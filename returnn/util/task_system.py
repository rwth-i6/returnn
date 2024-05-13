"""
Here are all subprocess, threading etc related utilities,
most of them quite low level.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Set
from threading import Lock
import sys
import os
import io
import pickle
import types
import struct
import marshal
from importlib import import_module
import numpy


_abs_mod_file = os.path.abspath(__file__)


class SharedMem:
    class ShmException(Exception):
        pass

    class CCallException(ShmException):
        pass

    if sys.platform != "win32":
        import ctypes
        import ctypes.util

        libc_so = ctypes.util.find_library("c")
        libc = ctypes.CDLL(libc_so, use_errno=True)
        shm_key_t = ctypes.c_int
        IPC_PRIVATE = 0
        IPC_RMID = 0

        # int shmget(key_t key, size_t size, int shmflg);
        shmget = libc.shmget
        shmget.restype = ctypes.c_int
        shmget.argtypes = (shm_key_t, ctypes.c_size_t, ctypes.c_int)
        # void* shmat(int shmid, const void *shmaddr, int shmflg);
        shmat = libc.shmat
        shmat.restype = ctypes.c_void_p
        shmat.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_int)
        # int shmdt(const void *shmaddr);
        shmdt = libc.shmdt
        shmdt.restype = ctypes.c_int
        shmdt.argtypes = (ctypes.c_void_p,)
        # int shmctl(int shmid, int cmd, struct shmid_ds *buf);
        shmctl = libc.shmctl
        shmctl.restype = ctypes.c_int
        shmctl.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        # void* memcpy( void *dest, const void *src, size_t count );
        memcpy = libc.memcpy
        memcpy.restype = ctypes.c_void_p
        memcpy.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)

        @classmethod
        def check_ccall_error(cls, check, f):
            import ctypes

            if not check:
                errno = ctypes.get_errno()
                errstr = os.strerror(errno)
                raise cls.CCallException("SharedMem: %s failed with error %i (%s)" % (f, errno, errstr))

        @classmethod
        def is_shmget_functioning(cls):
            shmid = cls.shmget(cls.IPC_PRIVATE, 4 * 1024 * 1024, 0o600)
            if shmid <= 0:
                return False
            cls.shmctl(shmid, cls.IPC_RMID, 0)
            return True

        def __init__(self, size, shmid=None):
            self.size = size
            self.shmid = None
            self.ptr = None
            if shmid is None:
                self.is_creator = True
                self.shmid = self.shmget(self.IPC_PRIVATE, self.size, 0o600)
                self.check_ccall_error(self.shmid > 0, "shmget")
                print("SharedMem[pid %i]: New shmid: %i (size %i)" % (os.getpid(), self.shmid, self.size))
                import atexit

                atexit.register(self.remove)
            else:
                self.is_creator = False
                self.shmid = shmid
                assert self.shmid > 0
            self.ptr = self.shmat(self.shmid, 0, 0)
            self.check_ccall_error(self.ptr != self.ctypes.c_void_p(-1).value, "shmat")
            self.check_ccall_error(self.ptr > 0, "shmat")

        def remove(self):
            if self.ptr:
                self.shmdt(self.ptr)
                self.ptr = None
            if self.shmid and self.shmid > 0:
                if self.is_creator:
                    print("SharedMem[pid %i]: Removing shmid %i (size %i)" % (os.getpid(), self.shmid, self.size))
                    self.shmctl(self.shmid, self.IPC_RMID, 0)
                self.shmid = None

        def __del__(self):
            self.remove()

        def __getstate__(self):
            return {"size": self.size, "shmid": self.shmid}

        def __setstate__(self, state):
            self.__init__(**state)

        def __repr__(self):
            return "<SharedMem shmid=%r size=%r is_creator=%r>" % (self.shmid, self.size, self.is_creator)


def next_power_of_two(n):
    return 2 ** (int(n - 1).bit_length())


class SharedNumpyArray:
    """
    This class provides a way to create Numpy arrays in shared memory.
    It adds some logic to mark whether some shared memory segment can be reused
    - that is when the client marks it as unused.

    Note that there are a few similar Python modules:
      https://pypi.python.org/pypi/SharedArray
      https://parad0x.org/git/python/shared-array/about
      https://bitbucket.org/cleemesser/numpy-sharedmem/src
      https://stackoverflow.com/questions/5033799/how-do-i-pass-large-numpy-arrays
      https://stackoverflow.com/questions/7894791/use-numpy-array-in-shared-memory
    """

    # cls members
    ServerLock = Lock()
    ServerInstances = set()
    ServerArrayId = 0

    class TooMuchInstances(SharedMem.ShmException):
        pass

    ExtraSpaceBytes = 4096
    # local members
    is_server = False
    mem = None
    shape, strides, typestr = None, None, None

    @staticmethod
    def numpy_strides_for_fortran(shape, typestr):
        itemsize = numpy.dtype(typestr).itemsize
        strides = [itemsize]
        for s in shape:
            strides += [strides[-1] * s]
        strides = strides[:-1]
        return tuple(strides)

    @staticmethod
    def numpy_strides_for_c_contiguous(shape, typestr):
        itemsize = numpy.dtype(typestr).itemsize
        strides = [numpy.prod(shape[i + 1 :], dtype="int") * itemsize for i in range(len(shape))]
        return tuple(strides)

    @classmethod
    def needed_mem_size(cls, shape, typestr):
        itemsize = numpy.dtype(typestr).itemsize
        mem_size = cls.ExtraSpaceBytes + itemsize * numpy.prod(shape)
        return mem_size

    @classmethod
    def as_shared(cls, array):
        assert isinstance(array, numpy.ndarray)
        if isinstance(array.base, SharedNumpyArray):
            assert array.base.is_in_use()
            return array.base
        return cls.create_copy(array)

    @classmethod
    def create_copy(cls, array):
        assert isinstance(array, numpy.ndarray)
        array_intf = array.__array_interface__
        shape = array_intf["shape"]
        strides = array_intf["strides"]
        typestr = array_intf["typestr"]
        if array.flags.c_contiguous or array.flags.f_contiguous:
            pass  # ok, we can reuse it like that
        else:
            assert strides
            # Use some similar strides so that the copying might be faster.
            if strides[0] == array.itemsize:
                strides = cls.numpy_strides_for_fortran(shape=shape, typestr=typestr)
            else:
                strides = None  # C-contiguous
        inst = cls.create_new(shape=shape, strides=strides, typestr=typestr)
        inst.create_numpy_array()[...] = array
        assert inst._get_sanity_check_flag_ref().value == 42
        assert inst.is_in_use()
        return inst

    @classmethod
    def create_new(cls, shape, strides, typestr):
        needed_mem_size = cls.needed_mem_size(shape=shape, typestr=typestr)
        with cls.ServerLock:
            for inst in cls.ServerInstances:
                assert isinstance(inst, SharedNumpyArray)
                assert inst._get_sanity_check_flag_ref().value == 42
                if inst.is_in_use():
                    continue
                if inst.mem.size < needed_mem_size:
                    inst._init_mem(shape=shape, typestr=typestr)
                # We can reuse it.
                inst._set_new_array_id()
                inst._set_is_used(1)
                inst._set_numpy_format(shape=shape, strides=strides, typestr=typestr)
                return inst
        return cls(shape=shape, strides=strides, typestr=typestr)

    @classmethod
    def _get_new_array_id(cls):
        array_id = cls.ServerArrayId
        cls.ServerArrayId += 1
        return array_id

    def _set_new_array_id(self):
        assert self.is_server
        self.array_id = self._get_new_array_id()
        self._get_array_id_ref().value = self.array_id

    def __init__(self, shape, strides, typestr, mem=None, array_id=None):
        if not mem:
            assert array_id is None
            if len(self.ServerInstances) >= SharedMemNumpyConfig["max_server_instances"]:
                raise self.TooMuchInstances("too much instances (%i)" % len(self.ServerInstances))
            self.is_server = True
            self._init_mem(shape=shape, typestr=typestr)
            self._set_new_array_id()
            self._set_is_used(1)
        else:
            assert array_id is not None
            self.is_server = False
            self.array_id = array_id
            mem_size = self.needed_mem_size(shape=shape, typestr=typestr)
            assert isinstance(mem, SharedMem)
            assert mem.size >= mem_size
            assert mem.shmid > 0
            assert mem.ptr > 0
            self.mem = mem
            assert self._get_sanity_check_flag_ref().value == 42
            assert self._get_array_id_ref().value == self.array_id
            assert self.is_in_use()
        self._set_numpy_format(shape=shape, strides=strides, typestr=typestr)
        if self.is_server:
            with self.ServerLock:
                self.ServerInstances.add(self)

    def _set_numpy_format(self, shape, strides, typestr):
        itemsize = numpy.dtype(typestr).itemsize
        if strides:
            assert all([st > 0 for st in strides])
            assert sum([st * (sh - 1) for (st, sh) in zip(strides, shape)]) + itemsize == numpy.prod(shape) * itemsize
        self.shape = shape
        self.strides = strides
        self.typestr = typestr

    def _init_mem(self, shape, typestr):
        assert self.is_server
        if self.mem:
            self.mem.remove()
            self.mem = None
        assert numpy.prod(shape) > 0
        mem_size = next_power_of_two(self.needed_mem_size(shape=shape, typestr=typestr))
        mem_size = max(SharedMemNumpyConfig["min_shared_mem_size"], mem_size)
        self.mem = SharedMem(size=mem_size)
        self._get_sanity_check_flag_ref().value = 42

    def get_numpy_array_data_ptr(self):
        assert self.mem.ptr > 0
        return self.mem.ptr + self.ExtraSpaceBytes

    @property
    def __array_interface__(self):
        assert self.shape
        # https://docs.scipy.org/doc/numpy/reference/arrays.interface.html
        return {
            "data": (self.get_numpy_array_data_ptr(), False),
            "shape": self.shape,
            "strides": self.strides,
            "typestr": self.typestr,
            "version": 3,
        }

    def create_numpy_array(self):
        assert self._get_sanity_check_flag_ref().value == 42
        assert self._get_array_id_ref().value == self.array_id
        assert self.is_in_use()
        a = numpy.array(self, copy=False)
        assert a.__array_interface__["data"][0] == self.get_numpy_array_data_ptr()
        assert not a.flags.owndata, "a.__array_interface__ = %r" % a.__array_interface__
        assert a.base is self
        assert a.nbytes + self.ExtraSpaceBytes <= self.mem.size
        assert (
            sum([st * (sh - 1) for (st, sh) in zip(a.strides, a.shape)]) + a.itemsize
            == numpy.prod(a.shape) * a.itemsize
            == a.nbytes
        )
        return a

    def _get_sanity_check_flag_ref(self):
        assert self.mem.ptr > 0
        import ctypes

        return ctypes.cast(ctypes.c_void_p(self.mem.ptr), ctypes.POINTER(ctypes.c_uint64)).contents

    def _get_array_id_ref(self):
        assert self.mem.ptr > 0
        import ctypes

        return ctypes.cast(ctypes.c_void_p(self.mem.ptr + 8), ctypes.POINTER(ctypes.c_uint64)).contents

    def _get_in_use_flag_ref(self):
        assert self.mem.ptr > 0
        import ctypes

        return ctypes.cast(ctypes.c_void_p(self.mem.ptr + 16), ctypes.POINTER(ctypes.c_uint64)).contents

    def _set_is_used(self, n):
        self._get_in_use_flag_ref().value = n

    def is_in_use(self):
        return self._get_in_use_flag_ref().value > 0

    def set_unused(self):
        if self.is_server:
            return
        if self.mem:
            self._set_is_used(0)
            self.mem.remove()
            self.mem = None

    def __getstate__(self):
        return {
            "shape": self.shape,
            "strides": self.strides,
            "typestr": self.typestr,
            "mem": self.mem,
            "array_id": self.array_id,
        }

    def __setstate__(self, state):
        self.__init__(**state)

    def __del__(self):
        # On the server side, we will get deleted at program end
        # because we are referenced in the global SharedNumpyArray.ServerInstances.
        # On the client side, we will get deleted once we are not used anymore.
        # Note that self.array holds a reference to self.
        self.set_unused()

    def __repr__(self):
        return "<%s is_server=%r state=%r>" % (self.__class__.__name__, self.is_server, self.__getstate__())


def attrChain(base, *attribs, **kwargs):
    default = kwargs.get("default", None)
    obj = base
    for attr in attribs:
        if obj is None:
            return default
        obj = getattr(obj, attr, None)
    if obj is None:
        return default
    return obj


# This is needed in some cases to avoid pickling problems with bounded funcs.
def funcCall(attrChainArgs, args=()):
    f = attrChain(*attrChainArgs)
    return f(*args)


Unpickler = pickle.Unpickler


def get_func_closure(f):
    return f.__closure__


# (code, globals[, name[, argdefs[, closure]]])
def get_func_tuple(f):
    return (
        f.__code__,
        f.__globals__,
        f.__name__,
        f.__defaults__,
        f.__closure__,
    )


_closure = (lambda x: lambda: x)(0)
_cell = get_func_closure(_closure)[0]
CellType = type(_cell)
ModuleType = type(sys)
DictType = dict
NewStyleClass = type


def makeFuncCell(value):
    return get_func_closure((lambda: value))[0]


def getModuleDict(modname, path=None):
    """
    :param str modname: such that "import <modname>" would work
    :param list[str] path: sys.path
    :return: the dict of the mod
    :rtype: dict[str]
    """
    try:
        mod = import_module(modname)
    except ImportError:
        # Try again with extended sys.path.
        assert path
        for p in path:
            if p not in sys.path:
                sys.path.append(p)
        mod = import_module(modname)
    return mod.__dict__


def getModNameForModDict(obj):
    """
    :type obj: dict
    :rtype: str | None
    :returns The module name or None. It will not return '__main__' in any case
    because that likely will not be the same in the unpickling environment.
    Also see: https://stackoverflow.com/questions/56171796/
    """
    if "__name__" not in obj:
        return None  # this is not a module
    mod_name = obj["__name__"]
    if mod_name == "__main__":
        return None
    if mod_name not in sys.modules:
        return None  # does not look like we have it loaded
    mod = sys.modules[mod_name]
    if mod.__dict__ is obj:
        return mod_name
    return None


def getNormalDict(d):
    """
    :type d: dict[str] | dictproxy
    :rtype: dict[str]
    It also removes getset_descriptor. New-style classes have those.
    """
    r = {}
    for k, v in d.items():
        if isinstance(v, types.GetSetDescriptorType):
            continue
        r[k] = v
    return r


def assign_obj_attribs(obj, d: Dict[str, Any]):
    """
    :param obj:
    :param d:
    :return: obj

    Note that obj.__dict__.update(d) does not always work,
    e.g. when obj is a type (then obj.__dict__ is a readonly mappingproxy).
    """
    for k, v in d.items():
        setattr(obj, k, v)
    return obj


def make_numpy_ndarray_fromstring(s, dtype, shape):
    return numpy.fromstring(s, dtype=dtype).reshape(shape)


SharedMemNumpyConfig = {
    "enabled": False,
    "auto_pickling_min_size": 8 * 1024 * 1024,  # 8MB
    "min_shared_mem_size": 32 * 1024 * 1024,  # 32MB
    "max_server_instances": 10,
}


def use_shared_mem_for_numpy_array(obj):
    assert isinstance(obj, numpy.ndarray)
    if obj.shape == ():  # scalar
        return False  # cannot use shared memory because it will always use its own memory
    if isinstance(obj.base, SharedNumpyArray):
        assert obj.base.is_in_use()
        return True
    if not SharedMemNumpyConfig["enabled"]:
        return False
    return obj.nbytes >= SharedMemNumpyConfig["auto_pickling_min_size"]


def numpy_set_unused(v):
    """
    :param numpy.ndarray v: array which will be marked as not-used-anymore
    This will tell mechanisms like SharedNumpyArray that it can reuse the memory.
    On the client side, this will even unmap the memory, so any further access
    to it will cause a SEGFAULT.
    """
    if v is None:
        return
    assert isinstance(v, numpy.ndarray)
    if isinstance(v.base, SharedNumpyArray):
        assert v.base.is_in_use()  # must not be called multiple times
        v.base.set_unused()


def numpy_copy_and_set_unused(v):
    """
    :param dict[str,numpy.ndarray|object] | numpy.ndarray | object v: object to be handled
    If v is a dict, we will return a new copied dict where every value is mapped through numpy_copy_and_set_unused.
    If v is a numpy.ndarray and its base is a SharedNumpyArray, we will copy it and
      call numpy_set_unused on the old value.
    If v is a numpy.ndarray and its base is not a SharedNumpyArray, we will just return it as it is and do nothing.
    In all other cases, we will also just return the object as it is and do nothing.
    """
    if isinstance(v, numpy.ndarray):
        if isinstance(v.base, SharedNumpyArray):
            newv = v.copy(order="A")
            numpy_set_unused(v)
            return newv
        return v
    if isinstance(v, dict):
        return {k: numpy_copy_and_set_unused(vv) for (k, vv) in v.items()}
    return v


def numpy_alloc(shape, dtype, fortran_for_shared=False):
    """
    If EnableAutoNumpySharedMemPickling is True, this will allocate a Numpy array
    in shared memory so we avoid a copy later on when this Numpy array would
    be transferred to another process via pickling.
    """
    if SharedMemNumpyConfig["enabled"]:
        dtype = numpy.dtype(dtype)
        typestr = dtype.str
        strides = None
        if fortran_for_shared:
            strides = SharedNumpyArray.numpy_strides_for_fortran(shape=shape, typestr=typestr)
        try:
            return SharedNumpyArray.create_new(shape=shape, strides=strides, typestr=typestr)
        except SharedMem.ShmException as e:
            print("numpy_alloc: SharedMem exception: %s" % e)
    # Fallback.
    return numpy.ndarray(shape, dtype=dtype)


try:
    _BasePickler = pickle._Pickler  # use the pure Python implementation
except AttributeError:
    _BasePickler = pickle.Pickler


class Pickler(_BasePickler):
    """
    We extend the standard Pickler to be able to pickle some more types,
    such as lambdas and functions, code, func cells, buffer and more.
    """

    def __init__(self, *args, **kwargs):
        if not "protocol" in kwargs:
            kwargs["protocol"] = pickle.HIGHEST_PROTOCOL
        _BasePickler.__init__(self, *args, **kwargs)

    dispatch = _BasePickler.dispatch.copy()
    memo: Dict[int, Tuple[int, Any]]

    def save_func(self, obj):
        try:
            self.save_global(obj)
            return
        except pickle.PicklingError:
            pass
        assert type(obj) is types.FunctionType
        self.save(types.FunctionType)
        self.save(get_func_tuple(obj))
        self.write(pickle.REDUCE)
        if id(obj) not in self.memo:  # Could be if we recursively landed here. See also pickle.save_tuple().
            self.memoize(obj)

        self._extend_save_extra_attribs(obj)

    dispatch[types.FunctionType] = save_func

    def save_method(self, obj):
        try:
            self.save_global(obj)
            return
        except pickle.PicklingError:
            pass
        assert type(obj) is types.MethodType
        self.save(types.MethodType)
        self.save((obj.__func__, obj.__self__))
        self.write(pickle.REDUCE)
        self.memoize(obj)

    dispatch[types.MethodType] = save_method

    def save_code(self, obj):
        assert type(obj) is types.CodeType
        self.save(marshal.loads)
        self.save((marshal.dumps(obj),))
        self.write(pickle.REDUCE)
        self.memoize(obj)

    dispatch[types.CodeType] = save_code

    def save_cell(self, obj):
        assert type(obj) is CellType
        self.save(makeFuncCell)
        self.save((obj.cell_contents,))
        self.write(pickle.REDUCE)
        self.memoize(obj)

    dispatch[CellType] = save_cell

    module_name_black_list: Set[str] = set()

    # We also search for module dicts and reference them.
    # This is for FunctionType.func_globals.
    def intellisave_dict(self, obj):
        modname = getModNameForModDict(obj)
        if modname and modname not in self.module_name_black_list:
            self.save(getModuleDict)
            self.save((modname, sys.path))
            self.write(pickle.REDUCE)
            self.memoize(obj)
            return
        self.save_dict(obj)

    dispatch[DictType] = intellisave_dict

    def save_module(self, obj):
        modname = getModNameForModDict(obj.__dict__)
        if modname and modname not in self.module_name_black_list:
            self.save(import_module)
            self.save((modname,))
            self.write(pickle.REDUCE)
            self.memoize(obj)
            return
        # We could maybe construct it manually. For now, just fail.
        raise pickle.PicklingError("cannot pickle module %r" % obj)

    dispatch[ModuleType] = save_module

    def save_ndarray(self, obj):
        if use_shared_mem_for_numpy_array(obj):
            try:
                shared = SharedNumpyArray.as_shared(obj)
            except SharedMem.ShmException as e:
                print("SharedNumpyArray exception: %s" % e)
                # fallback to default
            else:
                self.save(shared.create_numpy_array)
                self.save(())
                self.write(pickle.REDUCE)
                return
        # For some reason, Numpy fromstring/tostring is faster than Numpy loads/dumps.
        self.save(make_numpy_ndarray_fromstring)
        self.save((obj.tostring(), str(obj.dtype), obj.shape))
        self.write(pickle.REDUCE)

    dispatch[numpy.ndarray] = save_ndarray

    def save_iobuffer_dummy(self, obj):
        # Not supported but we want to not fail and just store None.
        self.save_none(None)

    dispatch[io.BufferedReader] = save_iobuffer_dummy
    dispatch[io.BufferedWriter] = save_iobuffer_dummy

    use_whichmodule: bool = True

    # Overwrite to avoid the broken pickle.whichmodule() which might return "__main__".
    def save_global(self, obj, name=None):
        assert obj
        assert id(obj) not in self.memo
        if name is None:
            name = obj.__name__

        module = getattr(obj, "__module__", None)
        if module == "__main__" and globals().get(name, None) is obj:
            # Can happen if this is directly executed.
            module = __name__  # should be correct now
        if (module is None or module == "__main__") and self.use_whichmodule:
            module = pickle.whichmodule(obj, name)
        if module is None or module == "__main__":
            raise pickle.PicklingError("Can't pickle %r: module not found: %s" % (obj, module))
        if module in self.module_name_black_list:
            raise pickle.PicklingError("Can't pickle %r: module blacklisted: %s" % (obj, module))

        try:
            __import__(module)
            mod = sys.modules[module]
            klass = getattr(mod, name)
        except (ImportError, KeyError, AttributeError):
            raise pickle.PicklingError("Can't pickle %r: it's not found as %s.%s" % (obj, module, name))
        else:
            if klass is not obj:
                raise pickle.PicklingError("Can't pickle %r: it's not the same object as %s.%s" % (obj, module, name))

        assert "\n" not in module
        assert "\n" not in name
        self.write(pickle.GLOBAL + bytes(module + "\n" + name + "\n", "utf8"))
        self.memoize(obj)

    def save_type(self, obj):
        try:
            self.save_global(obj)
            return
        except pickle.PicklingError:
            pass
        # Some types in the types modules are not correctly referenced,
        # such as types.FunctionType. This is fixed here.
        for modname in ["types"]:
            moddict = sys.modules[modname].__dict__
            for modobjname, modobj in moddict.items():
                if modobj is obj:
                    self.write(pickle.GLOBAL + bytes(modname + "\n" + modobjname + "\n", "utf8"))
                    self.memoize(obj)
                    return
        # Generic serialization of new-style classes.
        self._save_type_fallback(obj)

    def _save_type_fallback(self, obj):
        self.save(type)
        self.save((obj.__name__, obj.__bases__, {}))
        self.write(pickle.REDUCE)
        self.memoize(obj)
        self._extend_save_extra_attribs(obj)

    def _extend_save_extra_attribs(self, obj):
        # Assumes the obj is already saved and at the top of the stack.
        self.write(pickle.POP)  # we will put it back on the stack below

        # Assign the attribs after it is already memoized
        # to resolve recursive references.
        self.save(assign_obj_attribs)
        d = getNormalDict(obj.__dict__)
        if obj.__module__:
            d["__module__"] = obj.__module__
        self.save((obj, d))
        self.write(pickle.REDUCE)

    dispatch[NewStyleClass] = save_type

    # avoid pickling instances of ourself. this mostly doesn't make sense and leads to trouble.
    # however, also doesn't break. it mostly makes sense to just ignore.
    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass
