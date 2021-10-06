
"""
Here are all subprocess, threading etc related utilities,
most of them quite low level.
"""

from __future__ import print_function
from threading import Lock, currentThread
import sys
PY3 = sys.version_info[0] >= 3

import os
import io

if PY3:
  from io import BytesIO
else:
  # noinspection PyUnresolvedReferences,PyCompatibility
  from StringIO import StringIO as BytesIO

from contextlib import contextmanager
import pickle
import types
import struct
import marshal
from importlib import import_module
import errno
import time
import numpy
try:
  from _multiprocessing import Connection
except ImportError:
  from multiprocessing.connection import Connection

_abs_mod_file = os.path.abspath(__file__)


def execInMainProc(func):
  global isMainProcess
  if isMainProcess:
    return func()
  else:
    assert _AsyncCallQueue.Self, "works only if called via asyncCall"
    return _AsyncCallQueue.Self.asyncExecClient(func)


def ExecInMainProcDecorator(func):
  def decoratedFunc(*args, **kwargs):
    return execInMainProc(lambda: func(*args, **kwargs))
  return decoratedFunc


class AsyncInterrupt(BaseException):
  pass


class ForwardedKeyboardInterrupt(Exception):
  pass


class _AsyncCallQueue:
  Self = None

  class Types:
    result = 0
    exception = 1
    asyncExec = 2

  def __init__(self, queue):
    assert not self.Self
    self.__class__.Self = self
    self.mutex = Lock()
    self.queue = queue

  def put(self, type, value):
    self.queue.put((type, value))

  def asyncExecClient(self, func):
    with self.mutex:
      self.put(self.Types.asyncExec, func)
      t, value = self.queue.get()
      if t == self.Types.result:
        return value
      elif t == self.Types.exception:
        raise value
      else:
        assert False, "bad behavior of asyncCall in asyncExec (%r)" % t

  @classmethod
  def asyncExecHost(clazz, task, func):
    q = task
    name = "<unknown>"
    try:
      name = repr(func)
      res = func()
    except Exception as exc:
      print("Exception in asyncExecHost %s %s" % (name, exc))
      q.put((clazz.Types.exception, exc))
    else:
      try:
        q.put((clazz.Types.result, res))
      except IOError:
        # broken pipe or so. parent quit. treat like a SIGINT
        raise KeyboardInterrupt


def asyncCall(func, name=None, mustExec=False):
  """
  This executes func() in another process and waits/blocks until
  it is finished. The returned value is passed back to this process
  and returned. Exceptions are passed back as well and will be
  reraised here.

  If `mustExec` is set, the other process must `exec()` after the `fork()`.
  If it is not set, it might omit the `exec()`, depending on the platform.
  """

  def doCall(queue):
    q = _AsyncCallQueue(queue)
    try:
      try:
        res = func()
      except KeyboardInterrupt as exc:
        print("Exception in asyncCall %s: KeyboardInterrupt" % name)
        q.put(q.Types.exception, ForwardedKeyboardInterrupt(exc))
      except BaseException as exc:
        print("Exception in asyncCall %s" % name)
        sys.excepthook(*sys.exc_info())
        q.put(q.Types.exception, exc)
      else:
        q.put(q.Types.result, res)
    except (KeyboardInterrupt, ForwardedKeyboardInterrupt):
      print("asyncCall: SIGINT in put, probably the parent died")
    # ignore

  task = AsyncTask(func=doCall, name=name, mustExec=mustExec)

  while True:
    # If there is an unhandled exception in doCall or the process got killed/segfaulted or so,
    # this will raise an EOFError here.
    # However, normally, we should catch all exceptions and just reraise them here.
    t,value = task.get()
    if t == _AsyncCallQueue.Types.result:
      return value
    elif t == _AsyncCallQueue.Types.exception:
      raise value
    elif t == _AsyncCallQueue.Types.asyncExec:
      _AsyncCallQueue.asyncExecHost(task, value)
    else:
      assert False, "unknown _AsyncCallQueue type %r" % t


class SharedMem:
  class ShmException(Exception): pass
  class CCallException(ShmException): pass

  if sys.platform != "win32":
    import ctypes
    import ctypes.util

    libc_so = ctypes.util.find_library('c')
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
  class TooMuchInstances(SharedMem.ShmException): pass
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
    strides = [numpy.prod(shape[i + 1:], dtype="int") * itemsize for i in range(len(shape))]
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
        if inst.is_in_use(): continue
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
      'typestr': self.typestr,
      "version": 3
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
    assert sum([st * (sh - 1) for (st, sh) in zip(a.strides, a.shape)]) + a.itemsize == numpy.prod(a.shape) * a.itemsize == a.nbytes
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
    if self.is_server: return
    if self.mem:
      self._set_is_used(0)
      self.mem.remove()
      self.mem = None

  def __getstate__(self):
    return {
      "shape": self.shape, "strides": self.strides, "typestr": self.typestr,
      "mem": self.mem, "array_id": self.array_id
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
    if obj is None: return default
    obj = getattr(obj, attr, None)
  if obj is None: return default
  return obj


# This is needed in some cases to avoid pickling problems with bounded funcs.
def funcCall(attrChainArgs, args=()):
  f = attrChain(*attrChainArgs)
  return f(*args)


Unpickler = pickle.Unpickler
if PY3:
  def get_func_closure(f): return f.__closure__
  # (code, globals[, name[, argdefs[, closure]]])
  def get_func_tuple(f):
    return (
      f.__code__,
      f.__globals__,
      f.__name__,
      f.__defaults__,
      f.__closure__,
    )
else:
  def get_func_closure(f): return f.func_closure
  def get_func_tuple(f):
    return (
      f.func_code,
      f.func_globals,
      f.func_name,
      f.func_defaults,
      f.func_closure,
    )
_closure = (lambda x: lambda: x)(0)
# noinspection PyUnresolvedReferences
_cell = get_func_closure(_closure)[0]
CellType = type(_cell)
ModuleType = type(sys)
# noinspection PyUnresolvedReferences
DictType = dict if PY3 else types.DictionaryType

if PY3:
  class BufferType: "Dummy"
  def make_buffer(*args): assert False
else:
  # noinspection PyUnresolvedReferences
  make_buffer = buffer
  # noinspection PyUnresolvedReferences
  BufferType = types.BufferType
  def bytes(x, *args): return str(x)

if PY3:
  _old_style_class = None
  class OldStyleClass: "Dummy"
  class _new_style_class: pass
  NewStyleClass = type
else:
  class _old_style_class: pass
  class _new_style_class(object): pass
  OldStyleClass = type(_old_style_class)  # == types.ClassType (classobj)
  NewStyleClass = type(_new_style_class)  # (type)

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
    if isinstance(v, types.GetSetDescriptorType): continue
    r[k] = v
  return r

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
  if v is None: return
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
  dispatch[types.FunctionType] = save_func

  def save_method(self, obj):
    try:
      self.save_global(obj)
      return
    except pickle.PicklingError:
      pass
    assert type(obj) is types.MethodType
    self.save(types.MethodType)
    if PY3:
      self.save((obj.__func__, obj.__self__))
    else:
      self.save((obj.im_func, obj.im_self, obj.im_class))
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

  # We also search for module dicts and reference them.
  # This is for FunctionType.func_globals.
  def intellisave_dict(self, obj):
    modname = getModNameForModDict(obj)
    if modname:
      self.save(getModuleDict)
      self.save((modname, sys.path))
      self.write(pickle.REDUCE)
      self.memoize(obj)
      return
    self.save_dict(obj)
  dispatch[DictType] = intellisave_dict

  def save_module(self, obj):
    modname = getModNameForModDict(obj.__dict__)
    if modname:
      self.save(import_module)
      self.save((modname,))
      self.write(pickle.REDUCE)
      self.memoize(obj)
      return
    # We could maybe construct it manually. For now, just fail.
    raise pickle.PicklingError('cannot pickle module %r' % obj)
  dispatch[ModuleType] = save_module

  def save_buffer(self, obj):
    self.save(buffer)
    self.save((str(obj),))
    self.write(pickle.REDUCE)
  dispatch[BufferType] = save_buffer

  def save_string(self, obj, pack=struct.pack):
    # Difference to base: We just always use BINSTRING (simpler)
    # and use a separate write for the obj itself.
    # For a huge obj, this avoids one unnecessary copy of the data.
    self.write(pickle.BINSTRING + pack("<i", len(obj)))
    self.write(bytes(obj, "utf8"))
  dispatch[str] = save_string

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
    if module is None or module == "__main__":
      module = pickle.whichmodule(obj, name)
    if module is None or module == "__main__":
      raise pickle.PicklingError(
          "Can't pickle %r: module not found: %s" % (obj, module))

    try:
      __import__(module)
      mod = sys.modules[module]
      klass = getattr(mod, name)
    except (ImportError, KeyError, AttributeError):
      raise pickle.PicklingError(
          "Can't pickle %r: it's not found as %s.%s" % (obj, module, name))
    else:
      if klass is not obj:
        raise pickle.PicklingError(
            "Can't pickle %r: it's not the same object as %s.%s" % (obj, module, name))

    assert "\n" not in module
    assert "\n" not in name
    self.write(pickle.GLOBAL + bytes(module + '\n' + name + '\n', "utf8"))
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
      for modobjname,modobj in moddict.items():
        if modobj is obj:
          self.write(pickle.GLOBAL + bytes(modname + '\n' + modobjname + '\n', "utf8"))
          self.memoize(obj)
          return
    # Generic serialization of new-style classes.
    self.save(type)
    self.save((obj.__name__, obj.__bases__, getNormalDict(obj.__dict__)))
    self.write(pickle.REDUCE)
    self.memoize(obj)
  dispatch[NewStyleClass] = save_type

  # This is about old-style classes.
  def save_class(self, cls):
    try:
      # First try with a global reference. This works normally. This is the default original pickle behavior.
      self.save_global(cls)
      return
    except pickle.PicklingError:
      pass
    # It didn't worked. But we can still serialize it.
    # Note that this could potentially confuse the code if the class is reference-able in some other way
    # - then we will end up with two versions of the same class.
    self.save(types.ClassType)
    self.save((cls.__name__, cls.__bases__, cls.__dict__))
    self.write(pickle.REDUCE)
    self.memoize(cls)
    return
  dispatch[OldStyleClass] = save_class

  # avoid pickling instances of ourself. this mostly doesn't make sense and leads to trouble.
  # however, also doesn't break. it mostly makes sense to just ignore.
  def __getstate__(self): return None
  def __setstate__(self, state): pass


class ExecingProcess:
  """
  This is a replacement for multiprocessing.Process which always
  uses fork+exec, not just fork.
  This ensures that you have a separate independent process.
  This can avoid many types of bugs, such as:
    https://stackoverflow.com/questions/24509650
    https://bugs.python.org/issue6721
    https://stackoverflow.com/questions/8110920
    https://stackoverflow.com/questions/23963997
    https://github.com/numpy/numpy/issues/654
    https://comments.gmane.org/gmane.comp.python.numeric.general/60204
  """

  def __init__(self, target, args, name, env_update):
    self.target = target
    self.args = args
    self.name = name
    self.env_update = env_update
    self.daemon = True
    self.pid = None
    self.exit_status = None

  def start(self):
    assert self.pid is None
    assert self.exit_status is None
    def pipeOpen():
      readend, writeend = os.pipe()
      if hasattr(os, "set_inheritable"):
        # Python 3 by default will close all fds in subprocesses. This will avoid that.
        os.set_inheritable(readend, True)
        os.set_inheritable(writeend, True)
      readend = os.fdopen(readend, "rb")
      writeend = os.fdopen(writeend, "wb")
      return readend, writeend
    self.pipe_c2p = pipeOpen()
    self.pipe_p2c = pipeOpen()
    self.parent_pid = os.getpid()
    pid = os.fork()
    flags = {key: value for (key, value) in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]}
    if 'base_compiledir' in flags:
      offset = flags['base_compiledir'].find("_-_", 1)
      if offset > 1:
        flags['base_compiledir'] = flags['base_compiledir'][:offset]
      flags['base_compiledir'] += '_-_' + self.name.replace(' ','_')
    else:
      flags['base_compiledir'] = '/tmp/theano/' + self.name.replace(' ','_')
    os.environ["THEANO_FLAGS"] = ",".join(["=".join(x) for x in flags.items()])
    if pid == 0:  # child
      try:
        sys.stdin.close()  # Force no tty stdin.
        self.pipe_c2p[0].close()
        self.pipe_p2c[1].close()
        py_mod_file = os.path.splitext(_abs_mod_file)[0] + ".py"
        assert os.path.exists(py_mod_file)
        args = [sys.executable,
                py_mod_file,
                "--forkExecProc",
                str(self.pipe_c2p[1].fileno()),
                str(self.pipe_p2c[0].fileno())]
        if self.env_update:
          os.environ.update(self.env_update)
        os.execv(args[0], args)  # Does not return if successful.
      except BaseException:
        print("ExecingProcess: Error at initialization.")
        sys.excepthook(*sys.exc_info())
        sys.exit(1)
      finally:
        sys.exit()
    else:  # parent
      self.pipe_c2p[1].close()
      self.pipe_p2c[0].close()
      self.pid = pid
      self.pickler = Pickler(self.pipe_p2c[1])
      self.pickler.dump(self.name)
      self.pickler.dump(self.target)
      self.pickler.dump(self.args)
      self.pipe_p2c[1].flush()

  def _wait(self, options=0):
    assert self.parent_pid == os.getpid()
    assert self.pid
    assert self.exit_status is None
    pid, exit_status = os.waitpid(self.pid, options)
    if pid != self.pid:
      assert pid == 0
      # It's still alive, otherwise we would have get the same pid.
      return
    self.exit_status = exit_status
    self.pid = None

  def is_alive(self):
    if self.pid is None:
      return False
    self._wait(os.WNOHANG)
    return self.pid is not None

  def join(self, timeout=None):
    if not self.is_alive():
      return
    if timeout:
      # Simple and stupid implementation.
      while self.is_alive():
        if timeout < 0:
          break
        if timeout < 1.0:
          time.sleep(timeout)
          break
        else:
          time.sleep(1)
          timeout -= 1
      return
    self._wait()

  Verbose = False

  @staticmethod
  def checkExec():
    if "--forkExecProc" in sys.argv:
      try:
        from returnn.util import better_exchook
      except ImportError:
        pass  # Doesn't matter.
      else:
        better_exchook.install()
      argidx = sys.argv.index("--forkExecProc")
      writeFileNo = int(sys.argv[argidx + 1])
      readFileNo = int(sys.argv[argidx + 2])
      readend = os.fdopen(readFileNo, "rb")
      writeend = os.fdopen(writeFileNo, "wb")
      unpickler = Unpickler(readend)
      name = unpickler.load()
      if ExecingProcess.Verbose: print("ExecingProcess child %s (pid %i)" % (name, os.getpid()))
      try:
        target = unpickler.load()
        args = unpickler.load()
      except EOFError:
        print("Error: unpickle incomplete")
        raise SystemExit
      ret = target(*args)
      sys.exited = True
      # IOError is probably broken pipe. That probably means that the parent died.
      try: Pickler(writeend).dump(ret)
      except IOError: pass
      try: readend.close()
      except IOError: pass
      try: writeend.close()
      except IOError: pass
      if ExecingProcess.Verbose: print("ExecingProcess child %s (pid %i) finished" % (name, os.getpid()))
      raise SystemExit


class ProcConnectionDied(Exception):
  pass


class ExecingProcess_ConnectionWrapper(object):
  """
  Wrapper around multiprocessing.connection.Connection.
  This is needed to use our own Pickler.
  """

  def __init__(self, fd=None, conn=None):
    self.fd = fd
    if self.fd:
      self.conn = Connection(fd)
    elif conn:
      self.conn = conn
    else:
      self.conn = None

  def __repr__(self):
    if self.conn is not None:
      return "<ExecingProcess_ConnectionWrapper fileno=%r>" % self.conn.fileno()
    else:
      return "<ExecingProcess_ConnectionWrapper None>"

  def __getstate__(self):
    if self.fd is not None:
      return {"fd": self.fd}
    elif self.conn is not None:
      return {"conn": self.conn}  # Try to pickle the connection.
    else:
      return {}

  def __setstate__(self, state):
    self.__init__(**state)

  def __getattr__(self, attr):
    return getattr(self.conn, attr)

  def _check_closed(self):
    if self.conn.closed:
      raise ProcConnectionDied("connection closed")

  def _check_writable(self):
    if not self.conn.writable:
      raise ProcConnectionDied("connection not writeable")

  def _check_readable(self):
    if not self.conn.readable:
      raise ProcConnectionDied("connection not readable")

  def poll(self, *args, **kwargs):
    while True:
      try:
        return self.conn.poll(*args, **kwargs)
      except IOError as e:
        if e.errno == errno.EINTR:
          # https://stackoverflow.com/questions/14136195
          # We can just keep trying.
          continue
        raise ProcConnectionDied("poll IOError: %s" % e)
      except EOFError as e:
        raise ProcConnectionDied("poll EOFError: %s" % e)

  def send_bytes(self, value):
    try:
      self.conn.send_bytes(value)
    except (EOFError, IOError) as e:
      raise ProcConnectionDied("send_bytes EOFError/IOError: %s" % e)

  def send(self, value):
    self._check_closed()
    self._check_writable()
    buf = BytesIO()
    Pickler(buf).dump(value)
    self.send_bytes(buf.getvalue())

  def recv_bytes(self):
    while True:
      try:
        return self.conn.recv_bytes()
      except IOError as e:
        if e.errno == errno.EINTR:
          # https://stackoverflow.com/questions/14136195
          # We can just keep trying.
          continue
        raise ProcConnectionDied("recv_bytes IOError: %s" % e)
      except EOFError as e:
        raise ProcConnectionDied("recv_bytes EOFError: %s" % e)

  def recv(self):
    self._check_closed()
    self._check_readable()
    buf = self.recv_bytes()
    f = BytesIO(buf)
    res = Unpickler(f).load()
    return res


def ExecingProcess_Pipe():
  """
  This is like multiprocessing.Pipe(duplex=True).
  It uses our own ExecingProcess_ConnectionWrapper.
  """
  import socket
  s1, s2 = socket.socketpair()
  # We duplicate the fds because the socket objects will close the fds after they go out of scope.
  fd1 = os.dup(s1.fileno())
  fd2 = os.dup(s2.fileno())
  s1.close()
  s2.close()
  if hasattr(os, "set_inheritable"):
    # Python 3 by default will close all fds in subprocesses. This will avoid that.
    os.set_inheritable(fd1, True)
    os.set_inheritable(fd2, True)
  c1 = ExecingProcess_ConnectionWrapper(fd1)
  c2 = ExecingProcess_ConnectionWrapper(fd2)
  return c1, c2


def Pipe_ConnectionWrapper(*args, **kwargs):
  from multiprocessing import Pipe
  c1, c2 = Pipe(*args, **kwargs)
  c1 = ExecingProcess_ConnectionWrapper(conn=c1)
  c2 = ExecingProcess_ConnectionWrapper(conn=c2)
  return c1, c2


if sys.platform == "win32":
  try:
    from multiprocessing.forking import Popen as mp_Popen
  except ImportError:
    from multiprocessing.popen_spawn_win32 import Popen as mp_Popen

  class Win32_mp_Popen_wrapper:
    def __init__(self, env_update):
      self.env = os.environ.copy()
      self.env.update(env_update)

    class Popen(mp_Popen):
      # noinspection PyMissingConstructor
      def __init__(self, process_obj, env):
        # No super init call by intention!

        from multiprocessing.forking import duplicate, get_command_line, _python_exe, close, get_preparation_data, HIGHEST_PROTOCOL, dump
        import msvcrt
        import _subprocess

        # create pipe for communication with child
        rfd, wfd = os.pipe()

        # get handle for read end of the pipe and make it inheritable
        rhandle = duplicate(msvcrt.get_osfhandle(rfd), inheritable=True)
        os.close(rfd)

        # start process
        cmd = get_command_line() + [rhandle]
        cmd = ' '.join('"%s"' % x for x in cmd)
        hp, ht, pid, tid = _subprocess.CreateProcess(
          _python_exe, cmd, None, None, 1, 0, env, None, None
        )
        ht.Close()
        close(rhandle)

        # set attributes of self
        self.pid = pid
        self.returncode = None
        self._handle = hp

        # send information to child
        prep_data = get_preparation_data(process_obj._name)
        to_child = os.fdopen(wfd, 'wb')
        mp_Popen._tls.process_handle = int(hp)
        try:
          dump(prep_data, to_child, HIGHEST_PROTOCOL)
          dump(process_obj, to_child, HIGHEST_PROTOCOL)
        finally:
          del mp_Popen._tls.process_handle
          to_child.close()

    def __call__(self, process_obj):
      return self.Popen(process_obj, self.env)


isFork = False  # fork() without exec()
isMainProcess = True


class AsyncTask:
  """
  This uses multiprocessing.Process or ExecingProcess to execute some function.
  In addition, it provides a duplex pipe for communication. This is either
  multiprocessing.Pipe or ExecingProcess_Pipe.
  """

  def __init__(self, func, name=None, mustExec=False, env_update=None):
    """
    :param func: a function which gets a single parameter,
      which will be a reference to our instance in the fork,
      so that it can use our communication methods put/get.
    :type str name: name for the sub process
    :param bool mustExec: if True, we do fork+exec, not just fork
    :param dict[str,str] env_update: for mustExec, also update these env vars
    """
    self.name = name or "unnamed"
    self.func = func
    self.mustExec = mustExec
    self.env_update = env_update
    self.parent_pid = os.getpid()
    self.SharedMemNumpyConfig = SharedMemNumpyConfig
    proc_args = {
      "target": funcCall,
      "args": ((AsyncTask, "_asyncCall"), (self,)),
      "name": self.name + " worker process"
    }
    if mustExec and sys.platform != "win32":
      self.Process = ExecingProcess
      self.Pipe = ExecingProcess_Pipe
      proc_args["env_update"] = env_update
    else:
      from multiprocessing import Process, Pipe
      self.Process = Process
      self.Pipe = Pipe_ConnectionWrapper
    self.parent_conn, self.child_conn = self.Pipe()
    self.proc = self.Process(**proc_args)
    self.proc.daemon = True
    if sys.platform == 'win32':
      self.proc._Popen = Win32_mp_Popen_wrapper(env_update=env_update)
    self.proc.start()
    self.child_conn.close()
    self.child_pid = self.proc.pid
    assert self.child_pid
    self.conn = self.parent_conn

  @staticmethod
  def _asyncCall(self):
    assert self.isChild
    assert isinstance(self.parent_conn, (ExecingProcess_ConnectionWrapper, Connection))
    parent_conn_handle = self.parent_conn.fileno()
    try:
      self.parent_conn.close()
    except Exception:
      # Print this here because in the backtrace, the handle will likely be reset already.
      print("parent connection close failed; file handle was: %r" % parent_conn_handle)
      raise
    self.conn = self.child_conn  # we are the child
    if not self.mustExec and sys.platform != "win32":
      global isFork
      isFork = True
    global isMainProcess
    isMainProcess = False
    global SharedMemNumpyConfig
    SharedMemNumpyConfig = self.SharedMemNumpyConfig
    try:
      self.func(self)
    except KeyboardInterrupt:
      print("Exception in AsyncTask %s: KeyboardInterrupt" % self.name)
      sys.exit(1)
    except SystemExit:
      raise
    except BaseException:
      print("Exception in AsyncTask %s" % self.name)
      sys.excepthook(*sys.exc_info())
      sys.exit(1)
    finally:
      self.conn.close()

  def put(self, value):
    self.conn.send(value)

  def get(self):
    thread = currentThread()
    try:
      thread.waitQueue = self
      res = self.conn.recv()
    except EOFError: # this happens when the child died
      raise ForwardedKeyboardInterrupt()
    except Exception:
      raise
    finally:
      thread.waitQueue = None
    return res

  @property
  def isParent(self):
    return self.parent_pid == os.getpid()

  @property
  def isChild(self):
    if self.isParent: return False
    # Note: self.parent_pid != os.getppid() if the parent died.
    return True

  # This might be called from the module code.
  # See OnRequestQueue which implements the same interface.
  def setCancel(self):
    self.conn.close()
    if self.isParent and self.child_pid:
      import signal
      try:
        os.kill(self.child_pid, signal.SIGINT)
      except OSError:
        # Could be that the process already died or so. Just ignore and assume it is dead.
        pass
      self.child_pid = None

  terminate = setCancel  # alias

  def join(self, timeout=None):
    return self.proc.join(timeout=timeout)

  def is_alive(self):
    return self.proc.is_alive()


def WarnMustNotBeInForkDecorator(func):
  class Ctx:
    didWarn = False
  def decoratedFunc(*args, **kwargs):
    global isFork
    if isFork:
      if not Ctx.didWarn:
        print("Must not be in fork!")
        Ctx.didWarn = True
      return None
    return func(*args, **kwargs)
  return decoratedFunc


class ReadWriteLock(object):
  """Classic implementation of ReadWriteLock.
  Note that this partly supports recursive lock usage:
  - Inside a readlock, a writelock will always block!
  - Inside a readlock, another readlock is fine.
  - Inside a writelock, any other writelock or readlock is fine.
  """
  def __init__(self):
    import threading
    self.lock = threading.RLock()
    self.writeReadyCond = threading.Condition(self.lock)
    self.readerCount = 0
  @property
  @contextmanager
  def readlock(self):
    with self.lock:
      self.readerCount += 1
    try: yield
    finally:
      with self.lock:
        self.readerCount -= 1
        if self.readerCount == 0:
          self.writeReadyCond.notifyAll()
  @property
  @contextmanager
  def writelock(self):
    with self.lock:
      while self.readerCount > 0:
        self.writeReadyCond.wait()
      yield


if __name__ == "__main__":
  # Make this the right module package.
  sys.path.insert(0, os.path.realpath(os.path.dirname(os.path.abspath(__file__)) + "/../.."))
  import returnn.util  # make sure parent package exists
  __package__ = "returnn.util"
  __name__ = "returnn.util.task_system"
  mod = sys.modules["__main__"]
  sys.modules[__name__] = mod
  returnn.util.task_system = mod  # need to set this
  import returnn.util.task_system as mod_  # make sure this works now
  assert mod is mod_
  try:
    ExecingProcess.checkExec()  # Never returns if this proc is called via ExecingProcess.
  except KeyboardInterrupt:
    sys.exit(1)
  print("You are not expected to call this. This is for ExecingProcess.")
  sys.exit(1)
