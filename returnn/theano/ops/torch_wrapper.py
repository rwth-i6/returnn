
# This is a Theano Op which can wrap Lua/Torch code.
# Some related projects / code:
# https://github.com/nouiz/theano_torch_bridge
# https://stackoverflow.com/questions/24712972/interfacing-python-and-torch7lua-via-shared-library
# https://pypi.python.org/pypi/lupa
# https://github.com/sermanet/OverFeat/blob/master/API/python/overfeatmodule.cpp
# https://github.com/albanD/lunatic-python  /  https://labix.org/lunatic-python
# https://github.com/hughperkins/pytorch
# https://github.com/facebook/fblualib/blob/master/fblualib/python/README.md

from __future__ import print_function

import os
import theano
import theano.tensor as T
from returnn.theano.util import try_register_gpu_opt
from theano.sandbox.cuda import GpuOp
from returnn.util.basic import make_hashable, make_dll_name, escape_c_str, long
from returnn.log import log
from pprint import pprint


_initialized = False
_torch_base_dir = None
_torch_include = "include"
_torch_lib = "lib"
_torch_share_lua = "share/lua/5.1"
_torch_lib_lua = "lib/lua/5.1"


def _init():
  global _initialized, _torch_base_dir
  if _initialized: return
  # Not sure about the best way. Maybe try multiple things.
  # We could set it in our config. get_global_config().
  # We could use some env var.
  # When torch-activate was called before, we should prefer that one.
  # In that case, and maybe in other cases, we will find the executable th in PATH.
  paths = []
  for binpath in os.environ.get("PATH", "").split(":"):
    if os.path.exists("%s/th" % binpath):  # e.g. "~/torch/install/bin"
      paths += [os.path.dirname(binpath)]  # parent dir
      break
  # Add some standard paths.
  paths += list(map(
    os.path.expanduser,
    ["~/torch/install", "~/code/torch/install", "~/Programmierung/torch/install",
     "/usr", "/usr/local"]))
  def is_torch_dir(p):
    if not os.path.exists("%s/lib/%s" % (p, make_dll_name("luajit"))): return False
    if not os.path.exists("%s/lib/%s" % (p, make_dll_name("TH"))): return False
    if not os.path.exists("%s/include/lua.h" % p): return False
    if not os.path.exists("%s/include/TH" % p): return False
    return True
  paths = list(filter(is_torch_dir, paths))
  print("Found Lua & Torch dirs (will use the first one):", file=log.v4)
  pprint(paths, log.v4)
  if not paths:
    print("ERROR: Did not found Lua & Torch.", file=log.v2)
  else:
    _torch_base_dir = paths[0]
    for p in ["%s/torch/init.lua" % _torch_share_lua, "%s/libtorch.so" % _torch_lib_lua]:
      fullp = "%s/%s" % (_torch_base_dir, p)
      if not os.path.exists(fullp):
        print("ERROR: Did not found Lua & Torch file:", fullp, file=log.v2)
  _initialized = True


class TorchWrapperOp(theano.Op):
  __props__ = ("in_info", "out_info", "lua_file", "lua_fw_func", "lua_bw_func", "name")

  def __init__(self, in_info, out_info, lua_fw_func, lua_bw_func=None, lua_file=None, name=None):
    _init()
    super(TorchWrapperOp, self).__init__()
    self.in_info = make_hashable(in_info)
    self.out_info = make_hashable(out_info)
    self.lua_file = lua_file  # if none, expect inplace definition
    self.lua_fw_func = lua_fw_func
    self.lua_bw_func = lua_bw_func
    self.name = name or "<anonymous>"
    for info in self.in_info + self.out_info:
      assert "ndim" in info
      assert "shape" in info
      assert len(info["shape"]) == info["ndim"]
    for info in self.out_info:
      for s in info["shape"]:
        assert s, "need output shape info or reference, %r" % info

  def make_node(self, *args):
    args = [T.as_tensor_variable(arg) for arg in args]
    assert len(args) == len(self.in_info)
    outputs = [T.TensorType(info.get("dtype", "float32"), (False,) * info["ndim"])()
               for info in self.out_info]
    return theano.Apply(self, args, outputs)

  def infer_shape(self, node, input_shapes):
    out_shapes = []
    for info in self.out_info:
      out_shape = list(info["shape"])
      for idx, s in enumerate(out_shape):
        if isinstance(s, tuple):  # we interpret this as a reference to input shapes
          assert len(s) == 2
          out_shape[idx] = input_shapes[s[0]][s[1]]
      assert not any([s is None for s in out_shape]), "out_shape %r, out_info %r" % (out_shape, self.out_info)
      out_shapes += [tuple(out_shape)]
    return out_shapes

  def perform(self, node, inputs, output_storage):
    raise NotImplementedError("TorchWrapper: no pure Python implementation, only C implementation")

  def c_header_dirs(self):
    _init()
    return ["%s/%s" % (_torch_base_dir, _torch_include)]

  def c_lib_dirs(self):
    _init()
    return ["%s/%s" % (_torch_base_dir, _torch_lib), "/usr/lib/atlas-base"]

  def c_libraries(self):
    return ["luaT", "luajit", "TH"]

  def c_compile_args(self):
    args = []
    # Some libs may use @rpath to reference to the lib. This is needed so that it finds it. (MacOSX)
    # This will also make the dynamic linker search in these paths. (Linux,Unix)
    args += ["-Wl,-rpath,%s" % l for l in self.c_lib_dirs()]
    return args

  def c_support_code(self):
    return """
    extern "C" {
    #include <lua.h>
    #include <luaT.h>
    #include <lualib.h>
    #include <lauxlib.h>
    #include <TH/TH.h>
    #include <dlfcn.h>
    #include <stdint.h>
    #include <assert.h>
    }
    #include <vector>

    #define ARRAY_LEN(x) (sizeof(x) / sizeof(x[0]))

    // Some Lua versions luaL_dostring ignore the result (https://stackoverflow.com/questions/12528820).
    #undef luaL_dostring
    #define luaL_dostring(L,s)  (luaL_loadstring(L, s) || lua_pcall(L, 0, LUA_MULTRET, 0))

    static const char* safe_lua_tostring(lua_State* L, int ud) {
      const char* s = lua_tostring(L, ud);
      if(!s) s = lua_typename(L, lua_type(L, ud));
      return s;
    }

    static lua_State* L = 0;
    static long L_ref_counter = 0;  // via c_init_code_struct/c_cleanup_code_struct

    // For TH documentation, best see the C code here: https://github.com/torch/torch7/blob/master/lib/TH/generic/
    // THC: https://github.com/torch/cutorch/tree/master/lib/THC
    // Also some documentation here: https://torch.ch/docs/developer-docs.html
    // LuaT: https://github.com/torch/torch7/blob/master/lib/luaT/luaT.c

    // Note: There is https://github.com/facebook/thpp for templated THTensor. But maybe overkill...
    // Some Tensor doc: https://github.com/torch/torch7/blob/master/doc/tensor.md
    template<typename T> struct TTH;
    template<> struct TTH<float> {
      typedef float type;
      typedef THFloatTensor Tensor;
      typedef THFloatStorage Storage;
      static Storage* Storage_newWithData(type* data, size_t size) { return THFloatStorage_newWithData(data, size); }
      static void Storage_free(Storage* storage) { THFloatStorage_free(storage); }
      static void Storage_clearFlag(Storage* storage, const char flag) { THFloatStorage_clearFlag(storage, flag); }
      static Tensor* Tensor_newWithStorage(
          Storage* storage, long storageOffset, THLongStorage* sizes, THLongStorage* strides) {
        return THFloatTensor_newWithStorage(storage, storageOffset, sizes, strides);
      }
      static const char* luaType() { return "torch.FloatTensor"; }
      static int numpyTypenum() { return NPY_FLOAT32; };
    };
    template<> struct TTH<int8_t> {
      typedef char type;
      typedef THCharTensor Tensor;
      typedef THCharStorage Storage;
      static Storage* Storage_newWithData(type* data, size_t size) { return THCharStorage_newWithData(data, size); }
      static void Storage_free(Storage* storage) { THCharStorage_free(storage); }
      static void Storage_clearFlag(Storage* storage, const char flag) { THCharStorage_clearFlag(storage, flag); }
      static Tensor* Tensor_newWithStorage(
          Storage* storage, long storageOffset, THLongStorage* sizes, THLongStorage* strides) {
        return THCharTensor_newWithStorage(storage, storageOffset, sizes, strides);
      }
      static const char* luaType() { return "torch.CharTensor"; }
      static int numpyTypenum() { return NPY_INT8; };
    };

    template<typename Base>
    static bool typed_lua_push_py_array(typename Base::type* data, size_t size, int ndim, long* shapes, long* strides) {
      typename Base::Storage* storage = Base::Storage_newWithData(data, size);
      Base::Storage_clearFlag(storage, TH_STORAGE_RESIZABLE|TH_STORAGE_FREEMEM);  // mem is owned by Python

      THLongStorage* shapes_storage = THLongStorage_newWithData(shapes, ndim);
      THLongStorage_clearFlag(shapes_storage, TH_STORAGE_RESIZABLE|TH_STORAGE_FREEMEM);  // mem is owned by Python
      THLongStorage* strides_storage = THLongStorage_newWithData(strides, ndim);
      THLongStorage_clearFlag(strides_storage, TH_STORAGE_RESIZABLE|TH_STORAGE_FREEMEM);  // mem is owned by Python

      typename Base::Tensor* tensor = Base::Tensor_newWithStorage(storage, 0, shapes_storage, strides_storage);
      luaT_pushudata(L, tensor, Base::luaType());

      THLongStorage_free(shapes_storage);
      THLongStorage_free(strides_storage);
      Base::Storage_free(storage);
      return true;
    }

    template<typename Base>
    static bool typed_lua_push_py_array(PyArrayObject* obj) {
      assert(PyArray_EquivTypenums(PyArray_TYPE(obj), Base::numpyTypenum()));
      typename Base::type* data = (typename Base::type*) PyArray_DATA(obj);
      size_t size = PyArray_NBYTES(obj) / sizeof(typename Base::type);

      int ndim = PyArray_NDIM(obj);
      std::vector<long> shapes(ndim);
      std::vector<long> strides(ndim);
      for(int i = 0; i < ndim; ++i) {
        shapes[i] = PyArray_DIM(obj, i);
        strides[i] = PyArray_STRIDE(obj, i) / sizeof(typename Base::type);  // Numpy strides are in bytes
      }

      return typed_lua_push_py_array<Base>(data, size, ndim, &shapes[0], &strides[0]);
    }

    static bool lua_push_py_array(PyArrayObject* obj) {
      if(PyArray_EquivTypenums(PyArray_TYPE(obj), TTH<float>::numpyTypenum()))
        return typed_lua_push_py_array<TTH<float> >(obj);

      if(PyArray_EquivTypenums(PyArray_TYPE(obj), TTH<int8_t>::numpyTypenum()))
        return typed_lua_push_py_array<TTH<int8_t> >(obj);

      PyErr_Format(PyExc_RuntimeError,
        "TorchWrapper: lua_push_py_array, cannot handle Numpy dtype %%s",
        PyArray_DESCR(obj)->typeobj->tp_name);
      return false;
    }

    template<typename Tensor>
    static bool isContiguous(const Tensor* tensor) {
      long z = 1;
      int d;
      for(d = tensor->nDimension - 1; d >= 0; d--) {
        if(tensor->size[d] != 1) {
          if(tensor->stride[d] == z)
            z *= tensor->size[d];
          else
            return false;
        }
      }
      return true;
    }

    template<typename Base>
    static bool typed_lua_pop_py_array(PyArrayObject** obj, const char* errmsg) {
      typename Base::Tensor* tensor = (typename Base::Tensor*) luaT_checkudata(L, -1, Base::luaType());
      if(!tensor) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: typed_lua_pop_py_array, luaT_checkudata returned NULL");
        return false;
      }

      // Cases:
      // 1. *obj == NULL. -> We need a new PyArrayObject.
      // 2. *obj != NULL. not sure... reuse? overwrite?
      //   To give Lua the possibility to reuse it, we could pass the old as a param to the Lua func?
      //   Or Lua could store the last return as a global? -> Then we cannot overtake the mem here.
      //   To make it possible to reuse it later by Lua, we need to let Lua own the memory.
      //   To make sure it does not go out of scope, we must use LUA_REGISTRYINDEX.

      // For now, very simple:
      if(*obj) Py_CLEAR(*obj);  // don't reuse any old one

      if(!tensor->storage) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: %%s, typed_lua_pop_py_array, tensor->storage == NULL", errmsg);
        return false;
      }
      if(tensor->storageOffset != 0) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: %%s, typed_lua_pop_py_array, tensor->storageOffset != 0, cannot handle that case yet", errmsg);
        return false;
      }
      if(tensor->storage->allocator != &THDefaultAllocator) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: %%s, typed_lua_pop_py_array, storage uses unknown mem allocator, cannot overtake memory", errmsg);
        return false;
      }
      if(!(tensor->storage->flag & TH_STORAGE_REFCOUNTED)) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: %%s, typed_lua_pop_py_array, storage not refcounted, not sure where memory came from", errmsg);
        return false;
      }
      if(!(tensor->storage->flag & TH_STORAGE_FREEMEM)) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: %%s, typed_lua_pop_py_array, storage does not own memory, not sure where memory came from", errmsg);
        return false;
      }
      if(tensor->storage->refcount != 1) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: %%s, typed_lua_pop_py_array, storage refcount is %%i, cannot overtake memory",
          errmsg, tensor->storage->refcount);
        return false;
      }
      tensor->storage->flag &= ~(TH_STORAGE_RESIZABLE | TH_STORAGE_FREEMEM);  // we overtake the memory
      typename Base::type* data = tensor->storage->data;

      int ndim = tensor->nDimension;
      std::vector<long> shapes(ndim);
      std::vector<long> strides(ndim);
      for(int i = 0; i < ndim; ++i) {
        shapes[i] = tensor->size[i];
        strides[i] = tensor->stride[i] * sizeof(typename Base::type);  // Numpy strides are in bytes
      }

      int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE | NPY_ARRAY_OWNDATA;
      if(isContiguous(tensor))
        flags |= NPY_ARRAY_C_CONTIGUOUS;

      *obj = (PyArrayObject*) PyArray_New(
        /*subtype*/&PyArray_Type,
        ndim, &shapes[0], Base::numpyTypenum(), &strides[0], data, /*itemsize*/0,
        flags, /*obj*/NULL);
      if(!*obj) {
        if(!PyErr_Occurred())
          PyErr_Format(PyExc_RuntimeError,
            "TorchWrapper: %%s, typed_lua_pop_py_array, failed to create PyArrayObject", errmsg);
        return false;
      }

      lua_pop(L, 1);
      return true;
    }

    static bool lua_pop_py_array(PyArrayObject** obj, const char* errmsg) {
      if(luaT_isudata(L, -1, "torch.FloatTensor"))
        return typed_lua_pop_py_array<TTH<float> >(obj, errmsg);

      PyErr_Format(PyExc_TypeError,
        "TorchWrapper: %%s, lua_pop_py_array: got type %%s",
        errmsg, lua_typename(L, lua_type(L, -1)));
      return false;
    }
    """ % {}

  def c_support_code_struct(self, node, name):
    return """
    int lua_user_func_ref_%(name)s;
    """ % {"name": name}

  def c_init_code_struct(self, node, name, sub):
    assert not self.lua_file, "not yet implemented..."
    return """
      lua_user_func_ref_%(name)s = LUA_REFNIL;
      L_ref_counter++;
      if(!L) {
        // workaround via https://groups.google.com/forum/#!topic/torch7/1Nl1OGEHxZw
        void *hdl = dlopen("libluajit.so", RTLD_NOW | RTLD_GLOBAL);
        if(hdl == 0) printf("TorchWrapper: dlopen luajit error: %%s\\n", dlerror());  // ignore...?

        // https://www.lua.org/manual/5.1/manual.html
        L = lua_open();
        if(!L) {
          PyErr_Format(PyExc_RuntimeError,
            "ERROR: TorchWrapper: Cannot create Lua state.\\n"
            "  If you are on MacOSX 64bit, Python must be linked with:\\n"
            "    -pagezero_size 10000 -image_base 100000000\\n"
            "  See here: https://luajit.org/install.html\\n"
            "  And here: https://groups.google.com/forum/#!topic/torch7/dW2rotAgijY\\n"
          );
          %(fail)s;
        }
        // If only specific ones: https://stackoverflow.com/questions/966162
        luaL_openlibs(L);  // all standard Lua libs

        // -e 'package.path="/u/zeyer/code/torch/install/share/lua/5.1/?.lua;/u/zeyer/code/torch/install/share/lua/5.1/?/init.lua;"..package.path'
        // -e 'package.cpath="/u/zeyer/code/torch/install/lib/lua/5.1/?.so;"..package.cpath'

        lua_getglobal(L, "package");
        lua_pushstring(L,
          %(_torch_base_dir)s "/" %(_torch_share_lua)s "/?.lua;"
          %(_torch_base_dir)s "/" %(_torch_share_lua)s "/?/init.lua;"
          "./?.lua"
        );
        lua_setfield(L, -2, "path");
        lua_pushstring(L,
          %(_torch_base_dir)s "/" %(_torch_lib_lua)s "/?.so;"
          %(_torch_base_dir)s "/" %(_torch_lib)s "/?.so;"
          "./?.so"
        );
        lua_setfield(L, -2, "cpath");
        lua_pop(L, 1);  // pops package

        lua_getglobal(L, "require");
        lua_pushstring(L, "torch");
        if(lua_pcall(L, 1, 0, 0) != 0) {
          PyErr_Format(PyExc_RuntimeError,
            "TorchWrapper: Error while loading Torch module: %%s",
            safe_lua_tostring(L, -1));
          %(fail)s;
        }

        // -e 'local k,l,_=pcall(require,"luarocks.loader") _=k
        // /u/zeyer/code/torch/install/lib/luarocks/rocks/trepl/scm-1/bin/th

        lua_getglobal(L, "torch");
        lua_getfield(L, -1, "updateerrorhandlers");
        lua_replace(L, -2);
        if(lua_pcall(L, 0, 0, 0) != 0) {
          PyErr_Format(PyExc_RuntimeError,
            "TorchWrapper: torch.updateerrorhandlers() error: %%s",
            safe_lua_tostring(L, -1));
          %(fail)s;
        }

        // https://github.com/torch/image/issues/107
        lua_getglobal(L, "torch");
        lua_getfield(L, -1, "setheaptracking");
        lua_replace(L, -2);
        lua_pushboolean(L, 1);
        if(lua_pcall(L, 1, 0, 0) != 0) {
          PyErr_Format(PyExc_RuntimeError,
            "TorchWrapper: torch.setheaptracking(true) error: %%s",
            safe_lua_tostring(L, -1));
          %(fail)s;
        }
      }

      const char* user_func_str = "return " %(user_func_str)s;
      if((luaL_loadstring(L, user_func_str) || lua_pcall(L, 0, 1, 0)) != 0) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: Error while getting %%s lua_fw_func: %%s\\nCode:\\n%%s\\n",
          %(op_name)s,
          safe_lua_tostring(L, -1),
          user_func_str);
        %(fail)s;
      }
      if(!lua_isfunction(L, -1)) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: %%s lua_fw_func is not a function but a %%s",
          %(op_name)s,
          lua_typename(L, lua_type(L, -1)));
        %(fail)s;
      }
      lua_user_func_ref_%(name)s = luaL_ref(L, LUA_REGISTRYINDEX);
    """ % {
      'name': name, 'fail': sub['fail'],
      'op_name': escape_c_str(self.name),
      "user_func_str": escape_c_str(self.lua_fw_func),
      "_torch_base_dir": escape_c_str(_torch_base_dir),
      "_torch_share_lua": escape_c_str(_torch_share_lua),
      "_torch_lib_lua": escape_c_str(_torch_lib_lua),
      "_torch_lib": escape_c_str(_torch_lib)
    }

  def c_cleanup_code_struct(self, node, name):
    return """
      if(L) {
        luaL_unref(L, LUA_REGISTRYINDEX, lua_user_func_ref_%(name)s);
        lua_user_func_ref_%(name)s = LUA_REFNIL;
      }
      L_ref_counter--;
      if(L_ref_counter == 0 && L) {
        lua_close(L);
        L = 0;
      }
    """ % {'name': name}

  def c_code(self, node, name, inputs, outputs, sub):
    assert len(inputs) == len(self.in_info)
    assert len(outputs) == len(self.out_info)
    return """
      PyArrayObject* inputs[] = {%(input_var_names_str)s};
      PyArrayObject** outputs[] = {%(output_var_names_str)s};
      int out_ndims[] = {%(output_ndims_str)s};
      int out_shapes_flat[] = {%(output_shapes_flat_str)s};
      int out_shape_idx = 0;
      int pcall_ret = 0;
      if(!L) {  // should have been initialized via c_init_code_struct()
        PyErr_Format(PyExc_RuntimeError, "Lua not initialized.");
        %(fail)s;
      }

      // TODO: I don't understand this. We do the same already in c_init_code_struct.
      // Is this another thread?
      {
        lua_getglobal(L, "torch");
        lua_getfield(L, -1, "updatethreadlocals");
        lua_replace(L, -2);
        if(lua_pcall(L, 0, 0, 0) != 0) {
          PyErr_Format(PyExc_RuntimeError,
            "TorchWrapper: torch.updatethreadlocals() error: %%s",
            safe_lua_tostring(L, -1));
          %(fail)s;
        }
      }

      // First push debug.traceback. This will be our pcall error handler.
      lua_getglobal(L, "debug");
      lua_getfield(L, -1, "traceback");
      lua_replace(L, -2);
      // Now the function itself which we want to call.
      lua_rawgeti(L, LUA_REGISTRYINDEX, lua_user_func_ref_%(name)s);
      // Now all the arguments.
      for(int i = 0; i < %(n_inputs)i; ++i) {
        if(!lua_push_py_array(inputs[i]))
          %(fail)s;
      }
      // And the call itself.
      pcall_ret = lua_pcall(L, %(n_inputs)i, %(n_outputs)i, /*error handler*/ - %(n_inputs)i - 2);
      if(pcall_ret != 0) {
        const char* errmsg = lua_tostring(L, -1);
        if(!errmsg) {
          errmsg = "(No Lua error message.)";
          printf("Unexpected Lua stack:\\n");
          luaT_stackdump(L);
        }
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: Error calling %%s lua_fw_func, error code %%i:\\n%%s",
          %(op_name)s, pcall_ret, errmsg);
        lua_pop(L, 2);  // remove error and debug.traceback from the stack
        %(fail)s;
      }
      // We don't want that there are any references floating around to the returned objects.
      // We want to overtake the memory of THStorage and we check that refcount == 1.
      lua_gc(L, LUA_GCCOLLECT, 0);
      // Now collect all the returned objects.
      for(int i = %(n_outputs)i - 1; i >= 0; --i) {
        if(!lua_pop_py_array(outputs[i], %(op_name)s))
          %(fail)s;

        if(PyArray_NDIM(*outputs[i]) != out_ndims[i]) {
          PyErr_Format(PyExc_ValueError,
            "TorchWrapper: %%s, lua_fw_func, in output %%i, got wrong ndim = %%i, expected %%i",
            %(op_name)s, i, PyArray_NDIM(*outputs[i]), out_ndims[i]);
          %(fail)s;
        }

        for(int j = 0; j < out_ndims[i]; ++j, ++out_shape_idx) {
          assert(out_shape_idx < ARRAY_LEN(out_shapes_flat));
          if(out_shapes_flat[out_shape_idx] >= 0) {  // otherwise we could infer it via input dim. TODO...
            if(PyArray_DIM(*outputs[i], j) != out_shapes_flat[out_shape_idx]) {
              PyErr_Format(PyExc_ValueError,
                "TorchWrapper: %%s lua_fw_func, in output %%i with ndim %%i, got wrong shape[%%i] = %%i, expected %%i",
                %(op_name)s, i, out_ndims[i], j, PyArray_DIM(*outputs[i], j), out_shapes_flat[out_shape_idx]);
              %(fail)s;
            }
          }
        }
      }
      lua_pop(L, 1); // remove debug.traceback from the stack
    """ % {
      'name': name, 'fail': sub['fail'],
      'op_name': escape_c_str(self.name),
      'n_inputs': len(inputs), 'n_outputs': len(outputs),
      'input_var_names_str': ", ".join(["%s" % inp for inp in inputs]),
      'output_var_names_str': ", ".join(["&%s" % out for out in outputs]),
      'output_ndims_str': ', '.join(["%i" % info["ndim"] for info in self.out_info]),
      'output_shapes_flat_str': (
        ', '.join([(("%i" % s) if isinstance(s, (int, long)) else "-1")
                   for info in self.out_info for s in info["shape"]]))
    }

  def grad(self, inputs, output_grads):
    if not self.lua_bw_func:
      # Unknown how to calculate gradient.
      return [T.DisconnectedType()() for inp in inputs]

    assert len(self.in_info) == len(inputs)
    assert len(self.out_info) == len(output_grads)
    out_info = [info.copy() for info in self.in_info]
    for idx, info in enumerate(out_info):
      # Refer to input shapes. See infer_shape().
      info["shape"] = [(idx, i) for i in range(info["ndim"])]
    out_info = [info for info in out_info if info.get("gradient", "") != "disconnected"]

    grad_op = TorchWrapperOp(
      name="grad-of-%s" % self.name,
      in_info=self.in_info + self.out_info,  # inputs + output_grads
      out_info=out_info,
      lua_file=self.lua_file,
      lua_fw_func=self.lua_bw_func
    )
    input_grads = grad_op(*(inputs + output_grads))
    assert len(out_info) == len(input_grads)

    results = []
    for info in self.in_info:
      if info.get("gradient", "") == "disconnected":
        results += [T.DisconnectedType()()]
      else:
        results += input_grads[:1]
        input_grads = input_grads[1:]
    assert len(input_grads) == 0
    assert len(results) == len(self.in_info)
    return results

  def connection_pattern(self, node):
    assert len(node.inputs) == len(self.in_info)
    pattern = [[info.get("gradient", "") != "disconnected"]
               for info in self.in_info]
    return pattern


class GpuTorchWrapperOp(GpuOp, TorchWrapperOp):
  # TODO...
  # Maybe helpful:
  # https://pydoc.net/Python/Theano/0.6.0/theano.sandbox.cuda.basic_ops/
  # https://deeplearning.net/software/theano/tutorial/aliasing.html
  # https://www.deeplearning.net/software/theano/extending/extending_theano_c.html#extending-theano-c
  # https://www.deeplearning.net/software/theano/extending/cop.html#Op.c_code
  # https://docs.scipy.org/doc/numpy/reference/c-api.array.html
  # https://docs.scipy.org/doc/numpy-1.9.2/reference/c-api.types-and-structures.html

  def c_support_code(self):
    cpu_code = super(GpuTorchWrapperOp, self).c_support_code()
    return """
    extern "C" {
    #include "THC.h"
    #include "THCTensor.h"
    }

    %(cpu_code)s

    // TODO?
    template<> struct THCuda {
      typedef float type;
      typedef THCudaTensor Tensor;
      typedef THCudaStorage Storage;
      static Storage* Storage_newWithData(type* data, size_t size) { return THCudaStorage_newWithData(data, size); }
      static void Storage_free(Storage* storage) { THCudaStorage_free(storage); }
      static void Storage_clearFlag(Storage* storage, const char flag) { THCudaStorage_clearFlag(storage, flag); }
      static Tensor* Tensor_newWithStorage(
          Storage* storage, long storageOffset, THLongStorage* sizes, THLongStorage* strides) {
        return THCudaTensor_newWithStorage(storage, storageOffset, sizes, strides);
      }
      static const char* luaType() { return "torch.CudaTensor"; }
    };

    // Generic function. support CudaNdarray (Theano CUDA) and PyArrayObject (Numpy).
    static bool lua_push_py_array(PyObject* obj) {
      if(PyArray_Check(obj))
        return lua_push_py_array((PyArrayObject*) obj);

      // https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/cuda_ndarray.cuh
      if(CudaNdarray_Check(obj)) {}  // TODO...

      PyErr_Format(PyExc_TypeError,
        "TorchWrapper: lua_push_py_array: cannot handle type %%s",
        obj->ob_type->tp_name);
      return false;
    }
    """ % {"cpu_code": cpu_code}

  def c_libraries(self):
    return super(GpuTorchWrapperOp, self).c_libraries() + ["THC"]



@try_register_gpu_opt(TorchWrapperOp)
def local_gpu_TorchWrapper(node):
  if isinstance(node.op, TorchWrapperOp):
    from theano.sandbox.cuda import host_from_gpu, gpu_from_host
    args = node.inputs
    if any([(x.owner and x.owner.op == host_from_gpu) for x in args]):
      gpu_op = GpuTorchWrapperOp(**{key: getattr(node.op, key) for key in node.op.__props__})
      args = [x.owner.inputs[0] if (x.owner and x.owner.op == host_from_gpu) else x
              for x in args]
      return [host_from_gpu(gpu_op(*args))]

