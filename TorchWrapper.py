
import os, sys
import theano
import theano.tensor as T
from TheanoUtil import try_register_gpu_opt
from theano.sandbox.cuda import GpuOp
from Util import make_hashable, make_dll_name, escape_c_str
from Log import log
from pprint import pprint


_initialized = False
_torch_include_dirs = None
_torch_lib_dirs = None

def _init():
  global _initialized, _torch_include_dirs, _torch_lib_dirs
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
  paths += map(
    os.path.expanduser,
    ["~/torch/install", "~/code/torch/install", "~/Programmierung/torch/install",
     "/usr", "/usr/local"])
  def is_torch_dir(p):
    if not os.path.exists("%s/lib/%s" % (p, make_dll_name("luajit"))): return False
    if not os.path.exists("%s/lib/%s" % (p, make_dll_name("TH"))): return False
    if not os.path.exists("%s/include/lua.h" % p): return False
    if not os.path.exists("%s/include/TH" % p): return False
    return True
  paths = filter(is_torch_dir, paths)
  print >>log.v4, "Found Lua & Torch dirs (will use the first one):"
  pprint(paths, log.v4)
  if not paths:
    print >>log.v2, "ERROR: Did not found Lua & Torch."
    _torch_include_dirs = _torch_lib_dirs = []
  else:
    _torch_include_dirs = ["%s/include" % paths[0]]
    _torch_lib_dirs = ["%s/lib" % paths[0]]
  _initialized = True


class TorchWrapperOp(theano.Op):
  __props__ = ("in_info", "out_info", "lua_file", "lua_fw_func", "lua_bw_func")

  def __init__(self, in_info, out_info, lua_fw_func, lua_bw_func=None, lua_file=None):
    _init()
    super(TorchWrapperOp, self).__init__()
    self.in_info = make_hashable(in_info)
    self.out_info = make_hashable(out_info)
    self.lua_file = lua_file  # if none, expect inplace definition
    self.lua_fw_func = lua_fw_func
    self.lua_bw_func = lua_bw_func
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
    return _torch_include_dirs

  def c_lib_dirs(self):
    _init()
    return _torch_lib_dirs

  def c_libraries(self):
    return ["luaT", "luajit"]

  def c_compile_args(self):
    args = []
    # Some libs may use @rpath to reference to the lib. This is needed so that it finds it. (MacOSX)
    # This will also make the dynamic linker search in these paths. (Linux,Unix)
    args += ["-Wl,-rpath,%s" % l for l in _torch_lib_dirs]
    return args

  def c_support_code(self):
    return """
    extern "C" {
    #include <lua.h>
    #include <luaT.h>
    #include <lualib.h>
    #include <lauxlib.h>
    #include <TH/TH.h>
    }
    // Some Lua versions luaL_dostring ignore the result (http://stackoverflow.com/questions/12528820).
    #undef luaL_dostring
    #define luaL_dostring(L,s)  (luaL_loadstring(L, s) || lua_pcall(L, 0, LUA_MULTRET, 0))

    static lua_State* L = 0;
    static long L_ref_counter = 0;  // via c_init_code_struct/c_cleanup_code_struct

    // for debugging
    static void stackDump (lua_State *L) {
      int i;
      int top = lua_gettop(L);
      printf("Lua stack (%i): ", top);
      for (i = 1; i <= top; i++) {  /* repeat for each level */
        int t = lua_type(L, i);
        switch (t) {
          case LUA_TSTRING:  /* strings */
            printf("`%s'", lua_tostring(L, i));
            break;
          case LUA_TBOOLEAN:  /* booleans */
            printf(lua_toboolean(L, i) ? "true" : "false");
            break;
          case LUA_TNUMBER:  /* numbers */
            printf("%g", lua_tonumber(L, i));
            break;
          default:  /* other values */
            printf("%s", lua_typename(L, t));
            break;
        }
        printf("  ");  /* put a separator */
      }
      printf("\\n");  /* end the listing */
    }
    """

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
        // http://www.lua.org/manual/5.1/manual.html
        L = lua_open();
        if(!L) {
          PyErr_Format(PyExc_RuntimeError,
            "ERROR: TorchWrapper: Cannot create Lua state.\\n"
            "  If you are on MacOSX 64bit, Python must be linked with:\\n"
            "    -pagezero_size 10000 -image_base 100000000\\n"
            "  See here: http://luajit.org/install.html\\n"
            "  And here: https://groups.google.com/forum/#!topic/torch7/dW2rotAgijY\\n"
          );
          %(fail)s;
        }
        // If only specific ones: http://stackoverflow.com/questions/966162
        luaL_openlibs(L);  // all standard Lua libs
        // TODO: load torch...?
      }
      const char* user_func_str = "return " %(user_func_str)s;
      if((luaL_loadstring(L, user_func_str) || lua_pcall(L, 0, 1, 0)) != 0) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: Error while getting lua_fw_func: %%s\\nCode:\\n%%s\\n",
          lua_tostring(L, -1),
          user_func_str);
        %(fail)s;
      }
      if(!lua_isfunction(L, -1)) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: lua_fw_func is not a function but a %%s",
          lua_typename(L, lua_type(L, -1)));
        %(fail)s;
      }
      lua_user_func_ref_%(name)s = luaL_ref(L, LUA_REGISTRYINDEX);
    """ % {'name': name, 'fail': sub['fail'], "user_func_str": escape_c_str(self.lua_fw_func)}

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
    return """
      if(!L) {  // should have been initialized via c_init_code_struct()
        PyErr_Format(PyExc_RuntimeError, "Lua not initialized.");
        %(fail)s;
      }
      lua_rawgeti(L, LUA_REGISTRYINDEX, lua_user_func_ref_%(name)s);
      if(lua_pcall(L, /*nargs*/0, /*nres*/0, /*error handler*/0) != 0) {
        PyErr_Format(PyExc_RuntimeError,
          "TorchWrapper: Error calling lua_fw_func: %%s",
          lua_tostring(L, -1));
        %(fail)s;
      }
      // TODO...
    """ % {'name': name, 'fail' : sub['fail']}

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
  pass  # TODO...


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

