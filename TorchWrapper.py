
import os, sys
import theano
import theano.tensor as T
from TheanoUtil import try_register_gpu_opt
from theano.sandbox.cuda import GpuOp
from Util import make_hashable, make_dll_name
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

  def __init__(self, in_info, out_info, lua_file, lua_fw_func, lua_bw_func=None):
    _init()
    super(TorchWrapperOp, self).__init__()
    self.in_info = make_hashable(in_info)
    self.out_info = make_hashable(out_info)
    self.lua_file = lua_file
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
    static lua_State* L;
    void init_torch() {
      if(L) return;
      // http://www.lua.org/manual/5.1/manual.html
      L = lua_open();
      if(!L) {
        printf("ERROR: TorchWrapper: Cannot create Lua state.\\n");
        printf("  If you are on MacOSX 64bit, Python must be linked with:\\n");
        printf("    -pagezero_size 10000 -image_base 100000000\\n");
        printf("  See here: http://luajit.org/install.html\\n");
        printf("  And here: https://groups.google.com/forum/#!topic/torch7/dW2rotAgijY\\n");
        return;
      }
      // http://stackoverflow.com/questions/966162/best-way-to-omit-lua-standard-libraries
      //luaL_openlibs(L);  // all standard Lua libs
      luaopen_base(L);
      // TODO...
    }
    """

  def c_init_code(self):
    return ["init_torch();"]

  def c_code(self, node, name, inputs, outputs, sub):
    return """
    init_torch();
    // TODO...
    PyErr_Format(PyExc_ValueError, "not implemented fully yet...");
    %(fail)s;
    """ % {'fail' : sub['fail']}

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

