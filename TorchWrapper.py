
import theano
import theano.tensor as T
from TheanoUtil import try_register_gpu_opt
from theano.sandbox.cuda import GpuOp
from Util import make_hashable


_initialized = False

def _init():
  global _initialized
  if _initialized: return
  # TODO... find Torch, etc.


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
    raise NotImplementedError  # only C code...

  def c_support_code(self):
    # TODO...
    return """
    #include <lua.h>
    """

  def c_code(self, node, name, inputs, outputs, sub):
    pass  # TODO...

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

