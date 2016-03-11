
import theano
import theano.tensor as T
from TheanoUtil import try_register_gpu_opt
from theano.sandbox.cuda import GpuOp, host_from_gpu, gpu_from_host


_initialized = False

def _init():
  global _initialized
  if _initialized: return
  # TODO... find Torch, etc.


class TorchWrapperOp(theano.Op):
  __props__ = ("n_in", "n_out", "ndim", "dtype", "lua_file", "lua_fw_func", "lua_bw_func")

  def __init__(self, n_in, n_out, lua_file, lua_fw_func, lua_bw_func=None, ndim=3, dtype="float32"):
    _init()
    super(TorchWrapperOp, self).__init__()
    assert ndim in [2, 3]  # sparse or not. dense is time*batch*feature.
    self.ndim = ndim
    self.dtype = dtype
    if isinstance(n_in, (int, long)): n_in = [n_in]
    assert isinstance(n_in, (tuple, list))
    self.n_in = n_in
    self.n_out = n_out
    self.lua_file = lua_file
    self.lua_fw_func = lua_fw_func
    self.lua_bw_func = lua_bw_func

  def make_node(self, *args):
    args = [T.as_tensor_variable(arg) for arg in args]
    index = args[-1]
    assert len(args) - 1 == len(self.n_in)
    assert index.ndim == 2  # time*batch
    return theano.Apply(self, args, [T.TensorType(self.dtype, (False,) * self.ndim)(), index.type()])

  def infer_shape(self, node, input_shapes):
    index_shape = input_shapes[-1]
    y_shape = [index_shape[0], index_shape[1]]  # time*batch
    if self.ndim == 3: y_shape += [self.n_out]
    return [tuple(y_shape), index_shape]

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
    # Only gradient of first arg supported at the moment.
    x = inputs[0]
    x_index = inputs[-1]
    D_y, D_y_index = output_grads
    if not self.lua_bw_func:
      D_x = T.DisconnectedType()()  # Unknown how to calculate gradient.
    elif x.ndim == 2:
      D_x = T.DisconnectedType()()  # Sparse input. Discrete values.
    else:
      assert x.ndim == 3
      grad_op = TorchWrapperOp(
        n_in=(self.n_out, self.n_in[0]),  # D_y, x, x_index
        n_out=self.n_in,
        lua_file=self.lua_file,
        lua_fw_func=self.lua_bw_func
      )
      D_x = grad_op(D_y, x, x_index)
    Ds_remaining = [T.DisconnectedType()() for inp in inputs[1:]]  # Discrete values or not supported.
    return [D_x] + Ds_remaining

  def connection_pattern(self, node):
    # The last is the index and it's disconnected (because discrete).
    # All others are connected.
    pattern = [[True] for inp in node.inputs[:-1]] + [[False]]
    assert len(pattern) == len(node.inputs)
    return pattern


class GpuTorchWrapperOp(GpuOp, TorchWrapperOp):
  pass  # TODO...


@try_register_gpu_opt
def local_gpu_TorchWrapper(node):
  if isinstance(node.op, TorchWrapperOp):
    args = node.inputs[:-1]
    index = node.inputs[-1]
    if all([(x.owner and x.owner.op == host_from_gpu) for x in args]):
      gpu_op = GpuTorchWrapperOp(**{key: getattr(node.op, key) for key in node.op.__props__})
      return [host_from_gpu(gpu_op([x.owner.inputs[0] for x in args] + [index]))]

