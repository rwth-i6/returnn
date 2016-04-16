
"""
Generic interface which automatically creates:
* CPU and GPU op
* inplace and not inplace
* grad variants
"""

import numpy
import theano
import theano.sandbox.cuda
import theano.tensor as T
from theano.gof.opt import OpSub
from theano.compile import optdb
from theano import gof
import os


class NativeOp(theano.Op):
  __props__ = ("inplace",)
  inplace = False

  def __init__(self):
    super(NativeOp, self).__init__()

  def as_tensor_var(self, v):
    return theano.tensor.as_tensor_variable(v)

  def contiguous(self, v):
    from theano.tensor.extra_ops import cpu_contiguous
    assert isinstance(v, theano.Variable)
    if getattr(v, 'owner', None):
      assert isinstance(v.owner, theano.Apply)
      if v.owner == cpu_contiguous:
        return v
    return cpu_contiguous(v)

  def c_support_code(self):
    src = open(os.path.dirname(__file__) + "/NativeOp.cpp").read()
    return "#define CUDA 0\n\n" + src

  def c_code(self, node, name, inputs, outputs, sub):
    raise NotImplementedError

  def make_node(self, *args):
    # TODO...
    args = [T.as_tensor_variable(arg) for arg in args]
    assert len(args) == len(self.in_info)
    outputs = [T.TensorType(info.get("dtype", "float32"), (False,) * info["ndim"])()
               for info in self.out_info]
    return theano.Apply(self, args, outputs)

  def infer_shape(self, node, input_shapes):
    # TODO..
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
    raise NotImplementedError("NativeOp: no pure Python implementation, only C implementation")

  def grad(self, inputs, output_grads):
    pass


class GpuNativeOpMixin(NativeOp, theano.sandbox.cuda.GpuOp):

  def as_tensor_var(self, v):
    from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
    return as_cuda_ndarray_variable(v)

  def contiguous(self, v):
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    assert isinstance(v, theano.sandbox.cuda.CudaNdarrayVariable)
    if getattr(v, 'owner', None):
      assert isinstance(v.owner, theano.Apply)
      if v.owner == gpu_contiguous:
        return v
    return gpu_contiguous(v)

  def c_support_code(self):
    src = open(os.path.dirname(__file__) + "/NativeOp.cpp").read()
    return "#define CUDA 1\n\n" + src


class NativeOpBase:
  """
  Base interface.
  """


def register_op(op):
  """
  :param NativeOpBase op: the op which we are going to register
  """
  pass


class LstmGenericBase:
  def code(self):
    pass

