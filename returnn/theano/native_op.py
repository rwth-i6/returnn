
import os

from returnn.native_op import NativeOpBaseMixin, Chunking, UnChunking, SubtensorBatchedIndex, SparseToDense, \
  MaxAndArgmaxSparse, CrossEntropySoftmaxAndGradientZSparse
from returnn.theano.util import softmax
from returnn.util.basic import escape_c_str, long

# noinspection PyPackageRequirements,PyUnresolvedReferences
import theano
# noinspection PyPackageRequirements,PyUnresolvedReferences
import theano.sandbox.cuda
# noinspection PyPackageRequirements,PyUnresolvedReferences
import theano.tensor as T
# noinspection PyPackageRequirements,PyUnresolvedReferences
from theano.compile import optdb
# noinspection PyPackageRequirements,PyUnresolvedReferences
from theano import gof
# noinspection PyPackageRequirements,PyUnresolvedReferences
from theano.gof.opt import OpSub
from returnn.theano.util import try_register_gpu_opt, make_var_tuple

TheanoNativeOpBase = theano.Op
TheanoGpuNativeOpBase = theano.sandbox.cuda.GpuOp


class TheanoNativeOp(TheanoNativeOpBase, NativeOpBaseMixin):
  """
  We wrap some C code which can define a forward pass
  and optionally a backward pass (for gradient calculation).
  The C code should be Numpy and CUDA compatible. See NativeOp.cpp.
  We also support inplace operations, i.e. we can operate inplace on some inputs.
  You can define in a flexible way all the inputs and the outputs.
  See __init__() for the details.

  All output variables are created automatically with the right shape
   but their content is not initialized,
   except when it's used by some input variable as the inplace output
   - in that case, it is either the input variable or it has a copy of its data.
  """

  __props__ = ("in_info", "out_info",
               "c_fw_code", "c_bw_code", "c_extra_support_code", "code_version",
               "grad_input_map", "name",
               "custom_grad")

  def __init__(self, custom_grad=None, **kwargs):
    """
    :param function custom_grad: if given, will use this instead for self.grad
    :param dict[str] kwargs: all passed to NativeOpBaseMixin
    """
    theano.Op.__init__(self)
    NativeOpBaseMixin.__init__(self, **kwargs)
    self.custom_grad = custom_grad

  def __str__(self):
    return "%s{%s,%s}" % (
      self.__class__.__name__,
      self.name,
      "inplace" if self.destroy_map else "no_inplace")

  @classmethod
  def as_tensor_var(cls, v):
    return theano.tensor.as_tensor_variable(v)

  @classmethod
  def tensor_type(cls, dtype, ndim):
    return T.TensorType(dtype=dtype, broadcastable=(False,) * ndim)

  @classmethod
  def contiguous(cls, v):
    from returnn.theano.util import Contiguous
    assert isinstance(v, theano.Variable)
    if getattr(v, 'owner', None):
      assert isinstance(v.owner, theano.Apply)
      if isinstance(v.owner.op, Contiguous.__base__):
        return v
    return Contiguous()(v)

  def _convert_input_var(self, v, info):
    v = self.as_tensor_var(v)
    dtype = "float32"  # Theano on GPU only supports float32 ... # info.get("dtype", "float32")
    if v.dtype != dtype:
      v = T.cast(v, dtype)
    if v.ndim != info["ndim"]:
      raise TypeError("input var ndim %i does not match with info %r" % (v.ndim, info))
    if info.get("need_contiguous", False):
      v = self.contiguous(v)
    return v

  def grad(self, inputs, output_grads):
    """
    For Theano.

    :param inputs:
    :param output_grads:
    :return:
    """
    if self.custom_grad:
      return self.custom_grad(self, inputs, output_grads)

    if not self.c_bw_code:
      # Unknown how to calculate gradient.
      return [T.DisconnectedType()() for inp in inputs]

    assert len(self.in_info) == len(inputs)
    assert len(self.out_info) == len(output_grads)

    # Some of output_grads might be of disconnected type.
    out_shapes = self.infer_shape(None, [v.shape for v in inputs])
    assert len(out_shapes) == len(output_grads)
    for i, out_grad in enumerate(output_grads):
      if isinstance(out_grad.type, T.DisconnectedType):
        output_grads[i] = T.zeros(out_shapes[i], dtype="float32")

    kwargs_for_grad = self.kwargs_for_grad_op()
    grad_op = self.__class__(**kwargs_for_grad)

    # noinspection PyCallingNonCallable
    grad_inputs = inputs + list(make_var_tuple(self(*inputs))) + output_grads
    grad_inputs = self._filter_grad_inputs(grad_inputs)
    assert len(grad_op.in_info) == len(grad_inputs)
    # noinspection PyCallingNonCallable
    grad_outputs = make_var_tuple(grad_op(*grad_inputs))
    assert len(grad_op.out_info) == len(grad_outputs)
    if grad_op.num_dummy_outs > 0:
      grad_outputs = grad_outputs[:-grad_op.num_dummy_outs]  # remove any dummy outputs

    return self.make_results_of_gradient(grad_outputs, disconnected_type=T.DisconnectedType())

  def connection_pattern(self, node):
    """
    For Theano.

    :param node:
    :return:
    """
    assert len(node.inputs) == len(self.in_info)
    pattern = [[info.get("gradient", "") != "disconnected"] * len(self.out_info)
               for info in self.in_info]
    return pattern

  def make_node(self, *args):
    """
    For Theano.

    :param args:
    :return:
    """
    assert len(args) == len(self.in_info)
    args = [self._convert_input_var(arg, info) for arg, info in zip(args, self.in_info)]
    outputs = [self.tensor_type(dtype=info.get("dtype", "float32"), ndim=info["ndim"])()
               for info in self.out_info]
    return theano.Apply(self, args, outputs)

  def perform(self, node, inputs, output_storage):
    """
    For Theano.

    :param node:
    :param inputs:
    :param output_storage:
    :return:
    """
    raise NotImplementedError("NativeOp: no pure Python implementation, only C implementation")

  def c_code_cache_version(self):
    """
    :type: tuple[int]
    """
    return self.code_version

  def c_support_code(self):
    """
    :return: Theano C++ code
    :rtype: str
    """
    base_src = open(os.path.dirname(__file__) + "/../native_op.cpp").read()
    return "\n\n".join([
      T.blas.blas_header_text(),
      "#define CUDA 0",
      base_src,
      self.c_extra_support_code])

  # noinspection PyMethodMayBeStatic
  def c_libraries(self):
    """
    :return: Theano libs
    :rtype: list[str]
    """
    return T.blas.ldflags()

  # noinspection PyMethodMayBeStatic
  def c_compile_args(self):
    """
    :return: Theano compile args
    :rtype: list[str]
    """
    return T.blas.ldflags(libs=False, flags=True)

  # noinspection PyMethodMayBeStatic
  def c_lib_dirs(self):
    """
    :return: Theano lib dirs
    :rtype: list[str]
    """
    return T.blas.ldflags(libs=False, libs_dir=True)

  # noinspection PyMethodMayBeStatic
  def c_header_dirs(self):
    """
    :return: Theano header dirs
    :rtype: list[str]
    """
    return T.blas.ldflags(libs=False, include_dir=True)

  def c_code(self, node, name, inputs, outputs, sub):
    """
    :param node:
    :param name:
    :param inputs:
    :param outputs:
    :param sub:
    :return:
    """
    assert len(inputs) == len(self.in_info)
    assert len(outputs) == len(self.out_info)
    return """
    {
      int n_inputs = %(n_inputs)i, n_outputs = %(n_outputs)i;
      Ndarray* inputs[] = {%(input_var_names_str)s};
      Ndarray** outputs[] = {%(output_var_names_str)s};
      int in_ndims[] = {%(input_ndims_str)s};
      int out_ndims[] = {%(output_ndims_str)s};
      Ndarray_DIM_Type output_shapes_flat[] = {%(output_shapes_flat_str)s};
      int in_want_inplace[] = {%(input_want_inplace_str)s};
      bool in_is_inplace[] = {%(input_is_inplace_str)s};

      // Check if we can reuse any preallocated output.
      // Reset those which we cannot reuse.
      {
        int out_shape_idx = 0;
        for(int i = 0; i < n_outputs; ++i) {
          assert_cmp(out_shape_idx + out_ndims[i], <=, ARRAY_LEN(output_shapes_flat));
          if(*outputs[i]) {
            bool can_reuse = true;
            for(int j = 0; j < out_ndims[i]; ++j)
              if(output_shapes_flat[out_shape_idx + j] != Ndarray_DIMS(*outputs[i])[j]) {
                can_reuse = false;
                break;
              }
            if(!can_reuse)
              Py_CLEAR(*outputs[i]);
          }
          out_shape_idx += out_ndims[i];
        }
        assert_cmp(out_shape_idx, ==, ARRAY_LEN(output_shapes_flat));
      }

      // Maybe reuse or otherwise copy input into output vars.
      for(int i = 0; i < n_inputs; ++i)
        if(in_want_inplace[i] >= 0) {
          assert_cmp(in_want_inplace[i], <, n_outputs);
          Py_XDECREF(*outputs[in_want_inplace[i]]);
          if(in_is_inplace[i]) {
            *(outputs[in_want_inplace[i]]) = inputs[i];
            Py_INCREF(inputs[i]);
          } else {
            *(outputs[in_want_inplace[i]]) = (Ndarray*) Ndarray_Copy(inputs[i]);
            if(!*(outputs[in_want_inplace[i]])) %(fail)s;
            inputs[i] = *(outputs[in_want_inplace[i]]);  // reset with copy
          }
        }

      // Init the remaining output vars. Note that they are initialized randomly!
      {
        int out_shape_idx = 0;
        for(int i = 0; i < n_outputs; ++i) {
          assert(out_shape_idx + out_ndims[i] <= ARRAY_LEN(output_shapes_flat));
          if(*(outputs[i])) {
            for(int j = 0; j < out_ndims[i]; ++j)
              // If this fails, we maybe have reused an input which has an invalid shape.
              assert_cmp(output_shapes_flat[out_shape_idx + j], ==, Ndarray_DIMS(*outputs[i])[j]);
          }
          else {
            *(outputs[i]) = (Ndarray*) Ndarray_NewDims(out_ndims[i], &output_shapes_flat[out_shape_idx]);
            if(!*(outputs[i])) %(fail)s;
          }
          out_shape_idx += out_ndims[i];
        }
        assert_cmp(out_shape_idx, ==, ARRAY_LEN(output_shapes_flat));
      }

      // And the user C code starts here.
      // --------------------------------
      %(c_code)s;
    }
    """ % {
      'name': name, 'fail': sub['fail'],
      'op_name': escape_c_str(self.name),
      'c_code': self.c_fw_code % {'fail': sub['fail']},
      'n_inputs': len(inputs), 'n_outputs': len(outputs),
      'input_var_names_str': ", ".join(["%s" % inp for inp in inputs]),
      'output_var_names_str': ", ".join(["&%s" % out for out in outputs]),
      'input_ndims_str': ', '.join(["%i" % info["ndim"] for info in self.in_info]),
      'output_ndims_str': ', '.join(["%i" % info["ndim"] for info in self.out_info]),
      'output_shapes_flat_str':
        ', '.join([(("%i" % s) if isinstance(s, (int, long))
                    else "Ndarray_DIMS(inputs[%i])[%i]" % s)
                   for info in self.out_info for s in info["shape"]]),
      "input_want_inplace_str": ", ".join([str(int(info.get("want_inplace", -1)))
                                           for info in self.in_info]),
      "input_is_inplace_str": ", ".join([str(int(info.get("is_inplace", False)))
                                         for info in self.in_info])
    }


class TheanoGpuNativeOp(TheanoNativeOp, TheanoGpuNativeOpBase):
  """
  Theano GPU native op.
  """

  @classmethod
  def as_tensor_var(cls, v):
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from theano.sandbox.cuda.basic_ops import as_cuda_ndarray_variable
    return as_cuda_ndarray_variable(v)

  @classmethod
  def tensor_type(cls, dtype, ndim):
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from theano.sandbox.cuda import CudaNdarrayType
    if dtype != "float32":
      print("%s: WARNING: cannot handle type %r, will use float32 instead" % ("GpuNativeOp", dtype))
      dtype = "float32"
    return CudaNdarrayType(dtype=dtype, broadcastable=(False,) * ndim)

  @classmethod
  def contiguous(cls, v):
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    assert isinstance(v, (theano.sandbox.cuda.CudaNdarrayVariable, theano.sandbox.cuda.CudaNdarrayConstant))
    if getattr(v, 'owner', None):
      assert isinstance(v.owner, theano.Apply)
      if v.owner.op == gpu_contiguous:
        return v
    return gpu_contiguous(v)

  def c_support_code(self):
    """
    :rtype: str
    """
    src = open(os.path.dirname(__file__) + "/NativeOp.cpp").read()
    return "\n\n".join([
      "#define CUDA 1",
      src,
      self.c_extra_support_code,
      "// end of c_support_code\n\n\n"])


def chunk(x, index, chunk_size, chunk_step):
  assert x.ndim == 3
  n_time = x.shape[0]
  n_batch = x.shape[1]
  n_dim = x.shape[2]
  if isinstance(chunk_size, T.TensorVariable):
    chunk_size = T.cast(chunk_size, "int64")
  if isinstance(chunk_step, T.TensorVariable):
    chunk_step = T.cast(chunk_step, "int64")
  n_chunks = T.maximum(n_time - chunk_size + chunk_step - 1, 0) // chunk_step + 1
  chunk_params = T.concatenate([T.as_tensor(chunk_size).reshape((1,)), T.as_tensor(chunk_step).reshape((1,))])
  out_buffer = T.zeros((chunk_size, n_batch * n_chunks, n_dim), dtype=x.dtype)
  oindex_buffer = T.zeros((chunk_size, n_batch * n_chunks), dtype=index.dtype)
  chunk_op = Chunking().make_theano_op()
  out, oindex = chunk_op(x, index, out_buffer, oindex_buffer, chunk_params)
  return out, oindex


def unchunk(x, index, chunk_size, chunk_step, n_time, n_batch):
  assert x.ndim == 3
  n_dim = x.shape[2]
  chunk_params = T.concatenate([T.as_tensor(chunk_size).reshape((1,)), T.as_tensor(chunk_step).reshape((1,))])
  out_buffer = T.zeros((n_time, n_batch, n_dim), dtype=x.dtype)
  oindex_buffer = T.zeros((n_time, n_batch), dtype=index.dtype)
  ofactors_buffer = T.zeros((n_time, n_batch), dtype=x.dtype)
  unchunk_op = UnChunking().make_theano_op()
  out, oindex, ofactors = unchunk_op(x, index, out_buffer, oindex_buffer, ofactors_buffer, chunk_params)
  return out, oindex, ofactors


def subtensor_batched_index(x, idx):
  if x.ndim == 2:
    assert idx.ndim == 1
    x = x.reshape((x.shape[0], 1, x.shape[1]))
    idx = idx.reshape((idx.shape[0], 1))
    y = subtensor_batched_index(x, idx)
    return y[:, 0]
  assert x.ndim == 3
  assert idx.ndim == 2
  op = SubtensorBatchedIndex().make_theano_op()
  return op(x, idx)


def sparse_to_dense(s0, s1, weight, mask, n_time, n_dim):
  assert s0.ndim == 2
  assert s1.ndim == 2
  assert weight.ndim == 2
  assert mask.ndim == 2
  n_batch = s0.shape[1]
  initial_W = T.zeros((n_time, n_batch, n_dim), dtype="float32")
  op = SparseToDense().make_theano_op()
  W = op(initial_W, s0, s1, weight, mask)
  assert isinstance(W, T.Variable)
  return W


def onehot_to_sparse(y, mask):
  assert y.ndim == 2
  assert mask.ndim == 2
  n_time = y.shape[0]
  n_batch = y.shape[1]
  y_t = T.arange(0, n_time, dtype="float32").dimshuffle(0, 'x') + T.zeros((n_time, n_batch), dtype="float32")
  y_i = y
  y_w = T.ones((n_time, n_batch), dtype="float32")
  return y_t, y_i, y_w, mask


def sparse_slice_offset(s0, idx):
  """
  :param s0: 1D tensor, ordered indices for sparse coo-format matrix (without batch)
  :param idx: scalar, index to find in s0
  :return: s0_idx, such that s0[i] >= idx for all i >= s0_idx, s0[i] < idx for all i < s0_idx.
  This assumes that the indices in s0 are ordered.
  """
  mask = s0 < idx
  return T.sum(mask)


def max_and_argmax_sparse(s0, s1, weight, mask, out_max, out_arg):
  op = MaxAndArgmaxSparse().make_theano_op()
  out_max, out_arg = op(s0, s1, weight, mask, out_max, out_arg)
  return out_max, out_arg


def crossentropy_softmax_and_gradient_z_sparse(z, z_mask, y_target_t, y_target_i, y_target_w, y_target_mask):
  op = CrossEntropySoftmaxAndGradientZSparse().make_theano_op()
  out_ce, out_grad_z, _out_max_z = op(z, z_mask, y_target_t, y_target_i, y_target_w, y_target_mask)
  return out_ce, out_grad_z


def crossentropy_softmax_and_gradient_z_sparse__slow(z, z_mask, y_target_t, y_target_i, y_target_w, y_target_mask):
  assert z.ndim == 3
  n_time = z.shape[0]
  n_batch = z.shape[1]
  n_dim = z.shape[2]
  y_target = sparse_to_dense(y_target_t, y_target_i, y_target_w, y_target_mask, n_time, n_dim)
  y = softmax(z)
  ce = -T.sum(y_target * T.log(y), axis=2)
  grad_z = y - y_target
  return ce, grad_z


@gof.local_optimizer([TheanoNativeOp], inplace=True)
def _inplace_native_op(node):
  if isinstance(node.op, TheanoNativeOp) and not node.op.destroy_map:
    kwargs = {k: getattr(node.op, k) for k in node.op.__props__}
    # TODO: We could try to make each input inplace individually.
    # What we do now is just to try to make all inplace.
    kwargs["in_info"] = [dict(info) for info in node.op.in_info]
    any_inplace = False
    for info in kwargs["in_info"]:
      if info.get("want_inplace", -1) >= 0:
        any_inplace = True
        info["is_inplace"] = True
    if not any_inplace:
      return False
    new_op = node.op.__class__(**kwargs)
    from returnn.theano.util import make_var_tuple
    # noinspection PyCallingNonCallable
    new_v = make_var_tuple(new_op(*node.inputs))
    return new_v
  return False


try:
  optdb.register('inplace_NativeOp',
                 gof.TopoOptimizer(_inplace_native_op
                                   , failure_callback=gof.TopoOptimizer.warn_inplace
                                   ),
                 60, 'fast_run', 'inplace')
except ValueError:  # can happen if it was already registered before, e.g. when we reload the module
  pass


@try_register_gpu_opt(TheanoNativeOp)
def _local_gpu_native_op(node):
  if isinstance(node.op, TheanoNativeOp):
    # see also: https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/opt.py
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from theano.sandbox.cuda import host_from_gpu, gpu_from_host, as_cuda_ndarray_variable
    args = node.inputs
    if any([(x.owner and x.owner.op == host_from_gpu) for x in args]):
      gpu_op = TheanoGpuNativeOp(**{key: getattr(node.op, key) for key in node.op.__props__})
      args = [x.owner.inputs[0] if (x.owner and x.owner.op == host_from_gpu) else x
              for x in args]
      from returnn.theano.util import make_var_tuple
      # noinspection PyCallingNonCallable
      outputs = make_var_tuple(gpu_op(*args))
      return [host_from_gpu(out) for out in outputs]
