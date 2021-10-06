
import numpy
import theano
import theano.sandbox.cuda
import theano.tensor as T
from theano.gof.opt import OpSub
from theano.compile import optdb
from theano import gof


def multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left=0, pad_right=0, idx_dim=0, batch_dim=1):
  """
  :param array: ndarray, at least 2D. symbolic
  :param start_idxs: ndarray, 1D. symbolic. can be float (for gpu)
  :param batch_lens: ndarray, 1D. symbolic. len of each batch. can be float (for gpu)
  :param beam_width: scalar. symbolic.
  :param wrap_mode: "wrap_around" or "pad". static.
  :param idx_dim: int. where to apply each start_idxs[i]. static.
  :param batch_dim: the same dim as in start_idxs. static.
  :param pad_value: used in wrap_mode "pad". automatically broadcasted. symbolic.
  :return: ndarray like array, but shape[idx_dim] == beam_width

  See also `_naive_multi_batch_beam` for one naive reference implementation.
  """
  assert array.ndim >= 2
  assert start_idxs.ndim == 1
  assert batch_lens.ndim == 1
  assert idx_dim < array.ndim
  assert batch_dim < array.ndim
  assert idx_dim != batch_dim
  assert wrap_mode in ("wrap_around", "pad")

  op = MultiBatchBeamOp(wrap_mode, idx_dim, batch_dim)
  return op(array, start_idxs, batch_lens, beam_width, pad_left, pad_right)


def _naive_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left=0, pad_right=0, idx_dim=0, batch_dim=1):
  assert array.ndim >= 2
  assert start_idxs.ndim == 1
  assert batch_lens.ndim == 1
  assert idx_dim < array.ndim
  assert batch_dim < array.ndim
  assert idx_dim != batch_dim
  n_batch = array.shape[batch_dim]
  assert start_idxs.shape == (n_batch, )
  assert batch_lens.shape == (n_batch, )
  pad_left = numpy.asarray(pad_left)
  pad_left_bc = pad_left.reshape(*([1] * (array.ndim - pad_left.ndim) + list(pad_left.shape)))
  pad_right = numpy.asarray(pad_right)
  pad_right_bc = pad_right.reshape(*([1] * (array.ndim - pad_right.ndim) + list(pad_right.shape)))

  if idx_dim != 0: raise NotImplementedError  # This is usually the time dim.
  if batch_dim != 1: raise NotImplementedError
  # Thus, array is usually in format (time,batch,dim).

  beam = numpy.zeros((beam_width, n_batch) + array.shape[2:], dtype=array.dtype)
  for i0 in range(beam_width):
    for i1 in range(n_batch):
      idx = start_idxs[i1] + i0
      if wrap_mode == "wrap_around":
        idx = idx % batch_lens[i1]
      elif wrap_mode == "pad":
        if idx < 0:
          beam[i0, i1] = pad_left_bc
          continue
        elif idx >= batch_lens[i1]:
          beam[i0, i1] = pad_right_bc
          continue
      beam[i0, i1] = array[idx, i1]
  return beam


def _theano_cpu_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left=0, pad_right=0, idx_dim=0, batch_dim=1):
  array = T.as_tensor(array)
  start_idxs = T.as_tensor(start_idxs)
  if start_idxs.dtype.startswith("float"):
    start_idxs = T.iround(start_idxs)
  batch_lens = T.as_tensor(batch_lens)
  if batch_lens.dtype.startswith("float"):
    batch_lens = T.iround(batch_lens)
  beam_width = T.as_tensor(beam_width)
  if beam_width.dtype.startswith("float"):
    beam_width = T.iround(beam_width)
  pad_left = T.as_tensor(pad_left)
  pad_right = T.as_tensor(pad_right)
  assert array.ndim >= 2
  assert start_idxs.ndim == 1
  assert batch_lens.ndim == 1
  assert beam_width.ndim == 0
  assert idx_dim < array.ndim
  assert batch_dim < array.ndim
  assert idx_dim != batch_dim
  n_batch = array.shape[batch_dim]

  if idx_dim != 0: raise NotImplementedError
  if batch_dim != 1: raise NotImplementedError
  if wrap_mode != "wrap_around": raise NotImplementedError

  idxs_0 = start_idxs.dimshuffle('x', 0)  # (beam,batch)
  idxs = idxs_0 + T.arange(beam_width).dimshuffle(0, 'x')  # (beam,batch)
  idxs_wrapped = idxs % batch_lens.dimshuffle('x', 0)  # (beam,batch)
  batches = T.arange(n_batch)  # (batch,)
  beam = array[idxs_wrapped[:, batches], batches]  # (beam,batch,...)
  if wrap_mode == "wrap_around":
    pass  # Done that.
  elif wrap_mode == "pad":
    cond_left = T.lt(idxs, 0)  # (beam,batch)
    cond_right = T.ge(idxs, batch_lens.dimshuffle('x', 0))  # (beam,batch)
    cond_left_bc = cond_left.dimshuffle(beam_width, n_batch, *([1] * (array.ndim - 2)))
    cond_right_bc = cond_right.dimshuffle(beam_width, n_batch, *([1] * (array.ndim - 2)))
    pad_left_bc = pad_left.dimshuffle(*(['x'] * (array.ndim - pad_left.ndim) +
                                        [pad_left.shape[i] for i in range(pad_left.ndim)]))
    pad_right_bc = pad_left.dimshuffle(*(['x'] * (array.ndim - pad_right.ndim) +
                                         [pad_right.shape[i] for i in range(pad_right.ndim)]))
    beam = T.switch(cond_left_bc, beam, T.cast(pad_left_bc, dtype=array.dtype))
    beam = T.switch(cond_right_bc, beam, T.cast(pad_right_bc, dtype=array.dtype))
  else:
    raise Exception("MultiBatchBeam: unknown wrap mode: %r" % wrap_mode)
  return beam


def _naive_multi_batch_beam_grad(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left=0, pad_right=0, idx_dim=0, batch_dim=1, output_grad=None):
  assert array.ndim >= 2
  assert start_idxs.ndim == 1
  assert batch_lens.ndim == 1
  assert idx_dim < array.ndim
  assert batch_dim < array.ndim
  assert idx_dim != batch_dim
  n_batch = array.shape[batch_dim]
  assert start_idxs.shape == (n_batch, )
  assert batch_lens.shape == (n_batch, )
  D_beam = output_grad
  pad_left = numpy.asarray(pad_left)
  pad_right = numpy.asarray(pad_right)

  if idx_dim != 0: raise NotImplementedError  # This is usually the time dim.
  if batch_dim != 1: raise NotImplementedError
  assert D_beam.shape == (beam_width, n_batch) + array.shape[2:]
  # Thus, array is usually in format (time,batch,dim).

  D_array = numpy.zeros_like(array, dtype="float32")
  if wrap_mode == "pad":
    D_pad_left = numpy.zeros(array.shape[2:], dtype="float32")
    D_pad_right = numpy.zeros(array.shape[2:], dtype="float32")
  else:
    D_pad_left = D_pad_right = None

  for i0 in range(beam_width):
    for i1 in range(n_batch):
      idx = start_idxs[i1] + i0
      if wrap_mode == "wrap_around":
        idx = idx % batch_lens[i1]
      elif wrap_mode == "pad":
        if idx < 0:
          D_pad_left += D_beam[i0, i1]
          continue
        if idx >= batch_lens[i1]:
          D_pad_right += D_beam[i0, i1]
          continue
      D_array[idx, i1] = D_beam[i0, i1]

  if wrap_mode == "pad":
    if D_pad_left.ndim > pad_left.ndim:
      D_pad_left = numpy.sum(D_pad_left, axis=tuple(range(D_pad_left.ndim - pad_left.ndim)))
    if D_pad_right.ndim > pad_right.ndim:
      D_pad_right = numpy.sum(D_pad_right, axis=tuple(range(D_pad_right.ndim - pad_right.ndim)))
    assert D_pad_left.shape == pad_left.shape
    assert D_pad_right.shape == pad_right.shape
  return D_array, D_pad_left, D_pad_right


def _theano_cpu_multi_batch_beam_grad(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_left=0, pad_right=0, idx_dim=0, batch_dim=1, output_grad=None):
  # Note: This is slow and hacky. This will create an index-array of the size of the original array.
  # This is calculated on the CPU. The subtensor then can be done on the GPU, but we should avoid the first part.
  D_beam = output_grad
  prod_array_shape = T.prod(array.shape)
  prod_pad_left_shape = T.prod(pad_left.shape)
  prod_pad_right_shape = T.prod(pad_right.shape)
  D_array_tmp_size = prod_array_shape
  if wrap_mode == "pad":
    D_array_tmp_size += prod_pad_left_shape + prod_pad_right_shape
  D_array_tmp_flat = T.zeros([D_array_tmp_size], dtype="float32")  # with pad values
  if wrap_mode == "pad":
    # Calculate the indices for D_pad_left/D_pad_right in D_array_tmp_flat.
    pad_left_idxs = T.arange(prod_pad_left_shape) + prod_array_shape
    pad_right_idxs = T.arange(prod_pad_right_shape) + prod_array_shape + prod_pad_left_shape
    pad_left_idxs = pad_left_idxs.reshape(pad_left.shape)
    pad_right_idxs = pad_right_idxs.reshape(pad_right.shape)
  else:
    pad_left_idxs = pad_right_idxs = 0
  all_idxs = T.arange(T.prod(array.shape)).reshape(array.shape)
  idxs = multi_batch_beam(array=all_idxs, start_idxs=start_idxs, batch_lens=batch_lens, beam_width=beam_width,
                          wrap_mode=wrap_mode,
                          pad_left=pad_left_idxs, pad_right=pad_right_idxs,
                          idx_dim=idx_dim, batch_dim=batch_dim)
  D_array_tmp_flat = T.inc_subtensor(D_array_tmp_flat[idxs.flatten()], D_beam.flatten())
  if wrap_mode == "pad":
    D_array = D_array_tmp_flat[:prod_array_shape].reshape(array.shape)
    D_pad_left = D_array_tmp_flat[pad_left_idxs.flatten()].reshape(pad_left.shape)
    D_pad_right = D_array_tmp_flat[pad_right_idxs.flatten()].reshape(pad_right.shape)
  else:
    D_array = D_array_tmp_flat.reshape(array.shape)
    D_pad_left = D_pad_right = T.DisconnectedType()()

  return D_array, D_pad_left, D_pad_right


class MultiBatchBeamOp(theano.Op):
  __props__ = ("wrap_mode", "idx_dim", "batch_dim")

  def __init__(self, wrap_mode, idx_dim=0, batch_dim=1):
    super(MultiBatchBeamOp, self).__init__()
    self.wrap_mode = wrap_mode
    self.idx_dim = idx_dim
    self.batch_dim = batch_dim

  def make_node(self, array, start_idxs, batch_lens, beam_width, pad_left, pad_right):
    array = T.as_tensor_variable(array)
    start_idxs = T.as_tensor_variable(start_idxs)
    batch_lens = T.as_tensor_variable(batch_lens)
    beam_width = T.as_tensor_variable(beam_width)
    pad_left = T.as_tensor_variable(pad_left)
    pad_right = T.as_tensor_variable(pad_right)
    assert array.ndim >= 2
    assert start_idxs.ndim == 1
    assert batch_lens.ndim == 1
    assert beam_width.ndim == 0
    return theano.Apply(self, [array, start_idxs, batch_lens, beam_width, pad_left, pad_right], [array.type()])

  def perform(self, node, inputs, output_storage):
    array, start_idxs, batch_lens, beam_width, pad_left, pad_right = inputs
    beam_out, = output_storage
    n_batches = array.shape[self.batch_dim]
    assert start_idxs.shape[0] == n_batches
    assert batch_lens.shape[0] == n_batches
    if not start_idxs.dtype.name.startswith("int"):
      start_idxs = numpy.round(start_idxs).astype("int64")
    if not batch_lens.dtype.name.startswith("int"):
      batch_lens = numpy.round(batch_lens).astype("int64")
    if not beam_width.dtype.name.startswith("int"):
      beam_width = int(numpy.round(beam_width))
    pad_left_bc = numpy.asarray(pad_left).reshape(*([1] * (array.ndim - pad_left.ndim) + list(pad_left.shape)))
    pad_right_bc = numpy.asarray(pad_right).reshape(*([1] * (array.ndim - pad_right.ndim) + list(pad_right.shape)))

    idxs_bc = numpy.arange(beam_width).reshape(beam_width, 1)  # dimshuffle(0, 'x')  (beam,batch)
    start_idxs_bc = start_idxs.reshape(1, n_batches)  # dimshuffle('x', 0)  (beam,batch)
    idxs = idxs_bc + start_idxs_bc  # (beam,batch)
    batch_lens_bc = batch_lens.reshape(1, n_batches)  # dimshuffle('x', 0)  (beam,batch)
    idxs_wrapped = idxs % batch_lens_bc
    array_remaining_dims = sorted(set(range(array.ndim)) - set([self.idx_dim, self.batch_dim]))
    array_trans_dims_order = [self.idx_dim, self.batch_dim] + array_remaining_dims
    array_trans = array.transpose(*array_trans_dims_order)  # (time,batch,...)
    beam_trans = array_trans[idxs_wrapped, numpy.arange(n_batches)]  # (beam,batch,...)
    if self.wrap_mode == "wrap_around":
      pass  # We have done exactly that.
    elif self.wrap_mode == "pad":
      cond_left = idxs < 0  # (beam,batch)
      cond_right = idxs >= batch_lens_bc  # (beam,batch)
      cond_left_bc = cond_left.reshape(beam_width, n_batches, *([1] * len(array_remaining_dims)))
      cond_right_bc = cond_right.reshape(beam_width, n_batches, *([1] * len(array_remaining_dims)))
      beam_trans = numpy.where(cond_left_bc, numpy.cast[array.dtype](pad_left_bc), beam_trans)
      beam_trans = numpy.where(cond_right_bc, numpy.cast[array.dtype](pad_right_bc), beam_trans)
    else:
      raise Exception("MultiBatchBeam: unknown wrap mode: %r" % self.wrap_mode)
    beam = beam_trans.transpose(*map(array_trans_dims_order.index, range(array.ndim)))
    beam_out[0] = beam

  def infer_shape(self, node, input_shapes):
    array, start_idxs, batch_lens, beam_width, pad_left, pad_right = node.inputs
    beam_width = T.cast(beam_width, dtype="int64")
    array_shape, start_idxs_shape, batch_lens_shape, beam_width_shape, pad_left_shape, pad_right_shape = input_shapes
    beam_shape = [beam_width if i == self.idx_dim else array_shape[i] for i in range(len(array_shape))]
    return [tuple(beam_shape)]

  def grad(self, inputs, output_grads):
    array, start_idxs, batch_lens, beam_width, pad_left, pad_right = inputs
    D_beam, = output_grads

    if not isinstance(pad_left, theano.Constant):
      raise NotImplementedError("D_pad_left not implemented...")
    if not isinstance(pad_right, theano.Constant):
      raise NotImplementedError("D_pad_right not implemented...")

    grad_op = MultiBatchBeamGradAddOp(wrap_mode=self.wrap_mode, zero_with_shape=True, array_ndim=array.ndim,
                                      idx_dim=self.idx_dim, batch_dim=self.batch_dim)
    D_array = grad_op(array.shape, start_idxs, batch_lens, beam_width, D_beam)

    if self.wrap_mode == "wrap_around":
      D_pad_left = D_pad_right = T.DisconnectedType()()
    elif self.wrap_mode == "pad":
      D_pad_left = D_pad_right = T.DisconnectedType()()
      # XXX...
      # D_pad_left = T.zeros(pad_left.shape, dtype="float32")
      # D_pad_right = T.zeros(pad_right.shape, dtype="float32")
    else:
      assert False, self.wrap_mode

    # Those are all discrete values. The gradient is 0 almost everywhere, except for integers where it is not defined.
    D_start_idxs = T.DisconnectedType()()
    D_batch_lens = T.DisconnectedType()()
    D_beam_width = T.DisconnectedType()()

    return [D_array, D_start_idxs, D_batch_lens, D_beam_width, D_pad_left, D_pad_right]

  def connection_pattern(self, node):
    # Only the gradient of the first input (array) will be connected.
    # All others are disconnected (because round() or floor() is used on them.).
    pattern = [[True], [False], [False], [False], [False], [False]]
    # if self.wrap_mode == "pad":  # XXX... we assume constant for now
    #   pattern[-2:] = [[True], [True]]
    assert len(pattern) == len(node.inputs)
    return pattern


def _len_of_shape(shape):
  if isinstance(shape, (list,tuple)):
    return len(shape)
  if isinstance(shape, T.Apply):
    if isinstance(shape.op, T.Shape):
      assert len(shape.inputs) == 1
      return shape.inputs[0].ndim
  raise NotImplementedError("cannot handle %r" % shape)


inplace_increment = None
if theano.config.cxx:
  import theano.gof.cutils  # needed to import cutils_ext
  try:
    from cutils_ext.cutils_ext import inplace_increment
  except ImportError:
    pass


class MultiBatchBeamGradAddOp(theano.Op):
  __props__ = ("wrap_mode", "idx_dim", "batch_dim", "inplace", "zero_with_shape", "array_ndim")

  def __init__(self, wrap_mode, idx_dim=0, batch_dim=1, inplace=False, zero_with_shape=False, array_ndim=None):
    """
    (D_array / D_array_shape, start_idxs, batch_lens, beam_width, D_beam) -> D_array + grad

    :param str wrap_mode: "wrap_around" or "pad"
    :param int idx_dim: usually that's time dim
    :param int batch_dim: batch dim
    :param bool inplace: operate inplace on input
    :param bool zero_with_shape: we get D_array_shape as the first input and init D_array with zero
    :param int array_ndim: ndim of array/D_array. needed for zero_with_shape
    """
    super(MultiBatchBeamGradAddOp, self).__init__()
    self.wrap_mode = wrap_mode
    self.idx_dim = idx_dim
    self.batch_dim = batch_dim
    self.inplace = inplace
    self.zero_with_shape = zero_with_shape
    self.array_ndim = array_ndim
    if zero_with_shape:
      assert not inplace
      assert array_ndim > 0
    if inplace:
      # We operate inplace on D_array.
      self.destroy_map = {0: [0]}

  def make_node(self, D_array_or_shape, start_idxs, batch_lens, beam_width, D_beam):
    # XXX: Currently without D_pad_left and D_pad_right.
    start_idxs = T.as_tensor_variable(start_idxs)
    batch_lens = T.as_tensor_variable(batch_lens)
    beam_width = T.as_tensor_variable(beam_width)
    D_beam = T.as_tensor_variable(D_beam)
    if self.zero_with_shape:
      D_array_ndim = self.array_ndim
    else:
      D_array_ndim = D_array_or_shape.ndim
    assert start_idxs.ndim == 1
    assert batch_lens.ndim == 1
    assert beam_width.ndim == 0
    return theano.Apply(self,
                        [D_array_or_shape, start_idxs, batch_lens, beam_width, D_beam],
                        [T.TensorType("float32", (False,) * D_array_ndim)("D_array")])

  def infer_shape(self, node, input_shapes):
    if self.zero_with_shape:
      D_array_ndim = self.array_ndim
      shape = node.inputs[0]  # This is symbolic.
      return [[shape[i] for i in range(D_array_ndim)]]
    else:
      return [input_shapes[0]]

  def perform(self, node, inputs, output_storage):
    D_array_or_shape, start_idxs, batch_lens, beam_width, D_beam = inputs
    out_D_array, = output_storage
    if self.inplace:
      out_D_array[0] = D_array = D_array_or_shape
    elif self.zero_with_shape:
      out_D_array[0] = D_array = numpy.zeros(D_array_or_shape, "float32")
    else:
      out_D_array[0] = D_array = D_array_or_shape.copy()
    n_batches = D_array.shape[self.batch_dim]

    idxs_bc = numpy.arange(beam_width).reshape(beam_width, 1)  # dimshuffle(0, 'x')  (beam,batch)
    start_idxs_bc = start_idxs.reshape(1, n_batches)  # dimshuffle('x', 0)  (beam,batch)
    idxs = idxs_bc + start_idxs_bc  # (beam,batch)
    batch_lens_bc = batch_lens.reshape(1, n_batches)  # dimshuffle('x', 0)  (beam,batch)
    assert idxs.shape == D_beam.shape[:2]

    idxs = idxs.astype("int32")
    if self.wrap_mode == "wrap_around":
      idxs = idxs % batch_lens_bc.astype("int32")
    elif self.wrap_mode == "pad":
      idxs = numpy.where(idxs >= batch_lens_bc, -1, idxs)
      cond_bc = (idxs < 0).reshape(*(D_beam.shape[:2] + (1,) * (D_beam.ndim - 2)))
      D_beam = numpy.where(cond_bc, numpy.float32(0), D_beam)  # XXX: ignore padding part
      idxs = numpy.where(idxs < 0, 0, idxs)
    else:
      assert False, self.wrap_mode

    if self.idx_dim != 0: raise NotImplementedError  # TODO...
    if self.batch_dim != 1: raise NotImplementedError  # TODO...
    # In Numpy, x[idx] += y doesn't work if the same index is present
    # many times: it does it only once. Is it a bug? In any case, for
    # this reason we implement our own 'inc' iteration.
    # See also theano.tensor.subtensor.AdvancedIncSubtensor documentation.
    if inplace_increment is None: raise NotImplementedError("need Numpy 1.8 or later")
    # This is like D_array_and_pad[idxs, numpy.arange(n_batches)] += D_beam .
    inplace_increment(D_array, (idxs, numpy.arange(n_batches)), D_beam)


# https://deeplearning.net/software/theano/extending/optimization.html
# See also theano/compile/mode.py for reference about the position priority numbers.
# After priority 50 we can do destructive inplace operations.

@gof.local_optimizer([T.add])
def add_merge_MultiBatchBeamGradAddOp(node):
  if node.op != T.add: return False
  if len(node.inputs) < 2: return False
  grad_op_idx = None
  grad_op_v = None
  grad_op = None
  for i, input in enumerate(node.inputs):
    if input.owner and isinstance(input.owner.op, MultiBatchBeamGradAddOp):
      grad_op = input.owner.op
      if not grad_op.inplace:  # we cannot merge when we operate inplace on it
        grad_op_v = input
        grad_op_idx = i
        break
  if grad_op_idx is None: return False
  sum_inputs = [node.inputs[i] for i in range(len(node.inputs)) if i != grad_op_idx]
  if grad_op.zero_with_shape:
    # Make new grad_op without zero_with_shape.
    kwargs = {k: getattr(grad_op, k) for k in grad_op.__props__}
    kwargs["zero_with_shape"] = False
    grad_op = grad_op.__class__(**kwargs)
  else:
    old_grad_op_input0 = grad_op_v.owner.inputs[0]
    sum_inputs = [old_grad_op_input0] + sum_inputs
  assert len(sum_inputs) > 0
  if len(sum_inputs) == 1:
    new_grad_op_input0 = sum_inputs[0]
  else:
    new_grad_op_input0 = T.add(*sum_inputs)
  new_grad_op_inputs = [new_grad_op_input0] + grad_op_v.owner.inputs[1:]
  new_v = grad_op(*new_grad_op_inputs)
  return [new_v]

optdb.register('add_merge_MultiBatchBeamGradAddOp',
               gof.TopoOptimizer(add_merge_MultiBatchBeamGradAddOp),
               0.1, 'fast_run')


@gof.local_optimizer([MultiBatchBeamGradAddOp], inplace=True)
def inplace_MultiBatchBeamGradAddOp(node):
  if isinstance(node.op, MultiBatchBeamGradAddOp) and not node.op.inplace and not node.op.zero_with_shape:
    kwargs = {k: getattr(node.op, k) for k in node.op.__props__}
    kwargs["inplace"] = True
    new_op = node.op.__class__(**kwargs)
    new_v = new_op(*node.inputs)
    return [new_v]
  return False

optdb.register('inplace_MultiBatchBeamGradAddOp',
               gof.TopoOptimizer(inplace_MultiBatchBeamGradAddOp
                                 , failure_callback=gof.TopoOptimizer.warn_inplace
                                 ),
               76,  # after ScanInplaceOptimizer
               'fast_run', 'inplace')


class GpuMultiBatchBeamOp(theano.sandbox.cuda.GpuOp):
  pass  # TODO
