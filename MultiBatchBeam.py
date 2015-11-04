
import numpy
import theano
import theano.sandbox.cuda
import theano.tensor as T


def multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_value=0, idx_dim=0, batch_dim=1):
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
  return op(array, start_idxs, batch_lens, beam_width, pad_value)


def _naive_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_value=0, idx_dim=0, batch_dim=1):
  assert array.ndim >= 2
  assert start_idxs.ndim == 1
  assert batch_lens.ndim == 1
  assert idx_dim < array.ndim
  assert batch_dim < array.ndim
  assert idx_dim != batch_dim
  n_batch = array.shape[batch_dim]
  assert start_idxs.shape == (n_batch, )
  assert batch_lens.shape == (n_batch, )
  pad_value = numpy.asarray(pad_value)
  pad_value_bc = pad_value.reshape(*([1] * (array.ndim - pad_value.ndim) + list(pad_value.shape)))

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
        if idx < 0 or idx >= batch_lens[i1]:
          beam[i0, i1] = pad_value_bc
          continue
      beam[i0, i1] = array[idx, i1]
  return beam


def _theano_cpu_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_value=0, idx_dim=0, batch_dim=1):
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
  pad_value = T.as_tensor(pad_value)
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
    cond = T.and_(T.ge(idxs, 0), T.lt(idxs, batch_lens.dimshuffle('x', 0)))  # (beam,batch)
    cond_bc = cond.dimshuffle(beam_width, n_batch, *([1] * (array.ndim - 2)))
    pad_value_bc = pad_value.dimshuffle(*(['x'] * (array.ndim - pad_value.ndim) +
                                          [pad_value.shape[i] for i in range(pad_value.ndim)]))
    beam = T.switch(cond_bc, beam, T.cast(pad_value_bc, dtype=array.dtype))
  else:
    raise Exception("MultiBatchBeam: unknown wrap mode: %r" % wrap_mode)
  return beam


def _simplified_numpy_multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_value=0, idx_dim=0, batch_dim=1):
  start_idxs = numpy.round(start_idxs).astype("int64")
  batch_lens = numpy.round(batch_lens).astype("int64")
  beam_width = int(numpy.round(beam_width))
  assert array.ndim >= 2
  assert start_idxs.ndim == 1
  assert batch_lens.ndim == 1
  assert idx_dim < array.ndim
  assert batch_dim < array.ndim
  assert idx_dim != batch_dim
  n_batch = array.shape[batch_dim]
  assert start_idxs.shape == (n_batch, )
  assert batch_lens.shape == (n_batch, )

  if idx_dim != 0: raise NotImplementedError  # This is usually the time dim.
  if batch_dim != 1: raise NotImplementedError
  if wrap_mode != "wrap_around": raise NotImplementedError
  # Thus, array is usually in format (time,batch,dim).

  idxs_bc = numpy.arange(beam_width).reshape(beam_width, 1)  # dimshuffle(0, 'x')  (beam,batch)
  start_idxs_bc = start_idxs.reshape(1, n_batch)  # dimshuffle('x', 0)  (beam,batch)
  idxs = idxs_bc + start_idxs_bc  # (beam,batch)
  batch_lens_bc = batch_lens.reshape(1, n_batch)  # dimshuffle('x', 0)  (beam,batch)
  idxs_wrapped = idxs % batch_lens_bc
  assert idxs_wrapped.shape[1] == array.shape[1]
  beam = array[idxs_wrapped, numpy.arange(n_batch)]
  return beam


def _naive_multi_batch_beam_grad(array, start_idxs, batch_lens, beam_width, wrap_mode, pad_value=0, idx_dim=0, batch_dim=1, output_grad=None):
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

  if idx_dim != 0: raise NotImplementedError  # This is usually the time dim.
  if batch_dim != 1: raise NotImplementedError
  assert D_beam.shape == (beam_width, n_batch) + array.shape[2:]
  # Thus, array is usually in format (time,batch,dim).

  D_array = numpy.zeros_like(array, dtype=D_beam.dtype)
  for i0 in range(beam_width):
    for i1 in range(n_batch):
      idx = start_idxs[i1] + i0
      if wrap_mode == "wrap_around":
        idx = idx % batch_lens[i1]
      elif wrap_mode == "pad":
        if idx < 0 or idx >= batch_lens[i1]:
          continue
      D_array[idx, i1] = D_beam[i0, i1]
  return D_array



class MultiBatchBeamOp(theano.Op):
  __props__ = ("wrap_mode", "idx_dim", "batch_dim")

  def __init__(self, wrap_mode, idx_dim=0, batch_dim=1):
    super(MultiBatchBeamOp, self).__init__()
    self.wrap_mode = wrap_mode
    self.idx_dim = idx_dim
    self.batch_dim = batch_dim

  def make_node(self, array, start_idxs, batch_lens, beam_width, pad_value):
    array = T.as_tensor_variable(array)
    start_idxs = T.as_tensor_variable(start_idxs)
    batch_lens = T.as_tensor_variable(batch_lens)
    beam_width = T.as_tensor_variable(beam_width)
    pad_value = T.as_tensor_variable(pad_value)
    assert array.ndim >= 2
    assert start_idxs.ndim == 1
    assert batch_lens.ndim == 1
    assert beam_width.ndim == 0
    return theano.Apply(self, [array, start_idxs, batch_lens, beam_width, pad_value], [array.type()])

  def perform(self, node, inputs, output_storage):
    array, start_idxs, batch_lens, beam_width, pad_value = inputs
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
    pad_value_bc = numpy.asarray(pad_value).reshape(*([1] * (array.ndim - pad_value.ndim) + list(pad_value.shape)))

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
      cond = numpy.logical_and(idxs >= 0, idxs < batch_lens_bc)  # (beam,batch)
      cond_bc = cond.reshape(beam_width, n_batches, *([1] * len(array_remaining_dims)))
      beam_trans = numpy.where(cond_bc, beam_trans, numpy.cast[array.dtype](pad_value_bc))
    else:
      raise Exception("MultiBatchBeam: unknown wrap mode: %r" % self.wrap_mode)
    beam = beam_trans.transpose(*map(array_trans_dims_order.index, range(array.ndim)))
    beam_out[0] = beam

  def infer_shape(self, node, input_shapes):
    array, start_idxs, batch_lens, beam_width, pad_value = node.inputs
    beam_width = T.cast(beam_width, dtype="int64")
    array_shape, start_idxs_shape, batch_lens_shape, beam_width_shape, pad_value_shape = input_shapes
    beam_shape = [beam_width if i == self.idx_dim else array_shape[i] for i in range(len(array_shape))]
    return [tuple(beam_shape)]

  def grad(self, inputs, output_grads):
    array, start_idxs, batch_lens, beam_width, pad_value = inputs
    D_beam, = output_grads

    zero_array_flat = T.zeros_like(array, dtype="float32").flatten()
    all_idxs = T.arange(T.prod(array.shape)).reshape(array.shape)
    idxs = multi_batch_beam(all_idxs, start_idxs, batch_lens, beam_width, self.wrap_mode, -1, self.idx_dim, self.batch_dim)
    if self.wrap_mode == "wrap_around":
      idxs_gt0 = idxs
    else:
      assert self.wrap_mode == "pad"
      idxs_gt0 = T.switch(T.ge(idxs, 0), idxs, 0)
    D_array_flat = T.set_subtensor(zero_array_flat[idxs_gt0.flatten()], D_beam.flatten())
    if self.wrap_mode != "wrap_around":
      D_array_flat = T.set_subtensor(D_array_flat[idxs_gt0.flatten()],
                                     T.switch(T.lt(idxs.flatten(), 0),
                                              numpy.float32(0),
                                              D_array_flat[idxs_gt0.flatten()]))
    D_array = D_array_flat.reshape(array.shape)

    # Those are all discrete values. The gradient is 0 almost everywhere, except for integers where it is not defined.
    D_start_idxs = T.DisconnectedType()()
    D_batch_lens = T.DisconnectedType()()
    D_beam_width = T.DisconnectedType()()
    D_pad_value = T.DisconnectedType()()  # TODO: that's not really correct...
    return [D_array, D_start_idxs, D_batch_lens, D_beam_width, D_pad_value]

  def connection_pattern(self, node):
    # Only the gradient of the first input (array) will be connected.
    # All others are disconnected (because round() or floor() is used on them.).
    # (TODO: Also pad_value is connected.)
    return [[True]] + [[False]] * len(node.inputs[1:])


class GpuMultiBatchBeamOp(theano.sandbox.cuda.GpuOp):
  pass  # TODO
