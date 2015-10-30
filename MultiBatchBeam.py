
import numpy
import theano
import theano.sandbox.cuda
import theano.tensor as T


def multi_batch_beam(array, start_idxs, batch_lens, beam_width, wrap_mode, idx_dim=0, batch_dim=1):
  """
  :param array: ndarray, at least 2D. symbolic
  :param start_idxs: ndarray, 1D. symbolic. can be float (for gpu)
  :param batch_lens: ndarray, 1D. symbolic. len of each batch. can be float (for gpu)
  :param beam_width: scalar. symbolic.
  :param wrap_mode: "wrap_around" or "pad_zero". static.
  :param idx_dim: int. where to apply each start_idxs[i]. static.
  :param batch_dim: the same dim as in start_idxs. static.
  :return: ndarray like array, but shape[idx_dim] == beam_width
  """
  assert array.ndim >= 2
  assert start_idxs.ndim == 1
  assert batch_lens.ndim == 1
  assert idx_dim < array.ndim
  assert batch_dim < array.ndim
  assert idx_dim != batch_dim
  assert wrap_mode in ("wrap_around", "pad_zero")

  op = MultiBatchBeamOp(wrap_mode, idx_dim, batch_dim)
  return op(array, start_idxs, batch_lens, beam_width)


class MultiBatchBeamOp(theano.Op):
  __props__ = ("wrap_mode", "idx_dim", "batch_dim")

  def __init__(self, wrap_mode, idx_dim=0, batch_dim=1):
    super(MultiBatchBeamOp, self).__init__()
    self.wrap_mode = wrap_mode
    self.idx_dim = idx_dim
    self.batch_dim = batch_dim

  def make_node(self, array, start_idxs, batch_lens, beam_width):
    array = T.as_tensor_variable(array)
    start_idxs = T.as_tensor_variable(start_idxs)
    batch_lens = T.as_tensor_variable(batch_lens)
    beam_width = T.as_tensor_variable(beam_width)
    assert array.ndim >= 2
    assert start_idxs.ndim == 1
    assert batch_lens.ndim == 1
    assert beam_width.ndim == 0
    return theano.Apply(self, [array, start_idxs, batch_lens, beam_width], [array.type()])

  def perform(self, node, inputs, output_storage):
    array, start_idxs, batch_lens, beam_width = inputs
    beam_out, = output_storage
    n_idx = array.shape[self.idx_dim]  # usually that is the time-dim or frame-dim
    n_beam = int(beam_width)
    n_batches = array.shape[self.batch_dim]
    assert start_idxs.shape[0] == n_batches
    assert batch_lens.shape[0] == n_batches
    if not start_idxs.dtype.name.startswith("int"):
      start_idxs = start_idxs.astype("int64")
    if not batch_lens.dtype.name.startswith("int"):
      batch_lens = batch_lens.astype("int64")

    idxs_bc = numpy.arange(n_beam).reshape(n_beam, 1)  # dimshuffle(0, 'x')  (beam,batch)
    start_idxs_bc = start_idxs.reshape(1, n_batches)  # dimshuffle('x', 0)  (beam,batch)
    idxs = idxs_bc + start_idxs_bc  # (beam,batch)
    batch_lens_bc = batch_lens.reshape(1, n_batches)  # dimshuffle('x', 0)  (beam,batch)
    idxs_wrapped = idxs % batch_lens_bc
    idxs_flat_offsets = numpy.tile(numpy.arange(n_batches), n_beam)  # (beam*batch)
    idxs_wrapped_flat = idxs_wrapped.reshape(n_beam * n_batches)
    idxs_wrapped_flat = idxs_wrapped_flat * n_batches + idxs_flat_offsets  # (beam*batch)
    array_remaining_dims = sorted(set(range(array.ndim)) - set([self.idx_dim, self.batch_dim]))
    array_trans_dims_order = [self.idx_dim, self.batch_dim] + array_remaining_dims
    array_trans = array.transpose(*array_trans_dims_order)
    array_trans_flat = array_trans.reshape(n_idx * n_batches, *array_trans.shape[2:])
    beam_trans_flat = array_trans_flat[idxs_wrapped_flat]
    beam_trans = beam_trans_flat.reshape(n_beam, n_batches, *array_trans.shape[2:])
    beam = beam_trans.transpose(*map(array_trans_dims_order.index, range(array.ndim)))
    if self.wrap_mode == "pad_zero":
      pass  # TODO...
    beam_out[0] = beam

  def infer_shape(self, node, input_shapes):
    array, start_idxs, batch_lens, beam_width = input_shapes
    return [tuple(beam_width if i == self.idx_dim else array[i] for i in range(len(array)))]

  def grad(self, inputs, output_grads):
    array, start_idxs, batch_lens, beam_width = inputs
    D_beam, = output_grads

    # TODO... HACK, working only for wrap_around
    zero_array_flat = T.zeros_like(array).flatten()
    all_idxs = T.arange(T.prod(array.shape)).reshape(array.shape)
    assert self.wrap_mode == "wrap_around"
    idxs = multi_batch_beam(all_idxs, start_idxs, batch_lens, beam_width, self.wrap_mode, self.idx_dim, self.batch_dim)
    D_array = T.set_subtensor(zero_array_flat[idxs.flatten()], D_beam.flatten())

    # Those are all discrete values. The gradient is 0 almost everywhere, except for integers where it is not defined.
    D_start_idxs = T.zeros_like(start_idxs)
    D_batch_lens = T.zeros_like(batch_lens)
    D_beam_width = T.zeros_like(beam_width)
    return [D_array, D_start_idxs, D_batch_lens, D_beam_width]


class GpuMultiBatchBeamOp(theano.sandbox.cuda.GpuOp):
  pass  # TODO
