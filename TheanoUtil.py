
import theano
import theano.sandbox.cuda
import theano.tensor as T
from theano.compile import ViewOp
import numpy


def time_batch_make_flat(val):
  """
  :rtype val: theano.Variable
  :rtype: theano.Variable

  Will flatten the first two dimensions and leave the others as is.
  """
  assert val.ndim > 1
  s0 = val.shape[0] * val.shape[1]
  newshape = [s0] + [val.shape[i] for i in range(2, val.ndim)]
  return T.reshape(val,
                   newshape,
                   ndim=val.ndim - 1,
                   name="flat_%s" % val.name)


def class_idx_seq_to_1_of_k(seq, num_classes, dtype="float32"):
  """
  :param theano.Variable seq: ndarray with indices
  :param int | theano.Variable num_classes: number of classes
  :param str dtype: eg "float32"
  :rtype: theano.Variable
  :returns ndarray with one added dimension of size num_classes.
  That is the one-hot-encoding.
  This function is like theano.tensor.extra_ops.to_one_hot
  but we can handle multiple dimensions.
  """
  shape = [seq.shape[i] for i in range(seq.ndim)] + [num_classes]
  eye = T.eye(num_classes, dtype=dtype)
  m = eye[T.cast(seq, 'int32')].reshape(shape)
  return m


def tiled_eye(n1, n2, dtype="float32"):
  r1 = T.maximum((n1 - 1) / n2 + 1, 1)
  r2 = T.maximum((n2 - 1) / n1 + 1, 1)
  small_eye = T.eye(T.minimum(n1, n2), dtype=dtype)
  tiled_big = T.tile(small_eye, (r1, r2))
  tiled_part = tiled_big[:n1,:n2]
  return tiled_part


def opt_contiguous_on_gpu(x):
  if theano.sandbox.cuda.cuda_enabled:
    return theano.sandbox.cuda.basic_ops.gpu_contiguous(x)
  return x


def windowed_batch(source, window):
  assert source.ndim == 3  # (time,batch,dim). not sure how to handle other cases
  n_time = source.shape[0]
  n_batch = source.shape[1]
  n_dim = source.shape[2]
  w_right = window / 2
  w_left = window - w_right - 1
  pad_left = T.zeros((w_left, n_batch, n_dim), dtype=source.dtype)
  pad_right = T.zeros((w_right, n_batch, n_dim), dtype=source.dtype)
  padded = T.concatenate([pad_left, source, pad_right], axis=0)  # shape[0] == n_time + window - 1
  tiled = T.tile(padded, (1, 1, window))  # shape[2] == n_dim * window
  tiled_reshape = T.reshape(tiled, ((n_time + window - 1), n_batch, window, n_dim))
  # We want to shift every dim*time block by one to the left.
  # To do this, we interpret that we have one more time frame (i.e. n_time+window).
  # We have to do some dimshuffling so that we get the right layout, then we can flatten,
  # add some padding, and then dimshuffle it back.
  # Then we can take out the first n_time frames.
  tiled_dimshuffle = tiled_reshape.dimshuffle(2, 0, 1, 3)  # (window,n_time+window-1,batch,dim)
  tiled_flat = T.flatten(tiled_dimshuffle)
  rem = n_batch * n_dim * window
  tiled_flat_pad_right = T.concatenate([tiled_flat, T.zeros((rem,), dtype=source.dtype)])
  tiled_reshape_shift = T.reshape(tiled_flat_pad_right, (window, n_time + window, n_batch, n_dim))  # add time frame
  final_dimshuffle = tiled_reshape_shift.dimshuffle(1, 2, 0, 3)  # (n_time+window,batch,window,dim)
  final_sub = final_dimshuffle[:n_time]  # (n_time,batch,window,dim)
  final_concat_dim = final_sub.reshape((n_time, n_batch, window * n_dim))
  return final_concat_dim


def slice_for_axis(axis, s):
  return (slice(None),) * (axis - 1) + (s,)


def downsample(source, axis, factor, method="average"):
  assert factor == int(factor), "factor is expected to be an int"
  factor = int(factor)
  # make shape[axis] a multiple of factor
  source = source[slice_for_axis(axis=axis, s=slice(0, (source.shape[axis] / factor) * factor))]
  # Add a temporary dimension as the factor.
  added_dim_shape = [source.shape[i] for i in range(source.ndim)]
  added_dim_shape = added_dim_shape[:axis] + [source.shape[axis] / factor, factor] + added_dim_shape[axis + 1:]
  source = T.reshape(source, added_dim_shape)
  if method == "average":
    return T.mean(source, axis=axis + 1)
  elif method == "max":
    return T.max(source, axis=axis + 1)
  elif method == "min":
    return T.min(source, axis=axis + 1)
  elif method == "concat": # concatenates in last dimension
    return T.reshape(source, added_dim_shape[:axis+1] + added_dim_shape[axis+2:-1] + [added_dim_shape[-1] * factor])
  elif method == "lstm":
    assert axis == 0
    return source
  elif method == "batch":
    assert axis == 0
    return source.dimshuffle(1,0,2,3).reshape((source.shape[1],source.shape[0]*source.shape[2],source.shape[3]))
  else:
    assert False, "unknown downsample method %r" % method


def upsample(source, axis, factor, method="nearest-neighbor", target_axis_len=None):
  if method == "nearest-neighbor":
    assert factor == int(factor), "factor is expected to be an int. not implemented otherwise yet."
    factor = int(factor)
    target = T.repeat(source, factor, axis=axis)
    if target_axis_len is not None:
      # We expect that we need to add a few frames. Just use the last frame.
      last = source[slice_for_axis(axis=axis, s=slice(-1, None))]
      target = pad(target, axis=axis, target_axis_len=target_axis_len, pad_value=last)
    return target
  else:
    assert False, "unknown upsample method %r" % method


def pad(source, axis, target_axis_len, pad_value=None):
  if pad_value is None:
    pad_value = T.zeros([source.shape[i] if i != axis else 1 for i in range(source.ndim)], dtype=source.dtype)
  num_missing = T.cast(target_axis_len, dtype="int32") - source.shape[axis]
  # There is some strange bug in Theano. If num_missing is 0, in some circumstances,
  # it crashes with Floating point exception.
  # Thus, do this workaround.
  num_missing = T.maximum(num_missing, 1)
  target = T.concatenate([source, T.repeat(pad_value, num_missing, axis=axis)], axis=axis)
  # Because of the workaround, we need this.
  target = target[slice_for_axis(axis=axis, s=slice(0, target_axis_len))]
  return target


def chunked_time_reverse(source, chunk_size):
  """
  :param source: >=1d array (time,...)
  :param chunk_size: int
  :return: like source
  Will not reverse the whole time-dim, but only every time-chunk.
  E.g. source=[0 1 2 3 4 5 6], chunk_size=3, returns [2 1 0 5 4 3 0].
  (Padded with 0, recovers original size.)
  """
  chunk_size = T.cast(chunk_size, dtype="int32")
  num_chunks = (source.shape[0] + chunk_size - 1) / chunk_size
  needed_time = num_chunks * chunk_size
  remaining_dims = [source.shape[i + 1] for i in range(source.ndim - 1)]
  padded_source = pad(source, axis=0, target_axis_len=needed_time)
  reshaped = padded_source.reshape([num_chunks, chunk_size] + remaining_dims)
  reshaped_rev = reshaped[:, ::-1]
  rev_correct_ndim = reshaped_rev.reshape([needed_time] + remaining_dims)
  return rev_correct_ndim[:source.shape[0]]


def try_register_canonicalize(f):
  try:
    return T.opt.register_canonicalize(f)
  except ValueError as e:
    print "try_register_canonicalize warning:", e
    return f  # just ignore


class GradDiscardOutOfBound(ViewOp):
  # See also theano.gradient.GradClip for a similar Op.
  __props__ = ()
  def __init__(self, lower_bound, upper_bound):
    super(GradDiscardOutOfBound, self).__init__()
    # We do not put those member in __eq__ or __hash__
    # as they do not influence the perform of this op.
    self.lower_bound = lower_bound
    self.upper_bound = upper_bound
    assert(self.lower_bound <= self.upper_bound)

  def grad(self, args, g_outs):
    return [T.switch(T.or_(T.lt(g_out, self.lower_bound), T.gt(g_out, self.upper_bound)),
                     T.cast(0, dtype=g_out.dtype),
                     g_out)
            for g_out in g_outs]

def grad_discard_out_of_bound(x, lower_bound, upper_bound):
  return GradDiscardOutOfBound(lower_bound, upper_bound)(x)

@try_register_canonicalize
@theano.gof.local_optimizer([GradDiscardOutOfBound])
def _local_grad_discard(node):
  if isinstance(node.op, GradDiscardOutOfBound):
    return node.inputs


def log_sum_exp(x, axis):
    x_max = T.max(x, axis=axis, keepdims=True)
    return T.log(T.sum(T.exp(x - x_max), axis=axis)) + x_max


def global_softmax_norm(z, index, log_norm=True):
  """
  :param theano.Variable z: 3D array. time*batch*feature
  :param theano.Variable index: 2D array, 0 or 1, time*batch
  :rtype: theano.Variable
  :returns 3D array. exp(z) / Z, where Z = sum(exp(z),axis=[0,2]) / z.shape[0].
  """
  assert z.ndim == 3
  assert index.ndim == 2
  index = T.cast(index, dtype="float32")  # 2D, time*batch
  index_bc = index.dimshuffle(0, 1, 'x')
  times = T.sum(index, axis=0)  # 1D, batch
  z_min = T.min(z, keepdims=True)
  z_filtered = z * index_bc + z_min * (numpy.float32(1) - index_bc)
  z_max = T.max(z_filtered, axis=2, keepdims=True)  # we ignore the out-of-index frames
  ez = T.exp(z - z_max)
  Z_frame = T.sum(ez, axis=2)  # 2D, time*batch
  if log_norm:
    Z_log_sum = T.log(Z_frame) + z_max.dimshuffle(0, 1)
    Z_log_norm = T.sum(Z_log_sum * index, axis=0) / times  # log-normalized. 1D, batch
    Z_log_norm_bc = Z_log_norm.dimshuffle('x', 0, 'x')  # 3D, time*batch*feature
  else:
    Z_norm = T.sum(Z_frame * index, axis=0) / times  # 1D, batch
    Z_log_norm = T.log(Z_norm)
    Z_log_norm_bc = Z_log_norm.dimshuffle('x', 0, 'x')  # 3D, time*batch*feature
    Z_log_norm_bc += z_max
  return T.exp(z - Z_log_norm_bc)
