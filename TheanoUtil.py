
import theano
import theano.sandbox.cuda
import theano.tensor as T


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
  added_dim_shape = added_dim_shape[:axis] + [factor, source.shape[axis] / factor] + added_dim_shape[axis + 1:]
  source = T.reshape(source, added_dim_shape)
  if method == "average":
    return T.mean(source, axis=axis)
  elif method == "max":
    return T.max(source, axis=axis)
  elif method == "min":
    return T.min(source, axis=axis)
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
      num_missing = target_axis_len - target.shape[axis]
      target = T.concatenate([target, T.repeat(last, num_missing, axis=axis)], axis=axis)
    return target
  else:
    assert False, "unknown upsample method %r" % method


