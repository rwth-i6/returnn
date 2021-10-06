
import theano
import theano.sandbox.cuda
import theano.tensor as T
from theano.compile import ViewOp
from theano import gof
from theano.compile import optdb
import numpy


def monkey_patches():
  """
  Applies a number of monkey patches,
  to make Theano work.
  """
  monkey_patch_numpy_get_ndarray_c_version()


def monkey_patch_numpy_get_ndarray_c_version():
  """
  Error:

    AttributeError: module 'numpy.core.multiarray' has no attribute '_get_ndarray_c_version'

  In Theano, when np.core.multiarray._get_ndarray_c_version() is called.

  Link:
    https://bugs.debian.org/cgi-bin/bugreport.cgi?bug=918090
  """
  import numpy.core.multiarray as ma
  if not hasattr(ma, "_get_ndarray_c_version"):  # later than 1.16.0
    print("Monkey patch Numpy _get_ndarray_c_version.")
    # Instead of patching Theano, we patch Numpy, which is simpler.
    import numpy.core._multiarray_umath as mau
    ma._get_ndarray_c_version = mau._get_ndarray_c_version


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
                   ndim=val.ndim - 1)
                   #name="flat_%s" % val.name)


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
  """
  :param theano.TensorVariable source: 3d tensor of shape (n_time, n_batch, n_dim)
  :param int|theano.Variable window: window size
  :return: tensor of shape (n_time, n_batch, window * n_dim)
  """
  assert source.ndim == 3  # (time,batch,dim). not sure how to handle other cases
  n_time = source.shape[0]
  n_batch = source.shape[1]
  n_dim = source.shape[2]
  w_right = window // 2
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


def delta_batch(source, window):
  """
  :param theano.TensorVariable source: 3d tensor of shape (n_time, n_batch, n_dim)
  :param int|theano.Variable window: window size
  :return: tensor of shape (n_time, n_batch, window * n_dim)
  Similar to numpy.diff. Also called delta.
  TODO with conv op
  """
  assert source.ndim == 3  # (time,batch,dim). not sure how to handle other cases
  w = windowed_batch(source, window + 1)
  return w[1:] - w[:-1]


def context_batched(source, window):
  """
  same as windowed_batch but with window center at the end of the window
  :param theano.TensorVariable source: 3d tensor of shape (n_time, n_batch, n_dim)
  :param int|theano.Variable window: window size
  :return: tensor of shape (n_time, n_batch, window * n_dim)
  """
  assert source.ndim == 3  # (time,batch,dim). not sure how to handle other cases
  n_time = source.shape[0]
  n_batch = source.shape[1]
  n_dim = source.shape[2]
  w_left = window - 1
  pad_left = T.zeros((w_left, n_batch, n_dim), dtype=source.dtype)
  padded = T.concatenate([pad_left, source], axis=0)  # shape[0] == n_time + window - 1
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

def window_batch_timewise(t,b,w,full_index):
  for i in range(w):
    full_index = T.set_subtensor(full_index[i], T.roll(full_index[i], i))
    if i > 0:
      full_index = T.inc_subtensor(full_index[i], T.where(full_index[i] > 0, i * t * b - i, 0))
  return full_index

def slice_for_axis(axis, s):
  return (slice(None),) * axis + (s,)


def downsample(source, axis, factor, method="average"):
  assert factor == int(factor), "factor is expected to be an int"
  factor = int(factor)
  # make shape[axis] a multiple of factor
  src = source
  source = source[slice_for_axis(axis=axis, s=slice(0, (source.shape[axis] // factor) * factor))]
  # Add a temporary dimension as the factor.
  added_dim_shape = [source.shape[i] for i in range(source.ndim)]
  added_dim_shape = added_dim_shape[:axis] + [source.shape[axis] // factor, factor] + added_dim_shape[axis + 1:]
  if method == "lstm":
    assert axis == 0
    return source
  source = T.reshape(source, added_dim_shape)
  if method == "average":
    return T.mean(source, axis=axis + 1)
  elif method == "max":
    return T.max(source, axis=axis + 1)
  elif method == "min":
    return T.min(source, axis=axis + 1)
  elif method == "concat" or method == 'mlp': # concatenates in last dimension
    #return T.reshape(source, added_dim_shape[:axis+1] + added_dim_shape[axis+2:-1] + [added_dim_shape[-1] * factor])
    return source.swapaxes(axis+1,src.ndim-1).reshape([source.shape[0],src.shape[1]] + [factor * source.shape[3]])
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
  target_axis_len = T.cast(target_axis_len, dtype="int32")
  num_missing = target_axis_len - source.shape[axis]
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
  num_chunks = (source.shape[0] + chunk_size - 1) // chunk_size
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
    print("try_register_canonicalize warning: %s" % e)
    return f  # just ignore


def try_register_gpu_opt(base_op_class):
  def do_register_gpu_opt(f):
    from theano.sandbox.cuda import cuda_available, register_opt
    from theano.gof.opt import local_optimizer
    f = local_optimizer([base_op_class])(f)
    if cuda_available:
      f = register_opt()(f)
    return f  # just ignore
  return do_register_gpu_opt


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



def gaussian_filter_1d(x, sigma, axis, window_radius=40):
  """
  Filter 1d input with a Gaussian using mode `nearest`.
  x is expected to be 2D/3D of type (time,batch,...).
  Adapted via: https://github.com/Theano/Theano/issues/3793
  Original Author: https://github.com/matthias-k
  """
  assert 2 <= x.ndim <= 3
  assert 0 <= axis < x.ndim

  # Construction of 1d kernel
  filter_1d = T.arange(2*window_radius + 1) - window_radius
  filter_1d = T.exp(-0.5*filter_1d**2/sigma**2)
  filter_1d = filter_1d / filter_1d.sum()
  filter_1d = filter_1d.astype(x.dtype)  # 1D, window-dim

  # transform `x` so that it is valid input for T.nnet.conv2d (batch size, stack size, nb row, nb col).
  # first, axis 1 of `x` because that is the number of batches.
  # then, a dummy stack size dim (not exactly sure about that. GpuDnnConv images and kernel must have the same stack size)
  # at the end the axis where we want to apply the conv on.
  # in between all remaining axes.
  # e.g., if axis==0 and x.ndim==3, we should get [1, 'x', 2, 0]
  blur_dims = [1] + ['x'] + [i for i in range(x.ndim) if i not in (1, axis)] + [axis]
  while len(blur_dims) < 4:
    blur_dims.insert(len(blur_dims) - 1, 'x')
  assert len(blur_dims) == 4
  blur_input = x.dimshuffle(blur_dims)
  assert blur_input.ndim == 4
  filter_W = filter_1d.dimshuffle(['x','x','x',0])

  # Construction of filter pipeline
  blur_input_start = blur_input[:, :, :, :1]
  blur_input_start_padding = T.repeat(blur_input_start, window_radius, axis=3)
  blur_input_end = blur_input[:, :, :, -1:]
  blur_input_end_padding = T.repeat(blur_input_end, window_radius, axis=3)

  padded_input = T.concatenate([blur_input_start_padding, blur_input, blur_input_end_padding], axis=3)

  # padded_input supposed to be 4D (batch size, stack size, nb row, nb col).
  # filter_W supposed to be 4D (nb filters, stack size, nb row, nb col).
  blur_op = T.nnet.conv2d(padded_input, filter_W, border_mode='valid',
                          filter_shape=[1, 1, 1, (2*window_radius + 1) if isinstance(window_radius, int) else None])
  # blur_op is 4D (batch size, nb filters, output row, output col).
  # output row = stack size * nb row.
  blur_op = blur_op[:, 0, :, :]  # only one filter, remove dimension
  # blur_op is 3D (batch size, output row, output col).
  y = blur_op.dimshuffle({0:2,2:1}[axis], 0, {0:1,2:2}[axis])
  if x.ndim == 2: y = y[:, :, 0]
  assert x.ndim == y.ndim
  return y


def log_sum_exp(x, axis):
  x_max = T.max(x, axis=axis)
  x_max_bc = T.makeKeepDims(x, x_max, axis=axis)
  return T.log(T.sum(T.exp(x - x_max_bc), axis=axis)) + x_max

def max_filtered(x, axis, index):
  index = T.cast(index, dtype="float32")  # 2D, time*batch
  index_bc = index.dimshuffle(*(range(index.ndim) + ['x'] * (x.ndim - index.ndim)))
  x_min = T.min(x, axis=axis, keepdims=True)
  x_filtered = x * index_bc + x_min * (numpy.float32(1) - index_bc)
  assert x_filtered.ndim == x.ndim
  x_max = T.max(x_filtered, axis=axis)  # we ignore the out-of-index frames
  return x_max

def log_sum_exp_index(x, axis, index):
  index = T.cast(index, dtype="float32")  # 2D, time*batch
  index_bc = index.dimshuffle(*(range(index.ndim) + ['x'] * (x.ndim - index.ndim)))
  assert index_bc.ndim == x.ndim
  x_max = max_filtered(x, axis=axis, index=index)  # we ignore the out-of-index frames
  x_max_bc = T.makeKeepDims(x, x_max, axis=axis)
  assert x.ndim == x_max_bc.ndim
  x_shift = (x - x_max_bc) * index_bc  # filter out out-of-index. exp() could be inf otherwise
  return T.log(T.sum(T.exp(x_shift) * index_bc, axis=axis)) + x_max


def global_softmax(z, index, mode):
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
  assert times.ndim == 1
  z_max2 = T.max(z, axis=2)
  z_max2_bc = z_max2.dimshuffle(0, 1, 'x')
  ez = T.exp(z - z_max2_bc)
  Z_frame = T.sum(ez, axis=2)  # 2D, time*batch
  if mode == "local":  # this is classic framewise softmax
    Z_log_norm_bc = T.log(Z_frame).dimshuffle(0, 1, 'x') + z_max2_bc
  elif mode == "log-norm":
    Z_log_frame = T.log(Z_frame) + z_max2
    Z_log_norm = T.sum(Z_log_frame * index, axis=0) / times  # log-normalized. 1D, batch
    Z_log_norm_bc = Z_log_norm.dimshuffle('x', 0, 'x')  # 3D, time*batch*feature
  elif mode == "maxshift-log-norm":
    Z_log_frame = T.log(Z_frame)
    Z_log_norm = T.sum(Z_log_frame * index, axis=0) / times  # log-normalized. 1D, batch
    Z_log_norm_bc = Z_log_norm.dimshuffle('x', 0, 'x')  # 3D, time*batch*feature
    Z_log_norm_bc = Z_log_norm_bc + z_max2_bc
  elif mode == "std-norm":
    #Z_log_norm = T.log( T.sum(T.exp(T.log(Z_frame) + z_max), axis=0) / times )  <- we want that
    Z_log_frame = T.log(Z_frame) + z_max2
    assert Z_log_frame.ndim == 2
    Z_log_norm = log_sum_exp_index(Z_log_frame, index=index, axis=0) - T.log(times)
    assert Z_log_norm.ndim == 1
    Z_log_norm_bc = Z_log_norm.dimshuffle('x', 0, 'x')  # 3D, time*batch*feature
  elif mode == "maxshift-std-norm":
    # We normalize each shifted frame.
    Z_norm = T.sum(Z_frame * index, axis=0) / times  # 1D, batch
    assert Z_norm.ndim == 1
    Z_log_norm = T.log(Z_norm)
    Z_log_norm_bc = Z_log_norm.dimshuffle('x', 0, 'x')  # 3D, time*batch*feature
    Z_log_norm_bc = Z_log_norm_bc + z_max2_bc
  elif mode.startswith("gauss-maxshift("):
    modeend = mode.find(")-")
    assert modeend >= 0
    sigma = float(mode[len("gauss-maxshift("):modeend])
    z_gmax2 = gaussian_filter_1d(z_max2, sigma=sigma, axis=0)
    z_gmax2_bc = z_gmax2.dimshuffle(0, 1, 'x')
    z = z - z_gmax2_bc
    return global_softmax(z - z_gmax2_bc, mode=mode[modeend + 2:], index=index)
  elif mode.startswith("gauss-std-norm("):
    modeend = mode.find(")")
    assert modeend >= 0 and modeend == len(mode) - 1
    sigma = float(mode[len("gauss-std-norm("):modeend])
    Z_log_frame = T.log(Z_frame) + z_max2
    assert Z_log_frame.ndim == 2
    Z_log_frame_g = gaussian_filter_1d(z_max2, sigma=sigma, axis=0)
    Z_log_norm = log_sum_exp_index(Z_log_frame_g, index=index, axis=0) - T.log(times)
    assert Z_log_norm.ndim == 1
    Z_log_norm_bc = Z_log_norm.dimshuffle('x', 0, 'x')  # 3D, time*batch*feature
  else:
    assert False, "invalid global_softmax mode %r" % mode
  return T.exp(z - Z_log_norm_bc)


def show_global_softmax_stats(z):
  """
  :param z: numpy.ndarray or Theano Var (eval-able), 2D time*features
  """
  def stats(y): return numpy.min(y), numpy.max(y), numpy.mean(y), numpy.var(y)
  if z.ndim == 3: z = z[:,0,:]
  assert z.ndim == 2
  z_numpy = z
  z = T.as_tensor_variable(z)
  if not isinstance(z, numpy.ndarray): z = z.eval()
  print("show_global_softmax_stats for shape %s" % (z_numpy.shape,))
  print(" z min/max/mean/var = %s" % (stats(z_numpy),))
  z_max1 = numpy.max(z_numpy, axis=1)
  print(" z max1 min/max/mean/var = %s" % (stats(z_max1),))
  z_dmax1 = z_max1[:-1] - z_max1[1:]
  print(" z dmax1 min/max/mean/var = %s" % (stats(z_dmax1),))
  z_gmax1 = gaussian_filter_1d(T.as_tensor_variable(z_max1).dimshuffle(0, 'x'), sigma=10.0, axis=0).eval()[:,0]
  print(" z gmax1 min/max/mean/var = %s" % (stats(z_gmax1),))
  z_dgmax1 = z_gmax1[:-1] - z_gmax1[1:]
  print(" z dgmax1 min/max/mean/var = %s" % (stats(z_dgmax1),))
  z = T.cast(z, "float32")  # we always expect this precision
  z = z.dimshuffle(0, 'x', 1)  # add batch-dim
  index = T.ones((z.shape[0], 1))
  for mode in ["local", "log-norm", "maxshift-log-norm", "std-norm", "maxshift-std-norm",
               "gauss-maxshift(2.0)-std-norm", "gauss-maxshift(5.0)-std-norm", "gauss-maxshift(2.0)-log-norm",
               "gauss-std-norm(2.0)", "gauss-std-norm(5.0)"]:
    print(" mode %s" % mode)
    y = global_softmax(z, index=index, mode=mode).eval()
    assert y.ndim == 3 and y.shape[1] == 1
    y = y[:,0,:]
    print("  min/max/mean/var = %s" % (stats(y),))
    log_y = numpy.log(y)
    print("  log min/max/mean/var = %s" % (stats(log_y),))
    y_sum1 = numpy.sum(y, axis=1)
    print("  sum1 min/max/mean/var = %s" % (stats(y_sum1),))
    log_y_sum1 = numpy.log(y_sum1)
    print("  log sum1 min/max/mean/var = %s" % (stats(log_y_sum1),))


def complex_elemwise_mult(a, b, axis=-1):
  assert a.ndim == b.ndim
  if axis < 0: axis %= a.ndim
  assert 0 <= axis < a.ndim
  a_real = a[slice_for_axis(axis=axis, s=slice(0, a.shape[axis] / 2))]
  a_imag = a[slice_for_axis(axis=axis, s=slice(a.shape[axis] / 2, None))]
  b_real = b[slice_for_axis(axis=axis, s=slice(0, b.shape[axis] / 2))]
  b_imag = b[slice_for_axis(axis=axis, s=slice(b.shape[axis] / 2, None))]
  r_real = a_real * b_real - a_imag * b_imag
  r_imag = a_real * b_imag + a_imag * b_real
  return T.concatenate([r_real, r_imag], axis=axis)

def complex_bound(a, axis=-1):
  # Via Associative LSTMs, https://arxiv.org/abs/1602.03032
  if axis < 0: axis %= a.ndim
  assert 0 <= axis < a.ndim
  a_real = a[slice_for_axis(axis=axis, s=slice(0, a.shape[axis] / 2))]
  a_imag = a[slice_for_axis(axis=axis, s=slice(a.shape[axis] / 2, None))]
  d = T.maximum(numpy.float32(1), T.sqrt(a_real * a_real + a_imag * a_imag))
  r_real = a_real / d
  r_imag = a_imag / d
  return T.concatenate([r_real, r_imag], axis=axis)

def complex_dot(a, b):
  assert a.ndim >= 1
  assert b.ndim >= 1
  a_axis = a.ndim - 1
  a_real = a[slice_for_axis(axis=a_axis, s=slice(0, a.shape[a_axis] / 2))]
  a_imag = a[slice_for_axis(axis=a_axis, s=slice(a.shape[a_axis] / 2, None))]
  b_axis = 0
  b_real = b[slice_for_axis(axis=b_axis, s=slice(0, b.shape[b_axis] / 2))]
  b_imag = b[slice_for_axis(axis=b_axis, s=slice(b.shape[b_axis] / 2, None))]
  r_real = T.dot(a_real, b_real) - T.dot(a_imag, b_imag)
  r_imag = T.dot(a_real, b_imag) + T.dot(a_imag, b_real)
  r_axis = r_real.ndim - 1
  return T.concatenate([r_real, r_imag], axis=r_axis)


def indices_in_flatten_array(ndim, shape, *args):
  """
  We expect that all args can be broadcasted together.
  So, if we have some array A with ndim&shape as given,
  A[args] would give us a subtensor.
  We return the indices so that A[args].flatten()
  and A.flatten()[indices] are the same.
  """
  assert ndim > 0
  assert len(args) == ndim
  indices_per_axis = [args[i] for i in range(ndim)]
  for i in range(ndim):
    for j in range(i + 1, ndim):
      indices_per_axis[i] *= shape[j]
  indices = indices_per_axis[0]
  for i in range(1, ndim):
    indices += indices_per_axis[i]
  return indices

def circular_convolution(a, b):
  from returnn.theano.device import is_using_gpu
  has_gpuarray = is_using_gpu()
  try:
    import pygpu
  except Exception:
    has_gpuarray = False
  if has_gpuarray:
    from theano.gpuarray.fft import curfft as fft
    from theano.gpuarray.fft import cuirfft as ifft
  else:
    from theano.tensor.fft import rfft as fft
    from theano.tensor.fft import irfft as ifft
  return ifft(fft(a) * fft(b))


def unroll_scan(fn, sequences=(), outputs_info=(), non_sequences=(), n_steps=None,
                go_backwards=False):
  """
  Helper function to unroll for loops. Can be used to unroll theano.scan.
  The parameter names are identical to theano.scan, please refer to here
  for more information.

  Note that this function does not support the truncate_gradient
  setting from theano.scan.

  Code adapted from https://github.com/Lasagne/Lasagne.
  Thank you!

  Parameters
  ----------

  fn : function
      Function that defines calculations at each step.

  sequences : TensorVariable or list of TensorVariables
      List of TensorVariable with sequence data. The function iterates
      over the first dimension of each TensorVariable.

  outputs_info : list of TensorVariables
      List of tensors specifying the initial values for each recurrent
      value.

  non_sequences: list of TensorVariables
      List of theano.shared variables that are used in the step function.

  n_steps: int
      Number of steps to unroll.

  go_backwards: bool
      If true the recursion starts at sequences[-1] and iterates
      backwards.

  Returns
  -------
  Tuple of the form (outputs, updates).
  outputs is a list of TensorVariables. Each element in the list gives the recurrent
  values at each time step.
  updates is an empty dict for now.
  """
  if not isinstance(sequences, (list, tuple)):
    sequences = [sequences]
  sequences = list(sequences)
  outputs_info = list(outputs_info)
  non_sequences = list(non_sequences)

  # When backwards reverse the recursion direction
  counter = range(n_steps)
  if go_backwards:
    counter = counter[::-1]

  output = []
  prev_vals = outputs_info
  until = []
  for i in counter:
    assert len(prev_vals) == len(outputs_info)
    prev_vals = [prev for prev, out_info in zip(prev_vals, outputs_info) if out_info is not None]
    step_input = [s[i] for s in sequences] + prev_vals + non_sequences
    out_ = fn(*step_input)
    # The returned values from step can be either a TensorVariable,
    # a list, or a tuple.  Below, we force it to always be a list.
    if isinstance(out_, T.TensorVariable):
      out_ = [out_]
    if isinstance(out_, tuple):
      if len(out_) >= 1 and isinstance(out_[0], (list, tuple)):
        if len(out_) >= 2:
          assert not out_[1], "shared var updates not supported"
        if len(out_) >= 3:
          assert isinstance(out_[2], theano.scan_module.until)
          until.append(T.neq(out_[2].condition, 0))
        out_ = list(out_[0])
      else:
        out_ = list(out_)
    output.append(out_)

    prev_vals = output[-1]

  # iterate over each scan output and convert it to same format as scan:
  # [[output11, output12,...output1n],
  # [output21, output22,...output2n],...]
  output_scan = []
  for i in range(len(output[0])):
    l = map(lambda x: x[i], output)
    output_scan.append(T.stack(*l))

  if until:
    assert len(until) == n_steps
    until_conds = T.stack(*until)
    new_len = T.switch(T.any(until_conds),
                       T.minimum(T.argmax(until_conds) + 1, n_steps),
                       n_steps)
    output_scan = [out[:new_len] for out in output_scan]

  if len(output_scan) == 1:
    output_scan = output_scan[0]
  updates = {}
  return output_scan, updates


def make_var_tuple(v):
  if isinstance(v, theano.Variable):
    return (v, )
  if isinstance(v, list):
    return tuple(v)
  assert isinstance(v, tuple)
  return v


class Contiguous(T.extra_ops.CpuContiguous):
  # Original CpuContiguous is missing the grad.
  # See https://github.com/Theano/Theano/issues/4399.
  def grad(self, inputs, output_grads):
    dout, = output_grads
    dout = T.as_tensor_variable(dout)
    return [dout]


@gof.local_optimizer([Contiguous], inplace=True)
def opt_remove_contiguous(node):
  if isinstance(node.op, Contiguous):
    x, = node.inputs
    if x.owner and isinstance(x.owner.op, (T.Alloc, T.AllocEmpty, T.extra_ops.CpuContiguous)):
      return [x]
  return False

optdb.register('opt_remove_contiguous',
               gof.TopoOptimizer(opt_remove_contiguous),
               10, 'fast_run')

# Theano will not do this optimization. So we register it now.
# See: https://github.com/Theano/Theano/issues/4400
@try_register_gpu_opt(Contiguous)
def local_gpu_Contiguous(node):
  if isinstance(node.op, Contiguous):
    # see also: https://github.com/Theano/Theano/blob/master/theano/sandbox/cuda/opt.py
    from theano.sandbox.cuda import host_from_gpu
    x, = node.inputs
    if x.owner and x.owner.op == host_from_gpu:
      from theano.sandbox.cuda.basic_ops import gpu_contiguous
      return [host_from_gpu(gpu_contiguous(x.owner.inputs[0]))]


class DumpOp(theano.Op):
  __props__ = ("filename", "with_grad")
  view_map = {0: [0]}

  def __init__(self, filename, container=None, with_grad=True, parent=None, step=1):
    super(DumpOp, self).__init__()
    self.filename = filename
    self.container = container
    self.with_grad = with_grad
    self.counter = 0
    self.parent = parent
    self.step = step

  def make_node(self, x):
    x = T.as_tensor_variable(x)
    return gof.Apply(self, [x], [x.type()])

  def perform(self, node, inputs, output_storage):
    x, = inputs
    self.dump(x)
    output_storage[0][0] = x

  def grad(self, inputs, output_grads):
    dout, = output_grads
    dout = T.as_tensor_variable(dout)
    if self.with_grad:
      # Note: This assumes that there will be only one such gradient.
      dout = DumpOp(filename=self.filename + ".grad", container=self.container, parent=self, step=self.step)(dout)
    return [dout]

  def dump(self, x):
    filename = self.get_full_filename()
    if filename is not None:
      if self.container is not None:
        assert filename not in self.container
        self.container[filename] = x.copy()
      else:
        import os, numpy
        assert not os.path.exists(filename), "%s already exists, not overwriting" % filename
        numpy.save(filename, x)

  def get_full_filename(self):
    counter = self.get_counter()
    self.inc_counter()
    if counter % self.step == 0:
      return "%s.%i.npy" % (self.filename, counter)
    else:
      return None

  def get_counter(self):
    if self.parent: return self.parent.get_counter() - 1
    return self.counter

  def inc_counter(self):
    if self.parent: return
    self.counter += 1


def softmax(z):
  dims = [z.shape[i] for i in range(z.ndim)]
  collapsed_dim = numpy.prod(dims[:-1])
  return T.nnet.softmax(z.reshape((collapsed_dim, dims[-1]))).reshape(dims)


def layer_normalization(x, bias=None, scale=None, eps=1e-5):
  """
  Layer Normalization, https://arxiv.org/abs/1607.06450
  x is mean and variance normalized along its feature dimension.
  After that, we allow a bias and a rescale. This is supposed to be trainable.
  :param x: 3d tensor (time,batch,dim) (or any ndim, last dim is expected to be dim)
  :param bias: 1d tensor (dim) or None
  :param scale: 1d tensor (dim) or None
  """
  mean = T.mean(x, axis=x.ndim - 1, keepdims=True)
  std = T.sqrt(T.var(x, axis=x.ndim - 1, keepdims=True) + numpy.float32(eps))
  assert mean.ndim == std.ndim == x.ndim
  output = (x - mean) / std
  assert output.ndim == x.ndim
  if scale is not None:
    assert scale.ndim == 1
    scale = scale.dimshuffle(*(('x',) * (x.ndim - 1) + (0,)))
    assert scale.ndim == x.ndim
    output = output * scale
  if bias is not None:
    assert bias.ndim == 1
    bias = bias.dimshuffle(*(('x',) * (x.ndim - 1) + (0,)))
    assert bias.ndim == x.ndim
    output = output + bias
  return output


def print_to_file(filename, x, argmax=None, sum=None, shape=False):
  def theano_print_to_file(op,x):
    with open(filename, 'a') as f:
      if argmax is not None:
        f.write(str(x.argmax(axis=argmax)) + '\n')
      elif sum is not None:
        f.write(str(x.sum(axis=sum)) + '\n')
      elif shape:
        f.write(str(x.shape) + '\n')
      else:
        #opt = numpy.get_printoptions()
        #numpy.set_printoptions(threshold='nan')
        f.write(str(x) + '\n')
        #numpy.set_printoptions(opt)
  return theano.printing.Print(global_fn=theano_print_to_file)(x)


def self_similarity_cosine(x):
  """
  :param x: shape (T,D)
  :returns cosine similarity matrix, shape (T,T), zeroed at diagonal and upper triangle
  """
  assert x.ndim == 2
  x2 = T.sqrt(T.clip(T.sum(T.sqr(x), axis=1), numpy.float32(1.e-20), numpy.float32(1.e20)))
  assert x2.ndim == 1
  x_ = x / x2.dimshuffle(0, 'x')
  xx = T.dot(x_, x_.T)
  assert xx.ndim == 2
  from theano.tensor.basic import tril
  return tril(xx, 1)
