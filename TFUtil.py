
import tensorflow as tf
from tensorflow.python.client import device_lib
import contextlib
import os
import sys
from Util import NotSpecified


class Data(object):
  """
  This class is to describe a tensor,
  i.e. it's shape and properties like
  whether we should consider it as sparse data (i.e. it represents indices).
  This is used in TFNetwork to describe the dataset external data
  as well as for every layer output.
  """

  size_dtype = "int32"

  def __init__(self, name,
               shape=None, dtype=None,
               placeholder=None,
               sparse=None,
               dim=None,
               size_placeholder=None,
               batch_dim_axis=0,
               time_dim_axis=NotSpecified,
               auto_create_placeholders=False):
    """
    :param str name:
    :param tuple[int|None] shape: including time-dim (can be None). excluding batch-dim. e.g. (time,feat)=(None,128)
    :param str dtype: e.g. "float32" or "int64"
    :param tf.Tensor|None placeholder: with added batch-dim
    :param bool sparse: whether to treat the value as an index. do not confuse with tf.SparseTensor
    :param None|int dim: feature dimension, shape[-1] if not sparse, otherwise like num_classes
    :param int batch_dim_axis: where we add the batch-dim.
      e.g. shape=(time,...), 0 -> (batch,time,...), 1 -> (time,batch,...)
    :param int|None time_dim_axis: where we have the time dim axis, after we added the batch-dim.
      this is often 1. however, can be None if there is no time-dim.
    :param dict[int,tf.Tensor] tf.Tensor size_placeholder: for every None in shape, this will describe the size
    """
    self.name = name
    if sparse is None:
      if dtype and dtype.startswith("int"):
        sparse = True
      elif name == "classes":
        sparse = True
      elif shape is not None and len(shape) == 1:
        sparse = True
      if sparse is None:
        sparse = False
    self.sparse = sparse
    if shape is None:
      assert dim, "no shape specified, need dim"
      if sparse:
        shape = (None,)  # assume common (time,)
      else:
        shape = (None, dim)  # assume common (time,feat)
    self.shape = shape  # excluding batch-dim. see self.batch_shape
    if dtype is None:
      if sparse:
        dtype = "int32"
      else:
        dtype = "float32"
    if dim is None and len(shape):
      assert not sparse, "need dim"
      dim = shape[-1]
    self.dim = dim
    self.batch_dim_axis = batch_dim_axis
    if time_dim_axis is NotSpecified:
      if (sparse and len(shape) >= 1) or ((not sparse) and len(shape) >= 2):
        if batch_dim_axis >= 1:
          time_dim_axis = 0
        else:
          time_dim_axis = 1
      else:
        time_dim_axis = None
    self.time_dim_axis = time_dim_axis
    self.dtype = dtype
    if placeholder is None and auto_create_placeholders:
      with tf.name_scope("extern_data/placeholders/%s/" % name):
        placeholder = tf.placeholder(name=name, dtype=dtype, shape=self.batch_shape)
    self.placeholder = placeholder
    # The size_placeholder is for each variable length dimension in shape, i.e. excluding the batch-dim.
    if size_placeholder is None and auto_create_placeholders:
      size_placeholder = {}  # type: dict[int,tf.Tensor]
      with tf.name_scope("extern_data/placeholders/%s/" % name):
        for i, dim in enumerate(shape):
          if dim is None:
            # For each batch a separate size.
            size_placeholder[i] = tf.placeholder(
              name="%s_dim%i_size" % (name, i), dtype=self.size_dtype, shape=(None,))
    self.size_placeholder = size_placeholder

  def get_kwargs(self):
    keys = ["name", "shape", "dtype", "sparse", "dim", "batch_dim_axis", "time_dim_axis"]
    return {key: getattr(self, key) for key in keys}

  def _get_variable_dim_pattern(self):
    """
    :return: tuple with bools specifying which dims of the shape (excluding batch-dim) are of variable length.
     e.g. (time,feature), shape=(None,128), this returns (True, False)
    :rtype: tuple[bool]
    """
    return tuple([dim is None for dim in self.shape])

  def _get_var_len_axes(self):
    return sorted([i for (i, d) in enumerate(self._get_variable_dim_pattern()) if d is None])

  def matches_dim_pattern(self, other):
    """
    :param Data other:
    :return: whether the dim pattern matches, i.e. same variable dims (get_variable_dim_pattern), same time dim,
      excluding batch-dim.
    :rtype: bool
    """
    if self.time_dim_axis_excluding_batch != other.time_dim_axis_excluding_batch:
      return False
    return self._get_var_len_axes() == other._get_var_len_axes()

  def get_description(self, with_name=True, with_placeholder=False):
    keys = ["shape", "dtype", "sparse"]
    if self.sparse:
      keys.append("dim")
    if with_name:
      keys.insert(0, "name")
    if with_placeholder:
      keys.append("placeholder")
    return "Data(%s)" % ", ".join(["%s=%r" % (key, getattr(self, key)) for key in keys])

  @property
  def batch_shape(self):
    """
    :return: shape with added batch-dim. e.g. (batch,time,feat) = (None,None,128)
    :rtype: tuple[int|None]
    """
    return self.shape[:self.batch_dim_axis] + (None,) + self.shape[self.batch_dim_axis:]

  @property
  def is_time_major(self):
    """
    :return: whether this is in time-major format, i.e. (time,batch,...)
    :rtype: bool
    """
    return self.time_dim_axis == 0

  @property
  def time_dim_axis_excluding_batch(self):
    if self.time_dim_axis is None:
      return None
    return self.get_batch_axis_excluding_batch(self.time_dim_axis)

  def get_placeholder_as_time_major(self):
    if self.is_time_major:
      return self.placeholder
    assert self.batch_dim_axis == 0
    assert self.time_dim_axis == 1
    return swapaxes(self.placeholder, 0, 1)  # (time,batch,dim)

  def get_axes(self, exclude_time=False, exclude_batch=False):
    """
    :param bool exclude_time: will filter out the time-axis
    :param bool exclude_batch: will filter out the batch-axis
    :return: list of axes, like `range(len(self.shape))`, calculated with batch dim.
    :rtype: list[int]
    """
    axes = list(range(len(self.batch_shape)))
    if exclude_time and self.time_dim_axis is not None:
      axes.pop(axes.index(self.time_dim_axis))
    if exclude_batch and self.batch_dim_axis is not None:
      axes.pop(axes.index(self.batch_dim_axis))
    return axes

  def get_batch_axis_excluding_batch(self, axis):
    if axis == self.batch_dim_axis:
      return None
    if axis < self.batch_dim_axis:
      return axis
    return axis - 1

  @property
  def default_broadcast_noise_shape(self):
    """
    :return: noise-shape which will broadcast along all dynamic dimensions and time/batch dim
    :rtype: tuple[int]
    """
    return [1 if (dim is None or axis in [self.batch_dim_axis, self.time_dim_axis]) else dim
            for axis, dim in enumerate(self.batch_shape)]


class OutputWithActivation(object):
  def __init__(self, x, act_func=None):
    """
    :param tf.Tensor x:
    :param None|(tf.Tensor)->tf.Tensor act_func:
    """
    self.x = x
    self.act_func = act_func
    if act_func:
      with tf.name_scope("activation"):
        self.y = act_func(x)
    else:
      self.y = x

  def is_softmax_act_func(self):
    return self.act_func is tf.nn.softmax

  def get_logits(self):
    """
    :rtype: tf.Tensor
    :return: logits. logits are (not necessarily normalized) log probabilities, i.e. the input of softmax.
    This call assumes that self.y is in probability space.
    """
    if self.is_softmax_act_func():
      return self.x
    return tf.log(self.y)


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_%s' % name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('%s_mean' % name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('%s_stddev' % name, stddev)
    tf.summary.scalar('%s_max' % name, tf.reduce_max(var))
    tf.summary.scalar('%s_min' % name, tf.reduce_min(var))
    tf.histogram_summary('%s_histogram' % name, var)


def get_current_name_scope(graph=None):
  """
  :param tf.Graph|None graph:
  :return: current name scope
  :rtype: str
  """
  if graph is None:
    graph = tf.get_default_graph()
  assert isinstance(graph, tf.Graph)
  return graph._name_stack


@contextlib.contextmanager
def reuse_name_scope(name):
  """
  :param str name: relative name scope

  http://stackoverflow.com/questions/40907769/how-to-get-current-tensorflow-name-scope
  """
  assert name
  g = tf.get_default_graph()
  assert isinstance(g, tf.Graph)
  current_name_scope = get_current_name_scope(g)
  if current_name_scope:
    name = current_name_scope + "/" + name
  if name[-1] != "/":
    name += "/"
  # Actually, tf.variable_scope doesn't fully support the absolute name-scope with ending "/"
  # but it uses tf.name_scope internally which uses this syntax.
  with tf.variable_scope(name) as scope:
    assert isinstance(scope, tf.VariableScope)
    # remove "/" from the end of the var-scope.
    # This is a work-around to fix up the variable scope behavior for nested variable scopes.
    # WARNING: This might break at some future point.
    assert scope.name is scope._name
    assert scope.name[-1] == "/"
    scope._name = scope._name[:-1]
    yield scope


class FlipGradientBuilder(object):
  """
  Gradient Reversal Layer.
  Discussion:
      https://github.com/fchollet/keras/issues/3119
      https://github.com/tensorflow/tensorflow/issues/4342
  Code from here:
      https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
  """

  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, l=1.0):
    grad_name = "FlipGradient%d" % self.num_calls

    from tensorflow.python.framework import ops
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      return [tf.neg(grad) * l]

    g = tf.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x)

    self.num_calls += 1
    return y

flip_gradient = FlipGradientBuilder()


def check_input_ndim(x, ndim):
  """
  :param tf.Tensor x:
  :param int ndim:
  :return: x with check added
  :rtype: tf.Tensor
  """
  dyn_shape = x.get_shape()
  if dyn_shape.ndims is not None:
    assert dyn_shape.ndims == ndim
    return x
  # Need to fall-back to runtime check.
  with reuse_name_scope("checks"):
    with tf.control_dependencies(
      [tf.assert_equal(tf.rank(x), ndim, data=["ndim not %i" % ndim, tf.shape(x)])]):
      return tf.identity(x, "identity_with_ndim_check")


def check_input_ndim_equal_offset(x, y, y_ndim_offset=0):
  """
  :param tf.Tensor x:
  :param tf.Tensor y:
  :param int y_ndim_offset:
  :return: x with check added such that ndim(x) == ndim(y) + y_ndim_offset
  :rtype: tf.Tensor
  """
  x_dyn_shape = x.get_shape()
  y_dyn_shape = y.get_shape()
  if x_dyn_shape.ndims is not None and y_dyn_shape.ndims is not None:
    assert x_dyn_shape.ndims == y_dyn_shape.ndims + y_ndim_offset
    return x
  # Need to fall-back to runtime check.
  with reuse_name_scope("checks"):
    with tf.control_dependencies(
      [tf.assert_equal(tf.rank(x), tf.rank(y) + y_ndim_offset,
                       data=["ndim not equal with offset %i" % y_ndim_offset,
                             tf.shape(x), tf.shape(y)])]):
      return tf.identity(x, "identity_with_ndim_equal_check")


def check_input_dim(x, axis, dim):
  """
  :param tf.Tensor x:
  :param int axis: which axis to check
  :param int dim:
  :return: x with check added
  :rtype: tf.Tensor
  """
  dyn_shape = x.get_shape()
  if dyn_shape.ndims is not None:
    if dyn_shape.dims[axis].value is not None:
      assert dyn_shape.dims[axis].value == dim
      return x
  # Need to fall-back to runtime check.
  with reuse_name_scope("checks"):
    with tf.control_dependencies(
      [tf.assert_equal(tf.shape(x)[axis], dim, data=["shape[%i] not %i" % (axis, dim), tf.shape(x)])]):
      return tf.identity(x, "identity_with_dim_check")


def check_dim_equal(x, x_axis, y, y_axis):
  """
  :param tf.Tensor x:
  :param int x_axis: which axis to check
  :param tf.Tensor y:
  :param int y_axis: which axis to check
  :return: x with check added that shape(x)[x_axis] == shape(y)[y_axis]
  :rtype: tf.Tensor
  """
  x_dyn_shape = x.get_shape()
  y_dyn_shape = y.get_shape()
  if x_dyn_shape.ndims is not None and y_dyn_shape.ndims is not None:
    if x_dyn_shape.dims[x_axis].value is not None and y_dyn_shape.dims[y_axis].value is not None:
      assert x_dyn_shape.dims[x_axis].value == y_dyn_shape.dims[y_axis].value
      return x
  # Need to fall-back to runtime check.
  with reuse_name_scope("checks"):
    with tf.control_dependencies(
      [tf.assert_equal(
        tf.shape(x)[x_axis], tf.shape(y)[y_axis],
        data=["x.shape[%i] not y.shape[%i]" % (x_axis, y_axis),
              tf.shape(x), tf.shape(y)])]):
      return tf.identity(x, "identity_with_dim_equal_check")


def check_shape_equal(x, y):
  """
  :param tf.Tensor x:
  :param tf.Tensor y:
  :return: x with check added that shape(x) == shape(y)
  :rtype: tf.Tensor
  """
  x_dyn_shape = x.get_shape()
  y_dyn_shape = y.get_shape()
  if x_dyn_shape.ndims is not None and y_dyn_shape.ndims is not None:
    assert x_dyn_shape.ndims == y_dyn_shape.ndims
    have_unknown = False
    for axis in range(x_dyn_shape.ndims):
      if x_dyn_shape.dims[axis].value is not None and y_dyn_shape.dims[axis].value is not None:
        assert x_dyn_shape.dims[axis].value == y_dyn_shape.dims[axis].value
      else:
        have_unknown = True
    if not have_unknown:
      return x  # all dims are checked, we can return
  # We need to fall-back to runtime check.
  with reuse_name_scope("checks"):
    with tf.control_dependencies(
      [tf.assert_equal(
        tf.shape(x), tf.shape(y),
        data=["x.shape not y.shape",
              tf.shape(x), tf.shape(y)])]):
      return tf.identity(x, "identity_with_shape_equal_check")


def get_shape_dim(x, axis):
  """
  :param tf.Tensor x:
  :param int axis: which axis
  :return: x.shape[axis] either as a static int or otherwise as an expression
  :rtype: int|tf.Tensor
  """
  dyn_shape = x.get_shape()
  if dyn_shape.ndims is not None:
    if dyn_shape.dims[axis].value is not None:
      return dyn_shape.dims[axis].value
  # Need to fall-back to runtime.
  return tf.shape(x)[axis]


def get_ndim(x):
  """
  :param tf.Tensor x:
  :return: x.ndim either as a static int or otherwise as an expression
  :rtype: int|tf.Tensor
  """
  dyn_shape = x.get_shape()
  if dyn_shape.ndims is not None:
    return dyn_shape.ndims
  # Need to fall-back to runtime.
  return tf.rank(x)


def get_range(start, stop=NotSpecified):
  """
  :param int|tf.Tensor|None start:
  :param int|tf.Tensor|None stop:
  :return: either tuple(range(start, stop)) or the same as a symbolic expression
  :rtype: tuple[int]|tf.Tensor
  """
  if stop is NotSpecified:
    stop = start
    start = 0
  if isinstance(start, tf.Tensor) or isinstance(stop, tf.Tensor):
    return tf.range(start, stop)
  return tuple(range(start, stop))


def identity_with_ops(x, ops):
  """
  :param tf.Tensor x:
  :param () -> list[tf.Operation|tf.Tensor] ops:
  :return: x with all ops executed
  :rtype: tf.Tensor
  """
  with reuse_name_scope("checks"):
    with tf.control_dependencies(ops()):
      return tf.identity(x, name="identity_with_ops")


_list_local_devices = None

def _get_tf_list_local_devices():
  global _list_local_devices
  if _list_local_devices:
    return _list_local_devices
  print("Collecting TensorFlow device list...")
  _list_local_devices = list(device_lib.list_local_devices())
  return _list_local_devices


def _parse_physical_device_desc(s):
  """
  :param str s: string via dev.physical_device_desc. e.g. "device: 0, name: GeForce GTX 980, pci bus id: 0000:41:00.0"
  :return: dict key -> value
  :rtype: dict[str,str]
  """
  d = {}
  for part in s.split(","):
    part = part.strip()
    key, value = part.split(":", 1)
    key, value = key.strip(), value.strip()
    d[key] = value
  return d


def print_available_devices():
  cuda_visible_devs = None
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES is set to %r." % os.environ["CUDA_VISIBLE_DEVICES"])
    cuda_visible_devs = dict(enumerate([int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d]))
  else:
    print("CUDA_VISIBLE_DEVICES is not set.")
  devs = _get_tf_list_local_devices()
  print("Local devices available to TensorFlow:")
  for i, dev in enumerate(devs):
    print("  %i/%i: %s" % (i + 1, len(devs), "\n       ".join(str(dev).splitlines())))

  # Theano prints sth like: Using gpu device 2: GeForce GTX 980 (...)
  # Print in a similar format so that some scripts which grep our stdout work just as before.
  for dev in devs:
    if dev.device_type == "GPU":
      d = _parse_physical_device_desc(dev.physical_device_desc)
      dev_id = int(d["device"])
      if cuda_visible_devs:
        dev_id = cuda_visible_devs[dev_id]
      dev_name = d["name"]
      print("Using gpu device %i: %s" % (dev_id, dev_name))


def is_gpu_available():
  """Returns whether TensorFlow can access a GPU."""
  return any(x.device_type == 'GPU' for x in _get_tf_list_local_devices())


def dot(a, b):
  """
  :param tf.Tensor a: shape [...da...,d]
  :param tf.Tensor b: shape [d,...db...]
  :return: tensor of shape [...da...,d,...db...]
  :rtype: tf.Tensor
  """
  with tf.name_scope("dot"):
    a_ndim = a.get_shape().ndims
    b_ndim = b.get_shape().ndims
    assert a_ndim is not None
    if a_ndim == 0:
      return tf.scalar_mul(a, b)
    assert b_ndim is not None
    if b_ndim == 0:
      return tf.scalar_mul(b, a)
    a = check_dim_equal(a, -1, b, 0)
    if a_ndim == b_ndim == 1:
      return tf.reduce_sum(a * b)
    a_shape = tf.shape(a)
    b_shape = tf.shape(b)
    d = get_shape_dim(b, 0)
    assert a_ndim >= 2 and b_ndim >= 2
    if a_ndim > 2:
      a = tf.reshape(a, (-1, d))
    if b_ndim > 2:
      b = tf.reshape(b, (d, -1))
    res = tf.matmul(a, b)
    if a_ndim > 2 or b_ndim > 2:
      res = tf.reshape(
        res, [a_shape[i] for i in range(0, a_ndim - 1)] + [b_shape[i] for i in range(1, b_ndim)])
    return res


def get_activation_function(s):
  """
  :param str s:
  :rtype: (tf.Tensor) -> tf.Tensor
  """
  act_func = getattr(tf.nn, s)  # e.g. relu, elu, sigmoid, softmax, ...
  return act_func


def swapaxes(x, axis1, axis2):
  """
  :param tf.Tensor x:
  :param tf.Tensor|int axis1:
  :param tf.Tensor|int axis2:
  :return: tensor with swapped axes, like numpy.swapaxes
  :rtype: tf.Tensor
  """
  with tf.name_scope("swapaxes"):
    shape = tf.shape(x)
    ndim = x.get_shape().ndims
    if ndim is not None:
      if isinstance(axis1, tf.Tensor) or isinstance(axis2, tf.Tensor):
        perm = [tf.select(tf.equal(axis1, i), axis2,
                          tf.select(tf.equal(axis2, i), axis1,
                                    i))
                for i in range(ndim)]
      else:
        perm = list(range(ndim))
        perm[axis1] = axis2
        perm[axis2] = axis1
    else:
      # Just fall back to the very generic pure symbolic variant.
      rank = tf.rank(x)
      all_axes = tf.range(rank)
      assert all_axes.get_shape().ndims == 1
      axis1 = tf.convert_to_tensor(axis1)
      axis2 = tf.convert_to_tensor(axis2)
      assert axis1.get_shape().ndims == 0
      assert axis2.get_shape().ndims == 0
      axis1_bc = tf.expand_dims(axis1, 0)
      axis2_bc = tf.expand_dims(axis2, 0)
      perm = tf.select(tf.equal(axis1_bc, all_axes), axis2_bc,
                       tf.select(tf.equal(axis2_bc, all_axes), axis1_bc,
                                 all_axes))
    return tf.transpose(x, perm=perm)


def sequence_mask(lengths, **kwargs):
  """
  Wraps around tf.sequence_mask().
  It will cache the value inside the passed object so that we don't recompute it multiple times.

  :param tf.Tensor lengths: shape (batch,)
  :param dict[str] kwargs: passed on to tf.sequence_mask
  :return: tensor mask of shape (batch,maxlen/time). default dtype is bool unless you specify something else
  :rtype: tf.Tensor
  """
  if hasattr(lengths, "_sequence_mask"):
    return lengths._sequence_mask
  mask = tf.sequence_mask(lengths, **kwargs)
  lengths._sequence_mask = mask
  return mask


def sequence_mask_time_major(lengths, **kwargs):
  """
  Wraps around tf.transpose(tf.sequence_mask(), (1,0)).
  It will cache the value inside the passed object so that we don't recompute it multiple times.

  :param tf.Tensor lengths: shape (batch,)
  :param dict[str] kwargs: passed on to tf.sequence_mask
  :return: mask of shape (maxlen/time,batch)
  """
  if hasattr(lengths, "_sequence_mask_time_major"):
    return lengths._sequence_mask_time_major
  mask = sequence_mask(lengths=lengths, **kwargs)  # shape (time,batch)
  mask = tf.transpose(mask, (1, 0))  # shape (batch,time)
  lengths._sequence_mask_time_major = mask
  return mask


def directed(x, direction):
  """
  If direction == 1 or direction is None, returns just x.
  If direction == -1, returns reversed(x).

  :param tf.Tensor x:
  :param int|None direction: -1 or 1 (or None)
  :rtype: tf.Tensor
  """
  if direction == 1 or direction is None:
    return x
  if direction == -1:
    return reversed(x)
  raise ValueError("invalid direction: %r" % direction)


def reversed(x):
  """
  Just returns x[::-1].
  It will cache the value inside the passed object so that we don't recompute it multiple times.

  :param tf.Tensor x:
  :rtype: tf.Tensor
  """
  if hasattr(x, "_reversed_dim0"):
    return x._reversed_dim0
  y = x[::-1]
  x._reversed_dim0 = y
  y._reversed_dim0 = x
  return y


def flatten_with_seq_len_mask(x, seq_lens, time_major=False):
  """
  :param tf.Tensor x: shape (batch,time,...s...) with time_major=False or otherwise shape (time,batch,...s....)
  :param tf.Tensor seq_lens: shape (batch,) of int32
  :param bool time_major: if the time-dim is the first dimension in x
  :return: tensor of shape (time', ...s...) where time' = sum(seq_len) <= batch*time
  :rtype: tf.Tensor
  """
  with tf.name_scope("flatten_with_seq_len_mask"):
    seq_lens = check_input_ndim(seq_lens, 1)
    if time_major:
      x = swapaxes(x, 0, 1)  # get (batch,time,...s...)
    x = check_dim_equal(x, 0, seq_lens, 0)  # batch dim
    # int64? -> https://github.com/tensorflow/tensorflow/issues/6518
    mask = sequence_mask(seq_lens, maxlen=tf.shape(x)[1])  # shape (batch,time)
    mask = check_input_ndim(mask, 2)
    mask = check_dim_equal(mask, 0, x, 0)
    mask = check_dim_equal(mask, 1, x, 1)
    res = tf.boolean_mask(x, mask)
    res = check_input_ndim_equal_offset(res, x, -1)
    return res


def expand_dims_unbroadcast(x, axis, dim):
  """
  :param tf.Tensor x:
  :param int|tf.Tensor axis: new axis
  :param int|tf.Tensor dim: dimension for axis
  :return: if x is of shape (a,b,c) and axis=0, then we return (dim,a,b,c)
  :rtype: tf.Tensor
  """
  with tf.name_scope("expand_dims_unbroadcast"):
    x = tf.expand_dims(x, axis)
    new_ndim = x.get_shape().ndims
    assert new_ndim is not None
    x = tf.tile(x, [dim if (axis == i) else 1 for i in range(new_ndim)])
    return x


def sparse_labels(x, seq_lens, dtype=tf.int32, collapse_repeated=False):
  """
  :param tf.Tensor x: shape (batch,time) -> index, some int type
  :param tf.Tensor|None seq_lens: shape (batch,) of int32|int64
  :param tf.DType|None dtype: if given, will cast the `x` values to this type. ctc_loss() wants int32
  :param bool collapse_repeated: like uniq() behavior
  :return: SparseTensor, e.g. input for tf.nn.ctc_loss()
  :rtype: tf.SparseTensor
  """
  with tf.name_scope("sparse_labels"):
    x = check_input_ndim(x, ndim=2)
    if seq_lens is not None:
      x = check_dim_equal(x, 0, seq_lens, 0)
    if dtype:
      x = tf.cast(x, dtype)
    batch_size = tf.shape(x)[0]
    max_time = tf.shape(x)[1]
    if seq_lens is not None:
      mask = sequence_mask(seq_lens, maxlen=max_time)  # shape (batch,time)
    else:
      mask = tf.ones(dtype=tf.bool, shape=(batch_size, max_time))
    if collapse_repeated:
      with tf.name_scope("collapse_repeated"):
        diffs = tf.concat(1, [tf.ones_like(x[:, :1]), x[:, 1:] - x[:, :-1]])  # shape (batch,time)
        zero = diffs.dtype.as_numpy_dtype()
        mask = tf.logical_and(tf.not_equal(diffs, zero), mask)
    with tf.name_scope("flat_x"):
      flat_x = tf.boolean_mask(x, mask)  # (N, ...s...)
    with tf.name_scope("idxs"):
      time_idxs = expand_dims_unbroadcast(tf.range(max_time), 0, batch_size)  # shape (batch,time)
      flat_time_idxs = tf.boolean_mask(time_idxs, mask)  # (N,)
      batch_idxs = expand_dims_unbroadcast(tf.range(batch_size), 1, max_time)  # shape (batch,time)
      flat_batch_idxs = tf.boolean_mask(batch_idxs, mask)  # (N,)
      flat_idxs = tf.pack([flat_batch_idxs, flat_time_idxs], axis=1)  # shape (N, 2)
      # tf.SparseTensor requires int64 indices
      flat_idxs = tf.cast(flat_idxs, tf.int64)
    with tf.name_scope("shape"):
      shape = [batch_size, max_time]
      # tf.SparseTensor requires int64 shape
      shape = [tf.cast(d, tf.int64) for d in shape]
      shape = tf.convert_to_tensor(shape)
    # tf.SparseTensor args:
    #   indices: A 2-D int64 tensor of shape `[N, ndims]`.
    #   values: A 1-D tensor of any type and shape `[N]`.
    #   shape: A 1-D int64 tensor of shape `[ndims]`.
    return tf.SparseTensor(flat_idxs, flat_x, shape)


def uniq(x):
  """
  :param tf.Tensor x: 1D shape (time,) -> index, some int type
  :return: like numpy.uniq. unlike tf.unique which will never repeat entries.
  Example: uniq([0, 0, 1, 1, 0, 0]) == [0, 1, 0], tf.unique([0, 0, 1, 1, 0, 0]) == [0, 1].
  For a batched variant, see sparse_labels() with option collapse_repeated.
  """
  diffs = tf.concat(0, [tf.ones_like(x[:1]), x[1:] - x[:-1]])
  nonzero_idx = tf.where(diffs)
  x_uniq = tf.gather_nd(x, nonzero_idx)
  return x_uniq


class VariableAssigner(object):
  def __init__(self, var):
    """
    :param tf.Variable var:
    """
    self.var = var
    self.value_placeholder = tf.placeholder(
      name="%s_placeholder_assign_value" % var.name.split("/")[-1][:-2],
      shape=var.get_shape(),
      dtype=var.dtype)
    self.assign_op = tf.assign(self.var, self.value_placeholder)

  def assign(self, value, session):
    """
    :param numpy.ndarray|int|float value:
    :param tf.Session session:
    """
    session.run(self.assign_op, feed_dict={self.value_placeholder: value})


class CudaEnv(object):
  _instance = None
  verbose_find_cuda = False

  def __init__(self):
    self.cuda_path = self._find_cuda_path()

  @classmethod
  def _find_nvcc_in_path(cls):
    """
    :return: yields full path to nvcc
    :rtype: list[str]
    """
    for p in os.environ["PATH"].split(":"):
      pp = "%s/nvcc" % p
      if os.path.exists(pp):
        yield pp

  @classmethod
  def _find_lib_in_ld_path(cls):
    """
    :return: yields full path to libcudart.so
    :rtype: list[str]
    """
    if not os.environ.get("LD_LIBRARY_PATH"):
      return
    for p in os.environ["LD_LIBRARY_PATH"].split(":"):
      pp = "%s/libcudart.so" % p
      if os.path.exists(pp):
        yield pp

  @classmethod
  def _get_lib_dir_name(cls):
    from Util import is_64bit_platform
    if is_64bit_platform():
      return "lib64"
    return "lib"

  @classmethod
  def _cuda_path_candidates(cls):
    for p in cls._find_nvcc_in_path():
      # Expect p == "/usr/local/cuda-8.0/bin/nvcc" or so.
      postfix = "/bin/nvcc"
      if cls.verbose_find_cuda:
        print("found cuda nvcc (wanted postfix: %r): %s" % (postfix, p))
      if not p.endswith(postfix):
        continue
      yield p[:-len(postfix)]
    for p in cls._find_lib_in_ld_path():
      # Expect p == "/usr/local/cuda-8.0/lib64/libcudart.so" or so.
      postfix = "/%s/libcudart.so" % cls._get_lib_dir_name()
      if cls.verbose_find_cuda:
        print("found cuda lib (wanted postfix: %r): %s" % (postfix, p))
      if not p.endswith(postfix):
        continue
      yield p[:-len(postfix)]

  @classmethod
  def _check_valid_cuda_path(cls, p):
    """
    :param str p: path to CUDA, e.g. "/usr/local/cuda-8.0"
    :return: whether this is a valid CUDA path, i.e. we find all what we need
    :rtype: bool
    """
    if cls.verbose_find_cuda:
      print("check valid CUDA path: %s" % p)
    if not os.path.exists("%s/bin/nvcc" % p):
      return False
    if not os.path.exists("%s/include/cuda.h" % p):
      return False
    if not os.path.exists("%s/%s/libcudart.so" % (p, cls._get_lib_dir_name())):
      return False
    return True

  @classmethod
  def _find_cuda_path(cls):
    """
    :return: base CUDA path if we find one, otherwise None
    :rtype: str|None
    """
    for p in cls._cuda_path_candidates():
      if cls._check_valid_cuda_path(p):
        return p
    return None

  def __nonzero__(self):
    """
    :return: whether this is a valid usable CUDA env
    :rtype: bool
    """
    return bool(self.cuda_path)

  def get_compiler_opts(self):
    return [
      "-I", "%s/include" % self.cuda_path, "-L", "%s/%s" % (self.cuda_path, self._get_lib_dir_name()),
      "-x", "cu"]

  def get_compiler_bin(self):
    return "%s/bin/nvcc" % self.cuda_path

  @classmethod
  def get_instance(cls):
    """
    :rtype: CudaEnv
    """
    if cls._instance is not None:
      return cls._instance
    cls._instance = cls()
    return cls._instance


class OpCodeCompiler(object):
  """
  Helper class to compile TF ops on-the-fly, similar as Theano.
  https://www.tensorflow.org/versions/master/how_tos/adding_an_op/
  """

  def __init__(self, base_name, code_version, code, c_macro_defines=None, ld_flags=None, include_deps=None,
               static_version_name=None, should_cleanup_old_all=True, should_cleanup_old_mydir=False):
    """
    :param str base_name: base name for the module, e.g. "zero_out"
    :param int|tuple[int] code_version: check for the cache whether to reuse
    :param str code: the source code itself
    :param dict[str,str|int]|None c_macro_defines: e.g. {"TENSORFLOW": 1}
    :param list[str]|None ld_flags: e.g. ["-lblas"]
    :param list[str]|None include_deps: if provided and an existing lib file, we will check if any dependency is newer
      and we need to recompile. we could also do it automatically via -MD but that seems overkill and too slow.
    :param str|None static_version_name: normally, we use .../base_name/hash as the dir
      but this would use .../base_name/static_version_name.
    :param bool should_cleanup_old_all: whether we should look in the cache dir
      and check all ops if we can delete some old ones which are older than some limit (self._cleanup_time_limit_days)
    :param bool should_cleanup_old_mydir: whether we should delete our op dir before we compile there.
    """
    from Util import get_temp_dir
    self.cache_dir = "%s/returnn_tf_cache" % get_temp_dir()
    self._include_path = tf.sysconfig.get_include()  # e.g. "...python2.7/site-packages/tensorflow/include"
    self.base_name = base_name
    self.code_version = code_version
    self.code = code
    self.c_macro_defines = c_macro_defines or {}
    self.ld_flags = ld_flags or []
    self.include_deps = include_deps
    self.static_version_name = static_version_name
    self._cuda_env = CudaEnv.get_instance()
    self._code_hash = self._make_code_hash()
    self._info_dict = self._make_info_dict()
    self._hash = self._make_hash()
    self._mod = None
    if should_cleanup_old_all:
      self._cleanup_old()
    self._should_cleanup_old_mydir = should_cleanup_old_mydir

  @property
  def _mod_path(self):
    return "%s/ops/%s/%s" % (self.cache_dir, self.base_name, self.static_version_name or self._hash[:10])

  @property
  def _info_filename(self):
    return "%s/info.py" % (self._mod_path,)

  @property
  def _so_filename(self):
    return "%s/%s.so" % (self._mod_path, self.base_name)

  @property
  def _cc_filename(self):
    return "%s/%s.cc" % (self._mod_path, self.base_name)

  _cleanup_time_limit_days = 60

  def _cleanup_old(self):
    mod_path = self._mod_path  # .../base_name/hash
    base_mod_path = os.path.dirname(mod_path)  # .../base_name
    my_mod_path_name = os.path.basename(mod_path)
    if not os.path.exists(base_mod_path):
      return
    import time
    from Util import hms
    cleanup_time_limit_secs = self._cleanup_time_limit_days * 24 * 60 * 60
    for p in os.listdir(base_mod_path):
      if p == my_mod_path_name:
        continue
      full_dir_path = "%s/%s" % (base_mod_path, p)
      if not os.path.isdir(full_dir_path):
        continue  # ignore for now
      info_path = "%s/info.py" % full_dir_path
      if not os.path.exists(info_path):
        self._cleanup_old_path(full_dir_path, reason="corrupt dir, missing info.py")
        continue
      so_path = "%s/%s.so" % (full_dir_path, self.base_name)
      if not os.path.exists(so_path):
        self._cleanup_old_path(full_dir_path, reason="corrupt dir, missing so")
        continue
      dt = time.time() - os.path.getmtime(so_path)
      if dt > cleanup_time_limit_secs:
        self._cleanup_old_path(full_dir_path, reason="%s old" % hms(dt))

  def _cleanup_old_path(self, p, reason):
    print("OpCompiler delete old, %s: %s" % (reason, p))
    assert os.path.exists(p)
    import shutil
    shutil.rmtree(p)

  def _load_info(self):
    filename = self._info_filename
    if not os.path.exists(filename):
      return None
    s = open(filename).read()
    return eval(s)

  _relevant_info_keys = ("tf_version", "code_version", "with_cuda", "code_hash", "c_macro_defines", "ld_flags")

  def _make_info_dict(self):
    return {
      "base_name": self.base_name,
      "tf_version": tf.__version__,
      "tf_include_path": self._include_path,
      "code_version": self.code_version,
      "code_hash": self._code_hash,
      "c_macro_defines": self.c_macro_defines,
      "ld_flags": self.ld_flags,
      "with_cuda": bool(self._cuda_env)
    }

  def _make_code_hash(self):
    import hashlib
    hash = hashlib.md5()
    hash.update(self.code)
    return hash.hexdigest()

  def _make_hash(self):
    import hashlib
    hash = hashlib.md5()
    hash.update("{")
    for key in self._relevant_info_keys:
      hash.update("%s:{%s}" % (key, self._info_dict[key]))
    hash.update("}")
    return hash.hexdigest()

  def _save_info(self):
    filename = self._info_filename
    from Util import betterRepr
    with open(filename, "w") as f:
      f.write("%s\n" % betterRepr(self._info_dict))

  def _need_recompile(self):
    if not os.path.exists(self._so_filename):
      return True
    if self.include_deps:
      so_mtime = os.path.getmtime(self._so_filename)
      for fn in self.include_deps:
        if os.path.getmtime(fn) > so_mtime:
          return True
    old_info = self._load_info()
    new_info = self._make_info_dict()
    if not old_info:
      return True
    # The hash already matched but very unlikely, this could be a collision.
    # Anyway, just do this very cheap check.
    for key in self._relevant_info_keys:
      if key not in old_info:
        return True
      if old_info[key] != new_info[key]:
        return True
    # If no code version is provided, we could also check the code itself now.
    # But I think this is overkill.
    return False

  def _maybe_compile(self):
    if not self._need_recompile():
      # Touch it so that we can see that we used it recently.
      os.utime(self._info_filename, None)
      return
    if self._should_cleanup_old_mydir:
      if os.path.exists(self._mod_path):
        self._cleanup_old_path(self._mod_path, reason="need recompile")
    if not os.path.exists(self._mod_path):
      print("OpCompiler create dir: %s" % self._mod_path)
      os.makedirs(self._mod_path)
    with open(self._cc_filename, "w") as f:
      f.write(self.code)
    common_opts = ["-shared", "-O2", "-std=c++11"]
    if sys.platform == "darwin":
      common_opts += ["-undefined", "dynamic_lookup"]
    common_opts += ["-I", self._include_path]
    compiler_opts = ["-fPIC"]
    if self._cuda_env:
      common_opts += self._cuda_env.get_compiler_opts()
      common_opts += ["-DGOOGLE_CUDA=1"]
      for opt in compiler_opts:
        common_opts += ["-Xcompiler", opt]
    else:
      common_opts += compiler_opts
    common_opts += ["-D_GLIBCXX_USE_CXX11_ABI=0"]  # might be obsolete in the future
    common_opts += ["-D%s=%s" % item for item in sorted(self.c_macro_defines)]
    common_opts += self.ld_flags
    opts = common_opts + [self._cc_filename, "-o", self._so_filename]
    cmd_bin = "g++"
    if self._cuda_env:
      cmd_bin = self._cuda_env.get_compiler_bin()
    cmd_args = [cmd_bin] + opts
    from subprocess import Popen, PIPE, STDOUT, CalledProcessError
    print("OpCompiler call: %s" % " ".join(cmd_args))
    proc = Popen(cmd_args, cwd=self._mod_path, stdout=PIPE, stderr=STDOUT)
    stdout, stderr = proc.communicate()
    assert stderr is None  # should only have stdout
    if proc.returncode != 0:
      print("OpCompiler: %s failed." % cmd_bin)
      print("Original stdout/stderr:")
      print(stdout)
      raise CalledProcessError(returncode=proc.returncode, cmd=cmd_args)
    assert os.path.exists(self._so_filename)
    self._save_info()
    assert not self._need_recompile()

  def load_module(self):
    if self._mod:
      return self._mod
    self._maybe_compile()
    self._mod = tf.load_op_library(self._so_filename)
    return self._mod


def make_var_tuple(v):
  """
  :param tf.Tensor|list[tf.Tensor]|tuple[tf.Tensor] v:
  :return: tuple of tensors
  :rtype: tuple[tf.Tensor]
  """
  if isinstance(v, (int, float, tf.Tensor, tf.Operation)):
    return (v,)
  if isinstance(v, list):
    return tuple(v)
  assert isinstance(v, tuple)
  return v


def add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale):
  """
  Adds scaled noise from a 0-mean normal distribution to gradients.
  Adapted from tf.contrib.layers.optimizers.

  :param list[(tf.Tensor, tf.Variable)] grads_and_vars:
  :param float gradient_noise_scale: used as stddev for tf.truncated_normal().
  :return: adapted grads_and_vars
  :rtype: list[(tf.Tensor, tf.Variable)]
  """
  gradients, variables = zip(*grads_and_vars)
  noisy_gradients = []
  for gradient in gradients:
    if gradient is None:
      noisy_gradients.append(None)
      continue
    if isinstance(gradient, tf.IndexedSlices):
      gradient_shape = gradient.dense_shape
    else:
      gradient_shape = gradient.get_shape()
    noise = tf.truncated_normal(gradient_shape, stddev=gradient_noise_scale)
    noisy_gradients.append(gradient + noise)
  return list(zip(noisy_gradients, variables))


class CustomGradient(object):
  def __init__(self):
    self.num_calls = 0
    self.registered_ops = {}  # func -> decorated func

  def Defun(self, *input_types, **kwargs):
    """
    :param (tf.Operation, tf.Tensor) -> tf.Tensor grad_op:
    :param list[tf.DType] input_types:
    :param dict[str] kwargs: passed to self.register()
    :return: function decorator
    :rtype: ((tf.Tensor) -> tf.Tensor) -> ((tf.Tensor) -> tf.Tensor)
    """

    def decorator(op):
      return self.register(input_types=input_types, op=op, **kwargs)

    return decorator

  def register(self, input_types, op, grad_op, name=None):
    """
    :param list[tf.DType] input_types:
    :param (tf.Tensor) -> tf.Tensor op:
    :param (tf.Operation, tf.Tensor) -> tf.Tensor grad_op:
    :param str name: optional func_name
    :return: op
    :rtype: (tf.Tensor) -> tf.Tensor
    """
    if op in self.registered_ops:
      return self.registered_ops[op]
    from tensorflow.python.framework import function
    op_with_new_grad = function.Defun(*input_types, python_grad_func=grad_op, func_name=name)(op)
    self.registered_ops[op] = op_with_new_grad
    # We need to add one instance of the new op to the graph now because of:
    # https://github.com/tensorflow/tensorflow/issues/6804
    op_with_new_grad(*[tf.placeholder(dtype) for dtype in input_types])
    return op_with_new_grad


custom_gradient = CustomGradient()


def debugRegisterBetterRepr():
  """
  Some types don't have good __repr__ implementations by default (for the current TF version).
  For debugging, it can be helpful to give some more info.
  """

  from tensorflow.python.framework import tensor_util

  def indexed_slices_repr(x):
    """
    :param tf.IndexedSlices x:
    :rtype: str
    """
    dense_shape = tensor_util.constant_value_as_shape(x.dense_shape)
    return "<tf.IndexedSlices %r dense_shape=%r dtype=%r>" % (x.name, dense_shape, x.dtype)

  def op_repr(x):
    """
    :param tf.Operation x:
    :rtype: str
    """
    return "<tf.Operation %r type=%r inputs=%r>" % (x.name, x.type, list(x.inputs))

  tf.IndexedSlices.__repr__ = indexed_slices_repr
  tf.Operation.__repr__ = op_repr
