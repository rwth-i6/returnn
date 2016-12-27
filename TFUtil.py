
import tensorflow as tf
from tensorflow.python.client import device_lib
import contextlib
import os


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries_%s' % name):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('%s_mean' % name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('%s_stddev' % name, stddev)
    tf.scalar_summary('%s_max' % name, tf.reduce_max(var))
    tf.scalar_summary('%s_min' % name, tf.reduce_min(var))
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


def print_available_devices():
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES is set to %r." % os.environ["CUDA_VISIBLE_DEVICES"])
  else:
    print("CUDA_VISIBLE_DEVICES is not set.")
  print("Collecting TensorFlow device list...")
  devs = list(device_lib.list_local_devices())
  print("Local devices available to TensorFlow:")
  for i, dev in enumerate(devs):
    print("  %i/%i: %s" % (i + 1, len(devs), "\n       ".join(str(dev).splitlines())))


def is_gpu_available():
  """Returns whether TensorFlow can access a GPU."""
  return any(x.device_type == 'GPU' for x in device_lib.list_local_devices())


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


def flatten_with_seq_len_mask(x, seq_lens):
  """
  :param tf.Tensor x: shape (batch,time,...s...)
  :param tf.Tensor seq_lens: shape (batch,) of int32
  :return: tensor of shape (time', ...s...) where time' = sum(seq_len) <= batch*time
  :rtype: tf.Tensor
  """
  with tf.name_scope("flatten_with_seq_len_mask"):
    seq_lens = check_input_ndim(seq_lens, 1)
    x = check_dim_equal(x, 0, seq_lens, 0)  # batch dim
    # int64? -> https://github.com/tensorflow/tensorflow/issues/6518
    mask = tf.sequence_mask(seq_lens, maxlen=tf.shape(x)[1])  # shape (batch,time)
    mask = check_input_ndim(mask, 2)
    mask = check_dim_equal(mask, 0, x, 0)
    mask = check_dim_equal(mask, 1, x, 1)
    res = tf.boolean_mask(x, mask)
    res = check_input_ndim_equal_offset(res, x, -1)
    return res


def sparse_labels(x, seq_lens):
  """
  :param tf.Tensor x: shape (batch,time)
  :param tf.Tensor seq_lens: shape (batch,) of int32
  :return: SparseTensor, e.g. input for tf.nn.ctc_loss()
  :rtype: tf.SparseTensor
  """
  with tf.name_scope("sparse_labels"):
    x = check_input_ndim(x, ndim=2)
    x = check_dim_equal(x, 0, seq_lens, 0)
    batch_size = tf.shape(x)[0]
    mask = tf.sequence_mask(seq_lens, maxlen=tf.shape(x)[1])  # shape (batch,time)
    flat_x = tf.boolean_mask(x, mask)  # (time', ...s...)
    idxs = tf.expand_dims(tf.range(tf.shape(x)[1]), 0)  # shape (batch,time)
    flat_idxs = tf.boolean_mask(idxs, mask)  # (time',)
    return tf.SparseTensor(flat_idxs, flat_x, [batch_size, tf.reduce_max(seq_lens)])
