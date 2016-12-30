
import tensorflow as tf
from tensorflow.python.client import device_lib
import contextlib
import os
import sys


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


class OpCodeCompiler(object):
  """
  Helper class to compile TF ops on-the-fly, similar as Theano.
  https://www.tensorflow.org/versions/master/how_tos/adding_an_op/
  """

  def __init__(self, base_name, code_version, code):
    """
    :param str base_name: base name for the module, e.g. "zero_out"
    :param int|tuple[int] code_version: check for the cache whether to reuse
    :param str code: the source code itself
    """
    self.cache_dir = "/tmp/.returnn_tf_cache"  # TODO...?
    self._include_path = tf.sysconfig.get_include()  # e.g. "...python2.7/site-packages/tensorflow/include"
    self.base_name = base_name
    self.code_version = code_version
    self.code = code
    self._info_dict = self._make_info_dict()
    self._hash = self._make_hash()
    self._mod = None
    self._cleanup_old()

  @property
  def _mod_path(self):
    return "%s/ops/%s/%s" % (self.cache_dir, self.base_name, self._hash[:10])

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
      info_path = "%s/info.py" % full_dir_path
      if not os.path.exists(info_path):
        self._cleanup_old_path(full_dir_path, reason="corrupt dir")
        continue
      dt = time.time() - os.path.getmtime()
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
    return eval(open(filename))

  _relevant_info_keys = ("tf_version", "code_version", "with_cuda")

  def _make_info_dict(self):
    return {
      "base_name": self.base_name,
      "tf_version": tf.__version__,
      "tf_include_path": self._include_path,
      "code_version": self.code_version,
      "with_cuda": False  # TODO...
    }

  def _make_hash(self):
    import hashlib
    hash = hashlib.md5()
    hash.update("{")
    hash.update("code:{%s}" % self.code)
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
    # Not sure if always needed to cleanup first...
    if os.path.exists(self._mod_path):
      self._cleanup_old_path(self._mod_path, reason="need recompile")
    if not os.path.exists(self._mod_path):
      print("OpCompiler create dir: %s" % self._mod_path)
      os.makedirs(self._mod_path)
    common_opts = ["-shared", "-fPIC", "-O2", "-std=c++11"]
    if sys.platform == "darwin":
      common_opts += ["-undefined", "dynamic_lookup"]
    common_opts += ["-I", self._include_path]
    # TODO cuda...
    common_opts += ["-D_GLIBCXX_USE_CXX11_ABI=0"]  # might be obsolete in the future
    opts = common_opts + [self._cc_filename, "-o", self._so_filename]
    cmd_args = ["g++"] + opts
    from subprocess import check_call
    print("OpCompiler call: %s" % cmd_args)
    check_call(cmd_args, cwd=self._mod_path)
    self._save_info()
    assert not self._need_recompile()

  def load_module(self):
    if self._mod:
      return self._mod
    self._maybe_compile()
    self._mod = tf.load_op_library(self._so_filename)
    return self._mod
