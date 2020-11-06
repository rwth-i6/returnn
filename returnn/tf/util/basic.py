
"""
Lots of random utility functions for TensorFlow.
"""

from __future__ import print_function, division

import tensorflow as tf
from tensorflow.python.client import device_lib
from tensorflow.python.ops import init_ops
import contextlib
import os
import sys
import threading
import typing
from returnn.util.basic import NotSpecified, NativeCodeCompiler
import returnn.tf.compat as tf_compat
# noinspection PyUnresolvedReferences
from .data import Data, SearchBeam, DimensionTag


class CollectionKeys:
  """
  Extension of :class:`tf.compat.v1.GraphKeys`
  """
  RETURNN_LAYERS = "_RETURNN_layers"  # LayerBase instances
  RETURNN_NET_STACK = "_RETURNN_network_stack"  # TFNetwork instance stack
  STATE_VARS = "_RETURNN_state_vars"  # tf.Variable, like e.g. tf.compat.v1.GraphKeys.LOCAL_VARIABLES


def tf_version_tuple():
  """
  :return: version tuple, e.g. (1, 1, 0), parsed from tf.__version__
  :rtype: tuple[int]
  """
  import re
  # noinspection PyUnresolvedReferences
  return tuple([int(s) for s in re.sub('-rc[0-9]|-dev[0-9]*', '', tf.__version__).split(".")])


def assert_min_tf_version(version, reason):
  """
  :param tuple[int] version: e.g. (1,2,0) or (1,2)
  :param str reason:
  """
  tf_version = tf_version_tuple()
  assert len(version) <= len(tf_version)
  assert tf_version >= version, "Your TF version %r is too old (older than %r). %s" % (tf_version, version, reason)


def have_min_tf_version(version):
  """
  :param tuple[int] version: e.g. (1,2,0) or (1,2)
  :return: True if we have at least that version, or newer
  :rtype: bool
  """
  tf_version = tf_version_tuple()
  assert len(version) <= len(tf_version)
  return tf_version >= version


class CustomUpdate(object):
  """
  Custom updates will be handled by :class:`TFUpdater`.
  """

  def set_on_var(self, var):
    """
    :param tf.Variable var: variable to update. this will be recognized by :class:`TFUpdater.Updater`
    """
    # A bit ugly, but simple.
    setattr(var, "returnn_custom_update", self)

  def update_var(self, var):
    """
    :param tf.Variable var: variable to update
    :return: operation which updates the variable, e.g. tf.compat.v1.assign_add(var, something)
    :rtype: tf.Operation
    """
    raise NotImplementedError


class CustomUpdateExpAverage(CustomUpdate):
  """
  exponential moving average
  """

  def __init__(self, average, alpha):
    """
    :param tf.Tensor average:
    :param float alpha:
    """
    self.average = average
    self.alpha = alpha

  def update_var(self, var):
    """
    :param tf.Variable var:
    :rtype: tf.Tensor
    """
    return tf_compat.v1.assign_add(var, self.alpha * (self.average - var))  # ((alpha - 1) * old + alpha * new)


def set_param_axes_split_info(param, axes_split_info):
  """
  :param tf.Variable|tf.Tensor param:
  :param list[list[int]|None] axes_split_info: e.g. [[n],[n]*4] for LSTM matrices
  """
  check_param_axes_split_info(param.get_shape().as_list(), axes_split_info)
  setattr(param, "returnn_axes_split_info", axes_split_info)


def check_param_axes_split_info(param_shape, axes_split_info):
  """
  :param list[int|None]|tuple[int|None] param_shape:
  :param list[list[int]|None] axes_split_info: e.g. [[n],[n]*4] for LSTM matrices
  """
  assert len(axes_split_info) == len(param_shape)
  for i, parts in enumerate(axes_split_info):
    if parts is not None:
      assert param_shape[i] == sum(parts)


def get_param_axes_split_info(param):
  """
  See :func:`set_param_axes_split_info`.

  :param tf.Variable|tf.Tensor param:
  :rtype: list[list[int]|None]|None
  """
  return getattr(param, "returnn_axes_split_info", None)


def transform_param_axes_split_info_to_new_shape(axes_split_info, new_shape, debug_name="<unknown>"):
  """
  new_shape can be bigger or smaller than the old shape.
  In some simple cases, it is obvious how that should be done, e.g. [[a],[b]*4], [a*2,b*8] -> [[a*2],[b*2]*4]
  In some, it is not so. E.g. [[a+b],[b]*4], [a+b*2,b*8] -> [[a+b*2],[b*2]*4].

  We should try to always return something, though.
  If some case is not covered yet, extend this.

  See test cases as well, :func:`test_transform_param_axes_split_info_to_new_shape`.
  No TF involved here, however, fits better to the functions above.

  :param list[list[int]] axes_split_info:
  :param list[int]|tuple[int] new_shape:
  :param str debug_name:
  :return: new axes-split-info for the new shape
  :rtype: list[list[int]]
  """
  new_axes_split_info = []
  assert len(axes_split_info) == len(new_shape)
  dim_diff = {}  # old-dim -> new-dim
  for new_dim, parts in zip(new_shape, axes_split_info):
    if len(parts) == 1:
      dim_diff[parts[0]] = new_dim
    elif len(set(parts)) == 1:  # all the same
      if new_dim % len(parts) == 0:
        dim_diff[parts[0]] = new_dim // len(parts)  # just a heuristic
  for i, (new_dim, parts) in enumerate(zip(new_shape, axes_split_info)):
    assert len(parts) >= 1, "%s transform %r %r diff %r" % (debug_name, axes_split_info, new_shape, dim_diff)
    if len(parts) == 1:  # simple case
      new_axes_split_info.append([new_dim])
      continue
    new_parts = [dim_diff.get(d) for d in parts]
    if any([d is None for d in new_parts]):
      while sum([d is None for d in new_parts]) > 1:
        # More than one entry is ambiguous. Assume the next one stayed the same.
        j = [d is None for d in new_parts].index(True)
        new_parts[j] = parts[j]
      j = [d is None for d in new_parts].index(True)
      new_parts[j] = new_dim - sum([d for d in new_parts if d is not None])
      assert new_parts[j] > 0, debug_name
    elif sum(new_parts) != new_dim:
      # another heuristic. assume that the first is wrong.
      new_parts[0] = new_dim - sum(new_parts[1:])
      assert new_parts[0] > 0, debug_name
    assert sum(new_parts) == new_dim, debug_name
    new_axes_split_info.append(new_parts)
  return new_axes_split_info


def copy_with_new_split_axes(old_axis_splits, new_axis_splits, old_values, new_values=None):
  """
  On Numpy arrays only, however, fits better to the functions above.

  :param list[list[int]] old_axis_splits:
  :param list[list[int]] new_axis_splits:
  :param numpy.ndarray old_values:
  :param numpy.ndarray new_values:
  :return: new values
  :rtype: numpy.ndarray
  """
  import numpy
  assert len(old_axis_splits) == len(new_axis_splits)
  assert all([len(old_parts) == len(new_parts) for (old_parts, new_parts) in zip(old_axis_splits, new_axis_splits)])
  old_shape = [sum(parts) for parts in old_axis_splits]
  assert tuple(old_shape) == old_values.shape
  new_shape = [sum(parts) for parts in new_axis_splits]
  if new_values is None:
    new_values = numpy.zeros(new_shape, dtype=old_values.dtype)
  for idxs in numpy.ndindex(tuple([len(parts) for parts in old_axis_splits])):
    assert len(idxs) == len(old_axis_splits) == len(new_axis_splits)
    old_offsets = [sum(parts[:i]) for i, parts in zip(idxs, old_axis_splits)]
    new_offsets = [sum(parts[:i]) for i, parts in zip(idxs, new_axis_splits)]
    dims = [min(old_parts[i], new_parts[i]) for i, old_parts, new_parts in zip(idxs, old_axis_splits, new_axis_splits)]
    old_slices = tuple([slice(offset, offset + dim) for offset, dim in zip(old_offsets, dims)])
    new_slices = tuple([slice(offset, offset + dim) for offset, dim in zip(new_offsets, dims)])
    new_values[new_slices] = old_values[old_slices]
  return new_values


class OutputWithActivation(object):
  """
  Stores some tensor before and after some activation function,
  and also the activation function itself.
  (Maybe obsolete when you directly access the TF computation graph; but simpler.)
  """

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
    """
    :rtype: bool
    """
    return self.act_func is tf.nn.softmax

  def get_logits(self):
    """
    :rtype: tf.Tensor
    :return: logits. logits are (not necessarily normalized) log probabilities, i.e. the input of softmax.
    This call assumes that self.y is in probability space.
    """
    if self.is_softmax_act_func():
      return self.x
    if self.act_func is tf.exp:
      return self.x
    return safe_log(self.y)

  def get_log_output(self):
    """
    :rtype: tf.Tensor
    :return: tf.math.log(output)
    """
    if self.is_softmax_act_func():
      return tf.nn.log_softmax(self.x)
    if self.act_func is tf.exp:
      return self.x
    if self.act_func is tf.sigmoid:
      return tf_compat.v1.log_sigmoid(self.x)
    return safe_log(self.y)


def variable_scalar_summaries_dict(x, name=None):
  """
  Collects all interesting information about `x`, such as min/max/mean, etc. (all scalars).
  This is used by :func:`variable_summaries`.

  :param tf.Tensor|tf.Variable x:
  :param str name:
  :return: dicth with key -> scalar info, e.g. with "%s_mean" % name -> tf.reduce_mean(x)
  :rtype: dict[str,tf.Tensor]
  """
  if x.dtype == tf.string:
    return {}
  if not name:
    name = get_base_name(x)
  if x.dtype.is_integer:
    x_float = tf.cast(x, tf.float32)
  else:
    x_float = x
  mean = tf.reduce_mean(x_float)
  stddev = tf.sqrt(tf.reduce_mean(tf.square(x_float - mean)))
  return {
    '%s_mean' % name: mean,
    '%s_stddev' % name: stddev,
    '%s_rms' % name: tf.sqrt(tf.reduce_mean(tf.square(x_float))),
    '%s_l2' % name: tf.sqrt(tf.nn.l2_loss(x_float) * 0.5),
    '%s_max' % name: tf.reduce_max(x),
    '%s_min' % name: tf.reduce_min(x)}


def variable_summaries(var, name=None, with_histogram=False):
  """
  Attach a lot of summaries to a Tensor (for TensorBoard visualization).
  Also see :func:`variable_scalar_summaries_dict`.

  :param tf.Tensor|tf.Variable var:
  :param str name:
  :param bool with_histogram: adds histogram. note that this can add noticeable overhead
  :return: nothing, use :func:`tf.compat.v1.summary.merge_all()` to collect the summaries
  """
  if var.dtype == tf.string:
    return
  if not name:
    name = get_base_name(var)
  with tf.name_scope('summaries_%s' % name):
    for k, v in variable_scalar_summaries_dict(var, name=name).items():
      tf_compat.v1.summary.scalar(k, v)
    if with_histogram:
      tf.summary.histogram('%s_histogram' % name, var)


def get_valid_scope_name_from_str(s):
  """
  :param str s: some name
  :return: valid scope name, might be just s. see tf._VALID_SCOPE_NAME_REGEX and tf._VALID_OP_NAME_REGEX
  :rtype: str
  """
  # For the root name scope, it's even more restrictive, and we must also cover this case.
  # NOTE: Be careful changing this logic. Try to never change the behavior for existing cases,
  # because this name is used e.g. for layers, and you might introduce incompatibility by changes here.
  s = s.replace(":", "__")
  s = s.replace("(", "__")
  s = s.replace(")", "__")
  if s[:1] in "_-\\/":  # invalid first chars
    s = (".%i." % ord(s[0])) + s[1:]
  return s


def get_current_var_scope_name():
  """
  :return: current absolute variable scope name, via tf.compat.v1.variable_scope
  :rtype: str
  """
  v = tf_compat.v1.get_variable_scope()
  return v.name


def get_current_name_scope():
  """
  :return: current absolute name scope, via tf.name_scope
  :rtype: str

  http://stackoverflow.com/questions/40907769/how-to-get-current-tensorflow-name-scope

  Note that this is a private member and might break at some point.
  Note also that this does not need to be the same as get_current_var_scope_name().
  """
  # noinspection PyProtectedMember
  return tf_compat.v1.get_default_graph()._name_stack or ""


@contextlib.contextmanager
def reuse_name_scope(name, absolute=None, **kwargs):
  """
  Context manager to reuse an already created scope.
  We try to both set the variable scope and the name scope.

  :param str|tf.compat.v1.VariableScope name: relative or absolute name scope
    (absolute if absolute=True or if tf.compat.v1.VariableScope).
    Must not end with "/".
  :param bool absolute: if True it will be absolute
  :param kwargs: passed on to `tf.compat.v1.variable_scope`
  :return: yields the variable_scope
  """
  kwargs = kwargs.copy()
  parent_var_scope = None  # type: typing.Optional[tf_compat.v1.VariableScope]
  if not absolute:
    parent_var_scope = tf_compat.v1.get_variable_scope()
  if isinstance(name, tf_compat.v1.VariableScope):
    parent_var_scope = name
    name = name.name
    if absolute is not None:
      assert absolute is True
    absolute = True
  if parent_var_scope:
    for attr in [
      "reuse", "initializer", "regularizer", "caching_device", "partitioner",
      "dtype", "custom_getter", "use_resource", "constraint"
    ]:
      if not hasattr(parent_var_scope, attr):
        continue  # e.g. "constraint" not available in older TF
      kwargs.setdefault(attr, getattr(parent_var_scope, attr))
  assert isinstance(name, str)
  if not absolute:
    assert name
    # First figure out the absolute name scope which we want to reuse/set.
    # The current name scope is more reliable because tf.compat.v1.variable_scope
    # will always also set the name scope.
    current_name_scope = get_current_name_scope()
    if current_name_scope:
      name = current_name_scope + "/" + name
  else:
    current_name_scope = None  # not needed
  assert name[-1:] != "/"
  abs_name = name + "/" if name else ""
  # tf.name_scope with a scope-name ending with "/" will interpret is as absolute name,
  # and use it as-is.
  # In all other cases, it would create a new name-scope with a new unique name,
  # which is not what we want.
  with tf.name_scope(abs_name):
    # tf.name_scope will not set the variable scope.
    # tf.compat.v1.variable_scope will also set the name scope, but the logic is broken
    # for absolute name scopes, thus we had to do the tf.name_scope manually above.
    # We create the dummy_var_scope to force it to reuse that name,
    # and the trailing "/" will work-around the broken tf.compat.v1.variable_scope() usage of tf.name_scope().
    # Afterwards we fix that name again.
    # Note that the reuse-argument might be miss-leading in this context:
    # It means that tf.compat.v1.get_variable() will search for existing variables and errors otherwise.
    var_scope = tf_compat.v1.VariableScope(name=abs_name, reuse=kwargs.get("reuse", None))
    with tf_compat.v1.variable_scope(var_scope, **kwargs) as scope:
      assert isinstance(scope, tf_compat.v1.VariableScope)
      # remove "/" from the end of the var-scope.
      # This is a work-around to fix up the variable scope behavior for nested variable scopes.
      # Warning: This might break at some future point.
      # noinspection PyProtectedMember
      assert scope.name is scope._name
      assert scope.name[-1:] == "/" or scope.name == ""
      # noinspection PyProtectedMember
      scope._name = scope._name[:-1]
      assert name == scope.name, "%r" % current_name_scope
      yield scope


@contextlib.contextmanager
def opt_reuse_name_scope(name):
  """
  :param str|tf.compat.v1.VariableScope name:
  :return: yields the variable_scope
  """
  if name:
    with reuse_name_scope(name) as scope:
      yield scope
  else:
    yield tf_compat.v1.get_variable_scope()


def get_name_scope_of_tensor(x):
  """
  :param tf.Tensor x: has name e.g. "layer0/rec/W:0"
  :return: the name scope of x, e.g. "layer0/rec"
  :rtype: str
  """
  parts = str(x.name).split("/")
  return "/".join(parts[:-1])


def get_base_name(x):
  """
  :param tf.Tensor|tf.Variable x: has name e.g. "layer0/rec/W:0"
  :return: return the base name, e.g. "W", without the output index
  """
  parts = str(x.name).split("/")
  return parts[-1].split(":")[0]


@contextlib.contextmanager
def reuse_name_scope_of_tensor(x, prefix="", postfix="", add_tensor_name=False):
  """
  :param tf.Tensor|tf.Variable x: has name e.g. "layer0/rec/W:0"
  :param str prefix:
  :param str postfix:
  :param bool add_tensor_name:
  :return: reuse the name scope of x, e.g. "layer0/rec", yields scope
  """
  if add_tensor_name:
    from returnn.util.basic import unicode_to_str
    postfix = "/%s%s" % (unicode_to_str(os.path.basename(x.name).split(":")[0]), postfix)
  with reuse_name_scope(prefix + get_name_scope_of_tensor(x) + postfix, absolute=True) as scope:
    yield scope


@contextlib.contextmanager
def default_control_flow_ctx():
  """
  This was earlier called ``var_creation_scope``.

  If you create a variable inside of a while-loop, you might get the following error:

    InvalidArgumentError: The node 'while/w/Assign' has inputs from different frames.
    The input 'while/j' is in frame 'while/while/'. The input 'while/w' is in frame ''.

  This happens when you directly call ``tf.Variable``, because the initial_value might be a tensor
  which depends on the current control flow context.
  See tests/test_TFUtil.py:test_loop_var_creation() for an example.

  Related TF bugs:

    https://github.com/tensorflow/tensorflow/issues/3114
    https://github.com/tensorflow/tensorflow/issues/4478
    https://github.com/tensorflow/tensorflow/issues/8604

  One solution is to reset the current control flow context.
  See also :func:`same_control_flow_ctx`.

  However, with respect to variables, you should instead use
  ``tf.get_variable``, which does not have this problem.
  """
  # Resetting all control dependencies has the effect of also resetting the current control flow context.
  with tf.control_dependencies(None) as dep:
    yield dep


class _ScaledGradientBuilder(object):
  """
  Use the ``scaled_gradient`` instance.
  tf.identity in forward pass, but scales the gradient in backprop.
  Can be used as gradient reversal layer (with negative scale).

  Discussion:
    https://github.com/fchollet/keras/issues/3119
    https://github.com/tensorflow/tensorflow/issues/4342
  Code from here:
    https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py

  Also see :class:`CustomGradient` which is more generic.
  """

  def __init__(self):
    self.num_calls = 0

  def __call__(self, x, scale=1.0):
    """
    :param tf.Tensor x:
    :param float scale:
    :rtype: tf.Tensor
    """
    grad_name = "ScaledGradient%d" % self.num_calls

    from tensorflow.python.framework import ops

    # noinspection PyUnusedLocal
    @ops.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
      return [grad * scale]

    g = tf_compat.v1.get_default_graph()
    with g.gradient_override_map({"Identity": grad_name}):
      y = tf.identity(x, name="scaled_gradient_identity")

    self.num_calls += 1
    return y


scaled_gradient = _ScaledGradientBuilder()


def flip_gradient(x, scale=1.0):
  """
  :param tf.Tensor x:
  :param float scale:
  :return: identity(x) but with flipped gradient (optionally scaled)
  :rtype: tf.Tensor
  """
  return scaled_gradient(x, scale=-scale)


def lookup_grad_func_by_name(op_type):
  """
  :param str op_type:
  :return: function grad_func(op, grad), or raises LookupError
  """
  from tensorflow.python.framework import ops
  # Also see ops.RegisterGradient and ops.get_gradient_function.
  # noinspection PyProtectedMember
  return ops._gradient_registry.lookup(op_type)


def opt_register_grad_func(op_type, grad_func, assert_is_same=True):
  """
  :param str op_type:
  :param grad_func: function grad_func(op, grad)
  :param bool assert_is_same:
  """
  try:
    f = lookup_grad_func_by_name(op_type)
  except LookupError:
    f = None
  if f is not None:
    if assert_is_same:
      assert f is grad_func, "already registered grad for %r, and not the same func: %r != %r" % (op_type, f, grad_func)
  else:
    from tensorflow.python.framework import ops
    ops.RegisterGradient(op_type)(grad_func)


def identity_with_check_numerics(x, with_grad=True, name="identity_with_check_numerics"):
  """
  Returns identity(x), but with additional check_numerics control dependency,
  and optionally the same for its gradient.
  See also :func:`TFUpdater.add_check_numerics_ops`, which will add checks for the whole graph.

  :param tf.Tensor x:
  :param bool with_grad: whether the check will also be added for the gradient
  :param str name:
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    with tf.control_dependencies([
          tf_compat.v1.check_numerics(x, message="%s check_numerics for tensor %s" % (name, x.name))]):
      if with_grad:
        # An alternative to gradient_override_map would be :class:`CustomGradient` which is more generic.
        # noinspection PyUnusedLocal
        def _identity_with_check_numerics_grad(op, grad):
          return identity_with_check_numerics(grad, with_grad=True, name="%s_grad" % name)

        grad_name = "%s_with_grad" % name
        opt_register_grad_func(
          op_type=grad_name,
          grad_func=_identity_with_check_numerics_grad,
          assert_is_same=False)

        g = tf_compat.v1.get_default_graph()
        with g.gradient_override_map({"Identity": grad_name}):
          y = tf.identity(x)

      else:
        y = tf.identity(x)

      return y


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
  with tf.name_scope("check_input_ndim"):
    with tf.control_dependencies(
      [tf_compat.v1.assert_equal(tf.rank(x), ndim, data=["ndim not %i" % ndim, tf.shape(x)])]):
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
  with tf.name_scope("check_input_ndim_equal_offset"):
    with tf.control_dependencies(
        [tf_compat.v1.assert_equal(
          tf.rank(x), tf.rank(y) + y_ndim_offset,
          data=["ndim not equal with offset %i" % y_ndim_offset, tf.shape(x), tf.shape(y)])]):
      return tf.identity(x, "identity_with_ndim_equal_check")


def check_input_dim(x, axis, dim):
  """
  :param tf.Tensor x:
  :param int axis: which axis to check
  :param int|tf.Tensor dim:
  :return: x with check added
  :rtype: tf.Tensor
  """
  dyn_shape = x.get_shape()
  if dyn_shape.ndims is not None and isinstance(dim, int):
    if dyn_shape.dims[axis].value is not None:
      assert dyn_shape.dims[axis].value == dim
      return x
  # Need to fall-back to runtime check.
  with tf.name_scope("check_input_dim"):
    with tf.control_dependencies([
          tf_compat.v1.assert_equal(
            tf.shape(x)[axis], dim, data=["shape[%i]:" % (axis,), tf.shape(x), "!=", "dim:", dim])]):
      return tf.identity(x, "identity_with_dim_check")


def check_dim_equal(x, x_axis, y, y_axis, extra_msg=()):
  """
  :param tf.Tensor x:
  :param int x_axis: which axis to check
  :param tf.Tensor y:
  :param int y_axis: which axis to check
  :param list[str]|tuple[str] extra_msg: will be printed additionally if it fails
  :return: x with check added that shape(x)[x_axis] == shape(y)[y_axis]
  :rtype: tf.Tensor
  """
  x_dyn_shape = x.get_shape()
  y_dyn_shape = y.get_shape()
  if x_dyn_shape.ndims is not None and y_dyn_shape.ndims is not None:
    if x_dyn_shape.dims[x_axis].value is not None and y_dyn_shape.dims[y_axis].value is not None:
      assert x_dyn_shape.dims[x_axis].value == y_dyn_shape.dims[y_axis].value, extra_msg
      return x
  # Need to fall-back to runtime check.
  with tf.name_scope("check_dim_equal"):
    shape_x = tf.shape(x)
    shape_y = tf.shape(y)
    with tf.control_dependencies(
      [tf_compat.v1.assert_equal(
         shape_x[x_axis], shape_y[y_axis],
         data=["x.shape[%i] != y.shape[%i]" % (x_axis, y_axis), shape_x, shape_y] + list(extra_msg))]):
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
  with tf.name_scope("check_shape_equal"):
    with tf.control_dependencies(
      [tf_compat.v1.assert_equal(
        tf.shape(x), tf.shape(y),
        data=["x.shape not y.shape",
              tf.shape(x), tf.shape(y)])]):
      return tf.identity(x, "identity_with_shape_equal_check")


def get_shape_dim(x, axis, name="shape_dim"):
  """
  :param tf.Tensor x:
  :param int axis: which axis
  :param str name:
  :return: x.shape[axis] either as a static int or otherwise as an expression
  :rtype: int|tf.Tensor
  """
  dyn_shape = x.get_shape()
  if dyn_shape.ndims is not None:
    if dyn_shape.dims[axis].value is not None:
      return dyn_shape.dims[axis].value
  # Need to fall-back to runtime.
  with tf.name_scope(name):
    return tf.shape(x)[axis]


def get_shape(x):
  """
  :param tf.Tensor|tf.Variable x:
  :return: list of scalars, which are either int if known statically, or otherwise expressions
  :rtype: list[int|tf.Tensor]
  """
  with tf.name_scope("get_shape"):
    static_shape = x.get_shape()
    dyn_shape = None if static_shape.is_fully_defined() else tf.shape(x)
    assert static_shape.ndims is not None
    return [static_shape.dims[i].value
            if static_shape.dims[i].value is not None
            else dyn_shape[i]
            for i in range(static_shape.ndims)]


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
  with tf.name_scope("identity_with_ops"):
    with tf.control_dependencies(ops()):
      return tf.identity(x, name="identity_with_ops")


_setup_tf_thread_pools_called_once = False


def setup_tf_thread_pools(num_threads=None, log_file=None, tf_session_opts=None):
  """
  See here for documentation of intra_op_parallelism_threads and inter_op_parallelism_threads:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto

  intra_op_parallelism_threads is used for the LocalDevice::EigenThreadPoolInfo, which is always global.
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/local_device.cc

  inter_op_parallelism_threads is used for the (global if not use_per_session_threads) session thread pool.
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/common_runtime/direct_session.cc

  TF will setup the thread pools on first usage. That can happen quite early, esp for intra_op_parallelism_threads.
  E.g. list_local_devices() will trigger this, i.e. any call to is_gpu_available() or print_available_devices().
  For debugging, you can set the env-var TF_CPP_MIN_VLOG_LEVEL=1 and then check for these message::

      Local device intra op parallelism threads: 4
      Direct session inter op parallelism threads: 4

  Thus, call this function as early as possible with your preferred number of threads,
  used for both thread pools.
  It will create a dummy session and directly close it again, but if you use the global thread pools,
  those settings will remain for further sessions.
  This function will only execute on the first call.

  :param int num_threads: used for both intra and inter parallelism thread pools
  :param stream|None log_file:
  :param dict[str] tf_session_opts:
  """
  global _setup_tf_thread_pools_called_once
  if _setup_tf_thread_pools_called_once:
    return
  _setup_tf_thread_pools_called_once = True
  if not num_threads:
    from returnn.util.basic import guess_requested_max_num_threads
    num_threads = guess_requested_max_num_threads(log_file=log_file, fallback_num_cpus=False)
  # See options here:
  # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto
  if tf_session_opts:
    opts = tf_session_opts.copy()
  else:
    opts = {}
  assert isinstance(opts, dict)
  opts.setdefault("log_device_placement", False)
  opts.setdefault("device_count", {}).setdefault("GPU", 0)
  if num_threads:
    opts.setdefault("intra_op_parallelism_threads", num_threads)
    opts.setdefault("inter_op_parallelism_threads", num_threads)
  if log_file:
    print("Setup TF inter and intra global thread pools, num_threads %r, session opts %r." % (num_threads, opts),
          file=log_file)
  with tf_compat.v1.Session(config=tf_compat.v1.ConfigProto(**opts)) as session:
    session.close()


def check_initial_tf_thread_pool_init(tf_session_opts=None):
  """
  Makes sure that the TF thread pools are initialized with the requested settings.
  You probably want to call this very early.

  :param dict[str]|None tf_session_opts:
  """
  if not _setup_tf_thread_pools_called_once:
    from returnn.util.basic import try_get_caller_name
    print("setup_tf_thread_pools() not yet called (via func %s), calling it now." %
          try_get_caller_name(fallback="<unknown>"))
    setup_tf_thread_pools(tf_session_opts=tf_session_opts, log_file=sys.stdout)


class _DeviceAttributes:
  """
  Like tf.python.client.session._DeviceAttributes but extended by physical_device_desc.
  """
  def __init__(self, dev):
    """
    :param tensorflow.python.client.session._DeviceAttributes dev:
    """
    self.name = dev.name  # type: str
    self.device_type = dev.device_type  # type: str
    self.memory_limit_bytes = dev.memory_limit_bytes  # type: int
    self.physical_device_desc = None  # type: typing.Optional[str]

  def set_physical_device_desc(self, session):
    """
    :param tf.compat.v1.Session session:
    """
    if "XLA_" in self.name:
      return  # XLA not supported currently by get_device_attr...
    physical_device_desc = session.run(get_device_attr(self.name))
    self.physical_device_desc = physical_device_desc.decode("utf8")

  def __str__(self):
    # Similar to tensorflow.core.framework.device_attributes_pb2.DeviceAttributes.
    return "".join([
      "%s: %r\n" % (k, getattr(self, k))
      for k in ["name", "device_type", "memory_limit_bytes", "physical_device_desc"]])

  def __repr__(self):
    return "<%s %s>" % (self.__class__.__name__, self.__str__().strip().replace("\n", ", "))


_list_local_devices = None


def get_tf_list_local_devices(tf_session_opts=None, file=None):
  """
  This uses tensorflow.device_lib.list_local_devices().
  Note that a call to this will trigger the internal TF thread pool inits,
  so you should call :func:`setup_tf_thread_pools` first.
  Note that this will list all available devices.
  Any TF session might only use a subset of these.
  You can get the list available in a given TF session by :func:`tf.compat.v1.Session.list_devices`.

  :param dict[str]|None tf_session_opts: if given, will init a temp tf.compat.v1.Session with these opts
  :param typing.TextIO|None file: log_stream stream for print statements, defaults to sys.stdout
  :rtype: list[tensorflow.core.framework.device_attributes_pb2.DeviceAttributes|_DeviceAttributes]
  """
  check_initial_tf_thread_pool_init(tf_session_opts=tf_session_opts)
  global _list_local_devices
  if _list_local_devices is not None:
    return _list_local_devices
  print("Collecting TensorFlow device list...", file=file)
  if tf_session_opts and tf_session_opts.get("gpu_options", {}):
    # On any gpu_options, e.g. gpu_options.per_process_gpu_memory_fraction:
    # The first usage of a device will use the provided options.
    # list_local_devices() will init the devices to get information, but it will just use the default GPU options.
    # To avoid that, we must do the custom session creation here.
    # The earlier session creation via setup_tf_thread_pools() is not enough,
    # because that uses device_count={"GPU":0}, which will not init the GPU.
    # On gpu_options.visible_device_list:
    # Note that setting CUDA_VISIBLE_DEVICES to the corresponding subset will not work because
    # CUDA will internally cache the devices, thus the first call to list_local_devices will init
    # all visible devices at that point, and TF/CUDA will get confused later
    # when another set of devices is visible.
    # However, getting the list via tf.compat.v1.Session.list_devices() will not provide us with a full DeviceAttributes
    # with all needed information, as dev.physical_device_desc is missing,
    # and we need that for e.g. get_available_gpu_min_compute_capability.
    # See also: https://github.com/tensorflow/tensorflow/issues/9374
    # However, we have get_device_attr, which provides gives us physical_device_desc.
    with tf_compat.v1.Session(config=tf_compat.v1.ConfigProto(**tf_session_opts)) as session:
      devs = list(session.list_devices())
      _list_local_devices = [_DeviceAttributes(dev=dev) for dev in devs]
      # Set physical_device_desc after we assigned _list_local_devices,
      # because there might happen recursive calls to this function, e.g. via is_gpu_available,
      # which will be called via get_device_attr, when the op will be compiled.
      for dev in _list_local_devices:
        dev.set_physical_device_desc(session=session)
      session.close()
  else:
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


def print_available_devices(tf_session_opts=None, file=None):
  """
  Prints the available TF devices on `file` (stdout by default).
  This uses tensorflow.device_lib.list_local_devices().
  Note that a call to this will trigger the internal TF thread pool inits,
  so you should call :func:`setup_tf_thread_pools` first.

  :param dict[str]|None tf_session_opts: if given, will init a temp Session with these opts
  :param typing.TextIO|None file: file stream for print statements, defaults to sys.stdout
  """
  if file is None:
    file = sys.stdout
  cuda_visible_devs = None
  if "CUDA_VISIBLE_DEVICES" in os.environ:
    print("CUDA_VISIBLE_DEVICES is set to %r." % os.environ["CUDA_VISIBLE_DEVICES"], file=file)
    cuda_visible_devs = dict(enumerate([int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d]))
  else:
    print("CUDA_VISIBLE_DEVICES is not set.", file=file)
  if tf_session_opts and tf_session_opts.get("gpu_options", {}).get("visible_device_list", None):
    print("TF session gpu_options.visible_device_list is set to %r." % (
      tf_session_opts["gpu_options"]["visible_device_list"],), file=file)
  devs = get_tf_list_local_devices(tf_session_opts=tf_session_opts, file=file)
  print("Local devices available to TensorFlow:", file=file)
  for i, dev in enumerate(devs):
    print("  %i/%i: %s" % (i + 1, len(devs), "\n       ".join(str(dev).splitlines())), file=file)

  # Theano prints sth like: Using gpu device 2: GeForce GTX 980 (...)
  # Print in a similar format so that some scripts which grep our stdout work just as before.
  for dev in devs:
    if dev.device_type == "GPU":
      d = _parse_physical_device_desc(dev.physical_device_desc)
      dev_id = int(d["device"])
      if cuda_visible_devs:
        dev_id = cuda_visible_devs[dev_id]
      dev_name = d["name"]
      print("Using gpu device %i: %s" % (dev_id, dev_name), file=file)


def is_gpu_available():
  """
  Returns whether TensorFlow can access a GPU.
  This uses tensorflow.device_lib.list_local_devices(),
  i.e. this is independent from the current TF session.
  If you want to know whether the current TF session has a GPU available,
  use :func:`is_gpu_available_in_session`.

  Note that a call to this will trigger the internal TF thread pool inits,
  so you should call :func:`setup_tf_thread_pools` first.

  :rtype: bool
  """
  # Also, we could maybe use tf.test.is_gpu_available().
  return len(get_available_gpu_devices()) > 0


def is_gpu_available_in_session(session=None):
  """
  :param tf.compat.v1.Session|None session:
    If None, will use current active/default session.
    If that is also not available (no current active session),
    we check a RETURNN global config,
    and return whether the RETURNN global config will use GPU or not.
    If the RETURNN global config is not available,
    we will use if a GPU is in general available for TF.
  :returns: whether the TensorFlow session has a GPU device.
    Also see :func:`is_gpu_available`.
  :rtype: bool
  """
  if not session:
    session = tf_compat.v1.get_default_session()
    if not session:
      # Now we assume a RETURNN environment, with a global config.
      import returnn.config
      config = returnn.config.get_global_config(raise_exception=False)  # use set_global_config if this fails
      if config:
        return returnn.config.tf_should_use_gpu(config)
      return is_gpu_available()
  devs = session.list_devices()
  return any([x for x in devs if x.device_type == 'GPU'])


def get_available_gpu_devices():
  """
  Returns a list of available GPU devices.
  This uses tensorflow.device_lib.list_local_devices().
  Note that a call to this will trigger the internal TF thread pool inits,
  so you should call :func:`setup_tf_thread_pools` first.

  :rtype: list[tensorflow.core.framework.device_attributes_pb2.DeviceAttributes|_DeviceAttributes]
  """
  return [x for x in get_tf_list_local_devices() if x.device_type == 'GPU']


def get_available_gpu_min_compute_capability():
  """
  Uses :func:`get_available_gpu_devices`.

  :return: e.g. 3.0, or 5.0, etc, or None
  :rtype: float|None
  """
  cap = None
  for dev in get_available_gpu_devices():
    assert dev.physical_device_desc is not None
    desc = _parse_physical_device_desc(dev.physical_device_desc)
    dev_cap = float(desc['compute capability'])
    if cap is None:
      cap = dev_cap
    else:
      cap = min(cap, dev_cap)
  return cap


def dot(a, b, transpose_b=False):
  """
  :param tf.Tensor a: shape [...da...,d]
  :param tf.Tensor b: shape [d,...db...] (or [...db...,d] if transpose_b)
  :param bool transpose_b:
  :return: tensor of shape [...da...,...db...]
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
    a = check_dim_equal(a, -1, b, -1 if transpose_b else 0)
    if a_ndim == b_ndim == 1:
      return tf.reduce_sum(a * b)
    d = get_shape_dim(b, -1 if transpose_b else 0)
    assert a_ndim >= 2 and b_ndim >= 2
    res_shape = None
    if a_ndim > 2 or b_ndim > 2:
      res_shape = (
        [get_shape_dim(a, i) for i in range(0, a_ndim - 1)] +
        [get_shape_dim(b, i + (0 if transpose_b else 1)) for i in range(0, b_ndim - 1)])
    if a_ndim > 2:
      a = tf.reshape(a, (-1, d))
    if b_ndim > 2:
      b = tf.reshape(b, (d, -1)) if transpose_b else tf.reshape(b, (d, -1))
    res = tf.matmul(a, b, transpose_b=transpose_b)
    if a_ndim > 2 or b_ndim > 2:
      res = tf.reshape(res, res_shape)
    return res


def identity(x):
  """
  :param tf.Tensor x:
  :rtype: tf.Tensor
  """
  return x


def _plus(a, b):
  return a + b


def _minus(a, b):
  return a - b


def _mul(a, b):
  return a * b


def _div(a, b):
  return a / b


_bin_ops = {"+": _plus, "-": _minus, "*": _mul, "/": _div}
_act_func_with_op_cache = {}  # type: typing.Dict[str,typing.Callable[[tf.Tensor],tf.Tensor]]


def _get_act_func_with_op(s):
  """
  :param str s: e.g. "2 * sigmoid" or even "3 + 2 * sigmoid"
  :rtype: (tf.Tensor) -> tf.Tensor
  """
  if s in _act_func_with_op_cache:
    return _act_func_with_op_cache[s]

  def _convert(v):
    v = v.strip()
    from returnn.util.basic import str_is_number
    if str_is_number(v):
      try:
        v = int(v)
      except ValueError:
        v = float(v)
      return lambda x: v
    else:
      return get_activation_function(v)

  a, b = None, None
  for k in "+-*/":
    if k in s:
      a, b = s.split(k, 2)
      a, b = _convert(a), _convert(b)

      def combined_op(x):
        """
        :param tf.Tensor x:
        :rtype: tf.Tensor
        """
        return _bin_ops[k](a(x), b(x))

      _act_func_with_op_cache[s] = combined_op
      return combined_op
  assert False


def get_activation_function(s):
  """
  :param str|None s:
  :rtype: (tf.Tensor) -> tf.Tensor
  """
  if not s or s in ["none", "identity"]:
    return identity
  if "(" in s:
    return eval("lambda x: %s" % s, {"tf": tf})
  if any(k in s for k in _bin_ops):
    return _get_act_func_with_op(s)
  if hasattr(tf.nn, s):
    return getattr(tf.nn, s)  # e.g. relu, elu, sigmoid, softmax, ...
  elif hasattr(tf, s):
    return getattr(tf, s)  # e.g. log, abs
  elif s in globals():
    return globals()[s]  # e.g. safe_log
  raise Exception("invalid activation function: %r" % s)


def gelu(x):
  """
  Gaussian Error Linear Units (GELUs) (https://arxiv.org/abs/1606.08415).
  Alternative to relu.

  :param tf.Tensor x:
  :rtype: tf.Tensor
  """
  import numpy
  return 0.5 * x * (1 + tf.tanh(numpy.sqrt(2 / numpy.pi) * (x + 0.044715 * tf.pow(x, 3))))


def gelu2(x):
  """
  Another approximation of the GELU (https://github.com/hendrycks/GELUs).
  Faster but less accurate than `gelu` (https://github.com/hendrycks/GELUs).

  :param tf.Tensor x:
  :rtype: tf.Tensor
  """
  return x * tf.sigmoid(1.702 * x)


def random_uniform_abs_initializer(limit, **kwargs):
  """
  :param float|int|tf.Tensor limit:
  :param kwargs: passed to tf.random_uniform_initializer
  :rtype: tensorflow.python.ops.init_ops.Initializer
  """
  return tf.random_uniform_initializer(minval=-limit, maxval=limit, **kwargs)


def xavier_initializer(uniform=True, seed=None, dtype=tf.float32):
  """
  Alias for tf.glorot_uniform_initializer or tf.glorot_normal_initializer.

  :param bool uniform: uniform or normal distribution
  :param int seed:
  :param tf.DType dtype:
  :return: ((tuple[int]) -> tf.Tensor) | tensorflow.python.ops.init_ops.Initializer
  """
  from tensorflow.python.ops import init_ops
  return init_ops.variance_scaling_initializer(
    scale=1.0, mode='fan_avg', distribution="uniform" if uniform else "normal", seed=seed, dtype=dtype)


def wrap_distribution_non_zero(x, zero_limit, limit):
  """
  :param tf.Tensor x: values in [-limit,limit]
  :param float zero_limit:
  :param float limit:
  :return: same shape as x.
    rescale and shifts such that values from [-zero_limit,zero_limit] are excluded.
    still values are in [-limit,limit].
  :rtype: tf.Tensor
  """
  assert limit > 0 and limit > zero_limit > 0
  # Rescale the range [0,limit] to [zero_limit,limit] (and same in negative).
  x_rescaled = x * ((limit - zero_limit) / limit)
  shift = tf.ones_like(x) * zero_limit
  return x_rescaled + tf.where(tf.greater_equal(x, 0.0), shift, -shift)


class VarianceScalingNonZero(init_ops.VarianceScaling):
  """
  Same as :class:`tf.VarianceScaling`, i.e. truncated normal or uniform from [-limit,limit] for some limit,
  except that we exclude the range [-limit*non_zero_fraction,limit*non_zero_fraction].
  non_zero_fraction=0 would yield no difference.

  For reference, to get the behavior of glorot_uniform, use these args:
    mode="fan_avg", distribution="uniform"
  """

  def __init__(self, non_zero_fraction=0.5, **kwargs):
    super(VarianceScalingNonZero, self).__init__(**kwargs)
    assert 0 <= non_zero_fraction <= 1
    self.non_zero_fraction = non_zero_fraction

  def __call__(self, shape, dtype=None, partition_info=None):
    """
    :param tuple[int] shape:
    :param tf.DType dtype:
    :param partition_info:
    :rtype: tf.Tensor
    """
    import numpy
    from tensorflow.python.ops import init_ops
    if dtype is None:
      dtype = self.dtype
    scale = self.scale
    scale_shape = shape
    if partition_info is not None:
      scale_shape = partition_info.full_shape
    # noinspection PyProtectedMember
    fan_in, fan_out = init_ops._compute_fans(scale_shape)
    if self.mode == "fan_in":
      scale /= max(1., fan_in)
    elif self.mode == "fan_out":
      scale /= max(1., fan_out)
    else:
      assert self.mode == "fan_avg"
      scale /= max(1., (fan_in + fan_out) / 2.)
    if self.distribution == "normal":
      stddev = numpy.sqrt(scale)
      limit = stddev * 2
      x = tf_compat.v1.truncated_normal(shape, mean=0.0, stddev=stddev, dtype=dtype, seed=self.seed)
    else:
      assert self.distribution == "uniform"
      limit = numpy.sqrt(3.0 * scale)
      x = tf_compat.v1.random_uniform(shape, minval=-limit, maxval=limit, dtype=dtype, seed=self.seed)
    x = wrap_distribution_non_zero(x, zero_limit=self.non_zero_fraction * limit, limit=limit)
    return x


variance_scaling_non_zero_initializer = VarianceScalingNonZero


def load_txt_file_initializer(filename, dtype=tf.float32):
  """
  :param str filename:
  :param tf.DType dtype:
  :return: function, when called, will return the content
  :rtype: ()->tf.Tensor
  """
  assert dtype == tf.float32, "only float32 supported currently"
  dtype_ = dtype

  def py_loader():
    """
    :rtype: numpy.ndarray
    """
    # Alternative: numpy.loadtxt.
    import numpy
    from returnn.util.basic import load_txt_vector
    return numpy.array(load_txt_vector(filename), dtype="float32")

  from tensorflow.python.ops import init_ops

  class LoadTxtFileInitializer(init_ops.Initializer):
    """
    Load TXT file TF initializer class.
    """
    # noinspection PyShadowingNames
    def __call__(self, shape, dtype=None, partition_info=None):
      v = tf_compat.v1.py_func(py_loader, [], dtype_)
      v.set_shape(shape)
      return v

  return LoadTxtFileInitializer()


class GammatoneFilterbankInitializer(init_ops.Initializer):
  """
  Initializer for a gammatone filterbank, e.g., to initialize weights of a convolutional layer.
  """

  def __init__(self, **kwargs):
    """
    :param kwargs: kwargs for GammatoneFilterbank
    """
    from returnn.util.sig_proc import GammatoneFilterbank
    self.gammatone_filterbank = GammatoneFilterbank(**kwargs)

  def __call__(self, shape, dtype=tf.float32, partition_info=None):
    """
    :param tuple[int] shape:
    :param tf.DType dtype:
    :param partition_info:
    :rtype: tf.Tensor
    """
    import numpy
    fbank = self.gammatone_filterbank.get_gammatone_filterbank()
    # Check whether shape and specified parameters for gammatone filterbank match. Currently, this is just implemented
    # for the case that the expected shape is (sample_rate * length, num_channels) with possibly additional 1-dim axes
    # in between.
    gammatone_shape = (
      int(self.gammatone_filterbank.sample_rate * self.gammatone_filterbank.length),
      self.gammatone_filterbank.num_channels
    )
    assert shape[0] == gammatone_shape[0], "Expected shape does not match given gammatone filterbank paramerers"
    assert shape[-1] == gammatone_shape[-1], "Expected shape does not match given gammatone filterbank paramerers"
    fbank = numpy.reshape(fbank, shape)
    fbank = fbank.astype(dtype.as_numpy_dtype)
    return tf.convert_to_tensor(fbank, dtype=dtype)


def get_initializer(s, seed=None, eval_local_ns=None, dtype=tf.float32):
  """
  :param str|dict[str]|float|numpy.ndarray s: e.g. "glorot_uniform" or "truncated_normal" or "orthogonal",
    or config dict with "class",
    or string to be `eval`ed if it contains "(". constant if a float is given.
  :param int|tf.Tensor seed:
  :param dict[str]|None eval_local_ns:
  :param tf.DType|str dtype:
  :return: (function (shape) -> tf.Tensor) | tf.Initializer
  :rtype: ((tuple[int]) -> tf.Tensor) | tf.Initializer
  """
  dtype = tf.as_dtype(dtype).base_dtype
  assert isinstance(dtype, tf.DType)
  import numpy
  if isinstance(s, (float, int)):
    if s == 0:
      return tf_compat.v1.zeros_initializer(dtype=dtype)
    if s == 1:
      return tf_compat.v1.ones_initializer(dtype=dtype)
    return tf_compat.v1.constant_initializer(s, dtype=dtype)
  if isinstance(s, numpy.ndarray):
    return tf_compat.v1.constant_initializer(s, dtype=dtype, verify_shape=True)
  if not s and dtype == tf.string:
    return tf_compat.v1.zeros_initializer(dtype=dtype)
  if not s and dtype.is_integer:
    return tf_compat.v1.zeros_initializer(dtype=dtype)
  import math
  from tensorflow.python.ops import init_ops

  def error():
    """
    Dump error info.
    """
    print("Error for initializer %r." % s)
    print("Possible initializers:")
    from inspect import isclass
    for key, value in sorted(vars(init_ops).items()):
      if isclass(value) and issubclass(value, init_ops.Initializer):
        print("  %s" % key)

  ns = dict(globals())
  ns.update(vars(tf_compat.v1))
  ns.update(vars(init_ops))
  ns.update(vars(math))
  ns["numpy"] = numpy
  for k in sorted(list(ns.keys())):
    if k.endswith("_initializer"):
      k_short = k[:-len("_initializer")]
      if k_short not in ns:
        ns[k_short] = ns[k]
  f = None
  try:
    if isinstance(s, str):
      if "(" in s:
        f = eval(s, ns, eval_local_ns)
      elif s + "_initializer" in ns:
        f = ns[s + "_initializer"]()
      elif s in ns:
        f = ns[s]()
    elif isinstance(s, dict):
      s = s.copy()
      class_name = s.pop("class")
      class_ = ns[class_name]
      assert issubclass(class_, init_ops.Initializer)
      f = class_.from_config(s)
    else:
      raise ValueError("invalid get_initializer argument, expected string or dict, got: %r" % s)
    if isinstance(f, (float, int)):
      if f == 0:
        return tf_compat.v1.zeros_initializer(dtype=dtype)
      if f == 1:
        return tf_compat.v1.ones_initializer(dtype=dtype)
      return tf_compat.v1.constant_initializer(f, dtype=dtype)
    elif isinstance(f, numpy.ndarray):
      return tf_compat.v1.constant_initializer(f, dtype=dtype, verify_shape=True)
    if not f:
      raise Exception("invalid initializer: %r" % s)
    if seed is not None:
      assert isinstance(f, init_ops.Initializer)
      if hasattr(f, "seed"):
        f.seed = seed
  except Exception:
    error()
    raise
  return f


def dropout(x, keep_prob, noise_shape=None, seed=None, name=None, cond_on_train=False, apply_correction_factor=True):
  """
  Computes dropout.
  Like :func:`tf.nn.dropout` but avoid :func:`tf.div` if possible.
  If `noise_shape` is statically known, and `x` is inside a recurrent loop,
  we will reuse the same mask for all frames.

  :param tf.Tensor x:
  :param float|tf.Tensor keep_prob:
  :param tf.Tensor|tuple[int|None] noise_shape: 1 will broadcast in that dimension, None will not broadcast
  :param int seed:
  :param str name:
  :param bool cond_on_train: automatically wrap through :func:`cond_on_train_flag`
  :param bool apply_correction_factor:
  """
  if cond_on_train:
    return cond_on_train_flag(
      lambda: dropout(x, keep_prob=keep_prob, noise_shape=noise_shape, seed=seed, name=name),
      lambda: x)
  with tf.name_scope(name or "dropout"):
    x = tf.convert_to_tensor(x, name="x")
    assert isinstance(x, tf.Tensor)
    if isinstance(keep_prob, (float, int)) and not 0 < keep_prob <= 1:
      raise ValueError("keep_prob must be a scalar tensor or a float in the "
                       "range (0, 1], got %g" % keep_prob)
    # Do nothing if we know keep_prob == 1
    if isinstance(keep_prob, (float, int)) and keep_prob == 1:
      return x
    inv_keep_prob = 1.0 / keep_prob

    if isinstance(noise_shape, (list, tuple)):
      assert len(noise_shape) == x.shape.ndims
      noise_shape = [d if isinstance(d, int) else get_shape_dim(x, i) for (i, d) in enumerate(noise_shape)]
    elif noise_shape is None:
      noise_shape = get_shape(x)
    else:
      assert isinstance(noise_shape, tf.Tensor) and noise_shape.shape.as_list() == [x.shape.ndims]
      from tensorflow.python.framework import tensor_util
      noise_shape_v = tensor_util.constant_value(noise_shape)
      if noise_shape_v is not None:
        noise_shape = [int(noise_shape_v[i]) for i in range(x.shape.ndims)]
    # If noise_shape is fully static, calculate it outside of any ctx (e.g. recurrent loop).
    # This effectively means that the mask would get reused for multiple frames, if `x` is inside a recurrent loop.
    with same_control_flow_ctx(noise_shape):
      # uniform [keep_prob, 1.0 + keep_prob)
      random_tensor = keep_prob
      random_tensor += tf_compat.v1.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
      # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
      binary_tensor = tf.floor(random_tensor)
      if apply_correction_factor:
        binary_tensor *= inv_keep_prob
    ret = x * binary_tensor
    assert isinstance(ret, tf.Tensor)
    ret.set_shape(x.get_shape())
    return ret


def layer_norm(x, gain, bias, axis, epsilon=1e-6):
  """
  Layer normalization.
  Also see :func:`openai_layer_norm`.
  Also see :func:`tensorflow.contrib.layers.layer_norm`.

  :param tf.Tensor x:
  :param tf.Tensor gain:
  :param tf.Tensor bias:
  :param int axis:
  :param float epsilon: OpenAI uses 1e-6, TF contrib uses 1e-12, pbhatia243 uses 1e-5.
  :rtype: tf.Tensor
  """
  with tf.name_scope('layer_norm'):
    ndim = x.get_shape().ndims
    if axis < 0:
      axis += ndim
      assert axis >= 0
    dim = get_shape_dim(x, axis=axis)
    if gain.get_shape().ndims == 1:
      gain = tf.reshape(gain, [dim if i == axis else 1 for i in range(ndim)], "gain_bc")
    if bias.get_shape().ndims == 1:
      bias = tf.reshape(bias, [dim if i == axis else 1 for i in range(ndim)], "bias_bc")
    m, v = tf_compat.v1.nn.moments(x, axes=[axis], keep_dims=True)
    inv = tf_compat.v1.rsqrt(v + epsilon)
    inv *= gain
    return x * inv - m * inv + bias


def openai_layer_norm(x, gain, bias, axis, epsilon=1e-6):
  """
  Layer normalization, like :func:`layer_norm`,
  but fast kernel by OpenAI (implemented as part of their blocksparse).
  To use it, init the git submodule in extern/blocksparse.

  :param tf.Tensor x:
  :param tf.Tensor gain:
  :param tf.Tensor bias:
  :param int axis:
  :param float epsilon:
  :rtype: tf.Tensor
  """
  with tf.name_scope("openai_layer_norm"):
    ndim = x.get_shape().ndims
    if axis < 0:
      axis += ndim
      assert axis >= 0
    assert axis == ndim - 1, "OpenAI kernel seems broken otherwise. see test_layer_norms."
    if gain.get_shape().ndims > 1:
      gain = tf.reshape(gain, [x.get_shape().dims[axis].value or -1], "gain_flat")
    if bias.get_shape().ndims > 1:
      bias = tf.reshape(bias, [x.get_shape().dims[axis].value or -1], "bias_flat")
    from returnn.tf.native_op import init_blocksparse
    init_blocksparse()
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    from blocksparse.norms import layer_norm
    return layer_norm(x, g=gain, b=bias, axis=axis, epsilon=epsilon)


def swapaxes(x, axis1, axis2):
  """
  Also see :func:`move_axis` or :func:`dimshuffle`.

  :param tf.Tensor x:
  :param tf.Tensor|int axis1:
  :param tf.Tensor|int axis2:
  :return: tensor with swapped axes, like numpy.swapaxes
  :rtype: tf.Tensor
  """
  with tf.name_scope("swapaxes"):
    ndim = x.get_shape().ndims
    if ndim is not None:
      if isinstance(axis1, tf.Tensor) or isinstance(axis2, tf.Tensor):
        perm = [tf.where(tf.equal(axis1, i), axis2,
                         tf.where(tf.equal(axis2, i), axis1,
                                  i))
                for i in range(ndim)]
      else:
        perm = list(range(ndim))
        axis1 = axis1 % len(perm)
        axis2 = axis2 % len(perm)
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
      perm = tf.where(tf.equal(axis1_bc, all_axes), axis2_bc,
                      tf.where(tf.equal(axis2_bc, all_axes), axis1_bc,
                               all_axes))
    return tf.transpose(x, perm=perm)


def move_axis(x, old_axis, new_axis, name="move_axis"):
  """
  Also see :func:`swapaxes` or :func:`dimshuffle`.

  :param tf.Tensor x:
  :param int old_axis: can also be negative
  :param int new_axis: can also be negative
  :param str name: name of the scope
  """
  with tf.name_scope(name):
    ndim = x.get_shape().ndims
    assert ndim is not None, "not supported currently: %r" % x
    if old_axis < 0:
      old_axis += ndim
      assert old_axis >= 0
    if new_axis < 0:
      new_axis += ndim
      assert new_axis >= 0
    if old_axis == new_axis:
      return x
    perm = list(range(ndim))
    old = perm.pop(old_axis)
    perm.insert(new_axis, old)
    return tf.transpose(x, perm)


class TensorCachedComputation:
  """
  Helper to cache some computation inside a ``tf.Tensor`` object.
  """

  def __init__(self, x, key):
    """
    :param tf.Tensor x:
    :param str|tuple[str|int|tf.Tensor] key:
    """
    self.x = x
    self.key = key

  def _get_cache_dict(self):
    """
    :rtype: dict
    """
    if not hasattr(self.x, "_RETURNN_cache"):
      self.x._RETURNN_cache = {}
    return self.x._RETURNN_cache

  def has_cache(self):
    """
    :return: whether we have stored the value already. if True, you can use :func:`get_cache`
    :rtype: bool
    """
    return self.key in self._get_cache_dict()

  def get_cache(self):
    """
    :rtype: tf.Tensor
    """
    return self._get_cache_dict()[self.key]

  def set_cache(self, value):
    """
    :param tf.Tensor value:
    """
    self._get_cache_dict()[self.key] = value


def sequence_mask(lengths, name=None, **kwargs):
  """
  Wraps around tf.sequence_mask().
  It will cache the value inside the passed object so that we don't recompute it multiple times.

  :param tf.Tensor lengths: shape (batch,)
  :param str|None name:
  :param kwargs: passed on to tf.sequence_mask
  :return: tensor mask of shape (batch,maxlen/time). default dtype is bool unless you specify something else
  :rtype: tf.Tensor
  """
  if kwargs:  # e.g. maxlen, dtype
    # Do not cache in this case, as it might be different depending on kwargs.
    return tf.sequence_mask(lengths, name=name, **kwargs)
  # Cache value if there are no other kwargs.
  cache = TensorCachedComputation(lengths, key="sequence_mask")
  if cache.has_cache():
    return cache.get_cache()
  with same_control_flow_ctx(lengths), reuse_name_scope_of_tensor(lengths):
    mask = tf.sequence_mask(lengths, name=name)
  cache.set_cache(mask)
  return mask


def sequence_mask_time_major(lengths, **kwargs):
  """
  Wraps around tf.transpose(tf.sequence_mask(), (1,0)).
  It will cache the value inside the passed object so that we don't recompute it multiple times.

  :param tf.Tensor lengths: shape (batch,)
  :param kwargs: passed on to tf.sequence_mask
  :return: mask of shape (maxlen/time,batch)
  :rtype: tf.Tensor
  """
  cache = None
  if not kwargs:
    cache = TensorCachedComputation(lengths, key="sequence_mask_time_major")
    if cache.has_cache():
      return cache.get_cache()
  mask = sequence_mask(lengths=lengths, **kwargs)  # shape (time,batch)
  with same_control_flow_ctx(mask), reuse_name_scope_of_tensor(lengths), tf.name_scope("sequence_mask_time_major"):
    mask = tf.transpose(mask, (1, 0))  # shape (batch,time)
  if cache:
    cache.set_cache(mask)
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


# noinspection PyShadowingBuiltins
def reversed(x):
  """
  Just returns x[::-1].
  It will cache the value inside the passed object so that we don't recompute it multiple times.

  :param tf.Tensor x:
  :rtype: tf.Tensor
  """
  cache_x = TensorCachedComputation(x, key="reversed_dim0")
  if cache_x.has_cache():
    return cache_x.get_cache()
  with reuse_name_scope_of_tensor(x), tf.name_scope("reversed"):
    y = x[::-1]
  cache_x.set_cache(y)
  TensorCachedComputation(y, key="reversed_dim0").set_cache(x)
  return y


def flatten_with_seq_len_mask(x, seq_lens, batch_dim_axis=None, time_dim_axis=None, time_major=None):
  """
  :param tf.Tensor x: shape (batch,...s..., time, ...s'...) or shape (time,...s...., batch, ...s'...)
  :param tf.Tensor seq_lens: shape (batch,) of int32
  :param int batch_dim_axis: index of batch_dim in x
  :param int time_dim_axis: index of time_dim in x
  :param bool time_major: whether time axis is 0 (redundant, kept for compatibility)
  :return: tensor of shape (time', ...s...s'...) where time' = sum(seq_len) <= batch*time
  :rtype: tf.Tensor
  """
  if time_major is not None:
    if time_major:
      batch_dim_axis, time_dim_axis = 1, 0
    else:
      batch_dim_axis, time_dim_axis = 0, 1
  assert batch_dim_axis is not None and time_dim_axis is not None
  assert batch_dim_axis != time_dim_axis
  with tf.name_scope("flatten_with_seq_len_mask"):
    seq_lens = check_input_ndim(seq_lens, 1)
    # If not (batch,time,...s...), transform.
    if batch_dim_axis != 0 or time_dim_axis != 1:
      dyn_axes = [batch_dim_axis, time_dim_axis]
      perm = dyn_axes + [i for i in range(len(x.shape)) if i not in dyn_axes]
      x = tf.transpose(x, perm=perm)
      batch_dim_axis = 0
      time_dim_axis = 1
    x = check_dim_equal(x, batch_dim_axis, seq_lens, batch_dim_axis, ["batch-dim does not match"])  # batch dim
    # int64? -> https://github.com/tensorflow/tensorflow/issues/6518
    # Batch and time dims have to be in front of the tensor in order to apply the mask.
    mask = sequence_mask(seq_lens, maxlen=tf.shape(x)[time_dim_axis])  # shape (batch,time)
    mask = check_input_ndim(mask, 2)
    mask = check_dim_equal(mask, 0, x, batch_dim_axis)
    mask = check_dim_equal(mask, 1, x, time_dim_axis)
    res = tf.boolean_mask(x, mask)
    res = check_input_ndim_equal_offset(res, x, -1)
    return res


def expand_dims_unbroadcast(x, axis, dim, name="expand_dims_unbroadcast"):
  """
  :param tf.Tensor|float|int x:
  :param int|tf.Tensor axis: new axis
  :param int|tf.Tensor dim: dimension for axis
  :param str name: scope name
  :return: if x is of shape (a,b,c) and axis=0, then we return (dim,a,b,c)
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    x = tf.convert_to_tensor(x)
    x = tf.expand_dims(x, axis)
    if dim is not 1:
      new_ndim = x.get_shape().ndims
      assert new_ndim is not None, "not implemented otherwise yet"
      assert isinstance(axis, int), "not implemented otherwise yet"
      if axis < 0:
        axis = new_ndim + axis
      x = tf.tile(x, [dim if (axis == i) else 1 for i in range(new_ndim)])
    return x


def expand_multiple_dims(x, axes, name="expand_multiple_dims"):
  """
  :param tf.Tensor x:
  :param list[int]|tuple[int] axes: after completion, tf.shape(y)[axis] == 1 for axis in axes
  :param str name: scope name
  :return: y where we have a new broadcast axis for each axis in axes
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    for i in sorted(axes):
      x = tf.expand_dims(x, axis=i, name="expand_axis_%i" % i)
    return x


def tile_transposed(x, axis, multiples):
  """
  Example: x with shape (D,), tf.tile(x, [N]) can be reshaped into (N,D),
  while tile_transposed(x, axis=0, multiples=N) can be reshaped into (D,N).

  :param tf.Tensor x:
  :param int axis:
  :param int|tf.Tensor multiples:
  :return: tensor with shape[axis] == x.shape[axis] * multiples
  :rtype: tf.Tensor
  """
  with tf.name_scope("tile_transposed"):
    ndim = x.get_shape().ndims
    assert ndim is not None
    assert 0 <= axis < ndim
    cache = TensorCachedComputation(x, key=("tile_transposed", axis, multiples))
    if cache.has_cache():
      return cache.get_cache()
    shape = get_shape(x)
    x = expand_dims_unbroadcast(x, axis=axis + 1, dim=multiples)  # new axis after `axis`
    y = tf.reshape(
      x,
      [shape[i] for i in range(axis)] +
      [shape[axis] * multiples] +
      [shape[i] for i in range(axis + 1, ndim)])
    cache.set_cache(y)
    return y


def constant_with_shape(x, shape, dtype=None, name="constant_with_shape"):
  """
  :param tf.Tensor|float|int|bool x: scalar
  :param list[tf.Tensor|int]|tuple[tf.Tensor|int]|tf.Tensor shape:
  :param tf.DType dtype:
  :param str name:
  :return: x of the specified shape
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    if type(x) in [int, float, bool] and type(shape) in [list, tuple] and all([type(d) == int for d in shape]):
      if dtype is None:
        dtype = {int: tf.int32, float: tf.float32, bool: tf.bool}[type(x)]
      if x in (0, 0.0, False):
        return tf.zeros(shape, dtype=dtype)
      if x in (1, 1.0, True):
        return tf.ones(shape, dtype=dtype)
    x = tf.convert_to_tensor(x, dtype=dtype)
    ones = tf.ones(shape, dtype=x.dtype)
    if x.dtype == tf.bool:
      return tf.logical_and(x, ones)
    return tf.multiply(x, ones)


def dimshuffle(x, axes, name="dimshuffle"):
  """
  Like Theanos dimshuffle.
  Combines tf.transpose, tf.expand_dims and tf.squeeze.

  :param tf.Tensor x:
  :param list[int|str]|tuple[int|str] axes:
  :param str name: scope name
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    assert all([i == "x" or isinstance(i, int) for i in axes])
    real_axes = [i for i in axes if isinstance(i, int)]
    bc_axes = [i for (i, j) in enumerate(axes) if j == "x"]
    if x.get_shape().ndims is None:
      x_shape = tf.shape(x)
      x = tf.reshape(x, [x_shape[i] for i in range(max(real_axes) + 1)])  # will have static ndims
    assert x.get_shape().ndims is not None

    # First squeeze missing axes.
    i = 0
    while i < x.get_shape().ndims:
      if i not in real_axes:
        x = tf.squeeze(x, axis=i)
        real_axes = [(j if (j < i) else (j - 1)) for j in real_axes]
      else:
        i += 1

    # Now permute.
    assert list(sorted(real_axes)) == list(range(x.get_shape().ndims))
    if real_axes != list(range(x.get_shape().ndims)):
      x = tf.transpose(x, real_axes)

    # Now add broadcast dimensions.
    if bc_axes:
      x = expand_multiple_dims(x, bc_axes)
    assert len(axes) == x.get_shape().ndims
    return x


def sparse_labels_with_seq_lens(x, seq_lens, dtype=tf.int32, collapse_repeated=False, post_filter_idx=None):
  """
  :param tf.Tensor x: shape (batch,time) -> index, some int type
  :param tf.Tensor|None seq_lens: shape (batch,) of int32|int64
  :param tf.DType|None dtype: if given, will cast the `x` values to this type. ctc_loss() wants int32
  :param bool collapse_repeated: like uniq() behavior
  :param int|list[int]|set[int]|None post_filter_idx: if given, after an optional collapse_repeated,
    will remove all those idx
  :return: SparseTensor, e.g. input for tf.nn.ctc_loss(), and seq_lens of shape (batch,)
  :rtype: (tf.SparseTensor, tf.Tensor)
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
        diffs = tf.concat(
          axis=1,
          values=[tf.ones_like(x[:, :1], dtype=tf.bool), tf.not_equal(x[:, 1:], x[:, :-1])])  # shape (batch,time)
        mask = tf.logical_and(diffs, mask)
    if post_filter_idx is not None:
      with tf.name_scope("post_filter_idx"):
        if isinstance(post_filter_idx, int):
          mask = tf.logical_and(mask, tf.not_equal(x, post_filter_idx))
        elif isinstance(post_filter_idx, (list, tuple, set)):
          for _idx in sorted(post_filter_idx):
            assert isinstance(_idx, int)
            mask = tf.logical_and(mask, tf.not_equal(x, _idx))
        else:
          raise TypeError("unexpected type post_filter_idx %r" % type(post_filter_idx))
    with tf.name_scope("flat_x"):
      flat_x = tf.boolean_mask(x, mask)  # (N, ...s...)
    with tf.name_scope("idxs"):
      if collapse_repeated or post_filter_idx is not None:
        # Recalculate mask, so that we have them all behind each other.
        seq_lens = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)  # (batch,)
        max_time = tf.reduce_max(seq_lens)
        mask = sequence_mask(seq_lens)
      time_idxs = expand_dims_unbroadcast(tf.range(max_time), 0, batch_size)  # shape (batch,time)
      flat_time_idxs = tf.boolean_mask(time_idxs, mask)  # (N,)
      batch_idxs = expand_dims_unbroadcast(tf.range(batch_size), 1, max_time)  # shape (batch,time)
      flat_batch_idxs = tf.boolean_mask(batch_idxs, mask)  # (N,)
      flat_idxs = tf.stack([flat_batch_idxs, flat_time_idxs], axis=1)  # shape (N, 2)
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
    return tf.SparseTensor(flat_idxs, flat_x, shape), seq_lens


def sparse_labels(x, seq_lens, dtype=tf.int32, collapse_repeated=False):
  """
  :param tf.Tensor x: shape (batch,time) -> index, some int type
  :param tf.Tensor|None seq_lens: shape (batch,) of int32|int64
  :param tf.DType|None dtype: if given, will cast the `x` values to this type. ctc_loss() wants int32
  :param bool collapse_repeated: like uniq() behavior
  :return: SparseTensor, e.g. input for tf.nn.ctc_loss()
  :rtype: tf.SparseTensor
  """
  y, _ = sparse_labels_with_seq_lens(x=x, seq_lens=seq_lens, dtype=dtype, collapse_repeated=collapse_repeated)
  return y


def uniq(x):
  """
  :param tf.Tensor x: 1D shape (time,) -> index, some int type
  :return: like numpy.uniq. unlike tf.unique which will never repeat entries.

  Example: uniq([0, 0, 1, 1, 0, 0]) == [0, 1, 0], tf.unique([0, 0, 1, 1, 0, 0]) == [0, 1].
  For a batched variant, see batched_uniq, or sparse_labels() with option collapse_repeated.
  """
  diffs = tf.concat(0, [tf.ones_like(x[:1]), x[1:] - x[:-1]])
  nonzero_idx = tf.where(diffs)
  x_uniq = tf.gather_nd(x, nonzero_idx)
  return x_uniq


def batched_uniq(x, seq_lens):
  """
  :param tf.Tensor x: shape (batch,time) -> index, some int type
  :param tf.Tensor|None seq_lens: shape (batch,) of int32|int64
  :return: tuple (z, new_seq_lens), where z is of shape (batch, max_new_time),
    max_new_time = max(new_seq_lens), seq_lens is of shape (batch,).
  :rtype: (tf.Tensor, tf.Tensor)
  """
  y, new_seq_lens = sparse_labels_with_seq_lens(x, seq_lens=seq_lens, collapse_repeated=True)
  z = tf_compat.v1.sparse_to_dense(sparse_indices=y.indices, sparse_values=y.values, output_shape=y.dense_shape)
  return z, new_seq_lens


def ctc_greedy_decode(logits, seq_lens, time_major):
  """
  Similar to :func:`tf.nn.ctc_greedy_decoder`,
  but simpler implementation, and should run on GPU.

  :param tf.Tensor logits: (time,batch,dim) or (batch,time,dim)
  :param tf.Tensor seq_lens: shape (batch,) of int32|int64
  :param bool time_major:
  :rtype: tf.SparseTensor
  :return: in batch-major, [batch,max_time] (like :func:`tf.nn.ctc_greedy_decoder`)
  """
  assert logits.get_shape().ndims == 3 and logits.get_shape().dims[-1].value
  dim = logits.get_shape().dims[-1].value
  assert isinstance(dim, int) and dim >= 2
  blank_idx = dim - 1
  if time_major:
    logits = tf.transpose(logits, [1, 0, 2])  # (batch,time,dim)
  greedy_labels = tf.argmax(logits, -1)  # (batch,time)
  y, _ = sparse_labels_with_seq_lens(
    greedy_labels, seq_lens=seq_lens, collapse_repeated=True, post_filter_idx=blank_idx)
  return y


def get_common_shape(values, ignore_axes=()):
  """
  Related: :func:`tf.broadcast_dynamic_shape`.
  Also see :func:`unbroadcast_to_common_shape`.

  :param list[tf.Tensor|float|int] values:
  :param list[int]|tuple[int] ignore_axes: these axes will be ignored
  :return: common shape of all values. broadcasts dims with 1. will use static dims when possible.
    Dim of axes which are in `ignore_axes` will be None.
  :rtype: list[tf.Tensor|int|None]
  """
  assert len(values) > 0
  assert all([isinstance(value, (tf.Tensor, float, int)) for value in values])
  # Filter out scalars.
  values = [value for value in values if isinstance(value, tf.Tensor)]
  assert all([value.shape.ndims is not None for value in values]), "some unknown ndim"
  values = [value for value in values if value.shape.ndims > 0]
  if not values:  # all were scalars?
    return []
  ndim = max([value.shape.ndims for value in values])
  for value in values:
    assert value.shape.ndims == ndim, "ndim does not match in values %r" % (values,)
  for axis in ignore_axes:
    assert 0 <= axis < ndim
  with tf.name_scope("common_shape"):
    common_shape = [None] * ndim  # type: typing.List[typing.Union[tf.Tensor,int,None]]
    for axis in range(ndim):
      if axis in ignore_axes:
        continue  # does not matter
      for value in values:
        static_dim = value.shape.dims[axis].value  # type: typing.Optional[int]
        if common_shape[axis] in (None, 1):
          if static_dim is not None:
            common_shape[axis] = static_dim
          else:
            common_shape[axis] = get_shape_dim(value, axis)
        if static_dim not in (None, 1):
          if isinstance(common_shape[axis], tf.Tensor):
            common_shape[axis] = static_dim
          else:  # common_shape is int
            assert isinstance(common_shape[axis], int)
            assert common_shape[axis] == static_dim, "non matching dim %r vs %r in axis %i, value %r of values %r" % (
              common_shape[axis], static_dim, axis, value, values)
    return common_shape


def unbroadcast_to_common_shape(value, common_shape, ignore_axes=(), allow_only_noop=False):
  """
  :param tf.Tensor|T value:
  :param list[tf.Tensor|int|None] common_shape: see :func:`get_common_shape`
  :param list[int]|tuple[int] ignore_axes:
  :param bool allow_only_noop: if False, and the unbroadcast is not a no-op, will raise an exception
  :return: (maybe) unbroadcasted value
  :rtype: tf.Tensor|T
  """
  if not isinstance(value, tf.Tensor):
    if isinstance(value, (float, int)) and not common_shape:
      return value
    value = tf.convert_to_tensor(value)
  ndim = value.shape.ndims
  assert ndim is not None, "value has unknown ndim: %r" % value
  if ndim == 0:
    if not common_shape:
      return value
    value = expand_multiple_dims(value, axes=list(range(len(common_shape))))
    ndim = len(common_shape)
  static_shape = value.shape.as_list()
  tile_multiples = [common_shape[_axis] if static_shape[_axis] == 1 else 1 for _axis in range(ndim)]
  for axis in ignore_axes:
    assert 0 <= axis < ndim
    tile_multiples[axis] = 1
  assert all([m is not None for m in tile_multiples]), (
    "ignore_axes %r probably missing some axis for common shape %r" % (ignore_axes, common_shape))
  if all([isinstance(m, int) and m == 1 for m in tile_multiples]):
    # We have a no-op.
    return value
  assert not allow_only_noop, "need to broadcast value %r to common shape %r with tile multiples %r" % (
    value, common_shape, tile_multiples)
  value = tf.tile(value, tile_multiples, name="unbroadcast_to_common_shape")
  return value


def concat_with_opt_broadcast(values, allow_broadcast, axis, name="concat_with_opt_broadcast"):
  """
  :param list[tf.Tensor] values: all with same ndim
  :param list[bool] allow_broadcast: same len as `values`
  :param int axis:
  :param str name:
  :return: basically tf.concat(values, axis), but we can allow broadcasting for some values
  :rtype: tf.Tensor
  """
  assert 0 < len(values) == len(allow_broadcast)
  if all([not a for a in allow_broadcast]):
    return tf.concat(values, axis=axis)
  ndim = values[0].shape.ndims
  assert ndim, "unknown ndim or scalar: %r" % (values,)
  for value in values:
    assert value.shape.ndims == ndim, "ndim does not match in values %r" % (values,)
  if axis < 0:
    axis += ndim
  assert 0 <= axis < ndim
  with tf.name_scope(name):
    common_shape = get_common_shape(values, ignore_axes=[axis])
    # Now check all, or maybe unbroadcast.
    for i in range(len(values)):
      values[i] = unbroadcast_to_common_shape(
        values[i], common_shape=common_shape, ignore_axes=[axis], allow_only_noop=not allow_broadcast[i])
    # Now do the concat.
    return tf.concat(values, axis=axis, name=name)


def matrix_triangular(shape, dtype=tf.float32, lower=False, upper=False):
  """
  :param tuple[int|tf.Tensor]|tf.Tensor shape:
  :param tf.DType dtype:
  :param bool lower:
  :param bool upper:
  :rtype: tf.Tensor
  """
  assert (lower or upper) and (not lower or not upper)
  x = tf.ones(shape, dtype=dtype)
  return tf_compat.v1.matrix_band_part(x, num_lower=-1 if lower else 0, num_upper=-1 if upper else 0)


class VariableAssigner(object):
  """
  Object helper to assign some var.
  (This is mostly obsolete now.)
  """

  def __init__(self, var):
    """
    :param tf.Variable var:
    """
    self.var = var
    assert isinstance(self.var.initializer, tf.Operation)
    assert self.var.initializer.type in ["Assign", "AssignVariableOp"]
    self.assign_op = self.var.initializer

  def assign(self, value, session):
    """
    :param numpy.ndarray|int|float|list[str] value:
    :param tf.compat.v1.Session session:
    """
    session.run(self.assign_op, feed_dict={self.assign_op.inputs[1]: value})


def _get_tf_gcc_path(bin_name):
  """
  :param str bin_name: e.g. "gcc" or "g++"
  :rtype: str
  """
  gcc_candidates = []
  tf_gcc_version = getattr(tf, "__compiler_version__", None)
  if tf_gcc_version:  # e.g. "4.8.5" or "5.4.0 20160609"
    if " " in tf_gcc_version:
      tf_gcc_version = tf_gcc_version[:tf_gcc_version.find(" ")]  # just "5.4.0"
    tf_gcc_version = tf_gcc_version.split(".")  # ["5", "4", "0"]
    for i in range(len(tf_gcc_version), 0, -1):
      gcc_candidates.append("%s-%s" % (bin_name, ".".join(tf_gcc_version[:i])))

    for gcc in gcc_candidates:
      for p in os.environ["PATH"].split(":"):
        pp = "%s/%s" % (p, gcc)
        if os.path.exists(pp):
          return pp
    # No match found (even not same major version).

    # Find all versions.
    # Maybe we can use some other version instead.
    # E.g. if we like to have GCC 5 (TF compiler version), but we have GCC 7, 8, 9 installed,
    # instead of picking the default GCC (version 9), picking version 7 might make more sense.
    # (Some nvcc versions also only support earlier GCC versions...)
    available_binaries = {}  # major-version -> path
    for p in os.environ["PATH"].split(":"):
      if os.path.isdir(p):
        for name in os.listdir(p):
          if name.startswith("%s-" % bin_name):
            version_str = name[len(bin_name) + 1:]
            version_parts = version_str.split(".")
            try:
              major_version = int(version_parts[0])
            except ValueError:
              pass  # just ignore
            else:
              pp = "%s/%s" % (p, name)
              available_binaries.setdefault(major_version, pp)
              # Prefer shorter. Just some heuristic...
              available_binaries[major_version] = min(available_binaries[major_version], pp)
    if available_binaries:
      tf_gcc_major_version = int(tf_gcc_version[0])  # e.g. 5
      assert tf_gcc_major_version not in available_binaries  # we should have found it before
      # Get closest version. This is just some heuristic...
      # (Instead of such an heuristic, we might additionally add a mapping of known supported versions...)
      version = min([(abs(v - tf_gcc_major_version), v) for v in sorted(available_binaries.keys())])[1]
      return available_binaries[version]

  # Just check for binary (without version) in PATH.
  for p in os.environ["PATH"].split(":"):
    pp = "%s/%s" % (p, bin_name)
    if os.path.exists(pp):
      return pp
  # Not found.

  # Dummy fallback.
  return bin_name


_tf_gcc_path = None


def get_tf_gcc_path():
  """
  :return: path to a GCC version which is most suitable for TF (to have correct C++ ABI)
  :rtype: str
  """
  if sys.platform == "darwin":
    return "gcc"  # better default
  global _tf_gcc_path
  if _tf_gcc_path is not None:
    return _tf_gcc_path
  _tf_gcc_path = _get_tf_gcc_path("gcc")
  return _tf_gcc_path


_tf_gpp_path = None


def get_tf_gpp_path():
  """
  :return: path to a G++ version which is most suitable for TF (to have correct C++ ABI)
  :rtype: str
  """
  if sys.platform == "darwin":
    return "g++"  # better default
  global _tf_gpp_path
  if _tf_gpp_path is not None:
    return _tf_gpp_path
  _tf_gpp_path = _get_tf_gcc_path("g++")
  return _tf_gpp_path


class CudaEnv(object):
  """
  Information about the Nvidia CUDA environment, and library.
  Also path to ``nvcc``, the CUDA compiler.
  """

  _instance = None
  verbose_find_cuda = False

  def __init__(self):
    from returnn.util.basic import to_bool
    if to_bool(os.environ.get("DISABLE_CUDA", "0")):
      self.cuda_path = None
      if self.verbose_find_cuda:
        print("CUDA disabled via env DISABLE_CUDA.")
    else:
      self.cuda_path = self._find_cuda_path()
      if self.verbose_find_cuda:
        print("CUDA path:", self.cuda_path)
    self._max_compute_capability = None

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
    from returnn.util.basic import get_ld_paths
    for p in get_ld_paths():
      pp = "%s/libcudart.so" % p
      if os.path.exists(pp):
        yield pp

  @classmethod
  def _get_lib_dir_name(cls, base_path):
    """
    :return: dir name in base path
    :rtype: str
    """
    from returnn.util.basic import is_64bit_platform, get_ld_paths
    for ld_path in get_ld_paths():
      # We also want to allow "lib/x86_64-linux-gnu" for "/usr".
      # However, this logic should not be triggered for incorrect cases.
      # E.g. base_path="/usr" would be the prefix for most LD paths.
      if ld_path.startswith(base_path + "/lib") and os.path.exists("%s/libcudart.so" % ld_path):
        return ld_path[len(base_path) + 1:]
    if is_64bit_platform():
      return "lib64"
    return "lib"

  @classmethod
  def _cuda_path_candidate_via_proc_map_libcudart(cls):
    from returnn.util.basic import find_libcudart_from_runtime
    fn = find_libcudart_from_runtime()
    if cls.verbose_find_cuda:
      print("libcudart.so found from /proc/maps:", fn)
    if not fn:
      return None
    # fn is e.g. '/usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudart.so.8.0.61',
    # or maybe '/usr/local/cuda-8.0/lib64/libcudart.so'
    p = os.path.dirname(os.path.dirname(fn))
    while not cls._check_valid_cuda_path(p):
      p = os.path.dirname(p)
      assert p not in ["", "/"], "No parent dir of %r is a valid CUDA path." % fn
    assert cls._check_valid_cuda_path(p)
    return p

  @classmethod
  def _cuda_path_candidates(cls):
    p = cls._cuda_path_candidate_via_proc_map_libcudart()
    if p:
      yield p
    for p in cls._find_nvcc_in_path():
      # Expect p == "/usr/local/cuda-8.0/bin/nvcc" or so.
      postfix = "/bin/nvcc"
      if cls.verbose_find_cuda:
        print("found cuda nvcc (wanted postfix: %r): %s" % (postfix, p))
      if not p.endswith(postfix):
        continue
      yield p[:-len(postfix)] or "/"
    for p in cls._find_lib_in_ld_path():
      # Expect p == "/usr/local/cuda-8.0/lib64/libcudart.so" or so.
      d = "/".join(p.split("/")[:-2]) or "/"  # Get "/usr/local/cuda-8.0".
      if cls.verbose_find_cuda:
        print("found cuda lib: %s (path %s)" % (p, d))
      yield d

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
    if not os.path.exists("%s/%s/libcudart.so" % (p, cls._get_lib_dir_name(p))):
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

  def is_available(self):
    """
    :rtype: bool
    """
    return bool(self.cuda_path)

  def get_max_compute_capability(self):
    """
    :return: the highest compute capability supported by nvcc, or float("inf") if not known
    :rtype: float
    """
    if self._max_compute_capability is None:
      cuda_occupancy_path = "%s/include/cuda_occupancy.h" % self.cuda_path
      if os.path.exists(cuda_occupancy_path):
        import re
        major, minor = None, 0
        for line in open(cuda_occupancy_path).read().splitlines():
          m = re.match("^#define\\s+__CUDA_OCC_(MAJOR|MINOR)__\\s+([0-9]+)$", line)
          if m:
            s, v = m.groups()
            v = int(v)
            if s == "MAJOR":
              major = v
            else:
              minor = v
        if major:
          self._max_compute_capability = float(major) + float(minor) * 0.1
    if self._max_compute_capability is None:
      self._max_compute_capability = float("inf")
    return self._max_compute_capability

  def get_compiler_opts(self):
    """
    :rtype: list[str]
    """
    return [
      "-ccbin", get_tf_gcc_path(),
      "-I", "%s/include" % self.cuda_path,
      "-L", "%s/%s" % (self.cuda_path, self._get_lib_dir_name(self.cuda_path)),
      "-x", "cu",
      "-v"]

  def get_compiler_bin(self):
    """
    :return: path
    :rtype: str
    """
    assert self.cuda_path
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


class OpCodeCompiler(NativeCodeCompiler):
  """
  Helper class to compile TF ops on-the-fly, similar to Theano.
  https://www.tensorflow.org/guide/extend/op
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/extend/adding_an_op.md
  """

  CacheDirName = "returnn_tf_cache/ops"

  def __init__(self, use_cuda_if_available=True, cuda_auto_min_compute_capability=True,
               include_paths=(), ld_flags=(), c_macro_defines=None, **kwargs):
    self._cuda_env = use_cuda_if_available and CudaEnv.get_instance()
    if use_cuda_if_available and is_gpu_available():
      # Currently we assume that if we provide CUDA code (thus set use_cuda_if_available=True),
      # that if there is a GPU available (as TF reports it),
      # we also expect that we find CUDA.
      # Otherwise you would end up with ops compiled for CPU only although they support CUDA
      # and the user expects them to run on GPU.
      assert self._with_cuda(), "OpCodeCompiler: use_cuda_if_available=True but no CUDA found"
    self._nvcc_opts = []
    if self._with_cuda() and cuda_auto_min_compute_capability:
      # Get CUDA compute capability of the current GPU device.
      min_compute_capability = get_available_gpu_min_compute_capability()
      if min_compute_capability:
        min_compute_capability = min(min_compute_capability, self._cuda_env.get_max_compute_capability())
        self._nvcc_opts += ["-arch", "compute_%i" % int(min_compute_capability * 10)]
    tf_include = tf.sysconfig.get_include()  # e.g. "...python2.7/site-packages/tensorflow/include"
    tf_include_nsync = tf_include + "/external/nsync/public"  # https://github.com/tensorflow/tensorflow/issues/2412
    include_paths = list(include_paths) + [tf_include, tf_include_nsync]
    c_macro_defines = {} if c_macro_defines is None else c_macro_defines.copy()
    # https://github.com/rwth-i6/returnn/issues/87
    # https://github.com/tensorflow/tensorflow/issues/17316
    # https://github.com/tensorflow/tensorflow/issues/22766
    c_macro_defines.setdefault("NDEBUG", 1)
    ld_flags = list(ld_flags)
    if have_min_tf_version((1, 14)):
      # https://github.com/tensorflow/tensorflow/issues/13607
      ld_flags += tf.sysconfig.get_link_flags()
    elif have_min_tf_version((1, 4)):
      ld_flags += ["-L%s" % tf.sysconfig.get_lib(), "-ltensorflow_framework"]
    use_cxx11_abi = getattr(getattr(tf, "sysconfig", tf), "CXX11_ABI_FLAG", getattr(tf, "CXX11_ABI_FLAG", False))
    super(OpCodeCompiler, self).__init__(
      include_paths=include_paths, c_macro_defines=c_macro_defines, ld_flags=ld_flags, use_cxx11_abi=use_cxx11_abi,
      **kwargs)
    self._tf_mod = None

  _relevant_info_keys = NativeCodeCompiler._relevant_info_keys + ("tf_version", "with_cuda", "cuda_path", "nvcc_opts")

  def _make_info_dict(self):
    from returnn.util.basic import describe_tensorflow_version
    d = super(OpCodeCompiler, self)._make_info_dict()
    d.update({
      "tf_version": describe_tensorflow_version(),
      "with_cuda": self._with_cuda(),
      "cuda_path": self._cuda_env.cuda_path if self._with_cuda() else None,
      "nvcc_opts": (tuple(self._cuda_env.get_compiler_opts()) + tuple(self._nvcc_opts)) if self._with_cuda() else None,
    })
    return d

  def _with_cuda(self):
    return bool(self._cuda_env and self._cuda_env.is_available())

  def _get_compiler_bin(self):
    if self._with_cuda():
      return self._cuda_env.get_compiler_bin()
    if self.is_cpp:
      return get_tf_gpp_path()
    else:
      return get_tf_gcc_path()

  def _transform_compiler_opts(self, opts):
    if self._with_cuda():
      nvcc_opts = self._cuda_env.get_compiler_opts()
      nvcc_opts += ["-DGOOGLE_CUDA=1"]
      for opt in opts:
        nvcc_opts += ["-Xcompiler", opt]
      nvcc_opts += self._nvcc_opts
      return nvcc_opts
    return super(OpCodeCompiler, self)._transform_compiler_opts(opts)

  def load_tf_module(self):
    """
    :return: module
    """
    if self._tf_mod:
      return self._tf_mod
    self._maybe_compile()
    self._tf_mod = tf.load_op_library(self._so_filename)
    return self._tf_mod


class TFNativeUtilCompiler(NativeCodeCompiler):
  """
  Helper class to compile TF utility functions on-the-fly.
  """

  CacheDirName = "returnn_tf_cache/tf_utils"

  def __init__(self, include_paths=(), ld_flags=(), c_macro_defines=None, **kwargs):
    tf_include = tf.sysconfig.get_include()  # e.g. "...python2.7/site-packages/tensorflow/include"
    tf_include_nsync = tf_include + "/external/nsync/public"  # https://github.com/tensorflow/tensorflow/issues/2412
    include_paths = list(include_paths) + [tf_include, tf_include_nsync]
    c_macro_defines = {} if c_macro_defines is None else c_macro_defines.copy()
    # https://github.com/rwth-i6/returnn/issues/87
    # https://github.com/tensorflow/tensorflow/issues/17316
    # https://github.com/tensorflow/tensorflow/issues/22766
    c_macro_defines.setdefault("NDEBUG", 1)
    ld_flags = list(ld_flags)
    if have_min_tf_version((1, 14)):
      # https://github.com/tensorflow/tensorflow/issues/13607
      ld_flags += tf.sysconfig.get_link_flags()
    elif have_min_tf_version((1, 4)):
      ld_flags += ["-L%s" % tf.sysconfig.get_lib(), "-ltensorflow_framework"]
    use_cxx11_abi = getattr(getattr(tf, "sysconfig", tf), "CXX11_ABI_FLAG", getattr(tf, "CXX11_ABI_FLAG", False))
    super(TFNativeUtilCompiler, self).__init__(
      include_paths=include_paths, c_macro_defines=c_macro_defines, ld_flags=ld_flags, use_cxx11_abi=use_cxx11_abi,
      **kwargs)

  _relevant_info_keys = NativeCodeCompiler._relevant_info_keys + ("tf_version",)

  def _make_info_dict(self):
    from returnn.util.basic import describe_tensorflow_version
    d = super(TFNativeUtilCompiler, self)._make_info_dict()
    # noinspection PyUnresolvedReferences
    d.update({"tf_version": describe_tensorflow_version()})
    return d


def make_var_tuple(v):
  """
  :param tf.Tensor|list[tf.Tensor]|tuple[tf.Tensor] v:
  :return: tuple of tensors
  :rtype: tuple[tf.Tensor]
  """
  if isinstance(v, (int, float, tf.Tensor, tf.Operation)):
    return v,
  if isinstance(v, list):
    return tuple(v)
  assert isinstance(v, tuple)
  return v


def add_scaled_noise_to_gradients(grads_and_vars, gradient_noise_scale, sparse_grads=False):
  """
  Adds scaled noise from a 0-mean normal distribution to gradients.
  Adapted from tf.contrib.layers.optimizers.

  :param list[(tf.Tensor|tf.IndexedSlices, tf.Variable)] grads_and_vars:
  :param float gradient_noise_scale: used as stddev for tf.truncated_normal().
  :param bool sparse_grads: for sparse gradients (tf.IndexedSlices), it will only add the noise to the indexed values.
    Seems broken in some cases? Needs debugging.
  :return: adapted grads_and_vars
  :rtype: list[(tf.Tensor|tf.IndexedSlices, tf.Variable)]
  """
  gradients, variables = zip(*grads_and_vars)
  noisy_gradients = []
  for gradient in gradients:
    if gradient is None:
      noisy_gradients.append(None)
      continue
    name = get_base_name(gradient)
    with reuse_name_scope_of_tensor(gradient):
      if isinstance(gradient, tf.IndexedSlices):
        if sparse_grads:
          gradient_values = gradient.values
          gradient_shape = gradient.values.get_shape()
        else:
          gradient_values = gradient
          gradient_shape = gradient.dense_shape
      else:
        assert isinstance(gradient, tf.Tensor)
        gradient_values = gradient
        gradient_shape = gradient_values.get_shape()
      if isinstance(gradient_shape, tf.TensorShape) and not gradient_shape.is_fully_defined():
        gradient_shape = tf.shape(gradient_values)
      noise = tf_compat.v1.truncated_normal(
        gradient_shape, stddev=gradient_noise_scale, name="%s_grad_noise" % name, seed=get_random_seed())
      gradient_values = tf.add(gradient_values, noise, name="%s_add_grad_noise" % name)
      if sparse_grads and isinstance(gradient, tf.IndexedSlices):
        gradient = tf.IndexedSlices(values=gradient_values, indices=gradient.indices, dense_shape=gradient.dense_shape)
      else:
        gradient = gradient_values
      noisy_gradients.append(gradient)
  return list(zip(noisy_gradients, variables))


class CustomGradient(object):
  """
  Utility functions to specify a custom gradient for a given function,
  which will be wrapped around via TF :func:`Defun`.

  Also see :class:`FlipGradientBuilder`.
  """

  def __init__(self):
    from weakref import ref
    self.num_calls = 0
    self.registered_ops_graph = ref(NotSpecified)
    self.registered_ops = {}  # (op,grad_op) -> decorated func

  def register(self, input_types, op, grad_op, name=None):
    """
    :param list[tf.DType]|tuple[tf.DType] input_types:
    :param ((tf.Tensor) -> tf.Tensor)|T op:
    :param (tf.Operation, tf.Tensor) -> tuple[tf.Tensor]|tf.Tensor grad_op: args are (op, out_grad)
      and it must return in_grad
    :param str name: optional func_name
    :return: op
    :rtype: ((tf.Tensor) -> tf.Tensor)|T
    """
    graph = tf_compat.v1.get_default_graph()
    assert isinstance(graph, tf.Graph)
    if graph is not self.registered_ops_graph():
      self.registered_ops.clear()
      from weakref import ref
      self.registered_ops_graph = ref(graph)
    cache_key = (op, grad_op)
    if cache_key in self.registered_ops:
      return self.registered_ops[cache_key]
    from tensorflow.python.framework import function
    op_with_new_grad = function.Defun(*input_types, python_grad_func=grad_op, func_name=name)(op)
    # We need to add one instance of the new op to the graph now because of:
    # https://github.com/tensorflow/tensorflow/issues/6804
    # In case this is done too late, which is if there was already a previous session.run call,
    # you might get an exception like this:
    # NotFoundError: Op type not registered 'generic_loss_and_error_signal'
    call = op_with_new_grad(*[tf_compat.v1.placeholder(dtype) for dtype in input_types])
    assert isinstance(call, tf.Tensor)
    assert call.graph is graph
    self.registered_ops[cache_key] = op_with_new_grad
    return op_with_new_grad

  # noinspection PyUnusedLocal
  @classmethod
  def _generic_loss_and_error_signal(cls, loss, x, grad_x):
    """
    :param tf.Tensor loss:
    :param tf.Tensor x:
    :param tf.Tensor grad_x:
    :return: just loss
    :rtype: tf.Tensor
    """
    return loss

  @classmethod
  def _generic_loss_and_error_signal_grad(cls, op, grad_loss):
    """
    :param tf.Operation op:
    :param tf.Tensor grad_loss: grad for loss
    :return: grad for op.outputs, only defined for op input x
    :rtype: (None, tf.Tensor, None)
    """
    loss, x, grad_x = op.inputs
    return None, grad_loss * grad_x, None

  def register_generic_loss_and_error_signal(self):
    """
    If you want to use :func:`generic_loss_and_error_signal` at some point,
    call this as early as possible, because of https://github.com/tensorflow/tensorflow/issues/6804.
    """
    return self.register(
      input_types=[tf.float32, tf.float32, tf.float32],
      op=self._generic_loss_and_error_signal,
      grad_op=self._generic_loss_and_error_signal_grad,
      name="generic_loss_and_error_signal")

  def generic_loss_and_error_signal(self, loss, x, grad_x):
    """
    Wrapper around self.register().
    Expects that loss = loss(x), and grad_x = \\partial loss / \\partial x.

    :param tf.Tensor loss:
    :param tf.Tensor x:
    :param tf.Tensor grad_x:
    :return: loss but with the gradient for x
    :rtype: tf.Tensor
    """
    loss = tf.convert_to_tensor(loss)
    x = tf.convert_to_tensor(x)
    grad_x = tf.convert_to_tensor(grad_x)
    x.set_shape(grad_x.get_shape())
    grad_x.set_shape(x.get_shape())
    generic_loss_and_error_signal = self.register_generic_loss_and_error_signal()
    loss_out = generic_loss_and_error_signal(loss, x, grad_x)  # noqa
    loss_out.set_shape(loss.get_shape())
    return loss_out


custom_gradient = CustomGradient()


class MetaLosses(object):
  """
  This provides a way to use an alternative gradient,
  or to use the original gradient (error signal) and do something with it.
  You can then define an additional (meta) loss using this.

  This implements synthetic gradients, see :func:`synthetic_gradient`.
  """

  class LossInfo:
    """
    Covers loss and other info.
    """

    def __init__(self, value, scale, norm_factor, name, source):
      """
      :param tf.Tensor value:
      :param float scale:
      :param tf.Tensor norm_factor:
      :param str name:
      :param object source: e.g. layer
      """
      self.value = value
      self.scale = scale
      self.norm_factor = norm_factor
      self.name = name
      self.source = source

  class Scope(object):
    """
    Defines the scope for a synthetic gradient.
    Create this object via :func:`MetaLosses.enter_gradient_scope`.
    Any meta-losses will be collected here via :func:`register_loss`.
    """

    def __init__(self):
      self.losses = []  # type: typing.List[MetaLosses.LossInfo]

    def register_loss(self, loss):
      """
      :param MetaLosses.LossInfo loss:
      """
      self.losses.append(loss)

    def exit(self):
      """
      Exit the scope.
      """
      assert MetaLosses.scope_ctx.scope is self
      MetaLosses.scope_ctx.scope = None

    def losses_as_fetch_dict(self):
      """
      :rtype: dict[str,tf.Tensor]
      """
      from collections import OrderedDict
      d = OrderedDict()
      for loss in self.losses:
        # Note: This is somewhat specific to the way we use it in TFEngine.
        d["cost:%s" % loss.name] = loss.value
        d["loss_norm_factor:%s" % loss.name] = loss.norm_factor
      return d

    def summed_loss_for_optimization(self):
      """
      :rtype: tf.Tensor
      """
      return tf.add_n([loss.value * loss.scale for loss in self.losses])

  class ScopeCtxThreadLocal(threading.local):
    """
    Thread local.
    """
    scope = None  # type: typing.Optional[MetaLosses.Scope]

  scope_ctx = ScopeCtxThreadLocal()

  @classmethod
  def enter_gradient_scope(cls):
    """
    :rtype: MetaLosses.Scope
    """
    assert not cls.scope_ctx.scope
    cls.scope_ctx.scope = cls.Scope()
    return cls.scope_ctx.scope

  @classmethod
  def exit_gradient_scope(cls):
    """
    Exit gradient scope.
    """
    cls.scope_ctx.scope.exit()

  # noinspection PyUnusedLocal
  @classmethod
  def _identity_ignore_second_fwd(cls, x, dummy):
    """
    :param tf.Tensor x:
    :param tf.Tensor dummy:
    :return: x
    :rtype: tf.Tensor
    """
    return x

  @classmethod
  def _synthetic_gradient_bwd(cls, op, grad_out):
    """
    :param tf.Operation op:
    :param tf.Tensor grad_out:
    :return: grad for x
    :rtype: (tf.Tensor,)
    """
    x, synthetic_grad_x = op.inputs
    if cls.scope_ctx.scope:
      with tf.name_scope("grad_prediction_loss"):
        grad_prediction_loss = tf.reduce_mean(tf.square(synthetic_grad_x - tf.stop_gradient(grad_out)))
        tf_compat.v1.summary.scalar("loss", grad_prediction_loss)
      # noinspection PyProtectedMember
      loss_info = op._RETURNN_loss_info
      cls.scope_ctx.scope.register_loss(MetaLosses.LossInfo(value=grad_prediction_loss, **loss_info))
    return synthetic_grad_x, None

  @classmethod
  def synthetic_gradient(cls, x, synthetic_grad_x, loss_scale=1.0, loss_name=None, loss_source=None):
    """
    Decoupled Neural Interfaces using Synthetic Gradients, https://arxiv.org/abs/1608.05343

    :param tf.Tensor x:
    :param tf.Tensor synthetic_grad_x:
    :param float loss_scale:
    :param str|None loss_name:
    :param object|None loss_source:
    :return: x, where the gradient is overwritten by synthetic_grad_x, and when calculated,
      the gradient prediction loss will be added to ``cls.scope``.
    :rtype: tf.Tensor
    """
    op = custom_gradient.register(
      [tf.float32, tf.float32],
      op=cls._identity_ignore_second_fwd,
      grad_op=cls._synthetic_gradient_bwd,
      name="synthetic_gradient")
    y = op(x, synthetic_grad_x)  # noqa
    y.op._RETURNN_loss_info = {
      "name": loss_name, "source": loss_source, "scale": loss_scale, "norm_factor": tf.size(x)}
    y.set_shape(x.get_shape())
    return y

  # noinspection PyUnusedLocal
  @classmethod
  def _tikhonov_gradient_bwd(cls, op, grad_out):
    """
    :param tf.Operation op:
    :param tf.Tensor grad_out:
    :return: grad for x
    :rtype: (tf.Tensor,)
    """
    if cls.scope_ctx.scope:
      with tf.name_scope("tikhonov_regularization_loss"):
        loss = tf.nn.l2_loss(grad_out)
        tf_compat.v1.summary.scalar("loss", loss)
      # noinspection PyProtectedMember
      loss_info = op._RETURNN_loss_info
      cls.scope_ctx.scope.register_loss(MetaLosses.LossInfo(value=loss, **loss_info))
    return grad_out, tf.constant(0.0)

  @classmethod
  def tikhonov_regularized(cls, x, dummy, loss_scale=1., loss_name=None, loss_source=None):
    """
    :param tf.Tensor x:
    :param tf.Tensor|tf.Variable dummy: scalar. can be used to enforce getting a gradient
    :param float loss_scale:
    :param str|None loss_name:
    :param object|None loss_source:
    :return: identity(x), where we add a Tikhonov regularization
    :rtype: tf.Tensor
    """
    op = custom_gradient.register(
      [tf.float32, tf.float32],
      op=cls._identity_ignore_second_fwd,
      grad_op=cls._tikhonov_gradient_bwd,
      name="tikhonov_regularized")
    y = op(x, dummy)  # noqa
    y.op._RETURNN_loss_info = {
      "name": loss_name, "source": loss_source, "scale": loss_scale, "norm_factor": tf.size(x)}
    y.set_shape(x.get_shape())
    return y


def filter_grad(x, threshold, axis):
  """
  :param tf.Tensor x:
  :param float threshold: all grads going through `x` which max(grad**2) is over the threshold are removed
  :param int|list[int] axis: max(grad**2) will be reduced over this axis
  :return: identity(x) with custom gradient
  :rtype: tf.Tensor
  """

  # noinspection PyShadowingNames
  def grad_op(op, out_grad):
    """
    :param tf.Operation op:
    :param tf.Tensor out_grad:
    :rtype: tf.Tensor
    """
    with tf.name_scope("filter_grad__grad_op"):
      assert isinstance(op, tf.Operation)
      assert isinstance(out_grad, tf.Tensor)
      out_grad.set_shape(op.inputs[0].get_shape())
      keep_filter = tf.less(tf.reduce_max(out_grad ** 2, axis=axis, keepdims=True), threshold)
      # keep_filter must be the same shape as out_grad.
      keep_filter = tf.logical_and(keep_filter, tf.ones_like(out_grad, dtype=tf.bool))
      out_grad = tf.where(keep_filter, out_grad, tf.zeros_like(out_grad))
      return out_grad

  with tf.name_scope("filter_grad"):
    op = custom_gradient.register([x.dtype], op=identity, grad_op=grad_op)
    y = op(x)
    y.set_shape(x.get_shape())
    return y


def _indexed_slices_repr(x):
  """
  :param tf.IndexedSlices x:
  :rtype: str
  """
  from tensorflow.python.framework import tensor_util
  dense_shape = tensor_util.constant_value_as_shape(x.dense_shape)
  return "<tf.IndexedSlices %r dense_shape=%r dtype=%r>" % (x.name, dense_shape, x.dtype)


def _op_repr(x):
  """
  :param tf.Operation x:
  :rtype: str
  """
  extra = ""
  if x.type == "Const":
    from tensorflow.python.framework import tensor_util
    extra += " value=%s" % (tensor_util.MakeNdarray(x.get_attr("value")),)
  return "<tf.Operation %r type=%s%s>" % (x.name, x.type, extra)


def _var_repr(x):
  """
  :param tf.Variable x:
  :rtype: str
  """
  return "<tf.Variable %r shape=%s initial_value=%r>" % (x.op.name, x.shape, x.initial_value)


def _tensorarray_repr(x):
  """
  :param tf.TensorArray x:
  :rtype: str
  """
  if x.handle is not None:
    op = x.handle.op
    assert isinstance(op, tf.Operation)
    return "<tf.TensorArray %r>" % op.name
  if x.flow is not None:
    op = x.flow.op
    assert isinstance(op, tf.Operation)
    return "<tf.TensorArray %r>" % op.name
  # noinspection PyProtectedMember
  return "<tf.TensorArray %r>" % x._implementation


def _variablescope_repr(x):
  """
  :param tf.compat.v1.VariableScope x:
  :rtype: str
  """
  return "<tf.compat.v1.VariableScope %r>" % x.name


def _saveable_repr(x):
  """
  :param tensorflow.python.training.saver.BaseSaverBuilder.SaveableObject x:
  :rtype: str
  """
  return "<tf..%s op=%r, specs=%r, name=%r>" % (x.__class__.__name__, x.op, x.specs, x.name)


def _savespec_repr(x):
  """
  :param tensorflow.python.training.saver.BaseSaverBuilder.SaveSpec x:
  :rtype: str
  """
  return "<tf..%s tensor=%r, slice_spec=%r, name=%r>" % (x.__class__.__name__, x.tensor, x.slice_spec, x.name)


def debug_register_better_repr():
  """
  Some types don't have good __repr__ implementations by default (for the current TF version).
  For debugging, it can be helpful to give some more info.
  This monkey-patches clazz.__repr__ of some TF classes.
  """
  from tensorflow.python.training import saver

  for cl, f in [
        (tf.IndexedSlices, _indexed_slices_repr),
        (tf.Operation, _op_repr),
        (tf.Variable, _var_repr),
        (tf.TensorArray, _tensorarray_repr),
        (tf_compat.v1.VariableScope, _variablescope_repr),
        (saver.BaseSaverBuilder.SaveableObject, _saveable_repr),
        (saver.BaseSaverBuilder.SaveSpec, _savespec_repr)]:
    setattr(cl, "__repr__", f)


def cond(pred, fn1, fn2, name=None):
  """
  This is a wrapper around tf.control_flow_ops.cond().
  This will be a branched execution, i.e. either fn1() or fn2() will be executed,
  or at least the resulting graph will be evaluated.
  If pred can is constant at the call, only the corresponding fn will be called.
  This is similar to the TF internal _smart_cond().
  And similar to tf.contrib.framework.smart_cond.

  :param tf.Tensor|bool pred:
  :param ()->(tf.Tensor|list[tf.Tensor]|T) fn1:
  :param ()->(tf.Tensor|list[tf.Tensor]|T) fn2:
  :param str name:
  :return: fn1() if pred else fn2()
  :rtype: tf.Tensor|list[tf.Tensor]|T
  """
  if not callable(fn1):
    raise TypeError("fn1 must be callable.")
  if not callable(fn2):
    raise TypeError("fn2 must be callable.")
  if pred is True:
    return fn1()
  if pred is False:
    return fn2()
  from tensorflow.python.framework import tensor_util
  pred_const = tensor_util.constant_value(pred)
  if pred_const is not None:
    if pred_const:
      return fn1()
    else:
      return fn2()
  from tensorflow.python.ops import control_flow_ops
  return control_flow_ops.cond(pred, fn1, fn2, name=name)


def single_strided_slice(x, axis, begin=None, end=None, step=None):
  """
  :param tf.Tensor x:
  :param int|tf.Tensor axis:
  :param int|tf.Tensor|None begin:
  :param int|tf.Tensor|None end:
  :param int|tf.Tensor|None step:
  :return: e.g. if axis == 0, returns x[begin:end:step], if axis == 1, returns x[:, begin:end:step], etc.
  :rtype: tf.Tensor
  """
  with tf.name_scope("single_strided_slice"):
    if isinstance(axis, int):
      if axis < 0 and x.get_shape().ndims is not None:
        axis %= x.get_shape().ndims
        assert axis >= 0
      if axis >= 0:
        return x[(slice(None),) * axis + (slice(begin, end, step),)]
    else:
      assert isinstance(axis, tf.Tensor)
    axis = axis % tf.rank(x)
    shape = tf.shape(x)
    if begin is None:
      begin = 0
    if end is None:
      end = shape[axis]
    begins = tf.concat([tf.zeros((axis,), tf.int32), (begin,)], axis=0)
    ends = tf.concat([shape[:axis], (end,)], axis=0)
    if step is not None:
      strides = tf.concat([tf.ones((axis,), tf.int32), (step,)], axis=0)
    else:
      strides = None
    return tf.strided_slice(x, begin=begins, end=ends, strides=strides)


def circular_pad(x, paddings, axes=None):
  """
  :param tf.Tensor x: shape (..., height, width)
  :param int|((int,int), (int,int))|tf.Tensor paddings: how much to add ((top,bottom),(left,right))
  :param None|tf.Tensor|(tf.Tensor|int,tf.Tensor|int) axes:
  :return: tensor with shape (..., top + height + bottom, left + width + right)
  :rtype: tf.Tensor
  """
  with tf.name_scope("circular_pad"):
    ndim = x.get_shape().ndims
    assert ndim is not None
    shape = tf.shape(x)
    if axes is None:
      axis_height = ndim - 2
      axis_width = ndim - 1
    elif isinstance(axes, tf.Tensor):
      axes = check_input_ndim(axes, 1)
      axes = check_input_dim(axes, 0, 2)
      axis_height, axis_width = axes[0], axes[1]
    else:
      axis_height, axis_width = axes
    height, width = shape[axis_height], shape[axis_width]
    if isinstance(paddings, tf.Tensor):
      paddings = check_input_ndim(paddings, 2)
      paddings = check_input_dim(paddings, 0, 2)
      paddings = check_input_dim(paddings, 1, 2)
      top, bottom = paddings[0, 0], paddings[0, 1]
      left, right = paddings[1, 0], paddings[1, 1]
    elif isinstance(paddings, int):
      top = bottom = left = right = paddings
    else:
      assert isinstance(paddings, (list, tuple))
      (top, bottom), (left, right) = paddings
    left_x = single_strided_slice(x, begin=width - left, axis=axis_width)
    right_x = single_strided_slice(x, end=right, axis=axis_width)
    left_right_and_x = tf.concat([left_x, x, right_x], axis=axis_width)  # shape (..., height, left + width + right)
    top_x = single_strided_slice(left_right_and_x, begin=height - top, axis=axis_height)
    bottom_x = single_strided_slice(left_right_and_x, end=bottom, axis=axis_height)
    all_combined_x = tf.concat([top_x, left_right_and_x, bottom_x], axis=axis_height)  # final shape
    assert isinstance(all_combined_x, tf.Tensor)
    return all_combined_x


def spatial_smoothing_energy(x, dim, use_circular_conv=True):
  """
  :param tf.Tensor x: shape (..., dim)
  :param int dim: last dimension of x
  :param bool use_circular_conv: whether to use circular convolution, via circular_pad
  :rtype: tf.Tensor
  :return: energy of shape (...)

  Via: Achieving Human Parity in Conversational Speech Recognition, Microsoft, 2017 (https://arxiv.org/abs/1610.05256).
  Interpret the last dimension as 2D (w, h) and apply some high-pass filter on it.
  """
  import math
  with tf.name_scope("spatial_smoothing_energy"):
    x = check_input_dim(x, -1, dim)
    shape = get_shape(x)
    w = int(math.sqrt(dim))
    while dim % w > 0:
      w -= 1
      assert w > 0
    h = dim // w
    assert w * h == dim
    assert w >= 3 and h >= 3, "too small"
    # input shape: [batch, in_height=h, in_width=w, in_channels=1]
    x = tf.reshape(x, [-1, h, w, 1])
    if use_circular_conv:
      x = circular_pad(x, paddings=1, axes=(1, 2))  # [batch, h+2, w+2, in_channels=1]
    # filter shape: [filter_height, filter_width, in_channels=1, out_channels=1]
    # noinspection PyShadowingBuiltins
    filter = tf.reshape(tf.constant(
      [[-0.125, -0.125, -0.125],
       [-0.125, 1.0, -0.125],
       [-0.125, -0.125, -0.125]]), [3, 3, 1, 1])
    # out shape: [batch, out_height, out_width, out_channels=1]
    out = tf.nn.conv2d(x, filter=filter, strides=[1, 1, 1, 1], padding="VALID")
    out = tf.reshape(out, shape[:-1] + [-1])  # (..., out_height*out_width)
    # Note: Square all the filter values.
    return tf.reduce_sum(out ** 2, axis=-1)


def nan_to_num(x, nan_num=0, inf_num=1e30):
  """
  Like numpy.nan_to_num().

  :param tf.Tensor|tf.IndexedSlices x:
  :param float|tf.Tensor nan_num:
  :param float|tf.Tensor inf_num:
  :return: x with replaced nan and inf
  """
  if isinstance(x, tf.IndexedSlices):
    return tf.IndexedSlices(values=nan_to_num(x.values), indices=x.indices, dense_shape=x.dense_shape)
  with tf.name_scope("nan_to_num"):
    nan_num = tf.convert_to_tensor(nan_num, dtype=x.dtype)
    inf_num = tf.convert_to_tensor(inf_num, dtype=x.dtype)
    x = where_bc(tf_compat.v1.is_nan(x), nan_num, x)
    x = where_bc(tf.logical_and(tf_compat.v1.is_inf(x), tf.greater(x, 0)), inf_num, x)
    x = where_bc(tf.logical_and(tf_compat.v1.is_inf(x), tf.less(x, 0)), -inf_num, x)
    return x


def where_bc(condition, x, y, name="where_bc"):
  """
  This is basically :func:`tf.where` but with additional broadcasting support.
  We explicitly require that the ndims match (or x, y can also be scalars).
  See also :func:`get_common_shape` and :func:`unbroadcast_to_common_shape`.

  https://github.com/tensorflow/tensorflow/issues/3945
  https://github.com/tensorflow/tensorflow/issues/9284

  :param tf.Tensor condition:
  :param tf.Tensor|float|int x:
  :param tf.Tensor|float|int y:
  :param str name:
  :return: basically tf.where(condition, x, y)
  :rtype: tf.Tensor
  """
  if tf_compat.v2:
    # where_v2 supports broadcasting. But we might still need to extend dims.
    # Note that the extend dims is on the opposite side as it would be common (in all other broadcasting ops).
    # However, this matches the old tf.compat.v1.where behavior.
    # (Actually the doc of this where_bc says we do not allow this anyway...? We should check where this is used...)
    condition = tf.convert_to_tensor(condition)
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)
    ndims = max(condition.get_shape().ndims, x.get_shape().ndims, y.get_shape().ndims)
    if x.get_shape().ndims < ndims:
      x = expand_multiple_dims(x, [-1] * (ndims - x.get_shape().ndims))
    if y.get_shape().ndims < ndims:
      y = expand_multiple_dims(y, [-1] * (ndims - y.get_shape().ndims))
    if condition.get_shape().ndims < ndims:
      condition = expand_multiple_dims(condition, [-1] * (ndims - condition.get_shape().ndims))
    return tf_compat.v2.where(condition=condition, x=x, y=y, name=name)
  with tf.name_scope(name):
    common_shape = get_common_shape([condition, x, y])
    condition = unbroadcast_to_common_shape(condition, common_shape=common_shape)
    x = unbroadcast_to_common_shape(x, common_shape=common_shape)
    y = unbroadcast_to_common_shape(y, common_shape=common_shape)
    return tf.where(condition, x, y)


def identity_op_nested(x, name="identity"):
  """
  :param tf.Tensor|list[tf.Tensor]|dict[str,tf.Tensor] x:
  :param str name:
  :rtype tf.Tensor|list[tf.Tensor]|dict[str,tf.Tensor]
  """
  if isinstance(x, dict):
    return {k: identity_op_nested(x[k], name="%s_%s" % (name, k)) for k in x}
  if isinstance(x, (list, tuple)):
    from returnn.util.basic import is_namedtuple
    if is_namedtuple(type(x)):
      return type(x)(*[identity_op_nested(x[i], name="%s_%i" % (name, i)) for i in range(len(x))])
    return [identity_op_nested(x[i], name="%s_%i" % (name, i)) for i in range(len(x))]
  if isinstance(x, tf.TensorArray):
    return x  # could be nicer, but good enough for now...
  assert isinstance(x, tf.Tensor)
  return tf.identity(x, name=name)


def nd_indices(indices, batch_axis=0, indices_batch_major=None):
  """
  :param tf.Tensor indices: e.g. (batch, ...) -> index (or (..., batch, ...) -> index)
  :param int batch_axis: of the indices tensor itself
  :param bool|None indices_batch_major: of the resulting 2-tuple,
    whether it represents (batch_idx, index) or (index, batch_idx). default is like batch_axis
  :return: extended indices with batch-idx which can be used for tf.gather_nd,
    i.e. in the example of shape (batch, ..., 2) where the 2-tuple represents (batch_idx, index) or (index, batch_idx).
    the shape[:-1] is exactly the same as the indices shape.
  :rtype: tf.Tensor
  """
  assert indices.get_shape().ndims >= 1
  assert batch_axis < indices.get_shape().ndims
  if indices_batch_major is None:
    assert batch_axis in [0, 1]
    indices_batch_major = batch_axis == 0
  with tf.name_scope("nd_indices"):
    batches_idxs = tf.range(tf.shape(indices)[batch_axis], name="batches_idxs")  # (batch,)
    batches_idxs = tf.cast(batches_idxs, dtype=indices.dtype)
    for axis in range(indices.get_shape().ndims):
      if axis == batch_axis:
        continue
      batches_idxs = expand_dims_unbroadcast(batches_idxs, axis=axis, dim=tf.shape(indices)[axis],
                                             name="batches_idxs_bc")  # (batch, ...)
    batches_idxs.set_shape(indices.get_shape())
    if indices_batch_major:
      idxs_exp = tf.stack([batches_idxs, indices], axis=-1,
                          name="idxs_exp")  # (batch,...,2), where the 2 stands for (batch_idx, index)
    else:
      idxs_exp = tf.stack([indices, batches_idxs], axis=-1,
                          name="idxs_exp")  # (...,2), where the 2 stands for (index, batch_idx)
    return idxs_exp


def stop_all_event_writer_threads():
  """
  Iterates through all running threads, and stops those which are TF event logger threads.
  See :func:`stop_event_writer_thread`.
  """
  import threading
  # noinspection PyProtectedMember
  from tensorflow.python.summary.writer.event_file_writer import _EventLoggerThread

  for thread in threading.enumerate():
    if isinstance(thread, _EventLoggerThread):
      stop_event_writer_thread(thread)


def stop_event_writer_thread(event_writer):
  """
  There is a bug in TensorFlow (at least 1.1.0) (https://github.com/tensorflow/tensorflow/issues/4820)
  that the event writer thread is never stopped.
  This will try to stop it. Only do it if you don't use the event writer anymore.

  :param tf.compat.v1.summary.FileWriter|tensorflow.python.summary.writer.event_file_writer.EventFileWriter|tensorflow.python.summary.writer.event_file_writer._EventLoggerThread event_writer:  # nopep8
  """
  try:
    from tensorflow.python.summary.writer.event_file_writer import CloseableQueue
  except ImportError:
    pass  # earlier TF versions. go on with code below
  else:
    # If TF has the CloseableQueue, this should have been fixed on the TF side already.
    return
  # noinspection PyProtectedMember
  from tensorflow.python.summary.writer.event_file_writer import EventFileWriter, _EventLoggerThread
  if isinstance(event_writer, tf_compat.v1.summary.FileWriter):
    event_writer = event_writer.event_writer
  if isinstance(event_writer, _EventLoggerThread):
    worker = event_writer
  else:
    assert isinstance(event_writer, EventFileWriter)
    # noinspection PyProtectedMember
    worker = event_writer._worker
    if not worker:  # maybe fixed already?
      return
  del event_writer

  # This solution is very ugly and dependent on TF internal code.
  class DummyStopThread:
    """
    Stub for EventFileWriter.
    """

    # noinspection PyPep8Naming
    @classmethod
    def WriteEvent(cls, *args, **kwargs):
      """
      Stub for EventFileWriter.WriteEvent.

      :param args:
      :param kwargs:
      :return: nothing, raises SystemExit
      """
      raise SystemExit  # stop the thread

  # noinspection PyProtectedMember
  assert isinstance(worker, _EventLoggerThread)
  worker._ev_writer = DummyStopThread
  # noinspection PyProtectedMember
  worker._queue.put(None)
  worker.join()


def optional_add(*args):
  """
  :param list[tf.Tensor|None]|int|float|tf.Tensor args:
  :rtype: tf.Tensor|int|float|None
  :return: sums all non-None values, or returns None if there are none
  """
  y = None
  for v in args:
    if v is not None:
      if y is None or (isinstance(y, (int, float)) and y == 0):
        y = v
      elif not (isinstance(v, (int, float)) and v == 0):
        y = y + v
  return y


def optional_mul(*args):
  """
  :param tf.Tensor|None|int|float args:
  :rtype: tf.Tensor|int|float|None
  :return: sums all non-None values, or returns None if there are none
  """
  y = None
  for v in args:
    if v is not None:
      if isinstance(v, (int, float)) and v == 0:
        return v
      if y is None or (isinstance(y, (int, float)) and y == 1):
        y = v
      elif not (isinstance(v, (int, float)) and v == 1):
        y = y * v
  return y


def opt_logical_and(*args):
  """
  :param tf.Tensor|bool args:
  :return: basically logical_and(*args), but leaves out all constants
  :rtype: tf.Tensor|bool
  """
  res = True
  for v in args:
    if v is True:
      continue
    if v is False:
      return False
    if res is True:
      res = v
    else:
      res = tf.logical_and(res, v)
  return res


def windowed_nd(source, window_size, window_left=None, window_right=None,
                padding="same", time_axis=1, new_window_axis=2, stride=1):
  """
  Constructs a new "window" axis which is a moving input over the time-axis.
  If you want to take out a window, i.e. a slice, see :func:`slice_nd`.

  :param tf.Tensor source: N-D tensor of shape (..., n_time, ...)
  :param int|tf.Tensor window_size: window size
  :param int|tf.Tensor|None window_left:
  :param int|tf.Tensor|None window_right:
  :param str padding: "same" or "valid"
  :param int time_axis:
  :param int new_window_axis:
  :param int stride: return only each Nth windwow
  :return: tensor of shape (..., n_time, ..., window, ...)
  :rtype: tf.Tensor
  """
  with tf.name_scope("windowed_batch"):
    if time_axis != 0:
      source = move_axis(source, time_axis, 0)  # (n_time,...)
    source_shape = tf.shape(source)
    n_time = source_shape[0]
    if padding == "same":
      n_out_time = n_time
      if window_right is None:
        if window_left is not None:
          window_right = window_size - window_left - 1
        else:
          window_right = window_size // 2
      if window_left is None:
        window_left = window_size - window_right - 1
      else:
        if isinstance(window_size, int) and isinstance(window_left, int) and isinstance(window_right, int):
          assert window_size == window_left + window_right + 1
        else:
          with tf.control_dependencies([tf_compat.v1.assert_equal(
                window_size, window_left + window_right + 1,
                data=["window != w_left + w_right + 1.", window_size, " ", window_left, " ", window_right])]):
            window_size = tf.identity(window_size)
      pad_left = tf.zeros(tf.concat([[window_left], source_shape[1:]], axis=0), dtype=source.dtype)
      pad_right = tf.zeros(tf.concat([[window_right], source_shape[1:]], axis=0), dtype=source.dtype)
      source = tf.concat([pad_left, source, pad_right], axis=0)  # shape[0] == n_time + window - 1
    elif padding == "valid":
      assert window_left is None and window_right is None
      n_out_time = n_time - window_size + 1
    else:
      raise Exception("invalid padding %r" % padding)
    tiled_dimshuffle = expand_dims_unbroadcast(source, axis=0, dim=window_size)  # (window,n_time+window-1,...)
    # We want to shift every dim*time block by one to the left.
    # To do this, we interpret that we have one more time frame (i.e. n_time+window).
    # We have to do some dimshuffling so that we get the right layout, then we can flatten,
    # add some padding, and then dimshuffle it back.
    # Then we can take out the first n_time frames.
    tiled_flat = tf.reshape(tiled_dimshuffle, [-1])
    rem = window_size * tf.reduce_prod(source_shape[1:])
    tiled_flat_pad_right = tf.concat([tiled_flat, tf.zeros((rem,), dtype=source.dtype)], axis=0)
    tiled_reshape_shift = tf.reshape(
      tiled_flat_pad_right,
      tf.concat([(window_size, n_out_time + window_size),
                 source_shape[1:]], axis=0))  # add time frame, (window,n_time+window,...)
    final = tiled_reshape_shift
    if new_window_axis != 0:
      final = move_axis(final, 0, new_window_axis)  # (n_time+window,...,window,...)
      final = final[:n_out_time:stride]  # (n_out_time//stride,...,window,...)
    else:
      final = final[:, :n_out_time:stride]  # (window,n_out_time//stride,...)
    # Move time-axis back to its original place.
    if new_window_axis <= time_axis:
      time_axis += 1  # New window axis was inserted before.
    if time_axis != 0:
      if new_window_axis != 0:
        final = move_axis(final, 0, time_axis)
      else:
        final = move_axis(final, 1, time_axis)
    return final


def slice_nd(x, start, size):
  """
  :param tf.Tensor x: shape (B, T, ...)
  :param tf.Tensor start: shape (B,), int32
  :param int|tf.Tensor size: scalar
  :return: [x[start_1:size], x[start_2:size], ..., x[start_B:size]], shape (B, size, ...)
    Like :func:`slice_pad_zeros`, the size in the first axis will always be ``size``,
    and we will pad with zeros.
  :rtype: tf.Tensor
  """
  with tf.name_scope("slice_nd"):
    shape = get_shape(x)
    n_batch = shape[0]

    batch_idxs = expand_dims_unbroadcast(tf.range(n_batch), 1, size)  # (n_batch, size)
    batch_idxs = tf.reshape(batch_idxs, (-1,))  # (n_batch*size,)

    window_pos = tf.expand_dims(start, 1) + tf.range(size)[None, :]  # (n_batch, size)
    window_pos = tf.reshape(window_pos, (-1,))  # (n_batch*size,)

    # build mask for zero-padding
    mask = tf.logical_or(
      tf.greater(window_pos, shape[1] - 1), tf.less(window_pos, 0))  # (n_batch*size,) tf.bool

    # clip indices so that gather_nd doesn't fail, will zero-pad later
    clip_time_idx = tf.clip_by_value(window_pos, 0, shape[1] - 1)
    indices = tf.stack([batch_idxs, clip_time_idx])  # (n_batch*size, 2)
    indices = tf.transpose(indices)  # (2, n_batch*size)

    slices = tf.gather_nd(x, indices)  # (n_batch*size, ...)

    # (B, size, ...), we assume time-axis is/was 1
    new_shape = [shape[0], size] + shape[2:]

    # zero-pad
    mask_bc = expand_multiple_dims(mask, [-1] * (slices.get_shape().ndims - 1))
    slices = where_bc(mask_bc, tf.zeros_like(slices), slices)

    slices = tf.reshape(slices, new_shape)  # (B, size, ...)
    return slices


def global_tensor(f, name):
  """
  This creates a global accessible tensor in the graph to be reused later,
  i.e. on the second call given a unique name, it will not create a new tensor
  but return the previously created tensor.
  This is for the current graph, i.e. if there is a new graph, it will recreate the tensor.

  :param () -> tf.Tensor f: callable which creates the tensor
  :param str name: global reference name for the tensor. should be a valid scope name
  :return: the tensor
  :rtype: tf.Tensor
  """
  graph = tf_compat.v1.get_default_graph()
  assert isinstance(graph, tf.Graph)
  abs_graph_name = "globals/%s:0" % name
  try:
    return graph.get_tensor_by_name(abs_graph_name)
  except KeyError:  # does not exist yet
    pass
  with tf.control_dependencies(None):  # reset any deps, e.g. being inside a while loop
    with tf.name_scope("global_tensor_%s" % name):  # relative to the current scope
      v = f()
    with tf.name_scope("globals/"):  # enter the absolute scope
      v = tf.identity(v, name=name)
  assert isinstance(v, tf.Tensor)
  assert v.name == abs_graph_name
  assert graph.get_tensor_by_name(abs_graph_name) is v
  return v


def get_global_train_flag_placeholder():
  """
  Also consider :func:`TFNetwork.get_current_network().train_flag`,
  or :func:`get_global_train_flag`.

  :return: bool scalar tensor
  :rtype: tf.Tensor
  """
  return global_tensor(
    lambda: tf_compat.v1.placeholder(tf.bool, shape=(), name="train_flag"),
    name="train_flag")


def get_global_train_flag():
  """
  :rtype: tf.Tensor|bool
  :return: global train flag
  """
  from returnn.tf.network import TFNetwork
  network = TFNetwork.get_current_network(must_exist=False)
  if network:
    return network.train_flag
  return get_global_train_flag_placeholder()


def cond_on_train_flag(fn_train, fn_eval):
  """
  Uses fn_train() or fn_eval() base on train_flag.
  It will be a branched evaluation.
  train_flag is determined via :func:`get_global_train_flag`.

  :param ()->tf.Tensor fn_train:
  :param ()->tf.Tensor fn_eval:
  :return: fn_train() if self.train_flag else fn_eval()
  :rtype: tf.Tensor
  """
  train_flag = get_global_train_flag()
  return cond(train_flag, fn_train, fn_eval)


def get_random_seed():
  """
  :rtype: int|None
  """
  from returnn.tf.network import TFNetwork
  network = TFNetwork.get_current_network(must_exist=False)
  if network:
    return network.random.randint(2 ** 31)
  return tf_compat.v1.get_seed(None)[1]


def encode_raw(x, axis=-1, seq_lens=None):
  """
  The inverse function of tf.compat.v1.decode_raw().
  Also see: https://stackoverflow.com/questions/43403147/how-to-create-a-encode-raw-tensorflow-function

  :param tf.Tensor x: of integer types [0,255], will get casted to uint8
  :param int axis: the axis to reduce-join the string. decode_raw has added it at the end
  :param tf.Tensor|None seq_lens: must have same shape as x after reduce-joining.
    Note that using seq_lens will make our output not compatible with tf.compat.v1.decode_raw() anymore
    because tf.compat.v1.decode_raw() requires all strings to be of the same length.
  :return: string tensor
  :rtype: tf.Tensor
  """
  with tf.name_scope("encode_raw"):
    character_lookup = global_tensor(
      lambda: tf.constant([chr(i) for i in range(256)]), name="character_lookup")
    raw_bytes = tf.bitcast(x, tf.uint8, name="raw_bytes")
    chars = tf.gather(character_lookup, indices=tf.cast(raw_bytes, tf.int32), name="chars")
    strings = tf_compat.v1.reduce_join(chars, axis=axis, name="strings")
    if seq_lens is not None:
      strings = tf_compat.v1.substr(strings, pos=tf.zeros_like(seq_lens), len=seq_lens)
    return strings


def get_shared_vocab(vocab_strings):
  """
  The vocab is shared across the current instance of the computation graph.
  The tensor name might be different in different runs.

  :param list[str] vocab_strings:
  :return: shape (len(vocab_strings),), tf.string
  :rtype: tf.Tensor
  """
  return global_tensor(
    lambda: tf.convert_to_tensor(vocab_strings),
    name="shared_vocab_%s" % hex(hash(tuple(vocab_strings))).replace("-", "_"))


def map_labels(x, label_map, name="map_labels"):
  """
  :param tf.Tensor|tf.SparseTensor x: values of integer types
  :param dict[int,int|None] label_map: should be dense on input
  :param str name:
  :return: mapped values
  :rtype: tf.Tensor|tf.SparseTensor
  """
  if any([v is None for v in label_map.values()]):
    assert isinstance(x, tf.SparseTensor), "not supported otherwise currently"
    x = remove_labels(x, labels=[k for (k, v) in label_map.items() if v is None])
    label_map = {k: v if v is not None else -1 for (k, v) in label_map.items()}
  if isinstance(x, tf.SparseTensor):
    return tf.SparseTensor(
      indices=x.indices,
      values=map_labels(x.values, label_map=label_map, name=name),
      dense_shape=x.dense_shape)
  with tf.name_scope(name):
    assert label_map
    assert 0 in label_map
    assert len(label_map) - 1 in label_map
    lookup = global_tensor(
      lambda: tf.constant([label_map[i] for i in range(len(label_map))]),
      name="label_map_lookup_id%i" % id(label_map))
    y = tf.gather(lookup, indices=x, name="mapped")
    return y


def remove_labels(x, labels):
  """
  :param tf.SparseTensor x: sequences, i.e. the indices are interpret as (batch,time)
  :param set[int]|list[int] labels:
  :return: x where all provided labels are removed, and the indices are changed accordingly
  :rtype: tf.SparseTensor
  """
  if not labels:
    return x
  x.indices.set_shape((tf.TensorShape((None, 2))))
  x.values.set_shape((tf.TensorShape((None,))))
  x.dense_shape.set_shape(tf.TensorShape((2,)))
  x_ = tf_compat.v1.sparse_to_dense(sparse_indices=x.indices, sparse_values=x.values, output_shape=x.dense_shape)
  seq_lens = get_sparse_tensor_length(x)
  z, _ = sparse_labels_with_seq_lens(x_, seq_lens=seq_lens, post_filter_idx=labels)
  return z


def pad_zeros_in_axis(x, before=0, after=0, axis=0):
  """
  :param tf.Tensor x:
  :param int|tf.Tensor before:
  :param int|tf.Tensor after:
  :param int axis:
  :return:
  """
  with tf.name_scope("pad_zeros_in_axis"):
    paddings = [[0, 0] for _ in range(x.get_shape().ndims)]
    paddings[axis] = [before, after]
    return tf.pad(x, paddings=paddings)


def slice_pad_zeros(x, begin, end, axis=0):
  """
  :param tf.Tensor x: of shape (..., time, ...)
  :param int|tf.Tensor begin:
  :param int|tf.Tensor end:
  :param int axis:
  :return: basically x[begin:end] (with axis==0) but if begin < 0 or end > x.shape[0],
   it will not discard these frames but pad zeros, such that the resulting shape[0] == end - begin.
  :rtype: tf.Tensor
  """
  with tf.name_scope("slice_pad_zeros"):
    min_frame = tf.minimum(begin, end)
    left_rem = -min_frame
    x, begin, end = tf.cond(
      tf.less_equal(left_rem, 0),
      lambda: [x, begin, end],
      lambda: [pad_zeros_in_axis(x, before=left_rem, axis=axis), begin + left_rem, end + left_rem])
    max_frame = tf.maximum(begin, end)
    right_rem = max_frame - tf.shape(x)[axis]
    x = tf.cond(
      tf.less_equal(right_rem, 0),
      lambda: x,
      lambda: pad_zeros_in_axis(x, after=right_rem, axis=axis))
    return single_strided_slice(x, axis=axis, begin=begin, end=end)


def post_control_dependencies(x, updates):
  """
  :param tf.Tensor|list[tf.Tensor]|dict[str,tf.Tensor] x:
  :param list[tf.Operation] updates:
  :return: identity(x) with control_dependencies(updates)
  :rtype: tf.Tensor|list[tf.Tensor]|dict[str,tf.Tensor]
  """
  with tf.name_scope("post_control_dependencies"):
    with tf.control_dependencies(updates):
      if isinstance(x, tf.Tensor):
        return tf.identity(x)
      elif isinstance(x, (tuple, list)):
        return [tf.identity(v) for v in x]
      elif isinstance(x, dict):
        return {k: tf.identity(v) for (k, v) in x.items()}
      else:
        raise ValueError("type of %r not expected" % x)


@contextlib.contextmanager
def sequential_control_dependencies(ls):
  """
  tf.control_dependencies but each operation will be created such that it is executed
  after the ones coming before in the list, i.e. l[0] is executed first, l[-1] is executed last.

  :param list[()->(tf.Operation|tf.Tensor)] ls:
  """
  with tf.control_dependencies([ls[0]()]) as dep:
    if len(ls) > 1:
      with sequential_control_dependencies(ls[1:]) as dep2:
        yield dep2
    else:
      yield dep


def global_queue(name, queue_type, capacity, dtypes, shapes=None, names=None):
  """
  :param str name: global name
  :param (...)->tf.QueueBase queue_type: some function which creates a queue
  :param capacity:
  :param list[tf.DType|str] dtypes:
  :param list[tf.TensorShape|tuple[int|None]]|None shapes:
  :param list[str]|None names:
  :rtype: tf.QueueBase
  """
  queue_ref = global_tensor(
    name=name,
    f=lambda: queue_type(capacity=capacity, dtypes=dtypes, shapes=shapes, names=names).queue_ref)
  queue = tf_compat.v1.QueueBase(dtypes=dtypes, shapes=shapes, names=names, queue_ref=queue_ref)
  return queue


def init_variable_if_needed(v):
  """
  :param tf.Variable v:
  :rtype: tf.Operation
  """
  def make_init():
    """
    :rtype: tf.Operation
    """
    # Cannot use tf.compat.v1.variables_initializer(), see here: https://stackoverflow.com/questions/44354964/
    with tf.control_dependencies([tf_compat.v1.assign(v, v.initial_value)]):
      return tf.no_op()

  maybe_init = tf.cond(
    tf_compat.v1.is_variable_initialized(v),
    lambda: tf.no_op(),
    make_init,
    name="maybe_init")

  return maybe_init


def auto_init_var(v):
  """
  :param tf.Variable v:
  :return: a reference to the var via tf.identity
  :rtype: tf.Tensor
  """
  with tf.control_dependencies(init_variable_if_needed(v)):
    return tf.identity(v, name="auto_init_var")


def true_once():
  """
  :return: tensor which will be True once and then always False
    Internally, this creates a non-trainable variable as a helper.
  :rtype: tf.Tensor
  """
  with tf_compat.v1.variable_scope("true_once"):
    v = tf.Variable(initial_value=True, trainable=False, name="true_once_var")
    with tf.control_dependencies([init_variable_if_needed(v)]):
      # Cannot use tf.identity because that would give us a reference to the var but we want to copy it now.
      x = tf.where(v.read_value(), True, False)
      with tf.control_dependencies([x]):
        x = tf.identity(x)
        reset = tf_compat.v1.assign(v, False)
        with tf.control_dependencies([x, reset]):
          x = tf.identity(x)
  return x


# noinspection PyPep8Naming
def raise_OutOfRangeError():
  """
  :return: an op which raises an OutOfRangeError
  :rtype: tf.Operation
  """
  # Kind of hacky. We create some dummy queue, close it and every time we call dequeue on it,
  # it will raise the desired exception.
  with tf.name_scope("raise_OutOfRangeError"):
    queue = global_queue(name="raise_exception/queue", queue_type=tf_compat.v1.FIFOQueue, capacity=1, dtypes=[tf.bool])
    # We must only close it once, otherwise we could get a CancelledError.
    queue_open = global_tensor(f=true_once, name="raise_exception/queue_open")
    with tf.control_dependencies([tf.cond(queue_open, lambda: queue.close(), lambda: tf.no_op())]):
      return queue.dequeue()


def enforce_copy(x):
  """
  :param tf.Tensor|tf.Variable x:
  :return: copy of input, i.e. enforces that this is not a ref
  :rtype: tf.Tensor
  """
  with tf.name_scope("copy"):
    zero = x.dtype.as_numpy_dtype()
    return tf.add(x, zero)


def copy_unknown_shape(x):
  """
  :param tf.Tensor x:
  :return: tensor, copy of x, which has an unknown shape by intention
  :rtype: tf.Tensor
  """
  x = tf.convert_to_tensor(x)

  # Currently we use py_func, which is somewhat inefficient, and also CPU-only.
  # But this is fine for our use cases.

  # noinspection PyShadowingNames
  def py_copy(x_np):
    """
    :param numpy.ndarray x_np:
    :rtype: numpy.ndarray
    """
    return x_np

  y, = tf_compat.v1.py_func(py_copy, [x], [x.dtype], name="py_copy")
  assert isinstance(y, tf.Tensor)
  assert y.shape.ndims is None
  return y


def view_as(x, dtype):
  """
  Does the numpy.view equivalent.
  Note that the current implementation is inefficient (uses tf.compat.v1.py_func) and CPU-only.
  Also see :func:`tf.bitcast`.

  :param tf.Tensor x:
  :param tf.DType dtype:
  :return: x.view(dtype) equivalent (see numpy.view)
  """
  import numpy

  # noinspection PyShadowingNames
  def py_wrap_numpy_view(x):
    """
    :param numpy.ndarray x:
    :rtype: numpy.ndarray
    """
    assert isinstance(x, numpy.ndarray)
    return x.view(dtype.as_numpy_dtype)

  y, = tf_compat.v1.py_func(
    py_wrap_numpy_view,
    [x], [dtype],
    name="py_wrap_numpy_view")
  assert isinstance(y, tf.Tensor)
  y.set_shape(x.get_shape())
  return y


def broadcast_gradient_args(shape_x, shape_y):
  """
  :param tf.Tensor shape_x:
  :param tf.Tensor shape_y:
  :return: (axis reduce arg for grad x, axis reduce arg for grad y)
  :rtype: (tf.Tensor, tf.Tensor)
  """
  from tensorflow.python.ops import gen_array_ops
  if hasattr(gen_array_ops, '_broadcast_gradient_args'):  # earlier TF
    # noinspection PyProtectedMember
    return gen_array_ops._broadcast_gradient_args(shape_x, shape_y)
  # Since TF 1.8.0, this is public.
  # noinspection PyUnresolvedReferences
  return gen_array_ops.broadcast_gradient_args(shape_x, shape_y)


def _alternative_minmax_grad(op, grad):
  """
  :param tf.Operation op: e.g. tf.minimum(x, y) or tf.maximum(x, y)
  :param tf.Tensor grad:
  :rtype: tf.Tensor, tf.Tensor
  """
  x = op.inputs[0]
  y = op.inputs[1]
  sx = tf.shape(x)
  sy = tf.shape(y)
  rx, ry = broadcast_gradient_args(sx, sy)
  gx = tf.reshape(tf.reduce_sum(grad, axis=rx), sx)
  gy = tf.reshape(tf.reduce_sum(grad, axis=ry), sy)
  return gx, gy


def _register_alternative_minmax_grad():
  """
  :return: the op name to use with gradient_override_map
  :rtype: str
  """
  grad_name = "alternative_minmax_grad"
  opt_register_grad_func(
    op_type=grad_name,
    grad_func=_alternative_minmax_grad,
    assert_is_same=True)
  return grad_name


def maximum_with_identity_grad(x, y):
  """
  :param tf.Tensor x:
  :param tf.Tensor|float y:
  :return: tf.maximum(x, y) where each will receive the gradient
  :rtype: tf.Tensor
  """
  with tf.name_scope("maximum_with_identity_grad"):
    # An alternative to gradient_override_map would be :class:`CustomGradient` which is more generic.
    grad_name = _register_alternative_minmax_grad()
    g = tf_compat.v1.get_default_graph()
    with g.gradient_override_map({"Maximum": grad_name}):
      return tf.maximum(x, y)


def minimum_with_identity_grad(x, y):
  """
  :param tf.Tensor x:
  :param tf.Tensor|float y:
  :return: tf.minimum(x, y) where each will receive the gradient
  :rtype: tf.Tensor
  """
  with tf.name_scope("minimum_with_identity_grad"):
    # An alternative to gradient_override_map would be :class:`CustomGradient` which is more generic.
    grad_name = _register_alternative_minmax_grad()
    g = tf_compat.v1.get_default_graph()
    with g.gradient_override_map({"Minimum": grad_name}):
      return tf.minimum(x, y)


def clip_by_value_with_identity_grad(x, clip_value_min, clip_value_max):
  """
  :param tf.Tensor x:
  :param tf.Tensor|float clip_value_min:
  :param tf.Tensor|float clip_value_max:
  :return: tf.clip_by_value(x, clip_value_min, clip_value_max) where each will receive the gradient
  :rtype: tf.Tensor
  """
  with tf.name_scope("clip_by_value_with_identity_grad"):
    # An alternative to gradient_override_map would be :class:`CustomGradient` which is more generic.
    grad_name = _register_alternative_minmax_grad()
    g = tf_compat.v1.get_default_graph()
    with g.gradient_override_map({"Minimum": grad_name, "Maximum": grad_name}):
      x = tf.maximum(x, clip_value_min)
      x = tf.minimum(x, clip_value_max)
      return x


def safe_log(x, eps=1e-20, use_fake_grad=True):
  """
  Safe wrapper around :func:`tf.log` which avoids infs or nans in the gradient.

  :param tf.Tensor x:
  :param float|tf.Tensor eps:
  :param bool use_fake_grad: True -> use maximum_with_identity_grad, False -> use tf.maximum
  :return: log(max(x, eps))
  :rtype: tf.Tensor
  """
  with tf.name_scope("safe_log"):
    y = check_base_op_type_and_replace(x, "Exp", "Identity")
    if y is not None:
      return y
    y = check_base_op_type_and_replace(x, "Softmax", "LogSoftmax")
    if y is not None:
      return y
    y = check_base_op_type_and_replace(x, "Sigmoid", "LogSigmoid")
    if y is not None:
      return y
    if use_fake_grad:
      x = maximum_with_identity_grad(x, eps)
    else:
      x = tf.maximum(x, eps)
    return tf_compat.v1.log(x)


def safe_exp(x, eps=1e-20):
  """
  :param tf.Tensor x:
  :param float eps:
  :return: exp(x), but does clipping before, such that it never returns inf nor exactly 0.0.
    Also, we make sure that we use the gradient in all cases.
  :rtype: tf.Tensor
  """
  import numpy
  with tf.name_scope("safe_exp"):
    y = check_base_op_type_and_replace(x, "Log", "Identity")
    if y is not None:
      return y
    y = check_base_op_type_and_replace(x, "LogSoftmax", "Softmax")
    if y is not None:
      return y
    y = check_base_op_type_and_replace(x, "LogSigmoid", "Sigmoid")
    if y is not None:
      return y
    clip_value_min = numpy.log(eps)
    clip_value_max = numpy.log(1.0 / eps)
    x = clip_by_value_with_identity_grad(x, clip_value_min, clip_value_max)
    return tf.exp(x)


def l1_normalized(x, axis=-1, eps=1e-20, use_logsumexp=False, is_not_negative=False):
  """
  :param tf.Tensor x: assumes != 0
  :param int|tf.Tensor axis: in range [-rank(x),rank(x)]
  :param float|tf.Tensor|None eps: for safety, to ensure that tf.reduce_sum(tf.abs(x)) >= eps
  :param bool use_logsumexp: eps must not be None
  :param bool is_not_negative:
  :return: y such that tf.reduce_sum(tf.abs(y)) == 1. i.e. y = x / tf.reduce_sum(tf.abs(x)).
  :rtype: tf.Tensor
  """
  with tf.name_scope("l1_normalized"):
    if not is_not_negative:
      x = tf.abs(x)
    if eps is not None:
      # Do that here, not after reduce_sum, so that we get a proper gradient to each entry.
      x = maximum_with_identity_grad(x, eps)
    if use_logsumexp:
      weighted_input_sum = tf.exp(tf.reduce_logsumexp(tf_compat.v1.log(x), axis=axis, keepdims=True))
    else:
      weighted_input_sum = tf.reduce_sum(x, axis=axis, keepdims=True)
    divisor = tf_compat.v1.reciprocal(weighted_input_sum)
    return tf.multiply(x, divisor)


def lin_exp(x, use_safe_exp=True):
  """
  :param tf.Tensor x:
  :param bool use_safe_exp:
  :return: x + 1 if x >= 0 else exp(x). this is smooth and differentiable everywhere
  :rtype: tf.Tensor
  """
  with tf.name_scope("lin_exp"):
    if use_safe_exp:
      neg_part = safe_exp(tf.minimum(x, 0))
    else:
      neg_part = tf.exp(tf.minimum(x, 0))
    return tf.where(tf.greater_equal(x, 0), x + 1, neg_part)


def lin_exp_normed(x, axis=-1, eps=1e-10):
  """
  This can be used as an alternative to softmax. It uses :func:`lin_exp` instead of exp.

  :param tf.Tensor x:
  :param int|tf.Tensor axis: in range [-rank(x),rank(x)]
  :param float|tf.Tensor|None eps: for safety, to ensure that tf.reduce_sum(tf.abs(x)) >= eps
  :return: y = l1_normalized(lin_exp(x)), i.e. tf.reduce_sum(y) == 1, and y >= 0.
  :rtype: tf.Tensor
  """
  with tf.name_scope("lin_exp_normed"):
    return l1_normalized(lin_exp(x), axis=axis, eps=eps, is_not_negative=True)


def check_base_op_type_and_replace(x, op_type, new_op_type):
  """
  Suppose you have ``x = tf.nn.softmax(z)`` and you want to get ``y = tf.nn.log_softmax(z)``.
  This function will test to see if ``x`` is of that kind and then return ``y``.

  :param tf.Tensor x:
  :param str op_type: e.g. "Softmax"
  :param str new_op_type: e.g. "LogSoftmax"
  :return: x with new_op_type instead of op_type, or None if not matched
  :rtype: tf.Tensor|None
  """
  assert isinstance(x, tf.Tensor)
  assert x.op.outputs[0] is x
  # Handle cases like f(tf.nn.softmax(z)) for f in tf.identity, tf.reshape, etc.
  safe_post_op_types = ["Identity", "Reshape", "Transpose", "GatherNd"]
  if op_type not in safe_post_op_types and x.op.type in safe_post_op_types:
    inner = check_base_op_type_and_replace(x.op.inputs[0], op_type=op_type, new_op_type=new_op_type)
    if inner is None:
      return None
    op = copy_op(x.op, inputs=[inner] + list(x.op.inputs[1:]))
    return op.outputs[0]
  if x.op.type != op_type:
    return None
  op = copy_op(x.op, op_type=new_op_type)
  return op.outputs[0]


def copy_op(op, op_type=None, inputs=None):
  """
  Copies a tf.Operation.

  :param tf.Operation op:
  :param str|None op_type:
  :param list[tf.Tensor]|None inputs:
  :return: copy of op but optionally change op.type == op_type or op.inputs == inputs
  :rtype: tf.Operation
  """
  assert isinstance(op, tf.Operation)
  g = op.graph
  if op_type is None:
    op_type = op.type
  if inputs is None:
    inputs = list(op.inputs)
  # Use some aliases, for simplicity.
  # Maybe in the future we would also wrap some deprecated/outdated ops.
  if op_type == "LogSigmoid":
    assert len(inputs) == 1
    return tf_compat.v1.log_sigmoid(inputs[0]).op
  # Fallback to the generic case.
  new_op = g.create_op(
    op_type=op_type,
    op_def=op.op_def if op_type == op.type else None,  # Can only copy op_def if it is the same op_type.
    inputs=inputs,
    input_types=[x.dtype for x in inputs],
    dtypes=[x.dtype for x in op.outputs],  # output types
    attrs=dict(op.node_def.attr.items()))
  return new_op


def copy_tensor(x):
  """
  Similar to tf.identity, but we ensure here that the return value has its own memory.
  This can be relevant when you want to keep a copy of the original variable value.
  See :func:`get_variable_value_copy_before_update_ops` for usage.

  :param tf.Tensor x:
  :return: a copy of x (points to new memory)
  :rtype: tf.Tensor
  """
  # I think there is a copy op also in TF, but I don't see it in the Python API.
  with tf.name_scope("copy"):
    return tf.add(x, tf.constant(0, dtype=x.dtype, name="dummy_zero"), name="copy")


def smoothing_cross_entropy(logits,
                            labels,
                            label_smoothing,
                            gaussian=False,
                            vocab_size=None,
                            logits_are_normalized=False):
  """
  Cross entropy with label smoothing to limit over-confidence.
  Code adapted from here:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_layers.py

  :param tf.Tensor logits: Tensor of size shape(labels) + [vocab_size]
  :param tf.Tensor labels: Tensor of size [...]
  :param int|tf.Tensor vocab_size: Tensor representing the size of the vocabulary.
  :param float label_smoothing: confidence = 1.0 - label_smoothing.
    Used to determine on and off values for label smoothing.
    If `gaussian` is true, `confidence` is the variance to the gaussian distribution.
    A common default value is 0.1. See:
      https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_hparams.py
  :param bool gaussian: Uses a gaussian distribution for label smoothing
  :param bool logits_are_normalized:
  :return: Tensor of the same shape as `labels` and of the same dtype as `logits`.
  :rtype: tf.Tensor
  """
  with tf.name_scope("smoothing_cross_entropy"):
    if vocab_size is None:
      vocab_size = get_shape_dim(logits, -1, name="vocab_size")
    confidence = 1.0 - label_smoothing
    # Low confidence is given to all non-true labels, uniformly.
    low_confidence = (1.0 - confidence) / tf.cast(vocab_size - 1, tf.float32)
    # Normalizing constant is the best cross-entropy value with soft targets.
    # We subtract it just for readability, makes no difference on learning.
    normalizing = -(
      confidence * tf_compat.v1.log(confidence) + tf.cast(vocab_size - 1, tf.float32) *
      low_confidence * tf_compat.v1.log(low_confidence + 1e-20))  # scalar

    if gaussian:
      labels = tf.cast(labels, tf.float32)
      normal_dist = tf_compat.v1.distributions.Normal(loc=labels, scale=confidence)
      # Locations to evaluate the probability distributions.
      soft_targets = normal_dist.prob(
        expand_multiple_dims(
          tf.cast(tf.range(vocab_size), tf.float32),
          axes=[i + 1 for i in range(labels.get_shape().ndims)]))  # [vocab_size] + shape(labels)
      soft_targets = move_axis(
        soft_targets, old_axis=0, new_axis=labels.get_shape().ndims)  # shape(labels) + [vocab_size]
    else:
      # TODO: We could implement an own native op which does not need to create this one-hot vector
      # which consumes lots of memory.
      soft_targets = tf.one_hot(
        tf.cast(labels, tf.int32),
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence)  # shape(labels) + [vocab_size
    if logits_are_normalized:
      xentropy = -tf.reduce_sum(soft_targets * logits, axis=-1)
    else:
      xentropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=soft_targets)  # shape(labels)
    return xentropy - normalizing  # shape(labels)


def softmax_cross_entropy_over_size(logits, labels, stable_gradient=True):
  """
  The last spatial axis with dyn size info will be used and interpret as the class probabilities
  over the size.
  We will mask logits outside of the size.
  We expect that the labels have the corresponding invalid frames already set to 0.0.
  This can be used to measure the cross entropy between two soft alignments / attention weights.

  :param Data logits: in log space, unscaled. shape (...,T,...).
    Shape can be eg. (B,dec-T,enc-T,H...), or (dec-T,enc-T,B,H...), etc.
    If it has multiple axes with dynamic size, we use the last one (enc-T in the example).
  :param Data labels: in prob space. shape compatible to `logits` (but axes can be ordered differently).
    Shape can be e.g. (B,dec-T,enc-T,H...) etc.
    If is has multiple spatial axes, we expect them to be in the same order as of `logits`
  :param bool stable_gradient: whether to use an explicit gradient
  :return: shape as logits, but the T axis removed.
  :rtype: tf.Tensor
  """
  assert len(logits.size_placeholder) == len(labels.size_placeholder) >= 1  # expect same number, and at least 1
  assert logits.batch_ndim == labels.batch_ndim
  logits_enc_time_axis = logits.get_batch_axis(max(logits.size_placeholder.keys()))
  enc_seq_len = logits.size_placeholder[logits.get_batch_axis_excluding_batch(logits_enc_time_axis)]
  logits_t = logits.placeholder
  labels_t = labels.placeholder
  # Assume that it is faster to transpose labels, as they are probably static.
  # Transpose such that it is compatible to logits.
  labels_perm = []
  labels_spatial_dims = labels.get_spatial_batch_axes()
  logits_spatial_dims = logits.get_spatial_batch_axes()
  assert len(labels_spatial_dims) == len(logits_spatial_dims)
  for i in range(logits.batch_ndim):
    if i == logits.batch_dim_axis:
      labels_perm.append(labels.batch_dim_axis)
    elif i in logits_spatial_dims:
      labels_perm.append(labels_spatial_dims[logits_spatial_dims.index(i)])
    elif i == logits.feature_dim_axis:
      assert logits.batch_shape[logits.feature_dim_axis] == labels.batch_shape[labels.feature_dim_axis]
      labels_perm.append(labels.feature_dim_axis)
    else:
      raise Exception("not matching %r vs %r, axis %i" % (logits, labels, i))
  labels_t = tf.transpose(labels_t, labels_perm)  # should be same shape as logits
  labels_shape = tf.shape(labels_t)
  n_batch = labels_shape[logits.batch_dim_axis]
  enc_time_dim = labels_shape[logits_enc_time_axis]
  # See SoftmaxOverSpatialLayer.
  if logits.batch_dim_axis < logits_enc_time_axis:
    mask = sequence_mask(enc_seq_len)  # (B,encT)
  else:
    mask = sequence_mask_time_major(enc_seq_len)  # (encT,B)
  mask_expand_dims_shape = []
  for i in range(logits.batch_ndim):
    if i == logits.batch_dim_axis:
      mask_expand_dims_shape.append(n_batch)
    elif i == logits_enc_time_axis:
      mask_expand_dims_shape.append(enc_time_dim)
    else:
      mask_expand_dims_shape.append(1)
  assert (any([dim is n_batch for dim in mask_expand_dims_shape]) and
          any([dim is enc_time_dim for dim in mask_expand_dims_shape]))
  mask = tf.reshape(mask, mask_expand_dims_shape)  # (...,B,...,enc-T), just like logits/labels
  logits_t = where_bc(mask, logits_t, float("-inf") * tf.ones_like(logits_t))
  # We only apply the mask to the logits. We expect that we already have it zeroed for labels.
  # Unfortunately we cannot use tf.nn.softmax_cross_entropy_with_logits because we would get inf loss.
  log_probs_t = tf.nn.log_softmax(logits_t, axis=logits_enc_time_axis)
  log_probs_t = where_bc(mask, log_probs_t, tf.zeros_like(logits_t))  # filter out the infs
  out = labels_t * log_probs_t
  out = -tf.reduce_sum(out, axis=logits_enc_time_axis, keepdims=True)
  if stable_gradient:
    probs_t = tf.nn.softmax(logits_t, axis=logits_enc_time_axis)
    out = custom_gradient.generic_loss_and_error_signal(loss=out, x=logits_t, grad_x=probs_t - labels_t)
  out = tf.squeeze(out, axis=logits_enc_time_axis)
  return out


def interpolate_bilinear(grid, query_points, name='interpolate_bilinear', indexing='ij'):
  """
  Similar to Matlab's interp2 function.
  Finds values for query points on a grid using bilinear interpolation.
  Adapted from tensorflow.contrib.image.dense_image_warp, from newer TF version which supports variable-sized images.

  :param tf.Tensor grid: a 4-D float `Tensor` of shape `[batch, height, width, channels]`.
  :param tf.Tensor query_points: a 3-D float `Tensor` of N points with shape `[batch, N, 2]`.
    Note that this function is not differentiable w.r.t. the query points.
  :param str name: a name for the operation (optional).
  :param str indexing: whether the query points are specified as row and column (ij), or Cartesian coordinates (xy).
  :returns: a 3-D `Tensor` with shape `[batch, N, channels]`
  :rtype: tf.Tensor
  """
  import numpy
  assert indexing in ('ij', 'xy')

  with tf.name_scope(name):
    grid = tf.convert_to_tensor(grid)
    query_points = tf.convert_to_tensor(query_points)

    shape = tf.shape(grid)
    batch_size, height, width, channels = [shape[i] for i in range(grid.get_shape().ndims)]
    shape = [batch_size, height, width, channels]
    query_type = query_points.dtype
    grid_type = grid.dtype

    query_points.set_shape((None, None, 2))
    num_queries = tf.shape(query_points)[1]

    with tf.control_dependencies([
        tf_compat.v1.assert_greater_equal(height, 2, message='Grid height must be at least 2.'),
        tf_compat.v1.assert_greater_equal(width, 2, message='Grid width must be at least 2.')
    ]):
      alphas = []
      floors = []
      ceils = []
      index_order = [0, 1] if indexing == 'ij' else [1, 0]
      unstacked_query_points = tf.unstack(query_points, axis=2)

    for dim in index_order:
      with tf.name_scope('dim-' + str(dim)):
        queries = unstacked_query_points[dim]

        size_in_indexing_dimension = shape[dim + 1]

        # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
        # is still a valid index into the grid.
        max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
        min_floor = tf.constant(0.0, dtype=query_type)
        floor = tf.minimum(tf.maximum(min_floor, tf.floor(queries)), max_floor)
        int_floor = tf.cast(floor, tf.int32)
        floors.append(int_floor)
        ceil = int_floor + 1
        ceils.append(ceil)

        # alpha has the same type as the grid, as we will directly use alpha
        # when taking linear combinations of pixel values from the image.
        alpha = tf.cast(queries - floor, grid_type)
        min_alpha = tf.constant(0.0, dtype=grid_type)
        max_alpha = tf.constant(1.0, dtype=grid_type)
        alpha = tf.minimum(tf.maximum(min_alpha, alpha), max_alpha)

        # Expand alpha to [b, n, 1] so we can use broadcasting
        # (since the alpha values don't depend on the channel).
        alpha = tf.expand_dims(alpha, 2)
        alphas.append(alpha)
    assert len(alphas) == len(floors) == len(ceils) == len(index_order) == 2

    with tf.control_dependencies([
        tf_compat.v1.assert_less_equal(
          tf.cast(batch_size, tf.float32) * tf.cast(height, tf.float32) * tf.cast(width, tf.float32),
          numpy.iinfo(numpy.int32).max / 8.,
          message="""The image size or batch size is sufficiently large
                     that the linearized addresses used by array_ops.gather
                     may exceed the int32 limit.""")
    ]):
      flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
      batch_offsets = tf.reshape(tf.range(batch_size) * height * width, [batch_size, 1])

    # This wraps array_ops.gather. We reshape the image data such that the
    # batch, y, and x coordinates are pulled into the first dimension.
    # Then we gather. Finally, we reshape the output back. It's possible this
    # code would be made simpler by using array_ops.gather_nd.
    def gather(y_coords, x_coords, name_):
      """
      :param tf.Tensor y_coords:
      :param tf.Tensor x_coords:
      :param str name_:
      :rtype: tf.Tensor
      """
      with tf.name_scope('gather-' + name_):
        linear_coordinates = batch_offsets + y_coords * width + x_coords
        gathered_values = tf.gather(flattened_grid, linear_coordinates)
        return tf.reshape(gathered_values, [batch_size, num_queries, channels])

    # grab the pixel values in the 4 corners around each query point
    top_left = gather(floors[0], floors[1], 'top_left')
    top_right = gather(floors[0], ceils[1], 'top_right')
    bottom_left = gather(ceils[0], floors[1], 'bottom_left')
    bottom_right = gather(ceils[0], ceils[1], 'bottom_right')

    # now, do the actual interpolation
    with tf.name_scope('interpolate'):
      interp_top = alphas[1] * (top_right - top_left) + top_left
      interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
      interp = alphas[0] * (interp_bottom - interp_top) + interp_top

    return interp


def dense_image_warp(image, flow, name='dense_image_warp'):
  """
  Image warping using per-pixel flow vectors.
  Adapted from tensorflow.contrib.image.dense_image_warp, from newer TF version which supports variable-sized images.

  :param tf.Tensor image: 4-D float `Tensor` with shape `[batch, height, width, channels]`.
  :param tf.Tensor flow: A 4-D float `Tensor` with shape `[batch, height, width, 2]`.
    E.g. via :func:`create_random_warp_flow_2d`.
    Note that this function is not differentiable w.r.t. the flow.
  :param str name: A name for the operation (optional).
  :returns: A 4-D float `Tensor` with shape`[batch, height, width, channels]` and same type as input image.
  :rtype: tf.Tensor
  """
  with tf.name_scope(name):
    image_shape = tf.shape(image)
    batch_size, height, width, channels = [image_shape[i] for i in range(image.get_shape().ndims)]
    # The flow is defined on the image grid. Turn the flow into a list of query points in the grid space.
    grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
    stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis=2), flow.dtype)
    batched_grid = tf.expand_dims(stacked_grid, axis=0)
    query_points_on_grid = batched_grid - flow
    query_points_flattened = tf.reshape(query_points_on_grid, [batch_size, height * width, 2])
    # Compute values at the query points, then reshape the result back to the image grid.
    interpolated = interpolate_bilinear(image, query_points_flattened)
    interpolated = tf.reshape(interpolated, [batch_size, height, width, channels])
    return interpolated


def create_random_warp_flow_2d(shape, std=None, scale=10., blur_std=2.):
  """
  Can be used with :func:`dense_image_warp`.

  :param tf.Tensor|(int,int,int) shape: 1D, contains (batch,height,width). e.g. ``tf.shape(image)[:-1]``
  :param float|(float,float) std:
  :param float|(float,float) scale:
  :param float|(float,float) blur_std:
  :return: [batch, height, width, 2]
  :rtype: tf.Tensor
  """
  if not isinstance(std, (tuple, list)):
    std = (std, std)
  if not isinstance(scale, (tuple, list)):
    scale = (scale, scale)
  if not isinstance(blur_std, (tuple, list)):
    blur_std = (blur_std, blur_std)
  if isinstance(shape, (tuple, list)):
    assert len(shape) == 3
  else:
    assert isinstance(shape, tf.Tensor)
    shape.set_shape((3,))  # b,h,w
  small_shape = [shape[0], shape[1] // int(scale[0]), shape[2] // int(scale[1])]
  flow1 = tf_compat.v1.random_normal(shape=small_shape, stddev=std[0])
  flow2 = tf_compat.v1.random_normal(shape=small_shape, stddev=std[1])
  flow = tf.stack([flow1, flow2], axis=-1)  # [batch, height, width, 2]
  flow.set_shape((None, None, None, 2))
  flow = gaussian_blur_2d(flow, kernel_std=blur_std)
  flow = tf_compat.v1.image.resize_images(flow, size=[shape[1], shape[2]])
  return flow


def gaussian_kernel_2d(size, std):
  """
  :param int|(int,int) size:
  :param float|(float,float) std:
  :return: (size_x*2+1,size_y*2+1), float32
  :rtype: tf.Tensor
  """
  if isinstance(size, (tuple, list)):
    size_x, size_y = size
  else:
    size_x, size_y, = size, size
  if isinstance(std, (tuple, list)):
    std_x, std_y = std
  else:
    std_x, std_y = std, std
  values_x = tf.range(start=-size_x, limit=size_x + 1, dtype=tf.float32)
  values_y = tf.range(start=-size_y, limit=size_y + 1, dtype=tf.float32)
  dx = tf_compat.v1.distributions.Normal(0.0, std_x)
  dy = tf_compat.v1.distributions.Normal(0.0, std_y)
  values_x = dx.prob(values_x)
  values_y = dy.prob(values_y)
  values = tf.einsum('i,j->ij', values_x, values_y)
  values.set_shape((size_x * 2 + 1, size_y * 2 + 1))
  return values / tf.reduce_sum(values)


def gaussian_blur_2d(image, kernel_size=None, kernel_std=None):
  """
  :param tf.Tensor image: (batch,width,height,channel)
  :param int|(int,int)|None kernel_size:
  :param float|(float,float)|None kernel_std:
  :return: image
  :rtype: tf.Tensor
  """
  if kernel_std is None:
    kernel_std = 1.
  if kernel_size is None:
    if isinstance(kernel_std, (tuple, list)):
      assert len(kernel_std) == 2
      kernel_size = (int(kernel_std[0] * 2 + 1), int(kernel_std[1] * 2 + 1))
    else:
      kernel_size = int(kernel_std * 2 + 1)
  image.set_shape((None, None, None, None))
  orig_shape = tf.shape(image)
  orig_shape = [orig_shape[i] for i in range(image.get_shape().ndims)]
  image = tf.transpose(image, [0, 3, 1, 2])  # (B,C,W,H)
  image = tf.reshape(image, [orig_shape[0] * orig_shape[3], orig_shape[1], orig_shape[2], 1])  # (B*C,W,H,1)
  gauss_kernel = gaussian_kernel_2d(size=kernel_size, std=kernel_std)
  gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]
  image = tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
  image = tf.reshape(image, [orig_shape[0], orig_shape[3], orig_shape[1], orig_shape[2]])  # (B,C,W,H)
  image = tf.transpose(image, [0, 2, 3, 1])  # (B,W,H,C)
  return image


def _py_bleu_score(hypothesis, truth, hyp_seq_lens, truth_seq_lens):
  """
  :param numpy.ndarray hypothesis:
  :param numpy.ndarray truth:
  :param numpy.ndarray hyp_seq_lens:
  :param numpy.ndarray truth_seq_lens:
  :rtype: numpy.ndarray
  """
  import numpy
  from returnn.util.basic import compute_bleu
  assert hypothesis.ndim == truth.ndim == 2 and hyp_seq_lens.ndim == truth_seq_lens.ndim == 1
  assert hypothesis.shape[0] == truth.shape[0] == hyp_seq_lens.shape[0] == truth_seq_lens.shape[0]
  return numpy.array([
    compute_bleu([truth[n, :truth_seq_lens[n]]], [hypothesis[n, :hyp_seq_lens[n]]])
    for n in range(hypothesis.shape[0])])


def bleu_score(hypothesis, truth, hyp_seq_lens, truth_seq_lens):
  """
  Calculates the BLEU score. See :func:`Util.compute_bleu`.
  This currently wraps a Python function and thus is not efficient.

  :param tf.Tensor hypothesis: (batch, max(hyp_seq_lens))
  :param tf.Tensor truth: (batch, max(truth_seq_lens))
  :param tf.Tensor hyp_seq_lens: (batch,)
  :param tf.Tensor truth_seq_lens: (batch,)
  :rtype: tf.Tensor
  :return: (batch,), float32
  """
  hypothesis = tf.convert_to_tensor(hypothesis)
  truth = tf.convert_to_tensor(truth)
  hyp_seq_lens = tf.convert_to_tensor(hyp_seq_lens)
  truth_seq_lens = tf.convert_to_tensor(truth_seq_lens)
  hypothesis.set_shape(tf.TensorShape((None, None)))
  truth.set_shape(tf.TensorShape((None, None)))
  hyp_seq_lens.set_shape(tf.TensorShape((None,)))
  truth_seq_lens.set_shape(tf.TensorShape((None,)))
  res = tf_compat.v1.py_func(
    _py_bleu_score, name="py_bleu_score", stateful=False,
    inp=[hypothesis, truth, hyp_seq_lens, truth_seq_lens], Tout=tf.float32)
  res.set_shape(tf.TensorShape((None,)))
  return res


def prod(ls):
  """
  :param list[T]|tuple[T]|numpy.ndarray|tf.Tensor ls:
  :rtype: T|int|float|tf.Tensor
  """
  if isinstance(ls, tf.Tensor):
    return tf.reduce_prod(ls, axis=0)
  from returnn.util.basic import prod as pure_prod
  if all([not isinstance(x, tf.Tensor) for x in ls]):
    return pure_prod(ls)  # not a tf.Tensor
  with tf.name_scope("prod"):
    return pure_prod(ls)  # tf.Tensor


class _DevMaxBytesInUse:
  """
  Like ``tf.contrib.contrib.memory_stats.MaxBytesInUse``.
  Reimplemented because ``tf.contrib`` got removed.

  See code for reference:
  https://github.com/tensorflow/tensorflow/blob/e210cb140a60a74d5e9ce3bf9ebedb21b4910f1c/tensorflow/contrib/memory_stats/kernels/memory_stats_ops.cc
  """

  _tf_mod = None

  @classmethod
  def get_mod(cls, verbose=False):
    """
    :param bool verbose:
    :return: module
    """
    if cls._tf_mod:
      return cls._tf_mod

    src_code = """
    #include "tensorflow/core/framework/common_shape_fns.h"
    #include "tensorflow/core/framework/op.h"
    #include "tensorflow/core/framework/op_kernel.h"

    using namespace tensorflow;

    REGISTER_OP("DevMaxBytesInUse")
      .Output("out: int64")
      .SetShapeFn(shape_inference::ScalarShape);

    class DevMaxBytesInUseOp : public OpKernel {
    public:
      explicit DevMaxBytesInUseOp(OpKernelConstruction* context) : OpKernel(context) {}

      void Compute(OpKernelContext* context) override {
        Allocator* allocator =
          context->device()->GetAllocator(AllocatorAttributes());
        OP_REQUIRES(context, allocator, errors::Internal("No device allocator available."));
        auto allocator_stats = allocator->GetStats();

        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(
          context, context->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<int64>()() = allocator_stats ? allocator_stats->peak_bytes_in_use : 0;
      }
    };

    REGISTER_KERNEL_BUILDER(Name("DevMaxBytesInUse").Device(DEVICE_CPU), DevMaxBytesInUseOp);
    REGISTER_KERNEL_BUILDER(Name("DevMaxBytesInUse").Device(DEVICE_GPU).HostMemory("out"), DevMaxBytesInUseOp);
    """

    compiler = OpCodeCompiler(
      base_name="DevMaxBytesInUse", code_version=1, code=src_code,
      is_cpp=True, use_cuda_if_available=True,
      # This would lead to a get_tf_list_local_devices call, which we might not want at this point.
      cuda_auto_min_compute_capability=False,
      verbose=verbose)
    tf_mod = compiler.load_tf_module()
    assert hasattr(tf_mod, "dev_max_bytes_in_use"), "content of mod: %r" % (dir(tf_mod),)
    cls._tf_mod = tf_mod
    return tf_mod

  @classmethod
  def max_bytes_in_use(cls):
    """
    :return: scalar string
    :rtype: tf.Tensor
    """
    return cls.get_mod().dev_max_bytes_in_use()


def mem_usage_for_dev(dev_name):
  """
  :param str dev_name: e.g. "/device:GPU:0" or "/job:localhost/replica:0/task:0/device:GPU:0"
  :return: int scalar, which is the peak memory usage in bytes of the given device
  :rtype: tf.Tensor

  This function will not create multiple nodes in the graph for multiple calls.
  Currently only works for GPU devices.
  """
  def get():
    """
    :rtype:  tf.Tensor
    """
    try:
      from tensorflow.contrib import memory_stats  # noqa
      # It's not so clear what BytesInUse returns. https://stackoverflow.com/questions/47903039/
      # Thus we always use MaxBytesInUse for now, although this is also not so nice.
      bytes_in_use = memory_stats.MaxBytesInUse
    except ImportError:  # TF 2
      bytes_in_use = _DevMaxBytesInUse.max_bytes_in_use
    # try:
    #   bytes_in_use = memory_stats.BytesInUse  # since TF 1.4.0
    # except AttributeError:
    #   bytes_in_use = memory_stats.MaxBytesInUse
    with tf.device(dev_name):
      return bytes_in_use()

  assert dev_name.startswith("/")  # e.g. "/cpu:0" or "/gpu:0"
  scope_name = dev_name[1:].replace(":", "").replace("/", "_")  # e.g. "cpu0" or "gpu0"
  return global_tensor(get, "mem_usage_%s" % scope_name)


def identity_with_debug_log(x, args, out, name="DebugLogOp"):
  """
  :param tf.Tensor x:
  :param dict[str,tf.Tensor|None] args:
  :param list[dict[str,numpy.ndarray]] out:
  :param str name:
  :return: x
  :rtype: tf.Tensor
  """
  from returnn.util.basic import dict_joined
  none_args = {k: None for (k, v) in args.items() if v is None}
  arg_keys = sorted([k for k in args.keys() if k not in none_args])

  # noinspection PyShadowingNames
  def py_func(x, *arg_values):
    """
    :param numpy.ndarray x:
    :rtype: numpy.ndarray
    """
    out.append(dict_joined(dict(zip(arg_keys, arg_values)), none_args))
    return x

  with tf.name_scope(name):
    y, = tf_compat.v1.py_func(
      py_func, [x] + [args[k] for k in arg_keys], [x.dtype], stateful=True)
    with tf.control_dependencies([y]):
      return tf.identity(x)


def add_check_numerics_ops(
      fetches=None, ignore_ops=None, use_check_numerics=True, debug_print_added_checks=True,
      name="add_check_numerics_ops"):
  """
  This is similar to :func:`tf.add_check_numerics_ops` and based on similar code.
  It adds some more logic and options.

  :param list[tf.Operation|tf.Tensor]|None fetches: in case this is given, will only look at these and dependent ops
  :param list[str] ignore_ops: e.g. ""
  :param bool use_check_numerics: if False, instead of :func:`tf.check_numerics`,
    it does the check manually (via :func:`tf.is_finite`) and in case there is inf/nan,
    it will also print the tensor (while `tf.check_numerics` does not print the tensor).
    Note that this can be about 50 times slower.
  :param bool debug_print_added_checks: prints info about each added check
  :param str name: op-name for the final tf.group
  :return: operation which performs all the checks
  :rtype: tf.Operation
  """
  if fetches is None:
    ops = tf_compat.v1.get_default_graph().get_operations()
  else:
    fetch_ops = [v.op if isinstance(v, tf.Tensor) else v for v in fetches]
    assert all([isinstance(op, tf.Operation) for op in fetch_ops])
    from returnn.extern import graph_editor
    ops = graph_editor.get_backward_walk_ops(fetch_ops, inclusive=True, control_inputs=True)
  if ignore_ops is None:
    # The checks could increase the memory usage a lot.
    # Ignore some common ops which should not be able to introduce inf/nan.
    ignore_ops = {
      "Add", "AddN", "Sum", "Mul", "MatMul", "Sub", "L2Loss", "Floor", "Neg", "UnsortedSegmentSum",
      "Switch", "Merge", "PreventGradient",
      "Select", "Maximum", "Minimum", "Abs", "Sign",
      "Const", "Identity", "Fill", "ZerosLike",
      "Reshape", "Tile", "ExpandDims", "ConcatV2", "Transpose",
      "Slice", "StridedSlice", "StridedSliceGrad", "Gather",
      "TruncatedNormal", "RandomUniform"}
  with tf.name_scope(name):
    check_op = []
    # This code relies on the ordering of ops in get_operations().
    # The producer of a tensor always comes before that tensor's consumer in
    # this list. This is true because get_operations() returns ops in the order
    # added, and an op can only be added after its inputs are added.
    for op in ops:
      assert isinstance(op, tf.Operation)
      if op.type in ignore_ops:
        continue
      # Frames from within a while-loop are partly broken.
      # https://github.com/tensorflow/tensorflow/issues/2211
      # noinspection PyProtectedMember
      if op._get_control_flow_context() != tf_compat.v1.get_default_graph()._get_control_flow_context():
        continue
      for output in op.outputs:
        if output.dtype not in [tf.float16, tf.float32, tf.float64]:
          continue
        message = op.name + ":" + str(output.value_index)
        with tf.control_dependencies(check_op):
          if debug_print_added_checks:
            print("add check for:", output, op.type)
          if use_check_numerics:
            check_op = [tf_compat.v1.check_numerics(output, message=message, name=op.name + "_check_numerics")]
          else:
            is_finite = tf.reduce_all(tf_compat.v1.is_finite(output))
            check_op = [tf.Assert(is_finite, [message, "Tensor had inf or nan values:", output])]
    return tf.group(*check_op)


def nested_get_shapes(x):
  """
  :param tf.Tensor|dict[str,tf.Tensor]|list[tf.Tensor]|object x: anything that nest supports
  :return: same structure as x, but tf.TensorShape for each tensor
  """
  if isinstance(x, tf.Tensor):
    return x.get_shape()
  if isinstance(x, (tuple, list)):
    from returnn.util.basic import make_seq_of_type
    return make_seq_of_type(type(x), [nested_get_shapes(v) for v in x])
  if isinstance(x, dict):
    return {k: nested_get_shapes(v) for (k, v) in x.items()}
  if isinstance(x, tf.TensorArray):
    return tf.TensorShape(())
  raise TypeError("invalid type %r of %r" % (type(x), x))


def get_current_control_flow_context():
  """
  :rtype: tensorflow.python.ops.control_flow_ops.ControlFlowContext|None
  """
  # noinspection PyProtectedMember
  return tf_compat.v1.get_default_graph()._get_control_flow_context()


def _get_control_flows(v, yield_none):
  """
  :param tf.Tensor|tf.Operation|int|float|None|list[tf.Tensor|tf.Operation|int|float] v:
  :param bool yield_none: the default context is None. specifies whether we should return that
  :return: yields control flow contexts
  :rtype: typing.Iterator[tensorflow.python.ops.control_flow_ops.ControlFlowContext|None]
  """
  import numpy
  from tensorflow.python.ops.control_flow_ops import ControlFlowContext
  if isinstance(v, (list, tuple)):
    for elem in v:
      for t in _get_control_flows(elem, yield_none=yield_none):
        yield t
    return
  if isinstance(v, (int, float, numpy.integer, type(None))):
    yield None
    return
  if isinstance(v, tf.Tensor):
    v = v.op
  assert isinstance(v, tf.Operation), "unexpected type %r" % type(v)
  # Control flow context will be set to the context of the loop or so, if there is one, otherwise None.
  # noinspection PyProtectedMember
  ctx = v._control_flow_context
  if ctx:
    assert isinstance(ctx, ControlFlowContext)
    if v.type in ["Exit", "RefExit"]:
      # We are just exiting this context, so return the outer context.
      ctx = ctx.outer_context
  if not yield_none and not ctx:
    return
  yield ctx


def has_control_flow_context(x):
  """
  :param tf.Tensor|tf.Operation|int|float|None|list[tf.Tensor|tf.Operation|int|float] x:
  :return: whether `x` has a control flow, i.e. is e.g. inside a while loop
  :rtype: bool
  """
  ops = list(_get_control_flows(x, yield_none=False))
  return len(ops) > 0


@contextlib.contextmanager
def same_control_flow_ctx(x):
  """
  Will use the same (flow) context as `x`.
  E.g. if `x` is a constant, it can be outside the loop,
  so we will yield a context which is not inside the loop.
  (This function was earlier called ``same_context``.)

  See also :func:`default_control_flow_ctx`.

  :param tf.Tensor|tf.Operation|int|float|None|list[tf.Tensor|tf.Operation|int|float] x:
  :return: yields context (via tf.control_dependencies)
  """
  ctxs = set(_get_control_flows(x, yield_none=True))
  if not ctxs:
    # There is no tensor given in `x` (just int or so).
    # Just stay in the current context.
    yield None
    return
  if len(ctxs) > 1 and None in ctxs:
    ctxs.remove(None)
  assert len(ctxs) == 1, "found multiple context: %r" % ctxs
  graph = tf_compat.v1.get_default_graph()
  ctx = list(ctxs)[0]
  # noinspection PyProtectedMember
  cur_ctx = graph._get_control_flow_context()
  if ctx == cur_ctx:
    yield ctx
    return
  if not ctx:  # None context, i.e. the default context.
    with tf.control_dependencies(None) as dep:  # this will reset the context
      yield dep
    return
  # noinspection PyProtectedMember
  graph._set_control_flow_context(ctx)
  yield ctx
  # noinspection PyProtectedMember
  graph._set_control_flow_context(cur_ctx)


def get_protobuf_fields(obj):
  """
  :param obj: protobuf object
  :rtype: dict[str]
  """
  return {k.name: v for (k, v) in obj.ListFields()}


def get_op_attrib_keys(op):
  """
  :param tf.Operation|tf.Tensor|tf.TensorArray op:
  :rtype: list[str]
  :return: list of attribs. op.get_attr(key) should work
  """
  if isinstance(op, tf.Tensor):
    op = op.op
  elif isinstance(op, tf.TensorArray):
    op = op.handle.op
  assert isinstance(op, tf.Operation)
  node_def_fields = get_protobuf_fields(op.node_def)
  attribs = node_def_fields.get("attr", {})
  return list(attribs.keys())


def get_op_input_names(op):
  """
  Also see: https://stackoverflow.com/questions/50723310/get-tensorflow-tf-operation-inputs-by-name

  :param tf.Operation op:
  :return: list of names with same len as op.inputs
  :rtype: list[str]
  """
  num_inputs = len(op.inputs)
  if op.op_def is None:
    # We could maybe do a lookup via the C++ API, similar to kernels_registered_for_op.
    # Or we could return None.
    # But this is simpler for now.
    names = []
  else:
    op_def_fields = get_protobuf_fields(op.op_def)
    args_pb = [get_protobuf_fields(a) for a in op_def_fields.get("input_arg", [])]
    names = [a["name"] for a in args_pb]
  assert len(names) <= num_inputs  # Not exactly sure why/when `<` can happen (except the unknown case above).
  names += ["?%i" % i for i in range(num_inputs - len(names))]
  assert len(names) == num_inputs
  return names


def get_op_inputs_by_name(op):
  """
  :param tf.Operation op:
  :return: dict input_name -> input
  :rtype: dict[str,tf.Tensor]
  """
  return dict(zip(get_op_input_names(op), op.inputs))


def tensor_array_is_dynamic_size(ta):
  """
  :param tf.TensorArray ta:
  :rtype: bool
  """
  return ta.handle.op.get_attr("dynamic_size")


def tensor_array_is_clear_after_read(ta):
  """
  :param tf.TensorArray ta:
  :rtype: bool
  """
  return ta.handle.op.get_attr("clear_after_read")


# noinspection PyProtectedMember
def tensor_array_element_shape(ta):
  """
  :param tf.TensorArray ta:
  :rtype: tf.TensorShape
  """
  if have_min_tf_version((1, 14)):
    assert isinstance(ta.element_shape, tf.TensorShape)
    return ta.element_shape
  else:
    # If it is know, _element_shape is a list with 1 entry, the element shape as tf.TensorShape.
    # Otherwise it is an empty list.
    assert isinstance(ta._element_shape, list)
    assert len(ta._element_shape) <= 1
    if ta._element_shape:
      assert isinstance(ta._element_shape[0], tf.TensorShape)
      return ta._element_shape[0]
    return tf.TensorShape(None)


def tensor_array_like(ta, **kwargs):
  """
  :param tf.TensorArray ta:
  :param kwargs: passed to tf.TensorArray constructor
  :return: another tensor array, just like ta
  :rtype: tf.TensorArray
  """
  # noinspection PyProtectedMember
  return tf.TensorArray(
    dtype=ta.dtype, size=ta.size(), dynamic_size=tensor_array_is_dynamic_size(ta),
    clear_after_read=tensor_array_is_clear_after_read(ta),
    infer_shape=ta._infer_shape, element_shape=tensor_array_element_shape(ta),
    **kwargs)


def _tensor_array_ref(ta):
  """
  :param tf.TensorArray ta:
  :rtype: tf.Tensor
  """
  if ta.handle is not None:
    return ta.handle
  if ta.flow is not None:
    return ta.flow
  raise Exception("Don't know how to handle TensorArray %r" % ta)


def tensor_array_stack(ta, start=0, stop=None, name="TensorArrayStack"):
  """
  Extends tf.TensorArray.stack by start/stop options.

  :param tf.TensorArray ta:
  :param int|tf.Tensor start:
  :param int|tf.Tensor|None stop:
  :param str name:
  :rtype: tf.Tensor
  """
  if start is 0 and stop is None:
    return ta.stack(name=name)
  with tf_compat.v1.colocate_with(_tensor_array_ref(ta)):
    with tf.name_scope(name):
      if stop is None:
        stop = ta.size()
      return ta.gather(tf.range(start, stop), name=name)


def _tensor_array_select_src_beams(ta, src_beams):
  """
  Currently this is a quite inefficient implementation.

  :param tf.TensorArray ta:
  :param tf.Tensor src_beams:
  :rtype: tf.TensorArray
  """
  x = ta.stack()  # (time,batch,...)
  x = swapaxes(x, 0, 1)  # (batch,time,...)
  x = select_src_beams(x, src_beams=src_beams)
  x = swapaxes(x, 0, 1)  # (time,batch,...)
  ta_new = tensor_array_like(ta)
  ta_new = ta_new.unstack(x)
  return ta_new


def beam_search(scores, beam_size, keep_beams=False,
                cheating_gold_targets=None, cheating_src_beam_idx=None, cheating_exclusive=True):
  """
  This is mostly a higher-level wrapper around :func:`tf.nn.top_k`.

  :param tf.Tensor scores: (batch,beam_in,dim). combined scores (i.e. base beam scores + new scores),
    dense over the dims, such that we have labels in [0,...,dim-1].
    These are supposed to be in +log space, although it just matters here that we take the maximum (or top-k).
  :param int|tf.Tensor beam_size:
  :param bool keep_beams: specifies that we keep the beam_in entries,
    i.e. we just expand, i.e. we just search on the dim. beam_size must be a multiple of beam_in.
  :param tf.Tensor|None cheating_gold_targets: (batch,), int32
  :param tf.Tensor|None cheating_src_beam_idx: (batch,), int32. If not given, assumes beam_in - 1. See code below.
  :param bool cheating_exclusive: make sure that the cheating target does not occur twice,
    i.e. no duplicates in search tree. This could have happened in our earlier implementation, or if this is disabled.
  :rtype: (tf.Tensor,tf.Tensor,tf.Tensor)
  :return: src_beams, labels, beam_scores.
    src_beams: (batch, beam) -> beam_in idx (int32),
    labels: (batch, beam) -> dim idx (int32),
    beam_scores: (batch, beam) -> beam score (float32).
  """
  batch_dim, beam_in, in_dim = get_shape(scores)
  if keep_beams:
    # It assumes that sorted=True in top_k, and the first entries in scores/labels are the best.
    scores = tf.reshape(scores, [batch_dim * beam_in, 1, in_dim])
    if cheating_gold_targets is not None:
      cheating_gold_targets = tile_transposed(cheating_gold_targets, axis=0, multiples=beam_in)
      if cheating_src_beam_idx is not None and cheating_src_beam_idx.shape.ndims > 0:
        cheating_src_beam_idx = tile_transposed(cheating_src_beam_idx, axis=0, multiples=beam_in)
    _, labels, beam_scores = beam_search(
      scores=scores, beam_size=beam_size // beam_in,
      cheating_gold_targets=cheating_gold_targets, cheating_src_beam_idx=cheating_src_beam_idx,
      cheating_exclusive=cheating_exclusive)
    src_beams = tf.zeros([batch_dim, beam_in, beam_size // beam_in], dtype=tf.int32)
    src_beams += tf.range(beam_in)[None, :, None]
    src_beams = tf.reshape(src_beams, [batch_dim, beam_size])
    labels = tf.reshape(labels, [batch_dim, beam_size])
    beam_scores = tf.reshape(beam_scores, [batch_dim, beam_size])
    return src_beams, labels, beam_scores
  # `tf.nn.top_k` is the core function performing our search.
  # We get scores/labels of shape (batch, beam) with indices in [0..beam_in*dim-1].
  top_k_size = beam_size
  if isinstance(in_dim, tf.Tensor) or isinstance(beam_size, tf.Tensor) or in_dim < beam_size:
    top_k_size = tf.minimum(beam_in * in_dim, top_k_size)
  gold_labels, gold_scores = None, None
  if cheating_gold_targets is not None:
    cheating_gold_targets = tf.clip_by_value(
      cheating_gold_targets, 0, in_dim - 1)  # safety, for invalid values...
    if cheating_src_beam_idx is None:
      # We also assume that the last choice also has the cheating target in the last beam index.
      cheating_src_beam_idx = beam_in - 1
    else:
      cheating_src_beam_idx = tf.clip_by_value(cheating_src_beam_idx, 0, beam_in - 1)  # safety
    gold_labels = cheating_src_beam_idx * in_dim + cheating_gold_targets  # (batch,)
    if cheating_src_beam_idx.shape.ndims == 0:
      gold_scores = scores[:, cheating_src_beam_idx]  # (batch,in_dim)
    else:
      assert cheating_src_beam_idx.shape.ndims == 1
      gold_scores = tf.gather_nd(scores, indices=nd_indices(cheating_src_beam_idx))  # (batch,in_dim)
    # Note: In case the seq ended, we assume that the gold_targets are all 0, such that we get the right score.
    gold_scores = tf.gather_nd(gold_scores, indices=nd_indices(cheating_gold_targets))  # (batch,)
    if cheating_exclusive:
      # Now mask this in the scores, such that top_k will not select the gold target,
      # because we later add it explicitly.
      cheating_src_beam_idx_bc = expand_multiple_dims(
        cheating_src_beam_idx, axes=[-1] * (2 - cheating_src_beam_idx.shape.ndims))  # [B|1,1]
      mask_beam = tf.equal(tf.range(beam_in)[None, :], cheating_src_beam_idx_bc)  # [B|1,beam_in]
      mask_dim = tf.equal(tf.range(in_dim)[None, :], cheating_gold_targets[:, None])  # [B,dim]
      mask = tf.logical_and(mask_beam[:, :, None], mask_dim[:, None, :])  # [B,beam_in,dim]
      scores = where_bc(mask, float("-inf"), scores)
  scores_flat = tf.reshape(scores, [batch_dim, beam_in * in_dim])  # (batch, beam_in * dim)
  # The main TF top_k call is here now:
  beam_scores, labels = tf.nn.top_k(scores_flat, k=top_k_size)
  if top_k_size is not beam_size:
    extra_shape = (batch_dim, beam_size - top_k_size)
    labels = tf.concat([labels, tf.zeros(extra_shape, dtype=labels.dtype)], axis=-1)
    beam_scores = tf.concat([beam_scores, tf.fill(extra_shape, float("-inf"))], axis=-1)
  if cheating_gold_targets is not None:
    # It assumes that sorted=True in top_k, and the last entries in scores/labels are the worst.
    # We replace them by the true labels.
    gold_labels_bc = tf.expand_dims(gold_labels, axis=1)  # (batch,1)
    labels = tf.concat([labels[:, :beam_size - 1], gold_labels_bc], axis=1)  # (batch,beam)
    gold_scores_bc = tf.expand_dims(gold_scores, axis=1)  # (batch,1)
    beam_scores = tf.concat([beam_scores[:, :beam_size - 1], gold_scores_bc], axis=1)  # (batch,beam)
  src_beams = labels // in_dim  # (batch, beam) -> beam_in idx
  labels = labels % in_dim  # (batch, beam) -> dim idx
  return src_beams, labels, beam_scores


def select_src_beams(x, src_beams, name="select_src_beams"):
  """
  :param tf.Tensor|tf.TensorArray|T x: (batch * src-beam, ...)
  :param tf.Tensor src_beams: (batch, beam) -> src-beam-idx
  :param str name:
  :return: (batch * beam, ...)
  :rtype: tf.Tensor|T
  """
  if isinstance(x, tf.TensorArray):
    return _tensor_array_select_src_beams(x, src_beams=src_beams)
  assert isinstance(x, tf.Tensor)
  assert isinstance(src_beams, tf.Tensor)
  with tf.name_scope(name):
    x_tshape = x.get_shape()
    src_beams.set_shape(tf.TensorShape([None, None]))
    src_beams_shape = get_shape(src_beams)
    batch_dim, beam_dim = src_beams_shape[0], src_beams_shape[1]
    x_ndim = x.get_shape().ndims
    assert x_ndim is not None
    x_shape = get_shape(x)
    x_shape_rem = [x_shape[i] for i in range(1, x_ndim)]
    src_beam_dim = x_shape[0] // batch_dim
    with reuse_name_scope_of_tensor(x, add_tensor_name=True, postfix="_reshape_split_beam"):
      x = tf.reshape(x, [batch_dim, src_beam_dim] + x_shape_rem)  # (batch, src-beam, ...)
    with reuse_name_scope_of_tensor(src_beams, add_tensor_name=True, postfix="_nd_indices"):
      indices = nd_indices(src_beams)  # (batch, beam, 2)
    x = tf.gather_nd(x, indices=indices)  # K=2, (batch, beam, ...)
    x = tf.reshape(x, [batch_dim * beam_dim] + x_shape_rem)
    x.set_shape(tf.TensorShape([None] + x_tshape.as_list()[1:]))
    return x


def filter_ended_scores(x, end_flags, batch_dim=None, dim=None, score_zero=0.0, score_rem=-1.e30):
  """
  This can e.g. used before tf.nn.top_k to let only one beam through for an ended hypothesis.
  Then, batch would also include the beam size, which does not matter here.

  :param tf.Tensor x: (batch, dim)
  :param tf.Tensor end_flags: (batch,)
  :param tf.Tensor|int|None batch_dim:
  :param tf.Tensor|int|None dim:
  :param float score_zero: x[..., 0] will have this score where end_flag is True
  :param float score_rem: x[..., 1:] will have this score where end_flag is False
  :return: filtered x, (batch, dim)
  :rtype: tf.Tensor
  """
  with tf.name_scope("filter_ended_scores"):
    end_flags.set_shape(tf.TensorShape([batch_dim if isinstance(batch_dim, int) else None]))
    end_flags = check_input_ndim(end_flags, 1)
    x = check_dim_equal(x, 0, end_flags, 0)
    x.set_shape(tf.TensorShape([
      batch_dim if isinstance(batch_dim, int) else None,
      dim if isinstance(dim, int) else None]))
    if batch_dim is None:
      batch_dim = tf.shape(end_flags)[0]
    if dim is None:
      dim = tf.shape(x)[-1]
    with same_control_flow_ctx(dim):  # force calculation outside loop if possible
      filter_score = tf.one_hot(
        0, dim, dtype=tf.float32, on_value=score_zero, off_value=score_rem)  # (dim,)
      filter_score.set_shape(tf.TensorShape([dim if isinstance(dim, int) else None]))
    with same_control_flow_ctx([dim, batch_dim]):  # force calculation outside loop if possible
      filter_score = expand_dims_unbroadcast(filter_score, axis=0, dim=batch_dim)  # (batch,dim)
      filter_score.set_shape(tf.TensorShape([
        batch_dim if isinstance(batch_dim, int) else None,
        dim if isinstance(dim, int) else None]))
    end_flags_bc = tf.expand_dims(end_flags, axis=1)
    x = where_bc(end_flags_bc, filter_score, x)
    x.set_shape(tf.TensorShape([
      batch_dim if isinstance(batch_dim, int) else None,
      dim if isinstance(dim, int) else None]))
    return x


def to_int32_64(x):
  """
  :param tf.Tensor x: dtype uint8, int8, int16, int32, int64
  :rtype: tf.Tensor
  :return: dtype int32 or int64
  """
  if x.dtype in [tf.int32, tf.int64]:
    return x
  assert x.dtype in [tf.uint8, tf.int8, tf.uint16, tf.int16]
  return tf.cast(x, tf.int32)


def to_float32(x):
  """
  :param tf.Tensor x:
  :return: x as float32
  :rtype: tf.Tensor
  """
  if x.dtype == tf.float32:
    return x
  if not hasattr(x, "cast_float32"):
    with reuse_name_scope_of_tensor(x):
      x_cast_float32 = tf.cast(x, dtype=tf.float32, name="cast_float32")
    x.cast_float32 = x_cast_float32
  return x.cast_float32


def batch_gather(x, indices, keepdims=False):
  """
  :param tf.Tensor x: (batch,dim,...)
  :param tf.Tensor indices: (batch,) -> [0..dim-1]
  :param bool keepdims:
  :return: x[batches,indices[batches]], (batch,...). or (batch,1,...) with keep_dims
  :rtype: tf.Tensor
  """
  with tf.name_scope('batch_gather'):
    idx_ext = nd_indices(to_int32_64(indices))
    y = tf.gather_nd(x, indices=idx_ext)
    if keepdims:
      y = tf.expand_dims(y, axis=1)
    return y


def unflatten_nd(x, nd_sizes, num_axes=None):
  """
  E.g. assume that for each x[b], we have an image flattened, i.e. of size width*height.
  Then nd_sizes[b] == (width, height) would provide the individual sizes.
  We return y such that y[b][i][j] == x[b][i * nd_sizes[b][0] + j].
  This is implemented for any number of axes.
  Kind of like the reverse of a ND version of flatten_with_seq_len_mask.

  :param tf.Tensor x: (B, T, <Ds>)
  :param tf.Tensor nd_sizes: (B, N = num_axes)
  :param int num_axes:
  :return: (B, T_1, ..., T_N, <Ds>), T_i == max(nd_sizes[:, i])
  :rtype: tf.Tensor
  """
  if num_axes is None:
    assert nd_sizes.shape.dims[-1].value
    num_axes = nd_sizes.shape.dims[-1].value
  assert num_axes >= 1
  nd_sizes.set_shape([None, num_axes])

  # indices for tf.gather_nd should be of shape (B, T_1, ..., T_N, 2).
  # Also see nd_indices.
  # Write in Python. Maybe convert to TF later...
  def py_get_indices(py_nd_sizes):
    """
    :param numpy.ndarray py_nd_sizes: (B, N)
    :return: (B, T_1, ..., T_N, 2)
    """
    import numpy
    assert py_nd_sizes.ndim == 2
    n_batch = py_nd_sizes.shape[0]
    num_axes_res = py_nd_sizes.shape[1]
    res = numpy.zeros([n_batch] + [numpy.max(py_nd_sizes[:, i]) for i in range(num_axes_res)] + [2], dtype="int32")
    for b in range(n_batch):
      idxs = numpy.arange(int(numpy.prod(py_nd_sizes[b])), dtype="int32")  # (t1*...*tN)
      idxs = idxs.reshape(py_nd_sizes[b])  # (t1,...,tN)
      res[b, ..., 0] = b
      res[tuple([b] + [slice(None, t) for t in py_nd_sizes[b]] + [1])] = idxs
    return res

  indices = tf_compat.v1.py_func(py_get_indices, [nd_sizes], tf.int32, stateful=False)
  indices.set_shape([None] + ([None] * num_axes) + [2])
  y = tf.gather_nd(x, indices)
  y.set_shape(indices.shape.as_list()[:-1] + x.shape.as_list()[2:])
  return y


def kernels_registered_for_op(op_name):
  """
  This just wraps the TF C++ function tensorflow::KernelsRegisteredForOp().

  :param str op_name: e.g. "Gather"
  :return: e.g. ["device='CPU'; ...", "device='GPU'; ...", ...]
  :rtype: list[str]
  """
  code = """
  #include <tensorflow/core/framework/op_kernel.h>
  using namespace tensorflow;

  extern "C" {
    typedef void (*set_string_callback) (const char* str, unsigned long size);

    void kernels_registered_for_op(const char* op_name, set_string_callback cb) {
      string s = KernelsRegisteredForOp(op_name);
      cb(s.c_str(), s.size());
    }
  };
  """
  native = TFNativeUtilCompiler(
    base_name="kernels_registered_for_op", code_version=1, code=code, is_cpp=True)
  lib = native.load_lib_ctypes()
  from ctypes import CFUNCTYPE, c_char_p, c_ulong
  set_string_callback_type = CFUNCTYPE(None, c_char_p, c_ulong)
  lib.kernels_registered_for_op.restype = None  # void
  lib.kernels_registered_for_op.argtypes = (c_char_p, set_string_callback_type)

  class Res:
    """
    Closure.
    """
    res = None

    # noinspection PyUnusedLocal
    @classmethod
    def callback(cls, string, size):
      """
      :param str string:
      :param int size:
      :rtype: None
      """
      cls.res = string

  cb = set_string_callback_type(Res.callback)
  lib.kernels_registered_for_op(str(op_name).encode("utf8"), cb)
  assert Res.res is not None
  s = Res.res.decode("utf8")
  lines = [line.strip() for line in s.splitlines()]
  if "<no registered kernels>" in lines:
    raise Exception("Op %r is unknown." % op_name)
  lines = [line for line in lines if line]
  return lines


def supported_devices_for_op(op_name):
  """
  :param str op_name:
  :return: list of devices, e.g. ["CPU", "GPU"]
  :rtype: list[str]
  """
  import re
  kernels_info = kernels_registered_for_op(op_name)
  devs_matches = [re.match("device='(.+)'.*", s) for s in kernels_info]
  if None in devs_matches:
    raise Exception("Got invalid output: %r" % kernels_info)
  devs = [m.group(1) for m in devs_matches]
  return list(sorted(set(devs)))


def find_unsupported_devices_in_graph(graph, dev_name, ignore=None):
  """
  :param tf.Graph graph:
  :param str dev_name: e.g. "GPU"
  :param list[str]|None ignore: list of op-names to ignore, e.g. ["ScalarSummary"] etc. If None, will use defaults.
  :rtype: list[tf.Operation]
  """
  if ignore is None:
    ignore = {"Assert", "ScalarSummary", "MergeSummary", "SaveV2", "RestoreV2"}
  ops = []
  for op in graph.get_operations():
    assert isinstance(op, tf.Operation)
    if op.type in ignore:
      continue
    if dev_name not in supported_devices_for_op(op.type):
      ops.append(op)
  return ops


class _DeviceAttrMod:

  _tf_mod = None

  @classmethod
  def get_mod(cls, verbose=False):
    """
    :param bool verbose:
    :return: module
    """
    if cls._tf_mod:
      return cls._tf_mod

    src_code = """
    #include "tensorflow/core/framework/common_shape_fns.h"
    #include "tensorflow/core/framework/op.h"
    #include "tensorflow/core/framework/op_kernel.h"
    #include "tensorflow/core/framework/device_attributes.pb.h"
    #include "tensorflow/core/public/version.h"

    using namespace tensorflow;

    #if (TF_MAJOR_VERSION < 2) || (TF_MAJOR_VERSION == 2 && TF_MINOR_VERSION < 2)
    using tstring = std::string;
    #else
    using tstring = tensorflow::tstring;
    #endif

    REGISTER_OP("GetDeviceAttr")
      .Output("out: string")
      .SetShapeFn(shape_inference::ScalarShape);

    class GetDeviceAttrOp : public OpKernel {
    public:
      explicit GetDeviceAttrOp(OpKernelConstruction* context) : OpKernel(context) {}

      void Compute(OpKernelContext* context) override {
        const DeviceAttributes& attribs = context->device()->attributes();
        Tensor* output_tensor = nullptr;
        OP_REQUIRES_OK(
            context, context->allocate_output(0, TensorShape({}), &output_tensor));
        output_tensor->scalar<::tstring>()() = attribs.physical_device_desc();
      }
    };

    REGISTER_KERNEL_BUILDER(Name("GetDeviceAttr").Device(DEVICE_CPU), GetDeviceAttrOp);
    REGISTER_KERNEL_BUILDER(Name("GetDeviceAttr").Device(DEVICE_GPU).HostMemory("out"), GetDeviceAttrOp);
    """

    compiler = OpCodeCompiler(
      base_name="GetDeviceAttr", code_version=1, code=src_code,
      is_cpp=True, use_cuda_if_available=True,
      # This would lead to a get_tf_list_local_devices call, which we might not want at this point.
      cuda_auto_min_compute_capability=False,
      verbose=verbose)
    tf_mod = compiler.load_tf_module()
    assert hasattr(tf_mod, "get_device_attr"), "content of mod: %r" % (dir(tf_mod),)
    cls._tf_mod = tf_mod
    return tf_mod

  @classmethod
  def get_device_attr(cls):
    """
    :return: scalar string
    :rtype: tf.Tensor
    """
    return cls.get_mod().get_device_attr()


def get_device_attr(dev):
  """
  :param str dev: eg. "/device:GPU:0", or any argument for :func:`tf.device`
  :return: scalar string, eg. b'device: 2, name: GeForce GTX 1080 Ti, pci bus id: 0000:82:00.0, compute capability: 6.1'
  :rtype: tf.Tensor
  """
  # `dev` should be one of the available devices, e.g. via session.list_devices(), and exactly that,
  # i.e. if that is XLA, it must be used as-is.
  with tf.device(dev):
    return _DeviceAttrMod.get_device_attr()


def print_graph_output(fetches, file=sys.stdout, max_depth=None):
  """
  :param tf.Operation|tf.Tensor|list[tf.Tensor|tf.Operation] fetches:
  :param typing.IO[str]|io.TextIOBase|io.StringIO file:
  :param int|None max_depth:
  """
  if not isinstance(fetches, (list, tuple)):
    fetches = [fetches]
  visited = set()

  def p(op, prefix="", indent=0):
    """
    :param tf.Operation|tf.Tensor op:
    :param str prefix:
    :param int indent:
    """
    postfix = ""
    if isinstance(op, tf.Tensor):
      postfix = " [%i], shape %s" % (op.value_index, op.shape.as_list() if op.shape.ndims is not None else "<unknown>")
      op = op.op
    assert isinstance(op, tf.Operation)
    print("%s%s%r%s" % ("  " * indent, prefix, op, postfix), file=file)
    if indent:
      if op in visited:
        return
    visited.add(op)
    if max_depth is not None and indent > max_depth:
      return
    if op.inputs:
      input_names = get_op_input_names(op)
      for i, x in enumerate(op.inputs):
        assert isinstance(x, tf.Tensor)
        p(x, prefix="inputs[%i] %r: " % (i, input_names[i]), indent=indent + 1)
    if op.control_inputs:
      for i, x in enumerate(op.control_inputs):
        p(x, prefix="control_inputs[%i]: " % (i,), indent=indent + 1)

  for fetch in fetches:
    p(fetch, prefix="fetch: ")


def var_handle_or_ref(var):
  """
  :param tf.Variable|tensorflow.python.ops.resource_variable_ops.ResourceVariable var:
  :rtype: tf.Tensor
  """
  from tensorflow.python.ops.resource_variable_ops import ResourceVariable
  if isinstance(var, ResourceVariable):
    return var.handle
  if isinstance(var, tf_compat.v1.Variable):
    # noinspection PyProtectedMember
    return var._ref()
  raise TypeError("invalid type for var %r" % var)


def find_ops_with_tensor_input(tensors, fetches=None, graph=None):
  """
  :param tf.Tensor|tf.Variable|list[tf.Tensor] tensors:
  :param tf.Operation|tf.Tensor|list[tf.Operation|tf.Tensor]|None fetches:
  :param tf.Graph|None graph:
  :return: list of ops
  :rtype: list[tf.Operation]
  """
  if isinstance(tensors, tf.Variable):
    from tensorflow.python.ops.resource_variable_ops import ResourceVariable
    if isinstance(tensors, ResourceVariable):
      # To keep the behavior more consistent to the old-style ref-variable,
      # we resolve all ReadVariableOp on the var resource handle,
      # and also remove VarIsInitializedOp.
      # If you do not want this, pass the var.handle directly to find_ops_with_tensor_input.
      ops_ = find_ops_with_tensor_input(tensors.handle, fetches=fetches, graph=graph)
      ops = []
      for op in ops_:
        if op.type == "ReadVariableOp":
          ops += find_ops_with_tensor_input(op.outputs[0], fetches=fetches, graph=graph)
        elif op.type == "VarIsInitializedOp":
          pass  # remove this
        else:
          ops.append(op)
      return ops
    else:
      # noinspection PyProtectedMember
      tensors = [tensors._ref(), tensors.value()]
  if isinstance(tensors, tf.Tensor):
    tensors = [tensors]
  assert all([isinstance(x, tf.Tensor) for x in tensors])
  assert len(tensors) > 0
  if fetches is not None:
    if isinstance(fetches, (tf.Operation, tf.Tensor)):
      fetches = [fetches]
    fetches = [x.op if isinstance(x, tf.Tensor) else x for x in fetches]
    assert all([isinstance(x, tf.Operation) for x in fetches])
    from returnn.extern import graph_editor
    all_ops = graph_editor.get_backward_walk_ops(
      fetches, inclusive=True, control_inputs=True, stop_at_ts=tensors)
  else:
    if graph is None:
      graph = tensors[0].graph
    all_ops = graph.get_operations()
  ops = []
  for op in all_ops:
    assert isinstance(op, tf.Operation)
    if any([x.op == op for x in tensors]):
      continue
    for x in tensors:
      if x in op.inputs:
        ops.append(op)
        break
  return ops


def find_ops_path_output_to_input(tensors, fetches):
  """
  Searches backwards like in :func:`extern.graph_editor.get_backward_walk_ops`
  and then returns a found traceback, if there is one.

  :param tf.Tensor|tf.Variable|list[tf.Tensor] tensors: input
  :param tf.Operation|tf.Tensor|list[tf.Operation|tf.Tensor] fetches: output
  :return: list of ops, input to output
  :rtype: list[tf.Operation]|None
  """
  if isinstance(tensors, tf.Variable):
    # noinspection PyProtectedMember
    tensors = [tensors._ref(), tensors.value()]
  if isinstance(tensors, tf.Tensor):
    tensors = [tensors]
  assert isinstance(tensors, (list, tuple, set))
  assert all([isinstance(x, tf.Tensor) for x in tensors])
  assert len(tensors) > 0
  # Operate on tensor names to not have problems with different object instances.
  tensor_names = [x.name for x in tensors]
  tensor_names = set(tensor_names)  # make unique
  if isinstance(fetches, (tf.Operation, tf.Tensor)):
    fetches = [fetches]
  fetches = [x.op if isinstance(x, (tf.Tensor, tf.Variable)) else x for x in fetches]
  fetches = set(fetches)
  for x in fetches:
    assert isinstance(x, tf.Operation)

  back_pointers = {}  # type: typing.Dict[tf.Operation,tf.Operation]
  cur_wave = fetches  # type: typing.Set[tf.Operation]
  visited = set()  # type: typing.Set[tf.Operation]

  while cur_wave:
    next_wave = set()  # type: typing.Set[tf.Operation]
    for op in cur_wave:
      visited.add(op)
      for x in op.inputs:
        if x.name in tensor_names:  # found a path
          result = [op]
          while op not in fetches:
            op = back_pointers[op]
            result.append(op)
          return result
      for next_op in [x.op for x in op.inputs] + list(op.control_inputs):
        assert isinstance(next_op, tf.Operation)
        if next_op in visited or next_op in next_wave:
          continue
        next_wave.add(next_op)
        back_pointers[next_op] = op
    cur_wave = next_wave

  return None


def get_var_update_ops(var, fetches=None):
  """
  :param tf.Variable var:
  :param tf.Operation|tf.Tensor|list[tf.Operation|tf.Tensor]|None fetches: e.g. the Optimizer.minimize() op
  :return: list of ops that update var; currently expected to be of length 1
  :rtype: list[tf.Operation]
  """
  ops = find_ops_with_tensor_input(var, fetches=fetches)
  assert ops, "we expect that var %r is used somewhere" % var
  apply_op_names = {
    "Assign", "AssignAdd", "AssignSub", "ScatterSub",
    # This list might need to be extended for your need...
    "ApplyAdam", "ApplyGradientDescent", "ApplyAdadelta", "ApplyAdagrad", "ApplyAdagradDA",
    "ApplyCenteredRMSProp", "ApplyFtrl", "ApplyMomentum", "ApplyProximalAdagrad",
    "ApplyProximalGradientDescent", "ApplyRMSProp"}
  apply_op_names.update(["Sparse%s" % name for name in apply_op_names])
  apply_op_names.update(["Resource%s" % name for name in apply_op_names])
  ops_ = [op for op in ops if op.type in apply_op_names]
  # Maybe we may loosen this restriction to be >= 1 or so later on.
  assert len(ops_) == 1, "we expect to have exactly one Assign/Apply op in %r" % (ops,)
  return ops_


def get_variable_value_copy_before_update_ops(var, update_ops):
  """
  :param tf.Variable var:
  :param list[tf.Operation] update_ops:
  :return: var value before any of the update_ops are executed
  :rtype: tf.Tensor
  """
  with tf.name_scope("get_variable_value_copy_before_update_ops"):
    with tf.control_dependencies(None):
      v_val = copy_tensor(var.value())
      for op in update_ops:
        add_control_input(op, v_val.op)  # Do it before op is executed.
    return v_val


def get_variable_grad_from_update_ops(var, update_ops):
  """
  :param tf.Variable var:
  :param list[tf.Operation] update_ops: via :func:`get_var_update_ops`
  :return: grad of loss w.r.t. var, as it is used in the update_ops, e.g. via ApplyAdam or ApplyGradientDescent
    (not all kind of updates are supported currently).
    If the gradient is sparse, it will return a tf.IndexedSlices.
  :rtype: tf.Tensor|tf.IndexedSlices
  """
  assert len(update_ops) == 1
  op = update_ops[0]
  op_inputs = get_op_inputs_by_name(op)
  if op.type == "ScatterSub":  # e.g. sparse grad with GradientDescentOptimizer
    # noinspection PyProtectedMember
    assert op_inputs["ref"] == var_handle_or_ref(var)
    indices = op_inputs["indices"]
    delta = op_inputs["updates"]
    assert delta.op.type == "Mul"  # mul with learning rate
    grad = delta.op.inputs[0]
    assert "gradients" in grad.name
    return tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.convert_to_tensor(get_shape(var)))
  if op.type == "AssignSub":
    op_name_prefix = os.path.dirname(op.name) + "/"
    # noinspection PyProtectedMember
    assert op_inputs["ref"] == var_handle_or_ref(var)
    # Case for sparse update in Adam:
    # m_scaled_g_values = grad * (1 - beta1_t)
    # m_t = scatter_add(m, indices, m_scaled_g_values)
    from returnn.extern import graph_editor
    all_ops = graph_editor.get_backward_walk_ops(update_ops, inclusive=True, control_inputs=True)
    all_ops = [x for x in all_ops if x.name.startswith(op_name_prefix)]
    scatter_add_ops = [x for x in all_ops if x.type == "ScatterAdd"]
    # print(scatter_add_ops, [x.inputs[0] for x in scatter_add_ops])
    indices = scatter_add_ops[0].inputs[1]
    m_scaled_g_values = scatter_add_ops[0].inputs[2]
    assert m_scaled_g_values.op.type == "Mul"
    grad = m_scaled_g_values.op.inputs[0]
    # We should either have the gradient directly now, or an UnsortedSegmentSum, via _apply_sparse_duplicate_indices.
    assert "gradients" in grad.name or grad.op.type == "UnsortedSegmentSum"
    return tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.convert_to_tensor(get_shape(var)))
  assert "var" in op_inputs
  # noinspection PyProtectedMember
  assert op_inputs["var"] == var_handle_or_ref(var)
  if "grad" in op_inputs:  # e.g. ApplyAdam
    grad = op_inputs["grad"]
  elif "delta" in op_inputs:  # e.g. ApplyGradientDescent
    grad = op_inputs["delta"]
  else:
    raise Exception("Don't know how to get grad from op %r with inputs %r." % (op, op_inputs))
  if op.type.startswith("SparseApply"):  # e.g. SparseApplyMomentum
    # We should either have the gradient directly now, or an UnsortedSegmentSum, via _apply_sparse_duplicate_indices.
    assert "gradients" in grad.name or grad.op.type == "UnsortedSegmentSum"
    indices = op_inputs["indices"]
    return tf.IndexedSlices(values=grad, indices=indices, dense_shape=tf.convert_to_tensor(get_shape(var)))
  assert "gradients" in grad.name
  return grad


def add_control_input(op, control_input):
  """
  :param tf.Operation op:
  :param tf.Operation control_input:
  """
  assert isinstance(op, tf.Operation)
  assert isinstance(control_input, tf.Operation)
  if hasattr(op, "_add_control_input"):  # some later TF version
    # noinspection PyProtectedMember
    op._add_control_input(control_input)
    return
  # Fallback. I think I have seen this in OpenAI code.
  # noinspection PyProtectedMember
  op._control_inputs.append(control_input)
  # noinspection PyProtectedMember
  op._recompute_node_def()


def vocab_idx_to_vocab_string(labels, vocab):
  """
  Just does a lookup on vocab.

  :param tf.Tensor labels: (batch,max_len), or any, int32, indices in vocab
  :param tf.Tensor vocab: (vocab_size,), string
  :return: (batch,max_len), or any, like labels, string
  :rtype: tf.Tensor
  """
  return tf.gather(params=vocab, indices=labels, axis=0)


def vocab_idx_repr(labels, data):
  """
  :param tf.Tensor labels: int32, indices in vocab
  :param Data data: might have vocab
  :return: string or int32, shape as labels, or maybe without last axis
  :rtype: tf.Tensor
  """
  if data.vocab:
    vocab = get_shared_vocab(data.vocab.labels)
    return vocab_idx_to_vocab_string(labels, vocab)
  if data.dim == 255:
    if labels.shape.ndims >= 2:  # currently encode_raw also joins the strings in last axis
      return encode_raw(labels)
  return labels


def string_merge(strings, seq_lens, separator=" "):
  """
  Also see TFEngine.Engine.search().

  :param tf.Tensor strings: (batch,max_len)
  :param tf.Tensor seq_lens: (batch,)
  :param str|tf.Tensor separator: string
  :return: (batch,), string
  :rtype: tf.Tensor
  """
  input_shape = tf.shape(strings)
  n_batch, max_len = input_shape[0], input_shape[1]
  strings = tf.reshape(strings, [n_batch, max_len, 1])
  seps = tf.zeros_like(strings, dtype=tf.string) + separator
  strings = tf.concat([strings, seps], axis=2)  # (batch,max_len,2)
  strings = tf.reshape(strings, [n_batch, max_len * 2])
  mask = tf.sequence_mask(seq_lens * 2 - 1, maxlen=max_len * 2)  # (batch,)
  strings = tf.where(mask, strings, tf.zeros_like(strings, dtype=tf.string))  # (batch,max_len*2)
  strings = tf_compat.v1.reduce_join(strings, axis=1)
  return strings


def string_replace(strings, old, new, count=-1):
  """
  Like str.replace.

  :param tf.Tensor strings: (batch,), string
  :param tf.Tensor|str old:
  :param tf.Tensor|str new:
  :param tf.Tensor|int count:
  :return: (batch,), string
  :rtype: tf.Tensor
  """
  import numpy

  # noinspection PyShadowingNames
  def str_replace(strings, old, new, count):
    """
    :param numpy.ndarray|bytes strings:
    :param bytes old:
    :param bytes new:
    :param numpy.int32 count:
    :rtype: numpy.ndarray|bytes
    """
    assert isinstance(strings, (numpy.ndarray, bytes)), "strings is %r" % (strings,)
    assert isinstance(old, bytes), "old is %r" % (new,)
    assert isinstance(new, bytes), "new is %r" % (new,)
    assert isinstance(count, numpy.int32), "count is %r" % (count,)
    if isinstance(strings, numpy.ndarray):
      return numpy.array(
        [s.replace(old, new, count) for s in strings.flatten()], dtype=strings.dtype).reshape(strings.shape)
    else:
      return strings.replace(old, new, count)

  res, = tf_compat.v1.py_func(
    str_replace,
    [tf.cast(strings, tf.string),
     tf.cast(old, tf.string),
     tf.cast(new, tf.string),
     tf.cast(count, tf.int32)],
    [tf.string],
    stateful=False,
    name="string_replace")
  assert isinstance(res, tf.Tensor)
  res.set_shape(strings.get_shape())
  return res


def bpe_merge(strings):
  """
  :param tf.Tensor strings: (batch,), string
  :return: (batch,), string. strings after BPE merging
  :rtype: tf.Tensor
  """
  return string_replace(strings, old="@@ ", new="")


def words_split(strings):
  """
  Basically just tf.string_split with delimiter=" ".

  :param tf.Tensor strings: (batch,), string
  :return: sparse tensor of shape (batch,max_len), string
  :rtype: tf.SparseTensor
  """
  return tf_compat.v1.string_split(strings)


def get_sparse_tensor_length(x):
  """
  :param tf.SparseTensor x: of shape prefix + (max_len,), where prefix can be anything, e.g. prefix=(batch,)
  :return: shape prefix, int64
  :rtype: tf.Tensor
  """
  # x.indices is of shape (N,R), where R==rank(x), and each x.indices[i] is the index entry.
  # So, x.indices[i, -1] is the position.
  # We just do it in a simple way here.
  mask = tf_compat.v1.sparse_to_dense(
    x.indices, output_shape=x.dense_shape, sparse_values=tf.ones_like(x.values, dtype=tf.int64))  # prefix+(max_len,)
  return tf.reduce_sum(mask, axis=-1)  # prefix


def string_words_calc_wer(hyps, refs):
  """
  :param tf.Tensor hyps: (batch,)
  :param tf.Tensor refs: (batch,)
  :return: (WER (batch,), num ref words (batch,))
  :rtype: (tf.Tensor, tf.Tensor)
  """
  refs.set_shape(hyps.get_shape())
  hyps.set_shape(refs.get_shape())
  hyps_sparse = words_split(hyps)
  refs_sparse = words_split(refs)
  wer = tf.edit_distance(hypothesis=hyps_sparse, truth=refs_sparse, normalize=False)
  wer.set_shape(hyps.get_shape())
  wer = tf.cast(wer, tf.int64)  # no normalization, should be an integer
  return wer, get_sparse_tensor_length(refs_sparse)


def py_print(pass_through_value, print_args, message=None, summarize=None, first_n=None, name="py_print"):
  """
  Like :func:`tf.Print`, but prints to Python stdout.
  Also see :func:`tf.print`, which however also does not print to Python stdout.

  :param tf.Tensor|int|float pass_through_value: will return tf.identity of this, but with side effect of printing
  :param list[str|tf.Tensor] print_args:
  :param str|None message: A string, prefix of the error message.
  :param int|None summarize: Only print this many entries of each tensor. If None, then a
    maximum of 3 elements are printed per input tensor.
  :param int|None first_n: Only log `first_n` number of times. Negative numbers log always; this is the default.
  :param str name:
  :return: tf.identity(pass_through_value) with side effect of printing
  :rtype: tf.Tensor
  """
  import numpy
  from numpy.lib import NumpyVersion
  if summarize is None:
    summarize = 3
  if first_n is None:
    first_n = -1
  np_a2s_kwargs = dict(formatter={"int": str, "object": bytes.decode}, edgeitems=summarize, threshold=summarize)
  if NumpyVersion(numpy.__version__) <= NumpyVersion('1.11.3'):
    # Seems some older Numpy versions don't support this.
    del np_a2s_kwargs["edgeitems"]

  class Counter:
    """
    Closure.
    """
    count = 0

  def _py_print(*_print_args):
    Counter.count += 1
    if first_n > 0 and first_n > Counter.count:
      return False
    next_item_add_new_line = False
    s = ""
    if message:
      s += message
    for arg in _print_args:
      # Try to keep somewhat consistent with the tf.Print output.
      if isinstance(arg, bytes):
        arg_s = "[%s]" % arg.decode("utf8")
      elif isinstance(arg, numpy.ndarray):
        if arg.size == 0:
          arg_s = "[]"
        else:
          arg_s = numpy.array2string(arg, **np_a2s_kwargs)
      else:
        arg_s = "[%r]" % arg
      if "\n" in arg_s:
        next_item_add_new_line = True
        s += "\n"
        s += arg_s
      else:
        if next_item_add_new_line:
          next_item_add_new_line = False
          s += "\n"
        s += arg_s
    try:
      print(s)
      return True
    except BrokenPipeError:  # be silent about those. probably only at exit
      return False

  with tf.name_scope(name):
    print_op = tf_compat.v1.py_func(_py_print, print_args, tf.bool, name=name)
    with tf.control_dependencies([print_op]):
      return tf.identity(pass_through_value)


def get_positional_encoding(num_channels, length=None, position=None, min_timescale=1.0, max_timescale=1.0e4):
  """
  Gets a bunch of sinusoids of different frequencies.

  Each channel of the input Tensor is incremented by a sinusoid of a different
  frequency and phase.

  This allows attention to learn to use absolute and relative positions.
  Timing signals should be added to some precursors of both the query and the
  memory inputs to attention.

  The use of relative position is possible because sin(x+y) and cos(x+y) can be
  expressed in terms of y, sin(x) and cos(x).

  In particular, we use a geometric sequence of timescales starting with
  min_timescale and ending with max_timescale.  The number of different
  timescales is equal to channels / 2. For each timescale, we
  generate the two sinusoidal signals sin(timestep/timescale) and
  cos(timestep/timescale).  All of these sinusoids are concatenated in
  the channels dimension.

  The code is adapted from Tensor2Tensor get_timing_signal_1d (https://github.com/tensorflow/tensor2tensor).

  :param int num_channels: scalar, size of timing embeddings to create. The number of
    different timescales is equal to channels / 2.
  :param tf.Tensor|None length: scalar, length of timing signal sequence.
  :param tf.Tensor|None position: could be provided directly. int32. Can have any shape.
  :param float min_timescale: a float.
  :param float max_timescale: a float.
  :return: a Tensor of timing signals of shape (length, channels) or (batch, length, channels).
  :rtype: tf.Tensor
  """
  import math
  if position is None:
    assert length is not None
    position = tf.range(length)
  else:
    assert length is None
  position = tf.cast(position, tf.float32)
  num_timescales = num_channels // 2
  log_timescale_increment = (
    math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales - 1)))
  inv_timescales = min_timescale * tf.exp(
    tf.cast(tf.range(num_timescales), tf.float32) * -log_timescale_increment)
  scale = tf.reshape(inv_timescales, [1] * len(position.shape) + [num_timescales])  # Usually (1, D//2) or (1, 1, D//2).
  scaled_time = tf.expand_dims(position, -1) * scale
  signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1)
  # (length, channels) or (batch, length, channels).
  signal = tf.pad(signal, [[0, 0]] * len(position.shape) + [[0, num_channels % 2]])
  return signal


def get_linear_alignment_out_to_in_indices(input_lens, output_lens, pad_value=0):
  """
  :param tf.Tensor|list[int] input_lens: [B]
  :param tf.Tensor|list[int] output_lens: [B]
  :param int pad_value:
  :return: [B,outT], mapping to input positions [0..input_len-1].
    Examples:
      * input_len=7, output_len=3, resulting indices [1,3,5].
      * input_len=3, output_len=3, resulting indices [0,1,2].
      * input_len=2, output_len=4, resulting indices [0,0,1,1].
  :rtype: tf.Tensor
  """
  input_lens = tf.convert_to_tensor(input_lens)
  output_lens = tf.convert_to_tensor(output_lens)
  out_time_dim = tf.reduce_max(output_lens)
  out_times = tf.expand_dims(tf.range(out_time_dim), axis=0)  # (1,outT)
  # Add 1, such that the first index is not necessarily at 0.
  idxs = tf.cast(out_times + 1, tf.float32)  # (1,outT)

  # We want: x[0] == (input_len+1)/(output_len+1), x[output_len] == input_len + 1.
  factors = (
    tf.maximum(tf.cast(input_lens + 1, tf.float32), 0.0) /
    tf.maximum(tf.cast(output_lens + 1, tf.float32), 1.0))  # (B,)
  factors = tf.expand_dims(factors, axis=1)  # (B,1)
  idxs = idxs * factors  # (B,outT)
  idxs = tf.cast(tf.round(idxs), tf.int32)
  idxs = idxs - 1  # correct from above
  # For safety, clip.
  idxs = tf.clip_by_value(idxs, 0, expand_dims_unbroadcast(input_lens - 1, axis=1, dim=out_time_dim))  # (B,outT)

  # Although not required for the padding area, it is helpful to have 0 there.
  # (E.g. cheating gold targets in the rec layer requires this.)
  idxs = where_bc(tf.less(out_times, tf.expand_dims(output_lens, axis=1)), idxs, pad_value)
  return idxs


def get_rnnt_linear_aligned_output(
  input_lens, targets, target_lens, blank_label_idx, pad_value=0,
  targets_consume_time=False):
  """
  RNN-T (https://arxiv.org/abs/1211.3711) has an output length of input_lens + target_lens.
  Here we create a linear alignment.
  Examples: (B is blank.)
    * input_len=4, targets=[a,b,c] (len 3), output=[B,a,B,b,B,c,B] (len 7).
    * input_len=0, targets=[a,b,c] (len 3), output=[a,b,c] (len 3).
    * input_len=4, targets=[a] (len 1), output=[B,B,a,B,B] (len 5).
    * input_len=3, targets=[a,b] (len 2), output=[B,a,B,b,B] (len 5)

  :param tf.Tensor|list[int] input_lens: [B], int32. the input (or encoder) lengths
  :param tf.Tensor|list[list[int]] targets: [B,targetT], int32
  :param tf.Tensor|list[int] target_lens: [B], int32. the targets length
  :param int blank_label_idx:
  :param int pad_value:
  :param bool targets_consume_time: In the standard RNN-T, the target labels do not consume a time frame.
    That is why the RNN-T label output length is input_lens + target_lens.
    In RNA (https://www.isca-speech.org/archive/Interspeech_2017/abstracts/1705.html),
    each target label consumes a time frame, thus the label output length is just input_lens.
  :return: output [B,outT], output_lens [B]. The output is basically the target filled with blank in between.
  :rtype: (tf.Tensor,tf.Tensor)
  """
  input_lens = tf.convert_to_tensor(input_lens)
  targets = tf.convert_to_tensor(targets)
  target_lens = tf.convert_to_tensor(target_lens)
  if targets_consume_time:
    out_lens = input_lens
  else:
    out_lens = input_lens + target_lens
  out_time_dim = tf.reduce_max(out_lens)
  batch_dim = tf.shape(out_lens)[0]
  target_time_dim = tf.shape(targets)[1]

  # Build idx to scatter_nd of out_times.
  idxs = get_linear_alignment_out_to_in_indices(
    input_lens=tf.maximum(out_lens, target_lens), output_lens=target_lens, pad_value=-1)  # (B,targetT)
  idxs = idxs + 1  # make valid pad_value
  idxs = where_bc(tf.greater(idxs, out_lens[:, None]), 0, idxs)  # fix if target_len > out_len

  target_times = tf.expand_dims(tf.range(target_time_dim), axis=0)  # (1,targetT)
  out_values = tf.concat([tf.tile([[blank_label_idx]], [batch_dim, 1]), targets], axis=1)  # (B,targetT+1)
  # Build idx to gather_nd of out_values. Values [0..targetT].
  idxs2 = tf.scatter_nd(
    indices=nd_indices(idxs),
    updates=tf.tile(target_times + 1, [batch_dim, 1]),
    shape=[batch_dim, out_time_dim + 1])  # (B,outT+1)
  idxs2 = idxs2[:, 1:]  # (B,outT). remove the scratch area.
  out = tf.gather_nd(out_values, indices=nd_indices(idxs2))  # (B,outT)
  # Although not required for the padding area, it is helpful to have 0 there.
  # (E.g. cheating gold targets in the rec layer requires this.)
  out = where_bc(tf.less(tf.range(out_time_dim)[None, :], out_lens[:, None]), out, pad_value)
  return out, out_lens


def get_non_deterministic_ops_from_graph():
  """
  Lists all non deterministic ops used in the default graph
  If a non deterministic op is used multiple times each instance will be listed

  currently doesn't check if user specified a specific computation device
  list of non deterministic ops is not jet complete

  :return: list of all non deterministic ops names (depending on device and tf version) used in current graph
  :rtype: list[tf.Operation]
  """
  device_types = {device.device_type for device in get_tf_list_local_devices()}
  non_det_ops = []
  tf_version = tf_version_tuple()
  for op in tf_compat.v1.get_default_graph().get_operations():
    if op.type == "Mean" and tf_version <= (1, 5, 0) and "GPU" in device_types:
      non_det_ops.append(op)
    elif op.type == "BiasAddGrad" and "GPU" in device_types:
      non_det_ops.append(op)
    if op.type == "UnsortedSegmentSum" and "GPU" in device_types:
      non_det_ops.append(op)
    # elif ... more non det ops to be added

  return non_det_ops


def compute_sampled_logits(weights,
                           biases,
                           labels,
                           inputs,
                           num_sampled,
                           num_classes,
                           num_true=1,
                           sampled_values=None,
                           subtract_log_q=True,
                           remove_accidental_hits=False,
                           partition_strategy="mod",
                           name="compute_sampled_logits",
                           seed=None):
  """Helper function for nce_loss and sampled_softmax_loss functions.
  Computes sampled output training logits and labels suitable for implementing
  e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
  sampled_softmax_loss).
  Note: In the case where num_true > 1, we assign to each target class
  the target probability 1 / num_true so that the target probabilities
  sum to 1 per-example.

  This is a copy of
    https://github.com/tensorflow/tensorflow/blob/e19c354920c3b246dda6598229210a582caaa1a9/tensorflow/python/ops/nn_impl.py#L1440

  :param tf.Tensor|list[tf.Tensor]|tuple[tf.Tensor] weights: A `Tensor` of shape `[num_classes, dim]`,
    or a list of `Tensor` objects whose concatenation along dimension 0 has shape `[num_classes, dim]`.
    The class embeddings.
  :param tf.Tensor biases: A `Tensor` of shape `[num_classes]`.  The class biases.
  :param tf.Tensor labels: A `Tensor` of type `int64` and shape `[batch_size, num_true]`.
    The target classes.  Note that this format differs from
    the `labels` argument of `tf.nn.softmax_cross_entropy_with_logits`.
  :param tf.Tensor inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
  :param int num_sampled: The number of classes to randomly sample per batch.
  :param int num_classes: The number of possible classes.
  :param int num_true: The number of target classes per training example.
  :param (tf.Tensor, tf.Tensor, tf.Tensor)|None sampled_values: a tuple of
    (`sampled_candidates`, `true_expected_count`, `sampled_expected_count`)
    returned by a `*_candidate_sampler` function.
    (if None, we default to `log_uniform_candidate_sampler`)
  :param bool subtract_log_q: whether to subtract the log expected count of
    the labels in the sample to get the logits of the true labels.
    Default is True.  Turn off for Negative Sampling.
  :param bool remove_accidental_hits: Whether to remove "accidental hits"
    where a sampled class equals one of the target classes.
  :param str partition_strategy: A string specifying the partitioning strategy, relevant
    if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
    Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
  :param str|None name: A name for the operation.
  :param int|None seed: random seed for candidate sampling. Default to None, which doesn't set
    the op-level random seed for candidate sampling.
  :return:
    out_logits: `Tensor` object with shape
        `[batch_size, num_true + num_sampled]`, for passing to either
        `nn.sigmoid_cross_entropy_with_logits` (NCE) or
        `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    out_targets: A Tensor object with the same shape and dtype as `out_logits`.
      These are the targets. If num_true > 1 the per-example labels are divided by num_true so they sum to 1.0.
  :rtype: (tf.Tensor, tf.Tensor)
  """
  if not isinstance(weights, (list, tuple)):
    weights = [weights]

  with tf.name_scope(name):
    if labels.dtype != tf.int64:
      labels = tf.cast(labels, tf.int64)
    labels_flat = tf.reshape(labels, [-1])

    if sampled_values is None:
      sampled_values = tf.random.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes,
          seed=seed)

    sampled, true_expected_count, sampled_expected_count = (
        tf.stop_gradient(s) for s in sampled_values)
    sampled = tf.cast(sampled, tf.int64)

    all_ids = tf.concat([labels_flat, sampled], 0)

    all_w = tf.nn.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)
    if all_w.dtype != inputs.dtype:
      all_w = tf.cast(all_w, inputs.dtype)

    true_w = tf.slice(all_w,
                      [0, 0],
                      [tf.shape(labels_flat)[0], -1])

    sampled_w = tf.slice(
        all_w, [tf.shape(labels_flat)[0], 0], [-1, -1])
    sampled_logits = tf.matmul(inputs, sampled_w, transpose_b=True)

    all_b = tf.nn.embedding_lookup(
        biases, all_ids, partition_strategy=partition_strategy)
    if all_b.dtype != inputs.dtype:
      all_b = tf.cast(all_b, inputs.dtype)
    true_b = tf.slice(all_b, [0], tf.shape(labels_flat))
    sampled_b = tf.slice(all_b, tf.shape(labels_flat), [-1])

    dim = tf.shape(true_w)[1:2]
    new_true_w_shape = tf.concat([[-1, num_true], dim], 0)
    row_wise_dots = tf.multiply(
        tf.expand_dims(inputs, 1),
        tf.reshape(true_w, new_true_w_shape))
    dots_as_matrix = tf.reshape(row_wise_dots,
                                tf.concat([[-1], dim], 0))
    true_logits = tf.reshape(tf.reduce_sum(dots_as_matrix, axis=1),
                             [-1, num_true])
    true_b = tf.reshape(true_b, [-1, num_true])
    true_logits += true_b
    sampled_logits += sampled_b

    if remove_accidental_hits:
      acc_hits = tf.nn.compute_accidental_hits(
          labels, sampled, num_true=num_true)
      acc_indices, acc_ids, acc_weights = acc_hits

      acc_indices_2d = tf.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = tf.reshape(tf.cast(acc_ids, tf.int32), [-1, 1])
      sparse_indices = tf.concat([acc_indices_2d, acc_ids_2d_int32], 1,
                                 "sparse_indices")
      sampled_logits_shape = tf.concat(
          [tf.shape(labels)[:1],
           tf.expand_dims(num_sampled, 0)], 0)
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = tf.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += tf_compat.v1.sparse_to_dense(
          sparse_indices,
          sampled_logits_shape,
          acc_weights,
          default_value=0.0,
          validate_indices=False)

    if subtract_log_q:
      true_logits -= tf_compat.v1.log(true_expected_count)
      sampled_logits -= tf_compat.v1.log(sampled_expected_count)

    out_logits = tf.concat([true_logits, sampled_logits], 1)

    out_targets = tf.concat([
        tf.ones_like(true_logits) / num_true,
        tf.zeros_like(sampled_logits)
    ], 1)

    return out_logits, out_targets


def safe_deep_copy(obj):
  """
  :param T obj:
  :return: deepcopy of obj, without copying TF types, Python modules, functions/lambdas
  :rtype: T
  """
  import types
  from returnn.util.basic import deepcopy
  stop_types = [
    # Python types.
    types.FunctionType, types.LambdaType, types.BuiltinFunctionType, types.BuiltinMethodType,
    types.ModuleType,
    # Common TF types.
    tf.Tensor, tf.Operation, tf.Variable, tf.Graph, tf_compat.v1.Session,
    # Our own types, which should not be copied.
    DimensionTag]
  return deepcopy(obj, stop_types=stop_types)


class FetchHelper:
  """
  ``session.run(tensor)`` does not work if ``tensor`` is inside a loop (``tf.while_loop``) (or ``tf.cond``).
  You would get an error like this::

    Operation '...' has been marked as not fetchable.

  This class is a helper to work around that. It will add an op to the graph, which stores the most recent value.
  To get this executed automatically, you likely want to add is as a control dependency to another op.
  Use :func:`add_to_control_inputs` for that, or better :func:`copy_graph_replace_tensors`,
  or better :func:`copy_graph`.
  """

  def __init__(self, tensor, verbose_stream=None):
    """
    :param tf.Tensor tensor:
    :param typing.IO[str]|None verbose_stream:
    """
    assert isinstance(tensor, tf.Tensor)
    self.tensor = tensor
    self.verbose_stream = verbose_stream
    self.most_recent_value = None
    self.callback_count = 0

    with same_control_flow_ctx(tensor):
      with tf.device("/cpu:0"):
        dummy_out, = tf_compat.v1.py_func(
          self._callback,
          [tensor],
          [tf.int64],  # dummy return value. will not be used
          name="FetchHelper_%s" % os.path.basename(tensor.op.name))
        assert isinstance(dummy_out, tf.Tensor)
        self.fetch_op = dummy_out.op

      with tf_compat.v1.colocate_with(tensor.op):
        with tf.control_dependencies([self.fetch_op]):
          self.identity_with_dep = tf.identity(tensor)

  def __repr__(self):
    return "%s(%r)" % (self.__class__.__name__, self.tensor)

  @classmethod
  def copy_graph(cls, fetches, target_op, fetch_helper_tensors, stop_at_ts=(), verbose_stream=None):
    """
    :param tf.Tensor|list[tf.Tensor]|T fetches:
    :param tf.Operation target_op: will add the fetch helpers as control dependencies to this op
    :param list[tf.Tensor] fetch_helper_tensors:
    :param typing.IO[str]|None verbose_stream:
    :param typing.Iterable[tf.Tensor] stop_at_ts: iterable of tensors at which the graph walk stops.
    :return: copied fetches, fetch helpers, transformed target op
    :rtype: (tf.Tensor|list[tf.Tensor]|T, list[FetchHelper], tf.Operation)
    """
    from pprint import pformat
    from tensorflow.python.util import nest
    fetches_flat = nest.flatten(fetches)
    from returnn.extern import graph_editor
    ops = graph_editor.get_backward_walk_ops(
      seed_ops=[x.op if isinstance(x, (tf.Tensor, tf.Variable)) else x for x in fetches_flat],
      stop_at_ts=stop_at_ts,
      inclusive=True, control_inputs=True)
    if target_op.name in [x.name for x in ops] and target_op not in ops:
      # What? Very strange. Replace by other instance.
      target_op = [x for x in ops if x.name == target_op.name][0]
    assert target_op in ops, "target_op %r,\nops\n%s" % (target_op, pformat(ops))
    for x in fetch_helper_tensors:
      assert x.op in ops
    sgv = graph_editor.make_view(ops)
    copier = graph_editor.Transformer()
    copier.transform_external_input_handler = lambda info_, t: t
    _, info = copier(sgv, dst_graph=sgv.graph, dst_scope="", reuse_dst_scope=True)
    assert isinstance(info, graph_editor.TransformerInfo)
    target_op_transformed = info.transformed(target_op)
    assert isinstance(target_op_transformed, tf.Operation), (
      "\ntarget_op\n%r,\nfetches\n%r,\nstop_at_ts\n%s,\nops\n%s" % (
        target_op, fetches, pformat(stop_at_ts), pformat(ops)))
    fetch_helpers = []
    for x in fetch_helper_tensors:
      fetch_helper = FetchHelper(tensor=info.transformed(x), verbose_stream=verbose_stream)
      fetch_helper.add_to_control_inputs(target_op_transformed)
      fetch_helpers.append(fetch_helper)
    fetches_flat_transformed = [info.transformed(x) for x in fetches_flat]
    return nest.pack_sequence_as(fetches, fetches_flat_transformed), fetch_helpers, target_op_transformed

  @classmethod
  def copy_graph_replace_tensors(cls, fetches, fetch_helpers):
    """
    :param tf.Tensor|list[tf.Tensor] fetches:
    :param list[FetchHelper] fetch_helpers:
    :return: as fetches
    :rtype: tf.Tensor|list[tf.Tensor]
    """
    from returnn.extern import graph_editor
    fetches_copied = graph_editor.graph_replace(
      target_ts=fetches,
      replacement_ts={fetch_helper.tensor: fetch_helper.identity_with_dep for fetch_helper in fetch_helpers},
      reuse_dst_scope=True)
    return fetches_copied

  def add_to_control_inputs(self, other_op):
    """
    Note: This will not work if you already did a ``session.run``.
    See `here <https://stackoverflow.com/questions/57707445/>`__.
    Use :func:`copy_graph_replace_tensors` instead. Or better :func:`copy_graph`.

    :param tf.Operation other_op:
    """
    add_control_input(other_op, self.fetch_op)

  @classmethod
  def _format_value(cls, value):
    """
    :param numpy.ndarray|object value:
    :rtype: str
    """
    import numpy
    if isinstance(value, numpy.ndarray):
      info = "shape %s, dtype %s" % (value.shape, value.dtype)
      if value.size > 0:
        v_minmax = numpy.min(value), numpy.max(value)
        info += ", min/max %s/%s" % v_minmax
        if value.dtype.kind == "f":
          info += ", mean/stddev %s/%s" % (numpy.mean(value), numpy.std(value))
        if value.ndim <= 1:
          info += ", (%s)" % numpy.array2string(value)
      else:
        info += ", EMPTY"
    elif isinstance(value, (numpy.floating, numpy.integer, numpy.bool_, float, int, bool, str, bytes)):
      info = "%s(%s)" % (type(value).__name__, value)
    elif value is None:
      info = "None"
    else:
      info = "type %r" % type(value)
    return info

  def _callback(self, value):
    """
    :param numpy.ndarray value:
    :return: dummy value
    :rtype: int
    """
    if self.verbose_stream:
      print(
        "FetchHelper(%i): %r = %s" % (self.callback_count, self.tensor, self._format_value(value)),
        file=self.verbose_stream)
    self.most_recent_value = value
    self.callback_count += 1
    return 0
