
from __future__ import print_function

import tensorflow as tf

from Log import log
from TFNetwork import TFNetwork
from TFUtil import tf_version_tuple, assert_min_tf_version, CustomUpdate

_OptimizerClassesDict = {}  # type: dict[str,()->tf.train.Optimizer]


def get_optimizer_class(class_name):
  """
  :param str class_name: e.g. "adam"
  :return:
  """
  if not _OptimizerClassesDict:
    potential_list = list(vars(tf.train).items())
    if tf_version_tuple() >= (1, 2, 0):
      import tensorflow.contrib.opt.python.training.nadam_optimizer as nadam
      potential_list += list(vars(nadam).items())
    for name, v in potential_list:
      assert isinstance(name, str)
      if name.endswith("Optimizer"):
        name = name[:-len("Optimizer")]
      else:
        continue
      if not issubclass(v, tf.train.Optimizer):
        continue
      name = name.lower()
      assert name not in _OptimizerClassesDict
      _OptimizerClassesDict[name] = v
  return _OptimizerClassesDict[class_name.lower()]


class Updater(object):
  """
  This will create the :class:`tf.train.Optimizer` instance given the config
  and the update-op for all trainable vars.
  See the code of :func:`Updater.create_optimizer` for valid config options.

  Note: `Vincent Vanhoucke says <https://github.com/tensorflow/tensorflow/issues/323#issuecomment-159116515>`_,
  in case you get nans, consider increasing the epsilon (for Adam, Nadam and similar).
  This is the config option ``optimizer_epsilon``.
  In some places in our Theano code, 1e-16 is our default epsilon, in some other parts, 1e-8 is.
  1e-8 might be more stable. Or even 1e-6.
  Note that when the gradient is suddenly zero in one step, the update can be proportional to lr / eps.

  From the :class:`tf.train.AdamOptimizer` documentation:

      The default value of 1e-8 for epsilon might not be a good default in
      general. For example, when training an Inception network on ImageNet a
      current good choice is 1.0 or 0.1. Note that since AdamOptimizer uses the
      formulation just before Section 2.1 of the Kingma and Ba paper rather than
      the formulation in Algorithm 1, the "epsilon" referred to here is "epsilon
      hat" in the paper.

  More from Vincent Vanhoucke:

      One thing you can do is run with a tiny learning rate, or even zero learning rate.
      If you still have divergence then, you have a bug in your setup.
      If not, increase your rate slowly and see if there is a regime in which things train without diverging.
      It's completely possible to have weights that are in a good range,
      but activations or gradients going to infinity because of the shape of the loss, or too high a learning rate.
      It's obviously always a possibility that there is a bug in the optimizers, but in my experience,
      every single instance of this kind of problem could be traced back to a weirdly wired model,
      learning rate issues, bad randomization of the input examples,
      or - in the case of Adam or RMSProp - issues with the epsilon value.

  In addition, you might also want to try ``gradient_nan_inf_filter`` or maybe set beta1=0.5.

  For further debugging, see :func:`tf.add_check_numerics_ops` or :func:`add_check_numerics_ops_and_debug_print`,
  which is config option ``debug_add_check_numerics_ops``.

  """

  def __init__(self, config, tf_session, network):
    """
    :param Config.Config config:
    :param tf.Session tf_session:
    :param TFNetwork network:
    """
    self.config = config
    self.tf_session = tf_session
    self.learning_rate_var = tf.Variable(name="learning_rate", initial_value=0.0, trainable=False, dtype="float32")
    self.trainable_vars = []  # type: list[tf.Variable]
    self.network = network
    self.loss = network.get_objective()
    self.optimizer = None  # type: tf.train.Optimizer
    self.optim_op = None  # type: tf.Operation
    self.optimizer_vars = []  # type: list[tf.Variable]
    self.optimizer_init_vars_op = None  # type: tf.Operation

  def reset_optim_op(self):
    """
    Call this if sth is changed which the optim_op depends on.
    See self.create_optim_op().
    """
    self.optim_op = None  # type: tf.Operation

  def set_trainable_vars(self, trainable_vars):
    """
    :param list[tf.Variable] trainable_vars:
    """
    if trainable_vars == self.trainable_vars:
      return
    self.trainable_vars = trainable_vars
    self.reset_optim_op()

  def set_learning_rate(self, value):
    """
    :param float value:
    """
    self.network.get_var_assigner(self.learning_rate_var).assign(value, session=self.tf_session)

  def create_optimizer(self):
    lr = self.learning_rate_var
    epsilon = self.config.float("optimizer_epsilon", 1e-16)
    momentum = self.config.float("momentum", 0.0)
    optim_config = self.config.typed_value("optimizer")
    if optim_config:
      if isinstance(optim_config, str):
        optim_config = {"class": optim_config}
      assert isinstance(optim_config, dict)
      optim_config = optim_config.copy()
      optim_class_name = optim_config.pop("class")
      optim_class = get_optimizer_class(optim_class_name)
      from Util import collect_class_init_kwargs
      optim_class_kwargs = collect_class_init_kwargs(optim_class)
      if "epsilon" in optim_class_kwargs:
        optim_config.setdefault("epsilon", epsilon)
      if "momentum" in optim_class_kwargs and momentum:
        optim_config.setdefault("momentum", momentum)
      assert "learning_rate" not in optim_config, "learning_rate will be set implicitly"
      optim_config["learning_rate"] = lr
      print("Create optimizer %s with options %r." % (optim_class, optim_config), file=log.v2)
      optimizer = optim_class(**optim_config)
      assert isinstance(optimizer, tf.train.Optimizer)
    elif self.config.bool("adam", False):
      assert not momentum
      print("Create Adam optimizer.", file=log.v2)
      # Default TF values: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.
      # Default Keras values: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8.
      # Our Theano default values: beta1=0.9, beta2=0.999, epsilon=1e-16
      # https://github.com/openai/improved-gan/blob/master/imagenet/train_imagenet.py: beta1=0.5
      optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
    elif self.config.bool("nadam", False):
      assert_min_tf_version((1, 2, 0), "NadamOptimizer introduced in TF 1.2.0")
      assert not momentum
      print("Create NAdam optimizer.", file=log.v2)
      # TF default values: like Adam: beta1=0.9, beta2=0.999, epsilon=1e-8
      # Our Theano default values: decay=0.004, beta1=0.9, beta2=0.999, epsilon=1e-8
      from tensorflow.contrib.opt.python.training.nadam_optimizer import NadamOptimizer
      optimizer = NadamOptimizer(learning_rate=lr, epsilon=epsilon)
    elif self.config.bool("adadelta", False):
      assert not momentum
      print("Create Adadelta optimizer.", file=log.v2)
      optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr, epsilon=epsilon)
    elif self.config.bool("adagrad", False):
      assert not momentum
      print("Create Adagrad optimizer.", file=log.v2)
      optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    elif self.config.is_of_type("rmsprop", float):
      print("Create RMSProp optimizer. With Decay %f" % (self.config.float("rmsprop", 0.9)), file=log.v2)
      optimizer = tf.train.RMSPropOptimizer(decay=self.config.float("rmsprop", 0.9), learning_rate=lr, momentum=momentum, epsilon=epsilon)
    elif self.config.bool("rmsprop", False):
      print("Create RMSProp optimizer.", file=log.v2)
      optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum, epsilon=epsilon)
    elif momentum:
      print("Create Momentum optimizer.", file=log.v2)
      optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
    else:
      print("Create SGD optimizer.", file=log.v2)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    self.optimizer = optimizer
    self.reset_optim_op()

  def create_optim_op(self):
    assert self.loss is not None
    assert self.trainable_vars, "no variables to update/optimize"

    # Keep track of all current available vars.
    # The optimizer could add some, even some which are not so-called "slot-vars",
    # and we want to keep track about them.
    all_vars = tf.global_variables()  # type: list[tf.Variable]

    if not self.optimizer:
      self.create_optimizer()

    trainable_vars_for_gradients = list(self.trainable_vars)
    trainable_vars_custom_update = []  # type: list[tf.Variable]
    for v in self.trainable_vars:
      if hasattr(v, "custom_update"):
        trainable_vars_custom_update.append(v)
        trainable_vars_for_gradients.remove(v)

    with tf.variable_scope("optimize"):
      if trainable_vars_for_gradients:
        # AccumulateN might not be deterministic but should be faster and should require less memory.
        # We might want to make this configurable.
        aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
        grad_noise = self.config.float("gradient_noise", 0.0)
        grad_clip = self.config.float("gradient_clip", 0.0)
        grad_clip_global_norm = self.config.float("gradient_clip_global_norm", 0.0)
        # E.g. https://github.com/openai/baselines/blob/master/baselines/deepq/simple.py: grad_norm_clipping=10 -> tf.clip_by_norm

        # Extended self.optimizer.minimize() to optionally modify gradients.
        grads_and_vars = self.optimizer.compute_gradients(
          self.loss, var_list=trainable_vars_for_gradients,
          aggregation_method=aggregation_method)
        if not [v for g, v in grads_and_vars if g is not None]:
          raise Exception("no single variable to train")
        # Also see tf.contrib.layers.optimizers.optimize_loss() for reference.
        if self.config.bool("gradient_nan_inf_filter", False):
          from TFUtil import nan_to_num
          grads_and_vars = [(nan_to_num(grad, nan_num=0.0, inf_num=0.0), var) for (grad, var) in grads_and_vars]
        if grad_noise:
          assert grad_noise > 0
          from TFUtil import add_scaled_noise_to_gradients
          grads_and_vars = add_scaled_noise_to_gradients(grads_and_vars, grad_noise)
        if grad_clip:
          assert grad_clip > 0
          grads_and_vars = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in grads_and_vars]
        if grad_clip_global_norm:
          assert grad_clip_global_norm > 0
          grads_clipped, _ = tf.clip_by_global_norm([grad for (grad, _) in grads_and_vars], grad_clip_global_norm)
          grads_and_vars = zip(grads_clipped, [var for (_, var) in grads_and_vars])
        apply_grads = self.optimizer.apply_gradients(grads_and_vars)
      else:
        apply_grads = tf.no_op(name="no_grad_vars_no_op")
      incr_step_op = tf.assign_add(self.network.global_train_step, 1, name="global_train_step_increment")
      self.optim_op = tf.group(apply_grads, incr_step_op, name="optim_and_step_incr")

    if trainable_vars_custom_update:
      with tf.variable_scope("custom_update"):
        updates = [self.optim_op]
        for param in trainable_vars_custom_update:
          custom_update = getattr(param, "custom_update")
          assert isinstance(custom_update, CustomUpdate)
          updates.append(custom_update.update_var(param))
        self.optim_op = tf.group(*updates)

    print("Initialize optimizer with slots %s." % self.optimizer.get_slot_names(), file=log.v3)
    slot_vars = []
    for slot_name in self.optimizer.get_slot_names():
      for v in self.trainable_vars:
        slot_var = self.optimizer.get_slot(var=v, name=slot_name)
        assert slot_var is not None
        assert isinstance(slot_var, tf.Variable)
        slot_vars.append(slot_var)
    self.optimizer_vars = slot_vars

    # Check if there were any other variables added.
    # E.g. currently (TF 1.0) the `AdamOptimizer` creates these additional vars
    # `[<tf.Variable 'optimize/beta1_power:0' shape=() dtype=float32_ref>,
    #   <tf.Variable 'optimize/beta2_power:0' shape=() dtype=float32_ref>]`
    # which do not correspond to trainable vars, thus we did not get them as slot vars above.
    other_new_vars = []
    for v in tf.global_variables():
      if v in all_vars:
        continue
      if v in self.optimizer_vars:
        continue
      other_new_vars.append(v)
    if other_new_vars:
      print("These additional variable were created by the optimizer: %s." % other_new_vars, file=log.v3)
      self.optimizer_vars += other_new_vars
    self.optimizer_init_vars_op = tf.variables_initializer(self.optimizer_vars, name="init_optim_slot_vars")
    self.init_optimizer_vars()

    if self.config.bool("debug_add_check_numerics_ops", False):
      print("Adding checks for inf/nan.", file=log.v3)
      self.optim_op = tf.group(self.optim_op, add_check_numerics_ops_and_debug_print([self.optim_op]))

  def get_optim_op(self, callback_on_new=None):
    """
    :param None|()->None callback_on_new:
    :rtype: tf.Operation
    """
    if self.optim_op is None:
      self.create_optim_op()
      if callback_on_new:
        callback_on_new()
    return self.optim_op

  def init_optimizer_vars(self):
    self.tf_session.run(self.optimizer_init_vars_op)


def add_check_numerics_ops_and_debug_print(
  fetches=None, ignore_ops=None, use_check_numerics=True, debug_print_added_checks=True):
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
  :return: operation which performs all the checks
  :rtype: tf.Operation
  """
  if fetches is None:
    ops = tf.get_default_graph().get_operations()
  else:
    fetch_ops = [v.op if isinstance(v, tf.Tensor) else v for v in fetches]
    assert all([isinstance(op, tf.Operation) for op in fetch_ops])
    from tensorflow.contrib import graph_editor
    ops = graph_editor.get_backward_walk_ops(fetch_ops, inclusive=True, control_inputs=True)
  if ignore_ops is None:
    # The checks could increase the memory usage a lot.
    # Ignore some common ops which should not be able to introduce inf/nan.
    ignore_ops = {
      "Add", "AddN", "Sum", "Mul", "MatMul", "Sub", "L2Loss", "Floor", "Neg", "UnsortedSegmentSum",
      "Switch", "Merge", "PreventGradient",
      "Const", "Identity", "Fill", "ZerosLike",
      "Reshape", "Tile", "ExpandDims", "ConcatV2", "Transpose",
      "Slice", "StridedSlice", "StridedSliceGrad", "Gather",
      "TruncatedNormal", "RandomUniform"}
  check_op = []
  # This code relies on the ordering of ops in get_operations().
  # The producer of a tensor always comes before that tensor's consumer in
  # this list. This is true because get_operations() returns ops in the order
  # added, and an op can only be added after its inputs are added.
  for op in ops:
    assert isinstance(op, tf.Operation)
    if op.type in ignore_ops:
      continue
    for output in op.outputs:
      if output.dtype in [tf.float16, tf.float32, tf.float64]:
        message = op.name + ":" + str(output.value_index)
        with tf.control_dependencies(check_op):
          if debug_print_added_checks:
            print("add check for:", output, op.type)
          if use_check_numerics:
            check_op = [tf.check_numerics(output, message=message)]
          else:
            is_finite = tf.reduce_all(tf.is_finite(output))
            check_op = [tf.Assert(is_finite, [message, "Tensor had inf or nan values:", output])]
  return tf.group(*check_op)
