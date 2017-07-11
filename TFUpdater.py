
from __future__ import print_function

import tensorflow as tf

from Log import log
from TFNetwork import TFNetwork
from TFUtil import tf_version_tuple, assert_min_tf_version

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
      assert "learning_rate" not in optim_config, "learning_rate will be set implicitely"
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
    # Keep track of all current available vars.
    # The optimizer could add some, even some which are not so-called "slot-vars",
    # and we want to keep track about them.
    all_vars = tf.global_variables()  # type: list[tf.Variable]

    if not self.optimizer:
      self.create_optimizer()

    assert self.loss is not None
    with tf.variable_scope("optimize"):
      # AccumulateN might not be deterministic but should be faster and should require less memory.
      # We might want to make this configurable.
      aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
      grad_noise = self.config.float("gradient_noise", 0.0)
      grad_clip = self.config.float("gradient_clip", 0.0)
      grad_clip_global_norm = self.config.float("gradient_clip_global_norm", 0.0)
      # E.g. https://github.com/openai/baselines/blob/master/baselines/deepq/simple.py: grad_norm_clipping=10 -> tf.clip_by_norm

      # Extended self.optimizer.minimize() to optionally modify gradients.
      grads_and_vars = self.optimizer.compute_gradients(
        self.loss, var_list=self.trainable_vars,
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
      incr_step_op = tf.assign_add(self.network.global_train_step, 1, name="global_train_step_increment")
      self.optim_op = tf.group(apply_grads, incr_step_op, name="optim_and_step_incr")

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
