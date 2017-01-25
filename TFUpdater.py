
from __future__ import print_function

import tensorflow as tf

from Log import log
from TFNetwork import TFNetwork


_OptimizerClassesDict = {}  # type: dict[str,()->tf.train.Optimizer]


def get_optimizer_class(class_name):
  """
  :param str class_name: e.g. "adam"
  :return:
  """
  if not _OptimizerClassesDict:
    for name, v in vars(tf.train).items():
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
    epsilon = 1e-16
    momentum = self.config.float("momentum", 0.0)
    optim_config = self.config.typed_value("optimizer")
    if optim_config:
      assert isinstance(optim_config, dict)
      optim_config = optim_config.copy()
      optim_class_name = optim_config.pop("class")
      optim_class = get_optimizer_class(optim_class_name)
      optim_config.setdefault("epsilon", epsilon)
      if momentum:
        optim_config.setdefault("momentum", momentum)
      optim_config["learning_rate"] = lr
      print("Create optimizer %s with options %r." % (optim_class, optim_config), file=log.v2)
      optimizer = optim_class(**optim_config)
      assert isinstance(optimizer, tf.train.Optimizer)
    elif self.config.bool("adam", False):
      assert not momentum
      print("Create Adam optimizer.", file=log.v2)
      optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon)
    elif self.config.bool("nadam", False):
      raise NotImplementedError("NAdam not implemented yet.")
    elif self.config.bool("adadelta", False):
      assert not momentum
      print("Create Adadelta optimizer.", file=log.v2)
      optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr, epsilon=epsilon)
    elif self.config.bool("adagrad", False):
      assert not momentum
      print("Create Adagrad optimizer.", file=log.v2)
      optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
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
    if not self.optimizer:
      self.create_optimizer()

    assert self.loss is not None
    with tf.variable_scope("optimize"):
      # AccumulateN might not be deterministic but should be faster and should require less memory.
      # We might want to make this configurable.
      aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
      grad_noise = self.config.float("gradient_noise", 0.0)
      grad_clip = self.config.float("gradient_clip", 0.0)

      # Extended self.optimizer.minimize() to optinally modify gradients.
      grads_and_vars = self.optimizer.compute_gradients(
        self.loss, var_list=self.trainable_vars,
        aggregation_method=aggregation_method)
      if not [v for g, v in grads_and_vars if g is not None]:
        raise Exception("no single variable to train")
      # Also see tf.contrib.layers.optimizers.optimize_loss() for reference.
      if grad_noise:
        assert grad_noise > 0
        from TFUtil import add_scaled_noise_to_gradients
        grads_and_vars = add_scaled_noise_to_gradients(grads_and_vars, grad_noise)
      if grad_clip:
        assert grad_clip > 0
        grads_and_vars = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in grads_and_vars]
      apply_grads = self.optimizer.apply_gradients(grads_and_vars)
      incr_step_op = tf.assign_add(self.network.global_train_step, 1, name="global_train_step_increment")
      self.optim_op = tf.group(apply_grads, incr_step_op, name="optim_and_step_incr")

    print("Initialize optimizer with slots %s." % self.optimizer.get_slot_names(), file=log.v3)
    slot_vars = []
    for slot_name in self.optimizer.get_slot_names():
      for v in self.trainable_vars:
        slot_var = self.optimizer.get_slot(var=v, name=slot_name)
        assert slot_var is not None
        slot_vars.append(slot_var)
    self.tf_session.run(tf.variables_initializer(slot_vars, name="init_optim_slot_vars"))

  def get_optim_op(self):
    """
    :rtype: tf.Operation
    """
    if self.optim_op is None:
      self.create_optim_op()
    return self.optim_op
