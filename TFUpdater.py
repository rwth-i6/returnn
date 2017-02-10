
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops

from Log import log
from TFNetwork import TFNetwork


_OptimizerClassesDict = {}  # type: dict[str,()->tf.train.Optimizer]


def get_optimizer_class(class_name):
  """
  :param str class_name: e.g. "adam"
  :return:
  """
  if not _OptimizerClassesDict:
    for name, v in list(vars(tf.train).items()) + list(globals().items()):
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


class NadamOptimizer(tf.train.Optimizer):
  def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False, name="Nadam"):
    """
    :param tf.Tensor|tf.Variable|float learning_rate:
    :param float beta1:
    :param float beta2:
    :param float epsilon:
    :param bool use_locking:
    :param str name:

    Nadam is Adam with Nesterov momentum, by Timothy Dozat (http://web.stanford.edu/~tdozat/).
    http://cs229.stanford.edu/proj2015/054_report.pdf

    Also see tf.train.AdamOptimizer for reference.
    For Nadam code, see also Theano Updater.
    Also see here, from the original author of Nadam:
    https://github.com/tdozat/Optimization/blob/master/tensorflow/nadam.py
    """
    super(NadamOptimizer, self).__init__(use_locking=use_locking, name=name)

    self._lr = learning_rate
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

    # Tensor scalar versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._beta1_t = None
    self._beta2_t = None
    self._epsilon_t = None

    # Helper scalars, created in _prepare().
    self._mu_t = None
    self._mu_t_next = None

    # Scalar variables to accumulate the powers of the beta parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._beta1_power = None
    self._beta2_power = None
    self._mu_prod = None

  def _create_slots(self, var_list):
    # This get's called before self.prepare().
    t = tf.cast(tf.train.get_global_step(), "float32") + 1
    # Create the beta1 and beta2 accumulators on the same device as the first variable.
    if self._beta1_power is None or self._beta1_power.graph is not var_list[0].graph:
      with ops.colocate_with(var_list[0]):
        self._beta1_power = tf.Variable(self._beta1 ** t, name="beta1_power", trainable=False)
        self._beta2_power = tf.Variable(self._beta2 ** t, name="beta2_power", trainable=False)
        self._mu_prod = tf.Variable(1.0, name="mu_prod", trainable=False)
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
    self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
    self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    self._t = tf.cast(tf.train.get_global_step(), "float32") + 1

    # momentum schedule, http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
    nadam_decay = 0.004  # Magical 250.0 denominator in nesterov scaling of i_t
    self._mu_t = (self._beta1_t * (1 - 0.5 * 0.96 ** (self._t * nadam_decay)))
    self._mu_t_next = self._beta1_t * (1 - 0.5 * 0.96 ** ((self._t + 1) * nadam_decay))  # for simplified NAG

    self._mu_prod_t_next = self._mu_prod * self._mu_t
    self._mu_prod_t_next2 = self._mu_prod_t_next * self._mu_t_next

  def _apply_dense(self, grad, var):
    """
    :param tf.Tensor grad:
    :param tf.Variable var:
    :return: group of update operations
    :rtype: tf.Operation
    """
    m_prev = self.get_slot(var, "m")
    v_prev = self.get_slot(var, "v")

    # called m_t in paper
    m = self._beta1_t * m_prev + (1 - self._beta1_t) * grad
    m_ = m / (1 - self._mu_prod_t_next2)  # bias correction (with momentum schedule (include the next t+1))

    # called n_t in paper
    v = self._beta2_t * v_prev + (1 - self._beta2_t) * (grad * grad)
    v_ = v / (1 - self._beta2_power)

    grad_ = grad / (1 - self._mu_prod_t_next)
    m__ = (1 - self._mu_t) * grad_ + self._mu_t_next * m_

    step = self._lr_t * m__ / (tf.sqrt(v_) + self._epsilon_t)
    var_update = tf.assign_sub(var, step, use_locking=self._use_locking)

    return tf.group(
      var_update,
      tf.assign(m_prev, m, use_locking=self._use_locking),
      tf.assign(v_prev, v, use_locking=self._use_locking))

  def _apply_sparse(self, grad, var):
    """
    :param tf.IndexedSlices grad:
    :param tf.Variable var:
    :return: group of update operations
    :rtype: tf.Operation
    """
    beta2_power = tf.cast(self._beta2_power, var.dtype.base_dtype)
    lr_t = tf.cast(self._lr_t, var.dtype.base_dtype)
    beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
    mu_t = tf.cast(self._mu_t, var.dtype.base_dtype)
    mu_t_next = tf.cast(self._mu_t_next, var.dtype.base_dtype)
    mu_prod_t_next = tf.cast(self._mu_prod_t_next, var.dtype.base_dtype)
    mu_prod_t_next2 = tf.cast(self._mu_prod_t_next2, var.dtype.base_dtype)

    m_prev = self.get_slot(var, "m")
    v_prev = self.get_slot(var, "v")

    # called m_t in paper
    m = beta1_t * m_prev
    m = tf.assign(m_prev, m, use_locking=self._use_locking)
    m = tf.scatter_add(m, grad.indices, (1 - beta1_t) * grad.values, use_locking=self._use_locking)
    m_update = m
    m_ = m / (1 - mu_prod_t_next2)  # bias correction (with momentum schedule (include the next t+1))

    # called n_t in paper
    v = beta2_t * v_prev
    v = tf.assign(v_prev, v, use_locking=self._use_locking)
    v = tf.scatter_add(v, grad.indices, (1 - beta2_t) * (grad.values * grad.values), use_locking=self._use_locking)
    v_update = v
    v_ = v / (1 - beta2_power)

    m__ = tf.sparse_add(
      mu_t_next * m_,
      tf.IndexedSlices((1 - mu_t) * grad.values / (1 - mu_prod_t_next), grad.indices, grad.dense_shape))

    step = lr_t * m__ / (tf.sqrt(v_) + epsilon_t)
    var_update = tf.assign_sub(var, step, use_locking=self._use_locking)

    return tf.group(var_update, m_update, v_update)

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      with ops.colocate_with(self._beta1_power):
        update_beta1 = self._beta1_power.assign(
            self._beta1_power * self._beta1_t,
            use_locking=self._use_locking)
        update_beta2 = self._beta2_power.assign(
            self._beta2_power * self._beta2_t,
            use_locking=self._use_locking)
        update_mu_prod = self._mu_prod.assign(
            self._mu_prod_t_next,
            use_locking=self._use_locking)
    return tf.group(*update_ops + [update_beta1, update_beta2, update_mu_prod], name=name_scope)


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
      assert not momentum
      print("Create NAdam optimizer.", file=log.v2)
      optimizer = NadamOptimizer(learning_rate=lr, epsilon=epsilon)
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
      grad_clip_global_norm = self.config.float("gradient_clip_global_norm", 0.0)

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
        slot_vars.append(slot_var)
    self.tf_session.run(tf.variables_initializer(slot_vars, name="init_optim_slot_vars"))

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
