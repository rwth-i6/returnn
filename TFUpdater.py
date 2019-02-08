
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.training.optimizer import Optimizer
from tensorflow.python.ops import resource_variable_ops

from Log import log
from TFNetwork import TFNetwork
from TFUtil import tf_version_tuple, assert_min_tf_version, CustomUpdate, add_check_numerics_ops

_OptimizerClassesDict = {}  # type: dict[str,()->Optimizer]


def get_optimizer_class(class_name):
  """
  :param str class_name: e.g. "adam"
  :return: the class
  :rtype: type[Optimizer]|()->Optimizer
  """
  if not _OptimizerClassesDict:
    potential_list = list(vars(tf.train).items())
    if tf_version_tuple() >= (1, 2, 0):
      from tensorflow.contrib import opt
      potential_list += list(vars(opt).items())
    potential_list += list(globals().items())
    for name, v in potential_list:
      assert isinstance(name, str)
      if v is Optimizer:
        continue
      if not isinstance(v, type) or not issubclass(v, Optimizer):
        continue
      assert name.lower() not in _OptimizerClassesDict
      _OptimizerClassesDict[name.lower()] = v
      if name.endswith("Optimizer"):
        name = name[:-len("Optimizer")]
        assert name.lower() not in _OptimizerClassesDict
        _OptimizerClassesDict[name.lower()] = v
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
  Also relevant are config options ``debug_add_check_numerics_on_output`` and ``debug_grad_summaries``.
  """

  def __init__(self, config, network, initial_learning_rate=1.):
    """
    :param Config.Config config:
    :param TFNetwork network:
    :param float initial_learning_rate:
    """
    self.config = config
    self.learning_rate_var = tf.Variable(name="learning_rate", initial_value=0.0, trainable=False, dtype="float32")
    self.trainable_vars = []  # type: list[tf.Variable]
    self.network = network
    self.use_locking = self.config.bool("optimizer_use_locking", False)
    self.initial_learning_rate = initial_learning_rate
    if self.config.bool("decouple_constraints", False):
      # https://arxiv.org/abs/1711.05101, Fixing Weight Decay Regularization in Adam
      self.loss = network.get_total_loss()
      self.constraints = network.get_total_constraints()
    else:
      self.loss = network.get_objective()
      self.constraints = None
    self.optimizer = None  # type: WrapOptimizer
    self.optim_op = None  # type: tf.Operation
    self.optim_meta_losses = None  # type: dict[str,tf.Tensor]
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

  def set_learning_rate(self, value, session):
    """
    :param float value:
    :param tf.Session session:
    """
    from TFUtil import VariableAssigner
    VariableAssigner(self.learning_rate_var).assign(value, session=session)

  def get_current_step_learning_rate(self):
    """
    :rtype: tf.Tensor
    """
    lr = self.learning_rate_var
    if self.config.typed_dict.get("dynamic_learning_rate"):
      # To implement any kind of cyclic learning rate during the epoch. E.g.: https://arxiv.org/abs/1608.03983
      with tf.name_scope("dynamic_learning_rate"):
        from Util import CollectionReadCheckCovered
        opts = CollectionReadCheckCovered(self.config.typed_dict["dynamic_learning_rate"])
        # Currently all intervals of same step size.
        interval_steps = tf.constant(opts["interval"], name="interval", dtype=self.network.global_train_step.dtype)
        step_in_interval = tf.mod(self.network.global_train_step, interval_steps, name="step_in_interval")
        factor = tf.pow(
          tf.constant(opts["decay"], name="decay", dtype=tf.float32),
          tf.to_float(step_in_interval, name="step_in_interval_float"), name="factor")
        lr *= factor
        opts.assert_all_read()
    if self.config.is_true("use_horovod") and self.config.is_true("horovod_scale_lr"):
      import horovod.tensorflow as hvd
      lr *= hvd.size()
    return lr

  def create_optim_op(self):
    assert self.loss is not None
    assert self.trainable_vars, "no variables to update/optimize"
    from TFUtil import SyntheticGradient

    # Keep track of all current available vars.
    # The optimizer could add some, even some which are not so-called "slot-vars",
    # and we want to keep track about them.
    all_prev_existing_vars = tf.global_variables()  # type: list[tf.Variable]

    trainable_vars_for_gradients = list(self.trainable_vars)
    trainable_vars_custom_update = []  # type: list[tf.Variable]
    for v in self.trainable_vars:
      if hasattr(v, "returnn_custom_update"):
        trainable_vars_custom_update.append(v)
        trainable_vars_for_gradients.remove(v)

    if not self.optimizer:
      self.optimizer = WrapOptimizer(
        config=self.config,
        learning_rate=self.get_current_step_learning_rate(),
        global_train_step=self.network.global_train_step,
        use_locking=self.use_locking)
      self.optimizer.create_all_needed_optimizers(trainable_vars_for_gradients)

    with tf.variable_scope("optimize"):
      synthetic_gradient_scope = SyntheticGradient.enter_gradient_scope()
      apply_grads = self.optimizer.get_apply_grads_op(self.loss, trainable_vars_for_gradients)
      synthetic_gradient_scope.exit()
      self.optim_meta_losses = synthetic_gradient_scope.as_fetch_dict()
      if synthetic_gradient_scope.losses:
        with tf.name_scope("meta_loss"):
          meta_loss = tf.add_n(synthetic_gradient_scope.losses)
          meta_apply_grads = self.optimizer.get_apply_grads_op(meta_loss, trainable_vars_for_gradients)
        apply_grads = tf.group(apply_grads, meta_apply_grads)
      incr_step_op = tf.assign_add(self.network.global_train_step, 1, name="global_train_step_increment")
      self.optim_op = tf.group(apply_grads, incr_step_op, name="optim_and_step_incr")

    if trainable_vars_custom_update:
      with tf.variable_scope("custom_update"):
        updates = [self.optim_op]
        for param in trainable_vars_custom_update:
          custom_update = getattr(param, "returnn_custom_update")
          assert isinstance(custom_update, CustomUpdate)
          updates.append(custom_update.update_var(param))
        self.optim_op = tf.group(*updates)

    if self.constraints is not None:
      with tf.variable_scope("optimize_constraints"):
        with tf.variable_scope("factor"):
          factor = (self.get_current_step_learning_rate() / float(self.initial_learning_rate))
          factor *= self.config.float("decouple_constraints_factor", 0.025)
        sgd_optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=factor, use_locking=self.use_locking)
        with tf.control_dependencies([self.optim_op]):
          self.optim_op = sgd_optimizer.minimize(self.constraints, var_list=self.trainable_vars)

    if self.config.opt_typed_value("extra_updates"):
      extra_updates = self.config.typed_dict["extra_updates"]
      assert isinstance(extra_updates, dict)  # dict var_name -> function(var)
      vars_by_name = {v.name[:-2]: v for v in all_prev_existing_vars}
      extra_updates_op_list = []
      from Util import getargspec
      from TFUtil import get_var_update_ops, get_variable_grad_from_update_ops
      for var_name, func in extra_updates.items():
        func_arg_names = getargspec(func).args
        assert var_name in vars_by_name, "var with name %r not found. vars:\n%s" % (
          var_name, "\n".join(sorted(vars_by_name.keys())))
        var = vars_by_name[var_name]
        assert isinstance(var, tf.Variable)
        ops = get_var_update_ops(var, fetches=self.optim_op)
        with tf.control_dependencies(ops):
          func_kwargs = {"var": var}
          if "network" in func_arg_names:
            func_kwargs["network"] = self.network
          if "update_ops" in func_arg_names:
            func_kwargs["update_ops"] = ops
          if "grad" in func_arg_names:
            func_kwargs["grad"] = get_variable_grad_from_update_ops(var, ops)
          op = func(**func_kwargs)
          assert isinstance(op, (tf.Operation, tf.Tensor))
          extra_updates_op_list.append(op)
        self.optim_op = tf.group(self.optim_op, *extra_updates_op_list)

    slot_names_per_optimizer = self.optimizer.get_slot_names_per_optimizer()
    print("Initialize optimizer with slots %s." % slot_names_per_optimizer, file=log.v3)
    slot_vars = []
    for opt_key, slot_names in slot_names_per_optimizer.items():
      for slot_name in slot_names:
        for v in self.optimizer.filter_var_list_per_optimizer_key(trainable_vars_for_gradients, opt_key=opt_key):
          slot_var = self.optimizer.get_slot(var=v, name=slot_name)
          if slot_var is None:
            print("Warning: No slot_var found for variable %r, slot_name %r. Maybe no gradient for this var?" % (
              v, slot_name), file=log.v3)
          else:
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
      if v in all_prev_existing_vars:
        continue
      if v in self.optimizer_vars:
        continue
      other_new_vars.append(v)
    if other_new_vars:
      print("These additional variable were created by the optimizer: %s." % other_new_vars, file=log.v3)
      self.optimizer_vars += other_new_vars
    with tf.name_scope("optimizer_init_vars"):
      self.optimizer_init_vars_op = tf.variables_initializer(self.optimizer_vars, name="init_optim_slot_vars")

    if self.config.bool("debug_grad_summaries", False):
      from TFUtil import variable_summaries, get_base_name, reuse_name_scope_of_tensor
      for key in self.network.used_data_keys:
        data = self.network.extern_data.data[key]
        if data.sparse:
          continue
        with reuse_name_scope_of_tensor(data.placeholder):
          variable_summaries(data.placeholder)

    if self.config.bool("debug_add_check_numerics_ops", False):  # also see debug_add_check_numerics_on_output
      print("Adding checks for inf/nan.", file=log.v3)
      self.optim_op = tf.group(self.optim_op, add_check_numerics_ops([self.optim_op]))

    if self.config.bool("debug_save_updater_vars", False):
      print("Save updater/optimizer vars:", file=log.v3)
      print(self.optimizer_vars)
      for v in self.optimizer_vars:
        if v not in self.network.extra_vars_to_save:
          self.network.extra_vars_to_save.append(v)
      self.network.reset_saver()

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

  def init_optimizer_vars(self, session):
    """
    :param tf.Session session:
    """
    self.get_optim_op()  # make sure it is initialized
    session.run(self.optimizer_init_vars_op)


def accum_grad_multiple_step(grad, var, train_step, num_accum_steps):
  """
  :param tf.Tensor|tf.IndexedSlices grad:
  :param tf.Variable var:
  :param tf.Tensor train_step: int, scalar
  :param int num_accum_steps:
  :return: modified grad
  :rtype: tf.Tensor
  """
  from TFUtil import reuse_name_scope_of_tensor, get_base_name
  with reuse_name_scope_of_tensor(grad, postfix="/%s_accum_grad" % get_base_name(grad)):
    shape = var.get_shape().as_list()
    v = tf.get_variable(
      name="var_accum_grad", shape=shape, dtype=grad.dtype,
      initializer=tf.zeros_initializer(), trainable=False)
    return tf.cond(
      tf.less_equal(tf.mod(train_step, num_accum_steps), 0),
      lambda: tf.assign(v, grad),
      lambda: tf.assign_add(v, grad))


class WrapOptimizer:
  """
  Wraps a tf.train.Optimizer (or multiple).
  This is wrapped for a simpler interface, and also to allow for multiple optimizers.
  This class is not derived from tf.train.Optimizer itself, to keep it simple.
  """

  def __init__(self, config, learning_rate, global_train_step, use_locking):
    """
    :param Config.Config config:
    :param tf.Tensor learning_rate:
    :param tf.Tensor global_train_step:
    :param bool use_locking:
    """
    self.config = config
    self.learning_rate = learning_rate
    self.global_train_step = global_train_step
    self.use_locking = use_locking
    from collections import OrderedDict
    self.optimizers = OrderedDict()  # optimizer_opts|None -> tf.train.Optimizer

  def get_default_optimizer(self):
    """
    :rtype: tf.train.Optimizer
    """
    return self.get_default_optimizer_item(auto_create_new=False)[1]

  def get_default_optimizer_item(self, auto_create_new):
    """
    :param bool auto_create_new:
    :return: key, optimizer
    :rtype: (object, tf.train.Optimizer)
    """
    return self._get_optimizer_item_for_opts(None, auto_create_new=auto_create_new)

  def create_all_needed_optimizers(self, train_vars):
    """
    :param list[tf.Variable] train_vars:
    """
    for var in train_vars:
      self._get_optimizer_item_for_variable(var, auto_create_new=True)

  def _get_optimizer_item_for_variable(self, var, auto_create_new=False):
    """
    :param tf.Variable var:
    :param bool auto_create_new:
    :return: key, optimizer
    :rtype: (object, tf.train.Optimizer)
    """
    updater_opts = getattr(var, "RETURNN_updater_opts", None)
    if not updater_opts:
      return self.get_default_optimizer_item(auto_create_new=auto_create_new)
    from Util import CollectionReadCheckCovered
    assert isinstance(updater_opts, CollectionReadCheckCovered)
    optimizer_opts = updater_opts.get("optimizer", None)
    if not optimizer_opts:
      return self.get_default_optimizer_item(auto_create_new=auto_create_new)
    return self._get_optimizer_item_for_opts(optimizer_opts, auto_create_new=auto_create_new)

  def _get_optimizer_item_for_opts(self, optimizer_opts, auto_create_new):
    """
    :param dict[str]|str|None optimizer_opts:
    :param bool auto_create_new:
    :return: key, optimizer
    :rtype: (object, tf.train.Optimizer)
    """
    from Util import make_hashable
    key = make_hashable(optimizer_opts)
    if key in self.optimizers:
      return key, self.optimizers[key]
    assert auto_create_new, "no optimizer found for opts %r" % (optimizer_opts,)
    optimizer = self._create_optimizer(optimizer_opts)
    self.optimizers[key] = optimizer
    return key, optimizer

  def _create_optimizer(self, optimizer_opts):
    """
    :param dict[str]|str|None optimizer_opts: if dict, contains "class": opt_name. if str, then opt_name.
    :rtype: tf.train.Optimizer
    """
    if optimizer_opts is None:
      return self._create_default_optimizer()
    lr = self.learning_rate
    epsilon = self.config.float("optimizer_epsilon", 1e-16)
    use_locking = self.use_locking
    momentum = self.config.float("momentum", 0.0)
    if isinstance(optimizer_opts, str):
      optimizer_opts = {"class": optimizer_opts}
    assert isinstance(optimizer_opts, dict)
    optimizer_opts = optimizer_opts.copy()
    optim_class_name = optimizer_opts.pop("class")
    optim_class = get_optimizer_class(optim_class_name)
    from Util import collect_class_init_kwargs
    optim_class_kwargs = collect_class_init_kwargs(optim_class)
    if "epsilon" in optim_class_kwargs:
      optimizer_opts.setdefault("epsilon", epsilon)
    if "momentum" in optim_class_kwargs and momentum:
      optimizer_opts.setdefault("momentum", momentum)
    if "use_locking" in optim_class_kwargs and use_locking:
      optimizer_opts.setdefault("use_locking", use_locking)
    assert "learning_rate" not in optimizer_opts, "learning_rate will be set implicitly"
    if "learning_rate_multiplier" in optimizer_opts:
      lr *= optimizer_opts.pop("learning_rate_multiplier")
    optimizer_opts["learning_rate"] = lr
    print("Create optimizer %s with options %r." % (optim_class, optimizer_opts), file=log.v2)
    optimizer = optim_class(**optimizer_opts)
    assert isinstance(optimizer, tf.train.Optimizer)
    return optimizer

  def _create_default_optimizer(self):
    """
    :rtype: tf.train.Optimizer
    """
    lr = self.learning_rate
    epsilon = self.config.float("optimizer_epsilon", 1e-16)
    use_locking = self.use_locking
    momentum = self.config.float("momentum", 0.0)
    optim_config = self.config.typed_value("optimizer")
    if optim_config:
      assert isinstance(optim_config, (dict, str))
      optimizer = self._create_optimizer(optim_config)
    elif self.config.bool("adam", False):
      assert not momentum
      print("Create Adam optimizer.", file=log.v2)
      # Default TF values: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.
      # Default Keras values: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8.
      # Our Theano default values: beta1=0.9, beta2=0.999, epsilon=1e-16
      # https://github.com/openai/improved-gan/blob/master/imagenet/train_imagenet.py: beta1=0.5
      optimizer = tf.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon, use_locking=use_locking)
    elif self.config.bool("nadam", False):
      assert_min_tf_version((1, 2, 0), "NadamOptimizer introduced in TF 1.2.0")
      assert not momentum
      print("Create NAdam optimizer.", file=log.v2)
      # TF default values: like Adam: beta1=0.9, beta2=0.999, epsilon=1e-8
      # Our Theano default values: decay=0.004, beta1=0.9, beta2=0.999, epsilon=1e-8
      from tensorflow.contrib.opt import NadamOptimizer
      optimizer = NadamOptimizer(learning_rate=lr, epsilon=epsilon, use_locking=use_locking)
    elif self.config.bool("adadelta", False):
      assert not momentum
      print("Create Adadelta optimizer.", file=log.v2)
      optimizer = tf.train.AdadeltaOptimizer(learning_rate=lr, epsilon=epsilon, use_locking=use_locking)
    elif self.config.bool("adagrad", False):
      assert not momentum
      print("Create Adagrad optimizer.", file=log.v2)
      optimizer = tf.train.AdagradOptimizer(learning_rate=lr, use_locking=use_locking)
    elif self.config.is_of_type("rmsprop", float):
      print("Create RMSProp optimizer. With Decay %f" % (self.config.float("rmsprop", 0.9)), file=log.v2)
      optimizer = tf.train.RMSPropOptimizer(decay=self.config.float("rmsprop", 0.9), learning_rate=lr, momentum=momentum, epsilon=epsilon, use_locking=use_locking)
    elif self.config.bool("rmsprop", False):
      print("Create RMSProp optimizer.", file=log.v2)
      optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum, epsilon=epsilon, use_locking=use_locking)
    elif momentum:
      print("Create Momentum optimizer.", file=log.v2)
      optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum, use_locking=use_locking)
    else:
      print("Create SGD optimizer.", file=log.v2)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr, use_locking=use_locking)
    return optimizer

  def _partition_var_list(self, var_list):
    """
    :param list[tf.Variable] var_list:
    :return: ordered dict: opt key -> list of vars
    :rtype: dict[object,list[tf.Variable]]
    """
    from collections import OrderedDict
    res = OrderedDict()
    for var in var_list:
      key, _ = self._get_optimizer_item_for_variable(var)
      res.setdefault(key, []).append(var)
    return res

  def _compute_gradients(self, loss, var_list):
    """
    :param tf.Tensor loss:
    :param list[tf.Variable] var_list:
    :return: list of (gradient, variable) pairs
    :rtype: list[(tf.Tensor,tf.Variable)]
    """
    # AccumulateN might not be deterministic but should be faster and should require less memory.
    # We might want to make this configurable.
    if self.config.is_true("deterministic_train"):
      aggregation_method = tf.AggregationMethod.ADD_N
    else:
      aggregation_method = tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N
    res = []
    vars_per_optimizer = self._partition_var_list(var_list)
    for key, sub_var_list in vars_per_optimizer.items():
      optimizer = self.optimizers[key]
      assert isinstance(optimizer, tf.train.Optimizer)
      res.extend(optimizer.compute_gradients(loss=loss, var_list=sub_var_list, aggregation_method=aggregation_method))
    return res

  def _apply_gradients(self, grads_and_vars, opt_key, accum_grad_multiple_num_steps=0):
    """
    :param list[(tf.Tensor,tf.Variable) grads_and_vars:
    :param object opt_key:
    :param int accum_grad_multiple_num_steps:
    :rtype: tf.Operation
    """
    optimizer = self.optimizers[opt_key]
    assert isinstance(optimizer, tf.train.Optimizer)
    if accum_grad_multiple_num_steps >= 1:
      return tf.cond(
        tf.equal(
          tf.mod(self.global_train_step, accum_grad_multiple_num_steps),
          accum_grad_multiple_num_steps - 1),
        true_fn=lambda: optimizer.apply_gradients(grads_and_vars),
        false_fn=lambda: tf.no_op(),
        name="apply_grads/accum_grad_multiple_step")
    return optimizer.apply_gradients(grads_and_vars)

  def get_slot_names_per_optimizer(self):
    """
    :return: ordered dict: opt key -> slot names
    :rtype: dict[object, list[str]]
    """
    from collections import OrderedDict
    res = OrderedDict()
    for key, optimizer in self.optimizers.items():
      assert isinstance(optimizer, tf.train.Optimizer)
      res[key] = optimizer.get_slot_names()
    return res

  def filter_var_list_per_optimizer_key(self, var_list, opt_key):
    """
    :param list[tf.Variable] var_list:
    :param object opt_key: should be in self.optimizer
    :rtype: list[tf.Variable]
    """
    res = []
    for var in var_list:
      key, _ = self._get_optimizer_item_for_variable(var)
      if key == opt_key:
        res.append(var)
    return res

  def get_slot(self, var, name):
    """
    :param tf.Variable var:
    :param str name:
    :rtype: tf.Variable|None
    """
    _, opt = self._get_optimizer_item_for_variable(var)
    return opt.get_slot(var, name)

  class _GetGlobalInfo:
    def __init__(self, optimizer, all_vars, all_grads):
      """
      :param WrapOptimizer optimizer:
      :param list[tf.Variable] all_vars:
      :param list[tf.Tensor] all_grads: not necessarily the same length as all_vars
      """
      self.optimizer = optimizer
      self.all_vars = all_vars
      self.all_grads = all_grads
      self._global_grad_norm = None
      self._maximize_grad_norm_var_grads = None

    def get_global_grad_norm(self):
      """
      :rtype: tf.Tensor
      """
      if self._global_grad_norm is None:
        self._global_grad_norm = tf.global_norm(self.all_grads, name="global_grad_norm")
      return self._global_grad_norm

    def get_maximize_grad_norm_var_grads(self, factor):
      """
      :param tf.Tensor|float factor:
      :return: dict: var -> grad
      :rtype: dict[tf.Variable,tf.Tensor]
      """
      if self._maximize_grad_norm_var_grads is None:
        loss_ext = self.get_global_grad_norm() * (-factor)
        grads_and_vars_ext = self.optimizer._compute_gradients(loss_ext, var_list=self.all_vars)
        self._maximize_grad_norm_var_grads = {var: grad for (grad, var) in grads_and_vars_ext if grad is not None}
      return self._maximize_grad_norm_var_grads

    def get_maximize_grad_norm_grad(self, factor, var):
      """
      :param float|tf.Tensor factor:
      :param tf.Variable var:
      :rtype: tf.Tensor|None
      """
      return self.get_maximize_grad_norm_var_grads(factor).get(var, None)

    def clip_by_global_norm(self, grad, clip_norm):
      """
      Wraps tf.clip_by_global_norm.

      :param tf.Tensor grad:
      :param tf.Tensor|float clip_norm:
      :rtype: tf.Tensor
      """
      (grad,), _ = tf.clip_by_global_norm([grad], clip_norm=clip_norm, use_norm=self.get_global_grad_norm())
      return grad

  def _post_process_grad(self, grad, var, global_info):
    """
    :param tf.Tensor grad:
    :param tf.Variable var:
    :param WrapOptimizer._GetGlobalInfo global_info:
    :return: new grad, apply grad opts
    :rtype: tf.Tensor, dict[str]
    """
    from Util import CollectionReadCheckCovered
    updater_opts = getattr(var, "RETURNN_updater_opts", None)
    if updater_opts is None:
      updater_opts = CollectionReadCheckCovered({})
    assert isinstance(updater_opts, CollectionReadCheckCovered)

    accum_grad_multiple_num_steps = updater_opts.get(
      "accum_grad_multiple_step", self.config.int("accum_grad_multiple_step", 0))
    grad_noise = updater_opts.get("gradient_noise", self.config.float("gradient_noise", 0.0))
    grad_clip = updater_opts.get("gradient_clip", self.config.float("gradient_clip", 0.0))
    grad_clip_norm = updater_opts.get("gradient_clip_norm", self.config.float("gradient_clip_norm", 0.0))
    grad_clip_avg_norm = updater_opts.get("gradient_clip_avg_norm", self.config.float("gradient_clip_avg_norm", 0.0))
    grad_clip_global_norm = updater_opts.get(
      "gradient_clip_global_norm", self.config.float("gradient_clip_global_norm", 0.0))
    # E.g. https://github.com/openai/baselines/blob/master/baselines/deepq/simple.py:
    #   grad_norm_clipping=10 -> tf.clip_by_norm
    maximize_grad_norm = updater_opts.get("maximize_grad_norm", self.config.float("maximize_grad_norm", 0))

    if maximize_grad_norm:
      grad_ext = global_info.get_maximize_grad_norm_grad(maximize_grad_norm, var)
      if grad_ext is not None:
        grad += grad_ext

    if accum_grad_multiple_num_steps >= 1:
      grad = accum_grad_multiple_step(
        grad, var, train_step=self.global_train_step, num_accum_steps=accum_grad_multiple_num_steps)

    if updater_opts.get("debug_grad_summaries", self.config.bool("debug_grad_summaries", False)):
      from TFUtil import variable_summaries, get_base_name, reuse_name_scope_of_tensor
      with reuse_name_scope_of_tensor(grad, prefix="grads/"):
        variable_summaries(grad, name="grad_of_%s" % get_base_name(var))
      with reuse_name_scope_of_tensor(var, prefix="vars/"):
        variable_summaries(var, name=get_base_name(var))

    # Also see tf.contrib.layers.optimizers.optimize_loss() for reference.
    if updater_opts.get("gradient_nan_inf_filter", self.config.bool("gradient_nan_inf_filter", False)):
      from TFUtil import nan_to_num
      grad = nan_to_num(grad, nan_num=0.0, inf_num=0.0)
    if grad_noise:
      assert grad_noise > 0
      from TFUtil import add_scaled_noise_to_gradients
      with tf.name_scope("grad_noise"):
        (grad, var), = add_scaled_noise_to_gradients([(grad, var)], grad_noise)
    if grad_clip:
      assert grad_clip > 0
      with tf.name_scope("grad_clip"):
        grad = tf.clip_by_value(grad, -grad_clip, grad_clip)
    if grad_clip_norm:
      assert grad_clip_norm > 0
      with tf.name_scope("grad_clip_norm"):
        grad = tf.clip_by_norm(grad, grad_clip_norm)
    if grad_clip_avg_norm:
      assert grad_clip_avg_norm > 0
      with tf.name_scope("grad_clip_avg_norm"):
        grad = tf.clip_by_average_norm(grad, grad_clip_avg_norm)
    if grad_clip_global_norm:
      assert grad_clip_global_norm > 0
      with tf.name_scope("grad_clip_global_norm"):
        grad = global_info.clip_by_global_norm(grad, clip_norm=grad_clip_global_norm)

    updater_opts.assert_all_read()

    opt_key, _ = self._get_optimizer_item_for_variable(var)
    apply_grad_opts = {
      "opt_key": opt_key, "accum_grad_multiple_num_steps": accum_grad_multiple_num_steps}
    return grad, apply_grad_opts

  def get_apply_grads_op(self, loss, var_list):
    """
    :param tf.Tensor loss:
    :param list[tf.Variable] var_list:
    :return: op with all variable updates combined, using the optimizer
    :rtype: tf.Operation
    """
    # The following code is basically extended self.optimizer.minimize(), to optionally modify gradients.
    from Util import make_hashable
    if not var_list:
      return tf.no_op(name="no_grad_vars_no_op")

    grads_and_vars = self._compute_gradients(loss, var_list=var_list)
    if self.config.is_true("use_horovod") and self.config.value("horovod_reduce_type", "") == "grad":
      import horovod.tensorflow as hvd
      grads_and_vars = [
        (hvd.allreduce(grad, average=self.config.is_true("horovod_avg_grad")) if grad is not None else None, var)
        for (grad, var) in grads_and_vars]

    var_grads = {var: grad for (grad, var) in grads_and_vars if grad is not None}
    if not var_grads:
      raise Exception("no single variable to train")
    global_info = self._GetGlobalInfo(optimizer=self, all_vars=var_list, all_grads=list(var_grads.values()))
    grads_per_apply_grad_opts = {}  # dict apply_grad_opts -> list of (grad, var)
    for grad, var in grads_and_vars:
      assert var in var_list
      if grad is None:
        continue
      new_grad, apply_grad_opts = self._post_process_grad(grad=grad, var=var, global_info=global_info)
      grads_per_apply_grad_opts.setdefault(make_hashable(apply_grad_opts), []).append((new_grad, var))

    all_apply_grads = []
    assert grads_per_apply_grad_opts
    for apply_grad_opts, grads_and_vars_per_opts in grads_per_apply_grad_opts.items():
      all_apply_grads.append(self._apply_gradients(grads_and_vars_per_opts, **apply_grad_opts))
    if len(all_apply_grads) == 1:
      return all_apply_grads[0]
    return tf.group(*all_apply_grads)


class _BaseCustomOptimizer(Optimizer):
  """
  Base class for our own optimizer implementations.
  This simplifies the interface to be implemented a bit from :class:`Optimizer`.
  """

  def __init__(self, learning_rate, use_locking=False, name=None):
    """Construct a new optimizer.

    Args:
      learning_rate: A Tensor or a floating point value.  The learning
        rate to use.
      use_locking: If True use locks for update operations.
      name: Optional name prefix for the operations created when applying
        gradients. Defaults to `self.__class__.__name__`.
    """
    if name is None:
      name = self.__class__.__name__
    super(_BaseCustomOptimizer, self).__init__(use_locking, name)
    self._learning_rate = learning_rate

  def _prepare(self):
    self._learning_rate_tensor = tf.convert_to_tensor(self._learning_rate, name="learning_rate")

  def _apply(self, grad, var, indices=None):
    """
    :param tf.Tensor grad:
    :param tf.Variable|resource_variable_ops.ResourceVariable var:
    :param tf.Tensor|None indices: if this is a sparse update, the indices of the grad values
    :return: update
    :rtype: tf.Tensor|tf.Operation
    """
    raise NotImplementedError

  def _apply_dense(self, grad, var):
    return self._apply(grad=grad, var=var)

  def _resource_apply_dense(self, grad, handle):
    return self._apply_dense(grad=grad, var=handle)

  def _resource_apply_sparse_duplicate_indices(self, grad, handle, indices):
    return self._apply(grad=grad, var=handle, indices=indices)

  def _resource_apply_sparse(self, grad, handle, indices):
    return self._resource_apply_sparse_duplicate_indices(grad=grad, handle=handle, indices=indices)

  def _apply_sparse_duplicate_indices(self, grad, var):
    return self._apply(grad=grad.values, var=var, indices=grad.indices)

  def _apply_sparse(self, grad, var):
    return self._apply_sparse_duplicate_indices(grad=grad, var=var)

  def _assign_add(self, ref, updates, indices=None):
    if indices is not None:
      if isinstance(ref, tf.Variable):
        return tf.scatter_add(ref, indices, updates, use_locking=self._use_locking)
      elif isinstance(ref, resource_variable_ops.ResourceVariable):
        with tf.control_dependencies([resource_variable_ops.resource_scatter_add(ref.handle, indices, updates)]):
          return ref.value()
      else:
        raise TypeError("did not expect type %r" % type(ref))
    else:
      return tf.assign_add(ref, updates, use_locking=self._use_locking)

  def _assign_sub(self, ref, updates, indices=None):
    if indices is not None:
      if isinstance(ref, tf.Variable):
        return tf.scatter_sub(ref, indices, updates, use_locking=self._use_locking)
      elif isinstance(ref, resource_variable_ops.ResourceVariable):
        with tf.control_dependencies([resource_variable_ops.resource_scatter_add(ref.handle, indices, -updates)]):
          return ref.value()
      else:
        raise TypeError("did not expect type %r" % type(ref))
    else:
      return tf.assign_sub(ref, updates, use_locking=self._use_locking)

  def _gather(self, dense, indices=None):
    if indices is not None:
      return tf.gather(dense, indices=indices)
    return dense


class CustomGradientDescentOptimizer(_BaseCustomOptimizer):
  """
  Just an example implementation for simple gradient descent.
  """

  def _apply(self, grad, var, indices=None):
    """
    :param tf.Tensor grad:
    :param tf.Variable|resource_variable_ops.ResourceVariable var:
    :param tf.Tensor|None indices: if this is a sparse update, the indices of the grad values
    :return: update
    :rtype: tf.Tensor|tf.Operation
    """
    lr = tf.cast(self._learning_rate_tensor, grad.dtype.base_dtype)
    return self._assign_sub(ref=var, updates=lr * grad, indices=indices).op


class NormalizedSGD(CustomGradientDescentOptimizer):
  """
  All grads are L2 normalized (via :func:`tf.nn.l2_normalize`), otherwise it's standard SGD.
  Via: https://github.com/kmkolasinski/deep-learning-notes/tree/master/max-normed-optimizer
  """

  def _apply(self, grad, var, indices=None):
    """
    :param tf.Tensor grad:
    :param tf.Variable|resource_variable_ops.ResourceVariable var:
    :param tf.Tensor|None indices: if this is a sparse update, the indices of the grad values
    :return: update
    :rtype: tf.Tensor|tf.Operation
    """
    return super(NormalizedSGD, self)._apply(grad=tf.nn.l2_normalize(grad, None), var=var, indices=indices)


class NeuralOptimizer1(_BaseCustomOptimizer):
  """
  Via Neural Optimizer Search with Reinforcement Learning (http://proceedings.mlr.press/v70/bello17a/bello17a.pdf).

  Equivalent to the optimizer g * exp(sign(g) * sign(m)), we use:

    g * where(sign(g) == sign(m), 1.0, decrease_factor)

  where m is the running average of g.

  Calculation of m: m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  Same beta1 default as in Adam and in the paper: beta1=0.9
  """

  def __init__(self, beta1=0.9, decrease_factor=0.1, **kwargs):
    """
    :param float beta1: used for the running average of m
    :param float decrease_factor: in the original paper, it is e^-2 ~= 0.135
    """
    super(NeuralOptimizer1, self).__init__(**kwargs)
    self._beta1 = beta1
    self._decrease_factor = decrease_factor

  def _prepare(self):
    super(NeuralOptimizer1, self)._prepare()
    self._beta1_t = tf.convert_to_tensor(self._beta1, name="beta1")

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "m", self._name)

  def _apply(self, grad, var, indices=None):
    lr = tf.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    m = self.get_slot(var, "m")
    # m_t = beta1 * m + (1 - beta1) * g_t
    beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = tf.assign(m, m * beta1_t, use_locking=self._use_locking)
    with tf.control_dependencies([m_t]):
      m_t = self._assign_add(m, updates=m_scaled_g_values, indices=indices)
    # update = lr * grad * where(...)
    m_gathered = self._gather(m_t, indices=indices)
    ones = tf.ones_like(grad)
    update = lr * grad * tf.where(tf.equal(tf.sign(m_gathered), tf.sign(grad)), ones, ones * self._decrease_factor)
    var_update = self._assign_sub(ref=var, updates=update, indices=indices)
    return tf.group(*[var_update, m_t])


class GradVarianceScaledOptimizer(_BaseCustomOptimizer):
  """
  Let m be the running average of g.
  Calculation of m: m_t <- beta1 * m_{t-1} + (1 - beta1) * g
  Same beta1 default as in Adam and in the paper: beta1=0.9

  Let v be the running average of the variance of g, i.e. of (g - m)^2.
  """

  def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
    """
    :param float beta1: used for the running average of g (m)
    :param float beta2: used for the running average of variance of g (v)
    :param float epsilon:
    """
    super(GradVarianceScaledOptimizer, self).__init__(**kwargs)
    self._beta1 = beta1
    self._beta2 = beta2
    self._epsilon = epsilon

  def _prepare(self):
    super(GradVarianceScaledOptimizer, self)._prepare()
    self._beta1_t = tf.convert_to_tensor(self._beta1, name="beta1")
    self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")
    self._epsilon_t = tf.convert_to_tensor(self._epsilon, name="epsilon")

  def _create_slots(self, var_list):
    for v in var_list:
      self._zeros_slot(v, "m", self._name)
      self._zeros_slot(v, "v", self._name)

  def _apply(self, grad, var, indices=None):
    lr = tf.cast(self._learning_rate_tensor, var.dtype.base_dtype)
    m = self.get_slot(var, "m")
    v = self.get_slot(var, "v")
    beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
    beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
    epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)

    # m_t = beta1 * m + (1 - beta1) * g_t
    m_scaled_g_values = grad * (1 - beta1_t)
    m_t = tf.assign(m, m * beta1_t, use_locking=self._use_locking)
    with tf.control_dependencies([m_t]):
      m_t = self._assign_add(m, updates=m_scaled_g_values, indices=indices)
    m_gathered = self._gather(m_t, indices=indices)

    # Also see tf.nn.moments.
    variance = tf.squared_difference(grad, m_gathered)

    # v_t = beta2 * v + (1 - beta2) * variance
    v_scaled_new_values = variance * (1 - beta2_t)
    v_t = tf.assign(v, v * beta2_t, use_locking=self._use_locking)
    with tf.control_dependencies([v_t]):
      v_t = self._assign_add(v, updates=v_scaled_new_values, indices=indices)
    v_gathered = self._gather(v_t, indices=indices)

    # update = lr * grad * v / (variance + eps)
    factor = v_gathered / (variance + epsilon_t)
    # with tf.get_default_graph().colocate_with(None, True):
    #   with tf.control_dependencies([tf.Print(factor, [tf.reduce_min(factor), tf.reduce_max(factor), tf.reduce_mean(factor)])]):
    #     factor = tf.identity(factor)
    update = lr * grad * tf.minimum(factor, 1.0)
    var_update = self._assign_sub(ref=var, updates=update, indices=indices)
    return tf.group(*[var_update, m_t])


class AMSGradOptimizer(tf.train.Optimizer):
  """
  https://colab.research.google.com/notebook#fileId=1xXFAuHM2Ae-OmF5M8Cn9ypGCa_HHBgfG&scrollTo=N1-2wPHN1Otn
  https://openreview.net/pdf?id=ryQu7f-RZ
  https://keras.io/optimizers/
  http://ruder.io/deep-learning-optimization-2017/index.html#fixingtheexponentialmovingaverage
  https://github.com/taki0112/AMSGrad-Tensorflow
  """
  def __init__(self, learning_rate=0.001, decay=False, beta1=0.9, beta2=0.99,
               epsilon=0.0, var_list=[]):
    self.learning_rate = learning_rate
    self.decay = decay
    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    self.var_list = var_list
    self.m = {}
    self.v = {}
    self.v_hat = {}
    self.t = tf.Variable(0.0, trainable=False)

    for var in self.var_list:
      self.m[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)
      self.v[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)
      self.v_hat[var] = tf.Variable(tf.zeros(tf.shape(var.initial_value)), trainable=False)

  def apply_gradients(self, gradient_variables):
    with tf.control_dependencies([self.t.assign_add(1.0)]):
      learning_rate = self.learning_rate
      if self.decay:
        learning_rate /= tf.sqrt(self.t)
      update_ops = []

      for (g, var) in gradient_variables:
        m = self.m[var].assign(self.beta1 * self.m[var] + (1 - self.beta1) * g)
        v = self.v[var].assign(self.beta2 * self.v[var] + (1 - self.beta2) * g * g)
        v_hat = self.v_hat[var].assign(tf.maximum(self.v_hat[var], v))

        update = -learning_rate * m / (self.epsilon + tf.sqrt(v_hat))
        update_ops.append(var.assign_add(update))

      return tf.group(*update_ops)
