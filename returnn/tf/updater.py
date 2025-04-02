"""
This module covers the optimizer (SGD, Adam, etc) logic,
and model param update logic in general.
"""

from __future__ import annotations

import typing
from collections import OrderedDict
import tensorflow as tf
from tensorflow.python.ops import resource_variable_ops

from returnn.log import log
from returnn.util.basic import BehaviorVersion
from returnn.tf.network import TFNetwork
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.util.basic import tf_version_tuple, assert_min_tf_version, CustomUpdate, add_check_numerics_ops

Optimizer = tf_compat.v1.train.Optimizer
KerasOptimizer = None
if tf_compat.v2:
    KerasOptimizer = tf_compat.v2.optimizers.Optimizer

_OptimizerClassesDictInitialized = False
_OptimizerClassesDict = {}  # type: typing.Dict[str, typing.Type[typing.Union[Optimizer, KerasOptimizer]]]


def _init_optimizer_classes_dict():
    global _OptimizerClassesDictInitialized
    if _OptimizerClassesDictInitialized:
        return
    _OptimizerClassesDictInitialized = True
    # Build up all potential candidates of such classes.
    # We will filter it below.
    # The order of the list matters, and the first optimizer is used.
    # E.g. our own NadamOptimizer here will be preferred over others.
    potential_list = list(globals().items())
    potential_list += list(vars(tf_compat.v1.train).items())
    if tf_version_tuple() >= (1, 2, 0):
        try:
            from tensorflow.contrib import opt  # noqa

            potential_list += list(vars(opt).items())
        except ImportError:  # TF 2
            pass
    allowed_types = (Optimizer,)
    if KerasOptimizer:
        potential_list += list(vars(tf_compat.v2.keras.optimizers).items())
        allowed_types += (KerasOptimizer,)
        # Special name for Nadam in Keras, as we provide our own Nadam here.
        potential_list += [("NadamKeras", tf_compat.v2.keras.optimizers.Nadam)]
    for name, v in potential_list:
        assert isinstance(name, str)
        # We might have duplicate names if TF v1 and v2 are mixed, etc.
        # Allow this at this point.
        if name.lower() in _OptimizerClassesDict:
            continue
        if v is Optimizer:
            continue
        if KerasOptimizer and v is KerasOptimizer:
            continue
        if not isinstance(v, type):
            continue
        if not issubclass(v, allowed_types):
            continue
        if v is _KerasOptimizerWrapper:
            continue
        register_optimizer_class(v, name=name)


def _check_valid_optimizer(optimizer_class):
    """
    :param type optimizer_class:
    """
    if KerasOptimizer:
        assert issubclass(optimizer_class, (Optimizer, KerasOptimizer))
    else:
        assert issubclass(optimizer_class, Optimizer)


def register_optimizer_class(cls, name=None):
    """
    :param type[Optimizer|KerasOptimizer] cls:
    :param str|None name:
    """
    _init_optimizer_classes_dict()
    if not name:
        name = cls.__name__
    _check_valid_optimizer(cls)
    assert name.lower() not in _OptimizerClassesDict
    _OptimizerClassesDict[name.lower()] = cls
    if name.endswith("Optimizer"):
        name = name[: -len("Optimizer")]
        assert name.lower() not in _OptimizerClassesDict
        _OptimizerClassesDict[name.lower()] = cls


def get_optimizer_class(class_name):
    """
    :param str|function|type[Optimizer|KerasOptimizer] class_name: e.g. "adam"
    :return: the class
    :rtype: type[Optimizer|KerasOptimizer]
    """
    _init_optimizer_classes_dict()
    if isinstance(class_name, type):
        _check_valid_optimizer(class_name)
        return class_name
    if callable(class_name):
        class_name = class_name()
    assert isinstance(class_name, str)
    return _OptimizerClassesDict[class_name.lower()]


class Updater:
    """
    This will create the :class:`tf.compat.v1.train.Optimizer` instance given the config
    and the update-op for all trainable vars.
    See the code of :func:`Updater.create_optimizer` for valid config options.

    Wraps one or multiple tf.compat.v1.train.Optimizer, and extends it by some further functionality.

    Note: `Vincent Vanhoucke says <https://github.com/tensorflow/tensorflow/issues/323#issuecomment-159116515>`_,
    in case you get nans, consider increasing the epsilon (for Adam, Nadam and similar).
    This is the config option ``optimizer_epsilon``.
    In some places in our Theano code, 1e-16 is our default epsilon, in some other parts, 1e-8 is.
    1e-8 might be more stable. Or even 1e-6.
    Note that when the gradient is suddenly zero in one step, the update can be proportional to lr / eps.

    From the :class:`tf.compat.v1.train.AdamOptimizer` documentation:

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

    def __init__(self, config, network, initial_learning_rate=1.0):
        """
        :param returnn.config.Config config:
        :param TFNetwork network:
        :param float initial_learning_rate:
        """
        self.config = config
        self.learning_rate_var = tf.Variable(name="learning_rate", initial_value=0.0, trainable=False, dtype="float32")
        self.initial_learning_rate = initial_learning_rate
        self.learning_rate = None  # type: typing.Optional[tf.Tensor]
        self.trainable_vars = []  # type: typing.List[tf.Variable]
        self.network = network
        self.global_train_step = self.network.global_train_step
        self.use_locking = self.config.bool("optimizer_use_locking", False)
        self.loss = network.get_objective()
        # https://arxiv.org/abs/1711.05101, Fixing Weight Decay Regularization in Adam
        self.decouple_constraints = self.config.bool("decouple_constraints", False)
        self.optimizers = OrderedDict()  # optimizer_opts|None -> tf.compat.v1.train.Optimizer
        self.optim_op = None  # type: typing.Optional[tf.Operation]
        self.optim_meta_losses_dict = None  # type: typing.Optional[typing.Dict[str,tf.Tensor]]
        self.optimizer_vars = []  # type: typing.List[tf.Variable]
        self.optimizer_init_vars_op = None  # type: typing.Optional[tf.Operation]

        # After graph was build: look if it only uses deterministic ops
        if self.config.is_true("deterministic_train"):
            non_det_ops = tf_util.get_non_deterministic_ops_from_graph()
            if non_det_ops:
                print("WARNING: The graph uses these non deterministic ops: {}".format(non_det_ops), file=log.v1)

    def reset_optim_op(self):
        """
        Call this if sth is changed which the optim_op depends on.
        See self.create_optim_op().
        """
        self.optim_op = None  # type: typing.Optional[tf.Operation]

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
        :param tf.compat.v1.Session session:
        """
        from returnn.tf.util.basic import VariableAssigner

        VariableAssigner(self.learning_rate_var).assign(value, session=session)

    def get_current_step_learning_rate(self):
        """
        :rtype: tf.Tensor
        """
        lr = self.learning_rate_var
        if callable(self.config.typed_dict.get("dynamic_learning_rate")):
            import inspect

            learning_rate_function = self.config.typed_dict.get("dynamic_learning_rate")
            signature = inspect.signature(learning_rate_function)
            assert any(
                [arg.kind == inspect.Parameter.VAR_KEYWORD for arg in signature.parameters.values()]
            ), "please specify **kwargs in dynamic_learning_rate for future compatibility"
            if "epoch" in signature.parameters:
                raise NotImplementedError("TF updater: dynamic_learning_rate with epoch not supported currently")
            lr = learning_rate_function(
                network=self.network, global_train_step=self.global_train_step, learning_rate=lr
            )
        elif self.config.typed_dict.get("dynamic_learning_rate"):
            # To implement any kind of cyclic learning rate during the epoch. E.g.: https://arxiv.org/abs/1608.03983
            with tf.name_scope("dynamic_learning_rate"):
                from returnn.util.basic import CollectionReadCheckCovered

                opts = CollectionReadCheckCovered(self.config.typed_dict["dynamic_learning_rate"])
                # Currently all intervals of same step size.
                interval_steps = tf.constant(opts["interval"], name="interval", dtype=self.global_train_step.dtype)
                step_in_interval = tf_compat.v1.mod(self.global_train_step, interval_steps, name="step_in_interval")
                factor = tf.pow(
                    tf.constant(opts["decay"], name="decay", dtype=tf.float32),
                    tf.cast(step_in_interval, dtype=tf.float32, name="step_in_interval_float"),
                    name="factor",
                )
                lr = lr * factor
                opts.assert_all_read()
        if self.config.is_true("use_horovod") and self.config.is_true("horovod_scale_lr"):
            # noinspection PyPackageRequirements,PyUnresolvedReferences
            import horovod.tensorflow as hvd

            lr = lr * hvd.size()
        return lr

    def create_optim_op(self):
        """
        Creates the optimize TF op.

        :return: nothing, will just set self.optim_op
        """
        assert isinstance(self.loss, tf.Tensor), "no loss defined?"
        assert self.trainable_vars, "no variables to update/optimize"
        from returnn.tf.util.basic import MetaLosses
        from returnn.tf.util.gradient_checkpoint import prepare_gradient_checkpointing

        prepare_gradient_checkpointing()

        # Keep track of all current available vars.
        # The optimizer could add some, even some which are not so-called "slot-vars",
        # and we want to keep track about them.
        all_prev_existing_vars = tf_compat.v1.global_variables()  # type: typing.List[tf.Variable]

        trainable_vars_for_gradients = list(self.trainable_vars)
        trainable_vars_custom_update = []  # type: typing.List[tf.Variable]
        for v in self.trainable_vars:
            if hasattr(v, "returnn_custom_update"):
                trainable_vars_custom_update.append(v)
                trainable_vars_for_gradients.remove(v)

        self.learning_rate = self.get_current_step_learning_rate()
        self.optimizers.clear()
        self.create_all_needed_optimizers(trainable_vars_for_gradients)

        with tf_compat.v1.variable_scope("optimize"):
            meta_losses_scope = MetaLosses.enter_gradient_scope()
            apply_grads = self.get_apply_grads_op(self.loss, trainable_vars_for_gradients)
            meta_losses_scope.exit()
            self.optim_meta_losses_dict = meta_losses_scope.losses_as_fetch_dict()
            if meta_losses_scope.losses:
                with tf.name_scope("meta_loss"):
                    meta_loss = meta_losses_scope.summed_loss_for_optimization()
                    meta_apply_grads = self.get_apply_grads_op(meta_loss, trainable_vars_for_gradients)
                apply_grads = tf.group(apply_grads, meta_apply_grads)
            self.optim_op = apply_grads

        if trainable_vars_custom_update:
            with tf_compat.v1.variable_scope("custom_update"):
                updates = [self.optim_op]
                for param in trainable_vars_custom_update:
                    custom_update = getattr(param, "returnn_custom_update")
                    assert isinstance(custom_update, CustomUpdate)
                    updates.append(custom_update.update_var(param))
                self.optim_op = tf.group(*updates)

        if self.config.opt_typed_value("extra_updates"):
            extra_updates = self.config.typed_dict["extra_updates"]
            assert isinstance(extra_updates, dict)  # dict var_name -> function(var)
            vars_by_name = {v.name[:-2]: v for v in all_prev_existing_vars}
            extra_updates_op_list = []
            from returnn.util.basic import getargspec
            from returnn.tf.util.basic import get_var_update_ops, get_variable_grad_from_update_ops

            for var_name, func in extra_updates.items():
                func_arg_names = getargspec(func).args
                assert var_name in vars_by_name, "var with name %r not found. vars:\n%s" % (
                    var_name,
                    "\n".join(sorted(vars_by_name.keys())),
                )
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

        slot_names_per_optimizer = self.get_slot_names_per_optimizer()
        slot_vars = []
        for opt_key, slot_names in slot_names_per_optimizer.items():
            print("Initialize optimizer (%s) with slots %s." % (opt_key or "default", slot_names), file=log.v3)
            for slot_name in slot_names:
                for v in self.filter_var_list_per_optimizer_key(trainable_vars_for_gradients, opt_key=opt_key):
                    slot_var = self.get_slot(var=v, name=slot_name)
                    if slot_var is None:
                        print(
                            "Warning: No slot_var found for variable %r, slot_name %r. Maybe no gradient for this var?"
                            % (v, slot_name),
                            file=log.v3,
                        )
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
        for v in tf_compat.v1.global_variables():
            if v in all_prev_existing_vars:
                continue
            if v in self.optimizer_vars:
                continue
            other_new_vars.append(v)
        if other_new_vars:
            print("These additional variable were created by the optimizer: %s." % other_new_vars, file=log.v3)
            self.optimizer_vars += other_new_vars
        with tf.name_scope("optimizer_init_vars"):
            self.optimizer_init_vars_op = tf_compat.v1.variables_initializer(
                self.optimizer_vars, name="init_optim_slot_vars"
            )

        if self.config.bool_or_other("debug_grad_summaries", False):
            from returnn.tf.util.basic import variable_summaries, get_base_name, reuse_name_scope_of_tensor

            for key in self.network.used_data_keys:
                data = self.network.extern_data.data[key]
                if data.sparse:
                    continue
                with reuse_name_scope_of_tensor(data.placeholder):
                    variable_summaries(data.placeholder)

        if self.config.bool("debug_add_check_numerics_ops", False):  # also see debug_add_check_numerics_on_output
            print("Adding checks for inf/nan.", file=log.v3)
            self.optim_op = tf.group(self.optim_op, add_check_numerics_ops([self.optim_op]))

        # Do this at the very end.
        with tf.control_dependencies([self.optim_op, self.network.global_train_step]):
            incr_step_op = tf_compat.v1.assign_add(
                self.network.global_train_step_var, 1, name="global_train_step_increment"
            )
        self.optim_op = tf.group(self.optim_op, incr_step_op, name="optim_and_step_incr")

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
        :param tf.compat.v1.Session session:
        """
        self.get_optim_op()  # make sure it is initialized
        session.run(self.optimizer_init_vars_op)

    def get_default_optimizer(self):
        """
        :rtype: tf.compat.v1.train.Optimizer
        """
        return self.get_default_optimizer_item(auto_create_new=False)[1]

    def get_default_optimizer_item(self, auto_create_new):
        """
        :param bool auto_create_new:
        :return: key, optimizer
        :rtype: (object, tf.compat.v1.train.Optimizer)
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
        :rtype: (object, tf.compat.v1.train.Optimizer)
        """
        updater_opts = getattr(var, "RETURNN_updater_opts", None)
        if not updater_opts:
            return self.get_default_optimizer_item(auto_create_new=auto_create_new)
        from returnn.util.basic import CollectionReadCheckCovered

        assert isinstance(updater_opts, CollectionReadCheckCovered)
        optimizer_opts = updater_opts.get("optimizer", None)
        if not optimizer_opts:
            return self.get_default_optimizer_item(auto_create_new=auto_create_new)
        assert isinstance(optimizer_opts, dict)
        return self._get_optimizer_item_for_opts(optimizer_opts, auto_create_new=auto_create_new)

    def _get_optimizer_item_for_opts(self, optimizer_opts, auto_create_new):
        """
        :param dict[str]|str|None optimizer_opts:
        :param bool auto_create_new:
        :return: key, optimizer
        :rtype: (object, tf.compat.v1.train.Optimizer)
        """
        from returnn.util.basic import make_hashable

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
        :rtype: tf.compat.v1.train.Optimizer
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
        if "class" in optimizer_opts:
            optim_class_name = optimizer_opts.pop("class")
            optim_class = get_optimizer_class(optim_class_name)
        else:
            _, default_opt = self._get_optimizer_item_for_opts(None, auto_create_new=True)
            optim_class = default_opt.__class__
        from returnn.util.basic import collect_class_init_kwargs

        optim_class_kwargs = collect_class_init_kwargs(optim_class)
        if "epsilon" in optim_class_kwargs:
            optimizer_opts.setdefault("epsilon", epsilon)
        if "momentum" in optim_class_kwargs and momentum:
            optimizer_opts.setdefault("momentum", momentum)
        if "use_locking" in optim_class_kwargs and use_locking:
            optimizer_opts.setdefault("use_locking", use_locking)
        assert "learning_rate" not in optimizer_opts, "learning_rate will be set implicitly"
        if "learning_rate_multiplier" in optimizer_opts:
            lr = lr * optimizer_opts.pop("learning_rate_multiplier")
        optimizer_opts["learning_rate"] = lr
        print("Create optimizer %s with options %r." % (optim_class, optimizer_opts), file=log.v2)
        if KerasOptimizer and issubclass(optim_class, KerasOptimizer):
            optim_class = _KerasOptimizerWrapper.get_factory(optim_class)
        optimizer = optim_class(**optimizer_opts)
        assert isinstance(optimizer, Optimizer)
        return optimizer

    def _create_default_optimizer(self):
        """
        :rtype: tf.compat.v1.train.Optimizer
        """
        lr = self.learning_rate
        epsilon = self.config.float("optimizer_epsilon", 1e-16)
        use_locking = self.use_locking
        momentum = self.config.float("momentum", 0.0)
        optim_config = self.config.typed_value("optimizer")
        behavior_valid_optimizer = False  # only via "optimizer" or nothing at all (default SGD)
        if optim_config:
            assert isinstance(optim_config, (dict, str))
            assert "class" in optim_config
            optimizer = self._create_optimizer(optim_config)
            behavior_valid_optimizer = True
        elif self.config.bool("adam", False):
            assert not momentum
            print("Create Adam optimizer.", file=log.v2)
            # Default TF values: learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8.
            # Default Keras values: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8.
            # Our Theano default values: beta1=0.9, beta2=0.999, epsilon=1e-16
            # https://github.com/openai/improved-gan/blob/master/imagenet/train_imagenet.py: beta1=0.5
            optimizer = tf_compat.v1.train.AdamOptimizer(learning_rate=lr, epsilon=epsilon, use_locking=use_locking)
        elif self.config.bool("nadam", False):
            assert_min_tf_version((1, 2, 0), "NadamOptimizer introduced in TF 1.2.0")
            assert not momentum
            print("Create NAdam optimizer.", file=log.v2)
            # TF default values: like Adam: beta1=0.9, beta2=0.999, epsilon=1e-8
            # Our Theano default values: decay=0.004, beta1=0.9, beta2=0.999, epsilon=1e-8
            try:
                from tensorflow.contrib.opt import NadamOptimizer  # noqa

                optimizer = NadamOptimizer(learning_rate=lr, epsilon=epsilon, use_locking=use_locking)
            except ImportError:  # TF 2
                optimizer = tf.keras.optimizers.Nadam(learning_rate=lr, epsilon=epsilon)
                optimizer = _KerasOptimizerWrapper(optimizer)
        elif self.config.bool("adadelta", False):
            assert not momentum
            print("Create Adadelta optimizer.", file=log.v2)
            optimizer = tf_compat.v1.train.AdadeltaOptimizer(learning_rate=lr, epsilon=epsilon, use_locking=use_locking)
        elif self.config.bool("adagrad", False):
            assert not momentum
            print("Create Adagrad optimizer.", file=log.v2)
            optimizer = tf_compat.v1.train.AdagradOptimizer(learning_rate=lr, use_locking=use_locking)
        elif self.config.is_of_type("rmsprop", float):
            print("Create RMSProp optimizer. With Decay %f" % (self.config.float("rmsprop", 0.9)), file=log.v2)
            optimizer = tf_compat.v1.train.RMSPropOptimizer(
                decay=self.config.float("rmsprop", 0.9),
                learning_rate=lr,
                momentum=momentum,
                epsilon=epsilon,
                use_locking=use_locking,
            )
        elif self.config.bool("rmsprop", False):
            print("Create RMSProp optimizer.", file=log.v2)
            optimizer = tf_compat.v1.train.RMSPropOptimizer(
                learning_rate=lr, momentum=momentum, epsilon=epsilon, use_locking=use_locking
            )
        elif momentum:
            print("Create Momentum optimizer.", file=log.v2)
            optimizer = tf_compat.v1.train.MomentumOptimizer(
                learning_rate=lr, momentum=momentum, use_locking=use_locking
            )
        else:
            print("Create SGD optimizer.", file=log.v2)
            optimizer = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=lr, use_locking=use_locking)
            behavior_valid_optimizer = True
        BehaviorVersion.require(
            condition=behavior_valid_optimizer,
            message="Please define an optimizer specifically via the 'optimizer=...' parameter",
            version=2,
        )
        return optimizer

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
        # Note: Do not call compute_gradients for each optimizer, because that would result in multiple independent
        # backprops, and would be much slower and require more memory. Also, it should not be needed.
        # So instead, just call from the default optimizer. This should almost always be correct,
        # as this is not much more than a wrapper around tf.gradients.
        # (Some special optimizers would add special losses though.)
        default_opt = self.get_default_optimizer()
        return default_opt.compute_gradients(loss=loss, var_list=var_list, aggregation_method=aggregation_method)

    def _apply_gradients(self, grads_and_vars, opt_key, accum_grad_multiple_num_steps=0):
        """
        :param list[(tf.Tensor,tf.Variable) grads_and_vars:
        :param object opt_key:
        :param int accum_grad_multiple_num_steps:
        :rtype: tf.Operation
        """
        optimizer = self.optimizers[opt_key]
        assert isinstance(optimizer, Optimizer)
        if accum_grad_multiple_num_steps >= 1:
            return tf.cond(
                tf.equal(
                    tf_compat.v1.mod(self.global_train_step, accum_grad_multiple_num_steps),
                    accum_grad_multiple_num_steps - 1,
                ),
                true_fn=lambda: optimizer.apply_gradients(grads_and_vars),
                false_fn=lambda: tf.no_op(),
                name="apply_grads/accum_grad_multiple_step",
            )
        return optimizer.apply_gradients(grads_and_vars)

    def get_slot_names_per_optimizer(self):
        """
        :return: ordered dict: opt key -> slot names
        :rtype: dict[object, list[str]]
        """
        from collections import OrderedDict

        res = OrderedDict()
        for key, optimizer in self.optimizers.items():
            assert isinstance(optimizer, Optimizer)
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
        def __init__(self, updater, all_vars, var_grads):
            """
            :param Updater updater:
            :param list[tf.Variable] all_vars:
            :param dict[tf.Variable,tf.Tensor] var_grads:
            """
            self.updater = updater
            self.all_vars = all_vars
            self.var_grads = var_grads
            self.all_grads = list(var_grads.values())  # not necessarily the same length as all_vars
            self.vars_by_tag = self._build_vars_by_tag_dict()  # tag name -> set of vars
            self._l2loss_cache = {}
            self._global_grad_norm = None
            self._global_grad_norm_per_tag = {}
            self._maximize_grad_norm_var_grads = None

        def _build_vars_by_tag_dict(self):
            """
            :return: tag name -> set of vars
            :rtype: dict[str,set[tf.Variable]]
            """
            res = {}
            for var in self.all_vars:
                opts = self.updater._get_updater_opts_from_var(var)
                var_tags = opts.get("tags", [])
                for tag in var_tags:
                    res.setdefault(tag, set()).add(var)
            return res

        def get_l2loss(self, x):
            """
            :param tf.Tensor|tf.IndexedSlices x:
            :return: tf.nn.l2_loss(x) (which is l2norm(x)**2 / 2, or sum(x**2) / 2)
            :rtype: tf.Tensor
            """
            if x not in self._l2loss_cache:
                with tf_compat.v1.colocate_with(x):
                    values = x
                    if isinstance(values, tf.IndexedSlices):
                        values = values.values
                    self._l2loss_cache[x] = tf.nn.l2_loss(values)
            return self._l2loss_cache[x]

        def _global_norm(self, grads):
            """
            :param list[tf.Tensor]|set[tf.Tensor] grads:
            :rtype: tf.Tensor
            """
            if not isinstance(grads, (list, tuple)):
                grads = sorted(grads, key=lambda v: v.name)  # make some deterministic order
            # We want tf.global_norm(values), which is sqrt(sum([l2norm(t)**2 for t in values])),
            # but we use self.get_l2loss which caches the calculation of l2norm.
            # Thus this is tf.global_norm somewhat reproduced:
            with tf.name_scope("global_norm"):
                half_squared_norms = [self.get_l2loss(grad) for grad in grads if grad is not None]
                half_squared_norm = tf.reduce_sum(tf.stack(half_squared_norms))
                norm = tf.sqrt(half_squared_norm * tf.constant(2.0, dtype=half_squared_norm.dtype), name="global_norm")
            return norm

        def get_global_grad_norm(self, tag=None):
            """
            :param str|None tag:
            :return: sqrt(sum(t**2 for t in all_grads))
            :rtype: tf.Tensor
            """
            if tag:
                return self.get_global_grad_norm_per_tag(tag=tag)
            if self._global_grad_norm is None:
                self._global_grad_norm = self._global_norm(self.all_grads)
            return self._global_grad_norm

        def get_global_grad_norm_per_tag(self, tag):
            """
            :param str tag:
            :return: sqrt(sum(t**2 for t in grads_of_vars_of_this_tag))
            :rtype: tf.Tensor
            """
            if tag not in self._global_grad_norm_per_tag:
                from returnn.tf.util.basic import get_valid_scope_name_from_str

                with tf.name_scope("global_norm_for_tag_%s" % get_valid_scope_name_from_str(tag)):
                    norm = self._global_norm({self.var_grads[var] for var in self.vars_by_tag[tag]})
                if self.updater.config.bool_or_other("debug_grad_summaries", False):
                    tf_compat.v1.summary.scalar("global_norm_for_tag_%s" % get_valid_scope_name_from_str(tag), norm)
                self._global_grad_norm_per_tag[tag] = norm
            return self._global_grad_norm_per_tag[tag]

        def get_maximize_grad_norm_var_grads(self, factor):
            """
            :param tf.Tensor|float factor:
            :return: dict: var -> grad
            :rtype: dict[tf.Variable,tf.Tensor]
            """
            if self._maximize_grad_norm_var_grads is None:
                loss_ext = self.get_global_grad_norm() * (-factor)
                grads_and_vars_ext = self.updater._compute_gradients(loss_ext, var_list=self.all_vars)
                self._maximize_grad_norm_var_grads = {
                    var: grad for (grad, var) in grads_and_vars_ext if grad is not None
                }
            return self._maximize_grad_norm_var_grads

        def get_maximize_grad_norm_grad(self, factor, var):
            """
            :param float|tf.Tensor factor:
            :param tf.Variable var:
            :rtype: tf.Tensor|None
            """
            return self.get_maximize_grad_norm_var_grads(factor).get(var, None)

        def clip_by_global_norm(self, grad, clip_norm, global_norm_tag=None):
            """
            Wraps tf.clip_by_global_norm.

            :param tf.Tensor grad:
            :param tf.Tensor|float clip_norm:
            :param str|None global_norm_tag:
            :rtype: tf.Tensor
            """
            norm = self.get_global_grad_norm(tag=global_norm_tag)
            (grad,), _ = tf.clip_by_global_norm([grad], clip_norm=clip_norm, use_norm=norm)
            return grad

        def set_zero_on_high_global_norm(self, grad, grad_norm_threshold, global_norm_tag=None):
            """
            :param tf.Tensor grad:
            :param float grad_norm_threshold:
            :param str|None global_norm_tag:
            :rtype: tf.Tensor
            """
            norm = self.get_global_grad_norm(tag=global_norm_tag)
            # Also check nan/inf. Treat them as if we would have been over grad_norm_threshold.
            zero_cond = tf.logical_or(tf_compat.v1.is_nan(norm), tf_compat.v1.is_inf(norm))
            zero_cond = tf.logical_or(zero_cond, tf.greater(norm, grad_norm_threshold))
            return tf.where(zero_cond, tf.zeros_like(grad), grad)

    @classmethod
    def _get_updater_opts_from_var(cls, var):
        """
        :param tf.Variable var:
        :rtype: returnn.util.basic.CollectionReadCheckCovered
        """
        from returnn.util.basic import CollectionReadCheckCovered

        updater_opts = getattr(var, "RETURNN_updater_opts", None)
        if updater_opts is None:
            updater_opts = CollectionReadCheckCovered({})
        assert isinstance(updater_opts, CollectionReadCheckCovered)
        return updater_opts

    def _post_process_grad(self, grad, var, global_info):
        """
        :param tf.Tensor grad:
        :param tf.Variable var:
        :param WrapOptimizer._GetGlobalInfo global_info:
        :return: new grad, apply grad opts
        :rtype: (tf.Tensor, dict[str])
        """
        updater_opts = self._get_updater_opts_from_var(var)

        accum_grad_multiple_num_steps = updater_opts.get(
            "accum_grad_multiple_step", self.config.int("accum_grad_multiple_step", 0)
        )
        grad_noise = updater_opts.get("gradient_noise", self.config.float("gradient_noise", 0.0))
        grad_clip = updater_opts.get("gradient_clip", self.config.float("gradient_clip", 0.0))
        # E.g. https://github.com/openai/baselines/blob/master/baselines/deepq/simple.py:
        #   grad_norm_clipping=10 -> tf.clip_by_norm
        grad_clip_norm = updater_opts.get("gradient_clip_norm", self.config.float("gradient_clip_norm", 0.0))
        grad_clip_avg_norm = updater_opts.get(
            "gradient_clip_avg_norm", self.config.float("gradient_clip_avg_norm", 0.0)
        )
        grad_clip_global_norm = updater_opts.get(
            "gradient_clip_global_norm", self.config.float("gradient_clip_global_norm", 0.0)
        )
        global_norm_tag = updater_opts.get("global_norm_tag", self.config.value("global_norm_tag", None))
        grad_clip_global_norm_tag = updater_opts.get(
            "gradient_clip_global_norm_tag", self.config.value("gradient_clip_global_norm_tag", global_norm_tag)
        )
        grad_norm_to_clip_to_zero = updater_opts.get(
            "grad_norm_to_clip_to_zero", self.config.float("grad_norm_to_clip_to_zero", 0.0)
        )
        maximize_grad_norm = updater_opts.get("maximize_grad_norm", self.config.float("maximize_grad_norm", 0))

        if maximize_grad_norm:
            grad_ext = global_info.get_maximize_grad_norm_grad(maximize_grad_norm, var)
            if grad_ext is not None:
                grad += grad_ext

        if accum_grad_multiple_num_steps is None:
            accum_grad_multiple_num_steps = 0
        if accum_grad_multiple_num_steps >= 1:
            grad = accum_grad_multiple_step(
                grad, var, train_step=self.global_train_step, num_accum_steps=accum_grad_multiple_num_steps
            )

        if updater_opts.get("debug_grad_summaries", self.config.bool_or_other("debug_grad_summaries", False)):
            from returnn.tf.util.basic import variable_summaries, get_base_name, reuse_name_scope_of_tensor

            with reuse_name_scope_of_tensor(grad, prefix="grads/"):
                variable_summaries(grad, name="grad_of_%s" % get_base_name(var))
            with reuse_name_scope_of_tensor(var, prefix="vars/"):
                variable_summaries(var, name=get_base_name(var))

        # Also see tf.contrib.layers.optimizers.optimize_loss() for reference.
        if grad_noise:
            assert grad_noise > 0
            from returnn.tf.util.basic import add_scaled_noise_to_gradients

            with tf.name_scope("grad_noise"):
                ((grad, var),) = add_scaled_noise_to_gradients([(grad, var)], grad_noise)
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
                grad = tf_compat.v1.clip_by_average_norm(grad, grad_clip_avg_norm)
        if grad_clip_global_norm:
            assert grad_clip_global_norm > 0
            with tf.name_scope("grad_clip_global_norm"):
                grad = global_info.clip_by_global_norm(
                    grad, clip_norm=grad_clip_global_norm, global_norm_tag=grad_clip_global_norm_tag
                )
        if updater_opts.get("gradient_nan_inf_filter", self.config.bool("gradient_nan_inf_filter", False)):
            from returnn.tf.util.basic import nan_to_num

            grad = nan_to_num(grad, nan_num=0.0, inf_num=0.0)
        if grad_norm_to_clip_to_zero:
            with tf.name_scope("grad_norm_to_clip_to_zero"):
                grad = global_info.set_zero_on_high_global_norm(
                    grad, grad_norm_threshold=grad_norm_to_clip_to_zero, global_norm_tag=global_norm_tag
                )

        updater_opts.assert_all_read()

        opt_key, _ = self._get_optimizer_item_for_variable(var)
        apply_grad_opts = {"opt_key": opt_key, "accum_grad_multiple_num_steps": accum_grad_multiple_num_steps}
        return grad, apply_grad_opts

    def get_apply_grads_op(self, loss, var_list):
        """
        :param tf.Tensor loss:
        :param list[tf.Variable] var_list:
        :return: op with all variable updates combined, using the optimizer
        :rtype: tf.Operation
        """
        # The following code is basically extended self.optimizer.minimize(), to optionally modify gradients.
        from returnn.util.basic import make_hashable

        if not var_list:
            return tf.no_op(name="no_grad_vars_no_op")

        grads_and_vars = self._compute_gradients(loss, var_list=var_list)
        if self.config.is_true("use_horovod"):
            import returnn.tf.horovod

            if returnn.tf.horovod.get_ctx().is_reduce_type_grad():
                # noinspection PyPackageRequirements,PyUnresolvedReferences
                import horovod.tensorflow as hvd

                grads_and_vars = [
                    (
                        (
                            hvd.allreduce(grad, average=self.config.is_true("horovod_avg_grad"))
                            if grad is not None
                            else None
                        ),
                        var,
                    )
                    for (grad, var) in grads_and_vars
                ]

        var_grads = {var: grad for (grad, var) in grads_and_vars if grad is not None}
        if not var_grads:
            raise Exception("no single variable to train")
        global_info = self._GetGlobalInfo(updater=self, all_vars=var_list, var_grads=var_grads)
        if self.config.bool_or_other("debug_grad_summaries", False):
            tf_compat.v1.summary.scalar("global_grad_norm", global_info.get_global_grad_norm())
        grads_per_apply_grad_opts = {}  # dict apply_grad_opts -> list of (grad, var)
        for grad, var in grads_and_vars:
            assert var in var_list
            if grad is None:
                continue
            new_grad, apply_grad_opts = self._post_process_grad(grad=grad, var=var, global_info=global_info)
            grads_per_apply_grad_opts.setdefault(make_hashable(apply_grad_opts), []).append((new_grad, var))

        if self.decouple_constraints:
            # Note: We want to perform the decoupled constraint updates after all the gradients (and post processing)
            # is calculated (i.e. forward + backprop used the original variable, not any weight decayed version).
            # We want to perform the decoupled constraint updates before we do the gradient update.
            # This is consistent to other frameworks, e.g. PyTorch.
            # https://github.com/rwth-i6/returnn/issues/1007
            # The constraints are given as losses (e.g. L2 norm of the weights) thus we use SGD
            # which is then equivalent to the standard weight decay.
            # Also see the paper: https://arxiv.org/abs/1711.05101, Fixing Weight Decay Regularization in Adam
            with tf_compat.v1.variable_scope("optimize_constraints"):
                with tf_compat.v1.variable_scope("factor"):
                    factor = self.learning_rate / float(self.initial_learning_rate)
                    factor *= self.config.float("decouple_constraints_factor", 0.025)
                for apply_grad_opts, grads_and_vars_per_opts in grads_per_apply_grad_opts.items():
                    for i, (grad, var) in enumerate(grads_and_vars_per_opts):
                        # See LayerBase.get_constraints_value().
                        assert isinstance(var, tf.Variable)
                        l2 = getattr(var, "RETURNN_constraint_L2", None)
                        if not l2:
                            continue
                        with tf.control_dependencies([grad]):
                            # Don't just add the diff to the var because we want to have it decoupled,
                            # which would not be the case with apply_gradients below.
                            def _get_apply_constraints_op():
                                return var.assign_sub(var * (l2 * 2.0), use_locking=self.use_locking, read_value=False)

                            accum_grad_multiple_num_steps = apply_grad_opts.get("accum_grad_multiple_num_steps", 0)
                            if accum_grad_multiple_num_steps > 1:
                                apply_constraint = tf.cond(
                                    tf.equal(
                                        tf_compat.v1.mod(self.global_train_step, accum_grad_multiple_num_steps),
                                        accum_grad_multiple_num_steps - 1,
                                    ),
                                    true_fn=_get_apply_constraints_op,
                                    false_fn=tf.no_op,
                                    name="apply_decoupled_constraints/accum_grad_multiple_step",
                                )
                            else:
                                apply_constraint = _get_apply_constraints_op()
                        with tf.control_dependencies([apply_constraint]):
                            grad = tf.identity(grad)
                        grads_and_vars_per_opts[i] = (grad, var)

        all_apply_grads = []
        assert grads_per_apply_grad_opts
        for apply_grad_opts, grads_and_vars_per_opts in grads_per_apply_grad_opts.items():
            all_apply_grads.append(self._apply_gradients(grads_and_vars_per_opts, **apply_grad_opts))
        if len(all_apply_grads) == 1:
            return all_apply_grads[0]
        return tf.group(*all_apply_grads)


def accum_grad_multiple_step(grad, var, train_step, num_accum_steps):
    """
    :param tf.Tensor|tf.IndexedSlices grad:
    :param tf.Variable var:
    :param tf.Tensor train_step: int, scalar
    :param int num_accum_steps:
    :return: modified grad
    :rtype: tf.Tensor
    """
    from returnn.tf.util.basic import reuse_name_scope_of_tensor, get_base_name

    with reuse_name_scope_of_tensor(grad, postfix="/%s_accum_grad" % get_base_name(grad)):
        shape = var.get_shape().as_list()
        v = tf_compat.v1.get_variable(
            name="var_accum_grad", shape=shape, dtype=grad.dtype, initializer=tf.zeros_initializer(), trainable=False
        )
        return tf.cond(
            tf.less_equal(tf_compat.v1.mod(train_step, num_accum_steps), 0),
            lambda: tf_compat.v1.assign(v, grad),
            lambda: tf_compat.v1.assign_add(v, grad),
        )


# noinspection PyAbstractClass
class _KerasOptimizerWrapper(Optimizer):
    """
    Wraps a TF optimizer into a standard TF optimizer.
    """

    @classmethod
    def get_factory(cls, keras_class):
        """
        :param type[T] keras_class: e.g. tf.keras.optimizers.Nadam
        :return function (kwargs)->Optimizer
        """

        def creator(**kwargs):
            """
            Factory.
            :rtype: T
            """
            kwargs = kwargs.copy()
            kwargs.pop("use_locking", None)  # this is not used. just ignore
            opt = keras_class(**kwargs)
            return cls(opt, name=kwargs.get("name", None))

        return creator

    def __init__(self, optimizer, name=None):
        """
        :param tf.keras.optimizers.Optimizer optimizer:
        :param str|None name:
        """
        if not name:
            # noinspection PyProtectedMember
            name = optimizer._name
        super(_KerasOptimizerWrapper, self).__init__(name=name, use_locking=True)  # always uses locking
        self.keras_optimizer = optimizer
        self._var_list = None

    def _create_slots(self, var_list):
        self._var_list = var_list
        # noinspection PyProtectedMember
        self.keras_optimizer._create_all_weights(var_list)

    def _prepare(self):
        # noinspection PyProtectedMember
        self.keras_optimizer._prepare(self._var_list)

    def _apply_dense(self, grad, var):
        # There should only be resource vars...
        return self._resource_apply_dense(grad, var)

    def _apply_sparse(self, grad, var):
        # There should only be resource vars...
        return self._resource_apply_sparse(grad.values, var, grad.indices)

    def _resource_apply_dense(self, grad, handle):
        # noinspection PyProtectedMember
        return self.keras_optimizer._resource_apply_dense(grad, handle, None)

    def _resource_apply_sparse(self, grad, handle, indices):
        # noinspection PyProtectedMember
        return self.keras_optimizer._resource_apply_sparse(grad, handle, indices, None)


# noinspection PyAbstractClass
class BaseCustomOptimizer(Optimizer):
    """
    Base class for our own optimizer implementations.
    This simplifies the interface to be implemented a bit from :class:`Optimizer`.
    You just have to implement :func:`_apply` here.
    See :class:`CustomGradientDescentOptimizer` or :class:`CustomAdamOptimizer` for as an example.
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
        super(BaseCustomOptimizer, self).__init__(use_locking, name)
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

    def _assign(self, ref, updates, indices=None):
        if indices is not None:
            if isinstance(ref, tf.Variable):
                return tf_compat.v1.scatter_update(ref, indices, updates, use_locking=self._use_locking)
            elif isinstance(ref, resource_variable_ops.ResourceVariable):
                with tf.control_dependencies(
                    [resource_variable_ops.resource_scatter_update(ref.handle, indices, updates)]
                ):
                    return ref.value()
            else:
                raise TypeError("did not expect type %r" % type(ref))
        else:
            return tf_compat.v1.assign(ref, updates, use_locking=self._use_locking)

    def _assign_add(self, ref, updates, indices=None):
        if indices is not None:
            if isinstance(ref, tf.Variable):
                return tf_compat.v1.scatter_add(ref, indices, updates, use_locking=self._use_locking)
            elif isinstance(ref, resource_variable_ops.ResourceVariable):
                with tf.control_dependencies(
                    [resource_variable_ops.resource_scatter_add(ref.handle, indices, updates)]
                ):
                    return ref.value()
            else:
                raise TypeError("did not expect type %r" % type(ref))
        else:
            return tf_compat.v1.assign_add(ref, updates, use_locking=self._use_locking)

    def _assign_sub(self, ref, updates, indices=None):
        if indices is not None:
            if isinstance(ref, tf.Variable):
                return tf_compat.v1.scatter_sub(ref, indices, updates, use_locking=self._use_locking)
            elif isinstance(ref, resource_variable_ops.ResourceVariable):
                with tf.control_dependencies(
                    [resource_variable_ops.resource_scatter_add(ref.handle, indices, -updates)]
                ):
                    return ref.value()
            else:
                raise TypeError("did not expect type %r" % type(ref))
        else:
            return tf_compat.v1.assign_sub(ref, updates, use_locking=self._use_locking)

    # noinspection PyMethodMayBeStatic
    def _gather(self, dense, indices=None):
        """
        This is a simple helper to implement :func:`_apply`.

        :param tf.Tensor dense:
        :param tf.Tensor|None indices: if this is a sparse update, the indices of the grad values
        :rtype: tf.Tensor
        """
        if indices is not None:
            return tf.gather(dense, indices=indices)
        return dense


# noinspection PyAbstractClass
class CustomGradientDescentOptimizer(BaseCustomOptimizer):
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


# noinspection PyAbstractClass
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


# noinspection PyAbstractClass
class NeuralOptimizer1(BaseCustomOptimizer):
    """
    Via Neural Optimizer Search with Reinforcement Learning (https://proceedings.mlr.press/v70/bello17a/bello17a.pdf).

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
        m_t = tf_compat.v1.assign(m, m * beta1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._assign_add(m, updates=m_scaled_g_values, indices=indices)
        # update = lr * grad * where(...)
        m_gathered = self._gather(m_t, indices=indices)
        ones = tf.ones_like(grad)
        update = lr * grad * tf.where(tf.equal(tf.sign(m_gathered), tf.sign(grad)), ones, ones * self._decrease_factor)
        var_update = self._assign_sub(ref=var, updates=update, indices=indices)
        return tf.group(*[var_update, m_t])


# noinspection PyAbstractClass
class GradVarianceScaledOptimizer(BaseCustomOptimizer):
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
        m_t = tf_compat.v1.assign(m, m * beta1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._assign_add(m, updates=m_scaled_g_values, indices=indices)
        m_gathered = self._gather(m_t, indices=indices)

        # Also see tf.compat.v1.nn.moments.
        variance = tf_compat.v1.squared_difference(grad, m_gathered)

        # v_t = beta2 * v + (1 - beta2) * variance
        v_scaled_new_values = variance * (1 - beta2_t)
        v_t = tf_compat.v1.assign(v, v * beta2_t, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._assign_add(v, updates=v_scaled_new_values, indices=indices)
        v_gathered = self._gather(v_t, indices=indices)

        factor = v_gathered / (variance + epsilon_t)
        update = lr * grad * tf.minimum(factor, 1.0)
        var_update = self._assign_sub(ref=var, updates=update, indices=indices)
        return tf.group(*[var_update, m_t])


# noinspection PyAbstractClass
class NadamOptimizer(tf_compat.v1.train.AdamOptimizer):
    """
    Optimizer that implements the Nadam algorithm.
    See [Dozat, T., 2015](http://cs229.stanford.edu/proj2015/054_report.pdf).

    Copied from:
    https://github.com/tensorflow/tensorflow/blob/v1.15.5/tensorflow/contrib/opt/python/training/nadam_optimizer.py

    We have this here to have this Nadam variant available in TF 2
    because the Keras Nadam behaves a bit different.
    https://github.com/rwth-i6/returnn/issues/766
    https://github.com/tensorflow/tensorflow/issues/53204

    We can still use this old code because the underlying kernel still supports the ``use_nesterov`` option.
    """

    def _apply_dense(self, grad, var):
        from tensorflow.python.training import training_ops
        from tensorflow.python.ops import math_ops

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power = self._get_beta_accumulators()
        return training_ops.apply_adam(
            var,
            m,
            v,
            math_ops.cast(beta1_power, var.dtype.base_dtype),
            math_ops.cast(beta2_power, var.dtype.base_dtype),
            math_ops.cast(self._lr_t, var.dtype.base_dtype),
            math_ops.cast(self._beta1_t, var.dtype.base_dtype),
            math_ops.cast(self._beta2_t, var.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, var.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
            use_nesterov=True,
        ).op

    def _resource_apply_dense(self, grad, var):
        from tensorflow.python.training import training_ops
        from tensorflow.python.ops import math_ops

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta1_power, beta2_power = self._get_beta_accumulators()
        return training_ops.resource_apply_adam(
            var.handle,
            m.handle,
            v.handle,
            math_ops.cast(beta1_power, grad.dtype.base_dtype),
            math_ops.cast(beta2_power, grad.dtype.base_dtype),
            math_ops.cast(self._lr_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta1_t, grad.dtype.base_dtype),
            math_ops.cast(self._beta2_t, grad.dtype.base_dtype),
            math_ops.cast(self._epsilon_t, grad.dtype.base_dtype),
            grad,
            use_locking=self._use_locking,
            use_nesterov=True,
        )

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        from tensorflow.python.ops import math_ops
        from tensorflow.python.ops import state_ops
        from tensorflow.python.ops import array_ops
        from tensorflow.python.framework import ops
        from tensorflow.python.ops import control_flow_ops

        beta1_power, beta2_power = self._get_beta_accumulators()
        beta1_power = math_ops.cast(beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
        lr = lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power)
        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)
            # m_bar = (1 - beta1) * g_t + beta1 * m_t
            m_bar = m_scaled_g_values + beta1_t * array_ops.gather(m_t, indices)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)
        v_t_slice = array_ops.gather(v_t, indices)
        v_sqrt = math_ops.sqrt(v_t_slice)
        var_update = scatter_add(var, indices, -lr * m_bar / (v_sqrt + epsilon_t))
        return control_flow_ops.group(*[var_update, m_bar, v_t])


# noinspection PyAbstractClass
class CustomAdamOptimizer(BaseCustomOptimizer):
    """
    Reimplementation of Adam.
    See also :class:`tf.compat.v1.train.AdamOptimizer`.

    ```
    t <- t + 1
    lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)

    m_t <- beta1 * m_{t-1} + (1 - beta1) * g
    v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
    variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
    ```
    """

    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        """
        :param float beta1: used for the running average of g (m)
        :param float beta2: used for the running average of g*g (v)
        :param float epsilon:
        """
        super(CustomAdamOptimizer, self).__init__(**kwargs)
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

    def _prepare(self):
        super(CustomAdamOptimizer, self)._prepare()
        self._beta1_t = tf.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = tf.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = tf.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):
        self._beta1_power = tf.Variable(initial_value=self._beta1, name="beta1_power")
        self._beta2_power = tf.Variable(initial_value=self._beta2, name="beta2_power")
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply(self, grad, var, indices=None):
        """
        :param tf.Tensor grad:
        :param tf.Variable|resource_variable_ops.ResourceVariable var:
        :param tf.Tensor|None indices: if this is a sparse update, the indices of the grad values
        :return: update
        :rtype: tf.Tensor|tf.Operation
        """
        lr = tf.cast(self._learning_rate_tensor, var.dtype.base_dtype)
        beta1_t = tf.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = tf.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = tf.cast(self._epsilon_t, var.dtype.base_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        # lr_t <- learning_rate * sqrt(1 - beta2^t) / (1 - beta1^t)
        lr *= tf.sqrt(1.0 - self._beta2_power) / (1.0 - self._beta1_power)

        # m_t <- beta1 * m_{t-1} + (1 - beta1) * g
        # v_t <- beta2 * v_{t-1} + (1 - beta2) * g * g
        m = self._assign(m, updates=beta1_t * self._gather(m, indices) + (1.0 - beta1_t) * grad, indices=indices)
        v = self._assign(
            v, updates=beta2_t * self._gather(v, indices) + (1.0 - beta2_t) * (grad * grad), indices=indices
        )

        # variable <- variable - lr_t * m_t / (sqrt(v_t) + epsilon)
        update = lr * (self._gather(m, indices) / (tf.sqrt(self._gather(v, indices)) + epsilon_t))
        var_update = self._assign_sub(ref=var, updates=update, indices=indices)
        return tf.group(*[var_update, m, v])

    def _finish(self, update_ops, name_scope):
        with tf.control_dependencies(update_ops), tf_compat.v1.colocate_with(self._beta1_power):
            update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
            update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
        return tf.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


# noinspection PyAbstractClass
class AMSGradOptimizer(Optimizer):
    """
    https://colab.research.google.com/notebook#fileId=1xXFAuHM2Ae-OmF5M8Cn9ypGCa_HHBgfG&scrollTo=N1-2wPHN1Otn
    https://openreview.net/pdf?id=ryQu7f-RZ
    https://keras.io/optimizers/
    https://ruder.io/deep-learning-optimization-2017/index.html#fixingtheexponentialmovingaverage
    https://github.com/taki0112/AMSGrad-Tensorflow
    """

    def __init__(self, learning_rate=0.001, decay=False, beta1=0.9, beta2=0.99, epsilon=0.0, var_list=()):
        super(AMSGradOptimizer, self).__init__(name="AMSGradOptimizer", use_locking=False)
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

    # noinspection PyMethodOverriding
    def apply_gradients(self, gradient_variables):
        """
        :param list[(tf.Tensor,tf.Variable)] gradient_variables:
        :rtype: tf.Operation
        """
        with tf.control_dependencies([self.t.assign_add(1.0)]):
            learning_rate = self.learning_rate
            if self.decay:
                learning_rate /= tf.sqrt(self.t)
            update_ops = []

            for g, var in gradient_variables:
                m = self.m[var].assign(self.beta1 * self.m[var] + (1 - self.beta1) * g)
                v = self.v[var].assign(self.beta2 * self.v[var] + (1 - self.beta2) * g * g)
                v_hat = self.v_hat[var].assign(tf.maximum(self.v_hat[var], v))

                update = -learning_rate * m / (self.epsilon + tf.sqrt(v_hat))
                update_ops.append(var.assign_add(update))

            return tf.group(*update_ops)

    def _apply_dense(self, grad, var):
        raise NotImplementedError

    def _resource_apply_dense(self, grad, handle):
        raise NotImplementedError

    def _resource_apply_sparse(self, grad, handle, indices):
        raise NotImplementedError

    def _apply_sparse(self, grad, var):
        raise NotImplementedError
