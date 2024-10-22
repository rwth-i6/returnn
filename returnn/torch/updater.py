"""
This module covers the optimizer (SGD, Adam, etc) logic,
and model param update logic in general.
"""

from __future__ import annotations

from typing import Optional, Union, Any, Type, Callable, Sequence, Iterable, Set, Dict, List, Tuple
import os
import gc
import torch
import typing

import returnn
from returnn.log import log
from returnn.util.basic import RefIdEq, get_fwd_compat_kwargs
import returnn.frontend as rf
from returnn.torch.frontend.bridge import wrapped_pt_module_to_rf_module

_OptimizerClassesDictInitialized = False
_OptimizerClassesDict = {}


def _init_optimizer_classes_dict():
    """
    Initializes a global dictionary with all optimizers available in PyTorch.
    """
    global _OptimizerClassesDictInitialized
    if _OptimizerClassesDictInitialized:
        return
    _OptimizerClassesDictInitialized = True
    for name, cls in list(vars(torch.optim).items()):
        assert isinstance(name, str)
        # Check if cls is a valid subclass of torch.optim.Optimizer
        if not isinstance(cls, type) or not issubclass(cls, torch.optim.Optimizer):
            continue
        assert name not in _OptimizerClassesDict
        _OptimizerClassesDict[name.lower()] = cls


def get_optimizer_class(
    class_name: Union[str, Type[torch.optim.Optimizer], Callable[[], Type[torch.optim.Optimizer]]]
) -> Type[torch.optim.Optimizer]:
    """
    :param class_name: Optimizer class, either as str (e.g. "adam"), as type (torch.optim.Adam) or callable.
        If str, we support all torch.optim optimizers (ignoring case) (e.g. "adam"),
        or class names with full module path (e.g. "returnn.torch.optim.lion.Lion").
    :return: Optimizer class, e.g. torch.optim.Adam
    """
    _init_optimizer_classes_dict()
    if isinstance(class_name, type):
        assert issubclass(class_name, torch.optim.Optimizer)
        return class_name
    elif callable(class_name):
        return class_name()
    elif isinstance(class_name, str):
        if "." in class_name:
            import importlib

            mod_name, class_name_ = class_name.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, class_name_)

        if class_name.lower() not in _OptimizerClassesDict:
            raise ValueError(
                "Optimizer %r not found in the available torch optimizers list: %s."
                % (
                    class_name.lower(),
                    ", ".join("'%s'" % key for key in _OptimizerClassesDict),
                )
            )
        return _OptimizerClassesDict[class_name.lower()]
    else:
        raise TypeError(f"Invalid optimizer class_name {class_name!r} type {type(class_name).__name__}")


def _get_class_init_kwargs(optim_class):
    """
    Obtains the keyword arguments of the class provided as parameter that the user can add to their optimizer.

    :param type[torch.optim.Optimizer] optim_class: Optimizer class.
    :return: Keyword arguments of the provided class.
    :rtype: List[str]
    """
    from returnn.util.basic import collect_class_init_kwargs

    optim_class_init_kwargs = collect_class_init_kwargs(optim_class)
    # We already provide params by default, remove it so that the user doesn't add it to the optimizer dict.
    optim_class_init_kwargs.remove("params")

    return optim_class_init_kwargs


class Updater:
    """
    Wraps a torch.optim.Optimizer, and extends it by some further functionality.
    """

    def __init__(self, *, config, network, device, initial_learning_rate=1.0):
        """
        :param returnn.config.Config config: config defining the training conditions.
        :param torch.nn.Module network: PyTorch Module defining the network.
        :param torch.device|str device:
        :param float initial_learning_rate:
        """
        self.config = config
        self.learning_rate = float(initial_learning_rate)
        self._effective_learning_rate = self.learning_rate
        self.network = network
        self._device = device
        # Just set the very first step as initial values here.
        # They will be overwritten via set_current_train_step() below.
        self._current_train_step = 0
        self._current_epoch = 1
        self._current_epoch_continuous = 0.0

        self.learning_rate_function = self.config.typed_value("dynamic_learning_rate", None)
        if self.learning_rate_function is not None:
            print("Using dynamic learning rate scheduler that updates based on global train steps", file=log.v2)
            if callable(self.learning_rate_function):
                import inspect

                signature = inspect.signature(self.learning_rate_function)
                assert any(
                    [arg.kind == inspect.Parameter.VAR_KEYWORD for arg in signature.parameters.values()]
                ), "please specify **kwargs in dynamic_learning_rate for future compatibility"
                if "network" in signature.parameters:
                    raise ValueError("Torch updater: dynamic_learning_rate network is TF specific")
            else:
                raise NotImplementedError("not implemented for not callable dynamic_learning_rate")

        self._optimizer_opts = None
        self.optimizer = None  # type: typing.Optional[torch.optim.Optimizer]

        self._grad_clip = self.config.float("gradient_clip", 0.0)
        self._grad_clip_global_norm = self.config.float("gradient_clip_global_norm", 0.0)
        self._grad_noise = self.config.float("gradient_noise", 0.0)

        # Check other options we have in TF updater, which we might support here later as well,
        # but currently do not support.
        for opt_name in [
            "gradient_clip_norm",
            "gradient_clip_avg_norm",
            "global_norm_tag",
            "gradient_clip_global_norm_tag",
            "grad_norm_to_clip_to_zero",
            "maximize_grad_norm",
            "debug_grad_summaries",
            "gradient_nan_inf_filter",
        ]:
            if self.config.float(opt_name, 0.0):
                raise NotImplementedError(f"PyTorch updater: option {opt_name} not supported currently")
        # Check for potential user mistakes.
        if self.config.float("grad_clip", 0.0):
            raise ValueError(
                "You set grad_clip in the config,"
                " but the option is called gradient_clip_global_norm (or other options)."
            )

        self._update_effective_learning_rate()

    def set_learning_rate(self, value):
        """
        Updates the learning rate of the optimizer at each (sub)epoch.

        :param float value: New learning rate.
        """
        self.learning_rate = float(value)
        self._update_effective_learning_rate()

    def get_effective_learning_rate(self) -> float:
        """
        :return: get the actual learning rate
        """
        return self._effective_learning_rate

    def _update_effective_learning_rate(self):
        self._effective_learning_rate = self.learning_rate
        if self.learning_rate_function is not None:
            lr = self.learning_rate_function(
                global_train_step=self._current_train_step,
                epoch=self._current_epoch,
                epoch_continuous=self._current_epoch_continuous,
                learning_rate=self.learning_rate,
                **get_fwd_compat_kwargs(),
            )
            self._effective_learning_rate = float(lr)
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self._effective_learning_rate

    def set_current_train_step(self, *, global_train_step: int, epoch: int, epoch_continuous: Optional[float] = None):
        """
        Obtains an updated learning rate for the current training step inside a (sub)epoch.

        :param global_train_step: Current global training step over the whole training process.
            In the first epoch, this starts at 0.
        :param epoch: Current epoch. (First epoch is 1 by RETURNN convention.)
        :param epoch_continuous: How much of the epoch is finished.
            In the first step of the first epoch, this starts at 0.0,
            and when the fist epoch is finished, this reaches 1.0,
            and the values in between are the fraction of the epoch that is finished.
            The second epoch (epoch=2) starts at 1.0,
            and when the second epoch is finished, this reaches 2.0, and so on.
            We usually calculate this based on ``epoch-1+(last_seq_idx+1)/num_seqs``,
            if the dataset can provide ``num_seqs``.
            Other schemes based on the step_idx might be used as well to calculate this,
            if the number of steps per epoch is known in advance.
        """
        self._current_train_step = global_train_step
        self._current_epoch = epoch
        self._current_epoch_continuous = epoch_continuous
        self._update_effective_learning_rate()

    def step(self, *, grad_scaler: Optional[torch.cuda.amp.GradScaler] = None):
        """
        Perform one step, i.e. update the parameters using the optimizer given the current calculated gradients.
        """
        if grad_scaler is not None:
            grad_scaler.unscale_(self.optimizer)

        if self._grad_noise:
            gradient_noise_(self.network.parameters(), self._grad_noise)
        if self._grad_clip:
            torch.nn.utils.clip_grad_value_(self.network.parameters(), self._grad_clip)
        if self._grad_clip_global_norm:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self._grad_clip_global_norm)

        if grad_scaler is not None:
            grad_scaler.step(self.optimizer)
            grad_scaler.update()
        else:
            self.optimizer.step()

    def create_optimizer(self):
        """
        Creates an optimizer and stores it in self.optimizer.
        """
        optimizer_opts = self.config.typed_value("optimizer", None)
        if optimizer_opts is None:
            raise ValueError("config field 'optimizer' needs to be set explicitely for the Torch backend")
        self._optimizer_opts = optimizer_opts
        self.optimizer = self._create_optimizer(optimizer_opts)

    def load_optimizer(self, filename):
        """
        Loads a torch.optim.Optimizer from disk and stores it in self.optimizer.

        :param str filename: File from which to load the optimizer state.
        """
        print("Load optimizer %s" % filename, file=log.v4)
        optimizer_state = torch.load(filename, map_location=self._device)
        assert isinstance(optimizer_state, dict), f"optimizer_state is not a dict but {type(optimizer_state)}"
        if "optimizer" not in optimizer_state and "param_groups" in optimizer_state and "state" in optimizer_state:
            # Old format, convert to new format.
            optimizer_state = {"optimizer": optimizer_state}
        if optimizer_state.get("param_names") is not None:
            if len(self.optimizer.param_groups) != len(optimizer_state["optimizer"]["param_groups"]):
                raise ValueError(
                    "loaded state dict has a different number of parameter groups: ckpt %i vs. self %i"
                    % (len(optimizer_state["optimizer"]["param_groups"]), len(self.optimizer.param_groups))
                )
            # Check if we have the same parameters in the same order.
            self_param_names, param_id_to_name = self._get_opt_param_names()
            ckpt_param_names = optimizer_state["param_names"]
            if self_param_names != ckpt_param_names:
                self_param_names_dict = {name: i for i, name in enumerate(self_param_names)}
                self_param_names_critical_set = set()
                ckpt_param_names_dict = {name: i for i, name in enumerate(ckpt_param_names)}
                map_ckpt_param_idx_to_self_param_idx = {}
                self_params_not_in_ckpt = []
                self_params_not_in_ckpt_critical = []
                for param_name in self_param_names:
                    param = self.network.get_parameter(param_name)
                    if param.requires_grad:
                        self_param_names_critical_set.add(param_name)
                    if param_name not in ckpt_param_names_dict:
                        self_params_not_in_ckpt.append(param_name)
                        if param.requires_grad:
                            self_params_not_in_ckpt_critical.append(param_name)
                ckpt_params_not_in_self = []
                for i, param_name in enumerate(ckpt_param_names):
                    if param_name not in self_param_names_dict:
                        ckpt_params_not_in_self.append(param_name)
                    else:
                        map_ckpt_param_idx_to_self_param_idx[i] = self_param_names_dict[param_name]
                if self_params_not_in_ckpt_critical:
                    raise ValueError(
                        "load_optimizer: required params not in ckpt: %s" % ", ".join(self_params_not_in_ckpt_critical)
                    )
                if self_params_not_in_ckpt or ckpt_params_not_in_self:
                    print(
                        "load_optimizer: params not in ckpt: %s\n    ckpt params not existing: %s"
                        % (
                            ", ".join(self_params_not_in_ckpt) or "(None)",
                            ", ".join(ckpt_params_not_in_self) or "(None)",
                        ),
                        file=log.v3,
                    )
                    if self_params_not_in_ckpt:
                        print(
                            "load_optimizer: All params not in ckpt have required_grad=False, thus not critical.",
                            file=log.v3,
                        )
                else:
                    print("load_optimizer: Params in different order.", file=log.v3)
                print("load_optimizer: Will remap the state dict.", file=log.v3)
                for ckpt_group, self_group in zip(
                    optimizer_state["optimizer"]["param_groups"], self.optimizer.param_groups
                ):
                    # Check whether it is matching for the critical params.
                    self_group_param_names = set(param_id_to_name[id(p)] for p in self_group["params"])
                    ckpt_group_param_names = set(ckpt_param_names[p] for p in ckpt_group["params"])
                    self_group_param_names.intersection_update(self_param_names_critical_set)
                    ckpt_group_param_names.intersection_update(self_param_names_critical_set)
                    if ckpt_group_param_names != self_group_param_names:
                        raise ValueError(
                            "load_optimizer: params in group not in ckpt: %s\n  ckpt params not existing: %s"
                            % (
                                ", ".join(ckpt_group_param_names - self_group_param_names) or "(None)",
                                ", ".join(self_group_param_names - ckpt_group_param_names) or "(None)",
                            )
                        )
                    ckpt_group["params"] = [
                        self_param_names_dict[param_id_to_name[id(p)]] for p in self_group["params"]
                    ]
                optimizer_state["optimizer"]["state"] = {
                    map_ckpt_param_idx_to_self_param_idx[i]: s
                    for (i, s) in optimizer_state["optimizer"]["state"].items()
                    if i in map_ckpt_param_idx_to_self_param_idx
                }
        self.optimizer.load_state_dict(optimizer_state["optimizer"])
        # https://github.com/rwth-i6/returnn/issues/1345
        del optimizer_state
        gc.collect()

    def _get_opt_param_names(self) -> Tuple[List[str], Dict[int, str]]:
        param_id_to_name = {}  # id -> name
        for name, p in self.network.named_parameters():
            param_id_to_name[id(p)] = name
        param_names = []  # param_idx -> name
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                param_names.append(param_id_to_name[id(p)])
        return param_names, param_id_to_name

    def save_optimizer(self, filename):
        """
        Saves the state of self.optimizer to a file.

        :param str filename: File in which to save the optimizer state.
        """
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        # We use optimizer.state_dict() below.
        # That will only save param order indices
        # but not the name of the parameters.
        # We also save a mapping of parameter indices to names.
        param_names, _ = self._get_opt_param_names()

        print("Save optimizer under %s" % filename, file=log.v4)
        # First write to a temp-file, to be sure that writing happens without errors,
        # and only afterward rename to the target file.
        tmp_filename = filename + ".tmp_write"
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "optimizer_class_name": self.optimizer.__class__.__name__,
                "optimizer_opts": self._optimizer_opts,
                "param_names": param_names,
                "epoch": self._current_epoch,
                "step": self._current_train_step,
                "effective_learning_rate": self.get_effective_learning_rate(),
                "returnn_version": returnn.__long_version__,
            },
            tmp_filename,
        )
        os.rename(tmp_filename, filename)

    def get_optimizer(self):
        """
        :return: Wrapped optimizer object.
        :rtype: torch.optim.Optimizer
        """
        return self.optimizer

    def _create_optimizer(self, optimizer_opts):
        """
        Returns a valid optimizer considering the dictionary given by the user in the config.

        :param dict[str]|str optimizer_opts: Optimizer configuration specified by the user.
            If it's a dict, it must contain "class" with the optimizer name or callable.
            If it's a str, it must be the optimizer name.
        :return: A valid optimizer.
        :rtype: torch.optim.Optimizer
        """
        lr = self.learning_rate

        # If the parameter is already a valid optimizer, return it without further processing
        if isinstance(optimizer_opts, torch.optim.Optimizer):
            return optimizer_opts
        elif callable(optimizer_opts):
            optimizer_opts: Dict[str, Any] = {"class": optimizer_opts}
        else:
            if not isinstance(optimizer_opts, dict):
                raise ValueError("'optimizer' must of type dict, callable or torch.optim.Optimizer instance.")
            if "class" not in optimizer_opts:
                raise ValueError("'class' field of 'optimizer' dict was not set (use e.g. 'SGD', 'Adam', ...)")
            optimizer_opts = optimizer_opts.copy()

        # Resolve the optimizer class
        optim_class_name = optimizer_opts.pop("class")
        optim_class = get_optimizer_class(optim_class_name)

        # Resolve the optimizer arguments
        opt_kwargs = optimizer_opts.copy()
        optim_class_init_kwargs = _get_class_init_kwargs(optim_class)
        # epsilon is named eps in torch.
        # If the user specified it as epsilon, parse it as eps for the optimizer
        if "eps" in optim_class_init_kwargs and "epsilon" in opt_kwargs:
            opt_kwargs["eps"] = opt_kwargs.pop("epsilon")
        if "learning_rate" in opt_kwargs or "lr" in opt_kwargs:
            raise ValueError("'learning_rate' should be set outside of the 'optimizer' dict.")
        lr = lr * opt_kwargs.pop("learning_rate_multiplier", 1.0)
        opt_kwargs["lr"] = lr

        param_groups = self._get_optimizer_param_groups(optim_class, opt_kwargs)
        optimizer = optim_class(param_groups, **opt_kwargs)
        print("Optimizer: %s" % optimizer, file=log.v1)
        assert isinstance(optimizer, torch.optim.Optimizer)

        return optimizer

    def _create_default_optimizer(self):
        """
        :return: SGD optimizer.
        :rtype: torch.optim.SGD
        """
        print("Create SGD optimizer (default).", file=log.v2)
        optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)

        return optimizer

    def _get_optimizer_param_groups(
        self, optim_class: Type[torch.optim.Optimizer], optimizer_opts: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        The weight_decay parameter from AdamW affects the weights of layers such as LayerNorm and Embedding.
        This function creates a blacklist of network modules and splits the optimizer groups in two:
        those who will receive weight decay, and those who won't receive it.
        The weight_decay parameter of the rest of the optimizers is L2 regularization.

        For further reading, see https://github.com/karpathy/minGPT/pull/24#issuecomment-679316025 and
        https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994.

        This code is based on https://github.com/karpathy/minGPT (MIT license):
        https://github.com/karpathy/minGPT/blob/3ed14b2cec0dfdad3f4b2831f2b4a86d11aef150/mingpt/model.py#L136.

        Three variants how this can be configured by the user in the optimizer options dict:

        - ``param_groups_custom``: callable which returns a list of param groups.
          This is the most flexible option, and could also go beyond just weight decay logic,
          or having more than two param groups (weight decay disabled/enabled).
        - ``weight_decay_custom_include_check``: callable which returns True/False for each param,
          to either include it in the weight decay group or not,
          or None to use the default logic.
        - ``weight_decay_modules_blacklist``: list of modules types which should not get weight decay.
          Those can be RF modules or pure PyTorch modules.
          The types can be specified as string (e.g. ``"torch.nn.LayerNorm"``) or as the type itself.

        :param optim_class: Optimizer class.
        :param optimizer_opts: Optimizer configuration specified by the user. Might be modified inplace here.
        :return: List of configurations for the different sets of parameters.
        """
        custom_param_groups = optimizer_opts.pop("param_groups_custom", None)
        if custom_param_groups is not None:
            assert callable(custom_param_groups), f"invalid param_groups_custom {custom_param_groups!r}"
            rf_model = wrapped_pt_module_to_rf_module(self.network)
            custom_param_groups = custom_param_groups(
                model=self.network, rf_model=rf_model, optimizer_class=optim_class, optimizer_opts=optimizer_opts
            )
            return custom_param_groups

        network_params = self.network.parameters()

        # By default, insert the weight_decay constraints in the optimizer, as this is default PyTorch behavior.
        # If the user doesn't accept this, throw an error message.
        assert self.config.bool("decouple_constraints", True), (
            "L2/weight_decay constraints are decoupled in PyTorch, but "
            "decouple_constraints=False was explicitly specified in the config."
        )

        # Split in parameter groups only if decouple_constraints is set and the optimizer accepts weight_decay.
        cls_init_kwargs = _get_class_init_kwargs(optim_class)
        if "weight_decay" not in cls_init_kwargs:
            assert (
                "weight_decay" not in optimizer_opts
            ), "weight_decay not accepted by the chosen optimizer. Accepted values: %s" % ", ".join(
                "%s" % optim_name for optim_name in cls_init_kwargs
            )
            return [{"params": network_params}]

        weight_decay = optimizer_opts.get("weight_decay", 0.0)
        if not weight_decay:
            return [{"params": network_params}]

        # Distinguish between parameters with and without weight_decay/L2 regularization.
        # Parameters without weight decay: biases + LayerNorm/Embedding layers.
        wd_params = set()
        no_wd_params = set()
        blacklist_wd_modules = optimizer_opts.pop("weight_decay_modules_blacklist", None)
        if blacklist_wd_modules is None:
            blacklist_wd_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        else:
            blacklist_wd_modules = _wrap_user_blacklist_wd_modules(blacklist_wd_modules)
        custom_include_check = optimizer_opts.pop("weight_decay_custom_include_check", None)
        if custom_include_check:
            assert callable(custom_include_check), f"invalid weight_decay_custom_include_check {custom_include_check!r}"
        # Tracker of visited parameters to only add each parameter once, in case two modules share common parameters.
        # We need the wrapper class RefIdEq because Parameters are compared by value and not by reference.
        visited_params: Set[RefIdEq[torch.nn.Parameter]] = set()
        for module_name, module in self.network.named_modules():
            module_name: str
            module: torch.nn.Module
            rf_module = wrapped_pt_module_to_rf_module(module)
            for param_name, param in module.named_parameters(recurse=False):
                param_name: str
                param: torch.nn.Parameter
                if RefIdEq(param) in visited_params:
                    continue
                visited_params.add(RefIdEq(param))
                full_param_name = "%s.%s" % (module_name, param_name) if module_name else param_name
                custom_include = None
                if custom_include_check:
                    custom_include = custom_include_check(
                        module=module, rf_module=rf_module, full_param_name=param_name, param=param
                    )
                if custom_include is not None:
                    assert isinstance(custom_include, bool), "weight_decay_custom_include_check did not return bool"
                    if custom_include:
                        wd_params.add(full_param_name)
                    else:
                        no_wd_params.add(full_param_name)
                elif (
                    param_name.endswith("bias")
                    or isinstance(module, blacklist_wd_modules)
                    or isinstance(rf_module, blacklist_wd_modules)
                ):
                    no_wd_params.add(full_param_name)
                else:
                    wd_params.add(full_param_name)

        param_dict = {pn: p for pn, p in self.network.named_parameters()}
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(wd_params))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_wd_params))], "weight_decay": 0.0},
        ]

        return optim_groups


def _wrap_user_blacklist_wd_modules(
    mods: Sequence[Union[str, Type[rf.Module], Type[torch.nn.Module]]]
) -> Tuple[type, ...]:
    assert isinstance(mods, (list, tuple)), f"invalid blacklist_weight_decay_modules {mods!r}"
    res = []
    for mod in mods:
        if isinstance(mod, str):
            assert mod.startswith("torch.") or mod.startswith("rf."), f"invalid blacklist_weight_decay_modules {mods!r}"
            mod = eval(mod)
        assert issubclass(mod, (rf.Module, torch.nn.Module)), f"invalid blacklist_weight_decay_modules {mods!r}"
        res.append(mod)
    return tuple(res)


def gradient_noise_(params: Iterable[torch.nn.Parameter], std: float):
    """
    Add gradient noise to parameters, using a truncated normal distribution.
    """
    a, b = -2 * std, 2 * std
    for param in params:
        if param.requires_grad and param.grad is not None:
            noise = torch.empty_like(param.grad)
            torch.nn.init.trunc_normal_(noise, std=std, a=a, b=b)
            param.grad += noise
