"""
This module covers the optimizer (SGD, Adam, etc) logic,
and model param update logic in general.
"""

from __future__ import annotations

from typing import Optional, Union, Any, Type, Callable, Sequence, Iterable, Set, Dict, List, Tuple
import os
import gc
import torch

import returnn
from returnn.log import log
from returnn.util.basic import RefIdEq, get_fwd_compat_kwargs
import returnn.frontend as rf
from returnn.torch.frontend.bridge import wrapped_pt_module_to_rf_module

_OptimizerClassesDictInitialized = False
_OptimizerClassesDict = {}

# Custom optimizers shipped with RETURNN, resolvable by short name like the torch.optim ones.
# torch.optim names take precedence (see :func:`get_optimizer_class`).
_ReturnnOptimizerClassPathsByName = {
    "lion": "returnn.torch.optim.lion.Lion",
    "amuse": "returnn.torch.optim.amuse.AMUSE",
    "multi": "returnn.torch.optim.multi.MultiOptimizer",
}


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
    class_name: Union[str, Type[torch.optim.Optimizer], Callable[[], Type[torch.optim.Optimizer]]],
) -> Type[torch.optim.Optimizer]:
    """
    :param class_name: Optimizer class, either as str (e.g. "adam"), as type (torch.optim.Adam) or callable.
        If str, we support all torch.optim optimizers (ignoring case) (e.g. "adam"),
        the custom optimizers shipped with RETURNN (e.g. "multi", "lion", "amuse"),
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
        import importlib

        if "." in class_name:
            mod_name, class_name_ = class_name.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, class_name_)

        if class_name.lower() in _OptimizerClassesDict:
            return _OptimizerClassesDict[class_name.lower()]
        if class_name.lower() in _ReturnnOptimizerClassPathsByName:
            mod_name, class_name_ = _ReturnnOptimizerClassPathsByName[class_name.lower()].rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, class_name_)

        raise ValueError(
            "Optimizer %r not found in the available optimizers list: %s."
            % (
                class_name.lower(),
                ", ".join(
                    "'%s'" % key for key in list(_OptimizerClassesDict) + list(_ReturnnOptimizerClassPathsByName)
                ),
            )
        )
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

    _OptimizerParamGroupsExtraOpts = ("learning_rate_multiplier",)

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
        self._num_consec_invalid_gradients_steps = 0

        self.learning_rate_function = self.config.typed_value("dynamic_learning_rate", None)
        if self.learning_rate_function is not None:
            print("Using dynamic learning rate scheduler that updates based on global train steps", file=log.v2)
            if callable(self.learning_rate_function):
                import inspect

                signature = inspect.signature(self.learning_rate_function)
                assert any([arg.kind == inspect.Parameter.VAR_KEYWORD for arg in signature.parameters.values()]), (
                    "please specify **kwargs in dynamic_learning_rate for future compatibility"
                )
                if "network" in signature.parameters:
                    raise ValueError("Torch updater: dynamic_learning_rate network is TF specific")
            else:
                raise NotImplementedError("not implemented for not callable dynamic_learning_rate")

        self._optimizer_opts: Optional[Dict[str, Any]] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self._optimizer_param_groups_extra_opts: Optional[List[Dict[str, Any]]] = None

        self._grad_clip = self.config.float("gradient_clip", 0.0)
        self._grad_clip_global_norm = self.config.float("gradient_clip_global_norm", 0.0)
        self._num_allowed_consec_invalid_gradient_steps = self.config.typed_value(
            "num_allowed_consec_invalid_gradient_steps", None
        )
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
            if self._optimizer_param_groups_extra_opts:
                assert len(self.optimizer.param_groups) == len(self._optimizer_param_groups_extra_opts)
                lr_multiplies = [
                    opts.get("learning_rate_multiplier", 1.0) for opts in self._optimizer_param_groups_extra_opts
                ]
            else:
                lr_multiplies = [1.0] * len(self.optimizer.param_groups)
            for i, param_group in enumerate(self.optimizer.param_groups):
                lr = self._effective_learning_rate * lr_multiplies[i]
                if isinstance(param_group["lr"], torch.Tensor):  # see :func:`use_device_lr_tensors`
                    param_group["lr"].fill_(lr)
                else:
                    param_group["lr"] = lr

    def use_device_lr_tensors(self, device: Union[str, torch.device]):
        """
        Convert each param group's lr to a scalar device tensor, from then on updated IN PLACE
        by the LR schedule (:func:`set_learning_rate` / :func:`set_current_train_step`).
        For the captured optimizer step (``torch_cuda_graph`` "capture_optimizer"):
        the capture records the tensor's storage as a graph input,
        so the per-step host-side updates are visible to every replay.
        Requires the capturable optimizer (which accepts tensor lr).
        """
        for param_group in self.optimizer.param_groups:
            lr = param_group["lr"]
            if not isinstance(lr, torch.Tensor):
                param_group["lr"] = torch.tensor(float(lr), dtype=torch.float32, device=device)

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
        if self._current_epoch != epoch:
            self._num_consec_invalid_gradients_steps = 0
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
            norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self._grad_clip_global_norm)
        else:
            norm = None

        has_invalid_gradient = False
        if self._num_allowed_consec_invalid_gradient_steps is not None:
            if norm is None:
                norm = torch.nn.utils.get_total_norm(self.network.parameters())
            has_invalid_gradient = torch.isnan(norm) or torch.isinf(norm)
            if has_invalid_gradient:
                self._num_consec_invalid_gradients_steps += 1
                if self._num_consec_invalid_gradients_steps > self._num_allowed_consec_invalid_gradient_steps:
                    raise RuntimeError(
                        f"Got {self._num_consec_invalid_gradients_steps} invalid gradients in succession, "
                        f"abort training"
                    )
                else:
                    invalid_grads_left = (
                        self._num_allowed_consec_invalid_gradient_steps - self._num_consec_invalid_gradients_steps
                    )
                    print(
                        f"Invalid gradient in step {self._current_train_step}, skipping. "
                        f"{invalid_grads_left} subsequent broken steps left until training is aborted.",
                        file=log.v2,
                    )
            else:
                self._num_consec_invalid_gradients_steps = 0

        if grad_scaler is not None:
            if not has_invalid_gradient:
                grad_scaler.step(self.optimizer)
            # update needs to be called even if we discard the update due to an invalid gradient
            grad_scaler.update()
        elif not has_invalid_gradient:
            self.optimizer.step()

    def create_optimizer(self):
        """
        Creates an optimizer and stores it in self.optimizer.
        """
        optimizer_opts = self.config.typed_value("optimizer", None)
        if optimizer_opts is None:
            raise ValueError("config field 'optimizer' needs to be set explicitely for the Torch backend")
        self._optimizer_opts = optimizer_opts
        self.optimizer, self._optimizer_param_groups_extra_opts = self._create_optimizer(optimizer_opts)

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
        # optimizer_opts is saved as metadata only (load_optimizer ignores it)
        # Drop callables like param_groups_custom or params_filter (also nested, e.g. in the "optimizers" list
        # of the multi optimizer) so torch.load (weights_only=True since torch 2.6) can read it.
        optimizer_opts_to_save = self._optimizer_opts
        if isinstance(optimizer_opts_to_save, dict):
            optimizer_opts_to_save = _drop_callables_deep(optimizer_opts_to_save)

        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "optimizer_class_name": self.optimizer.__class__.__name__,
                "optimizer_opts": optimizer_opts_to_save,
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

    def set_optimizer_training_mode(self, *, train: bool):
        """
        For optimizers following the schedule-free convention with ``train()``/``eval()`` methods
        (e.g. :class:`returnn.torch.optim.amuse.AMUSE`,
        or :class:`returnn.torch.optim.multi.MultiOptimizer` wrapping such),
        switch between train mode (params hold the training iterate)
        and eval mode (params hold the averaged weights, used for evaluation and checkpoints).
        No-op for optimizers without these methods.

        The engine switches to train mode at the start of each train epoch,
        and back to eval mode at the train epoch end,
        before the checkpoint is saved and before any evaluation runs,
        so saved checkpoints always hold the averaged weights.
        The ``epoch_start``/``epoch_end`` config callbacks run before the respective switch,
        so ``epoch_start`` sees the averaged weights and ``epoch_end`` sees the training weights
        (matching the earlier AMUSE config-callback wiring).
        Standalone evaluation outside training (e.g. task "eval") needs no switch,
        as the checkpoints already hold the averaged weights.

        :param train: whether to switch to train mode (True) or eval mode (False)
        """
        if self.optimizer is None:
            return
        func = getattr(self.optimizer, "train" if train else "eval", None)
        if callable(func):
            func()

    def _create_optimizer(self, optimizer_opts) -> Tuple[torch.optim.Optimizer, Optional[List[Dict[str, Any]]]]:
        """
        Returns a valid optimizer considering the dictionary given by the user in the config.

        :param dict[str]|str optimizer_opts: Optimizer configuration specified by the user.
            If it's a dict, it must contain "class" with the optimizer name or callable.
            If it's a str, it must be the optimizer name.
        :return: tuple (optimizer, optional optimizer_param_groups_extra_opts).
        """
        # If the parameter is already a valid optimizer, return it without further processing
        if isinstance(optimizer_opts, torch.optim.Optimizer):
            return optimizer_opts, None
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

        from returnn.torch.optim.multi import MultiOptimizer

        if issubclass(optim_class, MultiOptimizer):
            return self._create_multi_optimizer(optim_class, optimizer_opts)

        # Resolve the optimizer arguments
        opt_kwargs = optimizer_opts.copy()
        optim_class_init_kwargs = _get_class_init_kwargs(optim_class)
        # epsilon is named eps in torch.
        # If the user specified it as epsilon, parse it as eps for the optimizer
        if "eps" in optim_class_init_kwargs and "epsilon" in opt_kwargs:
            opt_kwargs["eps"] = opt_kwargs.pop("epsilon")
        if "learning_rate" in opt_kwargs or "lr" in opt_kwargs:
            raise ValueError("'learning_rate' should be set outside of the 'optimizer' dict.")
        # lr will anyway be updated in set_current_train_step / _update_effective_learning_rate,
        # so this value doesn't really matter here
        opt_kwargs["lr"] = self.learning_rate

        param_groups = self._get_optimizer_param_groups(optim_class, opt_kwargs)
        param_groups = list(param_groups)
        assert len(param_groups) > 0, "got an empty parameter list?"
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        optimizer_param_groups_extra_opts: Optional[List[Dict[str, Any]]] = None
        if any(any(key in group for key in self._OptimizerParamGroupsExtraOpts) for group in param_groups):
            param_groups = [dict(group) for group in param_groups]  # copy to make sure we can modify it
            optimizer_param_groups_extra_opts = [
                {key: group.pop(key) for key in self._OptimizerParamGroupsExtraOpts if key in group}
                for group in param_groups
            ]
        optimizer = optim_class(param_groups, **opt_kwargs)
        print("Optimizer: %s" % optimizer, file=log.v1)
        assert isinstance(optimizer, torch.optim.Optimizer)

        return optimizer, optimizer_param_groups_extra_opts

    def _create_default_optimizer(self):
        """
        :return: SGD optimizer.
        :rtype: torch.optim.SGD
        """
        print("Create SGD optimizer (default).", file=log.v2)
        optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate)

        return optimizer

    def _create_multi_optimizer(
        self, optim_class: Type[torch.optim.Optimizer], optimizer_opts
    ) -> Tuple[torch.optim.Optimizer, Optional[List[Dict[str, Any]]]]:
        """
        Create a :class:`returnn.torch.optim.multi.MultiOptimizer`
        composing multiple sub-optimizers over disjoint parameter subsets.
        See the module docstring of :mod:`returnn.torch.optim.multi` for the config interface.

        :param optim_class: the resolved optimizer class, :class:`MultiOptimizer` or a subclass of it.
            Subclasses must keep the keyword-only ``sub_optimizers`` constructor argument.
        :param dict[str] optimizer_opts: the optimizer options dict, "class" already popped.
        :return: tuple (optimizer, optional optimizer_param_groups_extra_opts), like :func:`_create_optimizer`.
        """
        from returnn.torch.optim.multi import MultiOptimizer

        sub_specs = optimizer_opts.pop("optimizers", None)
        if not isinstance(sub_specs, (list, tuple)) or not sub_specs:
            raise ValueError("optimizer 'multi': 'optimizers' must be a non-empty list of sub-optimizer dicts")
        if optimizer_opts:
            raise ValueError(f"optimizer 'multi': unexpected options {sorted(optimizer_opts.keys())}")
        sub_specs = [dict(spec) for spec in sub_specs]
        for i, spec in enumerate(sub_specs):
            if "class" not in spec:
                raise ValueError(f"optimizer 'multi': sub-optimizer {i} has no 'class'")
            if "params_filter" not in spec and i != len(sub_specs) - 1:
                raise ValueError(
                    f"optimizer 'multi': sub-optimizer {i} has no 'params_filter'."
                    " Only the last sub-optimizer may omit it and then acts as the catch-all."
                )
            if "params_filter" in spec and not callable(spec["params_filter"]):
                raise ValueError(
                    f"optimizer 'multi': sub-optimizer {i}: invalid params_filter {spec['params_filter']!r}"
                )

        # Assign each param to the first sub-optimizer whose filter accepts it.
        named_params = self._named_params_with_modules()
        assigned_named_params = [[] for _ in sub_specs]
        leftover_param_names = []
        for entry in named_params:
            full_param_name, param, module, rf_module = entry
            for i, spec in enumerate(sub_specs):
                params_filter = spec.get("params_filter")
                if params_filter is None or params_filter(
                    full_param_name=full_param_name,
                    param=param,
                    module=module,
                    rf_module=rf_module,
                    **get_fwd_compat_kwargs(),
                ):
                    assigned_named_params[i].append(entry)
                    break
            else:
                leftover_param_names.append(full_param_name)
        if leftover_param_names:
            raise ValueError(
                "optimizer 'multi': params matched by no sub-optimizer params_filter"
                " (add a catch-all sub-optimizer without params_filter, or extend the filters): %s"
                % ", ".join(leftover_param_names)
            )
        for i, assigned in enumerate(assigned_named_params):
            if not assigned:
                raise ValueError(f"optimizer 'multi': sub-optimizer {i} ({sub_specs[i]['class']!r}) got no params")
            print(
                "Multi optimizer: sub-optimizer %i (%r): %i params / %i elements"
                % (i, sub_specs[i]["class"], len(assigned), sum(p.numel() for _, p, _, _ in assigned)),
                file=log.v3,
            )

        # Build the param groups per sub-optimizer, like in the single-optimizer case.
        sub_builds = []  # list of (sub_class, sub_kwargs, num_groups)
        all_param_groups = []
        for i, spec in enumerate(sub_specs):
            spec.pop("params_filter", None)
            sub_class = get_optimizer_class(spec.pop("class"))
            assert not issubclass(sub_class, MultiOptimizer), "optimizer 'multi': cannot nest 'multi'"
            lr_multiplier = spec.pop("learning_rate_multiplier", None)
            sub_kwargs = spec
            sub_class_init_kwargs = _get_class_init_kwargs(sub_class)
            if "eps" in sub_class_init_kwargs and "epsilon" in sub_kwargs:
                sub_kwargs["eps"] = sub_kwargs.pop("epsilon")
            if "learning_rate" in sub_kwargs or "lr" in sub_kwargs:
                raise ValueError(
                    "optimizer 'multi': 'learning_rate'/'lr' not allowed in sub-optimizer opts."
                    " Use the global learning_rate and per-sub-optimizer 'learning_rate_multiplier'."
                )
            sub_kwargs["lr"] = self.learning_rate
            param_groups = self._get_optimizer_param_groups(
                sub_class, sub_kwargs, named_params=assigned_named_params[i]
            )
            if lr_multiplier is not None:
                for group in param_groups:
                    group["learning_rate_multiplier"] = lr_multiplier
            sub_builds.append((sub_class, sub_kwargs, len(param_groups)))
            all_param_groups += param_groups

        # Extract the extra opts (learning_rate_multiplier) over the concatenated groups,
        # in the same order as the MultiOptimizer exposes them.
        optimizer_param_groups_extra_opts: Optional[List[Dict[str, Any]]] = None
        if any(any(key in group for key in self._OptimizerParamGroupsExtraOpts) for group in all_param_groups):
            optimizer_param_groups_extra_opts = [
                {key: group.pop(key) for key in self._OptimizerParamGroupsExtraOpts if key in group}
                for group in all_param_groups
            ]

        sub_optimizers = []
        group_idx = 0
        for sub_class, sub_kwargs, num_groups in sub_builds:
            sub_param_groups = all_param_groups[group_idx : group_idx + num_groups]
            group_idx += num_groups
            sub_optimizers.append(sub_class(sub_param_groups, **sub_kwargs))
        optimizer = optim_class(sub_optimizers=sub_optimizers)
        print("Optimizer: %s" % optimizer, file=log.v1)

        return optimizer, optimizer_param_groups_extra_opts

    def _named_params_with_modules(self) -> List[Tuple[str, torch.nn.Parameter, torch.nn.Module, Optional[rf.Module]]]:
        """
        :return: list of (full_param_name, param, owning module, owning RF module or None),
            each param exactly once (shared params are listed for their first owning module).
        """
        entries = []
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
                entries.append((full_param_name, param, module, rf_module))
        return entries

    def _get_optimizer_param_groups(
        self,
        optim_class: Type[torch.optim.Optimizer],
        optimizer_opts: Dict[str, Any],
        named_params: Optional[List[Tuple[str, torch.nn.Parameter, torch.nn.Module, Optional[rf.Module]]]] = None,
    ) -> Union[Iterable[Dict[str, Any]], Iterable[torch.nn.Parameter]]:
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
        :param named_params: if given (the multi optimizer sub-optimizer case), build the groups only over
            this parameter subset (entries as returned by :func:`_named_params_with_modules`),
            and return a list of param group dicts, dropping empty groups.
            ``param_groups_custom`` is not supported in this case.
        :return: List of configurations for the different sets of parameters.
        """
        subset_mode = named_params is not None

        custom_param_groups = optimizer_opts.pop("param_groups_custom", None)
        if custom_param_groups is not None:
            if subset_mode:
                raise ValueError(
                    "param_groups_custom is not supported in multi optimizer sub-optimizer opts."
                    " Use params_filter to assign the params to the sub-optimizers,"
                    " and weight_decay_custom_include_check / weight_decay_modules_blacklist"
                    " for the weight-decay split within a sub-optimizer."
                )
            assert callable(custom_param_groups), f"invalid param_groups_custom {custom_param_groups!r}"
            rf_model = wrapped_pt_module_to_rf_module(self.network)
            custom_param_groups_ = custom_param_groups(
                model=self.network,
                rf_model=rf_model,
                optimizer_class=optim_class,
                optimizer_opts=optimizer_opts,
                **get_fwd_compat_kwargs(),
            )
            assert isinstance(custom_param_groups_, Iterable) and all(
                isinstance(group, dict) for group in custom_param_groups_
            ), f"invalid param_groups_custom {custom_param_groups!r} result {custom_param_groups_!r} type"
            return custom_param_groups_

        # By default, insert the weight_decay constraints in the optimizer, as this is default PyTorch behavior.
        # If the user doesn't accept this, throw an error message.
        assert self.config.bool("decouple_constraints", True), (
            "L2/weight_decay constraints are decoupled in PyTorch, but "
            "decouple_constraints=False was explicitly specified in the config."
        )

        # Split in parameter groups only if decouple_constraints is set and the optimizer accepts weight_decay.
        cls_init_kwargs = _get_class_init_kwargs(optim_class)
        if "weight_decay" not in cls_init_kwargs:
            assert "weight_decay" not in optimizer_opts, (
                "weight_decay not accepted by the chosen optimizer. Accepted values: %s"
                % ", ".join("%s" % optim_name for optim_name in cls_init_kwargs)
            )
            if subset_mode:
                return [{"params": [param for _, param, _, _ in named_params]}]
            return self.network.parameters()

        blacklist_wd_modules = wrap_user_blacklist_wd_modules(
            optimizer_opts.pop("weight_decay_modules_blacklist", None)
        )
        custom_include_check = optimizer_opts.pop("weight_decay_custom_include_check", None)
        if custom_include_check:
            assert callable(custom_include_check), f"invalid weight_decay_custom_include_check {custom_include_check!r}"

        weight_decay = optimizer_opts.get("weight_decay", 0.0)
        if not weight_decay:
            if subset_mode:
                return [{"params": [param for _, param, _, _ in named_params]}]
            return self.network.parameters()

        if named_params is None:
            named_params = self._named_params_with_modules()

        # Distinguish between parameters with and without weight_decay/L2 regularization.
        # Parameters without weight decay: biases + LayerNorm/Embedding layers.
        wd_named = []
        no_wd_named = []
        for full_param_name, param, module, rf_module in named_params:
            custom_include = None
            if custom_include_check:
                # For backward compatibility, full_param_name carries the module-local param name
                # (e.g. just "weight" or "bias"), not the full hierarchical name,
                # as existing callbacks rely on that (despite the misleading argument name).
                custom_include = custom_include_check(
                    module=module, rf_module=rf_module, full_param_name=full_param_name.rsplit(".", 1)[-1], param=param
                )
            if custom_include is not None:
                assert isinstance(custom_include, bool), "weight_decay_custom_include_check did not return bool"
                include_wd = custom_include
            elif (
                full_param_name.endswith("bias")
                or isinstance(module, blacklist_wd_modules)
                or isinstance(rf_module, blacklist_wd_modules)
            ):
                include_wd = False
            else:
                include_wd = True
            (wd_named if include_wd else no_wd_named).append((full_param_name, param))

        wd_named.sort(key=lambda entry: entry[0])
        no_wd_named.sort(key=lambda entry: entry[0])
        optim_groups = [
            {"params": [param for _, param in wd_named], "weight_decay": weight_decay},
            {"params": [param for _, param in no_wd_named], "weight_decay": 0.0},
        ]
        if subset_mode:
            # A sub-optimizer only gets the groups it has params for.
            # The full-network case keeps both groups (even if empty)
            # for compatibility with existing optimizer checkpoints (the group count must match).
            optim_groups = [group for group in optim_groups if group["params"]]
        return optim_groups


def wrap_user_blacklist_wd_modules(
    mods: Optional[Sequence[Union[str, Type[rf.Module], Type[torch.nn.Module]]]],
) -> Tuple[type, ...]:
    """
    Wraps the user-provided blacklist_weight_decay_modules into a tuple of types.
    This supports both pure PyTorch modules (e.g. "torch.nn.LayerNorm")
    and RF modules (e.g. "rf.LayerNorm"), which can be specified as strings or types.
    """
    if mods is None:
        return torch.nn.LayerNorm, torch.nn.Embedding
    assert isinstance(mods, (list, tuple)), f"invalid blacklist_weight_decay_modules {mods!r}"
    res = []
    for mod in mods:
        if isinstance(mod, str):
            assert mod.startswith("torch.") or mod.startswith("rf."), f"invalid blacklist_weight_decay_modules {mods!r}"
            mod = eval(mod)
        assert issubclass(mod, (rf.Module, torch.nn.Module)), f"invalid blacklist_weight_decay_modules {mods!r}"
        res.append(mod)
    return tuple(res)


def _drop_callables_deep(obj: Any) -> Any:
    """
    :param obj: nested structure of dicts/lists/tuples
    :return: copy with callable dict values and callable list/tuple entries dropped
    """
    if isinstance(obj, dict):
        return {k: _drop_callables_deep(v) for k, v in obj.items() if not callable(v)}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_drop_callables_deep(v) for v in obj if not callable(v))
    return obj


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
