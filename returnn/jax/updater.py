"""
Updater for the JAX backend: turns gradients into new parameter values.

JAX optimizers are pure functions over a state pytree (optax), so unlike the PyTorch updater
this holds no reference to the parameters themselves:
the engine passes params and grads in and gets new params back.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Dict, List, Sequence, Tuple, Type

import inspect

import optax

from returnn.config import Config
from returnn.log import log
from returnn.util.basic import get_fwd_compat_kwargs
import returnn.frontend as rf


__all__ = ["Updater"]


class Updater:
    """
    Wraps an optax optimizer, configured the same way as the other backends'
    (the ``optimizer`` config dict, i.e. a ``class`` plus its options).
    """

    def __init__(
        self,
        *,
        config: Optional[Config] = None,
        optimizer_opts: Optional[Dict[str, Any]] = None,
        model: Optional[rf.Module] = None,
        param_names: Optional[Sequence[str]] = None,
    ):
        """
        :param config: RETURNN config, to read the ``optimizer`` option (and the LR schedule) from
        :param optimizer_opts: the optimizer options directly, overriding the config
        :param model: needed for ``weight_decay_modules_blacklist``, which is per module
        :param param_names: names of the parameters which will be passed to :func:`init` / :func:`step`,
            in that order -- the weight-decay mask follows it
        """
        if optimizer_opts is None:
            optimizer_opts = config.typed_value("optimizer") if config else None
        if optimizer_opts is None:
            optimizer_opts = {"class": "adamw"}
        if not isinstance(optimizer_opts, dict):
            raise TypeError(f"JAX Updater: expected an optimizer dict, got {optimizer_opts!r}")
        self.optimizer_opts = dict(optimizer_opts)
        # gradient_clip_global_norm is a top-level config option in the other engines;
        # accept it in the optimizer dict as well, as this engine did before.
        self.gradient_clip_global_norm = self.optimizer_opts.pop("gradient_clip_global_norm", None)
        if config and not self.gradient_clip_global_norm:
            self.gradient_clip_global_norm = config.float("gradient_clip_global_norm", 0.0) or None
        if config:
            _check_unsupported_grad_opts(config)
        # log_grad_norm: True means the 2-norm, or give p directly, as the PyTorch updater reads it
        self.log_grad_norm_p: Optional[float] = _parse_log_grad_norm(config) if config else None
        self.learning_rate_function = config.typed_value("dynamic_learning_rate", None) if config else None
        if self.learning_rate_function is not None:
            if not callable(self.learning_rate_function):
                raise NotImplementedError(
                    f"JAX updater: dynamic_learning_rate {self.learning_rate_function!r} is not callable"
                )
            signature = inspect.signature(self.learning_rate_function)
            assert any(arg.kind == inspect.Parameter.VAR_KEYWORD for arg in signature.parameters.values()), (
                "please specify **kwargs in dynamic_learning_rate for future compatibility"
            )
            if "network" in signature.parameters:
                raise ValueError("JAX updater: dynamic_learning_rate network is TF specific")
            print("Using dynamic learning rate scheduler that updates based on global train steps", file=log.v2)
        wd_mask = _weight_decay_mask(self.optimizer_opts, model=model, param_names=param_names)
        self._optimizer = _make_optimizer(
            self.optimizer_opts, clip_global_norm=self.gradient_clip_global_norm, wd_mask=wd_mask
        )
        print(f"JAX updater: {self.optimizer_opts}", file=log.v3)

    def get_effective_learning_rate(
        self, *, learning_rate: float, global_train_step: int, epoch: int, epoch_continuous: Optional[float] = None
    ) -> float:
        """
        The learning rate of one step, which is the epoch-level one from the learning-rate control
        unless the config defines a ``dynamic_learning_rate`` function on top of it (as the schedules do).

        :param learning_rate: the epoch-level learning rate
        :param global_train_step: over the whole training, starting at 0
        :param epoch: starting at 1
        :param epoch_continuous: fraction of the training done, e.g. 1.5 in the middle of epoch 2,
            None when the dataset cannot say
        :return: the learning rate to use for this step
        """
        if self.learning_rate_function is None:
            return learning_rate
        return float(
            self.learning_rate_function(
                global_train_step=global_train_step,
                epoch=epoch,
                epoch_continuous=epoch_continuous,
                learning_rate=learning_rate,
                **get_fwd_compat_kwargs(),
            )
        )

    def init(self, params: Sequence[Any]) -> Any:
        """
        :param params: the trainable parameters, as raw arrays
        :return: initial optimizer state
        """
        return self._optimizer.init(list(params))

    def step(
        self, *, params: Sequence[Any], grads: Sequence[Any], opt_state: Any, learning_rate: float
    ) -> Tuple[Sequence[Any], Any]:
        """
        One update.

        The learning rate is applied here rather than baked into the optimizer,
        so that RETURNN's learning-rate control stays in charge of it
        (optax schedules would duplicate that).

        :param params: current parameter values
        :param grads: gradients, same structure as params
        :param opt_state: state from :func:`init` or a previous step
        :param learning_rate:
        :return: (new params, new optimizer state)
        """
        updates, opt_state = self._optimizer.update(list(grads), opt_state, list(params))
        updates = [u * learning_rate for u in updates]
        return optax.apply_updates(list(params), updates), opt_state


def global_grad_norm(grads: Sequence[Any], *, p: float) -> Any:
    """
    :param grads: the gradients, as raw arrays
    :param p: the norm order
    :return: the global p-norm over all gradients, as one scalar

    PRE-clip, as the PyTorch engine reports ``grad_norm:pN``:
    the raw gradients, before noise or clipping touch them.
    """
    import jax.numpy as jnp

    if p == 2:  # the common case, and the numerically stable one
        return jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads))
    return sum(jnp.sum(jnp.abs(g) ** p) for g in grads) ** (1.0 / p)


def _parse_log_grad_norm(config: Config) -> Optional[float]:
    """
    :param config: ``log_grad_norm``: True (meaning p=2), or the order p directly
    :return: the norm order, or None when off
    """
    value = config.opt_typed_value("log_grad_norm", False)
    if isinstance(value, str):
        if value.lower() not in ("true", "false", "none"):
            raise ValueError(f"JAX updater: invalid log_grad_norm {value!r}")
        value = {"true": True, "false": False, "none": None}[value.lower()]
    if value is None or value is False:
        return None
    if value is True:
        return 2.0
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"JAX updater: log_grad_norm {value} must be > 0")
        return float(value)
    raise TypeError(f"JAX updater: invalid log_grad_norm {value!r} of type {type(value)}")


def _make_optimizer(
    opts: Dict[str, Any], *, clip_global_norm: Optional[float], wd_mask: Optional[List[bool]] = None
) -> optax.GradientTransformation:
    """
    :param opts: the ``optimizer`` config dict, with a ``class`` entry
    :param clip_global_norm: if given, clip the global gradient norm before the update
    :param wd_mask: per parameter, whether weight decay applies to it (see :func:`_weight_decay_mask`)
    :return: the optax optimizer

    The learning rate is 1 here: the engine scales the updates by the current learning rate,
    so RETURNN's learning-rate control remains the single place that defines it.
    """
    opts = dict(opts)
    cls = opts.pop("class", "adamw").lower()
    # accept RETURNN's spelling of the options
    if "epsilon" in opts:
        opts["eps"] = opts.pop("epsilon")
    if "betas" in opts:
        opts["b1"], opts["b2"] = opts.pop("betas")
    if cls in ("adam", "adamw"):
        func = optax.adamw if cls == "adamw" else optax.adam
    elif cls == "sgd":
        func = optax.sgd
    else:
        raise NotImplementedError(f"JAX Updater: optimizer class {cls!r} not supported")
    if wd_mask is not None:
        opts["mask"] = wd_mask
    optimizer = func(learning_rate=1.0, **opts)
    if clip_global_norm:
        optimizer = optax.chain(optax.clip_by_global_norm(clip_global_norm), optimizer)
    return optimizer


def _weight_decay_mask(
    opts: Dict[str, Any], *, model: Optional[rf.Module], param_names: Optional[Sequence[str]]
) -> Optional[List[bool]]:
    """
    Which parameters weight decay applies to, mirroring the PyTorch updater:
    everything except biases and the parameters of the blacklisted module types.

    :param opts: the optimizer options; ``weight_decay_modules_blacklist`` is popped from it
    :param model:
    :param param_names: the parameters passed to the optimizer, in their order
    :return: one flag per parameter, or None when there is nothing to mask
    """
    blacklist = _wrap_blacklist(opts.pop("weight_decay_modules_blacklist", None))
    if not opts.get("weight_decay"):
        return None
    if model is None or param_names is None:
        return None
    # Note on the default (no blacklist): the PyTorch one is torch.nn.LayerNorm / torch.nn.Embedding,
    # which a pure-RF model never contains, so there only the bias rule below applies. Same here.
    no_wd = set()
    for module_prefix, module in model.named_modules():
        if not isinstance(module, blacklist):
            continue
        for key, value in vars(module).items():
            if isinstance(value, rf.Parameter):
                no_wd.add(f"{module_prefix}.{key}" if module_prefix else key)
    mask = [not (name.split(".")[-1].endswith("bias") or name in no_wd) for name in param_names]
    print(f"JAX updater: weight decay on {sum(mask)} of {len(mask)} parameters", file=log.v3)
    return mask


def _wrap_blacklist(mods: Optional[Sequence[Union[str, Type[rf.Module]]]]) -> Tuple[type, ...]:
    """
    :param mods: module types, as types or as strings (``"rf.Embedding"``)
    :return: the types
    """
    if mods is None:
        return ()
    if not isinstance(mods, (list, tuple)):
        raise TypeError(f"JAX updater: invalid weight_decay_modules_blacklist {mods!r}")
    res = []
    for mod in mods:
        if isinstance(mod, str):
            if not mod.startswith("rf."):
                # "torch.nn.LayerNorm" and the like cannot be checked against an RF model here
                raise NotImplementedError(f"JAX updater: weight_decay_modules_blacklist entry {mod!r} not supported")
            mod = eval(mod)  # noqa: S307  # same as the PyTorch updater does
        if not issubclass(mod, rf.Module):
            raise TypeError(f"JAX updater: invalid weight_decay_modules_blacklist entry {mod!r}")
        res.append(mod)
    return tuple(res)


# Gradient options the other updaters implement and this one does not.
# Silently not clipping (or not adding the configured noise) would change what a config trains.
_UnsupportedGradOpts = [
    "gradient_clip",
    "gradient_clip_norm",
    "gradient_clip_avg_norm",
    "gradient_noise",
    "global_norm_tag",
    "gradient_clip_global_norm_tag",
    "grad_norm_to_clip_to_zero",
    "maximize_grad_norm",
    "gradient_nan_inf_filter",
    "num_allowed_consec_invalid_gradient_steps",
]


def _check_unsupported_grad_opts(config: Config):
    """
    :param config:
    :raise NotImplementedError: on a gradient option this updater does not implement
    """
    unsupported = []
    for key in _UnsupportedGradOpts:
        value = config.typed_value(key, None)
        if value is None and config.has(key):
            value = config.value(key, None)
        if value:
            unsupported.append(key)
    if unsupported:
        raise NotImplementedError(f"JAX updater: options not supported currently: {', '.join(unsupported)}")
    if config.float("grad_clip", 0.0):
        raise ValueError("You set grad_clip in the config, but the option is called gradient_clip_global_norm.")
