"""
Updater for the JAX backend: turns gradients into new parameter values.

JAX optimizers are pure functions over a state pytree (optax), so unlike the PyTorch updater
this holds no reference to the parameters themselves:
the engine passes params and grads in and gets new params back.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Sequence, Tuple

import optax

from returnn.config import Config
from returnn.log import log


__all__ = ["Updater"]


class Updater:
    """
    Wraps an optax optimizer, configured the same way as the other backends'
    (the ``optimizer`` config dict, i.e. a ``class`` plus its options).
    """

    def __init__(self, *, config: Optional[Config] = None, optimizer_opts: Optional[Dict[str, Any]] = None):
        """
        :param config: RETURNN config, to read the ``optimizer`` option from
        :param optimizer_opts: the optimizer options directly, overriding the config
        """
        if optimizer_opts is None:
            optimizer_opts = config.typed_value("optimizer") if config else None
        if optimizer_opts is None:
            optimizer_opts = {"class": "adamw"}
        if not isinstance(optimizer_opts, dict):
            raise TypeError(f"JAX Updater: expected an optimizer dict, got {optimizer_opts!r}")
        self.optimizer_opts = dict(optimizer_opts)
        self.gradient_clip_global_norm = self.optimizer_opts.pop("gradient_clip_global_norm", None)
        self._optimizer = _make_optimizer(self.optimizer_opts, clip_global_norm=self.gradient_clip_global_norm)
        print(f"JAX updater: {self.optimizer_opts}", file=log.v3)

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


def _make_optimizer(opts: Dict[str, Any], *, clip_global_norm: Optional[float]) -> optax.GradientTransformation:
    """
    :param opts: the ``optimizer`` config dict, with a ``class`` entry
    :param clip_global_norm: if given, clip the global gradient norm before the update
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
    optimizer = func(learning_rate=1.0, **opts)
    if clip_global_norm:
        optimizer = optax.chain(optax.clip_by_global_norm(clip_global_norm), optimizer)
    return optimizer
