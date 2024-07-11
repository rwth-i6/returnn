"""
Parameterizations using the parametrization API (:func:`register_parametrization`).

Also see:
https://github.com/rwth-i6/returnn/issues/1518
https://pytorch.org/tutorials/intermediate/parametrizations.html
"""

from __future__ import annotations
from returnn.tensor import Tensor
import returnn.frontend as rf


__all__ = ["weight_dropout", "WeightDropout", "weight_noise", "WeightNoise"]


def weight_dropout(module: rf.Module, param_name: str, *, drop_prob: float) -> rf.Module:
    """
    Apply weight dropout to a parameter of a module.

    This is only done in training.

    It uses :func:`gradient_checkpoint_scope` to avoid any memory overhead.

    In RETURNN TF-layers, this corresponds to the ``param_dropout`` option in a layer.
    Or in the RETURNN TF-layers :class:`RecLayer` with `ùnit="NativeLstm2"``,
    this was the ``rec_weight_dropout`` option.

    :param module:
    :param param_name: name of the parameter
    :param drop_prob: dropout probability
    :return: module
    """
    return rf.register_parametrization(module, param_name, WeightDropout(drop_prob))


class WeightDropout:
    """
    Use this for :func:`register_parametrization`, or via :func:`weight_dropout`.
    """

    def __init__(self, drop_prob: float):
        self.drop_prob = drop_prob

    def __call__(self, param: Tensor) -> Tensor:
        def _on_train() -> Tensor:
            with rf.gradient_checkpoint_scope():
                # on_forward=True because we already checked for train_flag
                return rf.dropout(param, drop_prob=self.drop_prob, on_forward=True)

        return rf.cond(rf.get_run_ctx().train_flag, _on_train, lambda: param)


def weight_noise(module: rf.Module, param_name: str, *, std: float) -> rf.Module:
    """
    Apply weight noise to a parameter of a module.
    This is also called variational noise.

    This is only done in training.

    It uses :func:`gradient_checkpoint_scope` to avoid any memory overhead.

    In RETURNN TF-layers, this corresponds to the ``param_variational_noise`` option in a layer.

    :param module:
    :param param_name: name of the parameter
    :param std: standard deviation of the noise
    :return: module
    """
    return rf.register_parametrization(module, param_name, WeightNoise(std))


class WeightNoise:
    """
    Use this for :func:`register_parametrization`, or via :func:`weight_noise`.
    """

    def __init__(self, std: float):
        self.std = std

    def __call__(self, param: Tensor) -> Tensor:
        def _on_train() -> Tensor:
            with rf.gradient_checkpoint_scope():
                noise = rf.random_normal(param.dims, dtype=param.dtype, stddev=self.std)
                return param + noise

        return rf.cond(rf.get_run_ctx().train_flag, _on_train, lambda: param)
