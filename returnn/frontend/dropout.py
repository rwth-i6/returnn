"""
Dropout
"""

from __future__ import annotations
from typing import Union, Optional, Sequence
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["dropout", "dropout_broadcast_default"]


def dropout(
    source: Tensor,
    drop_prob: Union[float, Tensor],
    *,
    axis: Optional[Union[Dim, Sequence[Dim], bool]] = None,
    on_forward: bool = False,
) -> Tensor:
    """
    Applies dropout.

    Dropout will only be applied during training (unless you set on_forward=True).

    When dropout is applied, the output will be scaled by 1/dropout.

    :param source:
    :param drop_prob: 0.0 means to apply no dropout. 100% would mask everything.
        For every value in the tensor, the probability of it being dropped
        is drawn independently given this probability.
        The broadcasted axes are those not specified in ``axis``.
    :param axis: axis to apply dropout on. multiple axes can be specified.
        This defines the set of axes where the dropout mask is not broadcasted to.
        If None (default), it will not broadcast on any axis.
        False is the same as None, and allows to write ``axis=use_dropout_broadcast and ...feature_dim``.
        (RETURNN also has the ``noise_shape`` option but the ``axis`` option provides the same functionality.)
    :param on_forward: apply dropout during inference and training (so just always). otherwise only during training.
    """
    keep_prob = 1.0 - drop_prob
    if axis is None or axis is False:
        noise_dims = source.dims
    elif axis is True:
        raise ValueError("dropout axis=True is not valid")
    elif isinstance(axis, Dim):
        noise_dims = (axis,)
    else:
        noise_dims = axis
    if not set(noise_dims).issubset(source.dims):
        raise ValueError(f"dropout axis {axis} not in source {source}")

    if isinstance(keep_prob, (float, int)) and not 0 < keep_prob <= 1:
        raise ValueError("keep_prob must be a scalar tensor or a float in the range (0, 1], got %g" % keep_prob)

    # Do nothing if we know keep_prob == 1
    if isinstance(keep_prob, (float, int)) and keep_prob == 1:
        return source

    if on_forward:
        return _dropout(source, keep_prob, noise_dims=noise_dims)

    return rf.cond(
        pred=rf.get_run_ctx().is_train_flag_enabled(func=dropout),
        true_fn=lambda: _dropout(source, keep_prob, noise_dims=noise_dims),
        false_fn=lambda: source,
    )


def _dropout(
    x: Tensor, keep_prob: Union[float, Tensor], noise_dims: Sequence[Dim], seed=None, apply_correction_factor=True
) -> Tensor:
    """
    Computes dropout.

    Adopted from tf_util.dropout.
    Like :func:`tf.nn.dropout` but avoid :func:`tf.div` if possible.

    Note that in tf_util.dropout, we had special logic for recurrent loops:
    The mask would be created outside the loop
    and then the same mask would be used for every frame of the loop.
    We cannot really support such a logic for eager-based frameworks.

    :param x:
    :param keep_prob:
    :param noise_dims: other dims would broadcast
    :param seed: passed on to :func:`random` for the mask
    :param bool apply_correction_factor:
    """
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob + rf.random_uniform(dims=noise_dims, seed=seed, dtype=x.dtype, minval=0.0, maxval=1.0)
    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = rf.floor(random_tensor)
    if apply_correction_factor:
        # Apply the factor on binary_tensor, because that is potentially smaller than x.
        # Use `*(1/p)` instead of `/p` because that might be faster in some cases.
        binary_tensor *= 1.0 / keep_prob

    ret = x * binary_tensor
    return ret


def dropout_broadcast_default() -> bool:
    """
    Check the global RETURNN config
    whether we should broadcast on non-related dropout dimensions.

    Historically in RETURNN, when we did dropout in the feature dimension,
    we broadcasted the dropout mask over the other dimensions (e.g. time and batch).

    This function provides an easy global config controllable way to control this,
    via the option ``rf_dropout_broadcast``.

    The default for now:
    keep same as historical RETURNN, unless we find that this is really not a good idea.
    Then we might change the default via a new behavior version.

    Also see the option ``rf_att_dropout_broadcast``,
    which does the same for attention dropout.
    Although the default for attention dropout broadcasting was already changed with behavior version 19.

    :return: whether broadcasting should be used for dropout.
        Note that this does not actually effect :func:`dropout`.
        Any user of :func:`dropout` should check this explicitly.
    """
    from returnn.config import get_global_config

    default = True  # see docstring for reasoning
    config = get_global_config(raise_exception=False)
    if not config:
        return default

    return config.bool("rf_dropout_broadcast", default)
