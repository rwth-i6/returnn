"""
Normalization functions such as batch norm
"""


from __future__ import annotations
from typing import Optional, Sequence, Union, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["moments", "LayerNorm", "BatchNorm", "normalize", "Normalize"]


def moments(x: Tensor, axis: Union[Dim, Sequence[Dim]]) -> Tuple[Tensor, Tensor]:
    """
    :param x: input
    :param axis: the axis to be reduced, to calculate statistics over
    :return: mean, variance. it has the same shape as the input with the axis removed
    """
    mean = rf.reduce_mean(x, axis=axis)
    # stop_gradient does not change the gradient here
    variance = rf.reduce_mean(rf.squared_difference(x, rf.stop_gradient(mean)), axis=axis)
    return mean, variance


class LayerNorm(rf.Module):
    """
    `Layer normalization <https://arxiv.org/abs/1607.06450>`__.

    Note that we *just* normalize over the feature-dim axis here.
    This is consistent to the default behavior of :class:`tf.keras.layers.LayerNormalization`
    and also how it is commonly used in many models, including Transformer.

    However, there are cases where it would be common to normalize over all axes except batch-dim,
    or all axes except batch and time.
    For a more generic variant, see :func:`norm`.
    """

    def __init__(self, in_dim: Union[rf.Dim, Sequence[rf.Dim]], *, eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.scale = rf.Parameter([self.in_dim] if isinstance(self.in_dim, rf.Dim) else self.in_dim)
        self.scale.initial = 1.0
        self.bias = rf.Parameter(self.scale.dims)
        self.bias.initial = 0.0

    def __call__(self, x: Tensor) -> Tensor:
        mean, variance = rf.moments(x, axis=self.in_dim)
        norm_x = (x - mean) * rf.rsqrt(variance + self.eps)
        return norm_x * self.scale + self.bias


class BatchNorm(rf.Module):
    """
    Batch normalization. https://arxiv.org/abs/1502.03167

    Note that the default arguments differ from corresponding batch norm in RETURNN.
    See here for discussion on defaults: https://github.com/rwth-i6/returnn/issues/522

    We calculate statistics over all axes except the given in_dim.
    I.e. all other axes are reduced for the statistics.

    To compensate the normalization, there are learnable parameters gamma and beta
    (optional, used when option `affine` is True).

    The usual behavior depends on whether this is used in training or evaluation,
    although this often configurable in other frameworks.
    The usual behavior, in training::

        # Using statistics from current batch.
        mean_cur_batch, variance_cur_batch = moments(source, reduce_dims)
        y = (x - mean_cur_batch) / sqrt(variance_cur_batch + epsilon)
        y = gamma * y + beta

        # Updating running statistics for later use.
        mean = (1 - momentum) * mean + momentum * mean_cur_batch
        variance = (1 - momentum) * variance + momentum * variance_cur_batch

    The usual behavior, not in training (i.e. in evaluation)::

        # Using collected statistics. Not using statistics from current batch.
        y = (x - mean) / sqrt(variance + epsilon)
        y = gamma * y + beta

    """

    def __init__(
        self,
        in_dim: Dim,
        *,
        affine: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-3,
        use_mask: Optional[bool] = None,
    ):
        """
        :param in_dim: the feature dimension of the input
        :param affine: whether to use learnable parameters gamma and beta
        :param momentum: momentum for the running mean and variance
        :param eps: epsilon for the variance
        :param use_mask: whether to use a mask for dynamic spatial dims.
          This must be specified if the input has dynamic spatial dims.
          True would use the correct masking then. However, that is inconsistent to all other frameworks
            which ignore the masking, and also slower, and the fused op would not be used.
          False would be consistent to all other frameworks,
            and potentially allows for the use of an efficient fused op internally.
        """
        super().__init__()
        assert isinstance(in_dim, Dim)
        self.in_dim = in_dim
        self.use_mask = use_mask
        self.momentum = momentum
        self.eps = eps
        self.running_mean = rf.Parameter([in_dim], auxiliary=True)
        self.running_mean.initial = 0.0
        self.running_variance = rf.Parameter([in_dim], auxiliary=True)
        self.running_variance.initial = 1.0
        self.affine = affine
        self.gamma = None  # type: Optional[rf.Parameter]
        self.beta = None  # type: Optional[rf.Parameter]
        if self.affine:
            self.gamma = rf.Parameter([in_dim])
            self.gamma.initial = 1.0
            self.beta = rf.Parameter([in_dim])
            self.beta.initial = 0.0

    def __call__(self, source: Tensor) -> Tensor:
        assert self.in_dim in source.dims

        if any(d.need_masking() for d in source.dims if d != self.in_dim):
            if self.use_mask is None:
                raise ValueError(
                    f"{self}: use_mask must be specified if the input {source} has any dynamic spatial dims"
                )
            use_mask = self.use_mask
        else:
            use_mask = False  # not needed. False because this potentially enables an efficient fused op.

        if use_mask:
            # Generic implementation which supports masking.
            use_current_batch_stats = self.running_mean is None or rf.get_run_ctx().train_flag
            update_running_stats = self.running_mean is not None and rf.get_run_ctx().train_flag
            need_current_batch_stats = rf.opt_logical_or(use_current_batch_stats, update_running_stats)

            mean_cur_batch, variance_cur_batch = rf.cond(
                need_current_batch_stats,
                lambda: rf.moments(source, axis=[d for d in source.dims if d != self.in_dim]),
                lambda: (self.running_mean, self.running_variance),
            )

            def _update_running_stats():
                self.running_mean.assign_add((mean_cur_batch - self.running_mean) * self.momentum)
                self.running_variance.assign_add((variance_cur_batch - self.running_variance) * self.momentum)

            rf.cond(update_running_stats, _update_running_stats, lambda: None)

            mean, variance = rf.cond(
                use_current_batch_stats,
                lambda: (mean_cur_batch, variance_cur_batch),
                lambda: (self.running_mean, self.running_variance),
            )

            bn = (source - mean) * rf.rsqrt(variance + self.eps)
            if self.gamma is not None:
                bn *= self.gamma
            if self.beta is not None:
                bn += self.beta
            return bn

        # Fallback to specific backend implementation for the standard case without masking.
        # This fallback probably can internally use a more efficient implementation like from CuDNN.
        # In case of TF-net-dict backend, we wrap the RETURNN layer because we want efficient handling if possible,
        # which is potentially the use of a fused op,
        # and maybe reordering of dims.
        # https://github.com/rwth-i6/returnn_common/issues/89
        # noinspection PyProtectedMember
        return source._raw_backend.batch_norm(
            source=source,
            in_dim=self.in_dim,
            use_mask=use_mask,
            affine=self.affine,
            momentum=self.momentum,
            epsilon=self.eps,
            running_mean=self.running_mean,
            running_variance=self.running_variance,
            gamma=self.gamma,
            beta=self.beta,
        )


def normalize(a: Tensor, *, axis: Union[Dim, Sequence[Dim]], epsilon: float = 1e-6) -> Tensor:
    """
    Mean- and variance-normalize some input in the given input dimension(s),
    such that the resulting tensor has mean 0 and variance 1.

    If you want that this can be shifted and scaled again,
    you need additional parameters, cf. :class:`Normalize`.

    :param a: input
    :param axis: axis over which the mean and variance are computed
    :param epsilon: epsilon for numerical stability
    :return: (a - mean) / sqrt(variance + epsilon)
    """
    mean, variance = rf.moments(a, axis=axis)
    return (a - mean) * rf.rsqrt(variance + epsilon)


class Normalize(rf.Module):
    """
    :func:`normalize` with additional scale and bias
    """

    def __init__(
        self,
        *,
        param_dims: Union[Dim, Sequence[Dim]],
        epsilon: float = 1e-6,
        scale: bool = True,
        bias: bool = True,
    ):
        """
        :param param_dims: shape of the scale and bias parameters
        :param epsilon: epsilon for numerical stability
        :param scale: whether to include a trainable scale
        :param bias: whether to include a trainable bias
        """
        super(Normalize, self).__init__()
        self.epsilon = epsilon
        if isinstance(param_dims, Dim):
            param_dims = [param_dims]
        self.scale = None
        if scale:
            self.scale = rf.Parameter(dims=param_dims)
            self.scale.initial = 1.0
        self.bias = rf.Parameter(dims=param_dims) if bias else None

    def __call__(self, a: Tensor, *, axis: Union[Dim, Sequence[Dim]]):
        norm = normalize(a, axis=axis, epsilon=self.epsilon)
        if self.scale is not None:
            norm = self.scale * norm
        if self.bias is not None:
            norm = norm + self.bias
        return norm
