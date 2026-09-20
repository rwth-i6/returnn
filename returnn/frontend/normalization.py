"""
Normalization functions such as batch norm
"""

from __future__ import annotations
from typing import Optional, Sequence, Union, Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from . import _utils


__all__ = [
    "moments",
    "layer_norm",
    "rms_norm",
    "LayerNorm",
    "RMSNorm",
    "GroupNorm",
    "GroupNormSpatial",
    "BatchNorm",
    "batch_norm_distributed_default",
    "normalize",
    "Normalize",
]


def moments(
    x: Tensor,
    axis: Union[Dim, Sequence[Dim]],
    *,
    use_mask: bool = True,
    correction: Union[int, float, Tensor] = 0,
    distributed: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    :param x: input
    :param axis: the axis (or axes) to be reduced, to calculate statistics over
    :param use_mask: whether to use a mask for dynamic spatial dims in the reduction
    :param correction:
        The variance will be estimated by ``sum((x - mean)**2) / (n-correction)``
        where ``n`` is the number of elements in the axis (or the axes)
        (with ``use_mask=True``, taking masking into account, using :func:`num_elements_of_shape`).
        The default ``correction=0`` will return the biased variance estimation.
        ``correction=1`` is the `Bessel correction <https://en.wikipedia.org/wiki/Bessel%27s_correction>`__
        and will return the unbiased variance estimation.
        In PyTorch, there was an argument ``unbiased`` for this, but this changed recently to ``correction``
        (`PyTorch issue #61492 <https://github.com/pytorch/pytorch/issues/61492>`__,
         `Python Array API Standard
          <https://data-apis.org/array-api/latest/API_specification/generated/array_api.var.html>`__).
        In PyTorch, the default is ``correction=1``, which is the unbiased variance estimation,
        while in most other frameworks, the default is ``correction=0``, which is the biased variance estimation.
    :param distributed:
        If True and a Torch DDP process group exists (world size > 1),
        compute the statistics over the global batch across all workers,
        by all-reducing the per-worker sums and the count (differentiable),
        in two passes so that the variance cannot cancel, as in torch.nn.SyncBatchNorm.
        Default False keeps the per-worker (local) statistics.
    :return: tuple (mean, variance). it has the same shape as the input with the axis removed
    """
    if distributed:
        # Two-pass statistics over the global batch, accumulated in float32.
        # The one-pass variance E[x^2] - E[x]^2 catastrophically cancels whenever the mean dominates
        # the variance, in float32 just as in bf16: the difference is then noise and can go negative,
        # which gives NaNs via rsqrt(variance + eps).
        # torch.nn.SyncBatchNorm instead combines per-worker Welford statistics, which does not cancel.
        compute_dtype = x.dtype
        x = rf.cast(x, "float32")
        count = _global_num_elements(axis, use_mask=use_mask, device=x.device)
        mean = rf.reduce_sum(x, axis=axis, use_mask=use_mask, distributed=True) / count
        # stop_gradient does not change the gradient here: the deviations sum to zero over the global batch
        sq_dev = rf.squared_difference(x, rf.stop_gradient(mean))
        variance = rf.reduce_sum(sq_dev, axis=axis, use_mask=use_mask, distributed=True) / count
        if isinstance(correction, Tensor) or correction != 0:
            variance *= count / (count - correction)
        return rf.cast(mean, compute_dtype), rf.cast(variance, compute_dtype)
    # Accumulated in float32 like the distributed branch, and rounded once at the end.
    # A low-precision reduction drifts by several ulps over a few thousand frames, and it drifts
    # differently per storage: a padded or exact packed buffer takes a direct mean, a bounded one
    # divides a masked sum by its count, and under Torch CUDA autocast the sum promotes where the
    # mean does not. The statistics then depend on the layout, and a running mean keeps that error.
    compute_dtype = x.dtype
    x = rf.cast(x, "float32")
    mean = rf.reduce_mean(x, axis=axis, use_mask=use_mask)
    # stop_gradient does not change the gradient here
    variance = rf.reduce_mean(rf.squared_difference(x, rf.stop_gradient(mean)), axis=axis, use_mask=use_mask)
    if isinstance(correction, Tensor) or correction != 0:
        n = rf.num_elements_of_shape(axis, use_mask=use_mask)
        variance *= n / (n - correction)
    return rf.cast(mean, compute_dtype), rf.cast(variance, compute_dtype)


def _global_num_elements(axis: Union[Dim, Sequence[Dim]], *, use_mask: bool, device: Optional[str]) -> Tensor:
    """
    :param axis: the dim or dims which are reduced
    :param use_mask: whether padded frames are excluded, as in the reduction itself
    :param device: where the count is needed, so it does not force a host sync under graph capture
    :return: number of reduced elements, summed over the Torch DDP workers, as a float32 tensor
    """
    count = rf.num_elements_of_shape(axis, use_mask=use_mask, device=device)
    if isinstance(count, Tensor):
        count = rf.cast(rf.copy_to_device(count, device), "float32")
    else:
        # static dims (or use_mask=False) give a plain int, which is the same on every worker
        count = rf.constant(count, dims=(), dtype="float32", device=device)
    # noinspection PyProtectedMember
    return count._raw_backend.reduce_distributed(count, mode="sum")


def layer_norm(
    x: Tensor, *, in_dim: Union[Dim, Sequence[Dim]], scale: Tensor, bias: Optional[Tensor], eps: float
) -> Tensor:
    """
    Normalizes x over in_dim to zero mean and unit variance and applies the scale and the bias, see :class:`LayerNorm`.

    :param x: input
    :param in_dim: the dim or dims to normalize over
    :param scale: over in_dim
    :param bias: over in_dim, or None
    :param eps: added to the variance
    :return: the normalized x, with the dims of x
    """
    return _utils.get_backend_from_tensors(x).layer_norm(x, in_dim=in_dim, scale=scale, bias=bias, eps=eps)


def rms_norm(
    x: Tensor, *, in_dim: Union[Dim, Sequence[Dim]], scale: Tensor, bias: Optional[Tensor], eps: float
) -> Tensor:
    """
    Divides x by its root mean square over in_dim and applies the scale and the bias, see :class:`RMSNorm`.

    :param x: input
    :param in_dim: the dim or dims to normalize over
    :param scale: over in_dim
    :param bias: over in_dim, or None
    :param eps: added to the mean square
    :return: the normalized x, with the dims of x
    """
    return _utils.get_backend_from_tensors(x).rms_norm(x, in_dim=in_dim, scale=scale, bias=bias, eps=eps)


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

    def __init__(self, in_dim: Union[rf.Dim, Sequence[rf.Dim]], *, eps: float = 1e-6, with_bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.scale = rf.Parameter([self.in_dim] if isinstance(self.in_dim, rf.Dim) else self.in_dim)
        self.scale.initial = 1.0
        self.bias = None
        if with_bias:
            self.bias = rf.Parameter(self.scale.dims)
            self.bias.initial = 0.0

    def __call__(self, x: Tensor) -> Tensor:
        return layer_norm(x, in_dim=self.in_dim, scale=self.scale, bias=self.bias, eps=self.eps)


class RMSNorm(rf.Module):
    """
    `Root Mean Square Layer Normalization (RMSNorm) <https://arxiv.org/abs/1910.07467>`__.

    Alternative to :class:`LayerNorm` that uses the root-mean-square of the input as the normalization factor.
    I.e. the main difference to layer norm is: *No subtraction of mean.*

    Note, the bias here is optional, *and disabled by default* (in line with most implementations of RMSNorm),
    unlike our :class:`LayerNorm`, where the bias is enabled by default.
    """

    def __init__(self, in_dim: Union[rf.Dim, Sequence[rf.Dim]], *, eps: float = 1e-6, with_bias: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.scale = rf.Parameter([self.in_dim] if isinstance(self.in_dim, rf.Dim) else self.in_dim)
        self.scale.initial = 1.0
        self.bias = None
        if with_bias:
            self.bias = rf.Parameter(self.scale.dims)
            self.bias.initial = 0.0

    def __call__(self, x: Tensor) -> Tensor:
        return rms_norm(x, in_dim=self.in_dim, scale=self.scale, bias=self.bias, eps=self.eps)


class GroupNorm(rf.Module):
    """
    `Group normalization <https://arxiv.org/abs/1803.08494>`__.

    Note: this is non-standard.
    It reduces the statistics only over the in-group channels, independently per spatial/time position,
    i.e. it does *not* pool over the spatial dims.
    This differs from :class:`torch.nn.GroupNorm` / the GroupNorm paper.
    For the standard, spatially-pooled (batch-independent) variant, use :class:`GroupNormSpatial`.
    """

    def __init__(self, in_dim: Union[rf.Dim, Sequence[rf.Dim]], *, num_groups: Union[int, Dim], eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.num_groups = num_groups if isinstance(num_groups, Dim) else Dim(num_groups, name="groups")
        self.in_group_dim = in_dim.ceildiv_left(num_groups)
        self.eps = eps
        self.scale = rf.Parameter([self.in_dim] if isinstance(self.in_dim, rf.Dim) else self.in_dim)
        self.scale.initial = 1.0
        self.bias = rf.Parameter(self.scale.dims)
        self.bias.initial = 0.0

    def __call__(self, x: Tensor) -> Tensor:
        x = rf.split_dims(x, axis=self.in_dim, dims=[self.num_groups, self.in_group_dim])
        mean, variance = rf.moments(x, axis=self.in_group_dim)
        norm_x = (x - mean) * rf.rsqrt(variance + self.eps)
        norm_x, _ = rf.merge_dims(norm_x, dims=[self.num_groups, self.in_group_dim], out_dim=self.in_dim)
        return _utils.keep_dtype(norm_x * self.scale + self.bias, x.dtype)


class GroupNormSpatial(rf.Module):
    """
    Standard (spatially-pooled) `group normalization <https://arxiv.org/abs/1803.08494>`__,
    equivalent to :class:`torch.nn.GroupNorm`:
    the mean/variance are computed over the in-group channels AND the given spatial dim(s),
    per batch element and per group (so batch-independent).

    Unlike :class:`GroupNorm` (which reduces only over the in-group channels, i.e. per spatial position),
    the spatial dim(s) must be passed explicitly to ``__call__`` --
    they cannot be reliably inferred from the input tensor.
    ``spatial_dim`` may be a single :class:`Dim` or a sequence of dims.
    With ``num_groups=1`` this normalizes over all channels and spatial positions per sequence
    (what torchaudio's Conformer uses); it is not the same as :class:`LayerNorm`.
    """

    def __init__(self, in_dim: Union[rf.Dim, Sequence[rf.Dim]], *, num_groups: Union[int, Dim], eps: float = 1e-6):
        super().__init__()
        self.in_dim = in_dim
        self.num_groups = num_groups if isinstance(num_groups, Dim) else Dim(num_groups, name="groups")
        self.in_group_dim = in_dim.ceildiv_left(num_groups)
        self.eps = eps
        self.scale = rf.Parameter([self.in_dim] if isinstance(self.in_dim, rf.Dim) else self.in_dim)
        self.scale.initial = 1.0
        self.bias = rf.Parameter(self.scale.dims)
        self.bias.initial = 0.0

    def __call__(self, x: Tensor, *, spatial_dim: Union[Dim, Sequence[Dim]]) -> Tensor:
        x = rf.split_dims(x, axis=self.in_dim, dims=[self.num_groups, self.in_group_dim])
        spatial_dims = [spatial_dim] if isinstance(spatial_dim, Dim) else list(spatial_dim)
        mean, variance = rf.moments(x, axis=[self.in_group_dim] + spatial_dims)
        norm_x = (x - mean) * rf.rsqrt(variance + self.eps)
        norm_x, _ = rf.merge_dims(norm_x, dims=[self.num_groups, self.in_group_dim], out_dim=self.in_dim)
        return _utils.keep_dtype(norm_x * self.scale + self.bias, x.dtype)


def batch_norm_distributed_default() -> bool:
    """
    Global-config default for :class:`BatchNorm` ``distributed`` (SyncBatchNorm-style global stats).
    Controlled via the option ``rf_batch_norm_distributed``,
    mirroring ``rf_dropout_broadcast``.
    Default False (per-worker stats).
    Only has an effect under Torch DDP; enable it explicitly there.
    """
    from returnn.config import get_global_config

    config = get_global_config(raise_exception=False)
    if not config:
        return False
    return config.bool("rf_batch_norm_distributed", False)


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
        track_running_stats: bool = True,
        use_mask: Optional[bool] = None,
        distributed: Optional[bool] = None,
    ):
        """
        :param in_dim: the feature dimension of the input
        :param affine: whether to use learnable parameters gamma and beta
        :param momentum: momentum for the running mean and variance
        :param eps: epsilon for the variance
        :param track_running_stats:
            If True, uses statistics of the current batch for normalization during training,
            and the tracked statistics (running mean and variance) during evaluation.
            If False, uses statistics of the current batch for normalization during both training and evaluation.
        :param use_mask: whether to use a mask for dynamic spatial dims.
          This must be specified if the input has dynamic spatial dims.
          True would use the correct masking then. However, that is inconsistent to all other frameworks
            which ignore the masking, and also slower, and the fused op would not be used.
          False would be consistent to all other frameworks,
            and potentially allows for the use of an efficient fused op internally.
        :param distributed: compute batch statistics over the global batch across all DDP workers
          (SyncBatchNorm-style) instead of per-worker.
          None (default) reads the global config option ``rf_batch_norm_distributed`` (default False).
          Only meaningful under Torch DDP grad-sync.
        """
        super().__init__()
        assert isinstance(in_dim, Dim)
        self.in_dim = in_dim
        self.use_mask = use_mask
        self.distributed = distributed if distributed is not None else batch_norm_distributed_default()
        self.momentum = momentum
        self.eps = eps
        if track_running_stats:
            self.running_mean = rf.Parameter([in_dim], auxiliary=True)
            self.running_mean.initial = 0.0
            self.running_variance = rf.Parameter([in_dim], auxiliary=True)
            self.running_variance.initial = 1.0
        else:
            self.running_mean = None
            self.running_variance = None
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
                use_mask = rf.use_mask_default(default=True, func_name="BatchNorm")
            else:
                use_mask = self.use_mask
        else:
            use_mask = False  # not needed. False because this potentially enables an efficient fused op.

        if use_mask or self.distributed:
            # Generic implementation which supports masking (and required for distributed/global stats).
            train_flag = rf.get_run_ctx().is_train_flag_enabled(func=BatchNorm.__call__)
            use_current_batch_stats = self.running_mean is None or train_flag
            update_running_stats = self.running_mean is not None and train_flag
            need_current_batch_stats = rf.opt_logical_or(use_current_batch_stats, update_running_stats)

            mean_cur_batch, variance_cur_batch = rf.cond(
                need_current_batch_stats,
                lambda: rf.moments(
                    source,
                    axis=[d for d in source.dims if d != self.in_dim],
                    use_mask=use_mask,
                    distributed=self.distributed,
                ),
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
