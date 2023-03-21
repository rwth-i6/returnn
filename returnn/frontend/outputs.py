"""
mark_as_output, mark_as_loss
"""

from __future__ import annotations
from typing import Optional, Sequence, TypeVar
from returnn.tensor import Tensor

T = TypeVar("T")

__all__ = ["mark_as_output", "mark_as_loss"]


def mark_as_loss(
    loss: Tensor,
    name: str,
    *,
    scale: Optional[float] = 1.0,
    as_error: bool = False,
    use_normalized_loss: bool = False,
    use_flatten_frames: bool = True,
    custom_inv_norm_factor: Optional[Tensor] = None,
) -> None:
    """
    Mark the given loss tensor as a loss.
    This has the effect that it is specially handled by RETURNN.
    Specifically, the optimizer can use it in training,
    and it is used for reporting per batch or per epoch,
    and for learning rate scheduling.

    This currently uses :class:`AsIsLoss` in RETURNN
    but this is an implementation detail and might change.

    :param loss:
    :param name: name of the loss. this name is used for reporting by RETURNN, and also for LR scheduling.
    :param scale: scale the loss by this factor for the training optimizer
      (but not for any reporting). setting to 0.0 has the effect that this loss is not used by the optimizer.
    :param as_error: if True, this loss is reported as an error instead of a loss,
      and not used by the training optimizer.
      This is by convention sth like the frame-error or edit-distance, and usually not differentiable anyway.
    :param bool use_flatten_frames: If True, will use :func:`returnn.tf.util.basic.flatten_with_seq_len_mask`,
      i.e. a "packed" sequence with the padded frames removed, and accumulates over that.
      This can be more efficient, also because it will further optimize incoming computations
      and e.g. skip softmax computations right before on the padded frames.
      This can also avoid issues with inf/nan in some cases.
      If False, it will mask the loss to 0 in the padded frames and accumulate over that.
      Typically, setting this to True (default) is both more efficient and better.
    :param bool use_normalized_loss: the loss used in optimization will be normalized.
      E.g. if the overall normalization is sum(loss)/sum(num_frames), this is also what the optimizer will use,
      otherwise the optimizer will just use sum(loss).
    :param custom_inv_norm_factor:
      The standard norm factor is 1/sum(target_seq_len) if the target has a time-axis,
      or 1/sum(output_seq_len) if there is no target and the output has a time-axis,
      or 1 otherwise. (See :func:`Loss.init` for details.)
      This is used for proper normalization of accumulated loss/error per epoch
      and also proper normalization per batch for reporting,
      no matter if use_normalized_loss is True or False.
      If you want to change this norm factor, you can set this.
      Basically, for all reporting, it uses sum(loss) * sum(custom_inv_norm_factor).
    """
    # noinspection PyProtectedMember
    loss._raw_backend.mark_as_loss(
        loss=loss,
        name=name,
        scale=scale,
        as_error=as_error,
        use_normalized_loss=use_normalized_loss,
        use_flatten_frames=use_flatten_frames,
        custom_inv_norm_factor=custom_inv_norm_factor,
    )


def mark_as_output(tensor: Tensor, name: str, *, shape: Optional[Sequence[int]] = None) -> None:
    """
    Mark this as an output.
    This has the effect that RETURNN will in any case construct the corresponding layer.
    Also see :func:`mark_as_default_output`.

    This is intended mostly for forwarding, or exporting the model (TF graph, TFLite, ONNX, etc).
    You must specify a shape to have the output shape (order of dims) well-defined
    (if not specified, we check if some defaults are possible, like BTF, or BF).

    :param tensor:
    :param name:
    :param shape: this specifies the order of the dims of the output, such that it is well-defined
        for some external application.
        If not specified, we try to infer BTF or BF as default, if that works, otherwise it will be an error.
    """
    # noinspection PyProtectedMember
    tensor._raw_backend.mark_as_output(tensor=tensor, name=name, shape=shape)
