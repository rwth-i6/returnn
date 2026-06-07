"""
Differentiable distributed collectives for the Torch backend.

See also :mod:`returnn.torch.distributed` for the DDP context setup;
this module holds low-level autograd-aware collective ops.
"""

from __future__ import annotations
import torch


# noinspection PyMethodOverriding,PyAbstractClass,PyMissingOrEmptyDocstring
class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        import torch.distributed as dist

        ctx.group = group
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        import torch.distributed as dist

        # The gradient of an all-reduce-sum is the all-reduce-sum of the upstream gradient:
        # each rank's input feeds every rank's output 1:1.
        grad = grad_output.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


def all_reduce_sum(x: torch.Tensor, *, group=None) -> torch.Tensor:
    """
    Differentiable all-reduce (sum) across the distributed worker group.

    Unlike a plain ``torch.distributed.all_reduce``, this propagates gradients correctly
    (the backward all-reduce-sums the gradient),
    so it can be used inside the model forward, e.g. for SyncBatchNorm-style statistics.
    We avoid ``torch.distributed.nn.functional.all_reduce`` because it is deprecated,
    and its backward is only correct for sum anyway.

    :param x: local tensor, same shape on every worker
    :param group: process group, or None for the default group
    :return: the sum of ``x`` across all workers, same shape, differentiable
    """
    return _AllReduceSum.apply(x, group)
