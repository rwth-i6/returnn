"""
Differentiable distributed collectives for the Torch backend.

See also :mod:`returnn.torch.distributed` for the DDP context setup;
this module holds low-level autograd-aware collective ops.
"""

from __future__ import annotations
import torch


# noinspection PyMethodOverriding,PyAbstractClass
class _AllReduceSum(torch.autograd.Function):
    """
    All-reduce-sum, whose gradient is again an all-reduce-sum.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        """
        :return: the summed tensor
        """
        import torch.distributed as dist

        ctx.group = group
        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        :return: the summed gradient
        """
        import torch.distributed as dist

        # The gradient of an all-reduce-sum is the all-reduce-sum of the upstream gradient:
        # each rank's input feeds every rank's output 1:1.
        grad = grad_output.clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


_HAVE_LIB_OPS = False
if hasattr(torch.library, "custom_op"):  # torch >= 2.4
    # An opaque op with a fake implementation and a registered backward:
    # AOT tracing (the compiled step of torch_cuda_graph, no Dynamo) runs on fake tensors,
    # which the direct collective of the autograd.Function above cannot take.
    # Only for the default group, a process group is no op argument.

    @torch.library.custom_op("returnn::all_reduce_sum", mutates_args=())
    def _lib_all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        out = x.clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out

    @_lib_all_reduce_sum.register_fake
    def _lib_all_reduce_sum_fake(x):
        return torch.empty_like(x)

    def _lib_setup_context(ctx, inputs, output):
        del ctx, inputs, output

    def _lib_backward(ctx, grad_output):
        del ctx
        return torch.ops.returnn.all_reduce_sum(grad_output.contiguous())

    torch.library.register_autograd("returnn::all_reduce_sum", _lib_backward, setup_context=_lib_setup_context)

    _HAVE_LIB_OPS = True


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
    if _HAVE_LIB_OPS and group is None and type(x) not in (torch.Tensor, torch.nn.Parameter):
        # a traced call (fake or functional tensors)
        return torch.ops.returnn.all_reduce_sum(x)
    return _AllReduceSum.apply(x, group)
