"""
Tiled Triton depthwise 1-D convolution for the torch backend (stride 1, no dilation).

torch routes a depthwise conv1d on CUDA to its native depthwise-2d kernels,
whose weight gradient parallelises over the (channel, tap) outputs
and reduces the whole batch x time extent inside each of them,
which for the Conformer conv block costs more than everything else in the layer.
The kernels are in :mod:`returnn.triton.depthwise_conv`, which describes them.
This module launches them on the (batch, time, channel) input with the (channel, width) filter
and adds the gradients, as an autograd.Function and, for a traced step, as opaque custom ops.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.autograd.function import once_differentiable

from returnn.triton import depthwise_conv as kernels


def is_available() -> bool:
    """:return: whether the kernel can run (needs Triton and a CUDA device)"""
    return kernels.triton is not None and torch.cuda.is_available()


def _launch_fwd(x, w, bias, pad_l: int, n_time_out: int, blocks) -> torch.Tensor:
    """:return: the conv output (batch, time_out, channel), the arguments as in :class:`_DepthwiseConv1d`"""
    n_batch, n_time_in, n_chan = x.shape
    width = w.shape[1]
    out = x.new_empty((n_batch, n_time_out, n_chan))
    n_rows = n_batch * n_time_out
    if n_time_in <= width:
        w_t = w.t().contiguous()
        grid = (kernels.cdiv(n_rows, kernels.BLOCK_R_ROWS), kernels.cdiv(n_chan, kernels.BLOCK_C_ROWS))
        kernels.dw_fwd_rows[grid](
            x,
            w_t,
            bias if bias is not None else w_t,
            n_rows,
            n_time_in,
            n_time_out,
            n_chan,
            pad_l,
            w_t.stride(1),
            w_t.stride(0),
            out,
            HAS_BIAS=bias is not None,
            BLOCK_R=kernels.BLOCK_R_ROWS,
            BLOCK_C=kernels.BLOCK_C_ROWS,
            KW=width,
        )
        return out
    block_r, block_c = blocks[0], blocks[1]
    grid = (kernels.cdiv(n_rows, block_r), kernels.cdiv(n_chan, block_c))
    kernels.dw_fwd[grid](
        x,
        w,
        bias if bias is not None else w,
        n_rows,
        n_time_in,
        n_time_out,
        n_chan,
        pad_l,
        w.stride(0),
        w.stride(1),
        out,
        HAS_BIAS=bias is not None,
        BLOCK_R=block_r,
        BLOCK_C=block_c,
        KW=width,
    )
    return out


def _launch_bwd(x, w, d_out, *, has_bias: bool, pad_l: int, blocks, need_dx: bool, need_dw_db: bool):
    """
    :return: (dx or None, the f32 sums (width + has_bias, channel) the filter and bias gradients are read from or None)
    """
    n_batch, n_time_in, n_chan = x.shape
    n_time_out = d_out.shape[1]
    width = w.shape[1]
    block_r, block_c, block_r_dw, block_c_dw = blocks
    dx = summed = None
    if need_dx:
        dx = torch.empty_like(x)
        n_rows = n_batch * n_time_in
        if n_time_out <= width:
            w_t = w.t().contiguous()
            grid = (kernels.cdiv(n_rows, kernels.BLOCK_R_ROWS), kernels.cdiv(n_chan, kernels.BLOCK_C_ROWS))
            kernels.dw_bwd_dx_rows[grid](
                d_out,
                w_t,
                n_rows,
                n_time_in,
                n_time_out,
                n_chan,
                pad_l,
                w_t.stride(1),
                w_t.stride(0),
                dx,
                BLOCK_R=kernels.BLOCK_R_ROWS,
                BLOCK_C=kernels.BLOCK_C_ROWS,
                KW=width,
            )
        else:
            grid = (kernels.cdiv(n_rows, block_r), kernels.cdiv(n_chan, block_c))
            kernels.dw_bwd_dx[grid](
                d_out,
                w,
                n_rows,
                n_time_in,
                n_time_out,
                n_chan,
                pad_l,
                w.stride(0),
                w.stride(1),
                dx,
                BLOCK_R=block_r,
                BLOCK_C=block_c,
                KW=width,
            )
    if need_dw_db:
        n_rows = n_batch * n_time_out
        rows_per_prog, n_splits, width_p, width_p2, num_warps = kernels.dw_launch(
            n_rows, width, has_bias=has_bias, block_r=block_r_dw, block_c=block_c_dw
        )
        partial = torch.empty((n_splits, width_p, n_chan), dtype=torch.float32, device=x.device)
        grid = (n_splits, kernels.cdiv(n_chan, block_c_dw))
        # noinspection PyArgumentList
        kernels.dw_bwd_dw[grid](
            x,
            d_out,
            n_rows,
            n_time_in,
            n_time_out,
            n_chan,
            pad_l,
            rows_per_prog,
            partial,
            HAS_BIAS=has_bias,
            BLOCK_R=block_r_dw,
            BLOCK_C=block_c_dw,
            KW=width,
            KW_P2=width_p2,
            KW_P=width_p,
            num_warps=num_warps,
        )
        summed = partial.sum(dim=0)
    return dx, summed


class _DepthwiseConv1d(torch.autograd.Function):
    """The conv with its gradients through the three kernels."""

    @staticmethod
    def forward(ctx, x, w, bias, pad_l, n_time_out, blocks):
        """
        :param ctx: keeps x and w and the launch parameters for the backward
        :param x: (batch, time_in, channel), contiguous
        :param w: (channel, width), contiguous
        :param bias: (channel,) or None
        :param pad_l: frames of zero padding before the first input frame
        :param n_time_out: output frames per batch entry
        :param blocks: (row block, channel block) for the conv kernels and (row block, channel block) for the dw kernel
        :return: (batch, time_out, channel)
        """
        out = _launch_fwd(x, w, bias, pad_l, n_time_out, blocks)
        ctx.save_for_backward(x, w)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.pad_l = pad_l
        ctx.blocks = blocks
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, d_out):
        """
        :param ctx: from :func:`forward`
        :param d_out: (batch, time_out, channel)
        :return: the gradients for x, w and bias, None for the other arguments
        """
        x, w = ctx.saved_tensors
        width = w.shape[1]
        has_bias = ctx.bias_dtype is not None
        dx, summed = _launch_bwd(
            x,
            w,
            d_out.contiguous(),
            has_bias=has_bias,
            pad_l=ctx.pad_l,
            blocks=ctx.blocks,
            need_dx=ctx.needs_input_grad[0],
            need_dw_db=ctx.needs_input_grad[1] or ctx.needs_input_grad[2],
        )
        dw = db = None
        if summed is not None:
            if ctx.needs_input_grad[1]:
                dw = summed[:width].t().contiguous().to(w.dtype)
            if has_bias and ctx.needs_input_grad[2]:
                db = summed[width].to(ctx.bias_dtype)
        return dx, dw, db, None, None, None


_HAVE_LIB_OPS = False
# torch.library.custom_op came in torch 2.4, but only torch 2.7 resolves the string annotations
# of `from __future__ import annotations` in this module's globals (before, `Tuple` fails at import)
if torch.__version__ >= (2, 7):
    # Opaque ops with fake implementations and a registered backward, like in rel_pos_att_triton:
    # AOT tracing (the compiled step of torch_cuda_graph, no Dynamo) runs on fake tensors,
    # which the Triton launch of the autograd.Function above cannot take.
    # Only a traced call goes through them, the eager path stays the autograd.Function.

    @torch.library.custom_op("returnn::depthwise_conv1d_fwd", mutates_args=())
    def _lib_fwd(
        x: torch.Tensor,
        w: torch.Tensor,
        bias: Optional[torch.Tensor],
        pad_l: int,
        n_time_out: int,
        block_r: int,
        block_c: int,
        block_r_dw: int,
        block_c_dw: int,
    ) -> torch.Tensor:
        # contiguous: Inductor feeds custom ops in whatever layout it likes, the fake promises a contiguous output
        bias = bias.contiguous() if bias is not None else None
        blocks = (block_r, block_c, block_r_dw, block_c_dw)
        return _launch_fwd(x.contiguous(), w.contiguous(), bias, pad_l, n_time_out, blocks)

    @_lib_fwd.register_fake
    def _lib_fwd_fake(x, w, bias, pad_l, n_time_out, block_r, block_c, block_r_dw, block_c_dw):
        del w, bias, pad_l, block_r, block_c, block_r_dw, block_c_dw
        return x.new_empty((x.shape[0], n_time_out, x.shape[2]))

    @torch.library.custom_op("returnn::depthwise_conv1d_bwd", mutates_args=())
    def _lib_bwd(
        x: torch.Tensor,
        w: torch.Tensor,
        d_out: torch.Tensor,
        has_bias: bool,
        pad_l: int,
        block_r: int,
        block_c: int,
        block_r_dw: int,
        block_c_dw: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dx, summed = _launch_bwd(
            x.contiguous(),
            w.contiguous(),
            d_out.contiguous(),
            has_bias=has_bias,
            pad_l=pad_l,
            blocks=(block_r, block_c, block_r_dw, block_c_dw),
            need_dx=True,
            need_dw_db=True,
        )
        return dx, summed

    @_lib_bwd.register_fake
    def _lib_bwd_fake(x, w, d_out, has_bias, pad_l, block_r, block_c, block_r_dw, block_c_dw):
        del d_out, pad_l, block_r, block_c, block_r_dw, block_c_dw
        return x.new_empty(tuple(x.shape)), x.new_empty((w.shape[1] + int(has_bias), w.shape[0]), dtype=torch.float32)

    def _lib_setup_context(ctx, inputs, output):
        del output
        x, w, bias, pad_l, _, block_r, block_c, block_r_dw, block_c_dw = inputs
        ctx.save_for_backward(x, w)
        ctx.bias_dtype = bias.dtype if bias is not None else None
        ctx.pad_l = pad_l
        ctx.blocks = (block_r, block_c, block_r_dw, block_c_dw)

    def _lib_backward(ctx, d_out):
        x, w = ctx.saved_tensors
        width = w.shape[1]
        has_bias = ctx.bias_dtype is not None
        dx, summed = torch.ops.returnn.depthwise_conv1d_bwd(x, w, d_out, has_bias, ctx.pad_l, *ctx.blocks)
        dw = summed[:width].t().contiguous().to(w.dtype)
        db = summed[width].to(ctx.bias_dtype) if has_bias else None
        return dx, dw, db, None, None, None, None, None, None

    torch.library.register_autograd("returnn::depthwise_conv1d_fwd", _lib_backward, setup_context=_lib_setup_context)

    _HAVE_LIB_OPS = True


def depthwise_conv1d(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    pad_l: int,
    n_time_out: int,
    blocks: Tuple[int, int, int, int] = (
        kernels.BLOCK_R_CONV,
        kernels.BLOCK_C_CONV,
        kernels.BLOCK_R_DW,
        kernels.BLOCK_C_DW,
    ),
) -> torch.Tensor:
    """
    :param x: (batch, time_in, channel)
    :param w: (channel, width), one filter per channel
    :param bias: (channel,) or None
    :param pad_l: frames of zero padding before the first input frame, "same" uses (width - 1) // 2
    :param n_time_out: output frames per batch entry, time_in + pad_l + pad_r - width + 1
    :param blocks: row and channel block of the conv kernels, then of the dw kernel, powers of two.
        The row loops of a window no longer than the filter use their own blocks.
    :return: (batch, time_out, channel), in the dtype of x, accumulated in f32
    """
    assert x.ndim == 3 and w.ndim == 2 and x.shape[2] == w.shape[0]
    assert bias is None or bias.shape == (w.shape[0],)
    operands = (x, w) if bias is None else (x, w, bias)
    assert all(t.dtype in (torch.float16, torch.bfloat16, torch.float32) for t in operands), [t.dtype for t in operands]
    assert all(v > 0 and v & (v - 1) == 0 for v in blocks), blocks
    bias = bias.contiguous() if bias is not None else None
    if _HAVE_LIB_OPS and type(x) not in (torch.Tensor, torch.nn.Parameter):
        # a traced call (fake or functional tensors)
        return torch.ops.returnn.depthwise_conv1d_fwd(x.contiguous(), w.contiguous(), bias, pad_l, n_time_out, *blocks)
    return _DepthwiseConv1d.apply(x.contiguous(), w.contiguous(), bias, pad_l, n_time_out, tuple(blocks))
