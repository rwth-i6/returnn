"""
One LSTM layer over every frame of a padded buffer as one op, so a traced step keeps the recurrence of
:func:`torch.lstm` in one node instead of unrolling the frames, see :func:`padded_lstm`.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, List, Optional, Tuple

import torch

from .custom_op import custom_op

_LstmOutputs = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def padded_lstm(
    source: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    ff_weight: torch.Tensor,
    rec_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs one LSTM layer over every frame of a padded buffer, the padded frames included.

    Under tracing the whole recurrence is one op (torch >= 2.4), eager it is the plain :func:`torch.lstm`.

    :param source: [frames, batch, in]
    :param h0: [1, batch, out] hidden state before the first frame
    :param c0: [1, batch, out] cell state before the first frame
    :param ff_weight: [4 * out, in]
    :param rec_weight: [4 * out, out]
    :param bias: [4 * out], None for none
    :return: output [frames, batch, out], the hidden and the cell state after the last frame, [1, batch, out] each
    """
    if _OPS is not None:
        fwd, _ = _OPS
        return fwd(source, h0, c0, ff_weight, rec_weight, bias)
    return _lstm(source, h0, c0, ff_weight, rec_weight, bias, train=False)


def _lstm(
    source: torch.Tensor,
    h0: torch.Tensor,
    c0: torch.Tensor,
    ff_weight: torch.Tensor,
    rec_weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    :return: output, hidden state and cell state of :func:`torch.lstm`, the one bias given as the two of cuDNN
    """
    if bias is None:
        params, has_biases = (ff_weight, rec_weight), False
    else:
        half_bias = bias * 0.5
        params, has_biases = (ff_weight, rec_weight, half_bias, half_bias), True
    out, hy, cy = torch.lstm(source, (h0, c0), params, has_biases, 1, 0.0, train, False, False)
    return out, hy, cy


@contextlib.contextmanager
def _autograd_inside_op():
    """lets autograd record inside an op kernel, which runs with the autograd keys excluded"""
    excluded = torch._C._dispatch_tls_local_exclude_set()
    for key in (
        torch._C.DispatchKey.AutogradFunctionality,
        torch._C.DispatchKey.AutogradOther,
        torch._C.DispatchKey.ADInplaceOrView,
    ):
        excluded = excluded.remove(key)
    with torch._C._ForceDispatchKeyGuard(torch._C._dispatch_tls_local_include_set(), excluded):
        yield


def _register_ops() -> Optional[Tuple[Callable[..., _LstmOutputs], Callable[..., Tuple[torch.Tensor, ...]]]]:
    """
    :return: the forward op and its backward op, None for a torch without :func:`torch.library.custom_op`
    """
    if not hasattr(torch.library, "custom_op"):
        return None

    @custom_op("returnn::padded_lstm", mutates_args=())
    def fwd(
        source: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        ff_weight: torch.Tensor,
        rec_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _lstm(source, h0, c0, ff_weight, rec_weight, bias, train=False)

    @fwd.register_fake
    def _fwd_fake(
        source: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        ff_weight: torch.Tensor,
        rec_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del ff_weight, rec_weight, bias
        out = source.new_empty((source.shape[0], source.shape[1], h0.shape[-1]))
        return out, torch.empty_like(h0), torch.empty_like(c0)

    @custom_op("returnn::padded_lstm_bwd", mutates_args=())
    def bwd(
        source: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        ff_weight: torch.Tensor,
        rec_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        d_out: Optional[torch.Tensor],
        d_hy: Optional[torch.Tensor],
        d_cy: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """gradients of the forward's inputs, the recurrence run once more under autograd, each in its own storage"""
        leaves: List[torch.Tensor] = [t.detach() for t in (source, h0, c0, ff_weight, rec_weight)]
        if bias is not None:
            leaves.append(bias.detach())
        given = [(index, grad) for index, grad in enumerate((d_out, d_hy, d_cy)) if grad is not None]
        grads: List[Optional[torch.Tensor]] = [None] * len(leaves)
        if given:
            with _autograd_inside_op(), torch.enable_grad():
                for leaf in leaves:
                    leaf.requires_grad_()
                outs = _lstm(
                    leaves[0],
                    leaves[1],
                    leaves[2],
                    leaves[3],
                    leaves[4],
                    leaves[5] if bias is not None else None,
                    train=True,
                )
                grads = list(
                    torch.autograd.grad(
                        [outs[index] for index, _ in given], leaves, [grad for _, grad in given], allow_unused=True
                    )
                )
        result = [torch.zeros_like(leaf) if grad is None else grad.clone() for grad, leaf in zip(grads, leaves)]
        if bias is None:
            result.append(source.new_empty(0))
        return result[0], result[1], result[2], result[3], result[4], result[5]

    @bwd.register_fake
    def _bwd_fake(
        source: torch.Tensor,
        h0: torch.Tensor,
        c0: torch.Tensor,
        ff_weight: torch.Tensor,
        rec_weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        d_out: Optional[torch.Tensor],
        d_hy: Optional[torch.Tensor],
        d_cy: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        del d_out, d_hy, d_cy
        bias_grad = source.new_empty(0) if bias is None else torch.empty_like(bias)
        return (
            torch.empty_like(source),
            torch.empty_like(h0),
            torch.empty_like(c0),
            torch.empty_like(ff_weight),
            torch.empty_like(rec_weight),
            bias_grad,
        )

    def _setup_context(ctx: Any, inputs: Tuple[Optional[torch.Tensor], ...], output: Any) -> None:
        del output
        ctx.save_for_backward(*inputs)

    def _backward(
        ctx: Any, d_out: Optional[torch.Tensor], d_hy: Optional[torch.Tensor], d_cy: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        source, h0, c0, ff_weight, rec_weight, bias = ctx.saved_tensors
        grads = bwd(source, h0, c0, ff_weight, rec_weight, bias, d_out, d_hy, d_cy)
        return grads[0], grads[1], grads[2], grads[3], grads[4], None if bias is None else grads[5]

    fwd.register_autograd(_backward, setup_context=_setup_context)
    return fwd, bwd


_OPS = _register_ops()
