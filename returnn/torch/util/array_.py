"""
Array (Tensor) functions
"""

from __future__ import annotations
from typing import Optional, Tuple, Union
import torch


# noinspection PyShadowingBuiltins
def masked_select_bound(
    input: torch.Tensor, mask: torch.Tensor, *, bound: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Like :func:`masked_select`, but with a static output shape:
    a declared upper bound on the number of selected elements sizes the output,
    the selected elements packed at the front (in input order), zeros after.
    No data-dependent shapes and no host synchronization
    (:func:`torch.nonzero` has both),
    so this is usable under tracing / CUDA-graph capture
    (static traceable, see :mod:`returnn.frontend`).

    :param input: [mask_dims..., remaining_dims...]
    :param mask: [mask_dims...], binary mask to index with. if it has less dims than ``input``,
        the remaining dims are broadcasted.
    :param bound: upper bound on the number of selected elements.
        By default the full mask size (always valid).
        A tighter bound (e.g. the max total target len per batch, known from the batching constraints)
        shrinks the output buffer.
        The caller guarantees it -- selected elements beyond the bound are silently dropped.
    :return: (out, out_len):
        out [bound, remaining_dims...],
        out_len scalar int64 on the device (number of selected elements)
    """
    assert input.ndim >= mask.ndim
    assert all(input.shape[i] == mask.shape[i] for i in range(mask.ndim))
    mask_flat = mask.flatten()
    input_flat = input.flatten(end_dim=mask.ndim - 1)
    if bound is None:
        bound = mask_flat.shape[0]
    pos = torch.cumsum(mask_flat.to(torch.int64), dim=0) - 1
    # masked-out elements go to an extra dump slot, dropped below
    # (clamp also selected elements beyond the bound there, see the bound doc)
    pos = torch.where(mask_flat, pos.clamp(max=bound), torch.full_like(pos, bound))
    # Gather-based select (out[slot] = input_flat[inv[slot]]) via the inverse permutation,
    # with a custom backward that is ALSO a gather (grad_in[row] = grad_out[pos[row]]).
    # NEVER a scatter/index_put of a [rows, remaining...] tensor, in NEITHER direction:
    # under AOT/Inductor, index_put/scatter (also the index_add behind a plain index_select
    # backward, via functionalization) decomposes into a scatter whose index is broadcast
    # over the remaining dims and MATERIALIZED as a full int64 buffer [rows, remaining...]
    # (e.g. ~4-6GB for a [50-77k, 10k-vocab] log-prob tensor, plus full-size scatter buffers).
    # Here every index stays 1-D on both paths;
    # the only scatter left is the small int one building the inverse.
    n_rows = input_flat.shape[0]
    inv = torch.zeros((bound + 1,), dtype=torch.int64, device=input_flat.device)
    inv[pos] = torch.arange(n_rows, dtype=torch.int64, device=input_flat.device)
    out_len = mask_flat.sum()
    # slots beyond the selected count point at stale inv entries (0): zeroed inside the select,
    # like the zeros-init of a scatter formulation
    slot_valid = torch.arange(bound, dtype=torch.int64, device=input_flat.device) < out_len
    out = _GatherSelectBound.apply(input_flat, inv[:bound], pos, slot_valid)
    return out, out_len


def gather_relayout(
    values: torch.Tensor, *, inv: torch.Tensor, pos: torch.Tensor, slot_valid: torch.Tensor
) -> torch.Tensor:
    """
    Re-layout over the first axis: ``out[slot] = values[inv[slot]]`` where ``slot_valid``, else 0.
    Backward: ``grad_values[row] = grad_out[pos[row]]``
    (rows with ``pos == out size`` get 0).
    Gather in BOTH directions, see :class:`_GatherSelectBound`:
    a value scatter/index_put (also the index_add behind a plain index_select backward)
    would materialize its index broadcast over the remaining dims under AOT/Inductor.

    :param values: [rows, remaining...]
    :param inv: [out_size] int64, slot -> source row (arbitrary where not slot_valid)
    :param pos: [rows] int64, row -> slot, in [0..out_size] (out_size = dropped)
    :param slot_valid: [out_size] bool
    :return: [out_size, remaining...]
    """
    return _GatherSelectBound.apply(values, inv, pos, slot_valid)


class _GatherSelectBound(torch.autograd.Function):
    """see :func:`masked_select_bound` and :func:`gather_relayout` (the gather-both-ways core)"""

    @staticmethod
    def forward(ctx, input_flat: torch.Tensor, inv: torch.Tensor, pos: torch.Tensor, slot_valid: torch.Tensor):
        """
        :param ctx:
        :param input_flat: [rows, remaining...]
        :param inv: [bound] int64, slot -> source row (stale entries where slot_valid is False)
        :param pos: [rows] int64, row -> slot, in [0..bound] (bound = the dump slot)
        :param slot_valid: [bound] bool
        :return: [bound, remaining...]
        """
        ctx.save_for_backward(pos)
        out = torch.index_select(input_flat, 0, inv)
        out = torch.where(slot_valid.reshape((-1,) + (1,) * (input_flat.ndim - 1)), out, torch.zeros_like(out))
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """
        :param ctx:
        :param grad_out: [bound, remaining...]
        :return: grad for input_flat [rows, remaining...] (a gather: grad_out row at each input row's slot;
            masked-out rows (pos == bound) get 0), None for the index args
        """
        (pos,) = ctx.saved_tensors
        bound = grad_out.shape[0]
        # NOT via a [bound+1, remaining] concat with a zero dump row:
        # its constant zero-row write is schedulable anytime,
        # so Inductor emits it far before the single use and PINS the allocation there
        # (>1 GiB idle across half the step at the loq scale).
        # Clamp + zero the dumped rows instead: same values, no extra buffer.
        row_valid = (pos < bound).reshape((-1,) + (1,) * (grad_out.ndim - 1))
        grad_in = torch.index_select(grad_out, 0, pos.clamp(max=bound - 1))
        grad_in = torch.where(row_valid, grad_in, torch.zeros_like(grad_in))
        return grad_in, None, None, None


# noinspection PyShadowingBuiltins
def masked_select(input: torch.Tensor, mask: torch.Tensor, *, mask_len: Optional[Union[int, torch.Tensor]] = None):
    """
    Like :func:`torch.masked_select` but much more efficient,
    both in terms of memory and computation time,
    both on CPU and GPU.

    See here for the issues with :func:`torch.masked_select`:
    https://github.com/rwth-i6/returnn/issues/1584
    https://github.com/pytorch/pytorch/issues/30246
    https://github.com/pytorch/pytorch/issues/56896

    :param input: [mask_dims..., remaining_dims...]
    :param mask: [mask_dims...], binary mask to index with. if it has less dims than ``input``,
        the remaining dims are broadcasted.
    :param mask_len: if given, the length of the mask. this avoids a CUDA synchronization.
    :return: selected elements, shape [mask_len, remaining_dims...]
    """
    assert input.ndim >= mask.ndim
    assert all(input.shape[i] == mask.shape[i] for i in range(mask.ndim))
    mask_flat = mask.flatten()
    # Note: So far it seems that our custom nonzero is always slower than torch.nonzero,
    # thus we always use torch.nonzero here for now.
    # https://github.com/rwth-i6/returnn/pull/1593
    # We might change this in the future. See also:
    # https://github.com/pytorch/pytorch/issues/131256
    indices = torch.nonzero(mask_flat).squeeze(1)  # [out_len]
    if mask_len is not None:
        assert indices.shape[0] == mask_len
    input_flat = input.flatten(end_dim=mask.ndim - 1)
    return input_flat[indices]


def nonzero(mask: torch.Tensor, *, out_len: Union[int, torch.Tensor]) -> torch.Tensor:
    """
    This has the advantage over :func:`torch.nonzero`
    that we do not need to perform a CUDA synchronization.
    We can avoid that when we know the output length in advance.

    However, in my benchmarks, it seems this is slower than torch.nonzero.
    https://github.com/rwth-i6/returnn/pull/1593
    https://github.com/pytorch/pytorch/issues/131256

    :param mask: flattened (dim() == 1) mask, bool
    :param out_len:
    :return: indices of True elements, shape [out_len].
        like ``mask.nonzero().flatten()``
    """
    assert mask.dim() == 1 and mask.dtype == torch.bool
    # Sort currently does not support bool dtype on CUDA, thus cast to int.
    idx = torch.argsort(mask.to(torch.int8), stable=True, descending=True)  # [in_len]
    idx = idx[:out_len]  # [out_len]
    return idx


def sequence_mask(lengths: torch.Tensor, *, maxlen: Optional[int] = None) -> torch.Tensor:
    """
    Creates a boolean mask from sequence lengths.

    :param lengths: Tensor of shape [batch_size...] containing sequence lengths
    :param maxlen: Maximum length of the sequences. If None, uses the maximum value in lengths.
    :return: A boolean mask tensor of shape [batch_size..., maxlen]
    """
    if maxlen is None:
        maxlen = lengths.max()
    indices = torch.arange(0, maxlen, dtype=lengths.dtype, device=lengths.device)
    mask = indices < lengths[..., None]
    return mask


def sequence_mask_time_major(lengths: torch.Tensor, *, maxlen: Optional[int] = None) -> torch.Tensor:
    """
    Creates a boolean mask from sequence lengths.

    :param lengths: Tensor of shape [batch_size...] containing sequence lengths
    :param maxlen: Maximum length of the sequences. If None, uses the maximum value in lengths.
    :return: A boolean mask tensor of shape [maxlen, batch_size...]
    """
    if maxlen is None:
        maxlen = lengths.max()
    indices = torch.arange(0, maxlen, dtype=lengths.dtype, device=lengths.device)
    mask = indices[(slice(None),) + (None,) * lengths.ndim] < lengths[None]
    return mask
