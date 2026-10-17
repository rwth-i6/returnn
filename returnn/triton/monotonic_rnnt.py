"""
Triton kernels of the packed monotonic RNN-T loss, shared by the backends.
The torch backend launches them in :mod:`returnn.torch.util.monotonic_rnnt_triton`
for :mod:`returnn.torch.util.monotonic_rnnt`.

The loss needs three numbers per lattice cell, the log normalizer, the blank log probability and the log
probability of the next reference label, and its gradient needs the softmax of the cell again. Those two
passes are bandwidth bound over ``[cells, vocab]``.

The forward-backward recursion is the opposite, tiny tensors and one step per frame, so as torch ops it is
pure launch overhead, about 10 kernels per frame and two sweeps. It runs here as one kernel per sweep with
one program per sequence instead, the frames looped inside the kernel and the prefixes held as a block.

Nothing in here materializes a normalized ``[cells, vocab]`` tensor, which is what makes the loss fit: at
the production budget that tensor alone is several GB.

Every kernel takes its outputs last, since ``jax_triton.triton_call`` binds the outputs after the inputs.
The Triton import is guarded, so this module imports without Triton and then defines no kernels.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # optional dependency
    triton = tl = None


# vocabulary block and warps of the two cell kernels, one program per cell
CELL_BLOCK = 1024
CELL_NUM_WARPS = 8


def block_for(max_prefix: int) -> int:
    """
    :param max_prefix: prefixes the scan kernels hold as one block
    :return: the next power of two, at least 16
    """
    block = 16
    while block < max_prefix:
        block *= 2
    return block


if triton is not None:
    _NEG_INF = tl.constexpr(-3.4028234663852886e38)

    # noinspection PyPep8Naming
    @triton.jit
    def cell_stats_kernel(
        logits_ptr,
        label_ptr,
        vocab,
        blank,
        row_max_ptr,
        log_sum_ptr,
        blank_lp_ptr,
        label_lp_ptr,
        BLOCK: tl.constexpr,
    ):
        """one program per cell, reduces its row to the normalizer and the two log probabilities"""
        cell = tl.program_id(0)
        base = cell.to(tl.int64) * vocab
        # the running maximum starts at the sentinel, a block of minus infinity would give inf minus inf
        running_max = _NEG_INF
        running_sum = 0.0
        for start in range(0, vocab, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < vocab
            x = tl.load(logits_ptr + base + offs, mask=mask, other=_NEG_INF).to(tl.float32)
            block_max = tl.max(x, axis=0)
            new_max = tl.maximum(running_max, block_max)
            running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(
                tl.where(mask, tl.exp(x - new_max), 0.0), axis=0
            )
            running_max = new_max
        # a row of minus infinity has nothing to normalize, 0 keeps its log probabilities at minus infinity
        log_sum = tl.where(running_sum > 0.0, tl.log(running_sum), 0.0)
        label = tl.load(label_ptr + cell)
        blank_logit = tl.load(logits_ptr + base + blank).to(tl.float32)
        label_logit = tl.load(logits_ptr + base + label).to(tl.float32)
        tl.store(row_max_ptr + cell, running_max)
        tl.store(log_sum_ptr + cell, log_sum)
        # the maximum comes off first, added to it a large logit would swallow the log sum
        tl.store(blank_lp_ptr + cell, (blank_logit - running_max) - log_sum)
        tl.store(label_lp_ptr + cell, (label_logit - running_max) - log_sum)

    # noinspection PyPep8Naming
    @triton.jit
    def cell_grad_kernel(
        logits_ptr,
        label_ptr,
        row_max_ptr,
        log_sum_ptr,
        blank_grad_ptr,
        label_grad_ptr,
        vocab,
        blank,
        out_ptr,
        BLOCK: tl.constexpr,
    ):
        """one program per cell, writes the gradient of its row from the two edge posteriors"""
        cell = tl.program_id(0)
        base = cell.to(tl.int64) * vocab
        row_max = tl.load(row_max_ptr + cell)
        log_sum = tl.load(log_sum_ptr + cell)
        blank_grad = tl.load(blank_grad_ptr + cell)
        label_grad = tl.load(label_grad_ptr + cell)
        label = tl.load(label_ptr + cell)
        total = blank_grad + label_grad
        for start in range(0, vocab, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            mask = offs < vocab
            x = tl.load(logits_ptr + base + offs, mask=mask, other=float("-inf")).to(tl.float32)
            # a cell carrying no posterior contributes nothing, and its row may not even be normalizable
            grad = tl.where(total == 0.0, 0.0, total * tl.exp((x - row_max) - log_sum))
            grad = tl.where(offs == blank, grad - blank_grad, grad)
            grad = tl.where(offs == label, grad - label_grad, grad)
            tl.store(out_ptr + base + offs, grad.to(out_ptr.dtype.element_ty), mask=mask)

    # noinspection PyPep8Naming
    @triton.jit
    def _scan_preamble(offsets_ptr, frame_lens_ptr, label_lens_ptr, max_prefix, BLOCK: tl.constexpr):
        """
        The prefix lanes and lengths of the sequence this program sweeps.

        :param offsets_ptr: [B] first cell of every sequence
        :param frame_lens_ptr: [B] frames per sequence
        :param label_lens_ptr: [B] labels per sequence
        :param max_prefix: prefixes per lattice row, U_max + 1
        :param BLOCK: prefix lanes of the program, at least max_prefix
        :return: (sequence, lanes, lanes below max_prefix, lanes on a prefix of the sequence, label count,
            frame count, first cell, cells per frame, offset of the lanes in a [B, max_prefix] row)
        """
        seq = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        in_block = offs < max_prefix
        label_len = tl.load(label_lens_ptr + seq).to(tl.int64)
        frame_len = tl.load(frame_lens_ptr + seq).to(tl.int64)
        base = tl.load(offsets_ptr + seq).to(tl.int64)
        valid = in_block & (offs <= label_len)
        lanes = seq.to(tl.int64) * max_prefix + offs
        return seq, offs, in_block, valid, label_len, frame_len, base, label_len + 1, lanes

    @triton.jit
    def _log_add_exp(stay, emit, valid):
        """
        Combines the blank edge that keeps the prefix and the label edge that extends it.

        :param stay: log score over the blank edge
        :param emit: log score over the label edge
        :param valid: lanes on a prefix of the sequence
        :return: the log of the summed scores, the sentinel outside the valid lanes
        """
        # the floor keeps two impossible edges at minus infinity instead of taking inf minus inf
        top = tl.maximum(tl.maximum(stay, emit), _NEG_INF)
        return tl.where(valid, top + tl.log(tl.exp(stay - top) + tl.exp(emit - top)), _NEG_INF)

    # noinspection PyPep8Naming
    @triton.jit
    def forward_scan_kernel(
        blank_lp_ptr,
        label_lp_ptr,
        offsets_ptr,
        frame_lens_ptr,
        label_lens_ptr,
        num_cells,
        frame_stride,
        max_frames,
        max_prefix,
        alpha_ptr,
        total_ptr,
        BLOCK: tl.constexpr,
    ):
        """one program per sequence, sweeps the frames forward and writes the score of every lattice position"""
        seq, offs, in_block, valid, label_len, frame_len, base, stride, lanes = _scan_preamble(
            offsets_ptr, frame_lens_ptr, label_lens_ptr, max_prefix, BLOCK
        )
        row = alpha_ptr + lanes

        tl.store(row, tl.where(offs == 0, 0.0, _NEG_INF), mask=in_block)
        for frame in range(0, max_frames):
            tl.debug_barrier()
            here = row + frame * frame_stride
            alpha = tl.load(here, mask=in_block, other=_NEG_INF)
            alpha_prev = tl.load(here - 1, mask=in_block & (offs >= 1), other=_NEG_INF)
            cell = base + frame * stride + offs
            blank_lp = tl.load(blank_lp_ptr + tl.minimum(cell, num_cells - 1), mask=valid, other=_NEG_INF)
            label_prev = tl.load(
                label_lp_ptr + tl.maximum(tl.minimum(cell - 1, num_cells - 1), 0),
                mask=valid & (offs >= 1),
                other=_NEG_INF,
            )
            updated = _log_add_exp(alpha + blank_lp, alpha_prev + label_prev, valid)
            tl.store(here + frame_stride, tl.where(frame < frame_len, updated, alpha), mask=in_block)

        tl.debug_barrier()
        final = tl.load(row + max_frames * frame_stride, mask=in_block, other=0.0)
        tl.store(total_ptr + seq, tl.sum(tl.where(offs == label_len, final, 0.0), axis=0))

    # noinspection PyPep8Naming
    @triton.jit
    def backward_scan_kernel(
        blank_lp_ptr,
        label_lp_ptr,
        offsets_ptr,
        frame_lens_ptr,
        label_lens_ptr,
        alpha_ptr,
        weight_ptr,
        num_cells,
        frame_stride,
        max_frames,
        max_prefix,
        beta_ptr,
        blank_grad_ptr,
        label_grad_ptr,
        BLOCK: tl.constexpr,
    ):
        """one program per sequence, sweeps the frames backward and scatters both edge posteriors onto the cells"""
        seq, offs, in_block, valid, label_len, frame_len, base, stride, lanes = _scan_preamble(
            offsets_ptr, frame_lens_ptr, label_lens_ptr, max_prefix, BLOCK
        )
        weight = tl.load(weight_ptr + seq)
        start = tl.where(offs == label_len, 0.0, _NEG_INF)
        row = beta_ptr + lanes

        tl.store(row, start, mask=in_block)
        for back in range(0, max_frames):
            frame = max_frames - 1 - back
            tl.debug_barrier()
            beta = tl.load(row, mask=in_block, other=_NEG_INF)
            ahead = tl.load(row + 1, mask=in_block & (offs + 1 < max_prefix), other=_NEG_INF)
            cell = tl.minimum(base + frame * stride + offs, num_cells - 1)
            blank_lp = tl.load(blank_lp_ptr + cell, mask=valid, other=_NEG_INF)
            label_lp = tl.load(label_lp_ptr + cell, mask=valid, other=_NEG_INF)
            alpha = tl.load(alpha_ptr + lanes + frame * frame_stride, mask=in_block, other=_NEG_INF)
            inside = valid & (frame < frame_len)
            emits = inside & (offs < label_len)
            stay = tl.where(inside, alpha + blank_lp + beta, _NEG_INF)
            move = tl.where(emits, alpha + label_lp + ahead, _NEG_INF)
            # every path leaves the frame by exactly one of its edges, so the edge scores sum to the total,
            # normalizing by their own sum cancels the drift between the separately accumulated alpha and beta
            top = tl.maximum(tl.max(tl.maximum(stay, move), axis=0), _NEG_INF)
            log_mass = tl.log(tl.sum(tl.exp(stay - top) + tl.exp(move - top), axis=0))
            # the log mass stays apart from the shift, added to a large top it would round away
            live = top > _NEG_INF
            blank_post = tl.where(inside & live, tl.exp((stay - top) - log_mass) * weight, 0.0)
            label_post = tl.where(emits & live, tl.exp((move - top) - log_mass) * weight, 0.0)
            tl.store(blank_grad_ptr + cell, blank_post, mask=inside)
            tl.store(label_grad_ptr + cell, label_post, mask=inside)
            updated = _log_add_exp(blank_lp + beta, label_lp + ahead, valid)
            tl.debug_barrier()
            tl.store(row, tl.where(frame >= frame_len, start, updated), mask=in_block)
