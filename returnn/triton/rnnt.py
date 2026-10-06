"""
Triton kernels of the two sweeps of the packed RNN-T loss, shared by the backends.
The torch backend launches them in :mod:`returnn.torch.util.rnnt_triton` for :mod:`returnn.torch.util.rnnt`.
The reductions over the vocabulary are those of the monotonic loss, :mod:`returnn.triton.monotonic_rnnt`.

A label stays in its frame, so a cell depends on its left neighbour in the same frame and no sweep can take
a frame at once. The cells of one anti-diagonal (frame plus prefix constant) only depend on the anti-diagonal
before, so each sweep runs as one kernel with one program per sequence, the anti-diagonals looped inside the
kernel and the prefixes held as a block.

The scores are kept per packed cell. A dense buffer over (anti-diagonals, sequences, prefixes) would,
at the bounds of a captured step, be several times the lattice itself.

Every kernel takes its outputs last, since ``jax_triton.triton_call`` binds the outputs after the inputs.
The Triton import is guarded, so this module imports without Triton and then defines no kernels.
"""

from __future__ import annotations

try:
    import triton
    import triton.language as tl
except ImportError:  # optional dependency
    triton = tl = None


if triton is not None:
    _NEG_INF = tl.constexpr(-3.4028234663852886e38)

    # noinspection PyPep8Naming
    @triton.jit
    def _sweep_preamble(offsets_ptr, frame_lens_ptr, label_lens_ptr, max_prefix, BLOCK: tl.constexpr):
        """
        The prefix lanes and lengths of the sequence this program sweeps.

        :param offsets_ptr: [B] first cell of every sequence
        :param frame_lens_ptr: [B] frames per sequence
        :param label_lens_ptr: [B] labels per sequence
        :param max_prefix: prefixes per lattice row, U_max + 1
        :param BLOCK: prefix lanes of the program, at least max_prefix
        :return: (sequence, lanes, lanes on a prefix of the sequence, label count, frame count, first cell,
            cells per frame)
        """
        seq = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        label_len = tl.load(label_lens_ptr + seq).to(tl.int64)
        frame_len = tl.load(frame_lens_ptr + seq).to(tl.int64)
        base = tl.load(offsets_ptr + seq).to(tl.int64)
        in_lattice = (offs < max_prefix) & (offs <= label_len)
        return seq, offs, in_lattice, label_len, frame_len, base, label_len + 1

    # noinspection PyPep8Naming
    @triton.jit
    def forward_scan_kernel(
        blank_lp_ptr,
        label_lp_ptr,
        offsets_ptr,
        frame_lens_ptr,
        label_lens_ptr,
        num_cells,
        max_diagonals,
        max_prefix,
        alpha_ptr,
        total_ptr,
        BLOCK: tl.constexpr,
    ):
        """one program per sequence, sweeps the anti-diagonals forward and writes the score of every cell"""
        seq, offs, in_lattice, label_len, frame_len, base, stride = _sweep_preamble(
            offsets_ptr, frame_lens_ptr, label_lens_ptr, max_prefix, BLOCK
        )

        for diagonal in range(0, max_diagonals):
            tl.debug_barrier()
            frame = diagonal - offs
            inside = in_lattice & (frame >= 0) & (frame < frame_len)
            cell = tl.maximum(tl.minimum(base + frame * stride + offs, num_cells - 1), 0)
            # the cell above, which a blank leaves to this one
            from_above = inside & (frame >= 1)
            above = tl.maximum(cell - stride, 0)
            stay = tl.load(alpha_ptr + above, mask=from_above, other=_NEG_INF) + tl.load(
                blank_lp_ptr + above, mask=from_above, other=_NEG_INF
            )
            # the left neighbour in the same frame, which a label leaves to this one
            from_left = inside & (offs >= 1)
            left = tl.maximum(cell - 1, 0)
            emit = tl.load(alpha_ptr + left, mask=from_left, other=_NEG_INF) + tl.load(
                label_lp_ptr + left, mask=from_left, other=_NEG_INF
            )
            # the floor keeps two impossible edges at minus infinity instead of taking inf minus inf
            top = tl.maximum(tl.maximum(stay, emit), _NEG_INF)
            reached = top + tl.log(tl.exp(stay - top) + tl.exp(emit - top))
            # the first cell has no edge into it, every path starts there
            reached = tl.where((diagonal == 0) & (offs == 0), 0.0, reached)
            tl.store(alpha_ptr + cell, reached, mask=inside)

        tl.debug_barrier()
        # every path ends by the blank of the last cell, a slot without frames has no cell at all
        last = tl.maximum(tl.minimum(base + (frame_len - 1) * stride + label_len, num_cells - 1), 0)
        ends = tl.load(alpha_ptr + last) + tl.load(blank_lp_ptr + last)
        tl.store(total_ptr + seq, tl.where(frame_len > 0, ends, _NEG_INF))

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
        max_diagonals,
        max_prefix,
        beta_ptr,
        blank_grad_ptr,
        label_grad_ptr,
        BLOCK: tl.constexpr,
    ):
        """one program per sequence, sweeps the anti-diagonals backward and writes both edge posteriors of the cells"""
        seq, offs, in_lattice, label_len, frame_len, base, stride = _sweep_preamble(
            offsets_ptr, frame_lens_ptr, label_lens_ptr, max_prefix, BLOCK
        )
        weight = tl.load(weight_ptr + seq)

        for back in range(0, max_diagonals):
            diagonal = max_diagonals - 1 - back
            tl.debug_barrier()
            frame = diagonal - offs
            inside = in_lattice & (frame >= 0) & (frame < frame_len)
            cell = tl.maximum(tl.minimum(base + frame * stride + offs, num_cells - 1), 0)
            blank_lp = tl.load(blank_lp_ptr + cell, mask=inside, other=_NEG_INF)
            label_lp = tl.load(label_lp_ptr + cell, mask=inside, other=_NEG_INF)
            alpha = tl.load(alpha_ptr + cell, mask=inside, other=_NEG_INF)
            # a blank leaves to the cell below, and from the last cell of the sequence out of the lattice
            to_below = inside & (frame + 1 < frame_len)
            below = tl.minimum(cell + stride, num_cells - 1)
            ahead = tl.load(beta_ptr + below, mask=to_below, other=_NEG_INF)
            ends = inside & (frame + 1 == frame_len) & (offs == label_len)
            ahead = tl.where(ends, 0.0, ahead)
            # a label leaves to the right neighbour in the same frame
            to_right = inside & (offs < label_len)
            right = tl.minimum(cell + 1, num_cells - 1)
            beside = tl.load(beta_ptr + right, mask=to_right, other=_NEG_INF)
            stay = blank_lp + ahead
            emit = label_lp + beside
            top = tl.maximum(tl.maximum(stay, emit), _NEG_INF)
            remaining = top + tl.log(tl.exp(stay - top) + tl.exp(emit - top))
            tl.store(beta_ptr + cell, remaining, mask=inside)
            # every path crosses an anti-diagonal exactly once, so the edge scores of its cells sum to the total,
            # and normalizing by their own sum cancels the drift between the separately accumulated alpha and beta
            blank_edge = tl.where(to_below | ends, alpha + stay, _NEG_INF)
            label_edge = tl.where(to_right, alpha + emit, _NEG_INF)
            edge_top = tl.maximum(tl.max(tl.maximum(blank_edge, label_edge), axis=0), _NEG_INF)
            # the log mass stays apart from the shift, added to a large top it would round away,
            # and a sequence without any alignment has no live edge on any anti-diagonal and gets no gradient
            log_mass = tl.log(tl.sum(tl.exp(blank_edge - edge_top) + tl.exp(label_edge - edge_top), axis=0))
            live = edge_top > _NEG_INF
            blank_post = tl.where(inside & live, tl.exp((blank_edge - edge_top) - log_mass) * weight, 0.0)
            label_post = tl.where(to_right & live, tl.exp((label_edge - edge_top) - log_mass) * weight, 0.0)
            tl.store(blank_grad_ptr + cell, blank_post, mask=inside)
            tl.store(label_grad_ptr + cell, label_post, mask=inside)
