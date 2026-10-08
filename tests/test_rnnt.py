"""
Tests for the packed RNN-T loss, see :mod:`returnn.torch.util.rnnt`.
"""

from __future__ import annotations

import itertools
import sys
import unittest

import torch

import _setup_test_env  # noqa
from returnn.torch.util.rnnt import rnnt_loss


def _brute_force(logits: torch.Tensor, labels, blank: int) -> float:
    """
    Sums over every alignment explicitly, as the definition of the loss.

    An alignment gives every label the frame it is emitted in, the frames never going back,
    and leaves every frame by a blank behind the labels emitted in it.

    :param logits: [T, U + 1, V] of one sequence
    :param labels: the U labels
    :param blank: blank index
    :return: the negative log likelihood
    """
    num_frames, num_prefix, _ = logits.shape
    num_labels = num_prefix - 1
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    totals = []
    for emits in itertools.combinations_with_replacement(range(num_frames), num_labels):
        u = 0
        total = 0.0
        for t in range(num_frames):
            while u < num_labels and emits[u] == t:
                total += float(log_probs[t, u, labels[u]])
                u += 1
            total += float(log_probs[t, u, blank])
        totals.append(total)
    if not totals:
        return 0.0
    return -float(torch.logsumexp(torch.tensor(totals, dtype=torch.float64), dim=0))


def _pack(per_seq_logits):
    """
    :param per_seq_logits: one [T, U + 1, V] tensor per sequence
    :return: the packed [cells, V] tensor the loss takes, frame index outer
    """
    return torch.cat([x.reshape(-1, x.shape[-1]) for x in per_seq_logits], dim=0)


def _batch(cases, vocab: int, *, extra_label_columns: int = 0):
    """
    :param cases: (frames, labels) per sequence
    :param vocab: vocabulary size with blank at index 0
    :param extra_label_columns: columns the padded labels are wider than the longest sequence
    :return: (logits per sequence, labels per sequence, padded labels, frame lens, label lens)
    """
    per_seq, labels = [], []
    for num_frames, num_labels in cases:
        per_seq.append(torch.randn(num_frames, num_labels + 1, vocab))
        seq_labels = torch.randint(1, vocab, (num_labels,))
        if num_labels >= 2:
            seq_labels[1] = seq_labels[0]
        labels.append(seq_labels)
    width = max(u for _t, u in cases) + extra_label_columns
    labels_padded = torch.zeros(len(cases), width, dtype=torch.int32)
    for i, seq_labels in enumerate(labels):
        labels_padded[i, : len(seq_labels)] = seq_labels
    frame_lens = torch.tensor([t for t, _u in cases], dtype=torch.int32)
    label_lens = torch.tensor([u for _t, u in cases], dtype=torch.int32)
    return per_seq, labels, labels_padded, frame_lens, label_lens


def test_rnnt_matches_the_sum_over_alignments():
    """
    The last sequences have more labels than frames, which the monotonic loss cannot align and this one can,
    since several labels may share a frame.
    """
    torch.manual_seed(42)
    vocab, blank = 5, 0
    cases = [(4, 2), (5, 0), (3, 3), (6, 2), (2, 4), (1, 3)]
    per_seq, labels, labels_padded, frame_lens, label_lens = _batch(cases, vocab, extra_label_columns=3)
    want = [_brute_force(logits, seq_labels.tolist(), blank) for logits, seq_labels in zip(per_seq, labels)]

    got = rnnt_loss(_pack(per_seq), labels_padded, frame_lens, label_lens, blank=blank, max_frames=6)
    torch.testing.assert_close(got.double(), torch.tensor(want, dtype=torch.float64), rtol=0, atol=1e-5)


def test_rnnt_runs_over_a_buffer_and_a_recursion_above_the_batch():
    """
    A captured step hands over a buffer of the declared capacity and the declared frames, both above what
    the batch holds, and neither changes the loss or the gradient of the batch, whatever the rows past it hold.
    """
    torch.manual_seed(9)
    vocab, blank = 5, 0
    cases = [(4, 2), (3, 3), (2, 0)]
    per_seq, labels, labels_padded, frame_lens, label_lens = _batch(cases, vocab, extra_label_columns=2)
    want = [_brute_force(logits, seq_labels.tolist(), blank) for logits, seq_labels in zip(per_seq, labels)]
    exact = _pack(per_seq)

    packed = torch.cat([exact, torch.full((11, vocab), float("nan"))])
    got = rnnt_loss(packed, labels_padded, frame_lens, label_lens, blank=blank, max_frames=9)
    torch.testing.assert_close(got.double(), torch.tensor(want, dtype=torch.float64), rtol=0, atol=1e-5)

    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        args = (labels_padded.to(device), frame_lens.to(device), label_lens.to(device))
        results = []
        for buffer, max_frames in ((exact, 4), (packed, 9)):
            x = buffer.to(device).clone().requires_grad_()
            loss = rnnt_loss(x, *args, blank=blank, max_frames=max_frames)
            loss.sum().backward()
            results.append((loss.detach().cpu(), x.grad.cpu()))
        torch.testing.assert_close(results[1][0], results[0][0])
        torch.testing.assert_close(results[1][1][: len(exact)], results[0][1])
        assert not results[1][1][len(exact) :].any(), results[1][1][len(exact) :]


def test_rnnt_gradient_matches_finite_differences():
    torch.manual_seed(7)
    vocab, blank = 4, 0
    num_frames, num_labels = 3, 2
    logits = torch.randn(num_frames, num_labels + 1, vocab, dtype=torch.float64, requires_grad=True)
    args = (
        torch.tensor([[1, 3]], dtype=torch.int32),
        torch.tensor([num_frames], dtype=torch.int32),
        torch.tensor([num_labels], dtype=torch.int32),
    )
    loss = rnnt_loss(logits.reshape(-1, vocab), *args, blank=blank, max_frames=num_frames)
    want = _brute_force(logits.detach(), [1, 3], blank)
    torch.testing.assert_close(loss, torch.tensor([want], dtype=torch.float64), rtol=0, atol=1e-9)
    torch.autograd.gradcheck(
        lambda x: rnnt_loss(x.reshape(-1, vocab), *args, blank=blank, max_frames=num_frames),
        (logits,),
        eps=1e-6,
        atol=1e-6,
    )


def test_rnnt_handles_degenerate_batches():
    torch.manual_seed(3)
    vocab, blank = 6, 0

    frame_lens = torch.tensor([0, 0], dtype=torch.int32)
    label_lens = torch.tensor([5, 7], dtype=torch.int32)
    got = rnnt_loss(
        torch.randn(0, vocab),
        torch.randint(1, vocab, (2, 7), dtype=torch.int32),
        frame_lens,
        label_lens,
        blank=blank,
        max_frames=3,
    )
    torch.testing.assert_close(got, torch.zeros(2))

    frame_lens = torch.tensor([4, 3], dtype=torch.int32)
    label_lens = torch.tensor([0, 0], dtype=torch.int32)
    logits = torch.randn(7, vocab)
    got = rnnt_loss(logits, torch.zeros((2, 0), dtype=torch.int32), frame_lens, label_lens, blank=blank, max_frames=4)
    blank_lp = torch.log_softmax(logits.double(), dim=-1)[:, blank]
    want = torch.tensor([-float(blank_lp[:4].sum()), -float(blank_lp[4:].sum())])
    torch.testing.assert_close(got.double(), want.double(), rtol=0, atol=1e-5)

    # a slot without frames next to a sequence, zero loss and a finite gradient
    frame_lens = torch.tensor([2, 0], dtype=torch.int32)
    label_lens = torch.tensor([1, 3], dtype=torch.int32)
    logits = torch.randn(4, vocab, requires_grad=True)
    labels = torch.randint(1, vocab, (2, 3), dtype=torch.int32)
    got = rnnt_loss(logits, labels, frame_lens, label_lens, blank=blank, max_frames=2)
    want = _brute_force(logits.detach().reshape(2, 2, vocab), labels[0, :1].tolist(), blank)
    torch.testing.assert_close(got.double(), torch.tensor([want, 0.0], dtype=torch.float64), rtol=0, atol=1e-5)
    got.sum().backward()
    assert torch.isfinite(logits.grad).all(), logits.grad


def test_rnnt_survives_impossible_edges():
    """
    Two edges have no probability, the label at frame 1 from the empty prefix and the blank at frame 0 after the
    label, so one alignment remains and the cell of frame 1 and prefix 1 lies on none. The gradient stays finite
    where the loss is, and a row of minus infinity at that dead cell changes nothing.
    """
    logits = torch.zeros(3, 2, 3)
    logits[1, 0, 1] = float("-inf")
    logits[0, 1, 0] = float("-inf")
    dead = logits.clone()
    dead[1, 1] = float("-inf")
    args = (
        torch.tensor([[1]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        results = []
        for x in (logits, dead):
            x = x.reshape(-1, 3).to(device).requires_grad_()
            loss = rnnt_loss(x, *(a.to(device) for a in args), blank=0, max_frames=3)
            loss.sum().backward()
            results.append((loss.detach().cpu(), x.grad.cpu()))
        torch.testing.assert_close(results[0][0], torch.tensor([54.0]).log())
        assert torch.isfinite(results[0][1]).all(), results[0][1]
        torch.testing.assert_close(results[1], results[0])


def test_rnnt_gives_no_gradient_without_a_possible_alignment():
    """
    The label of the first sequence has no probability in either of its frames, so none of its alignments has any
    and its loss is infinite. Its cells get no gradient, and the second sequence keeps the loss and the gradient
    it has alone.
    """
    torch.manual_seed(8)
    vocab, blank = 3, 0
    impossible = torch.randn(2, 2, vocab)
    impossible[:, 0, 1] = float("-inf")
    per_seq = [impossible, torch.randn(3, 2, vocab)]
    labels = torch.tensor([[1], [2]], dtype=torch.int32)
    frame_lens = torch.tensor([2, 3], dtype=torch.int32)
    label_lens = torch.tensor([1, 1], dtype=torch.int32)
    for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        results = []
        for seqs in (slice(None), slice(1, None)):
            x = _pack(per_seq[seqs]).to(device).requires_grad_()
            args = (labels[seqs].to(device), frame_lens[seqs].to(device), label_lens[seqs].to(device))
            loss = rnnt_loss(x, *args, blank=blank, max_frames=3)
            loss.sum().backward()
            results.append((loss.detach().cpu(), x.grad.cpu()))
        (loss, grad), (alone_loss, alone_grad) = results
        others = grad.shape[0] - alone_grad.shape[0]
        assert torch.isposinf(loss[0]), loss
        assert not grad[:others].any(), grad[:others]
        torch.testing.assert_close(loss[1:], alone_loss)
        torch.testing.assert_close(grad[others:], alone_grad)


def test_rnnt_refuses_a_recursion_shorter_than_a_sequence():
    """
    A recursion cut short never reaches the last cell of the sequence and its score is not the loss,
    so the frames a caller declares have to cover every sequence.
    """
    torch.manual_seed(1)
    logits = torch.randn(12, 5)
    labels = torch.tensor([[1, 2]], dtype=torch.int32)
    frame_lens, label_lens = torch.tensor([4], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    fine = rnnt_loss(logits, labels, frame_lens, label_lens, blank=0, max_frames=4)
    assert torch.isfinite(fine).all(), fine

    try:
        rnnt_loss(logits, labels, frame_lens, label_lens, blank=0, max_frames=3)
    except AssertionError as exc:
        assert "frames" in str(exc), exc
    else:
        raise AssertionError("a recursion shorter than the sequence was accepted")


def test_rnnt_refuses_labels_outside_the_vocabulary():
    """
    The cell kernels index the row by blank and by label unchecked, so a stray id reads and writes out of
    bounds on cuda, and both ids are checked before any kernel runs.
    """
    torch.manual_seed(1)
    vocab = 5
    logits = torch.randn(12, vocab)
    frame_lens, label_lens = torch.tensor([4], dtype=torch.int32), torch.tensor([2], dtype=torch.int32)
    for labels, blank, error in (
        ([[1, 2]], vocab, ValueError),
        ([[1, vocab]], 0, AssertionError),
        ([[-1, 2]], 0, AssertionError),
    ):
        try:
            rnnt_loss(
                logits, torch.tensor(labels, dtype=torch.int32), frame_lens, label_lens, blank=blank, max_frames=4
            )
        except error as exc:
            assert "vocabulary" in str(exc), exc
        else:
            raise AssertionError(f"labels {labels} with blank {blank} were accepted")


def test_rnnt_reads_strided_lengths():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    torch.manual_seed(7)
    logits = torch.randn(10, 3, device="cuda")
    labels = torch.tensor([[1], [2]], dtype=torch.int32, device="cuda")
    frame_lens = torch.tensor([2, 7, 3, 8], dtype=torch.int32, device="cuda")[::2]
    label_lens = torch.tensor([1, 0, 1, 0], dtype=torch.int32, device="cuda")[::2]
    results = []
    for lens in ((frame_lens, label_lens), (frame_lens.contiguous(), label_lens.contiguous())):
        x = logits.clone().requires_grad_()
        loss = rnnt_loss(x, labels, *lens, blank=0, max_frames=3)
        loss.sum().backward()
        results.append((loss.detach(), x.grad))
    torch.testing.assert_close(results[0], results[1])


def test_rnnt_gradient_of_a_long_sequence():
    """
    Without labels there is one alignment, so every blank posterior is one and the gradient is the softmax minus
    the blank one hot, which the sweeps only give when the drift between alpha and beta over two thousand
    anti-diagonals cancels.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    frames, vocab = 2048, 2048
    x = torch.zeros(frames, vocab, device="cuda", requires_grad=True)
    no_labels = torch.zeros((1, 0), dtype=torch.int32, device="cuda")
    lens = (
        torch.tensor([frames], dtype=torch.int32, device="cuda"),
        torch.tensor([0], dtype=torch.int32, device="cuda"),
    )
    loss = rnnt_loss(x, no_labels, *lens, blank=0, max_frames=frames)
    loss.sum().backward()
    want = torch.full_like(x, 1 / vocab)
    want[:, 0] -= 1
    torch.testing.assert_close(x.grad, want, rtol=1e-4, atol=1e-5)


def test_rf_rnnt_loss_scores_the_lattice_the_joint_builds():
    """
    The rows of the lattice need not be single frames. Here the joint gets the index of the row of every cell
    and looks its contribution up by it, as a joint attending over a chunk of frames would find its chunk.
    """
    import returnn.frontend as rf
    from returnn.tensor import Dim, Tensor

    rf.select_backend_torch()
    torch.manual_seed(13)
    vocab, blank = 5, 0
    cases = [(4, 2), (3, 3), (2, 0), (1, 4)]
    num_seqs, max_rows, max_labels = len(cases), max(t for t, _u in cases), max(u for _t, u in cases)
    row_table = torch.randn(max_rows, vocab)
    pred_raw = torch.randn(num_seqs, max_labels + 1, vocab)
    labels_raw = torch.randint(1, vocab, (num_seqs, max_labels), dtype=torch.int32)
    want = [
        _brute_force(row_table[:t].unsqueeze(1) + pred_raw[i, : u + 1].unsqueeze(0), labels_raw[i, :u].tolist(), blank)
        for i, (t, u) in enumerate(cases)
    ]

    batch = Dim(num_seqs, name="batch")
    rows_time = Dim(
        Tensor("row_lens", dims=[batch], dtype="int32", raw_tensor=torch.tensor([t for t, _u in cases]).int()),
        name="rows",
    )
    labels_time = Dim(
        Tensor("label_lens", dims=[batch], dtype="int32", raw_tensor=torch.tensor([u for _t, u in cases]).int()),
        name="labels",
    )
    prefix_dim = labels_time + 1
    row_index_dim, vocab_dim = Dim(max_rows, name="row_index"), Dim(vocab, name="vocab")
    rows = Tensor(
        "rows",
        dims=[batch, rows_time],
        dtype="int32",
        raw_tensor=torch.arange(max_rows, dtype=torch.int32).expand(num_seqs, -1).contiguous(),
        sparse_dim=row_index_dim,
    )
    pred = Tensor(
        "pred", dims=[batch, prefix_dim, vocab_dim], dtype="float32", raw_tensor=pred_raw, feature_dim=vocab_dim
    )
    labels = Tensor("labels", dims=[batch, labels_time], dtype="int32", raw_tensor=labels_raw, sparse_dim=vocab_dim)
    table = Tensor("table", dims=[row_index_dim, vocab_dim], dtype="float32", raw_tensor=row_table)

    def _joint(row_cells: Tensor, pred_cells: Tensor) -> Tensor:
        return rf.gather(table, indices=row_cells, axis=row_index_dim) + pred_cells

    got = rf.rnnt_loss(
        enc=rows,
        pred=pred,
        enc_spatial_dim=rows_time,
        prefix_dim=prefix_dim,
        labels=labels,
        labels_spatial_dim=labels_time,
        joint=_joint,
        blank_index=blank,
    )
    assert got.dims == (batch,), got
    torch.testing.assert_close(got.raw_tensor.double(), torch.tensor(want, dtype=torch.float64), rtol=0, atol=1e-5)


def test_rnnt_on_cuda_matches_the_reference():
    """
    What the kernels see under capture, a buffer and a recursion above the batch's own cells and frames,
    a sequence with more labels than frames, slots without frames and a batch without any label, each
    against the cpu path.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    torch.manual_seed(3)
    vocab, blank, slack = 37, 0, 13
    for cases in ([(9, 4), (12, 0), (7, 7), (5, 2), (3, 5), (1, 6), (0, 3)], [(4, 0), (6, 0)], [(1, 40), (30, 1)]):
        per_seq, _labels, labels_padded, frame_lens, label_lens = _batch(cases, vocab, extra_label_columns=2)
        packed = torch.cat([_pack(per_seq), torch.randn(slack, vocab)])
        grad_out = torch.randn(len(cases))

        results = []
        for device in ("cpu", "cuda"):
            logits = packed.to(device).clone().requires_grad_()
            loss = rnnt_loss(
                logits,
                labels_padded.to(device),
                frame_lens.to(device),
                label_lens.to(device),
                blank=blank,
                max_frames=int(frame_lens.max()) + 2,
            )
            loss.backward(grad_out.to(device))
            results.append((loss.detach().cpu(), logits.grad.detach().cpu()))
        torch.testing.assert_close(results[1][0], results[0][0], rtol=1e-5, atol=1e-5, msg=str(cases))
        torch.testing.assert_close(results[1][1], results[0][1], rtol=1e-4, atol=1e-5, msg=str(cases))


def test_rnnt_scans_on_cuda_give_the_posteriors_of_every_edge():
    """
    The edge posteriors of a frame's blanks sum to one, since every alignment leaves every frame exactly once,
    and those of all label edges sum to the number of labels.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    from returnn.torch.util.rnnt_triton import backward_scan, forward_scan

    torch.manual_seed(4)
    frame_lens = torch.tensor([3, 4, 0, 1], dtype=torch.int32, device="cuda")
    label_lens = torch.tensor([5, 2, 3, 0], dtype=torch.int32, device="cuda")
    cells = frame_lens.long() * (label_lens.long() + 1)
    offsets = torch.cumsum(cells, dim=0) - cells
    num_cells = int(cells.sum()) + 5
    log_probs = torch.log_softmax(torch.randn(num_cells, 3, device="cuda"), dim=-1)
    blank_lp, label_lp = log_probs[:, 0].contiguous(), log_probs[:, 1].contiguous()

    _total, alpha = forward_scan(blank_lp, label_lp, offsets, frame_lens, label_lens, 6, 8)
    blank_post, label_post = backward_scan(
        blank_lp, label_lp, offsets, frame_lens, label_lens, alpha, torch.ones(4, device="cuda"), 6, 8
    )
    for seq in range(4):
        first, count = int(offsets[seq]), int(cells[seq])
        frames, prefixes = int(frame_lens[seq]), int(label_lens[seq]) + 1
        per_frame = blank_post[first : first + count].reshape(frames, prefixes).sum(dim=1)
        torch.testing.assert_close(per_frame, torch.ones(frames, device="cuda"), rtol=1e-4, atol=1e-4)
        labels = label_post[first : first + count].sum()
        want = float(prefixes - 1) if frames else 0.0
        torch.testing.assert_close(labels, torch.tensor(want, device="cuda"), rtol=1e-4, atol=1e-4)
    assert not blank_post[int(cells.sum()) :].any() and not label_post[int(cells.sum()) :].any()


def test_rnnt_traces_under_aot():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    from functorch.compile import aot_function, nop

    torch.manual_seed(5)
    vocab, blank, max_frames = 32, 0, 11
    # unequal lengths, a frame bound above the longest sequence and rows past the cell sum, as a capacity leaves them
    frame_lens = torch.tensor([8, 5, 9], dtype=torch.int32)
    label_lens = torch.tensor([3, 7, 0], dtype=torch.int32)
    num_cells = int((frame_lens * (label_lens + 1)).sum())
    labels = torch.randint(1, vocab, (3, 7), dtype=torch.int32)
    packed = torch.randn(num_cells + 7, vocab)
    grad_out = torch.randn(3)

    def loss_of(x, lab, flens, ulens):
        return rnnt_loss(x, lab, flens, ulens, blank=blank, max_frames=max_frames)

    # the traced program against the eager one and both against the reference on cpu
    results = []
    for device, func in (
        ("cpu", loss_of),
        ("cuda", loss_of),
        ("cuda", aot_function(loss_of, fw_compiler=nop, bw_compiler=nop)),
    ):
        x = packed.to(device).detach().clone().requires_grad_()
        loss = func(x, labels.to(device), frame_lens.to(device), label_lens.to(device))
        loss.backward(grad_out.to(device))
        results.append((loss.detach().cpu(), x.grad.cpu()))
    for loss, grad in results[1:]:
        torch.testing.assert_close(loss, results[0][0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(grad, results[0][1], rtol=1e-4, atol=1e-6)
    # the rows past the cell sum belong to no sequence
    assert not results[2][1][num_cells:].any()


def test_rnnt_keeps_the_input_dtype():
    """
    The kernels read every logit anyway, so they convert on load and the gradient comes back in the dtype of
    the logits, without a float32 copy of the lattice in between.
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no cuda")
    torch.manual_seed(4)
    vocab, blank, batch, frames, num_labels = 2048, 0, 4, 32, 7
    frame_lens = torch.full((batch,), frames, dtype=torch.int32, device="cuda")
    label_lens = torch.full((batch,), num_labels, dtype=torch.int32, device="cuda")
    labels = torch.randint(1, vocab, (batch, num_labels), dtype=torch.int32, device="cuda")
    packed = torch.randn(batch * frames * (num_labels + 1), vocab, device="cuda").bfloat16()

    def run(dtype):
        x = packed.to(dtype, copy=True).requires_grad_()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        loss = rnnt_loss(x, labels, frame_lens, label_lens, blank=blank, max_frames=frames)
        loss.sum().backward()
        return loss.detach(), x.grad, torch.cuda.max_memory_allocated() - before

    want_loss, want_grad, _ = run(torch.float32)
    loss, grad, peak = run(torch.bfloat16)
    assert grad.dtype == torch.bfloat16, grad.dtype
    torch.testing.assert_close(loss, want_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(grad, want_grad.bfloat16())
    # the bf16 gradient is the one [cells, vocab] allocation, no float32 copy of the logits or of the gradient
    assert peak < 1.5 * packed.numel() * packed.element_size(), peak


if __name__ == "__main__":
    if len(sys.argv) > 1:
        globals()[sys.argv[1]]()
    else:
        for name, func in sorted(globals().items()):
            if name.startswith("test_"):
                print(f"-- {name}")
                try:
                    func()
                except unittest.SkipTest as exc:
                    print(f"   (skipped, {exc})")
        print("all passed")
