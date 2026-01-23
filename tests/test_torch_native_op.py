"""
Test torch native ops.
"""

import _setup_test_env  # noqa
import os
from unittest import SkipTest
from typing import Optional, Union, Sequence, List
import numpy
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.ndimage
import torch
from returnn.torch.util.native_op import (
    make_fast_baum_welch_op,
    fast_baum_welch,
    fast_viterbi,
    get_ctc_fsa_fast_bw,
    edit_distance,
    optimal_completion_edit_distance,
    optimal_completion_edit_distance_per_successor,
    optimal_completion_edit_distance_per_successor_via_next_edit_distance,
    edit_distance_via_next_edit_distance_row,
)
from returnn.util.fsa import FastBwFsaShared, get_ctc_fsa_fast_bw as get_ctc_fsa_fast_bw_np
from fsa_utils import py_baum_welch, py_viterbi
from numpy_ref_edit_distance import edit_distance_ref_np, edit_distance_ref_np_b1


os.environ["RETURNN_NATIVE_CODE_COMPILER_VERBOSE"] = "1"


def test_compile():
    # In case CUDA is available, it will use nvcc to compile, and compiles the op with both CPU and CUDA support.
    make_fast_baum_welch_op(compiler_opts=dict(verbose=True))


def test_compile_cpu_only():
    # This forces CPU-only compilation, using c++, even if CUDA is available.
    make_fast_baum_welch_op(compiler_opts=dict(verbose=True), with_cuda=False)


def test_compile_with_cuda():
    if not torch.cuda.is_available():
        raise SkipTest("CUDA not available")
    # Explicitly compile with CUDA support.
    make_fast_baum_welch_op(compiler_opts=dict(verbose=True), with_cuda=True)


def test_FastBaumWelch():
    print("Make op...")
    op = make_fast_baum_welch_op(
        compiler_opts=dict(verbose=True)
    )  # will be cached, used inside :func:`fast_baum_welch`
    print("Op:", op)
    n_batch = 3
    seq_len = 5
    n_classes = 10

    fsa = FastBwFsaShared()
    fsa.add_inf_loop(state_idx=0, num_emission_labels=n_classes)
    fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
    edges = torch.tensor(fast_bw_fsa.edges, dtype=torch.int32)
    weights = torch.tensor(fast_bw_fsa.weights, dtype=torch.float32)
    start_end_states = torch.tensor(fast_bw_fsa.start_end_states, dtype=torch.int32)
    am_scores_np = numpy.random.RandomState(42).normal(size=(seq_len, n_batch, n_classes)).astype("float32")
    am_scores = torch.tensor(am_scores_np, dtype=torch.float32)  # in -log space
    float_idx_np = numpy.ones((seq_len, n_batch), dtype="float32")
    float_idx = torch.ones((seq_len, n_batch), dtype=torch.float32)
    print("Construct call...")
    fwdbwd, obs_scores = fast_baum_welch(
        am_scores=am_scores, seq_mask=float_idx, edges=edges, weights=weights, start_end_states=start_end_states
    )
    print("Done.")
    print("Eval:")
    fwdbwd_np, score = fwdbwd.numpy(), obs_scores.numpy()
    print("score:", score)
    print("Baum-Welch soft alignment:")
    print(repr(fwdbwd_np))
    fwdbwd_np2, score2 = py_baum_welch(
        am_scores=am_scores_np,
        float_idx=float_idx_np,
        edges=fast_bw_fsa.edges,
        weights=fast_bw_fsa.weights,
        start_end_states=fast_bw_fsa.start_end_states,
    )
    print("ref score:", score2)
    print("ref Baum-Welch soft alignment:")
    print(repr(fwdbwd_np2))
    numpy.testing.assert_allclose(score, score2, rtol=1e-5)
    numpy.testing.assert_allclose(fwdbwd_np, fwdbwd_np2, rtol=1e-5)


def test_fast_bw_uniform():
    print("Make op...")
    op = make_fast_baum_welch_op(
        compiler_opts=dict(verbose=True)
    )  # will be cached, used inside :func:`fast_baum_welch`
    # args: (am_scores, edges, weights, start_end_states, float_idx, state_buffer)
    print("Op:", op)
    n_batch = 3
    seq_len = 7
    n_classes = 5

    fsa = FastBwFsaShared()
    for i in range(n_classes):
        fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
        fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
    assert n_classes <= seq_len
    fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
    edges = torch.tensor(fast_bw_fsa.edges, dtype=torch.int32)
    weights = torch.tensor(fast_bw_fsa.weights, dtype=torch.float32)
    start_end_states = torch.tensor(fast_bw_fsa.start_end_states, dtype=torch.int32)
    am_scores = numpy.ones((seq_len, n_batch, n_classes), dtype="float32") * numpy.float32(1.0 / n_classes)
    am_scores = -numpy.log(am_scores)  # in -log space
    am_scores = torch.tensor(am_scores, dtype=torch.float32)
    float_idx = torch.ones((seq_len, n_batch), dtype=torch.float32)
    print("Construct call...")
    fwdbwd, obs_scores = fast_baum_welch(
        am_scores=am_scores, seq_mask=float_idx, edges=edges, weights=weights, start_end_states=start_end_states
    )
    print("Done.")
    print("Eval:")
    fwdbwd, score = fwdbwd.numpy(), obs_scores.numpy()
    print("score:")
    print(repr(score))
    assert score.shape == (seq_len, n_batch)
    bw = numpy.exp(-fwdbwd)
    print("Baum-Welch soft alignment:")
    print(repr(bw))
    assert bw.shape == (seq_len, n_batch, n_classes)

    if seq_len == n_classes:
        print("Extra check identity...")
        for i in range(n_batch):
            assert_almost_equal(numpy.identity(n_classes), bw[:, i])
    if seq_len == 7 and n_classes == 5:
        print("Extra check ref_align (7,5)...")
        assert_allclose(score, 8.55801582, rtol=1e-5)  # should be the same everywhere
        ref_align = numpy.array(
            [
                [[1.0, 0.0, 0.0, 0.0, 0.0]],
                [[0.33333316, 0.66666663, 0.0, 0.0, 0.0]],
                [[0.06666669, 0.53333354, 0.40000018, 0.0, 0.0]],
                [[0.0, 0.20000014, 0.60000014, 0.19999999, 0.0]],
                [[0.0, 0.0, 0.39999962, 0.53333312, 0.06666663]],
                [[0.0, 0.0, 0.0, 0.66666633, 0.33333316]],
                [[0.0, 0.0, 0.0, 0.0, 0.99999982]],
            ],
            dtype=numpy.float32,
        )
        assert ref_align.shape == (seq_len, 1, n_classes)
        ref_align = numpy.tile(ref_align, (1, n_batch, 1))
        assert ref_align.shape == bw.shape
        # print("Reference alignment:")
        # print(repr(ref_align))
        print("mean square diff:", numpy.mean(numpy.square(ref_align - bw)))
        print("max square diff:", numpy.max(numpy.square(ref_align - bw)))
        assert_allclose(ref_align, bw, rtol=1e-5)
    print("Done.")


def _log_softmax(x, axis=-1):
    assert isinstance(x, numpy.ndarray)
    xdev = x - x.max(axis=axis, keepdims=True)
    lsm = xdev - numpy.log(numpy.sum(numpy.exp(xdev), axis=axis, keepdims=True))
    return lsm


def check_ctc_fsa(targets, target_seq_lens, n_classes, with_native_fsa=False, label_loop=True):
    """
    :param numpy.ndarray targets:
    :param numpy.ndarray target_seq_lens:
    :param int n_classes:
    :param bool with_native_fsa:
    :param bool label_loop: ctc_merge_repeated in tf.nn.ctc_loss
    :return: nothing, just checks
    """
    n_batch, n_target_time = targets.shape
    assert n_batch == len(target_seq_lens) and n_target_time == max(target_seq_lens)
    n_time = n_target_time * 3
    # am_scores are logits, unnormalized, i.e. the values before softmax.
    am_scores = numpy.random.RandomState(42).normal(size=(n_time, n_batch, n_classes)).astype("float32")
    # am_scores = numpy.zeros((n_time, n_batch, n_classes), dtype="float32")
    int_idx = numpy.zeros((n_time, n_batch), dtype="int32")
    seq_lens = numpy.array(
        [
            n_time,
            max(n_time - 4, (target_seq_lens[1] if (len(target_seq_lens) >= 2) else 0) + 1, 1),
            max(n_time - 5, 1),
            max(n_time - 5, 1),
        ],
        dtype="int32",
    )[:n_batch]
    for t in range(n_time):
        int_idx[t] = t < seq_lens
    float_idx = int_idx.astype("float32")
    blank_idx = n_classes - 1

    targets_th = torch.tensor(targets)
    targets_seq_lens_th = torch.tensor(target_seq_lens)

    if label_loop:
        fsa = get_ctc_fsa_fast_bw_np(targets=targets, seq_lens=target_seq_lens, blank_idx=blank_idx)
        assert fsa.start_end_states.shape == (2, len(target_seq_lens))
        edges = fsa.edges.astype("int32")
        weights = fsa.weights.astype("float32")
        start_end_states = fsa.start_end_states.astype("int32")
        if with_native_fsa:
            print("python edges:")
            print(edges)
            print("python start_end_states:")
            print(start_end_states)
    else:
        native_edges_th, native_weights_th, native_start_end_states_th = get_ctc_fsa_fast_bw(
            targets=targets_th, seq_lens=targets_seq_lens_th, blank_idx=blank_idx, label_loop=label_loop
        )
        edges, weights, start_end_states = (
            native_edges_th.numpy(),
            native_weights_th.numpy(),
            native_start_end_states_th.numpy(),
        )

    fwdbwd, obs_scores = py_baum_welch(
        am_scores=-_log_softmax(am_scores),
        float_idx=float_idx,
        edges=edges,
        weights=weights,
        start_end_states=start_end_states,
    )
    fwdbwd = numpy.exp(-fwdbwd)  # -log space -> prob space
    print(fwdbwd)
    print(obs_scores)

    if with_native_fsa:
        native_edges_th, native_weights_th, native_start_end_states_th = get_ctc_fsa_fast_bw(
            targets=targets_th, seq_lens=targets_seq_lens_th, blank_idx=blank_idx
        )
        native_edges, native_weights, native_start_end_states = (
            native_edges_th.numpy(),
            native_weights_th.numpy(),
            native_start_end_states_th.numpy(),
        )
        # Note: The native FSA vs the Python FSA are not exactly identical
        # (they just should be equivalent; although almost identical).
        # We introduce a dummy state (last before end state), and some dummy edges.
        print("native edges:")
        print(native_edges)
        print("native_start_end_states:")
        print(native_start_end_states)

        native_fwdbwd, native_obs_scores = py_baum_welch(
            am_scores=-_log_softmax(am_scores),
            float_idx=float_idx,
            edges=native_edges,
            weights=native_weights,
            start_end_states=native_start_end_states,
        )
        native_fwdbwd = numpy.exp(-native_fwdbwd)  # -log space -> prob space
        print(native_fwdbwd)
        print(native_obs_scores)
        for b in range(n_batch):
            for t in range(seq_lens[b]):
                numpy.testing.assert_almost_equal(fwdbwd[t, b], native_fwdbwd[t, b], decimal=5)
        for b in range(n_batch):
            numpy.testing.assert_almost_equal(obs_scores[b], native_obs_scores[b], decimal=5)
        fwdbwd = native_fwdbwd
        obs_scores = native_obs_scores

    if not label_loop:
        raise SkipTest("skip ref ctc_loss for label_loop=False")

    am_scores_th = torch.tensor(am_scores)
    seq_lens_th = torch.tensor(seq_lens)
    am_scores_th.requires_grad_()
    log_probs_th = torch.log_softmax(am_scores_th, dim=-1)
    ref_ctc_loss_th = torch.nn.functional.ctc_loss(
        log_probs=log_probs_th,
        input_lengths=seq_lens_th,
        targets=targets_th,
        target_lengths=targets_seq_lens_th,
        blank=blank_idx,
        reduction="none",
    )
    ref_ctc_loss_th.sum().backward()
    # See grad definition of CTCLoss.
    # The op will calculate the gradient w.r.t. the logits (log softmax).
    # I.e. with y = softmax(z), this is \partial loss / \partial z = y - soft_align.
    # Also see CtcLoss.get_soft_alignment.
    ref_ctc_loss_grad_th = am_scores_th.grad  # time major, i.e. (time, batch, dim)
    y_th = torch.softmax(am_scores_th, dim=-1)  # (time, batch, dim)
    soft_align_th = y_th - ref_ctc_loss_grad_th
    ref_fwdbwd, ref_obs_score = (soft_align_th.detach().numpy(), ref_ctc_loss_th.detach().numpy())
    print(ref_fwdbwd)
    print(ref_obs_score)

    for b in range(n_batch):
        for t in range(seq_lens[b]):
            numpy.testing.assert_almost_equal(fwdbwd[t, b], ref_fwdbwd[t, b], decimal=5)
    for b in range(n_batch):
        numpy.testing.assert_almost_equal(obs_scores[0, b], ref_obs_score[b], decimal=5)


def test_ctc_fsa_batch3_len6_c8():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[1, 3, 4, 2, 1, 0], [2, 6, 3, 4, 0, 0], [0, 3, 2, 0, 0, 0]], dtype="int32")
    target_seq_lens = numpy.array([6, 4, 3], dtype="int32")
    n_classes = 8  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes)


def test_ctc_fsa_batch1_len2():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[0, 1]], dtype="int32")
    target_seq_lens = numpy.array([2], dtype="int32")
    n_classes = 3  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes)


def test_ctc_fsa_batch1_len1():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[0]], dtype="int32")
    target_seq_lens = numpy.array([1], dtype="int32")
    n_classes = 2  # +1 because of blank

    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes)


def test_ctc_fsa_batch3_len6_c8_native():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[1, 3, 4, 2, 1, 0], [2, 6, 3, 4, 0, 0], [0, 3, 2, 0, 0, 0]], dtype="int32")
    target_seq_lens = numpy.array([6, 4, 3], dtype="int32")
    n_classes = 8  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch4_len6_c8_native():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array(
        [[1, 2, 4, 4, 1, 0], [2, 6, 3, 4, 0, 0], [3, 3, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]], dtype="int32"
    )
    target_seq_lens = numpy.array([6, 4, 2, 1], dtype="int32")
    n_classes = 8  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch4_len6_c8_native_no_loop():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    No label loop (ctc_merge_repeated False) is equivalent to the Recurrent Neural Aligner (RNA) topology.
    """
    targets = numpy.array(
        [[1, 2, 4, 4, 1, 0], [2, 6, 3, 4, 0, 0], [3, 3, 0, 0, 0, 0], [5, 0, 0, 0, 0, 0]], dtype="int32"
    )
    target_seq_lens = numpy.array([6, 4, 2, 1], dtype="int32")
    n_classes = 8  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, label_loop=False)


def test_ctc_fsa_batch2_len2a():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[0, 1], [1, 0]], dtype="int32")
    target_seq_lens = numpy.array([2, 1], dtype="int32")
    n_classes = 3  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch2_len2():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[0, 1], [0, 0]], dtype="int32")
    target_seq_lens = numpy.array([2, 2], dtype="int32")
    n_classes = 3  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch2_len1():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[0], [1]], dtype="int32")
    target_seq_lens = numpy.array([1, 1], dtype="int32")
    n_classes = 3  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch1_len2rep_native():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[0, 0]], dtype="int32")
    target_seq_lens = numpy.array([2], dtype="int32")
    n_classes = 2  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch1_len2_native():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[0, 1]], dtype="int32")
    target_seq_lens = numpy.array([2], dtype="int32")
    n_classes = 3  # +1 because of blank
    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch1_len1_native():
    """
    This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
    """
    targets = numpy.array([[0]], dtype="int32")
    target_seq_lens = numpy.array([1], dtype="int32")
    n_classes = 2  # +1 because of blank

    check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_fast_viterbi(device: torch.device = torch.device("cpu")):
    n_batch = 3
    seq_len = 7
    n_classes = 5

    fsa = FastBwFsaShared()
    for i in range(n_classes):
        fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
        fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
    assert n_classes <= seq_len
    fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
    edges = fast_bw_fsa.edges
    weights = fast_bw_fsa.weights
    start_end_states = fast_bw_fsa.start_end_states
    am_scores = numpy.eye(n_classes, n_classes, dtype="float32")  # (dim,dim)

    am_scores = scipy.ndimage.zoom(am_scores, zoom=(float(seq_len) / n_classes, 1), order=1, prefilter=False)
    assert am_scores.shape == (seq_len, n_classes)
    am_scores = am_scores[:, None]
    am_scores = am_scores + numpy.zeros((seq_len, n_batch, n_classes), dtype="float32")
    print(am_scores[:, 0])
    # am_scores = numpy.ones((seq_len, n_batch, n_classes), dtype="float32") * numpy.float32(1.0 / n_classes)
    am_scores = numpy.log(am_scores)  # in +log space
    print("Construct call...")
    alignment, obs_scores = fast_viterbi(
        am_scores=torch.tensor(am_scores, device=device),
        am_seq_len=torch.tensor(numpy.array([seq_len] * n_batch, dtype="int32"), device=device),
        edges=torch.tensor(edges, device=device),
        weights=torch.tensor(weights, device=device),
        start_end_states=torch.tensor(start_end_states, device=device),
    )
    alignment, obs_scores = alignment.cpu().numpy(), obs_scores.cpu().numpy()
    print("Done.")
    print("score:")
    print(repr(obs_scores))
    assert obs_scores.shape == (n_batch,)
    print("Hard alignment:")
    print(repr(alignment))
    assert alignment.shape == (seq_len, n_batch)
    if seq_len == n_classes:
        print("Extra check identity...")
        for i in range(n_batch):
            for t in range(seq_len):
                assert alignment[t, i] == t
    if seq_len == 7 and n_classes == 5:
        print("Extra check ref_align (7,5)...")
        assert_allclose(obs_scores, -1.6218603, rtol=1e-5)  # should be the same everywhere
        for i in range(n_batch):
            assert alignment[:, i].tolist() == [0, 1, 1, 2, 3, 3, 4]
    print("Done.")


def test_fast_viterbi_cuda():
    if not torch.cuda.is_available():
        raise SkipTest("CUDA not available")
    test_fast_viterbi(device=torch.device("cuda"))


def test_fast_viterbi_rnd():
    n_batch = 4
    seq_len = 23
    n_classes = 5

    fsa = FastBwFsaShared()
    for i in range(n_classes):
        fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
        fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
    assert n_classes <= seq_len
    fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
    edges = fast_bw_fsa.edges
    weights = fast_bw_fsa.weights
    start_end_states = fast_bw_fsa.start_end_states
    am_scores = numpy.random.RandomState(42).normal(size=(seq_len, n_batch, n_classes)).astype("float32")
    am_seq_len = numpy.array([seq_len] * n_batch, dtype="int32")
    am_seq_len[1] -= 1
    am_seq_len[-1] -= 2
    am_seq_len[-2] = max(n_classes - 1, 1)  # no path possible
    ref_alignment, ref_scores = py_viterbi(
        am_scores=am_scores, am_seq_len=am_seq_len, edges=edges, weights=weights, start_end_states=start_end_states
    )
    print("ref score:")
    print(repr(ref_scores))
    assert ref_scores.shape == (n_batch,)
    print("ref hard alignment:")
    print(repr(ref_alignment))
    assert ref_alignment.shape == (seq_len, n_batch)
    print("Construct fast_viterbi call...")
    alignment, scores = fast_viterbi(
        am_scores=torch.tensor(am_scores),
        am_seq_len=torch.tensor(am_seq_len),
        edges=torch.tensor(edges),
        weights=torch.tensor(weights),
        start_end_states=torch.tensor(start_end_states),
    )
    alignment, scores = alignment.numpy(), scores.numpy()
    print("Done.")
    print("score:")
    print(repr(scores))
    assert scores.shape == (n_batch,)
    print("Hard alignment:")
    print(repr(alignment))
    assert alignment.shape == (seq_len, n_batch)
    assert_allclose(scores, ref_scores, rtol=1e-5)
    assert_allclose(alignment, ref_alignment, rtol=1e-5)
    print("Done.")


def test_edit_distance():
    rnd = numpy.random.RandomState(42)
    n_batch = 15
    n_a_max_len = 13
    n_b_max_len = 11
    num_classes = 10
    a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
    b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
    a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
    b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
    # Likely some high error. So make some explicit examples.
    expected_results: List[Optional[int]] = [None] * n_batch
    i = 0
    # One insertion/deletion.
    a_np[i, :1] = [1]
    a_len_np[i] = 1
    b_len_np[i] = 0
    expected_results[i] = 1
    i += 1
    # One deletion.
    a_np[i, :2] = [1, 2]
    b_np[i, :1] = [1]
    a_len_np[i] = 2
    b_len_np[i] = 1
    expected_results[i] = 1
    i += 1
    # One substitution + deletion.
    a_np[i, :2] = [1, 2]
    b_np[i, :1] = [3]
    a_len_np[i] = 2
    b_len_np[i] = 1
    expected_results[i] = 2
    i += 1
    # One substitution error.
    a_np[i, :4] = [1, 2, 3, 4]
    b_np[i, :4] = [1, 2, 4, 4]
    a_len_np[i] = 4
    b_len_np[i] = 4
    expected_results[i] = 1
    i += 1
    # One deletion error.
    a_np[i, :6] = [1, 2, 3, 3, 4, 5]
    b_np[i, :5] = [1, 2, 3, 4, 5]
    a_len_np[i] = 6
    b_len_np[i] = 5
    expected_results[i] = 1
    i += 1
    # One insertion error.
    a_np[i, :6] = [1, 2, 3, 4, 5, 6]
    b_np[i, :7] = [1, 2, 3, 4, 4, 5, 6]
    a_len_np[i] = 6
    b_len_np[i] = 7
    expected_results[i] = 1
    i += 1
    # Same.
    a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
    b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
    a_len_np[i] = 11
    b_len_np[i] = 11
    expected_results[i] = 0
    i += 1
    # Both full length. Error should be 2.
    a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
    b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
    a_len_np[i] = n_a_max_len
    b_len_np[i] = n_b_max_len
    expected_results[i] = 2
    i += 1
    assert n_batch - i >= 5  # still some random left
    a = torch.tensor(a_np)
    b = torch.tensor(b_np)
    a_len = torch.tensor(a_len_np)
    b_len = torch.tensor(b_len_np)

    for i in range(n_batch):
        print("testing batch", i, "/", n_batch)
        _a = a[i : i + 1, : a_len_np[i]]
        _a_len = a_len[i : i + 1]
        _b = b[i : i + 1, : b_len_np[i]]
        _b_len = b_len[i : i + 1]
        print("seq a:", a_np[i, : a_len_np[i]])
        print("seq b:", b_np[i, : b_len_np[i]])
        ref_edit_dist_np = edit_distance_ref_np(_a.numpy(), _a_len.numpy(), _b.numpy(), _b_len.numpy())
        native_edit_dist_np = edit_distance(_a, _a_len, _b, _b_len).numpy()
        assert isinstance(ref_edit_dist_np, numpy.ndarray)
        assert isinstance(native_edit_dist_np, numpy.ndarray)
        print("Ref edit dist:", ref_edit_dist_np)
        print("Native edit dist:", native_edit_dist_np)
        print("Expected edit dist:", expected_results[i])
        assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[0] == native_edit_dist_np[0]
        else:
            assert ref_edit_dist_np[0] == native_edit_dist_np[0]
        print("swapped:")
        ref_edit_dist_np = edit_distance_ref_np(_b.numpy(), _b_len.numpy(), _a.numpy(), _a_len.numpy())
        native_edit_dist_np = edit_distance(_b, _b_len, _a, _a_len).numpy()
        assert isinstance(ref_edit_dist_np, numpy.ndarray)
        assert isinstance(native_edit_dist_np, numpy.ndarray)
        print("Ref edit dist:", ref_edit_dist_np)
        print("Native edit dist:", native_edit_dist_np)
        print("Expected edit dist:", expected_results[i])
        assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[0] == native_edit_dist_np[0]
        else:
            assert ref_edit_dist_np[0] == native_edit_dist_np[0]
        print()

    print("Now the whole batch.")
    ref_edit_dist_np = edit_distance_ref_np(a.numpy(), a_len.numpy(), b.numpy(), b_len.numpy())
    native_edit_dist_np = edit_distance(a, a_len, b, b_len).numpy()
    assert isinstance(ref_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("Ref edit dist:", ref_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results)
    assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
    for i in range(n_batch):
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[i] == native_edit_dist_np[i]
        else:
            assert ref_edit_dist_np[i] == native_edit_dist_np[i]
    print()

    print("Now the whole batch, flipped.")
    ref_edit_dist_np = edit_distance_ref_np(b.numpy(), b_len.numpy(), a.numpy(), a_len.numpy())
    native_edit_dist_np = edit_distance(b, b_len, a, a_len).numpy()
    assert isinstance(ref_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("Ref edit dist:", ref_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results)
    assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
    for i in range(n_batch):
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[i] == native_edit_dist_np[i]
        else:
            assert ref_edit_dist_np[i] == native_edit_dist_np[i]


def test_edit_distance_2():
    from returnn.tensor import Tensor, Dim
    import numpy
    import torch
    from collections import namedtuple
    import itertools
    from numpy_ref_edit_distance import edit_distance_ref

    # noinspection PyShadowingNames
    def _check_edit_distance(a: Tensor, a_spatial_dim: Dim, b: Tensor, b_spatial_dim: Dim):
        ref = edit_distance_ref(a, a_spatial_dim, b, b_spatial_dim).raw_tensor
        res = edit_distance(a.raw_tensor, a_spatial_dim.dyn_size, b.raw_tensor, b_spatial_dim.dyn_size)
        assert res.shape == ref.shape == a_spatial_dim.dyn_size.shape == b_spatial_dim.dyn_size.shape
        assert len(ref.shape) == 1
        print("ref:", ref, "res:", res)
        batch_size = ref.shape[0]
        for i in range(batch_size):
            assert res[i] == ref[i], (
                f"batch idx i={i}, a[i]={a.raw_tensor[i]} len {a_spatial_dim.dyn_size[i]},"
                f" b[i]={b.raw_tensor[i]} len {b_spatial_dim.dyn_size[i]},"
                f" ref[i]={ref[i]}, res[i]={res[i]};\n"
                f" a={a.raw_tensor} lens {a_spatial_dim.dyn_size},"
                f" b={b.raw_tensor} lens {b_spatial_dim.dyn_size}"
            )
        assert (res.numpy() == ref).all()

    SizedTensor = namedtuple("SizedTensor", ["tensor", "seq_lens"])

    _SeqsB1 = [
        SizedTensor(torch.tensor([[1, 2, 3, 4]]), torch.tensor([4])),
        SizedTensor(torch.tensor([[1, 2, 3]]), torch.tensor([3])),
        SizedTensor(torch.tensor([[1, 2, 4]]), torch.tensor([3])),
        SizedTensor(torch.tensor([[1, 4]]), torch.tensor([2])),
        SizedTensor(torch.tensor([[5, 2, 4]]), torch.tensor([3])),
        SizedTensor(torch.tensor([[]], dtype=torch.int64), torch.tensor([0])),
    ]

    for a, b in itertools.product(_SeqsB1, _SeqsB1):
        a: SizedTensor
        b: SizedTensor
        # noinspection PyShadowingNames
        batch_dim = Dim(1, name="batch")
        a_spatial_dim = Dim(Tensor("a_sizes", [batch_dim], dtype="int64", raw_tensor=a.seq_lens))
        b_spatial_dim = Dim(Tensor("b_sizes", [batch_dim], dtype="int64", raw_tensor=b.seq_lens))
        a_ = Tensor("a", [batch_dim, a_spatial_dim], dtype="int64", raw_tensor=a.tensor)
        b_ = Tensor("b", [batch_dim, b_spatial_dim], dtype="int64", raw_tensor=b.tensor)
        _check_edit_distance(a_, a_spatial_dim, b_, b_spatial_dim)

    rnd = numpy.random.RandomState(42)
    for a, b in itertools.product(_SeqsB1, _SeqsB1):
        batch_size = rnd.randint(2, 11)
        a_max_len = rnd.randint(a.seq_lens[0], a.seq_lens[0] + 5)
        b_max_len = rnd.randint(b.seq_lens[0], b.seq_lens[0] + 5)
        a_sizes = rnd.randint(0, a_max_len + 1, size=(batch_size,))
        b_sizes = rnd.randint(0, b_max_len + 1, size=(batch_size,))
        a_sizes[0] = a.seq_lens[0]
        b_sizes[0] = b.seq_lens[0]
        a_max_len = max(a_sizes)
        b_max_len = max(b_sizes)
        a_values = rnd.randint(0, 10, (batch_size, a_max_len))
        b_values = rnd.randint(0, 10, (batch_size, b_max_len))
        a_values[0, : a.seq_lens[0]] = a.tensor[0, : a.seq_lens[0]]
        b_values[0, : b.seq_lens[0]] = b.tensor[0, : b.seq_lens[0]]
        a_sizes = torch.tensor(a_sizes, dtype=torch.int32)
        b_sizes = torch.tensor(b_sizes, dtype=torch.int32)

        # noinspection PyShadowingNames
        batch_dim = Dim(batch_size, name="batch")
        a_spatial_dim = Dim(Tensor("a_sizes", [batch_dim], dtype="int32", raw_tensor=a_sizes))
        b_spatial_dim = Dim(Tensor("b_sizes", [batch_dim], dtype="int32", raw_tensor=b_sizes))
        a_ = Tensor("a", [batch_dim, a_spatial_dim], dtype="int64", raw_tensor=torch.tensor(a_values))
        b_ = Tensor("b", [batch_dim, b_spatial_dim], dtype="int64", raw_tensor=torch.tensor(b_values))
        _check_edit_distance(a_, a_spatial_dim, b_, b_spatial_dim)


def test_edit_distance_ref_np_b1():
    assert edit_distance_ref_np_b1([1], []) == 1
    assert edit_distance_ref_np_b1([1, 2], [1]) == 1
    assert edit_distance_ref_np_b1([2, 2], [1]) == 2
    assert edit_distance_ref_np_b1([2, 1], [1]) == 1
    assert edit_distance_ref_np_b1([2, 1], [1, 1]) == 1
    assert edit_distance_ref_np_b1([2, 1], [1, 1, 1]) == 2
    assert edit_distance_ref_np_b1([2, 1], [2, 1, 1]) == 1


def _naive_optimal_completion_edit_distance(
    a: Union[Sequence[int], numpy.ndarray], b: Union[Sequence[int], numpy.ndarray]
) -> int:
    distances = [edit_distance_ref_np_b1(a, b[:n]) for n in range(len(b) + 1)]
    return min(distances)


def test_optimal_completion_edit_distance():
    rnd = numpy.random.RandomState(42)
    n_batch = 15
    n_a_max_len = 11
    n_b_max_len = 13
    num_classes = 10
    a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
    b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
    a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
    b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
    # Likely some high error. So make some explicit examples.
    expected_results = [None] * n_batch
    i = 0
    # One deletion.
    a_np[i, :1] = [1]
    a_len_np[i] = 1
    b_len_np[i] = 0
    expected_results[i] = 1
    i += 1
    # One optional insertion.
    a_np[i, :1] = [1]
    b_np[i, :2] = [1, 2]
    a_len_np[i] = 1
    b_len_np[i] = 2
    expected_results[i] = 0
    i += 1
    # One substitution or deletion.
    a_np[i, :1] = [1]
    b_np[i, :2] = [3, 1]
    a_len_np[i] = 1
    b_len_np[i] = 2
    expected_results[i] = 1
    i += 1
    # One substitution error.
    a_np[i, :4] = [1, 2, 3, 4]
    b_np[i, :4] = [1, 2, 4, 4]
    a_len_np[i] = 4
    b_len_np[i] = 4
    expected_results[i] = 1
    i += 1
    # One insertion error.
    a_np[i, :5] = [1, 2, 3, 4, 5]
    b_np[i, :6] = [1, 2, 3, 3, 4, 5]
    a_len_np[i] = 5
    b_len_np[i] = 6
    expected_results[i] = 1
    i += 1
    # Same.
    a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
    b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
    a_len_np[i] = 11
    b_len_np[i] = 11
    expected_results[i] = 0
    i += 1
    # Both full length.
    a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
    b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
    a_len_np[i] = n_a_max_len
    b_len_np[i] = n_b_max_len
    expected_results[i] = 0
    i += 1
    assert n_batch - i >= 5  # still some random left
    a = torch.tensor(a_np)
    b = torch.tensor(b_np)
    a_len = torch.tensor(a_len_np)
    b_len = torch.tensor(b_len_np)
    for i in range(n_batch):
        print("testing batch", i, "/", n_batch)
        _a = a[i : i + 1, : a_len_np[i]]
        _a_len = a_len[i : i + 1]
        _b = b[i : i + 1, : b_len_np[i]]
        _b_len = b_len[i : i + 1]
        print("seq a:", a_np[i, : a_len_np[i]])
        print("seq b:", b_np[i, : b_len_np[i]])
        _native_edit_dist = optimal_completion_edit_distance(_a, _a_len, _b, _b_len)
        native_edit_dist_np = _native_edit_dist.numpy()
        ref_edit_dist_np = numpy.array(
            [_naive_optimal_completion_edit_distance(a_np[i, : a_len_np[i]], b_np[i, : b_len_np[i]])]
        )
        assert isinstance(ref_edit_dist_np, numpy.ndarray)
        assert isinstance(native_edit_dist_np, numpy.ndarray)
        print("Ref edit dist:", ref_edit_dist_np)
        print("Native edit dist:", native_edit_dist_np)
        print("Expected edit dist:", expected_results[i])
        assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[0] == native_edit_dist_np[0]
        else:
            assert ref_edit_dist_np[0] == native_edit_dist_np[0]
        print()

    print("Now the whole batch.")
    native_edit_dist = optimal_completion_edit_distance(a, a_len, b, b_len)
    native_edit_dist_np = native_edit_dist.numpy()
    ref_edit_dist_np = numpy.array(
        [
            _naive_optimal_completion_edit_distance(a_np[i, : a_len_np[i]], b_np[i, : b_len_np[i]])
            for i in range(n_batch)
        ]
    )
    assert isinstance(ref_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("Ref edit dist:", ref_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results)
    assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
    for i in range(n_batch):
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[i] == native_edit_dist_np[i]
        else:
            assert ref_edit_dist_np[i] == native_edit_dist_np[i]
    print()


def test_optimal_completion_edit_distance_per_successor():
    rnd = numpy.random.RandomState(42)
    n_batch = 15
    n_a_max_len = 11
    n_b_max_len = 13
    num_classes = 10
    a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
    b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
    a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
    b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
    # Likely some high error. So make some explicit examples.
    expected_results = [None] * n_batch
    i = 0
    # One deletion.
    a_np[i, :1] = [1]
    a_len_np[i] = 1
    b_len_np[i] = 0
    expected_results[i] = 1
    i += 1
    # One optional insertion.
    a_np[i, :1] = [1]
    b_np[i, :2] = [1, 2]
    a_len_np[i] = 1
    b_len_np[i] = 2
    expected_results[i] = 0
    i += 1
    # One substitution or deletion.
    a_np[i, :1] = [1]
    b_np[i, :2] = [3, 1]
    a_len_np[i] = 1
    b_len_np[i] = 2
    expected_results[i] = 1
    i += 1
    # One substitution error.
    a_np[i, :4] = [1, 2, 3, 4]
    b_np[i, :4] = [1, 2, 4, 4]
    a_len_np[i] = 4
    b_len_np[i] = 4
    expected_results[i] = 1
    i += 1
    # One insertion error.
    a_np[i, :5] = [1, 2, 3, 4, 5]
    b_np[i, :6] = [1, 2, 3, 3, 4, 5]
    a_len_np[i] = 5
    b_len_np[i] = 6
    expected_results[i] = 1
    i += 1
    # Same.
    a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 1, 3]
    b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 1, 3]
    a_len_np[i] = 11
    b_len_np[i] = 11
    expected_results[i] = 0
    i += 1
    # Both full length.
    a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
    b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
    a_len_np[i] = n_a_max_len
    b_len_np[i] = n_b_max_len
    expected_results[i] = 0
    i += 1
    assert n_batch - i >= 5  # still some random left
    a = torch.tensor(a_np)
    b = torch.tensor(b_np)
    assert all(a_len_np > 0)
    a_len = torch.tensor(a_len_np)
    b_len = torch.tensor(b_len_np)
    print("Now the whole batch.")
    # a_len - 1 such that we can do the check below.
    native_edit_dist = optimal_completion_edit_distance_per_successor(a, a_len - 1, b, b_len, num_classes)
    native_edit_dist_np = native_edit_dist.numpy()
    ref_edit_dist_np = numpy.array(
        [
            _naive_optimal_completion_edit_distance(a_np[i, : a_len_np[i]], b_np[i, : b_len_np[i]])
            for i in range(n_batch)
        ]
    )
    assert isinstance(ref_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("Ref edit dist:", ref_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results)
    assert ref_edit_dist_np.shape == (n_batch,)
    assert native_edit_dist_np.shape == (n_batch, num_classes)
    for i in range(n_batch):
        a_last = a_np[i, a_len_np[i] - 1]
        assert 0 <= a_last < num_classes
        native_res = native_edit_dist_np[i, a_last]
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[i] == native_res
        else:
            assert ref_edit_dist_np[i] == native_res
        for j in range(num_classes):
            ref_res = _naive_optimal_completion_edit_distance(
                list(a_np[i, : a_len_np[i] - 1]) + [j], list(b_np[i, : b_len_np[i]])
            )
            native_res = native_edit_dist_np[i, j]
            assert ref_res == native_res
    print()


def test_next_edit_distance_row():
    rnd = numpy.random.RandomState(42)
    n_batch = 15
    n_a_max_len = 13
    n_b_max_len = 11
    num_classes = 10
    a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
    b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
    a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
    b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
    # Likely some high error. So make some explicit examples.
    expected_results = [None] * n_batch
    i = 0
    # One insertion/deletion.
    a_np[i, :1] = [1]
    a_len_np[i] = 1
    b_len_np[i] = 0
    expected_results[i] = 1
    i += 1
    # One deletion.
    a_np[i, :2] = [1, 2]
    b_np[i, :1] = [1]
    a_len_np[i] = 2
    b_len_np[i] = 1
    expected_results[i] = 1
    i += 1
    # One substitution + deletion.
    a_np[i, :2] = [1, 2]
    b_np[i, :1] = [3]
    a_len_np[i] = 2
    b_len_np[i] = 1
    expected_results[i] = 2
    i += 1
    # One substitution error.
    a_np[i, :4] = [1, 2, 3, 4]
    b_np[i, :4] = [1, 2, 4, 4]
    a_len_np[i] = 4
    b_len_np[i] = 4
    expected_results[i] = 1
    i += 1
    # One deletion error.
    a_np[i, :6] = [1, 2, 3, 3, 4, 5]
    b_np[i, :5] = [1, 2, 3, 4, 5]
    a_len_np[i] = 6
    b_len_np[i] = 5
    expected_results[i] = 1
    i += 1
    # One insertion error.
    a_np[i, :6] = [1, 2, 3, 4, 5, 6]
    b_np[i, :7] = [1, 2, 3, 4, 4, 5, 6]
    a_len_np[i] = 6
    b_len_np[i] = 7
    expected_results[i] = 1
    i += 1
    # Same.
    a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
    b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
    a_len_np[i] = 11
    b_len_np[i] = 11
    expected_results[i] = 0
    i += 1
    # Both full length. Error should be 2.
    a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
    b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
    a_len_np[i] = n_a_max_len
    b_len_np[i] = n_b_max_len
    expected_results[i] = 2
    i += 1
    assert n_batch - i >= 5  # still some random left
    a = torch.tensor(a_np)
    b = torch.tensor(b_np)
    a_len = torch.tensor(a_len_np)
    b_len = torch.tensor(b_len_np)

    for i in range(n_batch):
        print("testing batch", i, "/", n_batch)
        _a = a[i : i + 1, : a_len_np[i]]
        _a_len = a_len[i : i + 1]
        _b = b[i : i + 1, : b_len_np[i]]
        _b_len = b_len[i : i + 1]
        print("seq a:", a_np[i, : a_len_np[i]])
        print("seq b:", b_np[i, : b_len_np[i]])
        ref_edit_dist_np = edit_distance_ref_np(_a.numpy(), _a_len.numpy(), _b.numpy(), _b_len.numpy())
        native_edit_dist_np = edit_distance_via_next_edit_distance_row(_a, _a_len, _b, _b_len).numpy()
        assert isinstance(ref_edit_dist_np, numpy.ndarray)
        assert isinstance(native_edit_dist_np, numpy.ndarray)
        print("Ref edit dist:", ref_edit_dist_np)
        print("Native edit dist:", native_edit_dist_np)
        print("Expected edit dist:", expected_results[i])
        assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[0] == native_edit_dist_np[0]
        else:
            assert ref_edit_dist_np[0] == native_edit_dist_np[0]
        print("swapped:")
        ref_edit_dist_np = edit_distance_ref_np(_b.numpy(), _b_len.numpy(), _a.numpy(), _a_len.numpy())
        native_edit_dist_np = edit_distance_via_next_edit_distance_row(_b, _b_len, _a, _a_len).numpy()
        assert isinstance(ref_edit_dist_np, numpy.ndarray)
        assert isinstance(native_edit_dist_np, numpy.ndarray)
        print("Ref edit dist:", ref_edit_dist_np)
        print("Native edit dist:", native_edit_dist_np)
        print("Expected edit dist:", expected_results[i])
        assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[0] == native_edit_dist_np[0]
        else:
            assert ref_edit_dist_np[0] == native_edit_dist_np[0]
        print()

    print("Now the whole batch.")
    ref_edit_dist_np = edit_distance_ref_np(a.numpy(), a_len.numpy(), b.numpy(), b_len.numpy())
    native_edit_dist_np = edit_distance_via_next_edit_distance_row(a, a_len, b, b_len).numpy()
    assert isinstance(ref_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("Ref edit dist:", ref_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results)
    assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
    for i in range(n_batch):
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[i] == native_edit_dist_np[i]
        else:
            assert ref_edit_dist_np[i] == native_edit_dist_np[i]
    print()

    print("Now the whole batch, flipped.")
    ref_edit_dist_np = edit_distance_ref_np(b.numpy(), b_len.numpy(), a.numpy(), a_len.numpy())
    native_edit_dist_np = edit_distance_via_next_edit_distance_row(b, b_len, a, a_len).numpy()
    assert isinstance(ref_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("Ref edit dist:", ref_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results)
    assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
    for i in range(n_batch):
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[i] == native_edit_dist_np[i]
        else:
            assert ref_edit_dist_np[i] == native_edit_dist_np[i]


def test_next_edit_distance_row_optimal_completion():
    rnd = numpy.random.RandomState(42)
    n_batch = 15
    n_a_max_len = 11
    n_b_max_len = 13
    num_classes = 10
    a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
    b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
    a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
    b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
    # Likely some high error. So make some explicit examples.
    expected_results = [None] * n_batch
    i = 0
    # One deletion.
    a_np[i, :1] = [1]
    a_len_np[i] = 1
    b_len_np[i] = 0
    expected_results[i] = 1
    i += 1
    # One optional insertion.
    a_np[i, :1] = [1]
    b_np[i, :2] = [1, 2]
    a_len_np[i] = 1
    b_len_np[i] = 2
    expected_results[i] = 0
    i += 1
    # One substitution or deletion.
    a_np[i, :1] = [1]
    b_np[i, :2] = [3, 1]
    a_len_np[i] = 1
    b_len_np[i] = 2
    expected_results[i] = 1
    i += 1
    # One substitution error.
    a_np[i, :4] = [1, 2, 3, 4]
    b_np[i, :4] = [1, 2, 4, 4]
    a_len_np[i] = 4
    b_len_np[i] = 4
    expected_results[i] = 1
    i += 1
    # One insertion error.
    a_np[i, :5] = [1, 2, 3, 4, 5]
    b_np[i, :6] = [1, 2, 3, 3, 4, 5]
    a_len_np[i] = 5
    b_len_np[i] = 6
    expected_results[i] = 1
    i += 1
    # Same.
    a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
    b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
    a_len_np[i] = 11
    b_len_np[i] = 11
    expected_results[i] = 0
    i += 1
    # Both full length.
    a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
    b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
    a_len_np[i] = n_a_max_len
    b_len_np[i] = n_b_max_len
    expected_results[i] = 0
    i += 1
    assert n_batch - i >= 5  # still some random left
    a = torch.tensor(a_np)
    b = torch.tensor(b_np)
    a_len = torch.tensor(a_len_np)
    b_len = torch.tensor(b_len_np)
    print("Now the whole batch.")
    native_edit_dist = edit_distance_via_next_edit_distance_row(a, a_len, b, b_len, optimal_completion=True)
    native_edit_dist_np = native_edit_dist.numpy()
    ref_edit_dist_np = numpy.array(
        [
            _naive_optimal_completion_edit_distance(a_np[i, : a_len_np[i]], b_np[i, : b_len_np[i]])
            for i in range(n_batch)
        ]
    )
    assert isinstance(ref_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("Ref edit dist:", ref_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results)
    assert ref_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
    for i in range(n_batch):
        if expected_results[i] is not None:
            assert expected_results[i] == ref_edit_dist_np[i] == native_edit_dist_np[i]
        else:
            assert ref_edit_dist_np[i] == native_edit_dist_np[i]
    print()


def test_next_edit_distance_reduce_optimal_completion():
    rnd = numpy.random.RandomState(42)
    n_batch = 15
    n_a_max_len = 7
    n_b_max_len = 13
    num_classes = 10
    a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
    b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
    # Test only for full length seqs in a.
    a_len_np = numpy.array([n_a_max_len] * n_batch, dtype="int32")
    b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
    a = torch.tensor(a_np)
    b = torch.tensor(b_np)
    assert all(a_len_np > 0)
    a_len = torch.tensor(a_len_np)
    b_len = torch.tensor(b_len_np)
    print("Now the whole batch.")
    # a_len - 1 such that we can do the check below.
    native_edit_dist = optimal_completion_edit_distance_per_successor_via_next_edit_distance(
        a, a_len, b, b_len, num_classes
    )
    native_edit_dist_np = native_edit_dist.numpy()
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("Native edit dist:", native_edit_dist_np)
    assert native_edit_dist_np.shape == (n_batch, num_classes)
    for i in range(n_batch):
        for j in range(num_classes):
            ref_res = _naive_optimal_completion_edit_distance(
                list(a_np[i, : a_len_np[i]]) + [j], b_np[i, : b_len_np[i]]
            )
            native_res = native_edit_dist_np[i, j]
            assert ref_res == native_res
    print()
