"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from rf_utils import run_model


def test_while_loop_simple():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        model, extern_data  # noqa  # unused
        i = rf.while_loop(
            cond=lambda i_: i_ < time_dim.get_dim_value_tensor(),
            body=lambda i_: i_ + 1,
            initial=rf.constant(0, dims=()),
        )
        i.mark_as_default_output(shape=())

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_while_loop_two_state():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32", feature_dim=in_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        model  # noqa  # unused
        data = extern_data["data"]
        _, out = rf.while_loop(
            cond=lambda s: s[0] < 2, body=lambda s: (s[0] + 1, s[1] * 0.9), initial=(rf.constant(0, dims=()), data)
        )
        assert out.control_flow_ctx is None
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_while_loop():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            def _cond(s: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
                t, ended, s_ = s
                if t.raw_tensor.__class__.__module__.startswith("torch"):
                    print("**", t.raw_tensor, ended.raw_tensor, rf.reduce_sum(s_, axis=in_dim).raw_tensor)
                return rf.logical_not(rf.reduce_all(ended, axis=[batch_dim]))

            def _body(s):
                t, ended, s_ = s
                cont = rf.logical_and(rf.reduce_sum(s_, axis=in_dim) < 50, t < time_dim.get_size_tensor())
                ended = rf.logical_or(ended, rf.logical_not(cont))
                s__ = s_ + rf.abs(rf.gather(x, indices=t, axis=time_dim, clip_to_valid=True))
                s__ = rf.where(ended, s_, s__)
                return t + 1, ended, s__

            _, _, final_s = rf.while_loop(
                _cond,
                _body,
                initial=(
                    rf.zeros((), dtype=rf.get_default_array_index_dtype()),  # t
                    rf.zeros((batch_dim,), dtype="bool"),  # ended
                    rf.zeros((batch_dim, in_dim)),  # s
                ),
            )
            return final_s

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_tensorflow=False)


def test_scan_unknown_len():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Dim]:
            def _cond(_, s: Tuple[Tensor, Tensor]):
                t, s_ = s
                if t.raw_tensor.__class__.__module__.startswith("torch"):
                    print("**", t.raw_tensor, rf.reduce_sum(s_, axis=in_dim).raw_tensor)
                return rf.logical_and(rf.reduce_sum(s_, axis=in_dim) < 20, t < time_dim.get_dim_value_tensor())

            def _body(_, s):
                t, s_ = s
                y_ = s_ + rf.abs(rf.gather(x, indices=t, axis=time_dim))
                return y_, (t + 1, y_)

            y, _, out_time_dim = rf.scan(
                cond=_cond,
                body=_body,
                cond_dims=[batch_dim],
                initial=(rf.zeros((), dtype=rf.get_default_array_index_dtype()), rf.zeros((batch_dim, in_dim))),
                ys=Tensor("y", dims=(batch_dim, in_dim), dtype=x.dtype),
            )

            return y, out_time_dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, out_time_dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, out_time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_tensorflow=False)


def test_scan_existing_spatial_dim():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            def _body(x_, s):
                y_ = s + x_
                return y_, y_

            y, _, _ = rf.scan(
                spatial_dim=time_dim,
                body=_body,
                initial=rf.zeros((batch_dim, in_dim)),
                xs=x,
                ys=Tensor("y", dims=(batch_dim, in_dim), dtype=x.dtype),
            )

            return y

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_tensorflow=False)


def test_scan_changing_dim():
    # This is a common case for beam search.
    # https://github.com/rwth-i6/returnn/issues/1327
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Dim]:
            def _body(x_: Tensor, s):
                s_ = s["state"]
                beam_in_dim = s["beam_dim"]
                y_ = s_ + x_
                # Make new beam dim and then remove prev beam dim.
                # Effectively, this is what you would get with top_k on the logits.
                beam_dim = Dim(3, name="beam")
                r = rf.range_over_dim(beam_dim, dtype=x_.dtype)
                r.sparse_dim = None
                y_ = rf.combine_bc(y_, "mul", r)
                y_ = rf.reduce_mean(y_, axis=beam_in_dim)
                return y_, {"state": y_, "beam_dim": beam_dim}

            initial_beam_dim = Dim(1, name="initial-beam")
            y, last_state, _ = rf.scan(
                spatial_dim=time_dim,
                body=_body,
                initial={"state": rf.zeros((batch_dim, initial_beam_dim, in_dim)), "beam_dim": initial_beam_dim},
                xs=x,
                ys=Tensor("y", dims=(batch_dim, initial_beam_dim, in_dim), dtype=x.dtype),
                return_tensor_arrays=True,
            )
            final_beam_dim = last_state["beam_dim"]
            assert isinstance(y, TensorArray)
            last = y[-1]
            return last, final_beam_dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, beam_dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, beam_dim, in_dim))

    # TODO the way this is implemented, accessing y[-1], is not consistent w.r.t. different batch sizes...
    run_model(
        extern_data, lambda *, epoch, step: _Net(), _forward_step, test_tensorflow=False, test_single_batch_entry=False
    )


def test_while_loop_growing_dim():
    # https://github.com/rwth-i6/returnn/issues/1327
    # A dim which grows per iteration can only be a graph-loop var if it starts DYNAMIC:
    # its size then travels with the loop, while the dim object stays the same.
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32", feature_dim=in_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        del model  # unused
        xs = TensorArray.unstack(extern_data["data"], axis=time_dim)
        hist_dim = Dim(rf.zeros((), dtype="int32"), name="hist")
        hist = rf.zeros([batch_dim, hist_dim, in_dim])

        def _body(state):
            t, hist_, hist_dim_ = state
            hist_, hist_dim_ = rf.cum_concat_step(xs[t], prev_accum=hist_, axis=hist_dim_)
            return t + 1, hist_, hist_dim_

        _, hist, hist_dim = rf.while_loop(
            cond=lambda s: s[0] < rf.copy_to_device(time_dim.get_dim_value_tensor(), "cpu"),
            body=_body,
            initial=(rf.constant(0, dims=(), dtype="int32", device="cpu"), hist, hist_dim),
        )
        rf.reduce_sum(hist, axis=hist_dim).mark_as_default_output(shape=(batch_dim, in_dim))

    # test_single_batch_entry: the loop runs over the padded extent,
    # so a shorter sequence alone takes fewer steps and is not the same computation
    run_model(
        extern_data,
        lambda *, epoch, step: rf.Module(),
        _forward_step,
        tf_low_level=True,
        test_single_batch_entry=False,
    )


def test_gather_ragged_hist_after_mask():
    # Reorder hypotheses whose histories have DIFFERENT lengths.
    # The history is built as a search step builds it (cum_concat_step, then mask_nested,
    # so label 0 counts as blank), which makes its dim a derived one with per-hypothesis sizes.
    # Hypothesis 0 (empty) must end up with hypothesis 1's label.
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    vocab_dim = Dim(5, name="vocab")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, vocab_dim], dtype="float32", feature_dim=vocab_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        del model, extern_data  # unused, the case is deterministic
        beam_dim = Dim(3, name="beam")
        batch_dims = [batch_dim, beam_dim]
        target = rf.expand_dim(rf.cast(rf.range_over_dim(beam_dim), "int32"), dim=batch_dim)  # 0, 1, 2
        target.sparse_dim = None
        backrefs = rf.expand_dim(rf.cast((rf.range_over_dim(beam_dim) + 1) % 3, "int32"), dim=batch_dim)
        backrefs.sparse_dim = beam_dim

        hist_dim = Dim(rf.zeros(batch_dims, dtype="int32"), name="hist")
        hist = rf.zeros(batch_dims + [hist_dim], dtype="int32")
        new_hist, new_hist_dim = rf.cum_concat_step(target, prev_accum=hist, axis=hist_dim)
        hist, hist_dim = rf.nested.mask_nested((new_hist, new_hist_dim), mask=target != 0, mask_value=(hist, hist_dim))
        hist, hist_dim = rf.nested.gather_nested((hist, hist_dim), indices=backrefs)

        hist_dim.get_size_tensor().mark_as_output("lens", shape=(batch_dim, beam_dim))
        # one zero slot appended, so reading works while a history is still empty,
        # and read onto a static axis, so this compares values and not dim metadata
        hist_p, (dim_p,) = rf.pad(hist, axes=[hist_dim], padding=[(0, 1)], value=rf.zeros((), dtype=hist.dtype))
        pos_dim = Dim(2, name="pos")
        pos = rf.minimum(rf.range_over_dim(pos_dim), rf.copy_to_device(hist_dim.get_dim_value_tensor(), None))
        rf.gather(hist_p, axis=dim_p, indices=pos).mark_as_output("hist", shape=(batch_dim, beam_dim, pos_dim))

    run_model(
        extern_data,
        lambda *, epoch, step: rf.Module(),
        _forward_step,
        tf_low_level=True,
        test_single_batch_entry=False,
    )


def test_while_loop_beam_search_growing_hist():
    # Simplified beam search, as the recog defs do it:
    # fixed beam, top_k over (beam, vocab), backrefs to reorder the hypotheses,
    # and a label history which only the hypotheses with a non-blank label extend,
    # so the histories diverge in length and mask_nested merges them.
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    vocab_dim = Dim(5, name="vocab")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, vocab_dim], dtype="float32", feature_dim=vocab_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        del model  # unused
        label_log_prob_ta = TensorArray.unstack(rf.log_softmax(extern_data["data"], axis=vocab_dim), axis=time_dim)
        beam_dim = Dim(3, name="beam")
        step_beam_dim = Dim(3, name="beam-step")  # top_k out dim, mapped back onto beam_dim
        batch_dims = [batch_dim, beam_dim]
        # -inf for the not-yet-real hypotheses, so the first step selects as a growing beam would
        seq_log_prob = rf.where(rf.range_over_dim(beam_dim) == 0, rf.constant(0.0, dims=batch_dims), float("-inf"))
        hist_dim = Dim(rf.zeros(batch_dims, dtype="int32"), name="hist")
        hist = rf.zeros(batch_dims + [hist_dim], dtype="int32")

        def _body(state):
            t, seq_log_prob_, hist_, hist_dim_ = state
            seq_log_prob_ = rf.combine(
                seq_log_prob_, "+", label_log_prob_ta[t], allow_broadcast_all_sources=True
            )  # Batch, InBeam, Vocab
            seq_log_prob_, (backrefs, target), _ = rf.top_k(
                seq_log_prob_, k_dim=step_beam_dim, axis=[beam_dim, vocab_dim]
            )
            seq_log_prob_, _ = rf.replace_dim(seq_log_prob_, in_dim=step_beam_dim, out_dim=beam_dim)
            backrefs, _ = rf.replace_dim(backrefs, in_dim=step_beam_dim, out_dim=beam_dim)
            backrefs = rf.cast(backrefs, "int32")  # top_k index dtype is backend specific
            backrefs.sparse_dim = beam_dim
            target, _ = rf.replace_dim(target, in_dim=step_beam_dim, out_dim=beam_dim)
            target = rf.cast(target, "int32")
            target.sparse_dim = None
            hist_, hist_dim_ = rf.nested.gather_nested((hist_, hist_dim_), indices=backrefs)
            new_hist, new_hist_dim = rf.cum_concat_step(target, prev_accum=hist_, axis=hist_dim_)
            new_hist, new_hist_dim = rf.nested.mask_nested(
                (new_hist, new_hist_dim), mask=target != 0, mask_value=(hist_, hist_dim_)
            )
            return t + 1, seq_log_prob_, new_hist, new_hist_dim

        _, seq_log_prob, hist, hist_dim = rf.while_loop(
            cond=lambda s: s[0] < rf.copy_to_device(time_dim.get_dim_value_tensor(), "cpu"),
            body=_body,
            initial=(rf.constant(0, dims=(), dtype="int32", device="cpu"), seq_log_prob, hist, hist_dim),
        )
        seq_log_prob.mark_as_output("scores", shape=(batch_dim, beam_dim))
        hist_dim.get_size_tensor().mark_as_output("hist_lens", shape=(batch_dim, beam_dim))
        # weighted by position, so a buffer whose entries are right but misplaced also fails
        hist_checksum = rf.reduce_sum(
            rf.cast(hist, "float32") * rf.cast(rf.range_over_dim(hist_dim) + 1, "float32"), axis=hist_dim
        )
        hist_checksum.mark_as_output("hist_checksum", shape=(batch_dim, beam_dim))

    # test_single_batch_entry: see test_while_loop_growing_dim
    run_model(
        extern_data,
        lambda *, epoch, step: rf.Module(),
        _forward_step,
        tf_low_level=True,
        test_single_batch_entry=False,
    )


def _beam_search_fixed_shape(log_probs: Tensor, *, time_dim: Dim, vocab_dim: Dim, beam_dim: Dim, num_frames: Tensor):
    """
    Frame-synchronous beam search as in the graph-loop recog defs:
    fixed beam, top_k over (beam, vocab), labels and backrefs in TensorArrays with a capacity,
    then backtracking in a second loop, reading the arrays in reverse.

    :param log_probs: [batch, time, vocab]
    :param num_frames: scalar int32, number of frames to search (at most time_dim)
    :return: scores [batch, beam], labels TensorArray (one entry per frame, reversed),
        final labels and backrefs TensorArrays of the search loop
    """
    batch_dims = log_probs.remaining_dims((time_dim, vocab_dim))
    step_beam_dim = Dim(beam_dim.dimension, name="beam-step")  # top_k out dim, mapped back onto beam_dim
    log_probs_ta = TensorArray.unstack(log_probs, axis=time_dim)
    capacity = time_dim.dimension
    label_template = Tensor("label", dims=batch_dims + [beam_dim], dtype="int32", sparse_dim=vocab_dim)
    backref_template = Tensor("backref", dims=batch_dims + [beam_dim], dtype="int32", sparse_dim=beam_dim)

    def _body(state):
        t, seq_log_prob_, labels_ta_, backrefs_ta_ = state
        seq_log_prob_ = rf.combine(seq_log_prob_, "+", log_probs_ta[t], allow_broadcast_all_sources=True)
        seq_log_prob_, (backrefs, labels), _ = rf.top_k(seq_log_prob_, k_dim=step_beam_dim, axis=[beam_dim, vocab_dim])
        seq_log_prob_, _ = rf.replace_dim(seq_log_prob_, in_dim=step_beam_dim, out_dim=beam_dim)
        backrefs, _ = rf.replace_dim(rf.cast(backrefs, "int32"), in_dim=step_beam_dim, out_dim=beam_dim)
        backrefs.sparse_dim = beam_dim
        labels, _ = rf.replace_dim(rf.cast(labels, "int32"), in_dim=step_beam_dim, out_dim=beam_dim)
        labels.sparse_dim = vocab_dim
        return t + 1, seq_log_prob_, labels_ta_.push_back(labels), backrefs_ta_.push_back(backrefs)

    seq_log_prob = rf.where(
        rf.range_over_dim(beam_dim) == 0, rf.constant(0.0, dims=batch_dims + [beam_dim]), float("-inf")
    )
    _, seq_log_prob, labels_ta, backrefs_ta = rf.while_loop(
        cond=lambda s: s[0] < num_frames,
        body=_body,
        initial=(
            rf.zeros((), dtype="int32", device=num_frames.device),
            seq_log_prob,
            TensorArray(label_template, capacity=capacity),
            TensorArray(backref_template, capacity=capacity),
        ),
    )

    def _backtrack_body(state):
        t, indices, out_ta_ = state
        out_ta_ = out_ta_.push_back(rf.gather(labels_ta[t], indices=indices))
        indices = rf.gather(backrefs_ta[t], indices=indices)
        return t - 1, indices, out_ta_

    indices0 = rf.range_over_dim(beam_dim)
    for dim in batch_dims:
        indices0 = rf.expand_dim(indices0, dim=dim)
    indices0.sparse_dim = beam_dim
    _, _, out_ta = rf.while_loop(
        cond=lambda s: s[0] >= 0,
        body=_backtrack_body,
        initial=(num_frames - 1, indices0, TensorArray(label_template, capacity=capacity)),
    )
    return seq_log_prob, out_ta, labels_ta, backrefs_ta


def _beam_search_inputs(*, device: str, num_seqs: int = 2, num_frames: int = 6):
    import torch

    batch = Dim(num_seqs, name="batch")
    time_dim = Dim(num_frames, name="time")
    vocab_dim = Dim(5, name="vocab")
    log_probs = Tensor("log_probs", [batch, time_dim, vocab_dim], dtype="float32")
    raw = torch.randn(num_seqs, num_frames, 5, generator=torch.Generator().manual_seed(42))
    log_probs.raw_tensor = raw.log_softmax(-1).to(device)
    return log_probs, dict(time_dim=time_dim, vocab_dim=vocab_dim, beam_dim=Dim(3, name="beam"))


def _check_beam_search_same(res, ref, *, num_frames: int):
    import torch

    torch.testing.assert_close(res[0].copy_transpose(ref[0].dims).raw_tensor.cpu(), ref[0].raw_tensor.cpu())
    for res_ta, ref_ta in zip(res[1:], ref[1:]):
        for i in range(num_frames):
            a, b = res_ta[i], ref_ta[i]
            assert torch.equal(a.copy_transpose(b.dims).raw_tensor.cpu(), b.raw_tensor.cpu()), f"entry {i}"


def test_while_loop_torch_graph_loop_beam_search():
    # Under static tracing, the PT backend runs rf.while_loop as the torch while_loop op over a fixed-shape carry.
    # Same results as the eager Python loop, for several trip counts including zero.
    import torch
    import unittest

    if torch.__version__ < (2, 3):  # torch while_loop op since torch 2.3
        raise unittest.SkipTest("needs torch >= 2.3")
    rf.select_backend_torch()
    log_probs, dims = _beam_search_inputs(device="cpu")
    for n in [0, 1, 4, 6]:
        num_frames = rf.constant(n, dims=(), dtype="int32", device="cpu")
        ref = _beam_search_fixed_shape(log_probs, num_frames=num_frames, **dims)
        # noinspection PyProtectedMember
        assert isinstance(ref[1]._backend_tensor_array, list)
        with rf.set_static_traceable_ctx():
            res = _beam_search_fixed_shape(log_probs, num_frames=num_frames, **dims)
        # noinspection PyProtectedMember
        assert not isinstance(res[1]._backend_tensor_array, list), "graph loop not taken"
        assert res[1].stack(axis=dims["time_dim"]).raw_tensor.shape[0] == dims["time_dim"].dimension
        _check_beam_search_same(res, ref, num_frames=n)


def test_while_loop_torch_graph_loop_tensor_array_overflow():
    import torch
    import unittest

    if torch.__version__ < (2, 3):  # torch while_loop op since torch 2.3
        raise unittest.SkipTest("needs torch >= 2.3")
    rf.select_backend_torch()
    template = Tensor("x", dims=(), dtype="int32")
    with rf.set_static_traceable_ctx():
        try:
            rf.while_loop(
                cond=lambda s: s[0] < 4,
                body=lambda s: (s[0] + 1, s[1].push_back(s[0])),
                initial=(rf.zeros((), dtype="int32"), TensorArray(template, capacity=3)),
            )
        except (IndexError, RuntimeError) as exc:
            print("expected error:", exc)
        else:
            raise AssertionError("capacity overflow not detected")


def test_while_loop_torch_cuda_graph_while_node():
    # torch >= 2.14: the loop captured once as a CUDA graph while node, replayed for different trip counts.
    import torch
    import unittest

    if not torch.cuda.is_available():
        raise unittest.SkipTest("cuda only: real graph capture")
    if torch.__version__ < (2, 14):  # CUDA graph while nodes since torch 2.14
        raise unittest.SkipTest("needs torch >= 2.14")
    from torch._higher_order_ops.cudagraph_conditional_nodes import (
        CUDAGraphCaptureControlFlowOpDispatchMode,
        ControlFlowOpWarmupDispatchMode,
    )

    rf.select_backend_torch()
    log_probs_cpu, dims = _beam_search_inputs(device="cpu")
    log_probs = log_probs_cpu.copy()
    log_probs.raw_tensor = log_probs_cpu.raw_tensor.cuda()
    num_frames = rf.constant(6, dims=(), dtype="int32", device="cuda")

    def _search():
        with rf.set_default_device_ctx("cuda"), rf.set_static_traceable_ctx():
            return _beam_search_fixed_shape(log_probs, num_frames=num_frames, **dims)

    with ControlFlowOpWarmupDispatchMode():
        _search()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), CUDAGraphCaptureControlFlowOpDispatchMode():
        res = _search()
    for n in [0, 1, 4, 6]:
        num_frames.raw_tensor.fill_(n)
        graph.replay()
        torch.cuda.synchronize()
        num_frames_cpu = rf.constant(n, dims=(), dtype="int32", device="cpu")
        ref = _beam_search_fixed_shape(log_probs_cpu, num_frames=num_frames_cpu, **dims)
        _check_beam_search_same(res, ref, num_frames=n)
