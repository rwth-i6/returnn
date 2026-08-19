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
        hist, hist_dim = rf.nested.mask_nested(
            (new_hist, new_hist_dim), mask=target != 0, mask_value=(hist, hist_dim)
        )
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
