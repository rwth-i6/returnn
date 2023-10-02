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
            def _cond(s: Tuple[Tensor, Tensor]):
                t, s_ = s
                if t.raw_tensor.__class__.__module__.startswith("torch"):
                    print("**", t.raw_tensor, rf.reduce_sum(s_, axis=s_.dims).raw_tensor)
                return rf.logical_and(rf.reduce_sum(s_, axis=s_.dims) < 50, t < time_dim.get_dim_value_tensor())

            def _body(s):
                t, s_ = s
                return t + 1, s_ + rf.abs(rf.gather(x, indices=t, axis=time_dim))

            _, final_s = rf.while_loop(
                _cond,
                _body,
                initial=(rf.zeros((), dtype=rf.get_default_array_index_dtype()), rf.zeros((batch_dim, in_dim))),
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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_tensorflow=False)
