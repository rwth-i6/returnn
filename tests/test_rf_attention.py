"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_dot_attention():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    key_dim = Dim(7, name="key")
    value_dim = Dim(13, name="value")
    extern_data = TensorDict(
        {
            "q": Tensor("q", [batch_dim, time_dim, key_dim], dtype="float32"),
            "k": Tensor("k", [batch_dim, time_dim, key_dim], dtype="float32"),
            "v": Tensor("v", [batch_dim, time_dim, value_dim], dtype="float32", feature_dim_axis=2),
        }
    )

    class _Net(rf.Module):
        def __call__(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
            kv_axis = Dim(None, name=f"kv-axis")
            k, _ = rf.replace_dim(k, in_dim=time_dim, out_dim=kv_axis)
            v, _ = rf.replace_dim(v, in_dim=time_dim, out_dim=kv_axis)
            return rf.dot_attention(q, k, v, axis=kv_axis, key_dim=key_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(q=extern_data["q"], k=extern_data["k"], v=extern_data["v"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, value_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_self_attention():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.self_att = rf.SelfAttention(
                in_dim=in_dim,
                proj_dim=Dim(5, name="out"),
                key_dim_total=Dim(21, name="key-dim-total"),
                value_dim_total=Dim(33, name="value-dim-total"),
                num_heads=3,
            )
            self.out_dim = self.self_att.out_dim

        def __call__(self, x: Tensor, *, axis: Dim) -> Tensor:
            """forward"""
            return self.self_att(x, axis=axis)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"], axis=time_dim)
        out.mark_as_default_output(shape=(batch_dim, time_dim, model.out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_causal_self_attention():
    from returnn.tensor import single_step_dim

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.self_att = rf.CausalSelfAttention(
                in_dim=in_dim,
                proj_dim=Dim(5, name="out"),
                key_dim_total=Dim(21, name="key-dim-total"),
                value_dim_total=Dim(33, name="value-dim-total"),
                num_heads=3,
            )
            self.out_dim = self.self_att.out_dim

        def __call__(self, x: Tensor, *, axis: Dim) -> Tensor:
            """forward"""

            def _body(_x: Tensor, _state: rf.State) -> Tuple[Tensor, rf.State]:
                _y, _state.self_att = self.self_att(_x, axis=single_step_dim, state=_state.self_att)
                return _y, _state

            y, _, _ = rf.scan(
                spatial_dim=axis,
                xs=x,
                body=_body,
                ys=Tensor("y", dims=[batch_dim, self.out_dim], dtype="float32"),
                initial=rf.State(self_att=self.self_att.default_initial_state(batch_dims=[batch_dim])),
            )
            return y

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"], axis=time_dim)
        out.mark_as_default_output(shape=(batch_dim, time_dim, model.out_dim))

    run_model(
        extern_data,
        lambda *, epoch, step: _Net(),
        _forward_step,
        # TF needs TensorArray unstack, not implemented yet
        test_tensorflow=False,
    )


def test_relative_positional_encoding():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(8, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor, *, axis: Dim) -> Tuple[Tensor, Dim]:
            x, dim = rf.relative_positional_encoding(
                key_value_spatial_dim=axis, query_spatial_dim=axis, feat_dim=in_dim
            )
            return x, dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, dim = model(extern_data["data"], axis=time_dim)
        out.mark_as_default_output(shape=(dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_rel_pos_self_attention():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(8, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __init__(self):
            super().__init__()
            self.self_att = rf.RelPosSelfAttention(
                in_dim=in_dim,
                proj_dim=Dim(5, name="out"),
                key_dim_total=Dim(21, name="key-dim-total"),
                value_dim_total=Dim(33, name="value-dim-total"),
                num_heads=3,
            )
            self.out_dim = self.self_att.out_dim

        def __call__(self, x: Tensor, *, axis: Dim) -> Tensor:
            """forward"""
            return self.self_att(x, axis=axis)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"], axis=time_dim)
        out.mark_as_default_output(shape=(batch_dim, time_dim, model.out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
