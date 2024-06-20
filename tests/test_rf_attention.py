"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import numpy.testing
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model, tf_scope


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


def test_self_attention_to_pure_torch():
    # test whether the torch and returnn implementation of the mhsa layer are equivalent
    import torch
    import returnn.frontend as rf
    from returnn.tensor import Dim

    # torch.backends.mha.set_fastpath_enabled(False)

    rf.select_backend_torch()
    rf.init_forward_step_run_ctx()
    rf.set_random_seed(1)

    batch_dim = Dim(3, name="batch")
    spatial_dim = Dim(11, name="spatial")
    out_dim = Dim(54, name="out")
    num_heads = 2

    random_opts = {"distribution": "normal", "dtype": "float32"}
    rf_input = rf.random(dims=[batch_dim, spatial_dim, out_dim], **random_opts)
    qkv_weight = rf.random(dims=[out_dim, 3 * out_dim], **random_opts)
    qkv_bias = rf.random(dims=[3 * out_dim], **random_opts)
    proj_weight = rf.random(dims=[out_dim.copy(match_priority=1), out_dim], **random_opts)
    proj_bias = rf.random(dims=[out_dim], **random_opts)

    rf_mhsa = rf.SelfAttention(
        in_dim=out_dim,
        proj_dim=out_dim,
        key_dim_total=out_dim,
        value_dim_total=out_dim,
        num_heads=num_heads,
    )
    rf_mhsa.qkv.weight.initial = qkv_weight
    rf_mhsa.qkv.bias.initial = qkv_bias
    rf_mhsa.proj.weight.initial = proj_weight
    rf_mhsa.proj.bias.initial = proj_bias

    torch_input = rf_input.raw_tensor
    torch_mhsa = torch.nn.MultiheadAttention(
        out_dim.dimension,
        num_heads,
        batch_first=True,
    )
    torch_mhsa.load_state_dict(
        {
            "in_proj_weight": qkv_weight.raw_tensor.reshape(
                out_dim.dimension, num_heads, 3, out_dim.dimension // num_heads
            )
            .permute(2, 1, 3, 0)
            .reshape(-1, out_dim.dimension),
            "in_proj_bias": qkv_bias.raw_tensor.reshape(num_heads, 3, out_dim.dimension // num_heads)
            .permute(1, 0, 2)
            .reshape(-1),
            "out_proj.weight": proj_weight.raw_tensor.reshape(
                num_heads, out_dim.dimension // num_heads, out_dim.dimension
            )
            .permute(2, 0, 1)
            .reshape(-1, out_dim.dimension),
            "out_proj.bias": proj_bias.raw_tensor,
        }
    )
    torch_mhsa.eval()

    rf_output = rf_mhsa(rf_input, axis=spatial_dim)
    torch_output, torch_attn_weights = torch_mhsa(torch_input, torch_input, torch_input, key_padding_mask=None)

    print("RF output")
    print(rf_output.raw_tensor)
    print(rf_output.raw_tensor.shape)
    print("---------------------------")
    print("Torch output")
    print(torch_output)
    print(torch_output.shape)

    torch.testing.assert_allclose(rf_output.raw_tensor, torch_output, atol=1e-3, rtol=1e-4)


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


def test_sinusoidal_positional_encoding():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    feat_dim = Dim(8, name="feat")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, feat_dim], dtype="float32"),
        }
    )

    def _forward_step(**_kwargs):
        out = rf.sinusoidal_positional_encoding(spatial_dim=time_dim, feat_dim=feat_dim)
        out.mark_as_default_output(shape=(time_dim, feat_dim))

    res = run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)

    from returnn.tf.util import basic as tf_util

    with tf_scope() as session:
        tf_ref = tf_util.get_positional_encoding(
            num_channels=feat_dim.dimension, length=res.data["output"].raw_tensor.shape[0]
        )
        tf_ref_v = session.run(tf_ref)

    np.testing.assert_almost_equal(res.data["output"].raw_tensor, tf_ref_v, decimal=5)


def test_CausalSelfAttention():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    feat_dim = Dim(8, name="feat")
    key_dim = Dim(6, name="key")
    value_dim = Dim(10, name="value")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, feat_dim], dtype="float32"),
        }
    )

    def _forward_step(*, model: rf.CausalSelfAttention, extern_data: TensorDict):
        data = extern_data["data"]
        data.mark_as_output("data", shape=[batch_dim, time_dim, feat_dim])
        time_dim.dyn_size_ext.mark_as_output("seq_len", shape=[batch_dim])
        out, _ = model(data, axis=time_dim)
        out.mark_as_default_output(shape=(batch_dim, time_dim, value_dim))
        model.qkv.weight.mark_as_output("qkv_weight", shape=[feat_dim, 2 * key_dim + value_dim])

    res = run_model(
        extern_data,
        lambda *, epoch, step: rf.CausalSelfAttention(
            in_dim=feat_dim,
            proj_dim=None,
            key_dim_total=key_dim,
            value_dim_total=value_dim,
            num_heads=2,
            with_bias=False,
        ),
        _forward_step,
        # Some problem with dimension tags currently in the TF-layers-dict backend...
        # Anyway, we compare to the TF SelfAttentionLayer with attention_left_only=True below.
        test_tensorflow=False,
    )

    extern_data.reset_content()

    with tf_scope() as session:
        from returnn.tf.network import TFNetwork, ExternData

        net_dict = {
            "self_att": {
                "class": "self_attention",
                "from": "data",
                "num_heads": 2,
                "total_key_dim": key_dim.dimension,
                "attention_left_only": True,
                "out_dim": value_dim,
                "is_output_layer": True,
            }
        }
        net = TFNetwork(
            extern_data=ExternData(
                {
                    "data": {
                        "dims": [batch_dim, time_dim, feat_dim],
                        "time_dim_axis": 1,
                        "feature_dim_axis": 2,
                        "dtype": "float32",
                        "version": 1,
                    }
                }
            )
        )
        net.construct_from_dict(net_dict)
        layer = net.get_default_output_layer()
        layer.params["QKV"].load(res.data["qkv_weight"].raw_tensor, session=session)
        out = layer.output.copy_transpose([batch_dim, time_dim, value_dim]).copy_masked(0.0)

        out_tf_v = session.run(
            out.raw_tensor,
            feed_dict={
                net.extern_data.data["data"].placeholder: res.data["data"].raw_tensor,
                net.extern_data.data["data"].dims[1].dyn_size_ext.raw_tensor: res.data["seq_len"].raw_tensor,
            },
        )
        numpy.testing.assert_almost_equal(res.data["output"].raw_tensor, out_tf_v, decimal=5)
