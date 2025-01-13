"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Union, Tuple
import numpy as np
import numpy.testing
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model, tf_scope


def _setup():
    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        pass


_setup()


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


def test_rotary_embedding():
    import torch
    from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding

    config = LlamaConfig(
        vocab_size=11,
        hidden_size=64,
        intermediate_size=64 * 4,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
    )
    model_hf = LlamaRotaryEmbedding(config=config)

    rf.select_backend_torch()
    rf.set_random_seed(42)

    batch_dim = Dim(3, name="batch")
    seq_dim = Dim(rf.random_uniform([batch_dim], minval=7, maxval=13, dtype="int32"), name="seq")
    model_head_dim = Dim(config.hidden_size // config.num_attention_heads, name="model_head")

    # base is a bit different in rf.sinusoidal_positional_encoding (like the original)
    # vs how it's used for RoPE.
    # log(base) / (dim / 2 - 1) = log(10_000) * 2 / dim
    # <=> log(base) = log(10_000) * (dim / 2 - 1) * 2 / dim = log(10_000) * (1 - 2 / dim)
    # <=> base = 10_000 ** (1 - 2 / dim)
    out_rf = rf.sinusoidal_positional_encoding(
        spatial_dim=seq_dim, feat_dim=model_head_dim, base=10_000 ** (1 - 2 / model_head_dim.dimension)
    )
    print("out_rf:", out_rf)
    # For the comparison below.
    out_rf = rf.expand_dim(out_rf, batch_dim)
    out_rf = out_rf.copy_transpose((batch_dim, seq_dim, model_head_dim))
    # Split the sin/cos for the comparison below.
    out_rf_sin, out_rf_cos = rf.split(out_rf, axis=model_head_dim, out_dims=[model_head_dim.div_left(2)] * 2)
    print("out_rf':", out_rf_sin, out_rf_cos)

    position_ids = rf.expand_dim(rf.range_over_dim(seq_dim), batch_dim)  # LlamaRotaryEmbedding wants this
    out_hf_cos, out_hf_sin = model_hf(
        torch.zeros(()), position_ids=position_ids.copy_compatible_to_dims_raw((batch_dim, seq_dim))
    )
    print("out_hf:", out_hf_sin, out_hf_cos)
    # The values are just repeated. Cut off the second half for comparison.
    out_hf_cos = out_hf_cos[:, :, : out_hf_cos.shape[-1] // 2]
    out_hf_sin = out_hf_sin[:, :, : out_hf_sin.shape[-1] // 2]
    print("out_hf':", out_hf_sin, out_hf_cos)

    assert out_rf_sin.raw_tensor.shape == out_hf_sin.shape
    assert out_rf_cos.raw_tensor.shape == out_hf_cos.shape
    torch.testing.assert_allclose(out_rf_sin.raw_tensor, out_hf_sin)
    torch.testing.assert_allclose(out_rf_cos.raw_tensor, out_hf_cos)


def test_rope_causal_self_att():
    import torch
    from returnn.util.pprint import pprint
    from returnn.util.debug import PyTracer, check_py_traces_rf_to_pt_equal

    # noinspection PyProtectedMember
    from returnn.frontend.attention import _apply_rope as rf_apply_rope
    from returnn.frontend.conversions.hf_llama import import_params_hf_llama_att_to_rf_rotary_att

    from transformers.models.llama.modeling_llama import (
        LlamaAttention,
        LlamaRotaryEmbedding,
        LlamaConfig,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    config = LlamaConfig(
        vocab_size=11,
        hidden_size=64,
        intermediate_size=64 * 4,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
    )

    model_hf = LlamaAttention(config, layer_idx=0)

    rf.select_backend_torch()
    rf.set_random_seed(42)

    model_dim = Dim(config.hidden_size, name="model")
    model_rf = rf.RotaryPosCausalSelfAttention(
        in_dim=model_dim,
        proj_dim=model_dim,
        num_heads=config.num_attention_heads,
        key_dim_total=model_dim,
        value_dim_total=model_dim,
        with_bias=False,
    )
    import_params_hf_llama_att_to_rf_rotary_att(model_hf, model_rf)

    batch_dim = Dim(3, name="batch")
    seq_dim = Dim(rf.random_uniform([batch_dim], minval=7, maxval=13, dtype="int32"), name="seq")
    in_ = rf.random_uniform([batch_dim, seq_dim, model_dim])
    in_.name = "input"

    with PyTracer(
        [rf.RotaryPosCausalSelfAttention.__call__, rf.sinusoidal_positional_encoding, rf.dot_attention, rf_apply_rope],
        (Tensor, Dim),
    ) as trace_rf:
        out_rf, _ = model_rf(in_, axis=seq_dim, state=model_rf.default_initial_state(batch_dims=[batch_dim]))
        out_rf = out_rf.copy_transpose((batch_dim, seq_dim, model_dim))
    pprint(trace_rf.captured_locals)

    position_ids = rf.expand_dim(rf.range_over_dim(seq_dim), batch_dim)  # LlamaRotaryEmbedding wants this
    with PyTracer(
        [LlamaAttention.forward, LlamaRotaryEmbedding.forward, apply_rotary_pos_emb, eager_attention_forward],
        torch.Tensor,
    ) as trace_hf:
        # causal_mask code copied from LlamaAttention
        sequence_length = target_length = in_.raw_tensor.shape[1]
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=torch.finfo(in_.raw_tensor.dtype).min,
            dtype=in_.raw_tensor.dtype,
            device=in_.raw_tensor.device,
        )
        causal_mask = torch.triu(causal_mask, diagonal=1)
        cache_position = torch.arange(0, in_.raw_tensor.shape[1], device=in_.raw_tensor.device)
        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(in_.raw_tensor.shape[0], 1, -1, -1)
        rotary_emb = LlamaRotaryEmbedding(config=config)
        position_embeddings = rotary_emb(
            torch.zeros(()), position_ids=position_ids.copy_compatible_to_dims_raw((batch_dim, seq_dim))
        )
        out_hf, *_ = model_hf(in_.raw_tensor, attention_mask=causal_mask, position_embeddings=position_embeddings)
    pprint(trace_hf.captured_locals)

    print("First HF att weight tensor:")
    print(trace_hf.captured_locals[LlamaAttention.forward][0]["attn_weights"][-1][0, 0, 0].detach().numpy())

    check_py_traces_rf_to_pt_equal(
        trace_rf.captured_locals,
        trace_hf.captured_locals,
        [
            (
                (rf.RotaryPosCausalSelfAttention.__call__, 0, "q", 0),
                # input: batch_dim, seq_dim, model_dim
                # input_shape: batch_dim, seq_dim
                # HF query_states': (batch_dim, seq_dim, num_heads, self.head_dim),
                #   then transposed to (batch_dim, num_heads, seq_dim, self.head_dim)
                (LlamaAttention.forward, 0, "query_states", 0),
                lambda x, *, name, **_: rf.convert_to_tensor(
                    # reorder complex numbers
                    x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).flatten(-2),
                    dims=(batch_dim, model_rf.num_heads, seq_dim, model_rf.key_dim_per_head),
                    name=name,
                ),
            ),
            (
                (rf.RotaryPosCausalSelfAttention.__call__, 0, "k", 0),
                (LlamaAttention.forward, 0, "key_states", 0),
                lambda x, *, name, **_: rf.convert_to_tensor(
                    # reorder complex numbers
                    x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).flatten(-2),
                    dims=(batch_dim, model_rf.num_heads, seq_dim, model_rf.key_dim_per_head),
                    name=name,
                ),
            ),
            (
                (rf.sinusoidal_positional_encoding, 0, "div_term", 0),
                (LlamaRotaryEmbedding.forward, 0, "inv_freq_expanded", 0),
                lambda x, *, name, **_: rf.convert_to_tensor(
                    x[0, :, 0], dims=[model_rf.key_dim_per_head.div_left(2)], name=name
                ),
            ),
            (
                (rf.sinusoidal_positional_encoding, 0, "arg_sin", 0),
                (LlamaRotaryEmbedding.forward, 0, "freqs", 0),
                lambda x, *, name, resolve_dim, **_: rf.convert_to_tensor(
                    x[0],
                    dims=(resolve_dim("spatial_dim"), model_rf.key_dim_per_head.div_left(2)),
                    name=name,
                ),
            ),
            (
                (rf_apply_rope, 0, "pe_imag", 0),
                (apply_rotary_pos_emb, 0, "sin", 0),
                lambda x, *, name, **_: rf.convert_to_tensor(
                    x[0, :, : x.shape[2] // 2], dims=(seq_dim, model_rf.key_dim_per_head.div_left(2)), name=name
                ),
            ),
            (
                (rf_apply_rope, 0, "pe_imag", 0),
                (apply_rotary_pos_emb, 0, "sin", 0),
                lambda x, *, name, **_: rf.convert_to_tensor(
                    x[-1, :, x.shape[2] // 2 :], dims=(seq_dim, model_rf.key_dim_per_head.div_left(2)), name=name
                ),
            ),
            (
                (rf_apply_rope, 0, "pe_real", 0),
                (apply_rotary_pos_emb, 0, "cos", 0),
                lambda x, *, name, **_: rf.convert_to_tensor(
                    x[0, :, : x.shape[2] // 2],
                    dims=(seq_dim, model_rf.key_dim_per_head.div_left(2)),
                    name=name,
                ),
            ),
            (
                (rf.RotaryPosCausalSelfAttention.__call__, 0, "q", -1),
                (LlamaAttention.forward, 0, "query_states", -1),
                lambda x, *, name, **_: rf.convert_to_tensor(
                    x.reshape(*x.shape[:-1], 2, -1).transpose(-1, -2).flatten(-2),
                    dims=(batch_dim, model_rf.num_heads, seq_dim, model_rf.key_dim_per_head),
                    name=name,
                ),
            ),
            (
                (rf.dot_attention, 0, "energy", 0),
                (eager_attention_forward, 0, "attn_weights", 0),
                (batch_dim, model_rf.num_heads, seq_dim, "axis"),
            ),
            (
                (rf.dot_attention, 0, "att_weights", 0),
                (LlamaAttention.forward, 0, "attn_weights", -1),
                (batch_dim, model_rf.num_heads, seq_dim, "axis"),
            ),
            (
                (rf.dot_attention, 0, "att", 0),
                (LlamaAttention.forward, 0, "attn_output", 0),
                (batch_dim, seq_dim, model_rf.num_heads, model_rf.value_dim_per_head),
            ),
        ],
    )

    print("Final check...")
    assert out_rf.raw_tensor.shape == out_hf.shape
    torch.testing.assert_close(out_rf.raw_tensor, out_hf)
    print("  all matched!")


def test_causal_self_att_variants_single_step_vs_full_seq():
    from returnn.tensor import single_step_dim

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7 * 2, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: Union[rf.CausalSelfAttention], extern_data: TensorDict):
        x = extern_data["data"]

        out_seq_level, _ = model(x, axis=time_dim)
        out_seq_level.mark_as_output("out_seq_level", shape=[batch_dim, time_dim, model.out_dim])

        out_seq_level_explicit_initial_state, _ = model(
            x, axis=time_dim, state=model.default_initial_state(batch_dims=[batch_dim])
        )
        out_seq_level_explicit_initial_state.mark_as_output(
            "out_seq_level_explicit_initial_state", shape=[batch_dim, time_dim, model.out_dim]
        )

        def _body(
            _x: Tensor, _state: Union[rf.CausalSelfAttentionState]
        ) -> Tuple[Tensor, Union[rf.CausalSelfAttentionState]]:
            return model(_x, axis=single_step_dim, state=_state)

        out_single_steps, _, _ = rf.scan(
            spatial_dim=time_dim,
            xs=x,
            body=_body,
            ys=Tensor("y", dims=[batch_dim, model.out_dim], dtype="float32"),
            initial=model.default_initial_state(batch_dims=[batch_dim]),
        )
        out_single_steps.mark_as_output("out_single_steps", shape=[batch_dim, time_dim, model.out_dim])

    common_opts = dict(
        in_dim=in_dim,
        proj_dim=Dim(5, name="out"),
        key_dim_total=Dim(21 * 2, name="key-dim-total"),
        value_dim_total=Dim(33, name="value-dim-total"),
        num_heads=3,
    )

    def _make_causal_self_att(**_kwargs):
        return rf.CausalSelfAttention(**common_opts)

    def _make_rope_causal_self_att(**_kwargs):
        return rf.RotaryPosCausalSelfAttention(**common_opts)

    def _make_rel_pos_causal_self_att(**_kwargs):
        return rf.RelPosCausalSelfAttention(**common_opts)

    models = [_make_causal_self_att, _make_rope_causal_self_att, _make_rel_pos_causal_self_att]

    for get_model in models:
        print("> Testing model:", get_model.__name__)
        res = run_model(
            extern_data,
            get_model,
            _forward_step,
            # TF needs TensorArray unstack, not implemented yet
            test_tensorflow=False,
        )

        # Check that the single-step and the seq-level output are the same.
        res_seq_level = res.data["out_seq_level"].raw_tensor
        for key in ["out_seq_level_explicit_initial_state", "out_single_steps"]:
            res_other = res.data[key].raw_tensor
            assert res_seq_level.shape == res_other.shape
            numpy.testing.assert_allclose(
                res_other, res_seq_level, atol=1e-5, rtol=1e-5, err_msg=f"output {key} differs"
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


def test_relative_positional_encoding_cross():
    enc_spatial_dim = Dim(Tensor("enc_spatial", [batch_dim], dtype="int32"))
    dec_spatial_dim = Dim(Tensor("dec_spatial", [batch_dim], dtype="int32"))
    in_dim = Dim(8, name="in")
    extern_data = TensorDict(
        {
            "enc": Tensor("enc", [batch_dim, enc_spatial_dim, in_dim], dtype="float32"),
            "dec": Tensor("dec", [batch_dim, dec_spatial_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(**_kwargs):
        out, dim = rf.relative_positional_encoding(
            key_value_spatial_dim=enc_spatial_dim, query_spatial_dim=dec_spatial_dim, feat_dim=in_dim
        )
        out.mark_as_default_output(shape=(dim, in_dim))

    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step)


def test_rel_pos_self_attention():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(8, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )
    check_batching = False

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
            nonlocal check_batching
            if check_batching:
                assert rf.is_executing_eagerly()
                assert batch_dim in x.dims and axis != batch_dim
                y = self.self_att(x, axis=axis)
                for b in range(batch_dim.get_dim_value()):
                    x_b = rf.gather(x, axis=batch_dim, indices=b)
                    assert batch_dim in axis.dyn_size_ext.dims  # current assumption...
                    seq_len = rf.gather(axis.dyn_size_ext, axis=batch_dim, indices=b)
                    axis_b = Dim(seq_len)
                    # Note: The current order (replace_dim and then slice) is somewhat dependent
                    # on the current internal behavior of gather and replace_dim,
                    # which might change at some point...
                    x_b, _ = rf.replace_dim(x_b, in_dim=axis, out_dim=axis_b)
                    x_b, _ = rf.slice(x_b, axis=axis_b, start=0, end=seq_len, out_dim=axis_b)
                    y_b = self.self_att(x_b, axis=axis_b)
                    y_b_ = rf.gather(y, axis=batch_dim, indices=b)
                    y_b_, _ = rf.replace_dim(y_b_, in_dim=axis, out_dim=axis_b)
                    y_b_, _ = rf.slice(y_b_, axis=axis_b, start=0, end=seq_len, out_dim=axis_b)
                    y_b_ = y_b_.copy_transpose(y_b.dims)
                    # Assuming PyTorch...
                    np.testing.assert_almost_equal(
                        y_b.raw_tensor.cpu().detach().numpy(), y_b_.raw_tensor.cpu().detach().numpy(), decimal=5
                    )
                return y

            return self.self_att(x, axis=axis)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"], axis=time_dim)
        out.mark_as_default_output(shape=(batch_dim, time_dim, model.out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
    check_batching = True
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_tensorflow=False)


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
