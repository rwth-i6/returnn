"""
Testing returnn.frontend.decoder.transformer.
"""

from __future__ import annotations

import _setup_test_env  # noqa
import torch
from returnn.util.debug import PyTracer, check_py_traces_rf_to_pt_equal
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def _setup():
    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        pass


_setup()


def test_llama():
    """
    Test that we can reproduce the Llama model.

    This here is the final complete test.
    There are several other sub-tests:

    - :func:`test_rotary_embedding`
    - :func:`test_rope_causal_self_att`

    Some references for the whole Llama model:
    https://github.com/meta-llama/llama/blob/main/llama/model.py
    https://github.com/meta-llama/llama3/blob/main/llama/model.py
    https://github.com/karpathy/llama2.c/blob/master/model.py
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
    https://github.com/hkproj/pytorch-llama/blob/main/model.py
    https://github.com/likejazz/llama3.np/blob/main/llama3.py
    """
    from returnn.frontend.decoder.transformer import TransformerDecoder, TransformerDecoderLayer, FeedForwardGated
    from returnn.frontend.conversions.hf_llama import import_params_hf_llama_to_rf_transformer_decoder
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel, LlamaConfig

    config = LlamaConfig(
        vocab_size=11,
        hidden_size=64,
        intermediate_size=64 * 4,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
    )

    model_hf = LlamaForCausalLM(config)
    print("HF Model:")
    print(model_hf)
    print("Parameters:")
    num_params = 0
    for k, v in model_hf.named_parameters():
        print(f"{k}: {list(v.shape)} {v.dtype}")
        num_params += v.numel()
    print("Total number of parameters:", num_params)

    rf.select_backend_torch()

    model_dim = Dim(config.hidden_size, name="model")
    model_rf = TransformerDecoder(
        encoder_dim=None,
        vocab_dim=Dim(config.vocab_size, name="vocab"),
        model_dim=model_dim,
        num_layers=config.num_hidden_layers,
        pos_enc=None,
        norm=rf.RMSNorm,
        ff=FeedForwardGated,
        share_embedding=False,
        input_embedding_scale=1.0,
        decoder_layer_opts=dict(self_att=rf.RotaryPosCausalSelfAttention, self_att_opts=dict(with_bias=False)),
        num_heads=config.num_attention_heads,
        dropout=0,
        att_dropout=0,
    )
    print("RF Model:")
    print(model_rf)
    print("Parameters:")
    num_params = 0
    for k, v in model_rf.named_parameters():
        print(f"{k}: {list(v.dims)} {v.dtype}")
        num_params += v.num_elements()
    print("Total number of parameters:", num_params)

    import_params_hf_llama_to_rf_transformer_decoder(model_hf, model_rf)

    batch_dim = Dim(3, name="batch")
    seq_dim = Dim(rf.random_uniform([batch_dim], minval=7, maxval=13, dtype="int32"), name="seq")
    in_ = rf.random_uniform([batch_dim, seq_dim], sparse_dim=model_rf.vocab_dim)
    in_.name = "input_labels"

    with PyTracer([TransformerDecoder.__call__, TransformerDecoderLayer.__call__], Tensor) as trace_rf:
        out_rf, _ = model_rf(in_, spatial_dim=seq_dim, state=model_rf.default_initial_state(batch_dims=[batch_dim]))

    mask = rf.sequence_mask([batch_dim, seq_dim])
    with PyTracer([LlamaForCausalLM.forward, LlamaModel.forward, LlamaDecoderLayer.forward], torch.Tensor) as trace_hf:
        out_hf = model_hf(in_.raw_tensor, attention_mask=mask.raw_tensor)

    check_py_traces_rf_to_pt_equal(
        trace_rf.captured_locals,
        trace_hf.captured_locals,
        [
            (
                (TransformerDecoder.__call__, 0, "decoded", 0),
                (LlamaModel.forward, 0, "inputs_embeds", 0),
                (batch_dim, seq_dim, model_dim),
            ),
        ],
    )

    print("Check...")
    assert out_rf.raw_tensor.shape == out_hf.logits.shape
    torch.testing.assert_allclose(out_rf.raw_tensor, out_hf.logits)
    print("  all matched!")


def test_feed_forward_gated():
    from returnn.frontend.decoder.transformer import FeedForwardGated
    from returnn.frontend.conversions.hf_llama import import_params_hf_llama_mlp_to_rf_feed_forward_gated
    from transformers.models.llama.modeling_llama import LlamaMLP, LlamaConfig

    config = LlamaConfig(
        vocab_size=11,
        hidden_size=64,
        intermediate_size=64 * 4,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
    )

    model_hf = LlamaMLP(config)

    rf.select_backend_torch()
    rf.set_random_seed(42)

    model_dim = Dim(config.hidden_size, name="model")
    model_rf = FeedForwardGated(out_dim=model_dim, ff_dim=Dim(config.intermediate_size, name="inter"), dropout=0.0)

    import_params_hf_llama_mlp_to_rf_feed_forward_gated(model_hf, model_rf)

    batch_dim = Dim(3, name="batch")
    seq_dim = Dim(rf.random_uniform([batch_dim], minval=7, maxval=13, dtype="int32"), name="seq")
    in_ = rf.random_uniform([batch_dim, seq_dim, model_dim])
    in_.name = "input"

    out_rf = model_rf(in_)
    out_rf = out_rf.copy_transpose((batch_dim, seq_dim, model_dim))

    out_hf = model_hf(in_.raw_tensor)

    print("Check...")
    assert out_rf.raw_tensor.shape == out_hf.shape
    torch.testing.assert_allclose(out_rf.raw_tensor, out_hf)
    print("  all matched!")
