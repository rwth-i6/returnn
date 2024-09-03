"""
Testing returnn.frontend.decoder.transformer.
"""

from __future__ import annotations

import _setup_test_env  # noqa
import sys
import unittest
import torch
from returnn.util import better_exchook
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


def test_transformer_rel_pos_att():
    """
    This tests that TransformerDecoder together with RelPosCausalSelfAttention
    and FeedForwardGated works in a reasonable standard setup.
    Works = does not cause exceptions.

    Additionally, we test an issue that dim tags seems to be leaking.
    """
    from returnn.tensor import TensorDict, batch_dim
    from returnn.frontend.decoder.transformer import TransformerDecoder, FeedForwardGated
    from returnn.datasets.util.vocabulary import Vocabulary
    from returnn.torch.data.extern_data import raw_dict_to_extern_data

    rf.select_backend_torch()

    vocab = Vocabulary.create_vocab_from_labels(
        [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"], eos_label=0, bos_label=0
    )
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None
    vocab_dim = Dim(vocab.num_labels, name="vocab", vocab=vocab)

    model_def = rf.build_dict(
        TransformerDecoder,
        encoder_dim=None,
        num_layers=2,  # with higher number of layers, probelm occurs more, but also with 2
        model_dim=20,
        num_heads=2,
        pos_enc=None,
        norm=rf.build_dict(rf.RMSNorm),
        ff=rf.build_dict(FeedForwardGated),
        decoder_layer_opts=dict(self_att=rf.build_dict(rf.RelPosCausalSelfAttention, with_bias=False)),
        dropout=0.0,
        att_dropout=0.0,
    )
    model = rf.build_from_dict(model_def, vocab_dim=vocab_dim)
    assert isinstance(model, TransformerDecoder)

    leakages = []

    # Adapted from Dim reset_raw.
    def _num_referenced_dim_tags(self: Dim) -> int:
        visited = set()  # ids
        queue = [self]
        while queue:
            # noinspection PyShadowingNames
            dim: Dim = queue.pop()
            if id(dim) in visited:
                continue
            visited.add(id(dim))
            # noinspection PyProtectedMember
            dim_extra = dim._extra
            if dim_extra:
                # Any dims via dim math could also contain raw tensors,
                # so iterate through them.
                print("Dim:", dim)
                print(" cache_dim_math:", dim_extra.cache_dim_math)
                print(" same_as:", dim_extra.same_as)
                print(" copy_same_as:", dim_extra.copy_same_as)
                print(" same_for_batch_ctx:", dim_extra.same_for_batch_ctx)
                queue += dim_extra.cache_dim_math.values()
                if dim_extra.same_as:
                    queue.append(dim_extra.same_as)
                if dim_extra.copy_same_as:
                    queue.append(dim_extra.copy_same_as)
                queue += dim_extra.same_for_batch_ctx.values()
        print(f"{self} _num_referenced_dim_tags (reset_raw), visited {len(visited)}")
        return len(visited)

    time_dim = Dim(None, name="time")
    extern_data_template = TensorDict([Tensor("data", (batch_dim, time_dim), "int32", sparse_dim=vocab_dim)])

    prev_step_num_tags = 0
    for step in range(10):
        print("Step:", step)
        rf.init_train_step_run_ctx(train_flag=False, step=step)

        # Check that we don't have any dim tags leaking.
        # Do that right after init_train_step_run_ctx, because that might clean some previous caches.
        step_num_tags = _num_referenced_dim_tags(time_dim)
        if step > 1 and step_num_tags > prev_step_num_tags:
            leakages.append(step_num_tags - prev_step_num_tags)
        prev_step_num_tags = step_num_tags

        seq_lens = torch.randint(5, 11, (3,), dtype=torch.int32)
        extern_data = raw_dict_to_extern_data(
            {"data": torch.randint(0, vocab_dim.dimension, (3, seq_lens.max())), "data:seq_len": seq_lens},
            extern_data_template=extern_data_template,
            device="cpu",
        )

        targets = extern_data["data"]
        targets_spatial_dim = time_dim
        input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
        )
        targets_w_eos, _ = rf.pad(
            targets,
            axes=[targets_spatial_dim],
            padding=[(0, 1)],
            value=vocab.eos_label_id,
            out_dims=[targets_w_eos_spatial_dim],
        )

        batch_dims = [batch_dim]

        # Gradients not relevant for this test.
        with torch.no_grad():
            logits, _ = model(
                input_labels,
                spatial_dim=targets_w_eos_spatial_dim,
                encoder=None,
                state=model.default_initial_state(batch_dims=batch_dims),
            )

            logits_packed, pack_dim = rf.pack_padded(
                logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
            )
            targets_packed, _ = rf.pack_padded(
                targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
            )

            log_prob = rf.log_softmax(logits_packed, axis=model.vocab_dim)
            # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
            loss = rf.cross_entropy(
                target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.vocab_dim
            )
            loss.mark_as_loss("ce", use_normalized_loss=True)

            best = rf.reduce_argmax(logits_packed, axis=model.vocab_dim)
            frame_error = best != targets_packed
            frame_error.mark_as_loss(name="fer", as_error=True)

    assert not leakages, f"Leakages: {leakages}"


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
