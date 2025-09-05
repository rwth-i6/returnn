"""
Testing returnn.frontend.decoder.transformer.
"""

from __future__ import annotations

import _setup_test_env  # noqa
from typing import Sequence, Tuple
import sys
import unittest
import torch
from returnn.util import better_exchook
from returnn.util.debug import PyTracer, check_py_traces_rf_to_pt_equal
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf


def _setup():
    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        pass


_setup()


def test_transformer_prefix_single_step():
    # We have some prefix we want to feed and then only get the logits of the last step,
    # and also continue step-wise (single_step_dim) from that state.

    from returnn.frontend.decoder.transformer import TransformerDecoder

    rf.select_backend_torch()

    vocab_dim = Dim(11, name="vocab")
    lm = TransformerDecoder(
        encoder_dim=None, vocab_dim=vocab_dim, model_dim=Dim(32, name="model"), num_layers=2, num_heads=2
    )

    # We currently need this for the case without batch dim.
    # We might want to extend the test case later in the future.
    state = lm.default_initial_state(batch_dims=[])

    # Not sure if a static dim is reasonable?
    # But for the single seq case (no batch dim), and eager mode (PyTorch), we know the size.
    spatial_dim = Dim(7, name="spatial")
    prefix_seq = rf.convert_to_tensor([3, 4, 5, 6, 7, 2, 1], dims=[spatial_dim], sparse_dim=vocab_dim)
    logits, state = lm(prefix_seq, spatial_dim=spatial_dim, state=state)
    print(logits, state["0"].self_att)
    assert state["0"].self_att.accum_axis == spatial_dim
    assert state["0"].self_att.accum_axis in state["0"].self_att.k_accum.dims

    # Now feed some other labels step by step.
    step = 1
    for label_idx in [8, 9, 10]:
        label = rf.constant(label_idx, dims=[], sparse_dim=vocab_dim)
        logits, state = lm(label, spatial_dim=single_step_dim, state=state)
        print(logits, state["0"].self_att)
        # This assumes it's a static dim.
        assert state["0"].self_att.accum_axis.dimension == spatial_dim.dimension + step
        assert state["0"].self_att.accum_axis in state["0"].self_att.k_accum.dims
        step += 1


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


def test_transformer_decoder_time_sync_search():
    """
    Adapted and simplified from
    :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc.model_recog_with_recomb`.

    This is about a bug happening in nested masked_scatter
    and/or RotaryPosCausalSelfAttention and its usage of sinusoidal_positional_encoding,
    but we leave also the encoder part with cross attention to cover more logic for the test.
    """
    from returnn.frontend.decoder.transformer import TransformerDecoder
    from returnn.frontend.tensor_array import TensorArray

    rf.select_backend_torch()

    target_dim = Dim(11, name="vocab")
    wb_target_dim = target_dim + 1
    blank_idx = target_dim.dimension  # last index
    eos_idx = bos_idx = 0
    decoder = TransformerDecoder(
        encoder_dim=wb_target_dim,
        vocab_dim=target_dim,
        model_dim=Dim(32, name="model"),
        num_layers=2,
        num_heads=2,
        # Transformer++ / Llama-like
        norm=rf.build_dict(rf.RMSNorm),
        ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
        layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    )

    batch_dim = Dim(3, name="batch")
    enc_spatial_dim = Dim(rf.convert_to_tensor([7, 5, 6], dims=[batch_dim]), name="enc")
    data = rf.random_normal([batch_dim, enc_spatial_dim, wb_target_dim])  # used both as encoder out and CTC logits
    enc = decoder.transform_encoder(data, axis=enc_spatial_dim)

    beam_size = 3
    recomb = "max"  # None, "max", "sum"

    batch_dims = [batch_dim]

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    neg_inf = float("-inf")
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    ctc_logits = data
    ctc_label_log_prob = rf.log_softmax(ctc_logits, axis=wb_target_dim)  # Batch, Spatial, VocabWB
    # No CTC scale needed.
    ctc_label_log_prob_ta = TensorArray.unstack(ctc_label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(bos_idx, dims=batch_dims_, sparse_dim=target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(blank_idx, dims=batch_dims_, sparse_dim=wb_target_dim)  # Batch, InBeam -> VocabWB

    seq_label = _seq_label_history_init_state(vocab_dim=target_dim, batch_dims=batch_dims_)

    decoder_state = decoder.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    decoder_logits, decoder_state = decoder(
        target,
        encoder=enc,
        spatial_dim=single_step_dim,
        state=decoder_state,
    )  # Batch, InBeam, Vocab / ...
    decoder_log_probs = rf.log_softmax(decoder_logits, axis=target_dim)  # Batch, InBeam, Vocab

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + ctc_label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        if decoder is not None:
            # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
            seq_log_prob += rf.where(
                (prev_target_wb == blank_idx) | (prev_target_wb != rf.range_over_dim(wb_target_dim)),
                _target_dense_extend_blank(
                    decoder_log_probs,
                    target_dim=target_dim,
                    wb_target_dim=wb_target_dim,
                    blank_idx=blank_idx,
                    value=0.0,
                ),
                0.0,
            )  # Batch, InBeam, VocabWB

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        if decoder is not None:
            decoder_log_probs = rf.gather(decoder_log_probs, indices=backrefs)  # Batch, Beam, Vocab
            decoder_state = rf.nested.gather_nested(decoder_state, indices=backrefs)
        seq_label = rf.nested.gather_nested(seq_label, indices=backrefs)

        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB

        got_new_label: Tensor = (target_wb != blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(target_wb, target_dim=target_dim, wb_target_dim=wb_target_dim, blank_idx=blank_idx),
            prev_target,
        )  # Batch, Beam -> Vocab
        got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
        if got_new_label_cpu.raw_tensor.sum().item() > 0:
            seq_label = rf.nested.mask_nested(
                _seq_label_append(seq_label, target),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                mask_value=seq_label,
            )

            # Recombine paths with the same label seq.
            if not recomb:
                pass
            elif recomb in ("max", "sum"):
                # Set seq_log_prob for batch entries to neg_inf if they have the same label seq.
                same_seq_labels, beam_dual_dim = _same_seq_labels(
                    seq_label.history, spatial_dim=seq_label.hist_dim, beam_dim=beam_dim
                )
                seq_log_prob_ext = rf.where(
                    same_seq_labels, rf.replace_dim_v2(seq_log_prob, in_dim=beam_dim, out_dim=beam_dual_dim), neg_inf
                )  # Batch, Beam, BeamDual
                if recomb == "sum":
                    seq_log_prob = rf.reduce_logsumexp(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam
                argmax_seq_log_prob = rf.reduce_argmax(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam -> BeamDual
                mask = argmax_seq_log_prob == rf.range_over_dim(beam_dim)  # Batch, Beam -> 0|1
                seq_log_prob = rf.where(mask, seq_log_prob, neg_inf)
                got_new_label = got_new_label & mask  # don't re-eval the LM when masked out
                got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
            else:
                raise ValueError(f"invalid recog_recomb {recomb!r}")

        if decoder is not None and got_new_label_cpu.raw_tensor.sum().item() > 0:
            (target_, decoder_state_, enc_), packed_new_label_dim, packed_new_label_dim_map = (
                rf.nested.masked_select_nested(
                    (target, decoder_state, enc),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    dims=batch_dims + [beam_dim],
                )
            )
            # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
            assert packed_new_label_dim.get_dim_value() > 0

            decoder_logits_, decoder_state_ = decoder(
                target_,
                encoder=enc_,
                spatial_dim=single_step_dim,
                state=decoder_state_,
            )  # Flat_Batch_Beam, Vocab / ...
            decoder_log_probs_ = rf.log_softmax(decoder_logits_, axis=target_dim)  # Flat_Batch_Beam, Vocab

            decoder_log_probs, decoder_state = rf.nested.masked_scatter_nested(
                (decoder_log_probs_, decoder_state_),
                (decoder_log_probs, decoder_state),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                masked_select_dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...

    if decoder is not None:
        # seq_log_prob, lm_log_probs: Batch, Beam
        # Add LM EOS score at the end.
        decoder_eos_score = rf.gather(decoder_log_probs, indices=eos_idx, axis=target_dim)
        seq_log_prob += decoder_eos_score  # Batch, Beam -> VocabWB

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)

    # Select valid.
    mask = rf.is_finite(seq_log_prob)  # Batch, Beam
    mask_cpu = rf.copy_to_device(mask, "cpu")
    (seq_targets_wb, seq_log_prob, out_spatial_dim), beam_dim, _ = rf.nested.masked_select_nested(
        (seq_targets_wb, seq_log_prob, out_spatial_dim), mask=mask, mask_cpu=mask_cpu, dims=[beam_dim]
    )

    print("result:", seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim)


def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == wb_target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, target_dim)


def _target_dense_extend_blank(
    target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
) -> Tensor:
    assert target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
    return res


def _seq_label_history_init_state(*, vocab_dim: Dim, batch_dims: Sequence[Dim]) -> rf.State:
    hist_dim = Dim(0, name="hist0")
    history = rf.zeros(list(batch_dims) + [hist_dim], dtype="int64", sparse_dim=vocab_dim)
    return rf.State(hist_dim=hist_dim, history=history)


def _seq_label_append(state: rf.State, new_label: Tensor) -> rf.State:
    hist_dim: Dim = state.hist_dim
    new_history, new_hist_dim = rf.cum_concat_step(new_label, prev_accum=state.history, axis=hist_dim)
    return rf.State(hist_dim=new_hist_dim, history=new_history)


def _same_seq_labels(seq: Tensor, *, spatial_dim: Dim, beam_dim: Dim) -> Tuple[Tensor, Dim]:
    seq_label_dual, beam_dual_dim = rf.replace_dim(seq, in_dim=beam_dim)
    same_seq_labels = rf.compare_bc(seq, "==", seq_label_dual)  # Batch, Beam, BeamDual, Spatial
    same_seq_labels = rf.reduce_all(same_seq_labels, axis=spatial_dim)  # Batch, Beam, BeamDual
    if beam_dim in spatial_dim.get_size_tensor().dims:
        seq_labels_lens = spatial_dim.get_size_tensor(device=same_seq_labels.device)
        seq_labels_dual_lens = rf.replace_dim_v2(
            seq_labels_lens, in_dim=beam_dim, out_dim=beam_dual_dim
        )  # Batch, BeamDual
        same_seq_labels_lens = rf.compare_bc(seq_labels_lens, "==", seq_labels_dual_lens)  # Batch, Beam, BeamDual
        same_seq_labels = rf.logical_and(same_seq_labels, same_seq_labels_lens)
    return same_seq_labels, beam_dual_dim


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
