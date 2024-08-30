"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
from typing import Sequence
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def _setup_lovely_tensors():
    try:
        import lovely_tensors
    except ImportError:
        return
    lovely_tensors.monkey_patch()


_setup_lovely_tensors()


def test_conformer():
    # This test needs a huge stack size currently, due to the way RETURNN layer construction works currently.
    # On RETURNN side, there is the option flat_net_construction to solve this,
    # however, it's experimental and also does not work for this case.
    # https://github.com/rwth-i6/returnn/issues/957
    # https://stackoverflow.com/a/16248113/133374
    import resource
    import sys

    try:
        resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
    except Exception as exc:
        print(f"resource.setrlimit {type(exc).__name__}: {exc}")
    sys.setrecursionlimit(10**6)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

    # noinspection PyShadowingNames
    def _forward_step(*, model: ConformerEncoder, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, model.out_dim))

    run_model(
        extern_data,
        lambda *, epoch, step: ConformerEncoder(
            in_dim,
            Dim(14, name="out"),
            ff_dim=Dim(17, name="ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2")],
                filter_sizes=[(3, 3), (3, 3)],
                pool_sizes=[(2, 1), (2, 1)],
            ),
            num_heads=2,
            num_layers=2,
        ),
        _forward_step,
    )


def test_e_branchformer():
    # https://arxiv.org/abs/2210.00077
    # https://github.com/espnet/espnet/tree/master/egs2/librispeech/asr1#e-branchformer
    # https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml

    import numpy as np
    import torch
    import returnn.frontend as rf
    from returnn.util.debug import PyTracer, check_py_traces_rf_to_pt_equal

    rf.select_backend_torch()
    rf.set_random_seed(42)

    from returnn.datasets.util.vocabulary import Vocabulary

    vocab = Vocabulary.create_vocab_from_labels(
        ["<blank>", "<BOS>", "<EOS>"] + [f"label-{i}" for i in range(17)], eos_label="<EOS>", bos_label="<BOS>"
    )
    target_dim = Dim(vocab.num_labels, name="vocab", vocab=vocab)

    import os
    import tempfile
    import textwrap
    import atexit

    from espnet2.tasks.asr import ASRTask
    from espnet2.asr.espnet_model import ESPnetASRModel
    from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder, EBranchformerEncoderLayer
    from espnet.nets.pytorch_backend.transformer.attention import RelPositionMultiHeadedAttention
    from espnet.nets.pytorch_backend.transformer.embedding import RelPositionalEncoding
    from espnet2.asr.layers.cgmlp import ConvolutionalGatingMLP, ConvolutionalSpatialGatingUnit

    # Code copied and adapted from i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/espnet.py.
    espnet_config_file = tempfile.mktemp(suffix=".yaml", prefix="espnet_config_")
    atexit.register(os.remove, espnet_config_file)
    with open(espnet_config_file, "w") as f:
        f.write(
            textwrap.dedent(
                # Copied from ESPnet egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml.
                # Then reduced dimensions/size for the test case, and slightly simplified.
                # language=yaml
                """\
                encoder: e_branchformer
                encoder_conf:
                    output_size: 50
                    attention_heads: 2
                    attention_layer_type: rel_selfattn
                    pos_enc_layer_type: rel_pos
                    rel_pos_type: latest
                    cgmlp_linear_units: 310
                    cgmlp_conv_kernel: 3
                    use_linear_after_conv: false
                    gate_activation: identity
                    num_blocks: 2
                    dropout_rate: 0.0
                    positional_dropout_rate: 0.0
                    attention_dropout_rate: 0.0
                    input_layer: conv2d
                    layer_drop_rate: 0.0
                    linear_units: 100
                    positionwise_layer_type: linear
                    macaron_ffn: true
                    use_ffn: true
                    merge_conv_kernel: 3

                decoder: transformer
                decoder_conf:
                    attention_heads: 2
                    linear_units: 200
                    num_blocks: 1
                    dropout_rate: 0.0
                    positional_dropout_rate: 0.0
                    self_attention_dropout_rate: 0.0
                    src_attention_dropout_rate: 0.0
                    layer_drop_rate: 0.0

                model_conf:
                    ctc_weight: 0.3
                    lsm_weight: 0.1
                    length_normalized_loss: false

                frontend_conf:
                    n_fft: 512
                    hop_length: 160
                """
            )
        )

    parser = ASRTask.get_parser()
    args = parser.parse_args(["--config", espnet_config_file])
    args.token_list = target_dim.vocab.labels
    args.model_conf["sym_sos"] = target_dim.vocab.labels[vocab.bos_label_id]
    args.model_conf["sym_eos"] = target_dim.vocab.labels[vocab.eos_label_id]

    model = ASRTask.build_model(args)
    assert isinstance(model, ESPnetASRModel)
    print("Target dim:", target_dim)
    print("Vocab size:", model.vocab_size)
    print("Vocab:", vocab.labels[:5], "...", vocab.labels[-5:])
    print("Ignore:", model.ignore_id)
    print("Blank:", model.blank_id)
    print("SOS/EOS:", model.sos, model.eos)

    _torch_default_device = "cpu"
    rnd = np.random.RandomState(42)
    batch_dim_ = Dim(5, name="batch")
    raw_audio_spatial_dim = Dim(
        rf.convert_to_tensor(torch.tensor([8_457, 10_237, 17_145, 5_141, 12_015], device="cpu"), dims=[batch_dim_]),
        name="audio",
    )
    targets_spatial_dim = Dim(
        rf.convert_to_tensor(torch.tensor([11, 10, 8, 7, 5], device="cpu"), dims=[batch_dim_]), name="dec"
    )
    raw_audio = rf.convert_to_tensor(
        torch.tensor(
            rnd.randn(batch_dim_.dimension, raw_audio_spatial_dim.dyn_size_ext.raw_tensor.max()).astype(np.float32),
            device=_torch_default_device,
        ),
        dims=[batch_dim_, raw_audio_spatial_dim],
    )
    targets = rf.convert_to_tensor(
        torch.tensor(
            rnd.randint(
                3, target_dim.dimension, (batch_dim_.dimension, targets_spatial_dim.dyn_size_ext.raw_tensor.max())
            ),
            device=_torch_default_device,
        ),
        dims=[batch_dim_, targets_spatial_dim],
    )

    # espnet2.asr.espnet_model.ESPnetASRModel.forward
    #   espnet2.asr.espnet_model.ESPnetASRModel.encode
    #     espnet2.asr.frontend.default.DefaultFrontend.forward
    #       espnet2.asr.frontend.default.DefaultFrontend._compute_stft
    #       (espnet.nets.pytorch_backend.frontends.frontend.Frontend.forward) (does nothing)
    #       espnet2.layers.log_mel.LogMel.forward
    #     espnet2.layers.utterance_mvn.UtteranceMVN.forward
    #     (No preencoder)
    #     espnet2.asr.encoder.e_branchformer_encoder.EBranchformerEncoder.forward
    #       espnet.nets.pytorch_backend.transformer.subsampling.Conv2dSubsampling.forward
    #         espnet.nets.pytorch_backend.transformer.embedding.RelPositionalEncoding
    #       espnet2.asr.encoder.e_branchformer_encoder.EBranchformerEncoderLayer.forward
    #         espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward.forward
    #         espnet.nets.pytorch_backend.transformer.attention.RelPositionMultiHeadedAttention.forward
    #         espnet2.asr.layers.cgmlp.ConvolutionalGatingMLP.forward
    #         espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.PositionwiseFeedForward.forward
    #   (CTC loss, Att-decoder)
    with PyTracer(
        [
            ESPnetASRModel.forward,
            ESPnetASRModel.encode,
            EBranchformerEncoder.forward,
            EBranchformerEncoderLayer.forward,
            RelPositionMultiHeadedAttention.forward,
            RelPositionalEncoding.forward,
            ConvolutionalGatingMLP.forward,
            ConvolutionalSpatialGatingUnit.forward,
        ],
        torch.Tensor,
    ) as trace_espnet, torch.no_grad():
        loss, stats, weight = model(
            speech=raw_audio.raw_tensor,
            speech_lengths=raw_audio_spatial_dim.dyn_size,
            text=targets.raw_tensor.to(torch.int64),  # targets are without EOS; added internally via add_sos_eos
            text_lengths=targets_spatial_dim.dyn_size,
        )

    # ESPnet usually does divide the loss by num seqs (batch dim) but not by seq length.
    print("ESPnet model loss:", loss)
    print(" stats:", stats)

    # I think we can skip the log-mel feat extraction and the conv frontend for now
    # (although it would be interesting to do that later as well)
    # and only cover the main E-Branchformer encoder itself
    # (also skipping CTC and the att-dec on top).

    # features_raw = trace_espnet.captured_locals[ESPnetASRModel.encode][0]["feats"][-1]  # [B,T',D=80]
    enc_in_raw = trace_espnet.captured_locals[EBranchformerEncoderLayer.forward][0]["x"][0]  # [B,T,D=50]
    mask_raw = trace_espnet.captured_locals[EBranchformerEncoderLayer.forward][0]["mask"][0].squeeze(dim=1)  # [B,T]
    assert mask_raw[:, 0].all()  # sanity check
    enc_out_raw = trace_espnet.captured_locals[ESPnetASRModel.forward][0]["encoder_out"][-1]  # [B,T,D=50]

    first_layer = model.encoder.encoders[0]
    assert isinstance(first_layer, EBranchformerEncoderLayer)

    enc_seq_lens_raw = torch.sum(mask_raw, dim=1).to(torch.int32)  # [B]
    batch_dim.dyn_size_ext = rf.convert_to_tensor(torch.tensor(enc_seq_lens_raw.shape[0]))
    enc_seq_lens = Tensor("enc_seq_lens", [batch_dim], dtype="int32", raw_tensor=enc_seq_lens_raw)
    enc_spatial_dim = Dim(enc_seq_lens, name="enc")
    model_dim = Dim(first_layer.size, name="model")
    enc_in = Tensor("enc_in", [batch_dim, enc_spatial_dim, model_dim], dtype="float32", raw_tensor=enc_in_raw)

    from returnn.frontend.encoder.conformer import ConformerEncoder
    from returnn.frontend.encoder.e_branchformer import EBranchformerLayer, FeedForwardConvGated

    model_rf = ConformerEncoder(
        in_dim=model_dim,
        out_dim=model_dim,
        input_layer=None,
        num_layers=len(model.encoder.encoders),
        encoder_layer=rf.build_dict(
            EBranchformerLayer,
            ff_dim=first_layer.feed_forward.w_1.out_features,
            num_heads=first_layer.attn.h,
            cgmlp_ff_dim=first_layer.cgmlp.channel_proj2.in_features,  # half the cgmlp_linear_units
            cgmlp_conv_kernel=first_layer.cgmlp.csgu.conv.kernel_size[0],
            merge_conv_kernel=first_layer.depthwise_conv_fusion.kernel_size[0],
        ),
        input_dropout=0.0,
        dropout=0.0,
        att_dropout=0.0,
    )

    from returnn.frontend.conversions.espnet_e_branchformer import import_params_espnet_e_branchformer_layer_to_rf

    assert len(model.encoder.encoders) == len(model_rf.layers)
    for layer, layer_rf in zip(model.encoder.encoders, model_rf.layers):
        assert isinstance(layer, EBranchformerEncoderLayer)
        assert isinstance(layer_rf, EBranchformerLayer)
        import_params_espnet_e_branchformer_layer_to_rf(layer, layer_rf)

    with PyTracer(
        [
            ConformerEncoder.__call__,
            EBranchformerLayer.__call__,
            rf.RelPosSelfAttention.__call__,
            rf.relative_positional_encoding,
            FeedForwardConvGated.__call__,
        ],
        Tensor,
    ) as trace_rf, torch.no_grad():
        enc_out, _ = model_rf(enc_in, in_spatial_dim=enc_spatial_dim)
    enc_out = enc_out.copy_transpose((batch_dim, enc_spatial_dim, model_dim))
    enc_out = enc_out.copy_masked(0.0)

    first_rf_layer: EBranchformerLayer = model_rf.layers[0]
    num_heads_dim: Dim = first_rf_layer.self_att.num_heads
    key_dim_per_head: Dim = first_rf_layer.self_att.key_dim_per_head
    cgmlp_ff_dim = first_rf_layer.cgmlp.ff_dim

    def _tensor(x: torch.Tensor, name: str, dims: Sequence[Dim]) -> Tensor:
        return rf.convert_to_tensor(x, name=name, dims=dims)

    from returnn.frontend.conversions.espnet_e_branchformer import _reorder_rel_pos_emb_espnet_to_rf

    check_py_traces_rf_to_pt_equal(
        trace_rf.captured_locals,
        trace_espnet.captured_locals,
        [
            (
                (EBranchformerLayer.__call__, 0, "x_ffn1_ln", 0),
                (EBranchformerEncoderLayer.forward, 0, "x", 1),
                (batch_dim, enc_spatial_dim, model_dim),
            ),
            (
                (EBranchformerLayer.__call__, 0, "x_ffn1_out", 0),
                (EBranchformerEncoderLayer.forward, 0, "x", 2),
                (batch_dim, enc_spatial_dim, model_dim),
            ),
            (
                (EBranchformerLayer.__call__, 0, "x_mhsa_ln", 0),
                (EBranchformerEncoderLayer.forward, 0, "x1", 1),
                (batch_dim, enc_spatial_dim, model_dim),
            ),
            (
                (rf.RelPosSelfAttention.__call__, 0, "source", 0),
                (RelPositionMultiHeadedAttention.forward, 0, "query", 0),
                (batch_dim, enc_spatial_dim, model_dim),
            ),
            # Also see test_rope_causal_self_att for similar checks.
            (
                (rf.RelPosSelfAttention.__call__, 0, "q", 0),
                (RelPositionMultiHeadedAttention.forward, 0, "q", 0),
                lambda x, *, name, **_: _tensor(
                    x.transpose(1, 2), name, [batch_dim, enc_spatial_dim, num_heads_dim, key_dim_per_head]
                ),
            ),
            (
                (rf.RelPosSelfAttention.__call__, 0, "v", 0),
                (RelPositionMultiHeadedAttention.forward, 0, "v", 0),
                lambda x, *, name, **_: _tensor(
                    x.transpose(1, 2), name, [batch_dim, enc_spatial_dim, num_heads_dim, key_dim_per_head]
                ),
            ),
            (
                (rf.RelPosSelfAttention.__call__, 0, "q_with_bias_u", 0),
                (RelPositionMultiHeadedAttention.forward, 0, "q_with_bias_u", 0),
                (batch_dim, num_heads_dim, enc_spatial_dim, key_dim_per_head),
            ),
            # Check RelPositionalEncoding vs our relative_positional_encoding
            (
                (rf.RelPosSelfAttention.__call__, 0, "pos_emb", 0),
                (RelPositionMultiHeadedAttention.forward, 0, "pos_emb", 0),
                lambda x, **_: _tensor(
                    _reorder_rel_pos_emb_espnet_to_rf(x.squeeze(dim=0)),
                    "pos_emb",
                    [enc_spatial_dim - 1 + enc_spatial_dim, model_dim],
                ),
            ),
            (
                (EBranchformerLayer.__call__, 0, "x_mhsa", 0),
                (EBranchformerEncoderLayer.forward, 0, "x_att", 0),
                (batch_dim, enc_spatial_dim, model_dim),
            ),
            (
                (FeedForwardConvGated.__call__, 0, "x_g", 0),
                (ConvolutionalSpatialGatingUnit.forward, 0, "x_g", 0),
                (batch_dim, enc_spatial_dim, cgmlp_ff_dim),
            ),
            (
                (EBranchformerLayer.__call__, 0, "x_cgmlp", 0),
                (EBranchformerEncoderLayer.forward, 0, "x2", 2),
                (batch_dim, enc_spatial_dim, model_dim),
            ),
            (
                (ConformerEncoder.__call__, 0, "x", -1),
                (ESPnetASRModel.forward, 0, "encoder_out", -1),
                (batch_dim, enc_spatial_dim, model_dim),
            ),
        ],
    )

    # Final check.
    # Actually the check_py_traces_rf_to_pt_equal should already have covered also this final output,
    # but anyway do it again now to be sure.
    assert enc_out_raw.shape == enc_out.raw_tensor.shape  # [B,T,D]
    assert enc_out_raw.shape[:1] == enc_seq_lens_raw.shape  # [B]
    for b in range(enc_out_raw.shape[0]):
        seq_len = enc_seq_lens_raw[b]
        torch.testing.assert_allclose(enc_out_raw[b, :seq_len], enc_out.raw_tensor[b, :seq_len], rtol=1e-5, atol=1e-5)
    # Check that there is sth non-zero.
    assert enc_seq_lens_raw.max() > 0
    assert torch.mean(enc_out.raw_tensor**2) > 0.1
    print("All matching!")
