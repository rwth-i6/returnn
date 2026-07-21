"""
Tests for returnn.frontend._packed_backend (packed / ragged tensor storage).

Each test compares the packed path against the same computation on padded storage,
on all non-padded frames.
"""

from __future__ import annotations

import _setup_test_env  # noqa
import sys
import unittest
from typing import Tuple

import numpy
import torch

from returnn.util import better_exchook
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend import _packed_backend as packed


def _make_input(*, batch_size: int = 2, seq_lens=(5, 3), feat: int = 4, seed: int = 42) -> Tuple[Tensor, Dim, Dim, Dim]:
    batch_dim = Dim(batch_size, name="batch")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(list(seq_lens), dtype=torch.int32))
    )
    feat_dim = Dim(feat, name="feat")
    x = Tensor("x", dims=[batch_dim, time_dim, feat_dim], dtype="float32")
    raw = torch.randn(batch_size, max(seq_lens), feat, generator=torch.Generator().manual_seed(seed))
    x.raw_tensor = raw
    return x, batch_dim, time_dim, feat_dim


def _flex_attention_usable() -> bool:
    # FlexAttention exists since torch 2.5, usable CPU (eager) support only later; we validated 2.7.
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 7):
        return False
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa
    except ImportError:
        return False
    return True


def _assert_equal_non_padded(actual: Tensor, expected: Tensor, batch_dim: Dim, time_dim: Dim, **kwargs):
    """compare on all non-padded frames. actual can have packed storage."""
    actual = packed.unpack(actual)
    order = [batch_dim, time_dim] + [d for d in expected.dims if d not in (batch_dim, time_dim)]
    actual = actual.copy_transpose(order)
    expected = expected.copy_transpose(order)
    mask = rf.sequence_mask([batch_dim, time_dim]).copy_compatible_to_dims([batch_dim, time_dim]).raw_tensor.numpy()
    a = actual.raw_tensor.detach().numpy()
    e = expected.raw_tensor.detach().numpy()
    numpy.testing.assert_allclose(a[mask], e[mask], **{"rtol": 1e-5, "atol": 1e-6, **kwargs})


def test_pack_auto_dims():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)  # no dims given: auto = (batch, time)
    assert packed.is_packed(xp)
    assert xp.dims == x.dims
    raw = xp.raw_tensor
    assert raw.orig_dims == (batch_dim, time_dim)
    assert raw.inner.dims[0] == raw.packed_dim
    assert raw.packed_dim.get_dim_value() == sum([5, 3])
    _assert_equal_non_padded(xp, x, batch_dim, time_dim)


def test_elementwise():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.relu(xp) * 2.0 + 1.0
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, rf.relu(x) * 2.0 + 1.0, batch_dim, time_dim)


def test_linear():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    out_dim = Dim(3, name="out")
    layer = rf.Linear(feat_dim, out_dim)  # with bias: also covers packed + plain combine
    xp = packed.pack(x)
    out_p = layer(xp)
    assert packed.is_packed(out_p)  # matmul over feat + bias add must stay packed
    _assert_equal_non_padded(out_p, layer(x), batch_dim, time_dim)


def test_layer_norm():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    layer = rf.LayerNorm(feat_dim)
    xp = packed.pack(x)
    out_p = layer(xp)
    assert packed.is_packed(out_p)  # statistics are over feat only, must stay packed
    _assert_equal_non_padded(out_p, layer(x), batch_dim, time_dim)


def test_output_block_log_softmax():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    vocab_dim = Dim(11, name="vocab")
    layer = rf.Linear(feat_dim, vocab_dim)
    xp = packed.pack(x)
    out_p = rf.log_softmax(layer(xp), axis=vocab_dim)
    assert packed.is_packed(out_p)  # softmax over vocab must stay packed
    _assert_equal_non_padded(out_p, rf.log_softmax(layer(x), axis=vocab_dim), batch_dim, time_dim)


def test_reduce_mean_over_packed_dims():
    # e.g. the mean loss over (batch, time): reduces the packed dim directly, no masking needed.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.reduce_mean(xp, axis=[batch_dim, time_dim])
    assert not packed.is_packed(out_p)  # packed dims fully reduced
    out_ref = rf.reduce_mean(x, axis=[batch_dim, time_dim])  # masked mean on padded storage
    numpy.testing.assert_allclose(
        out_p.raw_tensor.detach().numpy(), out_ref.raw_tensor.detach().numpy(), rtol=1e-5, atol=1e-6
    )


def test_reduce_over_time_segment():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.reduce_max(xp, axis=time_dim)  # partial packed reduce: segment reduce via rf.scatter
    # time is reduced away: the result has no packed dims left, so it is a plain tensor
    assert not packed.is_packed(out_p)
    out_ref = rf.reduce_max(x, axis=time_dim)
    out_p = out_p.copy_compatible_to_dims(out_ref.dims)
    numpy.testing.assert_allclose(
        out_p.raw_tensor.detach().numpy(), out_ref.raw_tensor.detach().numpy(), rtol=1e-5, atol=1e-6
    )


def test_conformer():
    """
    The goal test: a full Conformer forward pass on packed input matches the padded path.

    Currently most of the Conformer internals (conv subsampling, attention) go through
    the unpack fallback and decay to padded storage --
    this test establishes end-to-end correctness first;
    packed attention (flash-varlen) and packed conv come incrementally.
    """
    rf.select_backend_torch()
    from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

    x, batch_dim, time_dim, in_dim = _make_input(batch_size=2, seq_lens=(11, 7), feat=7, seed=3)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(17)
        model = ConformerEncoder(
            in_dim,
            Dim(14, name="enc"),
            ff_dim=Dim(17, name="ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(8, name="conv1"), Dim(8, name="conv2")],
                filter_sizes=[(3, 3), (3, 3)],
                pool_sizes=[(2, 1), (2, 1)],
            ),
            num_heads=2,
            num_layers=2,
        )
        out_ref, out_spatial_dim = model(x, in_spatial_dim=time_dim)
        # layout derived by hand for this model:
        # align 4 = total downsampling (two stride-2 pools);
        # gap 64 -> after the two stages exactly 16 left, as needed by the depthwise conv kernel 32
        # (each stage: pad consumes 1, pool divides by 2).
        xp = packed.pack(x, gap=64, align=4)
        out_p, out_spatial_dim_p = model(xp, in_spatial_dim=time_dim)
        # the whole subsample chain + depthwise convs must have run packed (no fallback warnings)
        assert "conv" not in packed._warned_fallback_ops
        assert "pad" not in packed._warned_fallback_ops
        assert "pool" not in packed._warned_fallback_ops
        if _flex_attention_usable():
            # the rel-pos self-attention must have run via the FlexAttention fast path
            assert "rel_pos_self_attention" not in packed._warned_fallback_ops
    assert out_spatial_dim == out_spatial_dim_p
    # fallbacks repack, so the encoder output must still be packed (over (batch, subsampled time))
    assert packed.is_packed(out_p)
    assert out_p.raw_tensor.orig_dims == (batch_dim, out_spatial_dim_p)
    _assert_equal_non_padded(packed.unpack(out_p), out_ref, batch_dim, out_spatial_dim, rtol=1e-4, atol=1e-5)


def test_seq_starts_cu_seqlens():
    # the layout descriptor: per-seq start offsets + flash-varlen-style cu_seqlens
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()  # lens (5, 3)
    raw = packed.pack(x).raw_tensor
    starts, seqs_dim = raw.seq_starts()
    assert seqs_dim == batch_dim
    assert starts.raw_tensor.tolist() == [0, 5]
    cu, cu_dim = raw.cu_seqlens()
    assert cu.dtype == "int32"
    assert cu.raw_tensor.tolist() == [0, 5, 8]
    assert cu_dim.get_dim_value() == 3


def test_pack_gap_roundtrip():
    # gapped layout: gap zero-frames between the sequences in the packed buffer
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()  # lens (5, 3)
    xp = packed.pack(x, gap=2)
    raw = xp.raw_tensor
    assert raw.gap == 2
    assert raw.packed_dim.get_dim_value() == 8 + 2 * 2
    starts, _ = raw.seq_starts()
    assert starts.raw_tensor.tolist() == [0, 7]  # 5 + gap 2
    _assert_equal_non_padded(xp, x, batch_dim, time_dim)
    # aligned layout: footprints roundup(len + gap, align), all starts multiples of align
    xa = packed.pack(x, gap=2, align=4)
    raw = xa.raw_tensor
    assert raw.packed_dim.get_dim_value() == 16  # roundup(5+2,4) + roundup(3+2,4) = 8 + 8
    starts, _ = raw.seq_starts()
    assert starts.raw_tensor.tolist() == [0, 8]
    _assert_equal_non_padded(xa, x, batch_dim, time_dim)
    # regap: cheap re-layout back to dense
    xd = packed.regap(xa, 0, align=1)
    assert xd.raw_tensor.packed_dim.get_dim_value() == 8
    _assert_equal_non_padded(xd, x, batch_dim, time_dim)


def test_conv_packed_gap():
    # packed conv: runs directly over the gapped packed buffer; layout (and packed dim) unchanged
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(7)
        conv = rf.Conv1d(feat_dim, Dim(6, name="out"), filter_size=3, padding="same")
        out_ref, _ = conv(x, in_spatial_dim=time_dim)
        xp = packed.pack(x, gap=1)  # required for kernel 3: ((3-1)*1+1)//2 = 1
        out_p, out_sp = conv(xp, in_spatial_dim=time_dim)
        assert out_sp == time_dim
        assert packed.is_packed(out_p)
        assert out_p.raw_tensor.packed_dim == xp.raw_tensor.packed_dim
        _assert_equal_non_padded(out_p, out_ref, batch_dim, time_dim)
        # dense (gap 0): must warn and fall back, but still be correct
        out_d, _ = conv(packed.pack(x), in_spatial_dim=time_dim)
        _assert_equal_non_padded(out_d, out_ref, batch_dim, time_dim)


def test_conv_packed_valid_dense():
    # padding "valid": every kept output window lies fully inside its own sequence
    # (windows never extend beyond the frames they are computed from),
    # so the packed conv needs NO input gap; boundary-crossing junk windows
    # land exactly in the output's gap slots (out gap = window span).
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(9)
        conv = rf.Conv1d(feat_dim, Dim(6, name="out"), filter_size=3, padding="valid")
        out_ref, out_time_ref = conv(x, in_spatial_dim=time_dim)
        out_p, out_time = conv(packed.pack(x), in_spatial_dim=time_dim)
        assert packed.is_packed(out_p)
        # out gap == span proves the packed fast path ran (a fallback repack would keep gap 0)
        assert out_p.raw_tensor.gap == 2
        assert out_time == out_time_ref
        _assert_equal_non_padded(out_p, out_ref, batch_dim, out_time_ref)


def test_conv_packed_gap_junk_robust():
    # "same" windows DO read into the gap, so the conv must zero the gap frames beforehand
    # (rf.where(frame_mask, x, 0)); junk in the gaps (e.g. from a previous bias add) must not leak.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(11)
        conv = rf.Conv1d(feat_dim, Dim(6, name="out"), filter_size=3, padding="same")
        out_ref, _ = conv(x + 123.0, in_spatial_dim=time_dim)
        xp = packed.pack(x, gap=1) + 123.0  # elementwise also hits the gap frames: gaps now hold 123
        out_p, _ = conv(xp, in_spatial_dim=time_dim)
        assert packed.is_packed(out_p)
        assert out_p.raw_tensor.packed_dim == xp.raw_tensor.packed_dim  # fast path, layout unchanged
        _assert_equal_non_padded(out_p, out_ref, batch_dim, time_dim)


def test_conv_packed_strided():
    # strided packed conv: stride | align and align | gap; out layout = (lens', gap/st, align/st)
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()  # lens (5, 3)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(13)
        conv = rf.Conv1d(feat_dim, Dim(6, name="out"), filter_size=3, padding="same", strides=2)
        out_ref, out_time_ref = conv(x, in_spatial_dim=time_dim)
        xp = packed.pack(x, gap=2, align=2)
        out_p, out_time = conv(xp, in_spatial_dim=time_dim)
        assert out_time == out_time_ref
        assert packed.is_packed(out_p)
        raw = out_p.raw_tensor
        assert raw.gap == 1 and raw.align == 1  # (gap 2, align 2) / stride 2
        _assert_equal_non_padded(out_p, out_ref, batch_dim, out_time_ref)


def test_pad_packed_inplace():
    # right-pad of the packed time dim: in-place, the new frames come out of the gap
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x, gap=2)
    padded_p, (out_time,) = rf.pad(xp, axes=[time_dim], padding=[(0, 1)], value=0.0)
    assert packed.is_packed(padded_p)
    raw = padded_p.raw_tensor
    assert raw.packed_dim == xp.raw_tensor.packed_dim  # same buffer, in place
    assert raw.gap == 1
    ref, _ = rf.pad(x, axes=[time_dim], padding=[(0, 1)], value=0.0)
    _assert_equal_non_padded(padded_p, ref, batch_dim, out_time)


def test_softmax_over_packed_time():
    # segment softmax: normalizing over the packed spatial dim runs directly on packed data,
    # no masking involved (padded frames do not exist in packed storage).
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.softmax(xp, axis=time_dim)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, rf.softmax(x, axis=time_dim), batch_dim, time_dim)
    out_p = rf.log_softmax(xp, axis=time_dim)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, rf.log_softmax(x, axis=time_dim), batch_dim, time_dim)


def test_transformer_aed():
    """
    Standard Transformer AED: encoder + decoder + cross-attention.

    Two different packings are involved:
    the encoder side is packed over (batch, enc_time),
    the decoder side over (batch, dec_time),
    and the cross-attention mixes them.
    """
    rf.select_backend_torch()
    from returnn.frontend.encoder.transformer import TransformerEncoder
    from returnn.frontend.decoder.transformer import TransformerDecoder

    batch_dim = Dim(2, name="batch")
    enc_time = Dim(
        Tensor("enc_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([7, 5], dtype=torch.int32))
    )
    dec_time = Dim(
        Tensor("dec_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([5, 3], dtype=torch.int32))
    )
    src_vocab = Dim(13, name="src_vocab")
    tgt_vocab = Dim(11, name="tgt_vocab")
    gen = torch.Generator().manual_seed(5)
    src = Tensor("src", dims=[batch_dim, enc_time], dtype="int32", sparse_dim=src_vocab)
    src.raw_tensor = torch.randint(0, 13, (2, 7), dtype=torch.int32, generator=gen)
    tgt = Tensor("tgt", dims=[batch_dim, dec_time], dtype="int32", sparse_dim=tgt_vocab)
    tgt.raw_tensor = torch.randint(0, 11, (2, 5), dtype=torch.int32, generator=gen)

    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(23)
        model_dim = Dim(12, name="model")
        encoder = TransformerEncoder(src_vocab, model_dim, num_layers=2, num_heads=2, dropout=0.0, att_dropout=0.0)
        decoder = TransformerDecoder(
            model_dim,
            tgt_vocab,
            model_dim,
            ff_dim=Dim(19, name="dec-ff"),
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            att_dropout=0.0,
        )

        def _fwd(src_t, tgt_t):
            enc_out = encoder(src_t, spatial_dim=enc_time)
            enc_state = decoder.transform_encoder(enc_out, axis=enc_time)
            logits, _ = decoder(
                tgt_t,
                spatial_dim=dec_time,
                state=decoder.default_initial_state(batch_dims=[batch_dim]),
                encoder=enc_state,
            )
            return logits

        logits_ref = _fwd(src, tgt)
        logits_p = _fwd(packed.pack(src), packed.pack(tgt))
        # gapped encoder: the cross-attention K/V then carries gap frames
        # -- the realistic conv-subsampled Conformer encoder feeding a Transformer decoder.
        # The varlen path must strip them and build separate query / kv offsets.
        logits_pg = _fwd(packed.pack(src, gap=8, align=2), packed.pack(tgt))
    assert packed.is_packed(logits_p)  # output side follows the decoder packing
    assert logits_p.raw_tensor.orig_dims == (batch_dim, dec_time)
    _assert_equal_non_padded(logits_p, logits_ref, batch_dim, dec_time, rtol=1e-4, atol=1e-5)
    assert packed.is_packed(logits_pg)
    _assert_equal_non_padded(logits_pg, logits_ref, batch_dim, dec_time, rtol=1e-4, atol=1e-5)


def test_batch_norm_packed_gapped_train():
    # batch_norm statistics must ignore gap frames: on a gapped layout in training,
    # the packed impl re-layouts to dense internally (see _DENSE_ONLY_INNER_OPS).
    # Compare against the dense packed run, which is the known-correct masked behavior
    # (note: the padded path with use_mask=False would include padding frames in the statistics).
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(seq_lens=(5, 3), feat=4, seed=8)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(3)
        bn_dense = rf.BatchNorm(feat_dim, use_mask=False)
        bn_gapped = rf.BatchNorm(feat_dim, use_mask=False)
        with rf.get_run_ctx().train_flag_ctx(True):
            out_dense = bn_dense(packed.pack(x))
            out_gapped = bn_gapped(packed.pack(x, gap=3, align=2))
        assert packed.is_packed(out_dense)
        assert packed.is_packed(out_gapped)
    _assert_equal_non_padded(out_gapped, packed.unpack(out_dense), batch_dim, time_dim)
    for p_dense, p_gapped in [
        (bn_dense.running_mean, bn_gapped.running_mean),
        (bn_dense.running_variance, bn_gapped.running_variance),
    ]:
        numpy.testing.assert_allclose(
            p_dense.raw_tensor.detach().numpy(), p_gapped.raw_tensor.detach().numpy(), rtol=1e-5, atol=1e-6
        )


def test_conformer_mixed_parity_lens():
    # Real-data case: seq lens NOT multiples of the total subsample factor.
    # The strided pool output layout is then not expressible in the (lens, gap, align) form;
    # it gets re-layouted into the closed form (one extra gather) and must STAY packed.
    rf.select_backend_torch()
    from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

    x, batch_dim, time_dim, in_dim = _make_input(batch_size=3, seq_lens=(11, 10, 7), feat=7, seed=4)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(17)
        model = ConformerEncoder(
            in_dim,
            Dim(14, name="enc"),
            ff_dim=Dim(17, name="ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(8, name="conv1"), Dim(8, name="conv2")],
                filter_sizes=[(3, 3), (3, 3)],
                pool_sizes=[(2, 1), (2, 1)],
            ),
            num_heads=2,
            num_layers=2,
        )
        out_ref, out_spatial_dim = model(x, in_spatial_dim=time_dim)
        xp = packed.pack(x, gap=64, align=4)
        out_p, out_spatial_dim_p = model(xp, in_spatial_dim=time_dim)
    assert out_spatial_dim == out_spatial_dim_p
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, out_ref, batch_dim, out_spatial_dim)


def test_mixed_operand_order():
    # plain-first mixed binary ops (plain * packed):
    # the base Backend.combine/compare re-dispatch to the higher-priority backend
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    plain = Tensor("y", dims=[feat_dim], dtype="float32")
    plain.raw_tensor = torch.randn(feat_dim.dimension, generator=torch.Generator().manual_seed(7))
    xp = packed.pack(x)
    for out_p, out_ref in [
        (plain * xp, plain * x),
        (plain + xp, plain + x),
        (1.0 - xp, 1.0 - x),
        (plain < xp, plain < x),
    ]:
        assert packed.is_packed(out_p)
        _assert_equal_non_padded(out_p, out_ref, batch_dim, time_dim)


def test_rel_pos_self_attention_packed():
    # Conformer-style rel-pos self-attention: on packed input this runs via the FlexAttention fast path
    # (document block mask + rel-pos score_mod over the flat packed buffer).
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(seq_lens=(9, 6), feat=8, seed=11)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(23)
        att = rf.RelPosSelfAttention(
            feat_dim,
            proj_dim=feat_dim,
            key_dim_total=Dim(8, name="key_tot"),
            value_dim_total=Dim(8, name="val_tot"),
            num_heads=2,
            att_dropout=0.0,
        )
        out_ref = att(x, axis=time_dim)
        xp = packed.pack(x, gap=4)  # some gap, to also cover the regap inside the fast path
        out_p = att(xp, axis=time_dim)
        assert packed.is_packed(out_p)
        if _flex_attention_usable():
            # must have taken the FlexAttention fast path (works eagerly on CPU too)
            assert "rel_pos_self_attention" not in packed._warned_fallback_ops
    _assert_equal_non_padded(out_p, out_ref, batch_dim, time_dim)


def test_aed_aux_ctc_stripped_real_model():
    """
    Stripped-down version of a real AED training setup:
    Conformer with strided subsampling (1,1)/(3,1)/(2,1) -- total time downsampling 6 --
    relu_square FF without bias,
    Transformer decoder with RMSNorm + rotary causal self-attention + gated FF,
    aux CTC on the encoder output,
    and seq lens NOT multiples of the downsampling factor (per-seq strided re-layout).

    Covers the integration issues found with the real model:
    per-spatial-dim padding lists in the strided subsampling convs,
    ctc_loss routing (unpack fallback),
    log_softmax feature_dim preservation (the CTC loss checks it),
    plain-first matmul operand order,
    and dtype handling under autocast (smoke).
    (The CUDA flash-varlen specifics, e.g. the contiguous-last-dim guard,
    are covered by the benchmark job's attention-path assert instead.)

    The known-missing packed impls are tracked as warnings, and the exact set is asserted:
    nothing else may fall back.
    """
    rf.select_backend_torch()
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from returnn.frontend.decoder.transformer import TransformerDecoder, FeedForwardGated

    # seq lens with distinct residues mod 6 (the total downsampling): per-seq strided re-layout
    x, batch_dim, time_dim, in_dim = _make_input(batch_size=3, seq_lens=(29, 22, 15), feat=8, seed=6)
    vocab_dim = Dim(11, name="vocab")
    wb_vocab_dim = Dim(12, name="vocab_wb")  # + blank
    tgt_time = Dim(
        Tensor("tgt_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([3, 2, 2], dtype=torch.int32))
    )
    targets = Tensor("targets", dims=[batch_dim, tgt_time], dtype="int32", sparse_dim=vocab_dim)
    targets.raw_tensor = torch.randint(0, 11, (3, 3), dtype=torch.int32, generator=torch.Generator().manual_seed(8))

    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(31)
        enc_dim = Dim(16, name="enc")
        encoder = ConformerEncoder(
            in_dim,
            enc_dim,
            ff_dim=Dim(24, name="enc-ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(4, name="conv1"), Dim(4, name="conv2"), Dim(4, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],  # total time downsampling 6
            ),
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=2,
            ),
            num_layers=2,
        )
        decoder = TransformerDecoder(
            enc_dim,
            vocab_dim,
            Dim(16, name="dec"),
            num_layers=2,
            num_heads=2,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            dropout=0.0,
            att_dropout=0.0,
        )
        aux_logits = rf.Linear(enc_dim, wb_vocab_dim)

        def _losses(feats_t, targets_t):
            enc_out, enc_spatial = encoder(feats_t, in_spatial_dim=time_dim)
            log_probs = rf.log_softmax(aux_logits(enc_out), axis=wb_vocab_dim)
            # log_softmax must preserve the feature dim (the CTC loss checks it)
            assert log_probs.feature_dim == wb_vocab_dim
            ctc = rf.ctc_loss(
                logits=log_probs,
                logits_normalized=True,
                targets=targets,  # stays plain, the loss unpacks anyway
                input_spatial_dim=enc_spatial,
                targets_spatial_dim=tgt_time,
                blank_index=wb_vocab_dim.dimension - 1,
            )
            ctc_sum = rf.reduce_sum(ctc, axis=list(ctc.dims))
            enc_state = decoder.transform_encoder(enc_out, axis=enc_spatial)
            logits, _ = decoder(
                targets_t,
                spatial_dim=tgt_time,
                state=decoder.default_initial_state(batch_dims=[batch_dim]),
                encoder=enc_state,
            )
            ce = rf.cross_entropy(estimated=logits, target=targets_t, axis=vocab_dim, estimated_type="logits")
            ce_sum = rf.reduce_sum(ce, axis=list(ce.dims))
            return ctc_sum, ce_sum

        ctc_ref, ce_ref = _losses(x, targets)

        # isolate this test's fallback warnings (the warn-once bookkeeping is global)
        warned_before = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.clear()
        packed.attention_path_counts.clear()
        # align 6 = total downsampling; gap 96 -> 16 at the subsampled rate, as the depthwise conv kernel 32 needs
        ctc_p, ce_p = _losses(packed.pack(x, gap=96, align=6), packed.pack(targets))
        warned_here = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.update(warned_before)

        for name, ref_t, p_t in [("ctc", ctc_ref, ctc_p), ("ce", ce_ref, ce_p)]:
            ref_v, p_v = float(ref_t.raw_tensor), float(p_t.raw_tensor)
            assert abs(ref_v - p_v) / max(abs(ref_v), 1e-6) < 1e-4, f"{name} loss: padded {ref_v} vs packed {p_v}"

        # NOTHING may fall back or even re-layout:
        # strided-conv outputs use per-seq layout lens (no strided-out gather),
        # and ctc_loss runs natively packed (FastBaumWelchPackedOp).
        expected = set()
        if _flex_attention_usable():
            assert warned_here == expected, f"unexpected fallbacks: {warned_here}"
            # 2 enc layers rel-pos flex; 2 dec layers x (self + cross) flex with document mask
            assert dict(packed.attention_path_counts) == {"rel_pos_flex": 2, "flex_doc": 4}
        else:
            assert expected <= warned_here, f"missing expected fallbacks: {expected - warned_here}"

        # plain-first matmul (plain a x packed b): must dispatch to the packed backend and stay packed
        w = Tensor("w", dims=[in_dim], dtype="float32")
        w.raw_tensor = torch.randn(in_dim.dimension, generator=torch.Generator().manual_seed(9))
        mm_p = rf.matmul(w, packed.pack(x), reduce=in_dim)
        assert packed.is_packed(mm_p)
        _assert_equal_non_padded(mm_p, rf.matmul(w, x, reduce=in_dim), batch_dim, time_dim)

        # autocast smoke: dtype handling, e.g. activations on the fp32 autocast list (relu_square -> pow)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            ctc_a, ce_a = _losses(packed.pack(x, gap=96, align=6), packed.pack(targets))
        assert numpy.isfinite(float(ctc_a.raw_tensor)) and numpy.isfinite(float(ce_a.raw_tensor))


def test_ctc_loss_packed_native():
    # Packed CTC via the native packed fast-baum-welch op (FastBaumWelchPackedOp):
    # loss and logits grads must match the padded path (torch F.ctc_loss).
    rf.select_backend_torch()
    batch_dim = Dim(3, name="batch")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([9, 7, 4], dtype=torch.int32))
    )
    vocab_dim = Dim(6, name="vocab")
    blank_index = 5
    tgt_time = Dim(
        Tensor("tgt_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([4, 3, 2], dtype=torch.int32))
    )
    targets = Tensor("targets", dims=[batch_dim, tgt_time], dtype="int32", sparse_dim=vocab_dim)
    targets.raw_tensor = torch.randint(0, 5, (3, 4), dtype=torch.int32, generator=torch.Generator().manual_seed(3))
    logits_raw = torch.randn(3, 9, 6, generator=torch.Generator().manual_seed(12))

    def _loss(raw_leaf, pack_gap=None):
        logits = Tensor("logits", dims=[batch_dim, time_dim, vocab_dim], dtype="float32", feature_dim_axis=2)
        logits.raw_tensor = raw_leaf
        if pack_gap is not None:
            logits = packed.pack(logits, gap=pack_gap)
        return rf.ctc_loss(
            logits=logits,
            targets=targets,
            input_spatial_dim=time_dim,
            targets_spatial_dim=tgt_time,
            blank_index=blank_index,
        )

    leaf_ref = logits_raw.clone().requires_grad_(True)
    loss_ref = _loss(leaf_ref)  # padded: torch F.ctc_loss
    rf.reduce_sum(loss_ref, axis=batch_dim).raw_tensor.backward()

    warned_before = set(packed._warned_fallback_ops)
    packed._warned_fallback_ops.clear()
    leaf_p = logits_raw.clone().requires_grad_(True)
    loss_p = _loss(leaf_p, pack_gap=0)
    assert "ctc_loss" not in packed._warned_fallback_ops  # must have taken the native packed path
    packed._warned_fallback_ops.update(warned_before)
    assert not packed.is_packed(loss_p) and loss_p.dims == (batch_dim,)
    rf.reduce_sum(loss_p, axis=batch_dim).raw_tensor.backward()

    numpy.testing.assert_allclose(
        loss_p.raw_tensor.detach().numpy(), loss_ref.raw_tensor.detach().numpy(), rtol=1e-4, atol=1e-5
    )
    mask = rf.sequence_mask([batch_dim, time_dim]).copy_compatible_to_dims([batch_dim, time_dim]).raw_tensor.numpy()
    numpy.testing.assert_allclose(leaf_p.grad.numpy()[mask], leaf_ref.grad.numpy()[mask], rtol=1e-4, atol=1e-5)

    # gapped input: re-layouted to dense internally, the loss must be the same
    loss_g = _loss(logits_raw.clone(), pack_gap=3)
    numpy.testing.assert_allclose(
        loss_g.raw_tensor.detach().numpy(), loss_ref.raw_tensor.detach().numpy(), rtol=1e-4, atol=1e-5
    )


def test_rel_pos_self_attention_per_seq_grad():
    # The per-seq CPU path (the train-mode dropout case, where flex bails):
    # called directly with att_dropout=0 for determinism, it must match the padded
    # reference exactly -- outputs AND grads (q/k/v inputs, pos_emb, biases).
    rf.select_backend_torch()
    batch_dim = Dim(3, name="batch")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([9, 6, 4], dtype=torch.int32))
    )
    # kv over its own (copied) spatial dim with the same lens, like the attention module does
    kv_time = Dim(
        Tensor("time_kv", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([9, 6, 4], dtype=torch.int32))
    )
    heads_dim = Dim(2, name="heads")
    qk_feat = Dim(4, name="qk_feat")
    v_feat = Dim(4, name="v_feat")
    pos_dim = Dim(2 * 9 - 1, name="pos")
    gen = torch.Generator().manual_seed(21)
    raws = {name: torch.randn(3, 9, 2, 4, generator=gen) for name in ("q", "k", "v")}
    pos_raw = torch.randn(2 * 9 - 1, 4, generator=gen)
    bias_u_raw = torch.randn(2, 4, generator=gen)
    bias_v_raw = torch.randn(2, 4, generator=gen)

    def _run(pack_gap=None):
        leaves = {name: raw.clone().requires_grad_(True) for name, raw in raws.items()}
        pos_leaf = pos_raw.clone().requires_grad_(True)
        bias_u_leaf, bias_v_leaf = bias_u_raw.clone().requires_grad_(True), bias_v_raw.clone().requires_grad_(True)
        qkv = {}
        for name, leaf in leaves.items():
            t = Tensor(
                name,
                dims=[
                    batch_dim,
                    time_dim if name == "q" else kv_time,
                    heads_dim,
                    qk_feat if name != "v" else v_feat,
                ],
                dtype="float32",
            )
            t.raw_tensor = leaf
            qkv[name] = t
        pos_emb = Tensor("pos_emb", dims=[pos_dim, qk_feat], dtype="float32")
        pos_emb.raw_tensor = pos_leaf
        bias_u = Tensor("bias_u", dims=[heads_dim, qk_feat], dtype="float32")
        bias_u.raw_tensor = bias_u_leaf
        bias_v = Tensor("bias_v", dims=[heads_dim, qk_feat], dtype="float32")
        bias_v.raw_tensor = bias_v_leaf
        kwargs = dict(
            pos_bias_u=bias_u,
            pos_bias_v=bias_v,
            att_dropout=0.0,
            att_dropout_broadcast=False,
            v_feat_dim=v_feat,
            qk_feat_dim=qk_feat,
            kv_spatial_dim=kv_time,
            query_spatial_dim=time_dim,
            pos_emb_spatial_dim=pos_dim,
        )
        if pack_gap is None:
            out = packed.Backend.rel_pos_self_attention(qkv["q"], qkv["k"], qkv["v"], pos_emb, **kwargs)
        else:
            out = packed._rel_pos_attention_per_seq(
                packed.pack(qkv["q"], gap=pack_gap),
                packed.pack(qkv["k"], gap=pack_gap),
                packed.pack(qkv["v"], gap=pack_gap),
                pos_emb,
                **kwargs,
            )
            assert out is not None and packed.is_packed(out)
            assert out.raw_tensor.gap == pack_gap  # layout restored
        loss = rf.reduce_sum(out, axis=list(out.dims))
        loss.raw_tensor.backward()
        return out, leaves, pos_leaf, bias_u_leaf, bias_v_leaf

    out_ref, leaves_ref, pos_g_ref, bu_g_ref, bv_g_ref = _run()
    packed.attention_path_counts.clear()
    out_p, leaves_p, pos_g_p, bu_g_p, bv_g_p = _run(pack_gap=0)
    assert packed.attention_path_counts.get("rel_pos_per_seq") == 1
    _assert_equal_non_padded(out_p, out_ref, batch_dim, time_dim, rtol=1e-4, atol=1e-5)
    mask = rf.sequence_mask([batch_dim, time_dim]).copy_compatible_to_dims([batch_dim, time_dim]).raw_tensor.numpy()
    for name in ("q", "k", "v"):
        numpy.testing.assert_allclose(
            leaves_p[name].grad.numpy()[mask], leaves_ref[name].grad.numpy()[mask], rtol=1e-4, atol=1e-5
        )
    numpy.testing.assert_allclose(pos_g_p.grad.numpy(), pos_g_ref.grad.numpy(), rtol=1e-4, atol=1e-5)
    numpy.testing.assert_allclose(bu_g_p.grad.numpy(), bu_g_ref.grad.numpy(), rtol=1e-4, atol=1e-5)
    numpy.testing.assert_allclose(bv_g_p.grad.numpy(), bv_g_ref.grad.numpy(), rtol=1e-4, atol=1e-5)
    # gapped layout roundtrip too
    out_g = _run(pack_gap=3)[0]
    _assert_equal_non_padded(out_g, out_ref, batch_dim, time_dim, rtol=1e-4, atol=1e-5)


def test_rel_pos_self_attention_dropout_train_packed():
    # att_dropout > 0 under the train flag: on CPU the per-seq path must be taken
    # (real weight dropout, no unpack); output packed, finite, and (per dropout)
    # equal to the no-dropout output in expectation -- here just sanity-bounded.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(9, 6, 4), feat=8, seed=23)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(31)
        att = rf.RelPosSelfAttention(
            feat_dim,
            proj_dim=feat_dim,
            key_dim_total=Dim(8, name="key_tot"),
            value_dim_total=Dim(8, name="val_tot"),
            num_heads=2,
            att_dropout=0.5,
        )
        packed.attention_path_counts.clear()
        warned_before = set(packed._warned_fallback_ops)  # isolate the warn-once bookkeeping
        with rf.get_run_ctx().train_flag_ctx(True):
            out = att(packed.pack(x), axis=time_dim)
        packed._warned_fallback_ops.clear()
        packed._warned_fallback_ops.update(warned_before)
        assert packed.is_packed(out)
        assert packed.attention_path_counts.get("rel_pos_per_seq") == 1
        assert bool(numpy.isfinite(packed.unpack(out).raw_tensor.detach().numpy()).all())


def test_rel_pos_att_triton_kernel_grad():
    # The Triton varlen rel-pos kernel (CUDA; the train-mode dropout fast path):
    # fwd + ALL grads (q, k, v, bd) vs autograd through an eager per-seq reference,
    # at dropout 0 (exact) and dropout > 0 with the kernel's own extracted mask (exact).
    rf.select_backend_torch()
    import torch as _torch

    if not _torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    try:
        from returnn.torch.util import rel_pos_att_triton as m
    except ImportError as exc:
        raise unittest.SkipTest(f"triton not available ({exc})")

    dev = "cuda"
    lens = [33, 21, 9]
    max_len = max(lens)
    total, n_heads, d = sum(lens), 2, 32
    r = 2 * max_len - 1
    starts = torch.tensor([0, 33, 54], dtype=torch.int32, device=dev)
    lens_t = torch.tensor(lens, dtype=torch.int32, device=dev)
    scale = 1.0 / (d**0.5)
    gen = torch.Generator(device="cpu").manual_seed(13)

    def _reference(q, k, v, bd, keep_mask, dropout_p):
        outs = []
        center = max_len - 1
        for b, ln in enumerate(lens):
            s0 = int(starts[b])
            qb, kb, vb, bdb = q[s0 : s0 + ln], k[s0 : s0 + ln], v[s0 : s0 + ln], bd[s0 : s0 + ln]
            s = torch.einsum("ihd,jhd->hij", qb, kb) * scale
            idx = center + torch.arange(ln, device=dev)[None, :] - torch.arange(ln, device=dev)[:, None]
            s = s + bdb.permute(1, 0, 2).gather(2, idx.unsqueeze(0).expand(s.shape[0], -1, -1))
            w = torch.softmax(s, dim=-1)
            if keep_mask is not None:
                w = w * keep_mask[s0 : s0 + ln, :, :ln].permute(1, 0, 2).float() / (1.0 - dropout_p)
            outs.append(torch.einsum("hij,jhd->ihd", w, vb))
        return torch.cat(outs, 0)

    for dropout_p, seed in [(0.0, 0), (0.3, 999)]:
        leaves = [torch.randn(total, n_heads, d, generator=gen).to(dev).requires_grad_(True) for _ in range(3)]
        bd_leaf = (torch.randn(total, n_heads, r, generator=gen) * 0.5).to(dev).requires_grad_(True)
        out = m.rel_pos_att_varlen(
            *leaves, bd_leaf, starts, lens_t, max_len, dropout_p=dropout_p, seed=seed, scale=scale
        )
        d_out = torch.randn(total, n_heads, d, generator=gen).to(dev)
        out.backward(d_out)
        grads_kernel = [t.grad.clone() for t in leaves] + [bd_leaf.grad.clone()]
        for t in leaves + [bd_leaf]:
            t.grad = None
        keep = None
        if dropout_p:
            keep = m.dump_mask(total, n_heads, max_len, r, dropout_p=dropout_p, seed=seed, device=dev)
        ref = _reference(*leaves, bd_leaf, keep, dropout_p)
        numpy.testing.assert_allclose(out.detach().cpu().numpy(), ref.detach().cpu().numpy(), rtol=1e-4, atol=1e-5)
        ref.backward(d_out)
        for g_kernel, t in zip(grads_kernel, leaves + [bd_leaf]):
            numpy.testing.assert_allclose(g_kernel.cpu().numpy(), t.grad.cpu().numpy(), rtol=1e-4, atol=1e-4)


def test_cast_packed():
    # rf.cast on packed data runs elementwise on the packed buffer (PackedBackend.cast_raw),
    # e.g. from the behavior_version>=27 keep-dtype path of LayerNorm/RMSNorm.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.cast(xp, "float64")
    assert packed.is_packed(out_p) and out_p.dtype == "float64"
    _assert_equal_non_padded(out_p, rf.cast(x, "float64"), batch_dim, time_dim)


def test_stft_packed():
    # stft on packed audio runs per-seq on the packed buffer (no unpack, no window crosses a seq),
    # bit-identical to the padded stft on the valid output frames.
    rf.select_backend_torch()
    batch_dim = Dim(3, name="batch")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([400, 320, 240], dtype=torch.int32))
    )
    audio = Tensor("audio", dims=[batch_dim, time_dim], dtype="float32")
    audio.raw_tensor = torch.randn(3, 400, generator=torch.Generator().manual_seed(1))
    opts = dict(in_spatial_dim=time_dim, frame_step=80, frame_length=160, fft_length=256)

    out_ref, out_sp, out_feat = rf.stft(audio, **opts)
    warned_before = set(packed._warned_fallback_ops)
    packed._warned_fallback_ops.clear()
    # frame_step | align, so the single-call packed stft applies (like the strided conv)
    xp = packed.regap(packed.pack(audio), 80, align=80)
    out_p, out_sp_p, _ = rf.stft(xp, out_dim=out_feat, **opts)
    warned = set(packed._warned_fallback_ops)
    packed._warned_fallback_ops.update(warned_before)
    assert "stft" not in warned  # ran the single-call packed stft, no unpack fallback
    assert packed.is_packed(out_p) and out_sp_p == out_sp
    _assert_equal_non_padded(out_p, out_ref, batch_dim, out_sp, rtol=1e-4, atol=1e-4)


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
