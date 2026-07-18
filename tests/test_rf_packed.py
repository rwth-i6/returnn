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
    assert packed.is_packed(logits_p)  # output side follows the decoder packing
    assert logits_p.raw_tensor.orig_dims == (batch_dim, dec_time)
    _assert_equal_non_padded(logits_p, logits_ref, batch_dim, dec_time, rtol=1e-4, atol=1e-5)


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
