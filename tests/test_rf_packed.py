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
    """
    compare on all non-padded frames. actual can have packed storage.

    The two sides can have different padded widths:
    under static tracing the packed side unpacks to the real max length,
    while the padded reference keeps its capacity-derived width.
    So slice each seq against its own storage, instead of one shared mask.
    """
    actual = packed.unpack(actual)
    order = [batch_dim, time_dim] + [d for d in expected.dims if d not in (batch_dim, time_dim)]
    a = actual.copy_transpose(order).raw_tensor.detach().cpu().numpy()
    e = expected.copy_transpose(order).raw_tensor.detach().cpu().numpy()
    opts = {"rtol": 1e-5, "atol": 1e-6, **kwargs}
    if time_dim.dyn_size_ext is None:
        numpy.testing.assert_allclose(a, e, **opts)
        return
    lens = time_dim.dyn_size_ext.copy_compatible_to_dims([batch_dim]).raw_tensor.cpu().numpy()
    assert len(lens) == a.shape[0] == e.shape[0], f"batch {len(lens)} vs {a.shape[0]} vs {e.shape[0]}"
    for b, n in enumerate(lens):
        numpy.testing.assert_allclose(a[b, :n], e[b, :n], err_msg=f"seq {b}", **opts)


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
        warned_before = set(packed._warned_fallback_ops)  # isolate the warn-once bookkeeping
        packed._warned_fallback_ops.clear()
        out_p, out_spatial_dim_p = model(xp, in_spatial_dim=time_dim)
        warned_here = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.update(warned_before)
        # the whole subsample chain + depthwise convs must have run packed (no fallback warnings)
        assert "conv" not in warned_here
        assert "pad" not in warned_here
        assert "pool" not in warned_here
        if _flex_attention_usable():
            # the rel-pos self-attention must have run via the FlexAttention fast path
            assert "rel_pos_self_attention" not in warned_here
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


def test_regap_bound_from_declared_total():
    # A static-traceable regap (the packed conv widening a too-small gap) derives its bound from the
    # total bound DECLARED at pack() time, plus the gap it adds.
    # The per-seq capacity product would instead put EVERY seq at its full capacity at once,
    # which the declared total already rules out (for loq that was 68_400 frames instead of 20_667,
    # and every downstream op inherits the regapped static shape).
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([5, 3], dtype=torch.int32)),
        capacity=8,  # static tracing needs a declared capacity
    )
    feat_dim = Dim(4, name="feat")
    x = Tensor("x", dims=[batch_dim, time_dim, feat_dim], dtype="float32")
    x.raw_tensor = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(23))  # padded to capacity
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(23)
        conv = rf.Conv1d(feat_dim, Dim(6, name="out"), filter_size=7, padding="same")  # span 6 -> needs gap 3
        out_ref, _ = conv(x, in_spatial_dim=time_dim)
        with rf.set_static_traceable_ctx():
            xp = packed.pack(x, gap=1, total_bound=12)
            assert xp.raw_tensor.packed_dim.dimension == 12
            out_p, _ = conv(xp, in_spatial_dim=time_dim)
            # the same packing, re-laid-out a second time: the base is the DECLARED bound,
            # never the one derived a step earlier, so the result does not depend on the path
            out_p2 = packed.regap(out_p, 3)
        assert packed.is_packed(out_p)
        # 12 + 2 seqs * ceil((3 - 1) / align 1) = 16.
        # (the per-seq capacity product would have been 2 * (8 + 3) = 22)
        assert out_p.raw_tensor.packed_dim.dimension == 16
        assert out_p.raw_tensor.gap == 3
        assert out_p2.raw_tensor.packed_dim.dimension == 16  # idempotent, not 16 + 2 * 2
        # and 16 really holds the content: footprints (5 + 3) + (3 + 3) = 14
        _assert_equal_non_padded(out_p, out_ref, batch_dim, time_dim)


def test_regap_gap_roundtrip_keeps_bound():
    # The varlen attention path densifies to gap 0 for the nested/jagged offsets and restores the
    # original layout afterwards (see _torch_sdpa_varlen_attention).
    # The bound must therefore be a function of the gap, not just grow:
    # with a one-sided delta each round trip added n_seqs frames, and after the decoder's
    # attention calls the result no longer matched a tensor that had not been through one.
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([5, 3], dtype=torch.int32)),
        capacity=8,
    )
    feat_dim = Dim(4, name="feat")
    x = Tensor("x", dims=[batch_dim, time_dim, feat_dim], dtype="float32")
    x.raw_tensor = torch.randn(2, 8, 4, generator=torch.Generator().manual_seed(5))
    with rf.set_default_device_ctx("cpu"):
        with rf.set_static_traceable_ctx():
            xp = packed.pack(x, gap=1, total_bound=14)
            assert xp.raw_tensor.packed_dim.dimension == 14
            dense = packed.regap(xp, 0)
            # a smaller gap needs less buffer: 14 - 2 seqs * 1
            assert dense.raw_tensor.packed_dim.dimension == 12
            back = packed.regap(dense, 1)
            assert back.raw_tensor.packed_dim.dimension == 14  # exactly where we started
            # and again, to catch a per-round-trip drift
            for _ in range(3):
                back = packed.regap(packed.regap(back, 0), 1)
            assert back.raw_tensor.packed_dim.dimension == 14
        _assert_equal_non_padded(back, x, batch_dim, time_dim)


def test_pack_static_traceable_requires_total_bound():
    # Without a declared bound there is nothing sound to derive a static buffer from,
    # so pack must say so instead of silently inventing the capacity product.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    with rf.set_default_device_ctx("cpu"):
        with rf.set_static_traceable_ctx():
            try:
                packed.pack(x, gap=1)
            except AssertionError as exc:
                assert "total_bound" in str(exc)
            else:
                raise Exception("pack should require total_bound under static tracing")


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

    # torch/device limits can leave NO packed sdpa fast path (e.g. torch<2.5 on cpu:
    # no flash, no NJT-cpu sdpa, no flex) -- allow the gated unpack fallback;
    # this test checks correctness, the fast paths are asserted by the benches.
    packed.set_allowed_fallbacks({"scaled_dot_product_attention"})

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
    packed.set_allowed_fallbacks(None)


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
        warned_before = set(packed._warned_fallback_ops)  # isolate the warn-once bookkeeping
        packed._warned_fallback_ops.clear()
        out_p = att(xp, axis=time_dim)
        warned_here = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.update(warned_before)
        assert packed.is_packed(out_p)
        if _flex_attention_usable():
            # must have taken the FlexAttention fast path (works eagerly on CPU too)
            assert "rel_pos_self_attention" not in warned_here
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

    if not _flex_attention_usable():
        # then no packed sdpa fast path at all (e.g. torch<2.5 cpu): gated unpack fallback
        packed.set_allowed_fallbacks({"scaled_dot_product_attention"})

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
    packed.set_allowed_fallbacks(None)


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

    # gapped packing: the native op reads each seq at its start offset (no regap) -> same loss + grads
    packed._warned_fallback_ops.clear()
    leaf_pg = logits_raw.clone().requires_grad_(True)
    loss_pg = _loss(leaf_pg, pack_gap=3)
    assert "ctc_loss" not in packed._warned_fallback_ops  # native packed path, no unpack fallback
    packed._warned_fallback_ops.update(warned_before)
    rf.reduce_sum(loss_pg, axis=batch_dim).raw_tensor.backward()
    numpy.testing.assert_allclose(
        loss_pg.raw_tensor.detach().numpy(), loss_ref.raw_tensor.detach().numpy(), rtol=1e-4, atol=1e-5
    )
    numpy.testing.assert_allclose(leaf_pg.grad.numpy()[mask], leaf_ref.grad.numpy()[mask], rtol=1e-4, atol=1e-5)

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


def _full_model_packed_vs_padded(
    dev: str, *, rtol: float, expected_att_paths: dict, amp: bool = False, atol: float = 1e-4
):
    """
    Full CTC+AED model, packed vs padded storage: losses, encoder output, and ALL parameter grads.

    The other tests here cover the ops in isolation.
    This one covers their composition, which is where a wrong per-seq index
    (a cu_seqlens total, a seq start) can hide:
    the forward still looks plausible while the backward credits the wrong sequence,
    so comparing the grads is the essential part of this test, not an extra.

    :param dev: "cpu" or "cuda". The attention fast paths are device-dependent
        (CPU takes FlexAttention, CUDA the Triton rel-pos kernel),
        so a CPU-only check leaves the kernels that real training runs untested.
    :param rtol: relative tolerance on losses and grads
    :param expected_att_paths: the attention paths the packed run must take.
        Asserted so a regression that silently reroutes to another path
        fails here, instead of passing on a path we did not mean to test.
    :param amp: run the model under autocast bfloat16, as the real trainings do.
        This is also the only way to reach the flash varlen attention path,
        which is gated on cuda + fp16/bf16 (see _torch_sdpa_varlen_attention),
        so an fp32-only test silently covers flex instead.
    :param atol: absolute tolerance for the encoder output comparison
    """
    import contextlib

    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from returnn.frontend.decoder.transformer import TransformerDecoder, FeedForwardGated

    rf.select_backend_torch()

    batch_dim = Dim(3, name="batch")
    seq_lens = [29, 22, 15]  # distinct residues mod 6 (the total downsampling)
    in_dim = Dim(8, name="feat")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(seq_lens, dtype=torch.int32))
    )
    x = Tensor("x", dims=[batch_dim, time_dim, in_dim], dtype="float32")
    x.raw_tensor = torch.randn(3, max(seq_lens), 8, generator=torch.Generator().manual_seed(6)).to(dev)

    vocab_dim = Dim(11, name="vocab")
    wb_vocab_dim = Dim(12, name="vocab_wb")  # + blank
    tgt_time = Dim(
        Tensor("tgt_time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([3, 2, 2], dtype=torch.int32))
    )
    targets = Tensor("targets", dims=[batch_dim, tgt_time], dtype="int32", sparse_dim=vocab_dim)
    targets.raw_tensor = torch.randint(0, 11, (3, 3), dtype=torch.int32, generator=torch.Generator().manual_seed(8)).to(
        dev
    )

    with rf.set_default_device_ctx(dev):
        rf.set_random_seed(31)
        # per-head dim must be >= 16: the Triton rel-pos kernel's tl.dot has a 16x16x16 minimum,
        # so 2 heads need at least 32 model dim, else the CUDA path does not compile
        enc_dim = Dim(32, name="enc")
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
                # pin the conv-block BatchNorm masking instead of inheriting the behavior-version
                # default, so this test isolates the packed ops and does not re-test that default.
                # Unmasked, the statistics run over the raw storage (padding frames vs gap frames),
                # so padded and packed cannot agree, see behavior version 29.
                conv_norm_opts={"use_mask": True},
            ),
            num_layers=2,
        )
        decoder = TransformerDecoder(
            enc_dim,
            vocab_dim,
            Dim(32, name="dec"),
            num_layers=2,
            num_heads=2,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            dropout=0.0,
            att_dropout=0.0,
        )
        aux_logits = rf.Linear(enc_dim, wb_vocab_dim)

        params = {}
        for mod, prefix in [(encoder, "enc"), (decoder, "dec"), (aux_logits, "aux")]:
            for name, param in mod.named_parameters():
                params[f"{prefix}.{name}"] = param

        def _losses(feats_t, targets_t):
            enc_out, enc_spatial = encoder(feats_t, in_spatial_dim=time_dim)
            log_probs = rf.log_softmax(aux_logits(enc_out), axis=wb_vocab_dim)
            ctc = rf.ctc_loss(
                logits=log_probs,
                logits_normalized=True,
                targets=targets,
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
            return ctc_sum, ce_sum, enc_out, enc_spatial

        def _take_grads():
            grads = {}
            for name, param in params.items():
                g = param.raw_tensor.grad
                grads[name] = None if g is None else g.detach().float().cpu().clone()
                param.raw_tensor.grad = None
            return grads

        amp_ctx = torch.autocast(device_type=dev, dtype=torch.bfloat16) if amp else contextlib.nullcontext()

        for param in params.values():
            param.raw_tensor.grad = None
        with amp_ctx:
            ctc_ref, ce_ref, enc_ref, spatial_ref = _losses(x, targets)
        (ctc_ref.raw_tensor + ce_ref.raw_tensor).backward()
        grads_ref = _take_grads()

        # isolate this test's fallback bookkeeping (it is global and warn-once)
        warned_before = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.clear()
        packed.attention_path_counts.clear()
        # align 6 = total downsampling; gap 96 -> 16 at the subsampled rate, for the depthwise conv span
        with amp_ctx:
            ctc_p, ce_p, enc_p, spatial_p = _losses(packed.pack(x, gap=96, align=6), packed.pack(targets))
        (ctc_p.raw_tensor + ce_p.raw_tensor).backward()
        grads_p = _take_grads()
        warned_here = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.update(warned_before)

        assert not warned_here, f"unexpected unpack fallbacks: {warned_here}"
        assert dict(packed.attention_path_counts) == expected_att_paths, (
            f"attention paths {dict(packed.attention_path_counts)}, expected {expected_att_paths}"
        )

        for name, ref_t, p_t in [("ctc", ctc_ref, ctc_p), ("ce", ce_ref, ce_p)]:
            ref_v, p_v = float(ref_t.raw_tensor), float(p_t.raw_tensor)
            rel = abs(ref_v - p_v) / max(abs(ref_v), 1e-6)
            assert rel < rtol, f"{name} loss: padded {ref_v} vs packed {p_v} (rel {rel})"

        # encoder output, on the non-padded frames only
        actual = packed.unpack(enc_p).copy_transpose([batch_dim, spatial_p, enc_dim])
        expected = enc_ref.copy_transpose([batch_dim, spatial_ref, enc_dim])
        mask = (
            rf.sequence_mask([batch_dim, spatial_ref])
            .copy_compatible_to_dims([batch_dim, spatial_ref])
            .raw_tensor.cpu()
            .numpy()
        )
        numpy.testing.assert_allclose(
            actual.raw_tensor.detach().float().cpu().numpy()[mask],
            expected.raw_tensor.detach().float().cpu().numpy()[mask],
            rtol=rtol,
            atol=atol,
            err_msg="encoder output differs",
        )

        assert len(params) > 50, f"expected the whole model, got {len(params)} params"
        for name in sorted(params):
            g_ref, g_p = grads_ref[name], grads_p[name]
            assert (g_ref is None) == (g_p is None), f"grad {name}: padded {g_ref is None}, packed {g_p is None}"
            if g_ref is None:
                continue
            rel = float((g_ref - g_p).abs().max()) / max(float(g_ref.abs().max()), 1e-8)
            assert rel < rtol, f"grad {name}: max rel diff {rel}"


def test_full_model_packed_vs_padded_grads():
    # CPU: encoder rel-pos attention and the decoder both go through FlexAttention.
    # That needs a recent torch (CI also runs torch 2.0), and without it there is no packed
    # fast path for these attentions at all: the expected paths would not match, and the
    # gated unpack fallback would raise. Nothing to compare then, so skip.
    if not _flex_attention_usable():
        raise unittest.SkipTest("needs FlexAttention (torch >= 2.7)")
    _full_model_packed_vs_padded("cpu", rtol=1e-4, expected_att_paths={"rel_pos_flex": 2, "flex_doc": 4})


def test_full_model_packed_vs_padded_grads_gpu():
    # CUDA takes the Triton rel-pos kernel for the encoder, which the CPU test cannot reach.
    # That is the path real training runs, so a failure here is a training bug, not only a test bug.
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    # TF32 would lose more precision than the packed-vs-padded difference we are testing for
    tf32_matmul, tf32_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        _full_model_packed_vs_padded("cuda", rtol=1e-3, expected_att_paths={"rel_pos_triton": 2, "flex_doc": 4})
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn


def test_full_model_packed_vs_padded_grads_gpu_bf16():
    # autocast bf16, as the real trainings run.
    # Only here does the decoder reach the flash varlen attention path:
    # it is gated on cuda + fp16/bf16, so the fp32 test above silently covers flex instead,
    # and the flash path is where a wrong cu_seqlens total would live.
    # The tolerance is bf16-wide, but the failure this guards against is not subtle:
    # attending across a sequence boundary moves grads by O(1), not by a few percent.
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    _full_model_packed_vs_padded(
        "cuda", rtol=1e-1, atol=1e-1, amp=True, expected_att_paths={"rel_pos_triton": 2, "flash": 4}
    )


def _full_model_packed_traced_replay(dev: str, *, rtol: float, atol: float, allow_att_fallback: bool = False):
    # The graph-captured training regime traces the whole packed step ONCE
    # (aot_function under bound shapes)
    # and re-executes the traced aten program every step.
    # Anything wrongly baked static at trace time,
    # or an op whose backward differs under tracing
    # (found this way: FlexAttention silently drops the grads
    # of score_mod-captured tensors, see test below),
    # is invisible to the eager tests.
    # So: trace the packed step on one batch,
    # replay on batches with other lens and content,
    # and compare losses and every param grad against the non-traced packed run.
    # nop compiler = the traced graph runs with eager kernels:
    # this isolates the trace itself (no Inductor, no CUDA graphs).
    if allow_att_fallback:
        # unpack -> padded attention -> repack: traceable aten ops, numerically exact,
        # so the traced-replay coverage of everything else does not depend on flex/flash.
        # The fast paths must be gated OFF too (rf_packed_att_fast_paths):
        # allowing the fallback alone would not stop the fast paths from winning.
        from returnn.config import Config, global_config_ctx

        packed.set_allowed_fallbacks(["scaled_dot_product_attention", "rel_pos_self_attention"])
        try:
            with global_config_ctx(Config({"rf_packed_att_fast_paths": False})):
                _full_model_packed_traced_replay_impl(dev, rtol=rtol, atol=atol, allow_att_fallback=True)
        finally:
            packed.set_allowed_fallbacks(None)
        return
    # rel-pos self-att on CPU has no traceable grad-correct fast path
    # (Triton is CUDA-only, flex refuses traced grads, per-seq declines under tracing),
    # so its unpack fallback must be allowed;
    # the path assertions below verify CUDA really takes Triton and the decoder stays on flex
    packed.set_allowed_fallbacks(["rel_pos_self_attention"])
    try:
        _full_model_packed_traced_replay_impl(dev, rtol=rtol, atol=atol, allow_att_fallback=False)
    finally:
        packed.set_allowed_fallbacks(None)


def _full_model_packed_traced_replay_impl(dev: str, *, rtol: float, atol: float, allow_att_fallback: bool):
    from functorch.compile import aot_function, nop
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from returnn.frontend.decoder.transformer import TransformerDecoder, FeedForwardGated

    rf.select_backend_torch()
    n_batch, t_cap, s_cap = 3, 32, 6
    in_dim = Dim(8, name="feat")
    vocab_dim = Dim(11, name="vocab")
    wb_vocab_dim = Dim(12, name="vocab_wb")
    with rf.set_default_device_ctx(dev):
        rf.set_random_seed(31)
        enc_dim = Dim(32, name="enc")
        encoder = ConformerEncoder(
            in_dim,
            enc_dim,
            ff_dim=Dim(24, name="enc-ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(4, name="conv1"), Dim(4, name="conv2"), Dim(4, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=2,
                conv_norm_opts={"use_mask": True},
            ),
            num_layers=2,
            dropout=0.0,
            att_dropout=0.0,
        )
        decoder = TransformerDecoder(
            enc_dim,
            vocab_dim,
            Dim(32, name="dec"),
            num_layers=2,
            num_heads=2,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            dropout=0.0,
            att_dropout=0.0,
        )
        aux_logits = rf.Linear(enc_dim, wb_vocab_dim)
        # params are explicit trace inputs, like the training capture does
        # (closure tensors would be baked as graph constants, then FakeTensor tracing rejects them)
        rf_params = []
        param_names = []
        for mod, prefix in [(encoder, "enc"), (decoder, "dec"), (aux_logits, "aux")]:
            for name, param in mod.named_parameters():
                rf_params.append(param)
                param_names.append(f"{prefix}.{name}")
        orig_raws = [p.raw_tensor for p in rf_params]
        trainable = [r.requires_grad for r in orig_raws]

        def step(raws):
            x_raw, lens_raw, tgt_raw, tgt_lens_raw = raws[:4]
            param_raws = raws[4:]
            for p_, t in zip(rf_params, param_raws):
                p_.raw_tensor = t
            try:
                return _step_inner(x_raw, lens_raw, tgt_raw, tgt_lens_raw, param_raws)
            finally:
                for p_, r0 in zip(rf_params, orig_raws):
                    p_.raw_tensor = r0

        def _step_inner(x_raw, lens_raw, tgt_raw, tgt_lens_raw, param_raws):
            batch_dim = Dim(n_batch, name="batch")
            time_dim = Dim(Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=lens_raw), capacity=t_cap)
            tgt_time = Dim(Tensor("tgt_time", dims=[batch_dim], dtype="int32", raw_tensor=tgt_lens_raw), capacity=s_cap)
            x = Tensor("x", dims=[batch_dim, time_dim, in_dim], dtype="float32", raw_tensor=x_raw)
            targets = Tensor(
                "targets", dims=[batch_dim, tgt_time], dtype="int32", sparse_dim=vocab_dim, raw_tensor=tgt_raw
            )
            with rf.set_static_traceable_ctx():
                # align 6 = total downsampling; gap 96 -> 16 at the subsampled rate
                xp = packed.pack(x, gap=96, align=6, total_bound=n_batch * (t_cap + 96))
                tp = packed.pack(targets, total_bound=n_batch * s_cap)
                enc_out, enc_spatial = encoder(xp, in_spatial_dim=time_dim)
                log_probs = rf.log_softmax(aux_logits(enc_out), axis=wb_vocab_dim)
                ctc = rf.ctc_loss(
                    logits=log_probs,
                    logits_normalized=True,
                    targets=tp,
                    input_spatial_dim=enc_spatial,
                    targets_spatial_dim=tgt_time,
                    blank_index=wb_vocab_dim.dimension - 1,
                )
                ctc_sum = rf.reduce_sum(ctc, axis=list(ctc.dims))
                enc_state = decoder.transform_encoder(enc_out, axis=enc_spatial)
                logits, _ = decoder(
                    tp,
                    spatial_dim=tgt_time,
                    state=decoder.default_initial_state(batch_dims=[batch_dim]),
                    encoder=enc_state,
                )
                ce = rf.cross_entropy(estimated=logits, target=tp, axis=vocab_dim, estimated_type="logits")
                ce_sum = rf.reduce_sum(ce, axis=list(ce.dims))
            loss = ctc_sum.raw_tensor + ce_sum.raw_tensor
            train_raws = [t for t, tr in zip(param_raws, trainable) if tr]
            grads = torch.autograd.grad(loss, train_raws, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(t) for g, t in zip(grads, train_raws)]
            return tuple(t.detach() for t in [ctc_sum.raw_tensor, ce_sum.raw_tensor, *grads])

        gen = torch.Generator().manual_seed(7)

        def make_batch(seq_lens, tgt_lens):
            # lens on the SAME device as the data: the device-lens regime the engine uses
            # (host-side lens would need host reads inside the trace, which is untraceable)
            x_raw = torch.randn(n_batch, t_cap, 8, generator=gen).to(dev)
            lens_raw = torch.tensor(seq_lens, dtype=torch.int32).to(dev)
            tgt_raw = torch.randint(0, 11, (n_batch, s_cap), dtype=torch.int32, generator=gen).to(dev)
            tgt_lens_raw = torch.tensor(tgt_lens, dtype=torch.int32).to(dev)
            return x_raw, lens_raw, tgt_raw, tgt_lens_raw

        # trace on the first batch; the later batches differ in lens (order too) and content,
        # so anything the trace froze from batch 0 fails the comparison there.
        # Every batch holds one capacity-length seq (audio and targets):
        # the CPU flex fast path sizes pos_emb from the capacity under static tracing
        # but checks applicability against the batch max len,
        # so a batch below capacity would reroute to the generic path mid-test.
        batches = [
            make_batch([32, 22, 15], [6, 2, 2]),
            make_batch([17, 32, 9], [1, 6, 2]),
            make_batch([12, 26, 32], [5, 1, 6]),
        ]
        n_train = sum(trainable)
        packed.attention_path_counts.clear()
        compiled = aot_function(step, fw_compiler=nop)
        for i, batch in enumerate(batches):
            raws = list(batch) + orig_raws
            expected = step(raws)
            actual = compiled(raws)
            assert len(expected) == len(actual) == 2 + n_train
            names = ["ctc", "ce"] + [f"grad {n}" for n, tr in zip(param_names, trainable) if tr]
            for name, e, a in zip(names, expected, actual):
                numpy.testing.assert_allclose(
                    a.detach().cpu().numpy(),
                    e.detach().cpu().numpy(),
                    rtol=rtol,
                    atol=atol,
                    err_msg=f"batch {i}, {name} differs between traced replay and eager",
                )

        # a silent reroute to another attention path would test something else -- assert what ran
        paths = dict(packed.attention_path_counts)
        if allow_att_fallback:
            assert not paths, f"expected only the unpack fallbacks, got fast paths {paths}"
        elif dev == "cuda":
            assert paths.get("rel_pos_triton"), f"expected the Triton rel-pos path, got {paths}"
        else:
            assert paths.get("flex_doc") and not paths.get("rel_pos_triton"), f"expected flex_doc only, got {paths}"


def test_full_model_packed_traced_program_replay():
    # CPU, CI-runnable; encoder rel-pos att via the allowed unpack fallback,
    # decoder self/cross att through flex_doc (see the path assertions in the helper).
    # This found the FlexAttention captured-grads drop:
    # the traced grads for linear_pos / pos_bias_v came out exactly zero,
    # so the rel-pos flex path now refuses traced training steps (see the gate in _packed_backend).
    # The Triton rel-pos kernel (the CUDA path real training captures) is unaffected,
    # verified on real checkpoints (the pos params train) and covered by the _gpu variant.
    if not _flex_attention_usable():
        raise unittest.SkipTest("needs FlexAttention (torch >= 2.7)")
    # torch 2.7 dynamo quirk: a REAL flex HOP compile earlier in the process
    # (e.g. test_conformer's eager rel-pos flex) poisons the next flex compile UNDER FAKE TRACING
    # (dynamo skip error on torch._library.utils.is_builtin inside can_auto_functionalize).
    # Resetting dynamo clears the stale compile state; graph_capture does the same before tracing.
    torch._dynamo.reset()
    # atol covers fp32 reassociation noise of the decomposed traced ops (observed ~2e-6)
    _full_model_packed_traced_replay("cpu", rtol=1e-4, atol=1e-5)


def test_full_model_packed_traced_program_replay_fallback_att():
    # No flex/flash dependency: the attentions take the (explicitly allowed) unpack fallback,
    # which is numerically exact and fully traceable,
    # so this variant runs on older torch, is immune to the flex dynamo quirk above,
    # and keeps the REAL model structure (rel-pos self-att) incl. its grads.
    # Everything else (pack/regap, conv relayouts, packed CTC, losses, all grads)
    # still gets the traced-replay coverage.
    _full_model_packed_traced_replay("cpu", rtol=1e-4, atol=1e-5, allow_att_fallback=True)


def test_full_model_packed_traced_program_replay_gpu():
    # CUDA: the encoder takes the Triton rel-pos kernel, i.e. the path real training captures.
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    tf32_matmul, tf32_cudnn = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        _full_model_packed_traced_replay("cuda", rtol=1e-3, atol=1e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul
        torch.backends.cudnn.allow_tf32 = tf32_cudnn


def test_conv_packed_auto_realign():
    # A stride-incompatible align is repaired like an insufficient gap:
    # the conv re-aligns and stays packed, instead of dropping to the unpack fallback.
    rf.select_backend_torch()
    x, batch_dim, time_dim, in_dim = _make_input(batch_size=3, seq_lens=(29, 22, 15), feat=8, seed=6)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(31)
        conv = rf.Conv1d(in_dim, Dim(6, name="out"), filter_size=3, padding="same", strides=3)
        out_ref, sp_ref = conv(x, in_spatial_dim=time_dim)

        warned_before = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.clear()
        # align 1 does not divide stride 3
        out_p, sp_p = conv(packed.pack(x, gap=4, align=1), in_spatial_dim=time_dim)
        warned = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.update(warned_before)

        assert packed.is_packed(out_p), "conv left the packed representation"
        assert not warned, f"unexpected fallback: {warned}"
        # the out align is the realigned in align divided by the stride, so it is not checked here
        _assert_equal_non_padded(out_p, out_ref, batch_dim, sp_ref)


def test_conv_packed_auto_realign_static():
    # Same under static tracing, where the realign must also derive a new total bound
    # (the align change shifts every per-seq footprint).
    rf.select_backend_torch()
    batch_dim = Dim(3, name="batch")
    seq_lens = [29, 22, 15]
    cap = 36
    in_dim = Dim(8, name="feat")
    time_dim = Dim(
        Tensor("time", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(seq_lens, dtype=torch.int32)),
        capacity=cap,
    )
    x = Tensor("x", dims=[batch_dim, time_dim, in_dim], dtype="float32")
    x.raw_tensor = torch.randn(3, cap, 8, generator=torch.Generator().manual_seed(6))

    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(31)
        conv = rf.Conv1d(in_dim, Dim(6, name="out"), filter_size=3, padding="same", strides=3)
        out_ref, sp_ref = conv(x, in_spatial_dim=time_dim)

        warned_before = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.clear()
        with rf.set_static_traceable_ctx():
            xp = packed.pack(x, gap=4, align=1, total_bound=3 * (cap + 4))
            out_p, sp_p = conv(xp, in_spatial_dim=time_dim)
        warned = set(packed._warned_fallback_ops)
        packed._warned_fallback_ops.update(warned_before)

        assert packed.is_packed(out_p), "conv left the packed representation"
        assert not warned, f"unexpected fallback: {warned}"
        assert out_p.raw_tensor.packed_dim.dimension is not None, "lost the static bound"
        _assert_equal_non_padded(out_p, out_ref, batch_dim, sp_ref)


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
