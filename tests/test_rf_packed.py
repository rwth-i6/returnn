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
import pytest
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
    # torch 2.12 REMOVED FlexAttention backward on CPU (NotImplementedError), and every use here
    # runs backward on CPU, so probe exactly that (version checks alone cannot express it).
    global _flex_attention_usable_cache
    if _flex_attention_usable_cache is not None:
        return _flex_attention_usable_cache
    _flex_attention_usable_cache = False
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 7):
        return False
    try:
        from torch.nn.attention.flex_attention import flex_attention  # noqa
    except ImportError:
        return False
    try:
        q, k, v = (torch.randn(1, 1, 4, 8, requires_grad=True) for _ in range(3))
        flex_attention(q, k, v).sum().backward()
    except (NotImplementedError, RuntimeError):
        return False
    _flex_attention_usable_cache = True
    return True


_flex_attention_usable_cache = None


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
    xp = packed.pack(x)
    # the feature dim often lives only on the outer tensor, e.g. rf.Linear sets it after the matmul
    xp.feature_dim = feat_dim
    for layer in (rf.LayerNorm(feat_dim), rf.RMSNorm(feat_dim)):
        out_p = layer(xp)
        assert packed.is_packed(out_p)  # statistics are over feat only, must stay packed
        assert out_p.feature_dim == feat_dim, out_p
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


def test_window_over_packed_time():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    win_dim = Dim(3, name="win")
    out, out_spatial_dim = rf.window(x, spatial_dim=time_dim, window_dim=win_dim, stride=2)
    xp = packed.pack(x)
    out_p, out_spatial_dim_p = rf.window(xp, spatial_dim=time_dim, window_dim=win_dim, stride=2)
    assert out_spatial_dim_p == out_spatial_dim
    assert packed.is_packed(out_p)  # the window re-lays out the packing, it does not fall back to padded
    raw = out_p.raw_tensor
    assert raw.orig_dims == (batch_dim, out_spatial_dim)
    _assert_equal_non_padded(out_p, out, batch_dim, out_spatial_dim)


def test_generic_op_packs_plain_frame_operand():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    targets = Tensor(
        "targets",
        dims=[batch_dim, time_dim],
        dtype="int32",
        sparse_dim=feat_dim,
        raw_tensor=torch.randint(0, feat_dim.dimension, (3, 7), dtype=torch.int32),
    )
    out = rf.reduce_argmax(x, axis=feat_dim) != targets
    xp = packed.pack(x)
    packed._warned_fallback_ops.clear()
    out_p = rf.reduce_argmax(xp, axis=feat_dim) != targets
    assert packed.is_packed(out_p)
    assert not packed._warned_fallback_ops
    _assert_equal_non_padded(out_p, out, batch_dim, time_dim)


def test_same_layout_over_freshly_minted_packed_dim_does_not_rebuild():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    xp = packed.pack(x)
    relaid, _ = rf.slice(xp, axis=time_dim, size=time_dim)
    assert packed.is_packed(relaid)
    assert relaid.raw_tensor.packed_dim != xp.raw_tensor.packed_dim
    assert relaid.raw_tensor.same_layout(xp.raw_tensor)

    n_regap = [0]
    orig = packed.regap

    def counting(source, gap, **kwargs):
        n_regap[0] += 1
        return orig(source, gap, **kwargs)

    packed.regap = counting
    try:
        out_p = relaid + xp
    finally:
        packed.regap = orig
    assert n_regap[0] == 0, f"rebuilt the buffer {n_regap[0]} times for an identical layout"
    ref = packed.unpack(relaid) + packed.unpack(xp)
    _assert_equal_non_padded(out_p, ref, batch_dim, time_dim)


def test_generic_op_keeps_the_fallback_when_a_dim_is_named():
    rf.select_backend_torch()
    x, batch_dim, time_dim, _feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    xp = packed.pack(x)
    try:
        rf.top_k(xp, axis=time_dim, k=2)
    except Exception as exc:
        assert "references packed dims" in str(exc), exc
    else:
        raise AssertionError("top_k names the packed time dim, it must not run on the packed buffer")


def _assert_equal_per_seq(actual: Tensor, expected: Tensor, batch_dim: Dim, a_dim: Dim, e_dim: Dim, *rest: Dim):
    """compare two tensors whose spatial dims are separate Dim objects (e.g. an int vs a Tensor stride)"""
    actual = packed.unpack(actual)
    a = actual.copy_transpose([batch_dim, a_dim, *rest]).raw_tensor.detach().cpu().numpy()
    e = expected.copy_transpose([batch_dim, e_dim, *rest]).raw_tensor.detach().cpu().numpy()
    a_lens = a_dim.dyn_size_ext.copy_compatible_to_dims([batch_dim]).raw_tensor.cpu().numpy()
    e_lens = e_dim.dyn_size_ext.copy_compatible_to_dims([batch_dim]).raw_tensor.cpu().numpy()
    numpy.testing.assert_equal(a_lens, e_lens)
    for b, n in enumerate(e_lens):
        numpy.testing.assert_allclose(a[b, :n], e[b, :n], rtol=1e-5, atol=1e-6, err_msg=f"seq {b}")


def test_window_tensor_stride():
    # a device-valued stride, which a static graph needs: same result as the int stride,
    # but the out spatial dim gets its sizes from a device computation
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    win_dim = Dim(3, name="win")
    ref, ref_dim = rf.window(x, spatial_dim=time_dim, window_dim=win_dim, stride=2)
    stride = rf.convert_to_tensor(2, dtype="int32")
    out, out_dim = rf.window(x, spatial_dim=time_dim, window_dim=win_dim, stride=stride)
    _assert_equal_per_seq(out, ref, batch_dim, out_dim, ref_dim, win_dim, feat_dim)
    out_p, out_p_dim = rf.window(packed.pack(x), spatial_dim=time_dim, window_dim=win_dim, stride=stride)
    assert packed.is_packed(out_p)
    _assert_equal_per_seq(out_p, ref, batch_dim, out_p_dim, ref_dim, win_dim, feat_dim)


def test_window_over_packed_time_static_traceable():
    # the captured regime:
    # the window buffer must be a static size,
    # and one captured graph has to cover every batch, so nothing may depend on the actual lens
    rf.select_backend_torch()
    n_seq_cap, buf, cap = 8, 200, 20
    live = [12, 5, 20]
    lens = live + [0] * (n_seq_cap - len(live))
    b_dim = Dim(n_seq_cap, name="batch")
    sd = Dim(Tensor("len", dims=[b_dim], dtype="int32", raw_tensor=torch.tensor(lens, dtype=torch.int32)), name="time")
    sd.capacity = cap
    packed_dim = Dim(buf, name="packed")
    flat = Tensor("flat", dims=[packed_dim], dtype="float32", raw_tensor=torch.arange(buf, dtype=torch.float32))
    x = packed.pack_import(flat, batch_dim=b_dim, spatial_dim=sd, packed_dim=packed_dim)
    win_dim = Dim(3, name="win")
    with rf.set_static_traceable_ctx():
        out, out_dim = rf.window(x, spatial_dim=sd, window_dim=win_dim, stride=2)
    assert packed.is_packed(out)
    raw = out.raw_tensor
    assert raw.orig_dims == (b_dim, out_dim)
    out_lens = out_dim.dyn_size_ext.copy_compatible_to_dims([b_dim]).raw_tensor
    # the buffer must be a static size that holds every sequence at capacity
    # (what one capture has to cover),
    # while still staying under the padded seqs-times-capacity product
    worst_case = n_seq_cap * -(-(cap + win_dim.dimension - 1) // 2)
    assert worst_case <= raw.packed_dim.dimension < n_seq_cap * cap, f"buffer {raw.packed_dim}"
    assert out_lens[: len(live)].tolist() == [-(-n // 2) for n in live], out_lens[: len(live)].tolist()
    assert int(out_lens[len(live)]) == 0, "padding seq must stay empty"


def test_window_dynamic_window_dim():
    # a window whose width is drawn per step:
    # the buffer follows the declared capacity and the drawn width only masks,
    # so every draw gives the same shapes, which is what one capture needs
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(9, 5, 4))
    max_win, stride = 6, 2
    shapes = set()
    for drawn in (6, 4, 2):
        win_dim = Dim(rf.convert_to_tensor(drawn, dtype="int32"), name="win")
        win_dim.capacity = max_win
        out, out_dim = rf.window(
            packed.pack(x),
            spatial_dim=time_dim,
            window_dim=win_dim,
            window_left=0,
            stride=rf.convert_to_tensor(stride, dtype="int32"),
            pad_value=0.0,
        )
        assert packed.is_packed(out)
        raw = out.raw_tensor
        shapes.add((raw.packed_dim.dimension, tuple(raw.inner.raw_tensor.shape)))

        ref_win = Dim(drawn, name="ref_win")
        ref, ref_dim = rf.window(
            x, spatial_dim=time_dim, window_dim=ref_win, window_left=0, stride=stride, pad_value=0.0
        )
        a = packed.unpack(out).copy_transpose([batch_dim, out_dim, win_dim, feat_dim])
        a = a.raw_tensor.detach().cpu().numpy()[:, :, :drawn]
        e = ref.copy_transpose([batch_dim, ref_dim, ref_win, feat_dim]).raw_tensor.detach().cpu().numpy()
        lens = ref_dim.dyn_size_ext.copy_compatible_to_dims([batch_dim]).raw_tensor.cpu().numpy()
        numpy.testing.assert_equal(out_dim.dyn_size_ext.copy_compatible_to_dims([batch_dim]).raw_tensor.numpy(), lens)
        for b, n in enumerate(lens):
            numpy.testing.assert_allclose(a[b, :n], e[b, :n], rtol=1e-5, atol=1e-6, err_msg=f"draw {drawn} seq {b}")
    assert len(shapes) == 1, f"shapes vary across draws: {shapes}"


def test_slice_packed_time_shift():
    # the pad-then-slice pattern that builds chunk history: both stay packed re-layouts
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    xp = packed.pack(x)
    padded, (pad_dim,) = rf.pad(xp, axes=[time_dim], padding=[(2, 0)], value=0.0)
    out_p, _ = rf.slice(padded, axis=pad_dim, size=time_dim, out_dim=time_dim)
    assert packed.is_packed(out_p)
    ref = rf.shift_right(x, axis=time_dim, pad_value=0.0)
    ref = rf.shift_right(ref, axis=time_dim, pad_value=0.0)
    _assert_equal_non_padded(out_p, ref, batch_dim, time_dim)


def test_merge_dims_packed_with_static():
    # un-chunking: merge the innermost packed dim with a static inner dim back to one time axis
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(6, 4, 2))
    s_dim = Dim(3, name="s")
    y = rf.expand_dim(x, dim=s_dim) + rf.range_over_dim(s_dim, dtype=x.dtype)
    ref, ref_dim = rf.merge_dims(y, dims=(time_dim, s_dim))
    yp = rf.expand_dim(packed.pack(x), dim=s_dim) + rf.range_over_dim(s_dim, dtype=x.dtype)
    out_p, out_dim = rf.merge_dims(yp, dims=(time_dim, s_dim))
    assert packed.is_packed(out_p)
    _assert_equal_per_seq(out_p, ref, batch_dim, out_dim, ref_dim, feat_dim)


def test_matmul_packed_both_operands():
    # both operands packed over the same seqs (e.g. chunk-local attention scores):
    # the packed dims are batch dims of the matmul, it runs on the inners
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(5, 3, 2))
    y = x * 0.5 + 1.0  # same dims, different content
    ref = rf.matmul(x, y, reduce=feat_dim, use_mask=False)  # per-frame dot product
    xp = packed.pack(x)
    yp = packed.pack(y, out_dim=xp.raw_tensor.packed_dim)
    out_p = rf.matmul(xp, yp, reduce=feat_dim, use_mask=False)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, ref, batch_dim, time_dim)


def test_gather_packed_plain_axis():
    # gather along a plain (non-packed) axis with per-frame indices: elementwise on the inner buffer
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(5, 3, 2))
    s_dim = Dim(4, name="s")
    y = rf.expand_dim(x, dim=s_dim) + rf.range_over_dim(s_dim, dtype=x.dtype)
    idx = rf.cast(rf.range_over_dim(time_dim) % s_dim.dimension, "int32")
    idx.sparse_dim = s_dim
    ref = rf.gather(y, indices=idx, axis=s_dim)
    yp = rf.expand_dim(packed.pack(x), dim=s_dim) + rf.range_over_dim(s_dim, dtype=x.dtype)
    out_p = rf.gather(yp, indices=idx, axis=s_dim)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, ref, batch_dim, time_dim)


def test_cumsum_over_packed_time():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.cumsum(xp, spatial_dim=time_dim)  # segment scan over the flat buffer
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, rf.cumsum(x, spatial_dim=time_dim), batch_dim, time_dim)


def test_cumsum_over_packed_time_gap():
    # gap/align frames carry junk, and a total_bound buffer has a junk tail;
    # neither may leak into a sequence's running sum.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(5, 1, 4))
    xp = packed.pack(x, gap=2, align=4, total_bound=64)
    out_p = rf.cumsum(xp, spatial_dim=time_dim)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, rf.cumsum(x, spatial_dim=time_dim), batch_dim, time_dim)


def test_cumsum_over_non_packed_dim():
    # feature is not packed, so this is a plain scan on the flat buffer, storage untouched
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.cumsum(xp, spatial_dim=feat_dim)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, rf.cumsum(x, spatial_dim=feat_dim), batch_dim, time_dim)


def _repeat_case(seq_lens, dur_rows, **repeat_kwargs):
    """padded vs packed repeat on the same durations, compared on the real frames"""
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=len(seq_lens), seq_lens=seq_lens)
    dur = Tensor("dur", dims=[batch_dim, time_dim], dtype="int32", raw_tensor=torch.tensor(dur_rows, dtype=torch.int32))
    out_ref, out_dim = rf.repeat(x, in_spatial_dim=time_dim, repeats=dur, **repeat_kwargs)
    xp = packed.pack(x)
    # share the input packing, so the durations need no relayout
    durp = packed.pack(dur, dims=(batch_dim, time_dim), out_dim=xp.raw_tensor.packed_dim)
    out_p, out_dim_p = rf.repeat(xp, in_spatial_dim=time_dim, repeats=durp, out_spatial_dim=out_dim, **repeat_kwargs)
    assert packed.is_packed(out_p)
    assert out_dim_p == out_dim
    _assert_equal_non_padded(out_p, out_ref, batch_dim, out_dim)


def test_repeat_packed():
    rf.select_backend_torch()
    # includes a 0 duration (element dropped) and a one-element sequence
    _repeat_case((5, 1, 4), [[2, 1, 3, 0, 2], [4, 0, 0, 0, 0], [1, 1, 1, 5, 0]])


def test_repeat_packed_max_len_factor():
    rf.select_backend_torch()
    # row 0 expands 3.6x and gets scaled down, row 1 stays; the two sides must agree either way
    _repeat_case((5, 3), [[9, 3, 3, 2, 1], [1, 2, 1, 0, 0]], max_len_factor=2)


def test_repeat_padded_repeats():
    # repeats from creation ops (rf.random_uniform/rf.constant) come padded;
    # repeat must pack them onto the values' packing, like combine does for mixed operands
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(5, 1, 4))
    dur_rows = [[2, 1, 3, 0, 2], [4, 0, 0, 0, 0], [1, 1, 1, 5, 0]]
    dur = Tensor("dur", dims=[batch_dim, time_dim], dtype="int32", raw_tensor=torch.tensor(dur_rows, dtype=torch.int32))
    out_ref, out_dim = rf.repeat(x, in_spatial_dim=time_dim, repeats=dur)
    xp = packed.pack(x)
    out_p, out_dim_p = rf.repeat(xp, in_spatial_dim=time_dim, repeats=dur, out_spatial_dim=out_dim)
    assert packed.is_packed(out_p)
    assert out_dim_p == out_dim
    _assert_equal_non_padded(out_p, out_ref, batch_dim, out_dim)


def test_gather_packed_shift_within_seq():
    # successor lookup: index is a position inside the sequence, so it must not read the next one
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(5, 1, 4))
    idx = rf.range_over_dim(time_dim) + 1
    out_ref = rf.gather(x, indices=idx, axis=time_dim, clip_to_valid=True)
    xp = packed.pack(x)
    out_p = rf.gather(xp, indices=idx, axis=time_dim, clip_to_valid=True)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, out_ref, batch_dim, time_dim)


def test_gather_packed_keeps_sparse_dim():
    # a sparse dim assigned on the virtual tensor does not reach the inner buffer,
    # so an op rewrapping from the inner must not restore the old one
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=2, seq_lens=(4, 2))
    vocab = Dim(7, name="vocab")
    labels = Tensor(
        "labels",
        dims=[batch_dim, time_dim],
        dtype="int32",
        raw_tensor=torch.randint(0, 7, (2, 4), dtype=torch.int32),
        sparse_dim=vocab,
    )
    lp = packed.pack(labels, dims=(batch_dim, time_dim))
    wider = Dim(8, name="vocab+1")
    lp = lp.copy()
    lp.sparse_dim = wider
    out = rf.gather(lp, indices=rf.range_over_dim(time_dim) + 1, axis=time_dim, clip_to_valid=True)
    assert out.sparse_dim == wider, f"sparse dim lost: {out.sparse_dim}"


def test_repeat_packed_static_traceable():
    # the captured regime: batch_size_bound makes the batch dim static, so the repeat buffer
    # must be a static size derived from the declared expansion factor
    rf.select_backend_torch()
    n_seq_cap, buf, cap, factor = 8, 200, 20, 4
    live = [12, 5, 20]
    lens = live + [0] * (n_seq_cap - len(live))
    b_dim = Dim(n_seq_cap, name="batch")
    sd = Dim(Tensor("len", dims=[b_dim], dtype="int32", raw_tensor=torch.tensor(lens, dtype=torch.int32)), name="time")
    sd.capacity = cap
    packed_dim = Dim(buf, name="packed")
    flat = Tensor("flat", dims=[packed_dim], dtype="int32", raw_tensor=torch.arange(buf, dtype=torch.int32))
    x = packed.pack_import(flat, batch_dim=b_dim, spatial_dim=sd, packed_dim=packed_dim)
    dur_flat = Tensor("dur", dims=[packed_dim], dtype="int32", raw_tensor=torch.ones((buf,), dtype=torch.int32) * 3)
    dur = packed.pack_import(dur_flat, batch_dim=b_dim, spatial_dim=sd, packed_dim=packed_dim)
    with rf.set_static_traceable_ctx():
        out, out_dim = rf.repeat(x, in_spatial_dim=sd, repeats=dur, max_len_factor=factor)
    raw = out.raw_tensor
    assert raw.packed_dim.dimension == buf * factor, f"buffer {raw.packed_dim} != {buf * factor}"
    out_lens = out_dim.dyn_size_ext.raw_tensor
    assert out_lens[: len(live)].tolist() == [3 * n for n in live], out_lens[: len(live)].tolist()
    assert int(out_lens[len(live)]) == 0, "padding seq must stay empty"


def test_concat_packed_separate_packings():
    # joining two streams that live in different buffers, e.g. audio + pseudo speech
    rf.select_backend_torch()
    a, batch_dim, a_dim, feat_dim = _make_input(batch_size=3, seq_lens=(5, 1, 4), seed=1)
    b_dim = Dim(
        Tensor("b_len", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([2, 3, 0], dtype=torch.int32)),
        name="b_time",
    )
    b = Tensor(
        "b",
        dims=[batch_dim, b_dim, feat_dim],
        dtype="float32",
        raw_tensor=torch.randn(3, 3, 4, generator=torch.Generator().manual_seed(7)),
    )
    ref, out_dim = rf.concat((a, a_dim), (b, b_dim), handle_dynamic_dims=True)
    ap, bp = packed.pack(a), packed.pack(b)  # deliberately separate packings
    assert ap.raw_tensor.packed_dim != bp.raw_tensor.packed_dim
    out_p, out_dim_p = rf.concat((ap, a_dim), (bp, b_dim), handle_dynamic_dims=True, out_dim=out_dim)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, ref, batch_dim, out_dim)


def test_concat_packed_empty_source():
    # eval feeds no text stream, so one source is empty and its packed buffer has zero rows.
    # such a buffer has no last row to clip indices to, and it contributes no frames at all.
    rf.select_backend_torch()
    a, batch_dim, a_dim, feat_dim = _make_input(batch_size=3, seq_lens=(5, 1, 4), seed=1)
    b_dim = Dim(
        Tensor("b_len", dims=[batch_dim], dtype="int32", raw_tensor=torch.zeros(3, dtype=torch.int32)),
        name="b_time",
    )
    b = Tensor("b", dims=[batch_dim, b_dim, feat_dim], dtype="float32", raw_tensor=torch.zeros(3, 0, 4))
    ref, out_dim = rf.concat((a, a_dim), (b, b_dim), handle_dynamic_dims=True)
    ap, bp = packed.pack(a), packed.pack(b)
    out_p, _ = rf.concat((ap, a_dim), (bp, b_dim), handle_dynamic_dims=True, out_dim=out_dim)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, ref, batch_dim, out_dim)


def test_concat_packed_static_traceable():
    # captured regime: static batch dim and static per-stream buffers,
    # so the concat buffer must be a static size derived from the sources' content bounds
    rf.select_backend_torch()
    n_seq_cap, feat = 6, 4
    live_a, live_b = [4, 2, 5], [3, 1, 0]
    buf_a, buf_b = 40, 20
    b_dim = Dim(n_seq_cap, name="batch")
    f_dim = Dim(feat, name="feat")  # shared: a per-source dim would differ

    def mk(lens, buf, name):
        d = Dim(
            Tensor(
                f"{name}_len",
                dims=[b_dim],
                dtype="int32",
                raw_tensor=torch.tensor(lens + [0] * (n_seq_cap - len(lens)), dtype=torch.int32),
            ),
            name=name,
        )
        d.capacity = buf
        pd = Dim(buf, name=f"packed_{name}")
        flat = Tensor("flat", dims=[pd, f_dim], dtype="float32", raw_tensor=torch.randn(buf, feat))
        return packed.pack_import(flat, batch_dim=b_dim, spatial_dim=d, packed_dim=pd, feature_dim=f_dim), d

    ap, a_dim = mk(live_a, buf_a, "a")
    bp, b_sp = mk(live_b, buf_b, "b")
    with rf.set_static_traceable_ctx():
        out, out_dim = rf.concat((ap, a_dim), (bp, b_sp), handle_dynamic_dims=True)
    assert packed.is_packed(out)
    raw = out.raw_tensor
    assert raw.packed_dim.dimension == buf_a + buf_b, f"buffer {raw.packed_dim} != {buf_a + buf_b}"
    lens = out_dim.dyn_size_ext.raw_tensor
    assert lens[: len(live_a)].tolist() == [x + y for x, y in zip(live_a, live_b)], lens[:3].tolist()


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


def test_chunked_rel_pos_att_triton_kernel_grad():
    # The chunked variant (chunk of S rows over its own rows and the first C rows of MEM previous
    # chunks, zero keys before the first chunk): fwd + all grads vs an explicit per-row reference,
    # at dropout 0 and at dropout > 0 with the kernel's own extracted mask.
    rf.select_backend_torch()
    import torch as _torch

    if not _torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    try:
        from returnn.torch.util import chunked_rel_pos_att_triton as m
    except ImportError as exc:
        raise unittest.SkipTest(f"triton not available ({exc})")

    dev = "cuda"
    s_rows, c_rows, mem = 4, 2, 2
    chunks = [5, 3, 4]
    lens = [n * s_rows for n in chunks]
    starts = [0, 20, 32]
    total, n_heads, d = 48, 2, 32
    r = mem * c_rows + 2 * s_rows - 1
    center = mem * c_rows + s_rows - 1
    starts_t = torch.tensor(starts, dtype=torch.int32, device=dev)
    lens_t = torch.tensor(lens, dtype=torch.int32, device=dev)
    scale = 1.0 / (d**0.5)
    gen = torch.Generator(device="cpu").manual_seed(17)

    def _reference(q, k, v, bd, keep_mask, dropout_p):
        outs = []
        for b, n_chunks in enumerate(chunks):
            s0 = starts[b]
            rows = n_chunks * s_rows
            kb = k[s0 : s0 + rows].view(n_chunks, s_rows, n_heads, d)
            vb = v[s0 : s0 + rows].view(n_chunks, s_rows, n_heads, d)
            for c in range(n_chunks):
                keys = [(kc, kp) for kc in range(c - mem, c) for kp in range(c_rows)] + [
                    (c, kp) for kp in range(s_rows)
                ]
                zero = torch.zeros(n_heads, d, device=dev)
                kvec = torch.stack([kb[kc, kp] if kc >= 0 else zero for kc, kp in keys])
                vvec = torch.stack([vb[kc, kp] if kc >= 0 else zero for kc, kp in keys])
                for i in range(s_rows):
                    row = s0 + c * s_rows + i
                    idx = torch.tensor([center + (kc - c) * c_rows + kp - i for kc, kp in keys], device=dev)
                    sc = torch.einsum("hd,lhd->hl", q[row], kvec) * scale + bd[row][:, idx]
                    w = torch.softmax(sc, dim=-1)
                    if keep_mask is not None:
                        w = w * keep_mask[row][:, idx].float() / (1.0 - dropout_p)
                    outs.append(torch.einsum("hl,lhd->hd", w, vvec))
        return torch.stack(outs)

    for dropout_p, seed in [(0.0, 0), (0.3, 999)]:
        leaves = [torch.randn(total, n_heads, d, generator=gen).to(dev).requires_grad_(True) for _ in range(3)]
        bd_leaf = (torch.randn(total, n_heads, r, generator=gen) * 0.5).to(dev).requires_grad_(True)
        out = m.chunked_rel_pos_att(
            *leaves,
            bd_leaf,
            starts_t,
            lens_t,
            max(lens),
            chunk_size=s_rows,
            kept_rows=c_rows,
            history=mem,
            dropout_p=dropout_p,
            seed=seed,
            scale=scale,
        )
        d_out = torch.randn(total, n_heads, d, generator=gen).to(dev)
        out.backward(d_out)
        grads_kernel = [t.grad.clone() for t in leaves] + [bd_leaf.grad.clone()]
        for t in leaves + [bd_leaf]:
            t.grad = None
        keep = None
        if dropout_p:
            keep = m.dump_keep_mask(total, n_heads, r, dropout_p=dropout_p, seed=seed, device=dev)
        ref = _reference(*leaves, bd_leaf, keep, dropout_p)
        numpy.testing.assert_allclose(out.detach().cpu().numpy(), ref.detach().cpu().numpy(), rtol=1e-4, atol=1e-5)
        ref.backward(d_out)
        for g_kernel, t in zip(grads_kernel, leaves + [bd_leaf]):
            numpy.testing.assert_allclose(g_kernel.cpu().numpy(), t.grad.cpu().numpy(), rtol=1e-4, atol=1e-4)


def test_key_range_att_triton_kernel_grad():
    # The kernel where every query row attends one range of key rows: fwd + all grads vs masked energies,
    # at dropout 0 and at dropout > 0 with the kernel's own extracted mask, in f32, bf16 and f16.
    # The ranges overlap, repeat, come in any order, cross block borders, cover all attended keys or nothing,
    # and the head dims are no powers of 2.
    # The rows which take no part (keys nobody attends, queries without keys and their gradient) hold NaN,
    # as gap rows or the tail of a bound-sized packed buffer may: nothing of them may reach the other rows.
    rf.select_backend_torch()
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    try:
        from returnn.torch.util import key_range_att_triton as m
    except ImportError as exc:
        raise unittest.SkipTest(f"triton not available ({exc})")

    dev = "cuda"
    n_q, n_k, n_attended, n_heads = 150, 170, 150, 2
    gen = torch.Generator(device="cpu").manual_seed(21)
    lo = torch.randint(0, n_attended, (n_q,), generator=gen)
    hi = torch.clamp(lo + torch.randint(0, 90, (n_q,), generator=gen), max=n_attended)
    lo[:5], hi[:5] = 0, n_attended
    lo[5:10], hi[5:10] = 30, 30
    lo, hi = lo.to(dev), hi.to(dev)
    cols = torch.arange(n_k, device=dev)
    allowed = (cols[None, :] >= lo[:, None]) & (cols[None, :] < hi[:, None])
    empty_queries, unused_keys = ~allowed.any(1), ~allowed.any(0)
    tolerances = {  # rtol, atol of the output, atol of the grads
        torch.float32: (1e-4, 1e-5, 1e-4),
        torch.bfloat16: (5e-2, 5e-2, 5e-2),
        torch.float16: (5e-3, 5e-3, 5e-3),
    }

    def _reference(q, k, v, keep_mask, dropout_p, scale):
        s = torch.einsum("ihd,jhd->hij", q, k) * scale
        s = torch.where(allowed[None], s, float("-inf"))
        s = torch.where(allowed.any(-1)[None, :, None], s, 0.0)
        w = torch.where(allowed[None], torch.softmax(s, dim=-1), 0.0)
        if keep_mask is not None:
            w = w * keep_mask.permute(1, 0, 2).float() / (1.0 - dropout_p)
        return torch.einsum("hij,jhd->ihd", w, v)

    f32, bf16, f16 = torch.float32, torch.bfloat16, torch.float16
    for d, d_v, dropout_p, seed, dtype in [
        (32, 32, 0.0, 0, f32),
        (32, 32, 0.3, 999, f32),
        (24, 40, 0.0, 0, f32),
        (128, 128, 0.2, 7, f32),
        (32, 32, 0.2, 5, bf16),
        (128, 128, 0.0, 0, f16),
    ]:
        rtol, atol_out, atol_grad = tolerances[dtype]
        scale = d**-0.5
        clean = [
            torch.randn(n, n_heads, dim, generator=gen).to(dev, dtype) for n, dim in ((n_q, d), (n_k, d), (n_k, d_v))
        ]
        leaves = [
            torch.where(rows[:, None, None], float("nan"), x).requires_grad_(True)
            for x, rows in zip(clean, (empty_queries, unused_keys, unused_keys))
        ]
        out = m.key_range_att(*leaves, lo, hi, dropout_p=dropout_p, seed=seed, scale=scale)
        d_out = torch.randn(n_q, n_heads, d_v, generator=gen).to(dev, dtype)
        out.backward(torch.where(empty_queries[:, None, None], float("nan"), d_out))
        refs = [x.float().requires_grad_(True) for x in clean]
        keep = None
        if dropout_p:
            keep = m.dump_keep_mask(n_q, n_heads, n_k, dropout_p=dropout_p, seed=seed, device=dev)
        ref = _reference(*refs, keep, dropout_p, scale)
        where = f"d {d} d_v {d_v} dropout {dropout_p} {dtype}"
        numpy.testing.assert_allclose(
            out.detach().float().cpu().numpy(), ref.detach().cpu().numpy(), rtol=rtol, atol=atol_out, err_msg=where
        )
        ref.backward(d_out.float())
        for t, t_ref in zip(leaves, refs):
            numpy.testing.assert_allclose(
                t.grad.float().cpu().numpy(), t_ref.grad.cpu().numpy(), rtol=rtol, atol=atol_grad, err_msg=where
            )

    # The op traces under AOT autograd (fake tensors), as the compiled step of torch_cuda_graph does it.
    from functorch.compile import aot_function, nop

    def _loss(q_, k_, v_, lo_, hi_):
        return m.key_range_att(q_, k_, v_, lo_, hi_).square().sum()

    leaves = [torch.randn(n, n_heads, 32, generator=gen).to(dev).requires_grad_(True) for n in (n_q, n_k, n_k)]
    traced = aot_function(_loss, fw_compiler=nop, bw_compiler=nop)
    grads = torch.autograd.grad(traced(*leaves, lo, hi), leaves)
    for g, g_ref in zip(grads, torch.autograd.grad(_loss(*leaves, lo, hi), leaves)):
        torch.testing.assert_close(g, g_ref)

    # Under CUDA-graph capture, the default seed is drawn inside the graph, so every replay drops other weights.
    q, k, v = (torch.randn(n, n_heads, 32, generator=gen).to(dev) for n in (n_q, n_k, n_k))
    m.key_range_att(q, k, v, lo, hi, dropout_p=0.3)  # compiles the kernel, which a capture cannot
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = m.key_range_att(q, k, v, lo, hi, dropout_p=0.3)
    replays = []
    for _ in range(2):
        graph.replay()
        replays.append(out.clone())
    assert not torch.equal(replays[0], replays[1])


def test_cast_packed():
    # rf.cast on packed data runs elementwise on the packed buffer (PackedBackend.cast_raw),
    # e.g. from the behavior_version>=27 keep-dtype path of LayerNorm/RMSNorm.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input()
    xp = packed.pack(x)
    out_p = rf.cast(xp, "float64")
    assert packed.is_packed(out_p) and out_p.dtype == "float64"
    _assert_equal_non_padded(out_p, rf.cast(x, "float64"), batch_dim, time_dim)


def test_pack_like_plain_helper_tensors():
    # Helper tensors built on the virtual side (a frame mask, targets regridded onto the frames, positions)
    # can be put on the data's packing, so the ops consuming them work on the packed buffer.
    # Without it, an embedding-style lookup (plain source, per-frame indices) runs on padded storage.
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(5, 1, 4))
    xp = packed.pack(x)
    emb_dim = Dim(2, name="emb")
    emb = Tensor("emb", dims=[feat_dim, emb_dim], dtype="float32")
    emb.raw_tensor = torch.randn(feat_dim.dimension, emb_dim.dimension, generator=torch.Generator().manual_seed(5))
    targets = Tensor("targets", dims=[batch_dim, time_dim], dtype="int32", sparse_dim=feat_dim)
    targets.raw_tensor = torch.randint(0, feat_dim.dimension, (3, 5), dtype=torch.int32)
    ref = rf.gather(emb, indices=targets, axis=feat_dim)
    assert not packed.is_packed(ref)  # plain indices keep the lookup on padded storage

    targets_p = rf.pack_like(targets, xp)
    assert packed.is_packed(targets_p) and targets_p.sparse_dim == feat_dim
    _assert_equal_non_padded(targets_p, targets, batch_dim, time_dim)

    packed._warned_fallback_ops.clear()
    out = rf.gather(emb, indices=targets_p, axis=feat_dim)
    assert packed.is_packed(out) and not packed._warned_fallback_ops
    _assert_equal_non_padded(out, ref, batch_dim, time_dim)

    # a packed source is conformed to the template's layout, and a padded template is a no-op,
    # so the same model code runs packed and padded
    gapped = packed.pack(x, gap=2, align=1)
    conformed = rf.pack_like(targets_p, gapped)
    assert packed.is_packed(conformed) and conformed.raw_tensor.same_packing(gapped.raw_tensor)
    other = Tensor("other", dims=[feat_dim], dtype="float32", raw_tensor=torch.zeros(feat_dim.dimension))
    assert rf.pack_like(other, xp) is other
    assert rf.pack_like(targets, x) is targets


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


def test_cu_seqlens_with_host_lens_and_a_device_total():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("cuda only: needs a second device for the packed total")
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")
    lens = Tensor("lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([5, 3], dtype=torch.int32))
    time_dim = Dim(lens, name="time")
    total = Tensor("total", dims=(), dtype="int32", raw_tensor=torch.tensor(8, dtype=torch.int32, device="cuda"))
    packed_dim = Dim(total, name="packed")
    inner = Tensor("inner", dims=[packed_dim], dtype="float32", raw_tensor=torch.zeros(8, device="cuda"))
    raw = packed.PackedRawTensor(inner=inner, packed_dim=packed_dim, orig_dims=(batch_dim, time_dim))
    cu, _ = raw.cu_seqlens(device="cuda")
    assert cu.raw_tensor.tolist() == [0, 5, 8]


def test_regap_under_cuda_graph_capture():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("cuda only: real graph capture")
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")
    lens = Tensor(
        "lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([5, 3], dtype=torch.int32, device="cuda")
    )
    time_dim = Dim(lens, name="time", capacity=6)
    packed_dim = Dim(16, name="packed")
    inner = Tensor("inner", dims=[packed_dim], dtype="float32", raw_tensor=torch.arange(16.0, device="cuda"))
    x = packed.pack_import(inner, batch_dim=batch_dim, spatial_dim=time_dim, packed_dim=packed_dim)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), rf.set_static_traceable_ctx():
        packed.regap(x, 2, align=1, total_bound=20)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), rf.set_static_traceable_ctx():
        out = packed.regap(x, 2, align=1, total_bound=20)
    graph.replay()
    torch.cuda.synchronize()
    values = out.raw_tensor.inner.raw_tensor.tolist()
    assert values[:5] == [0.0, 1.0, 2.0, 3.0, 4.0] and values[7:10] == [5.0, 6.0, 7.0], values


def test_gather_with_a_static_extra_index_dim_keeps_the_packing():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    xp = packed.pack(x)
    mem_dim = Dim(2, name="mem")
    idx = rf.combine_bc(rf.range_over_dim(time_dim), "+", rf.range_over_dim(mem_dim) - mem_dim.dimension)
    ref = rf.gather(x, indices=idx, axis=time_dim, clip_to_valid=True)
    packed._warned_fallback_ops.clear()
    out = rf.gather(xp, indices=idx, axis=time_dim, clip_to_valid=True)
    assert not packed._warned_fallback_ops, packed._warned_fallback_ops
    assert out.raw_tensor.packed_dim is xp.raw_tensor.packed_dim
    _assert_equal_non_padded(out, ref, batch_dim, time_dim)


def test_shift_along_the_packed_dim_keeps_the_packing():
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    xp = packed.pack(x)
    for shift, amount in ((rf.shift_right, 2), (rf.shift_left, 1)):
        ref = shift(x, axis=time_dim, pad_value=7.0, amount=amount)
        packed._warned_fallback_ops.clear()
        out_p = shift(xp, axis=time_dim, pad_value=7.0, amount=amount)
        assert not packed._warned_fallback_ops, (shift.__name__, packed._warned_fallback_ops)
        assert out_p.raw_tensor.packed_dim is xp.raw_tensor.packed_dim, shift.__name__
        _assert_equal_non_padded(out_p, ref, batch_dim, time_dim)


def test_reduce_over_time_dense_bound_tail():
    """a dense bound-sized buffer has unused rows past the content, which no per-sequence op may count"""
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(seq_lens=(4, 2))
    xp = packed.regap(packed.pack(x), 0, total_bound=10)
    assert xp.raw_tensor.packed_dim.dimension == 10
    for mode in ("mean", "sum", "logsumexp"):
        out_p = rf.reduce(xp, mode=mode, axis=time_dim)
        out_ref = rf.reduce(x, mode=mode, axis=time_dim)
        assert not packed.is_packed(out_p)
        out_p = out_p.copy_compatible_to_dims(out_ref.dims)
        numpy.testing.assert_allclose(
            out_p.raw_tensor.detach().numpy(), out_ref.raw_tensor.detach().numpy(), rtol=1e-5, atol=1e-6, err_msg=mode
        )
    for fn in (rf.softmax, rf.log_softmax):
        out_p = fn(xp, axis=time_dim)
        assert packed.is_packed(out_p), fn.__name__
        _assert_equal_non_padded(out_p, fn(x, axis=time_dim), batch_dim, time_dim)


def test_batch_norm_packed_dense_bound_train():
    """batch_norm statistics ignore the unused tail of a dense bound-sized buffer"""
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(seq_lens=(5, 3), feat=4, seed=8)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(3)
        bn_dense = rf.BatchNorm(feat_dim, use_mask=False)
        bn_bound = rf.BatchNorm(feat_dim, use_mask=False)
        with rf.get_run_ctx().train_flag_ctx(True):
            out_dense = bn_dense(packed.pack(x))
            out_bound = bn_bound(packed.regap(packed.pack(x), 0, total_bound=16))
        assert packed.is_packed(out_bound)
    _assert_equal_non_padded(out_bound, packed.unpack(out_dense), batch_dim, time_dim)
    for p_dense, p_bound in [
        (bn_dense.running_mean, bn_bound.running_mean),
        (bn_dense.running_variance, bn_bound.running_variance),
    ]:
        numpy.testing.assert_allclose(
            p_dense.raw_tensor.detach().numpy(), p_bound.raw_tensor.detach().numpy(), rtol=1e-5, atol=1e-6
        )


def test_gather_per_seq_index_drops_the_time_dim():
    """indices without the gathered time dim select frames per sequence, so the result has no time dim"""
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    xp = packed.pack(x, gap=2)
    k_dim = Dim(2, name="k")
    idx_b = Tensor("idx", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([6, 0, 3], dtype=torch.int32))
    idx_bk = Tensor(
        "idx",
        dims=[batch_dim, k_dim],
        dtype="int32",
        raw_tensor=torch.tensor([[6, 1], [0, 4], [3, 2]], dtype=torch.int32),
    )
    for idx in (idx_b, idx_bk):
        ref = rf.gather(x, indices=idx, axis=time_dim)
        out = rf.gather(xp, indices=idx, axis=time_dim)
        assert time_dim not in out.dims, (idx.dims, out.dims)
        out = packed.unpack(out) if packed.is_packed(out) else out
        out = out.copy_compatible_to_dims(ref.dims)
        numpy.testing.assert_allclose(out.raw_tensor.detach().numpy(), ref.raw_tensor.detach().numpy(), rtol=1e-6)
    vocab = Dim(9, name="vocab")
    codes = Tensor("codes", dims=[batch_dim, time_dim], dtype="int32")
    codes.raw_tensor = torch.arange(21, dtype=torch.int32).reshape(3, 7) % 9
    codes_p = rf.set_sparse_dim(packed.pack(codes, gap=2), vocab)
    out = rf.gather(codes_p, indices=idx_b, axis=time_dim)
    assert out.sparse_dim == vocab, out
    ref = rf.gather(rf.set_sparse_dim(codes, vocab), indices=idx_b, axis=time_dim)
    numpy.testing.assert_array_equal(out.raw_tensor.numpy(), ref.raw_tensor.numpy())


def test_gather_with_indices_packed_over_another_time_dim():
    """
    indices packed over their own time dim select frames of the source per sequence,
    so the result takes the packing of the indices, e.g. label states gathered onto the frames of an alignment
    """
    rf.select_backend_torch()
    x, batch_dim, label_dim, feat_dim = _make_input(batch_size=3, seq_lens=(4, 2, 3))
    frame_dim = Dim(
        Tensor("frames", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([6, 7, 3], dtype=torch.int32)),
        name="frames",
    )
    idx = Tensor("idx", dims=[batch_dim, frame_dim], dtype="int32", sparse_dim=label_dim)
    idx.raw_tensor = torch.tensor(
        [[0, 0, 1, 2, 3, 3, 0], [0, 0, 0, 1, 1, 1, 1], [2, 1, 0, 0, 0, 0, 0]], dtype=torch.int32
    )
    beyond = Tensor("idx_beyond", dims=[batch_dim, frame_dim], dtype="int32", sparse_dim=label_dim)
    beyond.raw_tensor = idx.raw_tensor + 2
    for source_gap, index_gap in ((0, 0), (2, 0), (0, 3)):
        xp = packed.pack(x, gap=source_gap)
        for indices, clip_to_valid in ((idx, False), (beyond, True)):
            indices_p = packed.pack(indices, gap=index_gap)
            ref = rf.gather(x, indices=indices, axis=label_dim, clip_to_valid=clip_to_valid)
            packed._warned_fallback_ops.clear()
            out = rf.gather(xp, indices=indices_p, axis=label_dim, clip_to_valid=clip_to_valid)
            assert not packed._warned_fallback_ops, packed._warned_fallback_ops
            assert out.dims == ref.dims, (out.dims, ref.dims)
            assert out.raw_tensor.packed_dim is indices_p.raw_tensor.packed_dim
            _assert_equal_non_padded(out, ref, batch_dim, frame_dim)

    # the captured regime: bound-sized buffers on both sides, the result keeps the bound of the indices
    label_dim.capacity, frame_dim.capacity = 4, 7
    xp = packed.pack(x, total_bound=16)
    indices_p = packed.pack(beyond, total_bound=24)
    with rf.set_static_traceable_ctx():
        out = rf.gather(xp, indices=indices_p, axis=label_dim, clip_to_valid=True)
    assert out.raw_tensor.packed_dim.dimension == 24, out.raw_tensor
    ref = rf.gather(x, indices=beyond, axis=label_dim, clip_to_valid=True)
    _assert_equal_non_padded(out, ref, batch_dim, frame_dim)

    # the gradient reaches the source rows the valid frames read, the padded frames of the reference are masked out
    x.raw_tensor.requires_grad_(True)
    valid = (torch.arange(7)[None, :] < frame_dim.dyn_size_ext.raw_tensor[:, None])[:, :, None]
    ref = rf.gather(x, indices=idx, axis=label_dim)
    (grad_ref,) = torch.autograd.grad(((ref.raw_tensor * valid) ** 2).sum(), x.raw_tensor)
    out = packed.unpack(rf.gather(packed.pack(x), indices=packed.pack(idx), axis=label_dim))
    out_raw = out.copy_compatible_to_dims(ref.dims).raw_tensor
    (grad,) = torch.autograd.grad(((out_raw * valid) ** 2).sum(), x.raw_tensor)
    numpy.testing.assert_allclose(grad.numpy(), grad_ref.numpy(), rtol=1e-6)


def test_gather_with_an_index_time_dim_which_the_source_carries_too():
    """
    a dim shared by source and indices is a batch dim of the gather: frame t reads column t, not every column.
    Here the indices bring this dim as their own time dim, while the packed source carries it as a plain dim
    (e.g. a label by frame lattice read along an alignment), and the result is packed over the frames.
    """
    rf.select_backend_torch()
    batch_dim = Dim(3, name="batch")
    label_dim, frame_dim = (
        Dim(Tensor(name, dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(lens, dtype=torch.int32)), name=name)
        for name, lens in (("labels", [3, 2, 4]), ("frames", [4, 6, 3]))
    )
    feat_dim = Dim(2, name="feat")
    x = Tensor("x", dims=[batch_dim, label_dim, frame_dim, feat_dim], dtype="float32")
    x.raw_tensor = torch.randn(3, 4, 6, 2, generator=torch.Generator().manual_seed(1))
    idx = Tensor("idx", dims=[batch_dim, frame_dim], dtype="int32", sparse_dim=label_dim)
    idx.raw_tensor = torch.tensor([[0, 1, 2, 2, 0, 0], [0, 0, 1, 1, 1, 0], [3, 1, 0, 0, 0, 0]], dtype=torch.int32)
    beyond = Tensor("idx_beyond", dims=[batch_dim, frame_dim], dtype="int32", sparse_dim=label_dim)
    beyond.raw_tensor = idx.raw_tensor + 3
    for source_gap, index_gap in ((0, 0), (2, 3)):
        xp = packed.pack(x, dims=[batch_dim, label_dim], gap=source_gap)
        for plain, clip_to_valid in ((idx, False), (beyond, True)):
            ref = rf.gather(x, indices=plain, axis=label_dim, clip_to_valid=clip_to_valid)
            assert ref.dims_set == {batch_dim, frame_dim, feat_dim}
            for indices in (plain, packed.pack(plain, dims=[batch_dim, frame_dim], gap=index_gap)):
                packed._warned_fallback_ops.clear()
                out = rf.gather(xp, indices=indices, axis=label_dim, clip_to_valid=clip_to_valid)
                assert not packed._warned_fallback_ops, packed._warned_fallback_ops
                assert packed.is_packed(out) and out.raw_tensor.orig_dims == (batch_dim, frame_dim)
                assert out.dims_set == ref.dims_set, (packed.is_packed(indices), out.dims, ref.dims)
                _assert_equal_non_padded(out, ref, batch_dim, frame_dim)

    # the captured regime: bound-sized buffers, the columns come from the declared capacity
    label_dim.capacity, frame_dim.capacity = 4, 6
    xp = packed.pack(x, dims=[batch_dim, label_dim], total_bound=12)
    indices_p = packed.pack(idx, dims=[batch_dim, frame_dim], total_bound=20)
    with rf.set_static_traceable_ctx():
        out = rf.gather(xp, indices=indices_p, axis=label_dim)
    assert out.raw_tensor.packed_dim.dimension == 20, out.raw_tensor
    _assert_equal_non_padded(out, rf.gather(x, indices=idx, axis=label_dim), batch_dim, frame_dim)

    # the gradient reaches exactly the cells the valid frames read
    x.raw_tensor.requires_grad_(True)
    valid = (torch.arange(6)[None, :] < frame_dim.dyn_size_ext.raw_tensor[:, None])[:, :, None]
    ref = rf.gather(x, indices=idx, axis=label_dim)
    ref_raw = ref.copy_transpose([batch_dim, frame_dim, feat_dim]).raw_tensor
    (grad_ref,) = torch.autograd.grad(((ref_raw * valid) ** 2).sum(), x.raw_tensor)
    out = rf.gather(packed.pack(x, dims=[batch_dim, label_dim]), indices=packed.pack(idx), axis=label_dim)
    out_raw = packed.unpack(out).copy_transpose([batch_dim, frame_dim, feat_dim]).raw_tensor
    (grad,) = torch.autograd.grad(((out_raw * valid) ** 2).sum(), x.raw_tensor)
    numpy.testing.assert_allclose(grad.numpy(), grad_ref.numpy(), rtol=1e-6)


def test_scatter_along_the_packed_dim_into_another_time_dim():
    """
    the inverse of the re-laid-out gather: every frame of the packed source writes to a position of its own
    sequence along a new time dim, and the result is packed over that dim,
    e.g. audio frames written to their places in a longer stream which other frames fill up
    """
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(4, 2, 3))
    out_dim = Dim(
        Tensor("out_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([7, 3, 6], dtype=torch.int32)),
        name="out",
    )
    idx = Tensor("idx", dims=[batch_dim, time_dim], dtype="int32", sparse_dim=out_dim)
    # the last sequence writes two frames to one position, the padded entries of the others point anywhere
    idx.raw_tensor = torch.tensor([[0, 2, 3, 6], [1, 2, 6, 6], [0, 0, 5, 1]], dtype=torch.int32)
    for gap in (0, 2):
        xp = packed.pack(x, gap=gap)
        for indices in (idx, packed.pack(idx, gap=gap)):
            for mode in ("sum", "max"):
                ref = rf.scatter(x, indices=idx, indices_dim=time_dim, out_dim=out_dim, mode=mode)
                packed._warned_fallback_ops.clear()
                out = rf.scatter(xp, indices=indices, indices_dim=time_dim, out_dim=out_dim, mode=mode)
                assert not packed._warned_fallback_ops, packed._warned_fallback_ops
                assert packed.is_packed(out) and out.raw_tensor.orig_dims == (batch_dim, out_dim), out
                assert out.dims_set == ref.dims_set, (out.dims, ref.dims)
                _assert_equal_non_padded(out, ref, batch_dim, out_dim)

    # the captured regime: a bound-sized source, the result gets a static buffer from the capacities
    time_dim.capacity, out_dim.capacity = 4, 7
    with rf.set_static_traceable_ctx():
        out = rf.scatter(packed.pack(x, total_bound=12), indices=idx, indices_dim=time_dim, out_dim=out_dim)
    assert out.raw_tensor.packed_dim.dimension is not None, out.raw_tensor
    _assert_equal_non_padded(out, rf.scatter(x, indices=idx, indices_dim=time_dim, out_dim=out_dim), batch_dim, out_dim)

    # the gradient of a written position goes back to the frames which wrote it
    x.raw_tensor.requires_grad_(True)
    weight = torch.randn(3, 7, 4, generator=torch.Generator().manual_seed(5))
    valid = (torch.arange(7)[None, :] < out_dim.dyn_size_ext.raw_tensor[:, None])[:, :, None]
    ref = rf.scatter(x, indices=idx, indices_dim=time_dim, out_dim=out_dim)
    ref_raw = ref.copy_transpose([batch_dim, out_dim, feat_dim]).raw_tensor
    (grad_ref,) = torch.autograd.grad((ref_raw * weight * valid).sum(), x.raw_tensor)
    out = rf.scatter(packed.pack(x), indices=idx, indices_dim=time_dim, out_dim=out_dim)
    out_raw = packed.unpack(out).copy_transpose([batch_dim, out_dim, feat_dim]).raw_tensor
    (grad,) = torch.autograd.grad((out_raw * weight * valid).sum(), x.raw_tensor)
    numpy.testing.assert_allclose(grad.numpy(), grad_ref.numpy(), rtol=1e-6)


def _seqs(name: str, batch_dim: Dim, lens, values, **kwargs) -> Tuple[Tensor, Dim]:
    """:return: a [batch, time] tensor with the given per-sequence values (rows padded with zeros), and its time dim"""
    time_dim = Dim(
        Tensor(f"{name}_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(lens, dtype=torch.int32)),
        name=f"{name}_time",
    )
    width = max(max(lens), 1)
    raw = torch.tensor([list(row) + [0] * (width - len(row)) for row in values])
    x = Tensor(name, dims=[batch_dim, time_dim], dtype="int32" if raw.dtype == torch.int64 else "float32", **kwargs)
    x.raw_tensor = raw.to(torch.int32 if raw.dtype == torch.int64 else torch.float32)[:, : max(lens)]
    return x, time_dim


def test_dot_attention_over_own_group_stays_packed():
    """
    a query which attends only the keys of its own group, e.g. a label over the encoder frames of its chunk.
    With max_group_size every query reads its own run of keys, which needs no lattice over (queries, keys),
    so packed keys are never unpacked, and the result is packed like the queries
    """
    from test_rf_attention import _grouped_attention_inputs

    rf.select_backend_torch()
    query, keys, values, query_group, key_group, dims = _grouped_attention_inputs()
    batch_dim, q_time, kv_time, heads, feat, v_feat = dims
    opts = dict(key_dim=feat, axis=kv_time, query_group=query_group, key_group=key_group, max_group_size=3)
    keys.raw_tensor.requires_grad_(True)
    weight = torch.randn(2, 4, 2, 6, generator=torch.Generator().manual_seed(6))
    valid = (torch.arange(4)[None, :] < q_time.dyn_size_ext.raw_tensor[:, None])[:, :, None, None]
    order = [batch_dim, q_time, heads, v_feat]

    ref = rf.dot_attention(query, keys, values, **opts)
    (ref_grad,) = torch.autograd.grad((ref.copy_transpose(order).raw_tensor * weight * valid).sum(), keys.raw_tensor)
    for gap in (0, 2):
        for q in (query, packed.pack(query, gap=gap)):
            packed._warned_fallback_ops.clear()
            out = rf.dot_attention(q, packed.pack(keys, gap=gap), packed.pack(values, gap=gap), **opts)
            assert not packed._warned_fallback_ops, (gap, packed._warned_fallback_ops)
            assert packed.is_packed(out) and out.raw_tensor.orig_dims == (batch_dim, q_time), out
            _assert_equal_non_padded(out, ref, batch_dim, q_time)
            out_raw = packed.unpack(out).copy_transpose(order).raw_tensor
            (grad,) = torch.autograd.grad((out_raw * weight * valid).sum(), keys.raw_tensor)
            numpy.testing.assert_allclose(grad.numpy(), ref_grad.numpy(), rtol=1e-5, atol=1e-6)

    # the groups themselves can be packed, of the queries (e.g. computed from packed labels), of the keys, or both
    key_group_batch = rf.expand_dim(key_group, dim=batch_dim)
    for gap in (0, 2):
        for q_group in (query_group, packed.pack(query_group, gap=gap)):
            for k_group in (key_group, key_group_batch, packed.pack(key_group_batch, gap=gap)):
                packed._warned_fallback_ops.clear()
                out = rf.dot_attention(
                    packed.pack(query, gap=gap),
                    packed.pack(keys, gap=gap),
                    packed.pack(values, gap=gap),
                    **{**opts, "query_group": q_group, "key_group": k_group},
                )
                assert not packed._warned_fallback_ops, (gap, q_group, k_group, packed._warned_fallback_ops)
                assert packed.is_packed(out), (gap, q_group, k_group)
                _assert_equal_non_padded(out, ref, batch_dim, q_time)

    # the captured regime: bound-sized keys, every shape comes from a capacity or a bound
    q_time.capacity, kv_time.capacity = 4, 7
    with rf.set_static_traceable_ctx():
        bound = dict(total_bound=16)
        out = rf.dot_attention(query, packed.pack(keys, **bound), packed.pack(values, **bound), **opts)
    assert packed.is_packed(out) and out.raw_tensor.packed_dim.dimension is not None, out.raw_tensor
    _assert_equal_non_padded(out, ref, batch_dim, q_time)


def test_dot_attention_group_ranges_stay_packed():
    """
    "less_equal", and "equal" without a group size, give every query one range of keys (the key groups are sorted),
    which the packed backend hands to the Triton kernel of key_range_att_triton on cuda,
    so packed keys are never unpacked, packed or plain queries and groups alike
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    from test_rf_attention import _grouped_attention_inputs

    rf.select_backend_torch()
    query, keys, values, query_group, key_group, dims = _grouped_attention_inputs("cuda")
    batch_dim, q_time, kv_time, heads, feat, v_feat = dims
    keys.raw_tensor.requires_grad_(True)
    weight = torch.randn(2, 4, 2, 6, generator=torch.Generator().manual_seed(6)).cuda()
    valid = (torch.arange(4)[None, :] < q_time.dyn_size_ext.raw_tensor[:, None])[:, :, None, None].cuda()
    order = [batch_dim, q_time, heads, v_feat]
    for mode in ("equal", "less_equal"):
        opts = dict(key_dim=feat, axis=kv_time, key_group=key_group, group_mode=mode)
        ref = rf.dot_attention(query, keys, values, query_group=query_group, **opts)
        ref_raw = ref.copy_transpose(order).raw_tensor
        (ref_grad,) = torch.autograd.grad((ref_raw * weight * valid).sum(), keys.raw_tensor)
        for gap in (0, 2):
            for q, q_group in ((query, query_group), (packed.pack(query, gap=gap), packed.pack(query_group, gap=gap))):
                packed._warned_fallback_ops.clear()
                packed.attention_path_counts.clear()
                out = rf.dot_attention(
                    q, packed.pack(keys, gap=gap), packed.pack(values, gap=gap), query_group=q_group, **opts
                )
                where = (mode, gap, packed.is_packed(q))
                assert not packed._warned_fallback_ops, (where, packed._warned_fallback_ops)
                assert packed.attention_path_counts == {"key_range_triton": 1}, (where, packed.attention_path_counts)
                assert packed.is_packed(out) and out.raw_tensor.orig_dims == (batch_dim, q_time), (where, out)
                _assert_equal_non_padded(out, ref, batch_dim, q_time)
                out_raw = packed.unpack(out).copy_transpose(order).raw_tensor
                (grad,) = torch.autograd.grad((out_raw * weight * valid).sum(), keys.raw_tensor)
                numpy.testing.assert_allclose(grad.cpu().numpy(), ref_grad.cpu().numpy(), rtol=1e-5, atol=1e-6)


def test_dot_attention_group_ranges_without_the_kernel():
    """
    where the kernel does not run (cpu), group ranges over packed operands take the attention fallback,
    once allowed, with the result of plain tensors
    """
    from test_rf_attention import _grouped_attention_inputs

    rf.select_backend_torch()
    query, keys, values, query_group, key_group, dims = _grouped_attention_inputs()
    batch_dim, q_time, kv_time, heads, feat, v_feat = dims
    packed.set_allowed_fallbacks(["scaled_dot_product_attention"])
    try:
        for mode in ("equal", "less_equal"):
            opts = dict(key_dim=feat, axis=kv_time, key_group=key_group, group_mode=mode)
            ref = rf.dot_attention(query, keys, values, query_group=query_group, **opts)
            q, k, v, q_group = (packed.pack(x) for x in (query, keys, values, query_group))
            out = rf.dot_attention(q, k, v, query_group=q_group, **opts)
            _assert_equal_non_padded(out, ref, batch_dim, q_time)
    finally:
        packed.set_allowed_fallbacks(None)


def test_dot_attention_over_own_group_with_device_lens():
    """
    the regime of a captured train step: static buffers, the lengths live on the device,
    and one graph over forward and backward has to serve every batch, an empty sequence included.
    Packed keys, the queries once plain (a padded decoder stream) and once packed.
    With a group size every query reads its own run of keys, without one (or with "less_equal")
    every query gets a range of keys, which only the cuda kernel serves packed.
    Without cuda the traced step runs per batch instead of being replayed.
    """
    rf.select_backend_torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n_seqs, q_cap, kv_cap, n_heads, n_feat, n_v_feat, group_size = 3, 5, 7, 2, 4, 3, 3
    q_bound, kv_bound = 17, 23
    batch_dim = Dim(n_seqs, name="batch")
    heads, feat, v_feat = Dim(n_heads, name="heads"), Dim(n_feat, name="feat"), Dim(n_v_feat, name="v_feat")
    q_lens_buf, kv_lens_buf = (torch.zeros(n_seqs, dtype=torch.int32, device=dev) for _ in range(2))
    q_time, kv_time = (
        Dim(Tensor(name, dims=[batch_dim], dtype="int32", raw_tensor=buf), name=name, capacity=cap)
        for name, buf, cap in (("q_time", q_lens_buf, q_cap), ("kv_time", kv_lens_buf, kv_cap))
    )
    q_packed_dim, kv_packed_dim = Dim(q_bound, name="packed_q"), Dim(kv_bound, name="packed_kv")

    # the static buffers the graph reads, every batch is copied into them
    q_pad_buf = torch.zeros(n_seqs, q_cap, n_heads, n_feat, device=dev, requires_grad=True)
    q_buf = torch.zeros(q_bound, n_heads, n_feat, device=dev, requires_grad=True)
    k_buf = torch.zeros(kv_bound, n_heads, n_feat, device=dev, requires_grad=True)
    v_buf = torch.zeros(kv_bound, n_heads, n_v_feat, device=dev, requires_grad=True)
    group_buf = torch.zeros(n_seqs, q_cap, dtype=torch.int32, device=dev)
    group_flat_buf = torch.zeros(q_bound, dtype=torch.int32, device=dev)
    gen = torch.Generator().manual_seed(8)
    weight = torch.randn(n_heads, n_v_feat, generator=gen).to(dev)
    leaves = (q_pad_buf, q_buf, k_buf, v_buf)

    def _import(name, raw, packed_dim, spatial_dim, dims, dtype="float32"):
        flat = Tensor(name, dims=[packed_dim] + dims, dtype=dtype, raw_tensor=raw)
        return packed.pack_import(flat, batch_dim=batch_dim, spatial_dim=spatial_dim, packed_dim=packed_dim)

    def _total(out: Tensor) -> torch.Tensor:
        """a readout which needs no layout: the rows outside the sequences hold copies and must not count"""
        rows = out.raw_tensor.inner.copy_transpose([out.raw_tensor.packed_dim, heads, v_feat]).raw_tensor
        n_rows = q_lens_buf.sum()
        in_seq = torch.arange(rows.shape[0], device=dev) < n_rows
        return (rows * weight * in_seq[:, None, None]).sum()

    def _step(mode, max_group_size):
        # as the engine before every step, see returnn.torch.util.graph_capture
        for dim, lens_buf in ((q_time, q_lens_buf), (kv_time, kv_lens_buf)):
            dim.reset_eager()
            dim.dyn_size_ext.raw_tensor = lens_buf
        keys = _import("k", k_buf, kv_packed_dim, kv_time, [heads, feat])
        values = _import("v", v_buf, kv_packed_dim, kv_time, [heads, v_feat])
        q_plain = Tensor("q", dims=[batch_dim, q_time, heads, feat], dtype="float32", raw_tensor=q_pad_buf)
        query_group = Tensor("q_group", dims=[batch_dim, q_time], dtype="int32", raw_tensor=group_buf)
        # packed queries come with packed groups, as when both derive from packed labels
        q_packed = _import("q", q_buf, q_packed_dim, q_time, [heads, feat])
        query_group_packed = _import("q_group", group_flat_buf, q_packed_dim, q_time, [], dtype="int32")
        totals = []
        with rf.set_static_traceable_ctx():
            key_group = rf.range_over_dim(kv_time, device=dev) // group_size
            for query, group in ((q_plain, query_group), (q_packed, query_group_packed)):
                out = rf.dot_attention(
                    query,
                    keys,
                    values,
                    key_dim=feat,
                    axis=kv_time,
                    query_group=group,
                    key_group=key_group,
                    group_mode=mode,
                    max_group_size=max_group_size,
                )
                assert packed.is_packed(out) and out.raw_tensor.packed_dim.dimension is not None, out.raw_tensor
                totals.append(_total(out))
        (totals[0] + 2.0 * totals[1]).backward()
        return totals

    def _load(q_lens, kv_lens, seed):
        """:return: the padded data of this batch, after writing it densely packed into the static buffers"""
        g = torch.Generator().manual_seed(seed)
        q_pad = torch.randn(n_seqs, q_cap, n_heads, n_feat, generator=g)
        k_pad = torch.randn(n_seqs, kv_cap, n_heads, n_feat, generator=g)
        v_pad = torch.randn(n_seqs, kv_cap, n_heads, n_v_feat, generator=g)
        groups = torch.zeros(n_seqs, q_cap, dtype=torch.int32)
        for b, (n_q, n_kv) in enumerate(zip(q_lens, kv_lens)):
            if n_q:
                groups[b, :n_q] = torch.randint(0, -(-n_kv // group_size), (n_q,), generator=g, dtype=torch.int32)
        with torch.no_grad():
            for buf in (q_buf, k_buf, v_buf, group_flat_buf):
                buf.zero_()
            q_pad_buf.copy_(q_pad)
            group_buf.copy_(groups)
            group_flat_buf[: sum(q_lens)] = torch.cat([groups[b, :n] for b, n in enumerate(q_lens)]).to(dev)
            q_buf[: sum(q_lens)] = torch.cat([q_pad[b, :n] for b, n in enumerate(q_lens)]).to(dev)
            k_buf[: sum(kv_lens)] = torch.cat([k_pad[b, :n] for b, n in enumerate(kv_lens)]).to(dev)
            v_buf[: sum(kv_lens)] = torch.cat([v_pad[b, :n] for b, n in enumerate(kv_lens)]).to(dev)
            q_lens_buf.copy_(torch.tensor(q_lens, dtype=torch.int32))
            kv_lens_buf.copy_(torch.tensor(kv_lens, dtype=torch.int32))
        return q_pad, k_pad, v_pad, groups

    def _reference(q_pad, k_pad, v_pad, groups, q_lens, kv_lens, mode):
        """the same attention through masked energies, on plain padded tensors"""

        def _time(name, lens):
            lens = Tensor(name, dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor(lens, dtype=torch.int32))
            return Dim(lens, name=name)

        q_t, kv_t = _time("q_ref", q_lens), _time("kv_ref", kv_lens)
        n_q, n_kv = max(max(q_lens), 1), max(max(kv_lens), 1)
        raws = [x.clone().requires_grad_(True) for x in (q_pad[:, :n_q], k_pad[:, :n_kv], v_pad[:, :n_kv])]
        q, k, v = (
            Tensor(name, dims=dims, dtype="float32", raw_tensor=raw)
            for name, dims, raw in zip(
                "qkv", ([batch_dim, q_t, heads, feat], [batch_dim, kv_t, heads, feat], [batch_dim, kv_t, heads, v_feat]), raws
            )
        )
        query_group = Tensor("q_group", dims=[batch_dim, q_t], dtype="int32", raw_tensor=groups[:, :n_q].clone())
        key_group = rf.range_over_dim(kv_t) // group_size
        out = rf.dot_attention(
            q, k, v, key_dim=feat, axis=kv_t, query_group=query_group, key_group=key_group, group_mode=mode
        )
        out_raw = out.copy_transpose([batch_dim, q_t, heads, v_feat]).raw_tensor
        valid = (torch.arange(n_q)[None, :] < torch.tensor(q_lens)[:, None])[:, :, None, None]
        total = torch.where(valid, out_raw * weight.cpu(), torch.zeros(())).sum()
        q_grad, k_grad, v_grad = torch.autograd.grad(total, raws)

        def _rows(grad, lens):
            return torch.cat([grad[b, :n] for b, n in enumerate(lens)])

        return total.detach(), q_grad * valid, _rows(q_grad, q_lens), _rows(k_grad, kv_lens), _rows(v_grad, kv_lens)

    # an empty last sequence is what the rows behind the content of a bound-sized buffer get mapped to
    batches = [
        ((5, 2, 4), (7, 3, 5), 1),
        ((1, 5, 5), (2, 7, 6), 2),
        ((3, 0, 2), (4, 0, 1), 3),
        ((3, 2, 0), (4, 1, 0), 4),
        ((5, 5, 5), (7, 7, 7), 5),
    ]
    settings = [("equal", group_size)] + ([("equal", None), ("less_equal", None)] if dev == "cuda" else [])
    for mode, max_group_size in settings:
        _load(*batches[0])
        if dev == "cuda":
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                _step(mode, max_group_size)
            torch.cuda.current_stream().wait_stream(side)
            for leaf in leaves:
                leaf.grad = None
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                totals_static = _step(mode, max_group_size)

        for q_lens, kv_lens, seed in batches:
            q_pad, k_pad, v_pad, groups = _load(q_lens, kv_lens, seed)
            if dev == "cuda":
                graph.replay()
                torch.cuda.synchronize()
                totals = totals_static
            else:
                for leaf in leaves:
                    leaf.grad = None
                totals = _step(mode, max_group_size)
            ref_total, ref_q_pad, ref_q, ref_k, ref_v = _reference(q_pad, k_pad, v_pad, groups, q_lens, kv_lens, mode)
            where = f"{mode} {max_group_size} lens {q_lens} {kv_lens}"
            for total in totals:
                numpy.testing.assert_allclose(total.detach().cpu(), ref_total, rtol=1e-5, err_msg=where)
            n_q, n_kv = sum(q_lens), sum(kv_lens)
            width = ref_q_pad.shape[1]
            q_pad_grad = q_pad_buf.grad[:, :width].cpu()
            numpy.testing.assert_allclose(q_pad_grad, ref_q_pad, rtol=1e-5, atol=1e-6, err_msg=where)
            numpy.testing.assert_allclose(q_buf.grad[:n_q].cpu(), 2.0 * ref_q, rtol=1e-5, atol=1e-6, err_msg=where)
            numpy.testing.assert_allclose(k_buf.grad[:n_kv].cpu(), 3.0 * ref_k, rtol=1e-5, atol=1e-6, err_msg=where)
            numpy.testing.assert_allclose(v_buf.grad[:n_kv].cpu(), 3.0 * ref_v, rtol=1e-5, atol=1e-6, err_msg=where)
            for leaf in leaves:
                assert torch.isfinite(leaf.grad).all(), where


def test_masked_select_static_buffer_follows_a_declared_capacity():
    """
    under static tracing a selection along the packed dim gets a static buffer.
    The content of the source bounds it, and so does a capacity declared on the result dim:
    it promises how much one sequence can select (the padded buffer relies on the same promise),
    e.g. the few labels among many frames
    """
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")
    x, time_dim = _seqs("x", batch_dim, [5, 3], [[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    keep, _ = _seqs("keep", batch_dim, [5, 3], [[1, 0, 0, 1, 0], [0, 1, 0]])
    keep, _ = rf.replace_dim(keep, in_dim=keep.dims[1], out_dim=time_dim)
    time_dim.capacity = 5
    # without a capacity of its own the result takes the one of the source dim
    for capacity, rows in ((None, 2 * 5), (2, 2 * 2)):
        for mask in (keep > 0, packed.pack(keep > 0, total_bound=12)):
            out_dim = Dim(None, name="selected", capacity=capacity)
            with rf.set_static_traceable_ctx():
                out, _ = rf.masked_select(packed.pack(x, total_bound=12), mask=mask, dims=[time_dim], out_dim=out_dim)
            assert out.raw_tensor.packed_dim.dimension == out.raw_tensor.content_bound == rows, out.raw_tensor
            got = packed.unpack(out).copy_compatible_to_dims_raw([batch_dim, out_dim])
            assert got[0, :2].tolist() == [1.0, 4.0] and got[1, :1].tolist() == [7.0], got


def test_scatter_relayout_static_buffer_holds_every_result():
    """
    under static tracing the result buffer has to hold any result the capacities allow:
    the lengths of the result are independent of those of the source, so the size of the source bounds nothing
    """
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")
    x, time_dim = _seqs("x", batch_dim, [1, 1], [[10.0], [20.0]])
    out_dim = Dim(
        Tensor("out_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([6, 6], dtype=torch.int32)),
        name="out",
    )
    idx = Tensor("idx", dims=[batch_dim, time_dim], dtype="int32", sparse_dim=out_dim)
    idx.raw_tensor = torch.tensor([[5], [5]], dtype=torch.int32)
    time_dim.capacity, out_dim.capacity = 4, 6
    with rf.set_static_traceable_ctx():
        out = rf.scatter(packed.pack(x, total_bound=2), indices=idx, indices_dim=time_dim, out_dim=out_dim)
    assert out.raw_tensor.packed_dim.dimension >= 12, out.raw_tensor
    got = packed.unpack(out).copy_compatible_to_dims_raw([batch_dim, out_dim])
    assert got.tolist() == [[0, 0, 0, 0, 0, 10.0], [0, 0, 0, 0, 0, 20.0]], got

    # a result dim which is a sum containing the source dim is bounded by the source content plus the rest
    rest_dim = Dim(
        Tensor("rest_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([2, 1], dtype=torch.int32)),
        name="rest",
        capacity=3,
    )
    with rf.set_static_traceable_ctx():
        out = rf.scatter(
            packed.pack(x, total_bound=5),
            indices=rf.zeros_like(idx),
            indices_dim=time_dim,
            out_dim=time_dim + rest_dim,
        )
    assert out.raw_tensor.packed_dim.dimension == 5 + 2 * 3, out.raw_tensor


def test_scatter_relayout_only_valid_frames_write_into_their_own_sequence():
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")

    # a position beyond the result length of its own sequence is padding, it must not reach the next sequence
    x, time_dim = _seqs("x", batch_dim, [2, 2], [[10.0, 30.0], [20.0, 40.0]])
    out_dim = Dim(
        Tensor("out_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([1, 2], dtype=torch.int32)),
        name="out",
    )
    idx, _ = _seqs("idx", batch_dim, [2, 2], [[0, 1], [0, 1]])
    idx, _ = rf.replace_dim(idx, in_dim=idx.dims[1], out_dim=time_dim)
    out = rf.scatter(packed.pack(x), indices=idx, indices_dim=time_dim, out_dim=out_dim)
    got = packed.unpack(out).copy_compatible_to_dims_raw([batch_dim, out_dim])
    assert got[0, :1].tolist() == [10.0] and got[1].tolist() == [20.0, 40.0], got

    # gap frames write nothing, so an explicit fill value survives at the untouched positions
    out_dim = Dim(
        Tensor("out_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([3, 3], dtype=torch.int32)),
        name="out",
    )
    for mode in ("sum", "max", "min"):
        ref = rf.scatter(x, indices=idx, indices_dim=time_dim, out_dim=out_dim, mode=mode, fill_value=7, use_mask=False)
        for indices in (idx, packed.pack(idx, gap=2)):
            out = rf.scatter(
                packed.pack(x, gap=2),
                indices=indices,
                indices_dim=time_dim,
                out_dim=out_dim,
                mode=mode,
                fill_value=7,
                use_mask=False,
            )
            got = packed.unpack(out).copy_compatible_to_dims_raw([batch_dim, out_dim])
            assert got.tolist() == ref.raw_tensor.tolist() == [[10.0, 30.0, 7.0], [20.0, 40.0, 7.0]], (mode, got)

    # an empty result, while the source still has gap or reserved rows which would write
    _, empty_time = _seqs("empty", batch_dim, [0, 0], [[], []])
    empty = Tensor("empty", dims=[batch_dim, empty_time], dtype="float32", raw_tensor=torch.zeros(2, 0))
    no_idx = Tensor("idx", dims=[batch_dim, empty_time], dtype="int32", raw_tensor=torch.zeros(2, 0, dtype=torch.int32))
    out_dim = Dim(
        Tensor("out_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([0, 0], dtype=torch.int32)),
        name="out",
    )
    for layout in (dict(gap=2), dict(total_bound=4)):
        for indices in (no_idx, packed.pack(no_idx, **layout)):
            out = rf.scatter(
                packed.pack(empty, **layout), indices=indices, indices_dim=empty_time, out_dim=out_dim, use_mask=False
            )
            assert packed.is_packed(out) and out.raw_tensor.inner.raw_tensor.shape[0] == 0, (layout, out.raw_tensor)

    # a plain empty source which follows packed indices is still what the result depends on
    empty.raw_tensor.requires_grad_(True)
    out = rf.scatter(
        empty, indices=packed.pack(no_idx, total_bound=4), indices_dim=empty_time, out_dim=out_dim, use_mask=False
    )
    (grad,) = torch.autograd.grad(out.raw_tensor.inner.raw_tensor.sum(), empty.raw_tensor)
    assert grad.shape == (2, 0), grad


def test_scatter_modes_which_the_frontend_composes_from_several_scatters():
    """
    mean, logsumexp, logmeanexp and the correction for an explicit fill value are compositions:
    they count the writes per position by scattering plain ones, which has to follow packed indices,
    and they ask where a maximum is -inf, which is elementwise
    """
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")
    x, time_dim = _seqs("x", batch_dim, [3, 2], [[1.0, 2.0, 4.0], [8.0, 16.0]])
    idx, _ = _seqs("idx", batch_dim, [3, 2], [[0, 0, 2], [1, 1]])
    idx, _ = rf.replace_dim(idx, in_dim=idx.dims[1], out_dim=time_dim)
    out_dim = Dim(
        Tensor("out_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([3, 2], dtype=torch.int32)),
        name="out",
    )
    for kwargs in (
        dict(mode="mean"),
        dict(mode="logsumexp"),
        dict(mode="logmeanexp"),
        dict(mode="max", fill_value=7, use_mask=True),
        dict(mode="sum", fill_value=7, use_mask=True),
    ):
        ref = rf.scatter(x, indices=idx, indices_dim=time_dim, out_dim=out_dim, **kwargs)
        for indices in (idx, packed.pack(idx)):
            packed._warned_fallback_ops.clear()
            out = rf.scatter(packed.pack(x), indices=indices, indices_dim=time_dim, out_dim=out_dim, **kwargs)
            assert packed.is_packed(out) and not packed._warned_fallback_ops, (kwargs, packed._warned_fallback_ops)
            _assert_equal_non_padded(out, ref, batch_dim, out_dim)


def test_scatter_relayout_rows_which_write_nothing_get_a_zero_gradient():
    """
    whatever a row outside the sequences holds must not come back through the discarded row:
    the backward of a max or min divides by the number of sources equal to the result, and nan equals nothing
    """
    rf.select_backend_torch()
    batch_dim = Dim(2, name="batch")
    idx, time_dim = _seqs("idx", batch_dim, [1, 1], [[0], [1]])
    out_dim = Dim(
        Tensor("out_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([2, 2], dtype=torch.int32)),
        name="out",
    )
    packed_dim = Dim(4, name="packed")
    buf = torch.tensor([1.0, 2.0, float("nan"), float("nan")], requires_grad=True)
    x = packed.pack_import(
        Tensor("x", dims=[packed_dim], dtype="float32", raw_tensor=buf),
        batch_dim=batch_dim,
        spatial_dim=time_dim,
        packed_dim=packed_dim,
    )
    for mode in ("sum", "max", "min"):
        for use_mask in (False, True):
            out = rf.scatter(x, indices=idx, indices_dim=time_dim, out_dim=out_dim, mode=mode, use_mask=use_mask)
            got = packed.unpack(out).copy_compatible_to_dims_raw([batch_dim, out_dim])
            assert (got[0, 0], got[1, 1]) == (1.0, 2.0), (mode, use_mask, got)
            (grad,) = torch.autograd.grad(got[0, 0] + got[1, 1], buf)
            assert grad.tolist() == [1.0, 1.0, 0.0, 0.0], (mode, use_mask, grad)


def test_gather_into_the_indices_packing_under_cuda_graph_capture():
    """
    one captured graph, forward and backward, has to serve every batch:
    the layout of both packings is recomputed from the device lens inside the graph,
    so a replay over other lengths reads the right rows (and columns, for a source carrying the frame dim)
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("cuda only: real graph capture")
    rf.select_backend_torch()
    dev = "cuda"
    n_seqs, label_cap, frame_cap, feat, label_bound, frame_bound = 3, 4, 6, 2, 14, 20
    batch_dim = Dim(n_seqs, name="batch")
    label_lens, frame_lens = (
        Tensor(name, dims=[batch_dim], dtype="int32", raw_tensor=torch.zeros(n_seqs, dtype=torch.int32, device=dev))
        for name in ("label_lens", "frame_lens")
    )
    label_dim = Dim(label_lens, name="labels", capacity=label_cap)
    frame_dim = Dim(frame_lens, name="frames", capacity=frame_cap)
    label_packed, frame_packed = Dim(label_bound, name="packed_labels"), Dim(frame_bound, name="packed_frames")
    feat_dim = Dim(feat, name="feat")

    # the static buffers the graph reads, every batch is copied into them
    states_buf = torch.zeros(label_bound, feat, device=dev, requires_grad=True)
    lattice_buf = torch.zeros(label_bound, frame_cap, device=dev, requires_grad=True)
    idx_buf = torch.zeros(frame_bound, dtype=torch.int32, device=dev)
    valid_buf = torch.zeros(frame_bound, device=dev)

    def _import(name, raw, dims, spatial_dim, packed_dim, **kwargs):
        flat = Tensor(name, dims=dims, dtype=kwargs.pop("dtype", "float32"), raw_tensor=raw, **kwargs)
        return packed.pack_import(flat, batch_dim=batch_dim, spatial_dim=spatial_dim, packed_dim=packed_dim)

    states = _import("states", states_buf, [label_packed, feat_dim], label_dim, label_packed)
    lattice = _import("lattice", lattice_buf, [label_packed, frame_dim], label_dim, label_packed)
    idx = _import("idx", idx_buf, [frame_packed], frame_dim, frame_packed, dtype="int32", sparse_dim=label_dim)

    def _step():
        with rf.set_static_traceable_ctx():
            out = rf.gather(states, indices=idx, axis=label_dim, clip_to_valid=True)
            out_lattice = rf.gather(lattice, indices=idx, axis=label_dim, clip_to_valid=True)
        assert out.raw_tensor.packed_dim is frame_packed and out_lattice.raw_tensor.packed_dim is frame_packed
        out_raw = out.raw_tensor.inner.copy_transpose([frame_packed, feat_dim]).raw_tensor
        out_lattice_raw = out_lattice.raw_tensor.inner.raw_tensor
        ((out_raw * valid_buf[:, None]).sum() + (out_lattice_raw * valid_buf).sum()).backward()
        return out_raw, out_lattice_raw

    def _load(n_labels, n_frames, seed):
        """:return: the padded data of this batch, after writing it densely packed into the static buffers"""
        gen = torch.Generator().manual_seed(seed)
        states_pad = torch.randn(n_seqs, label_cap, feat, generator=gen)
        lattice_pad = torch.randn(n_seqs, label_cap, frame_cap, generator=gen)
        idx_pad = torch.randint(0, label_cap + 2, (n_seqs, frame_cap), generator=gen, dtype=torch.int32)
        with torch.no_grad():
            for buf in (states_buf, lattice_buf, idx_buf, valid_buf):
                buf.zero_()
            states_buf[: sum(n_labels)] = torch.cat([states_pad[b, :n] for b, n in enumerate(n_labels)]).to(dev)
            lattice_buf[: sum(n_labels)] = torch.cat([lattice_pad[b, :n] for b, n in enumerate(n_labels)]).to(dev)
            idx_buf[: sum(n_frames)] = torch.cat([idx_pad[b, :n] for b, n in enumerate(n_frames)]).to(dev)
            valid_buf[: sum(n_frames)] = 1.0
            label_lens.raw_tensor.copy_(torch.tensor(n_labels, dtype=torch.int32))
            frame_lens.raw_tensor.copy_(torch.tensor(n_frames, dtype=torch.int32))
        return states_pad, lattice_pad, idx_pad

    batches = [
        ((4, 2, 3), (6, 3, 5), 1),
        ((1, 4, 4), (2, 6, 6), 2),
        ((3, 0, 2), (4, 0, 1), 3),
        ((4, 4, 4), (6, 6, 6), 4),
    ]
    _load(*batches[0])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _step()
    torch.cuda.current_stream().wait_stream(side)
    states_buf.grad = lattice_buf.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static, out_lattice_static = _step()

    for n_labels, n_frames, seed in batches:
        states_pad, lattice_pad, idx_pad = _load(n_labels, n_frames, seed)
        graph.replay()
        torch.cuda.synchronize()
        out, out_lattice = out_static.detach().cpu(), out_lattice_static.detach().cpu()
        grad, grad_lattice = states_buf.grad.cpu(), lattice_buf.grad.cpu()
        ref_grad, ref_grad_lattice = torch.zeros_like(grad), torch.zeros_like(grad_lattice)
        frame_row = label_row = 0
        for b in range(n_seqs):
            for t in range(n_frames[b]):
                label = min(int(idx_pad[b, t]), n_labels[b] - 1)
                where = f"lens {n_labels} {n_frames} seq {b} frame {t}"
                numpy.testing.assert_allclose(out[frame_row + t], states_pad[b, label], rtol=1e-6, err_msg=where)
                numpy.testing.assert_allclose(
                    out_lattice[frame_row + t], lattice_pad[b, label, t], rtol=1e-6, err_msg=where
                )
                ref_grad[label_row + label] += 1.0
                ref_grad_lattice[label_row + label, t] += 1.0
            frame_row += n_frames[b]
            label_row += n_labels[b]
        numpy.testing.assert_allclose(grad, ref_grad, err_msg=f"lens {n_labels} {n_frames}")
        numpy.testing.assert_allclose(grad_lattice, ref_grad_lattice, err_msg=f"lens {n_labels} {n_frames}")


def test_scatter_relayout_under_cuda_graph_capture():
    """
    one captured graph, forward and backward, has to serve every batch:
    the result dim is a sum of the source dim and another one, as for a stream which other frames fill up,
    its layout is recomputed from the device lens inside the graph,
    and whatever must not be written (junk rows, positions outside the own result sequence) goes nowhere
    """
    if not torch.cuda.is_available():
        raise unittest.SkipTest("cuda only: real graph capture")
    rf.select_backend_torch()
    dev = "cuda"
    n_seqs, frame_cap, rest_cap, feat, frame_bound = 3, 4, 3, 2, 10
    batch_dim = Dim(n_seqs, name="batch")
    frame_lens_buf, rest_lens_buf = (torch.zeros(n_seqs, dtype=torch.int32, device=dev) for _ in range(2))
    frame_dim, rest_dim = (
        Dim(Tensor(name, dims=[batch_dim], dtype="int32", raw_tensor=buf), name=name, capacity=cap)
        for name, buf, cap in (("frames", frame_lens_buf, frame_cap), ("rest", rest_lens_buf, rest_cap))
    )
    frame_packed = Dim(frame_bound, name="packed_frames")
    feat_dim = Dim(feat, name="feat")

    # the static buffers the graph reads, every batch is copied into them
    x_buf = torch.zeros(frame_bound, feat, device=dev, requires_grad=True)
    idx_buf = torch.zeros(frame_bound, dtype=torch.int32, device=dev)
    idx_pad_buf = torch.zeros(n_seqs, frame_cap, dtype=torch.int32, device=dev)
    out_bound = frame_bound + n_seqs * rest_cap
    weight_buf = torch.randn(out_bound, feat, device=dev, generator=torch.Generator(dev).manual_seed(9))

    def _import(name, raw, dims, **kwargs):
        flat = Tensor(name, dims=dims, dtype=kwargs.pop("dtype", "float32"), raw_tensor=raw, **kwargs)
        return packed.pack_import(flat, batch_dim=batch_dim, spatial_dim=frame_dim, packed_dim=frame_packed)

    def _step():
        # as the engine before every step: a dim derived from the lens memoizes its sizes when it is built,
        # so the lens dims forget what earlier calls derived, and the sum is built (and computed) in here
        for dim, lens_buf in ((frame_dim, frame_lens_buf), (rest_dim, rest_lens_buf)):
            dim.reset_eager()
            dim.dyn_size_ext.raw_tensor = lens_buf
        out_dim = frame_dim + rest_dim
        x = _import("x", x_buf, [frame_packed, feat_dim])
        idx_packed = _import("idx", idx_buf, [frame_packed], dtype="int32")
        idx_plain = Tensor("idx_plain", dims=[batch_dim, frame_dim], dtype="int32", raw_tensor=idx_pad_buf)
        outs = []
        with rf.set_static_traceable_ctx():
            for indices, use_mask in ((idx_packed, False), (idx_plain, True)):
                out = rf.scatter(x, indices=indices, indices_dim=frame_dim, out_dim=out_dim, use_mask=use_mask)
                assert out.raw_tensor.packed_dim.dimension == out_bound, out.raw_tensor
                outs.append(out.raw_tensor.inner.copy_transpose([out.raw_tensor.packed_dim, feat_dim]).raw_tensor)
        (outs[0] * weight_buf + 2.0 * outs[1] * weight_buf).sum().backward()
        return outs

    def _load(n_frames, n_rest, seed):
        """:return: the padded data of this batch, after writing it densely packed into the static buffers"""
        gen = torch.Generator().manual_seed(seed)
        x_pad = torch.randn(n_seqs, frame_cap, feat, generator=gen)
        # positions up to two beyond the longest possible result, so some frames must not write at all
        idx_pad = torch.randint(0, frame_cap + rest_cap + 2, (n_seqs, frame_cap), generator=gen, dtype=torch.int32)
        with torch.no_grad():
            for buf in (x_buf, idx_buf):
                buf.zero_()
            x_buf[: sum(n_frames)] = torch.cat([x_pad[b, :n] for b, n in enumerate(n_frames)]).to(dev)
            idx_buf[: sum(n_frames)] = torch.cat([idx_pad[b, :n] for b, n in enumerate(n_frames)]).to(dev)
            idx_pad_buf.copy_(idx_pad)
            frame_lens_buf.copy_(torch.tensor(n_frames, dtype=torch.int32))
            rest_lens_buf.copy_(torch.tensor(n_rest, dtype=torch.int32))
        return x_pad, idx_pad

    batches = [
        ((4, 2, 3), (2, 0, 3), 1),
        ((1, 4, 4), (3, 3, 1), 2),
        ((3, 0, 2), (1, 0, 0), 3),
        ((4, 4, 2), (3, 3, 3), 4),
    ]
    _load(*batches[0])
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _step()
    torch.cuda.current_stream().wait_stream(side)
    x_buf.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs_static = _step()

    for n_frames, n_rest, seed in batches:
        x_pad, idx_pad = _load(n_frames, n_rest, seed)
        graph.replay()
        torch.cuda.synchronize()
        weight = weight_buf.cpu()
        ref = torch.zeros(out_bound, feat)
        ref_grad = torch.zeros(frame_bound, feat)
        frame_row = out_row = 0
        for b in range(n_seqs):
            out_len = n_frames[b] + n_rest[b]
            for t in range(n_frames[b]):
                pos = int(idx_pad[b, t])
                if pos < out_len:
                    ref[out_row + pos] += x_pad[b, t]
                    ref_grad[frame_row + t] = 3.0 * weight[out_row + pos]
            frame_row += n_frames[b]
            out_row += out_len
        for out_static in outs_static:
            numpy.testing.assert_allclose(
                out_static.detach().cpu(), ref, rtol=1e-6, atol=1e-6, err_msg=f"lens {n_frames} {n_rest}"
            )
        numpy.testing.assert_allclose(
            x_buf.grad.cpu(), ref_grad, rtol=1e-6, atol=1e-6, err_msg=f"lens {n_frames} {n_rest}"
        )


def _scores_and_labels_over_two_time_dims() -> Tuple[Tensor, Tensor, Dim, Dim, Dim]:
    scores, batch_dim, time_dim, vocab_dim = _make_input(batch_size=2, seq_lens=(4, 3), feat=5)
    label_dim = Dim(
        Tensor("labels", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([3, 2], dtype=torch.int32)),
        name="labels",
    )
    labels = Tensor("labels", dims=[batch_dim, label_dim], dtype="int32", sparse_dim=vocab_dim)
    labels.raw_tensor = torch.tensor([[1, 2, 3], [4, 0, 0]], dtype=torch.int32)
    return scores, labels, batch_dim, time_dim, label_dim


def test_gather_along_a_plain_axis_refuses_indices_of_another_packing():
    """
    per-frame indices along a plain axis are read row by row, which only holds when they share the packing.
    Indices packed over another time dim ask for the (time, labels) lattice, which no packing describes,
    so this has to take the gated fallback instead of crossing the rows of all sequences.
    """
    rf.select_backend_torch()
    scores, labels, batch_dim, time_dim, label_dim = _scores_and_labels_over_two_time_dims()
    vocab_dim = labels.sparse_dim
    ref = rf.gather(scores, indices=labels, axis=vocab_dim)
    packed.set_allowed_fallbacks(False)
    try:
        with pytest.raises(Exception, match="op 'gather'"):
            rf.gather(packed.pack(scores), indices=packed.pack(labels), axis=vocab_dim)
        packed.set_allowed_fallbacks(["gather"])
        out = rf.gather(packed.pack(scores), indices=packed.pack(labels), axis=vocab_dim)
    finally:
        packed.set_allowed_fallbacks(None)
    out = packed.unpack(out) if packed.is_packed(out) else out
    assert out.dims_set == ref.dims_set, (out.dims, ref.dims)
    out_raw = out.copy_compatible_to_dims(ref.dims).raw_tensor
    for b, (n_time, n_labels) in enumerate(((4, 3), (3, 2))):
        numpy.testing.assert_allclose(
            out_raw[b, :n_time, :n_labels].numpy(), ref.raw_tensor[b, :n_time, :n_labels].numpy(), rtol=1e-6
        )


def test_scaled_gradient_refuses_a_scale_of_another_packing():
    """a per-frame gradient scale has to share the packing, else its rows belong to other frames"""
    rf.select_backend_torch()
    scores, labels, batch_dim, time_dim, label_dim = _scores_and_labels_over_two_time_dims()
    x = rf.reduce_sum(scores, axis=labels.sparse_dim)
    scale = rf.cast(labels, "float32")
    with pytest.raises(AssertionError, match="scaled_gradient"):
        rf.scaled_gradient(packed.pack(x), packed.pack(scale))
    same = Tensor("scale", dims=[batch_dim, time_dim], dtype="float32", raw_tensor=torch.full((2, 4), 0.5))
    out = rf.scaled_gradient(packed.pack(x), packed.pack(same, gap=2))
    assert packed.is_packed(out) and out.dims == x.dims


def test_softmax_over_a_single_packed_axis_with_a_bound():
    """a bound-sized packing of one axis normalizes over its content rows only"""
    rf.select_backend_torch()
    size_dim = Dim(Tensor("size", dims=[], dtype="int32", raw_tensor=torch.tensor(3, dtype=torch.int32)), name="size")
    x = Tensor("x", dims=[size_dim], dtype="float32", raw_tensor=torch.tensor([0.0, 1.0, 2.0]))
    xp = packed.pack(x, dims=[size_dim], total_bound=5)
    for fn in (rf.softmax, rf.log_softmax):
        out = fn(xp, axis=size_dim)
        assert packed.is_packed(out), fn.__name__
        numpy.testing.assert_allclose(
            out.raw_tensor.inner.raw_tensor.numpy()[:3],
            fn(x, axis=size_dim).raw_tensor.numpy(),
            rtol=1e-6,
            err_msg=fn.__name__,
        )


def test_batch_norm_packed_dense_bound_with_a_static_axis():
    """the masked batch_norm statistics also cover a static axis next to the packed one, no re-layout loop"""
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(seq_lens=(3, 2), feat=2, seed=9)
    k_dim = Dim(2, name="k")
    xk = Tensor("xk", dims=[batch_dim, time_dim, k_dim, feat_dim], dtype="float32")
    xk.raw_tensor = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    with rf.set_default_device_ctx("cpu"):
        rf.set_random_seed(3)
        bn_dense = rf.BatchNorm(feat_dim, use_mask=False)
        bn_bound = rf.BatchNorm(feat_dim, use_mask=False)
        with rf.get_run_ctx().train_flag_ctx(True):
            out_dense = bn_dense(packed.pack(xk))
            out_bound = bn_bound(packed.regap(packed.pack(xk), 0, total_bound=8))
        assert packed.is_packed(out_bound)
    _assert_equal_non_padded(out_bound, packed.unpack(out_dense), batch_dim, time_dim)
    for p_dense, p_bound in [
        (bn_dense.running_mean, bn_bound.running_mean),
        (bn_dense.running_variance, bn_bound.running_variance),
    ]:
        numpy.testing.assert_allclose(
            p_dense.raw_tensor.detach().numpy(), p_bound.raw_tensor.detach().numpy(), rtol=1e-5, atol=1e-6
        )


def test_regap_of_entirely_empty_sequences():
    """a packing whose sequences are all empty can still be re-laid out"""
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(seq_lens=(0, 0), feat=1)
    x.raw_tensor = torch.empty(2, 0, 1)
    out = packed.regap(packed.pack(x), 2)
    assert packed.is_packed(out) and out.raw_tensor.gap == 2
    assert tuple(packed.unpack(out).copy_transpose([batch_dim, time_dim, feat_dim]).raw_tensor.shape) == (2, 0, 1)


def test_pack_dense_total_bound_static_buffer():
    """a dense pack with total_bound allocates the bound-sized static buffer, content first"""
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(seq_lens=(4, 2))
    xp = packed.pack(x, total_bound=10)
    raw = xp.raw_tensor
    assert raw.packed_dim.dimension == 10 and raw.inner.raw_tensor.shape[0] == 10, raw
    assert raw.content_bound == 10, raw
    _assert_equal_non_padded(xp, x, batch_dim, time_dim)
    content = torch.cat([x.raw_tensor[0, :4], x.raw_tensor[1, :2]])
    numpy.testing.assert_allclose(raw.inner.raw_tensor[:6].detach().numpy(), content.numpy())
    out = rf.reduce_mean(xp, axis=time_dim).copy_compatible_to_dims([batch_dim, feat_dim])
    ref = rf.reduce_mean(x, axis=time_dim).copy_compatible_to_dims([batch_dim, feat_dim])
    numpy.testing.assert_allclose(out.raw_tensor.detach().numpy(), ref.raw_tensor.detach().numpy(), rtol=1e-5)


def test_shift_and_pad_with_a_per_seq_pad_value():
    """a pad value over the batch dim applies per sequence in the packed shift and pad"""
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(batch_size=3, seq_lens=(7, 5, 4))
    xp = packed.pack(x, gap=2)
    pad = Tensor("pad", dims=[batch_dim], dtype="float32", raw_tensor=torch.tensor([100.0, 200.0, 300.0]))
    for shift, amount in ((rf.shift_right, 2), (rf.shift_left, 1)):
        ref = shift(x, axis=time_dim, pad_value=pad, amount=amount)
        out_p = shift(xp, axis=time_dim, pad_value=pad, amount=amount)
        assert packed.is_packed(out_p), shift.__name__
        _assert_equal_non_padded(out_p, ref, batch_dim, time_dim)
    ref, (padded_time,) = rf.pad(x, axes=[time_dim], padding=[(1, 0)], value=pad)
    out_p, _ = rf.pad(xp, axes=[time_dim], padding=[(1, 0)], out_dims=[padded_time], value=pad)
    assert packed.is_packed(out_p)
    _assert_equal_non_padded(out_p, ref, batch_dim, padded_time)


def test_regap_restoring_a_layout_lens_layout_needs_a_bound():
    """restoring an exact layout under static tracing takes its bound from the caller, not derived"""
    rf.select_backend_torch()
    x, batch_dim, time_dim, feat_dim = _make_input(seq_lens=(6, 4))
    time_dim.capacity = 6
    layout_lens = Tensor(
        "layout_lens", dims=[batch_dim], dtype="int32", raw_tensor=torch.tensor([6, 4], dtype=torch.int32)
    )

    src = packed.pack(x, dims=[batch_dim, time_dim], gap=2, align=2, total_bound=32)
    orig_total = src.raw_tensor.packed_dim.dimension
    assert orig_total == 32, src.raw_tensor

    # the kernel paths strip the gaps, run, then restore. regap derives a bound itself only when
    # layout_lens is None, so the restoring call has to pass the layout's own size.
    with rf.set_static_traceable_ctx(True):
        dense = packed.regap(src, 0, align=1)
        assert dense.raw_tensor.packed_dim.dimension is not None, dense.raw_tensor

        try:
            out = packed.regap(dense, 2, align=2, layout_lens=layout_lens)
            out.raw_tensor.packed_dim.get_dim_value_tensor()
        except Exception as exc:
            assert "no (derivable) capacity" in str(exc), exc
        else:
            raise Exception("regap without a bound should have no capacity for the restored dim")

        out = packed.regap(dense, 2, align=2, layout_lens=layout_lens, total_bound=orig_total)
        assert out.raw_tensor.packed_dim.dimension == orig_total, out.raw_tensor
        out.raw_tensor.packed_dim.get_dim_value_tensor()

    _assert_equal_non_padded(out, x, batch_dim, time_dim)


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
