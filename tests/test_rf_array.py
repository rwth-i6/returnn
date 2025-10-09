"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import numpy as np
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_pack_padded():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Dim]:
            pack, pack_dim = rf.pack_padded(x, dims=[batch_dim, time_dim], enforce_sorted=False)
            return pack, pack_dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, pack_dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(pack_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_pack_padded_md():
    batch_dim = Dim(3, name="batch")
    hyps_dim = Dim(5, name="hyps")
    hyps_spatial_dim = Dim(Tensor("hyps_spatial", [batch_dim, hyps_dim], dtype="int32"))
    vocab_dim = Dim(1000, name="vocab")
    extern_data = TensorDict(
        {"labels": Tensor("labels", [batch_dim, hyps_dim, hyps_spatial_dim], dtype="int32", sparse_dim=vocab_dim)}
    )
    hyps_packed_spatial_dim = Dim(None, name="hyps_packed_spatial")

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        labels = extern_data["labels"]
        assert hyps_spatial_dim in labels.dims
        assert hyps_spatial_dim.dyn_size_ext.dims == (batch_dim, hyps_dim)
        labels.mark_as_output("labels", shape=(batch_dim, hyps_dim, hyps_spatial_dim))
        hyps_spatial_dim.dyn_size_ext.mark_as_output("hyps_spatial_size", shape=[batch_dim, hyps_dim])
        labels_, _ = rf.pack_padded(labels, dims=[hyps_dim, hyps_spatial_dim], out_dim=hyps_packed_spatial_dim)
        assert hyps_packed_spatial_dim.dyn_size_ext.dims == (batch_dim,)
        hyps_packed_spatial_dim.dyn_size_ext.mark_as_output("packed_size", shape=[batch_dim])
        labels_.mark_as_default_output(shape=[batch_dim, hyps_packed_spatial_dim])

    out = run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)
    print("out:")
    print(out["output"].raw_tensor.shape)
    print(out["output"].raw_tensor)
    batch_size = out["labels"].raw_tensor.shape[0]
    for b in range(batch_size):
        num_hyps = hyps_dim.dimension
        hyps_spatial_sizes = out["hyps_spatial_size"].raw_tensor[b]  # [hyps]
        hyps_packed_spatial_size = out["packed_size"].raw_tensor[b]  # []
        print(f"batch {b}, hyps_spatial_size {hyps_spatial_sizes}, hyps_packed_spatial_size {hyps_packed_spatial_size}")
        assert hyps_packed_spatial_size == sum(hyps_spatial_sizes)
        labels = [out["labels"].raw_tensor[b, h, : hyps_spatial_sizes[h]] for h in range(num_hyps)]
        labels_ = [seq.tolist() for seq in labels]
        print("labels:", labels_)
        packed = out["output"].raw_tensor[b, :hyps_packed_spatial_size]  # [hyps_packed_spatial]
        packed_ = packed.tolist()
        print("packed:", packed_)
        assert sum(labels_, []) == packed_


def test_masked_select():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "mask": Tensor("mask", [batch_dim, time_dim], dtype="bool"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        x = extern_data["data"]
        mask = extern_data["mask"]
        out, pack_dim = rf.masked_select(x, mask=mask, dims=[batch_dim, time_dim])
        out.mark_as_default_output(shape=(pack_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_masked_select_single_dim():
    # https://github.com/rwth-i6/returnn/issues/1605
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "mask": Tensor("mask", [batch_dim, time_dim], dtype="bool"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        x = extern_data["data"]
        mask = extern_data["mask"]
        out, pack_dim = rf.masked_select(x, mask=mask, dims=[time_dim])
        out.mark_as_default_output(shape=(batch_dim, pack_dim, in_dim))
        pack_dim.dyn_size_ext.mark_as_output("out_size", shape=[batch_dim])
        x.mark_as_output("input", shape=[batch_dim, time_dim, in_dim])
        mask.mark_as_output("mask", shape=[batch_dim, time_dim])
        time_dim.dyn_size_ext.mark_as_output("in_size", shape=[batch_dim])

    res = run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step, test_tensorflow=False)
    for k, v in res.data.items():
        print(f"{k} {v} = {v.raw_tensor}")
    num_batches = res["input"].raw_tensor.shape[0]
    have_non_zero = False
    have_less = False
    for b in range(num_batches):
        in_size = res["in_size"].raw_tensor[b]
        out_size = res["out_size"].raw_tensor[b]
        print(f"batch {b}, in_size {in_size}, out_size {out_size}")
        assert 0 <= out_size <= in_size
        if out_size > 0:
            have_non_zero = True
        if out_size < in_size:
            have_less = True
        c = 0
        for t in range(in_size):
            if res["mask"].raw_tensor[b, t]:
                c += 1
        assert c == out_size
        t_ = 0
        for t in range(in_size):
            if res["mask"].raw_tensor[b, t]:
                assert (res["input"].raw_tensor[b, t] == res["output"].raw_tensor[b, t_]).all()
                t_ += 1
    assert have_non_zero and have_less  # just that the test case covered all cases


def test_pad_packed():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        x = extern_data["data"]
        x.mark_as_output("in", shape=(batch_dim, time_dim, in_dim))
        x_, batch_time_flat_dim = rf.pack_padded(x, dims=[batch_dim, time_dim])
        assert x_.dims_set == {batch_time_flat_dim, in_dim}
        # test [batch_fime_flat,feat], batch_fime_flat -> [batch, time]
        x__ = rf.pad_packed(x_, in_dim=batch_time_flat_dim, dims=[batch_dim, time_dim])
        assert x__.dims_set == {batch_dim, time_dim, in_dim}
        x__.mark_as_output("out", shape=(batch_dim, time_dim, in_dim))

    out_dict = run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)
    in_ = out_dict["in"]
    out = out_dict["out"]
    np.testing.assert_array_equal(in_.raw_tensor, out.raw_tensor)


def test_pad_packed_batched():
    batch_dim_ = Dim(3, name="batch")
    in_dim = Dim(2, name="in")

    # noinspection PyShadowingNames
    def _forward_step(**_kwargs):
        time1_dim = Dim(rf.convert_to_tensor(np.array([1, 3, 4]), name="time1", dims=[batch_dim_], dtype="int32"))
        time2_dim = Dim(rf.convert_to_tensor(np.array([3, 2, 2]), name="time2", dims=[batch_dim_], dtype="int32"))
        x = rf.random_uniform((batch_dim_, time1_dim, time2_dim, in_dim))
        x.mark_as_output("in", shape=(batch_dim_, time1_dim, time2_dim, in_dim))
        x_, time_flat_dim = rf.pack_padded(x, dims=[time1_dim, time2_dim])
        assert x_.dims_set == {batch_dim_, time_flat_dim, in_dim}
        x_.mark_as_output("flat", shape=(batch_dim_, time_flat_dim, in_dim))
        # test [batch,mult_fime_flat,feat], mult_fime_flat -> [time1, time2]
        x__ = rf.pad_packed(x_, in_dim=time_flat_dim, dims=[time1_dim, time2_dim])
        assert x__.dims_set == {batch_dim_, time1_dim, time2_dim, in_dim}
        x__.mark_as_output("out", shape=(batch_dim_, time1_dim, time2_dim, in_dim))

    out_dict = run_model(TensorDict(), lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)
    in_ = out_dict["in"]
    flat = out_dict["flat"]
    out = out_dict["out"]
    print("in:", in_, in_.raw_tensor.shape)
    print("in time1:", in_.dims[1].dyn_size)
    print("in time2:", in_.dims[2].dyn_size)
    print("flat:", flat, flat.raw_tensor.shape)
    print("out:", out, out.raw_tensor.shape)
    print("in raw:")
    print(in_.raw_tensor)
    print("flat raw:")
    print(flat.raw_tensor)
    print("out raw:")
    print(out.raw_tensor)
    np.testing.assert_array_equal(in_.raw_tensor, out.raw_tensor)


def test_masked_select_masked_scatter_vs_where_rev_dims():
    """
    Compare rf.where vs rf.masked_select+rf.masked_scatter.
    Some op (e.g. LM update) could be done more efficiently on just the packed data (rf.masked_select),
    that is why rf.masked_select+rf.masked_scatter can be useful over just using rf.where.
    (In general, when computing the new ``a`` is expensive.)
    The test does not cover this part on the computation (we just feed in some random ``a``),
    but it checks that the results are the same,
    as we had some problems with that in the past.
    """
    # noinspection PyShadowingNames
    batch_dim = Dim(2, name="batch")
    beam_dim = Dim(3, name="beam")

    extern_data = TensorDict(
        {
            "mask": Tensor("mask", [beam_dim, batch_dim], dtype="bool"),
            # Note: The dim order is relevant for this test. It passes when it is [batch_dim, beam_dim]...
            "a": Tensor("a", [beam_dim, batch_dim], dtype="int32"),
            "b": Tensor("b", [beam_dim, batch_dim], dtype="int32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        mask = extern_data["mask"]
        a = extern_data["a"]
        b = extern_data["b"]
        a.mark_as_output("a", shape=[beam_dim, batch_dim])
        b.mark_as_output("b", shape=[beam_dim, batch_dim])
        mask.mark_as_output("mask", shape=[beam_dim, batch_dim])

        # Code via rf.where.
        res_where = rf.where(mask, a, b)
        assert res_where.dims_set == {beam_dim, batch_dim}
        res_where.mark_as_output("res_where", shape=[beam_dim, batch_dim])

        # Code via rf.masked_select and rf.masked_scatter.
        a_packed, packed_dim = rf.masked_select(a, mask=mask, dims=[batch_dim, beam_dim])
        assert a_packed.dims_set == {packed_dim}
        res_packed = rf.masked_scatter(a_packed, b, mask=mask, dims=[batch_dim, beam_dim], in_dim=packed_dim)
        assert res_packed.dims_set == {batch_dim, beam_dim}
        res_packed.mark_as_output("res_packed", shape=[beam_dim, batch_dim])

    out_dict = run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)
    res_where = out_dict["res_where"]
    res_packed = out_dict["res_packed"]
    assert res_where.raw_tensor.shape == res_packed.raw_tensor.shape
    print("a:")
    print(out_dict["a"].raw_tensor)
    print("b:")
    print(out_dict["b"].raw_tensor)
    print("mask:")
    print(out_dict["mask"].raw_tensor)
    print("result with where:")
    print(res_where.raw_tensor)
    print("result with packing:")
    print(res_packed.raw_tensor)
    np.testing.assert_equal(res_where.raw_tensor, res_packed.raw_tensor)


def test_masked_select_masked_scatter_vs_where_md_rev_dims():
    """
    Like :func:`test_masked_select_masked_scatter_vs_where_rev_dims`
    but we add another spatial dim, which then needs some further handling.
    """
    # noinspection PyShadowingNames
    batch_dim = Dim(2, name="batch")
    beam_dim = Dim(3, name="beam")
    hist_a_dim = Dim(Tensor("hist_a", [batch_dim, beam_dim], dtype="int32"))
    hist_b_dim = Dim(Tensor("hist_b", [batch_dim, beam_dim], dtype="int32"))

    extern_data = TensorDict(
        {
            "mask": Tensor("mask", [beam_dim, batch_dim], dtype="bool"),
            "a": Tensor("a", [beam_dim, batch_dim, hist_a_dim], dtype="int32"),
            "b": Tensor("b", [beam_dim, batch_dim, hist_b_dim], dtype="int32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        mask = extern_data["mask"]
        a = extern_data["a"]
        b = extern_data["b"]
        a.mark_as_output("a", shape=[beam_dim, batch_dim, hist_a_dim])
        b.mark_as_output("b", shape=[beam_dim, batch_dim, hist_b_dim])
        mask.mark_as_output("mask", shape=[beam_dim, batch_dim])
        hist_a_size = hist_a_dim.get_size_tensor()
        hist_b_size = hist_b_dim.get_size_tensor()

        # Code via rf.where.
        hist_comb_sizes = rf.where(mask, hist_a_size, hist_b_size)
        hist_comb_dim = Dim(hist_comb_sizes, name="hist_comb")
        a_ = rf.replace_dim_v2(a, in_dim=hist_a_dim, out_dim=hist_comb_dim)
        b_ = rf.replace_dim_v2(b, in_dim=hist_b_dim, out_dim=hist_comb_dim)
        res_where = rf.where(mask, a_, b_)
        assert res_where.dims_set == {beam_dim, batch_dim, hist_comb_dim}
        res_where.mark_as_output("res_where", shape=[beam_dim, batch_dim, hist_comb_dim])

        # Code via rf.masked_select and rf.masked_scatter.
        hist_a_packed_size, packed_dim = rf.masked_select(hist_a_size, mask=mask, dims=[batch_dim, beam_dim])
        hist_a_packed_dim = Dim(hist_a_packed_size, name="hist_a_packed")
        a_packed, _ = rf.masked_select(a, mask=mask, dims=[batch_dim, beam_dim], out_dim=packed_dim)
        a_packed = rf.replace_dim_v2(a_packed, in_dim=hist_a_dim, out_dim=hist_a_packed_dim)
        assert a_packed.dims_set == {packed_dim, hist_a_packed_dim}
        hist_merged_size = rf.masked_scatter(
            hist_a_packed_size, hist_b_size, mask=mask, dims=[batch_dim, beam_dim], in_dim=packed_dim
        )
        hist_merged_dim = Dim(hist_merged_size, name="hist_merged")
        a_packed = rf.replace_dim_v2(a_packed, in_dim=hist_a_packed_dim, out_dim=hist_merged_dim)
        b_packed = rf.replace_dim_v2(b, in_dim=hist_b_dim, out_dim=hist_merged_dim)
        res_packed = rf.masked_scatter(a_packed, b_packed, mask=mask, dims=[batch_dim, beam_dim], in_dim=packed_dim)
        assert res_packed.dims_set == {batch_dim, beam_dim, hist_merged_dim}
        res_packed.mark_as_output("res_packed", shape=[beam_dim, batch_dim, hist_merged_dim])

    out_dict = run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)
    res_where = out_dict["res_where"]
    res_packed = out_dict["res_packed"]
    hist_where_dim = res_where.dims[-1]
    hist_packed_dim = res_packed.dims[-1]
    assert hist_where_dim.dyn_size_ext.dims_set == hist_packed_dim.dyn_size_ext.dims_set == {batch_dim, beam_dim}
    hist_where_size_raw = hist_where_dim.dyn_size_ext.copy_compatible_to_dims_raw((batch_dim, beam_dim))
    hist_packed_size_raw = hist_packed_dim.dyn_size_ext.copy_compatible_to_dims_raw((batch_dim, beam_dim))
    assert (hist_where_size_raw == hist_packed_size_raw).all()
    assert res_where.raw_tensor.shape == res_packed.raw_tensor.shape
    print("a:")
    print(out_dict["a"].raw_tensor)
    print("b:")
    print(out_dict["b"].raw_tensor)
    print("mask:")
    print(out_dict["mask"].raw_tensor)
    print("result with where:")
    print(res_where.raw_tensor)
    print("result with packing:")
    print(res_packed.raw_tensor)
    np.testing.assert_equal(res_where.raw_tensor, res_packed.raw_tensor)


def test_reshape():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.reshape(x, in_dims=(time_dim, in_dim), out_dims=(in_dim, time_dim))

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    # Note: The tested op here is a bit meaningless. It also is not consinstent for different batch sizes...
    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step, test_single_batch_entry=False)


def test_expand_dim():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    expand_dim = Dim(3, name="expand")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.expand_dim(x, expand_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, expand_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_concat():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Dim]:
            return rf.concat((x, in_dim), (x, in_dim))

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, dim = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_concat_partly_dyn_dim():
    time_static_dim = Dim(5, name="time_static")
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "left": Tensor("left", [batch_dim, time_static_dim, in_dim], dtype="float32"),
            "right": Tensor("right", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        left, right = extern_data["left"], extern_data["right"]
        out, out_time_dim = rf.concat((left, time_static_dim), (right, time_dim))
        out.mark_as_default_output(shape=(batch_dim, out_time_dim, in_dim))

    run_model(extern_data, lambda **_: rf.Module(), _forward_step)


def test_concat_dyn_time():
    time1_dim = Dim(Tensor("time1", [batch_dim], dtype="int32"))
    time2_dim = Dim(Tensor("time2", [batch_dim], dtype="int32"))
    extern_data = TensorDict(
        {
            "left": Tensor("left", [batch_dim, time1_dim], dtype="float32"),
            "right": Tensor("right", [batch_dim, time2_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        left, right = extern_data["left"], extern_data["right"]
        out, out_time_dim = rf.concat((left, time1_dim), (right, time2_dim))
        out.mark_as_default_output(shape=(batch_dim, out_time_dim))

    # test_single_batch_entry should test the interesting case.
    run_model(extern_data, lambda **_: rf.Module(), _forward_step, test_tensorflow=False)


def test_pad():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Tuple[Dim, Dim]]:
            pack, (new_time, new_feat) = rf.pad(x, axes=[time_dim, in_dim], padding=[(1, 2), (3, 4)], value=0)
            return pack, (new_time, new_feat)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, (new_time, new_feat) = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, new_time, new_feat))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_pad_time():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Tuple[Dim]]:
            pack, (new_time,) = rf.pad(x, axes=[time_dim], padding=[(1, 0)], value=0)
            return pack, (new_time,)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, (new_time,) = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, new_time, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_pad_time_right():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Tuple[Dim]]:
            pack, (new_time,) = rf.pad(x, axes=[time_dim], padding=[(0, 1)], value=1)
            return pack, (new_time,)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        data = extern_data["data"]
        data.mark_as_output("data", shape=(batch_dim, time_dim, in_dim))
        out, (new_time,) = model(data)
        out.mark_as_default_output(shape=(batch_dim, new_time, in_dim))

    res = run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
    data_: Tensor = res["data"]
    out_: Tensor = res["output"]
    assert data_.dims == (batch_dim, time_dim, in_dim)
    new_time_dim = out_.dims[1]
    assert out_.dims == (batch_dim, new_time_dim, in_dim) and new_time_dim != time_dim
    # assert new_time_dim == time_dim + 1  # math dim... not really necessary check here...
    assert time_dim.dyn_size_ext.dims == new_time_dim.dyn_size_ext.dims == (batch_dim,)
    batch_size = batch_dim.get_dim_value()
    assert batch_size > 1
    assert len(set(time_dim.dyn_size_ext.raw_tensor)) > 1  # not all the same
    for b in range(batch_size):
        seq_len = time_dim.dyn_size_ext.raw_tensor[b]
        new_seq_len = new_time_dim.dyn_size_ext.raw_tensor[b]
        print(f"batch {b}, seq_len {seq_len}, new_seq_len {new_seq_len}")
        assert new_seq_len == seq_len + 1
        np.testing.assert_allclose(data_.raw_tensor[b, :seq_len], out_.raw_tensor[b, :seq_len])
        print(out_.raw_tensor[b])
        assert all(out_.raw_tensor[b, seq_len] == 1.0)


def test_pad_time_right_non_scalar():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "value": Tensor("value", [batch_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        data, value = extern_data["data"], extern_data["value"]
        data.mark_as_output("data", shape=(batch_dim, time_dim, in_dim))
        value.mark_as_output("value", shape=(batch_dim,))
        out, (new_time,) = rf.pad(data, axes=[time_dim], padding=[(0, 1)], value=value)
        out.mark_as_default_output(shape=(batch_dim, new_time, in_dim))

    # TF-layers currently does not support this.
    res = run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)
    data_: Tensor = res["data"]
    value_: Tensor = res["value"]
    out_: Tensor = res["output"]
    assert data_.dims == (batch_dim, time_dim, in_dim)
    new_time_dim = out_.dims[1]
    assert out_.dims == (batch_dim, new_time_dim, in_dim) and new_time_dim != time_dim
    assert time_dim.dyn_size_ext.dims == new_time_dim.dyn_size_ext.dims == (batch_dim,)
    batch_size = batch_dim.get_dim_value()
    assert batch_size > 1
    assert len(set(time_dim.dyn_size_ext.raw_tensor)) > 1  # not all the same
    for b in range(batch_size):
        seq_len = time_dim.dyn_size_ext.raw_tensor[b]
        new_seq_len = new_time_dim.dyn_size_ext.raw_tensor[b]
        print(f"batch {b}, seq_len {seq_len}, new_seq_len {new_seq_len}")
        assert new_seq_len == seq_len + 1
        np.testing.assert_allclose(data_.raw_tensor[b, :seq_len], out_.raw_tensor[b, :seq_len])
        print(out_.raw_tensor[b])
        assert all(out_.raw_tensor[b, seq_len] == value_.raw_tensor[b])


def test_stack():
    batch_dim_ = Dim(3, name="batch")
    time_dim = Dim(5, name="time")

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        seq = rf.range_over_dim(time_dim)  # [T]
        out, _ = rf.stack([seq, seq, seq], out_dim=batch_dim_)  # [B,T]
        out.mark_as_default_output(shape=(batch_dim_, time_dim))

    out = run_model(TensorDict(), lambda *, epoch, step: rf.Module(), _forward_step, test_tensorflow=False)
    out = out["output"]
    assert out.dims == (batch_dim_, time_dim)
    assert out.raw_tensor.tolist() == [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]


def test_gather():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.gather(x, indices=0, axis=time_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_gather_2d_indices():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
            "classes": Tensor("classes", [batch_dim, time_dim], dtype="int32", sparse_dim=in_dim),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor, y: Tensor) -> Tensor:
            return rf.gather(x, indices=y, axis=in_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"], extern_data["classes"])
        out.mark_as_default_output(shape=(batch_dim, time_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_gather_feature_dim():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], feature_dim=in_dim, dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.gather(x, indices=0, axis=time_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        assert out.feature_dim == in_dim
        out.mark_as_default_output(shape=(batch_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_gather_time_static_clip_to_valid():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data_template = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], feature_dim=in_dim, dtype="float32"),
        }
    )

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        x = extern_data["data"]
        out = rf.gather(x, indices=0, axis=time_dim, clip_to_valid=True)
        out.mark_as_default_output(shape=(batch_dim, in_dim))

    run_model(extern_data_template, lambda *, epoch, step: rf.Module(), _forward_step)


def test_gather_3d_embed():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    num_embeddings_dim = Dim(2, name="num_embeddings")
    embed_dim = Dim(11, name="embed")
    extern_data_template = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], sparse_dim=in_dim, dtype="int32"),
            "embed": Tensor("embed", [in_dim, num_embeddings_dim, embed_dim], dtype="float32"),
        }
    )

    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        x, embed = extern_data["data"], extern_data["embed"]
        out = rf.gather(embed, indices=x)
        out.mark_as_default_output(shape=(batch_dim, time_dim, num_embeddings_dim, embed_dim))
        rf.reduce_sum(out, axis=out.dims).mark_as_output("loss")

    run_model(extern_data_template, lambda *, epoch, step: rf.Module(), _forward_step)


def test_scatter_fill_inf():
    batch_dim_ = Dim(3, name="batch")
    states_dim = Dim(7, name="states")

    def _forward_step(**_kwargs):
        start_states = rf.convert_to_tensor(
            [2, 4, 5], name="start_states", dims=[batch_dim_], sparse_dim=states_dim, dtype="int32"
        )
        batch_dim_.get_size_tensor().mark_as_output("batch_size", shape=[])
        start_states.mark_as_output("start_states", shape=[batch_dim_])
        scores = rf.scatter(
            rf.zeros([batch_dim_]),
            indices=start_states,
            indices_dim=[batch_dim_],
            fill_value=float("-inf"),
        )  # [S], per state
        scores.mark_as_default_output(shape=[states_dim])

    res = run_model(
        TensorDict(),
        lambda *, epoch, step: rf.Module(),
        _forward_step,
        test_tensorflow=False,
        allow_inf_nan_in_output=True,
    )
    batch_size = res["batch_size"].raw_tensor.item()
    assert res["start_states"].raw_tensor.shape == (batch_size,)
    assert res["output"].raw_tensor.shape == (states_dim.dimension,)
    assert res["output"].raw_tensor.tolist().count(0.0) == batch_size
    assert res["output"].raw_tensor.tolist().count(float("-inf")) == states_dim.dimension - batch_size
    assert states_dim.dimension > batch_size
    for i in range(states_dim.dimension):
        if i in res["start_states"].raw_tensor:
            assert res["output"].raw_tensor[i] == 0.0
        else:
            assert res["output"].raw_tensor[i] == float("-inf")


def test_slice():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Dim]:
            pack, new_time = rf.slice(x, axis=time_dim, start=1, size=2)
            return pack, new_time

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out, new_time = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, new_time, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_slice_dyn_size():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        x = extern_data["data"]
        size = time_dim.get_size_tensor()
        size = rf.reduce_min(size, axis=size.dims)  # do whatever, but get some scalar
        pack, new_time = rf.slice(x, axis=time_dim, size=size)
        pack.mark_as_default_output(shape=(batch_dim, new_time, in_dim))

    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step)


def test_shift_right():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim], sparse_dim=in_dim, dtype="int32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.shift_right(x, axis=time_dim, pad_value=0)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_reverse_sequence():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.reverse_sequence(x, axis=time_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_reverse_sequence_no_dyn():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, extern_data: TensorDict, **_kwargs):
        out = rf.reverse_sequence(extern_data["data"], axis=time_dim, handle_dynamic_dims=False)
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step, test_single_batch_entry=False)


def test_where():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "cond": Tensor("cond", [batch_dim, time_dim], dtype="bool"),
            "true": Tensor("true", [batch_dim, time_dim, in_dim], dtype="float32"),
            "false": Tensor("false", [batch_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        out = rf.where(extern_data["cond"], extern_data["true"], extern_data["false"])
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_search_sorted():
    batch_dim_ = Dim(3, name="batch")
    time_dim = Dim(13, name="time")

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        sorted_seq = rf.range_over_dim(time_dim, dtype="float32")  # [T]
        index1 = rf.search_sorted(sorted_seq, rf.constant(4.5, dims=()), axis=time_dim)  # [] -> T
        assert index1.dims == () and index1.sparse_dim == time_dim and index1.dtype == "int32"
        index1.mark_as_output("index1", shape=())
        index2 = rf.search_sorted(sorted_seq, rf.constant(4.0, dims=()), axis=time_dim)  # [] -> T
        assert index2.dims == () and index2.sparse_dim == time_dim and index2.dtype == "int32"
        index2.mark_as_output("index2", shape=())
        index3 = rf.search_sorted(sorted_seq, rf.constant(4.0, dims=()), axis=time_dim, side="right")  # [] -> T
        assert index3.dims == () and index3.sparse_dim == time_dim and index3.dtype == "int32"
        index3.mark_as_output("index3", shape=())
        index4 = rf.search_sorted(
            sorted_seq, rf.convert_to_tensor(np.array([4.0, 4.5, 5.0]), dims=[batch_dim_]), axis=time_dim
        )  # [] -> T
        assert index4.dims == (batch_dim_,) and index4.sparse_dim == time_dim and index4.dtype == "int32"
        index4.mark_as_output("index4", shape=(batch_dim_,))

    out = run_model(TensorDict(), lambda *, epoch, step: rf.Module(), _forward_step, test_tensorflow=False)
    assert out["index1"].dims == () and out["index1"].raw_tensor.item() == 5
    assert out["index2"].dims == () and out["index2"].raw_tensor.item() == 4
    assert out["index3"].dims == () and out["index3"].raw_tensor.item() == 5
    assert out["index4"].dims == (batch_dim_,) and out["index4"].raw_tensor.tolist() == [4, 5, 5]


def test_where_int():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "cond": Tensor("cond", [batch_dim, time_dim], dtype="bool"),
            "true": Tensor("true", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        out = rf.where(extern_data["cond"], extern_data["true"], 0)
        out.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


def test_copy_masked():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.Conv1d, extern_data: TensorDict):
        x = extern_data["data"]
        x = x.copy_masked(1)
        # Do some pooling to make sure the copy_masked has an effect on the output.
        x, _ = rf.pool1d(x, mode="avg", pool_size=3, strides=1, padding="same", in_spatial_dim=time_dim)
        x.mark_as_default_output(shape=(batch_dim, time_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step, test_single_batch_entry=False)


def test_cast_sparse():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.Conv1d, extern_data: TensorDict):
        x = rf.reduce_argmax(extern_data["data"], axis=in_dim)
        assert x.sparse_dim == in_dim
        x.mark_as_output("argmax", shape=[batch_dim, time_dim])
        rf.cast(x, "float32").mark_as_output("float", shape=[batch_dim, time_dim])

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)
