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
        labels_, _ = rf.pack_padded(labels, dims=[hyps_dim, hyps_spatial_dim], out_dim=hyps_packed_spatial_dim)
        assert hyps_packed_spatial_dim.dyn_size_ext.dims == (batch_dim,)
        labels_.mark_as_default_output(shape=[batch_dim, hyps_packed_spatial_dim])

    run_model(extern_data, lambda **_kwargs: rf.Module(), _forward_step, test_tensorflow=False)


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

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


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

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


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

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)


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
