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
    assert new_time_dim == time_dim + 1  # math dim... not really necessary check here...
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
