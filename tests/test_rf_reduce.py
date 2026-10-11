"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_reduce_max():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.reduce_max(x, axis=in_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_reduce_argmax():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.reduce_argmax(x, axis=in_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, time_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_reduce_mean_dyn_time():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.reduce_mean(x, axis=time_dim)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(batch_dim, in_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_reduce_mean_dyn_batch_time():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tensor:
            return rf.reduce_mean(x, axis=(batch_dim, time_dim))

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        out = model(extern_data["data"])
        out.mark_as_default_output(shape=(in_dim,))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_reduce_bounded_batch_padding_seqs():
    """
    a batch dim with the real batch size as dyn size and a capacity (a batch filled up to a bound with padding seqs):
    the padding seqs have no frames, in the seq mask and in every reduction
    """
    rf.select_backend_torch()
    batch_size = rf.convert_to_tensor(3, dims=(), dtype="int32")
    bounded_batch_dim = Dim(batch_size, name="batch", kind=Dim.Types.Batch, capacity=5)
    lens = rf.convert_to_tensor([7, 4, 2, 0, 0], dims=[bounded_batch_dim], dtype="int32")
    time_dim = Dim(lens, name="time", capacity=7)
    valid = rf.compare_bc(rf.range_over_dim(time_dim), "<", lens)  # padding frames, also those of the padding seqs
    x = rf.where(valid, 1.0, 100.0)
    assert (
        time_dim.get_mask().copy_compatible_to_dims_raw([bounded_batch_dim, time_dim]).tolist()
        == valid.copy_compatible_to_dims_raw([bounded_batch_dim, time_dim]).tolist()
    )
    assert rf.reduce_sum(x, axis=[bounded_batch_dim, time_dim]).raw_tensor.tolist() == 13.0
    assert rf.reduce_mean(x, axis=[bounded_batch_dim, time_dim]).raw_tensor.tolist() == 1.0
    assert rf.reduce_max(x, axis=[bounded_batch_dim, time_dim]).raw_tensor.tolist() == 1.0
    per_seq = rf.reduce_sum(x, axis=time_dim)
    assert per_seq.raw_tensor.tolist() == [7.0, 4.0, 2.0, 0.0, 0.0]
    assert abs(rf.reduce_mean(per_seq, axis=bounded_batch_dim).raw_tensor.tolist() - 13 / 3) < 1e-6
    # not the padding seqs' 0 - 10
    assert rf.reduce_max(per_seq - 10.0, axis=bounded_batch_dim).raw_tensor.tolist() == -3.0


def test_top_k():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Dim]:
            return rf.top_k(x, axis=in_dim, k=2)

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        values, indices, k_dim = model(extern_data["data"])
        values.mark_as_output("values", shape=(batch_dim, time_dim, k_dim))
        indices.mark_as_output("indices", shape=(batch_dim, time_dim, k_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)


def test_top_k_beam_search():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    vocab_dim = Dim(7, name="vocab")
    beam_in_dim = Dim(3, name="beam_in")
    beam_out_dim = Dim(5, name="beam_out")
    extern_data = TensorDict(
        {
            "log_probs": Tensor("log_probs", [batch_dim, beam_in_dim, time_dim, vocab_dim], dtype="float32"),
        }
    )

    class _Net(rf.Module):
        def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
            log_probs, (indices_beam_in, indices_vocab), _ = rf.top_k(
                x, axis=[beam_in_dim, vocab_dim], k_dim=beam_out_dim, k=beam_out_dim.dimension
            )
            return log_probs, indices_beam_in, indices_vocab

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        log_probs, indices_beam_in, indices_vocab = model(extern_data["log_probs"])
        log_probs.mark_as_output("log_probs", shape=(batch_dim, time_dim, beam_out_dim))
        indices_beam_in.mark_as_output("indices_beam_in", shape=(batch_dim, time_dim, beam_out_dim))
        indices_vocab.mark_as_output("indices_vocab", shape=(batch_dim, time_dim, beam_out_dim))

    run_model(extern_data, lambda *, epoch, step: _Net(), _forward_step)
