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
