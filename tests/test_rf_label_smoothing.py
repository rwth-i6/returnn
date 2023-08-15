"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_label_smoothed_log_prob_gradient():
    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    vocab_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, vocab_dim], dtype="float32", feature_dim=vocab_dim),
            "targets": Tensor("targets", [batch_dim, time_dim], dtype="int32", sparse_dim=vocab_dim),
        }
    )

    # noinspection PyShadowingNames
    def _forward_step(*, model: rf.Module, extern_data: TensorDict):
        model  # noqa  # unused
        data = extern_data["data"]
        targets = extern_data["targets"]
        rf.set_requires_gradient(data)

        log_prob = rf.log_softmax(data, axis=vocab_dim)
        out = rf.label_smoothed_log_prob_gradient(log_prob, 0.1)
        loss = rf.cross_entropy(target=targets, estimated=log_prob, estimated_type="log-probs", axis=vocab_dim)

        out.mark_as_default_output(shape=(batch_dim, time_dim, vocab_dim))
        loss.mark_as_output("loss")

        grad = rf.gradient(rf.reduce_sum(loss, axis=loss.dims), data)
        grad.mark_as_output("grad")

    run_model(extern_data, lambda *, epoch, step: rf.Module(), _forward_step)
