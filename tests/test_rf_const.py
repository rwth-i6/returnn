"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
from typing import Tuple
import _setup_test_env  # noqa
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, TensorDict
from rf_utils import run_model


def test_constant_bool():
    class _Net(rf.Module):
        def __call__(self) -> Tuple[Tensor, Dim]:
            dim = Dim(3, name="dim")
            return rf.constant(False, dims=[dim]), dim

    # noinspection PyShadowingNames
    def _forward_step(*, model: _Net, extern_data: TensorDict):
        extern_data  # noqa
        out, dim = model()
        assert out.dtype == "bool"
        out.mark_as_default_output(shape=[dim])

    run_model(TensorDict(), lambda *, epoch, step: _Net(), _forward_step)
