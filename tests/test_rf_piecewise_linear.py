"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
import numpy as np
import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict
from rf_utils import run_model


def test_piecewise_linear():
    batch_dim_ = Dim(13, name="batch")

    eps = 1e-5
    tests = np.array(
        [
            (0.0, 2.0),
            (1.0 - eps, 2.0),
            (1.0, 2.0),
            (1.0 + eps, 2.0),
            (2.0, 3.0),
            (3.0 - eps, 4.0),
            (3.0, 4.0),
            (3.0 + eps, 4.0),
            (4.0, 2.5),
            (5.0 - eps, 1.0),
            (5.0, 1.0),
            (5.0 + eps, 1.0),
            (6.0, 1.0),
        ],
        dtype="float32",
    )

    # noinspection PyShadowingNames,PyUnusedLocal
    def _forward_step(*, model: rf.PiecewiseLinear, extern_data: TensorDict):
        values = rf.convert_to_tensor(tests[:, 0], dims=[batch_dim_])
        out = model(values)
        out.mark_as_default_output(shape=(batch_dim_,))

    out = run_model(
        TensorDict(),
        lambda *, epoch, step: rf.PiecewiseLinear({1: 2.0, 3: 4.0, 5: 1.0}),
        _forward_step,
        test_tensorflow=False,
    )
    out = out["output"]
    np.testing.assert_almost_equal(out.raw_tensor, tests[:, 1], decimal=4)
