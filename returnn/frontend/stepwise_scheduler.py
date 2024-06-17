"""
Stepwise scheduler, e.g. for learning rate or other hyperparameters.

All these modules will accept any args/kwargs but leave them unused,
and instead uses :func:`get_run_ctx` to get the current train step from the current run context.
"""

from __future__ import annotations
from returnn.tensor import Tensor
import returnn.frontend as rf
from .piecewise_linear import PiecewiseLinear


__all__ = ["PiecewiseLinearStepwiseScheduler"]


class PiecewiseLinearStepwiseScheduler(PiecewiseLinear):
    """
    Piecewise linear scheduler based on the current global train step.

    Example::

        scheduler = PiecewiseLinearStepwiseScheduler(
            {0: 1.0, 10000: 0.1, 20000: 0.01}
        )

    This will start with 1.0, and then linearly decay to 0.1 at step 10000, and then to 0.01 at step 20000.
    """

    def __call__(self, *args, **kwargs) -> Tensor:
        return super().__call__(rf.get_run_ctx_step())
