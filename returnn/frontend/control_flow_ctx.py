"""
Control flow context logic
"""

from __future__ import annotations
from typing import Optional
from contextlib import contextmanager
from returnn.tensor import ControlFlowContext


__all__ = ["control_flow_ctx"]


@contextmanager
def control_flow_ctx(ctx: Optional[ControlFlowContext] = None):
    """
    Activates the given control flow context.
    """
    ctx  # noqa  # TODO ...
    # TODO ...
    yield
