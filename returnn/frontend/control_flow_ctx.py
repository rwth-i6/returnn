"""
Control flow context logic
"""

from __future__ import annotations
from typing import Optional
from contextlib import contextmanager
from returnn.tensor import ControlFlowContext


__all__ = ["control_flow_ctx", "get_current_control_flow_ctx"]


_ctx: Optional[ControlFlowContext] = None


@contextmanager
def control_flow_ctx(ctx: Optional[ControlFlowContext] = None):
    """
    Activates the given control flow context.
    """
    global _ctx
    prev_ctx = _ctx
    try:
        _ctx = ctx
        yield ctx
    finally:
        _ctx = prev_ctx


def get_current_control_flow_ctx() -> Optional[ControlFlowContext]:
    """
    :return: current control flow context
    """
    return _ctx
