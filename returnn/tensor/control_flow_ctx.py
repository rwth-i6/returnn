"""
Control flow context
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING, List
import returnn.util.basic as util

if TYPE_CHECKING:
    from tensorflow.python.ops.control_flow_ops import ControlFlowContext as _TFControlFlowCtx
    from .dim import Dim


__all__ = ["ControlFlowContext"]


class ControlFlowContext:
    """
    This represents the current control flow context, e.g. whether this runs in a loop or a conditional branch.

    In case of TF,
    this is a simple wrapper around the TF ControlFlowContext which comes from tf.while_loop or tf.cond.

    We have this wrapper to refer to a context which might not exist yet (e.g. at template construction time).
    Also, we might want to store additional information, such the spatial dim tag of the loop.
    """

    class Types:
        """
        Possible types of context.
        """

        Loop = "loop"
        CondTrue = "cond-true"
        CondFalse = "cond-false"

    def __init__(self, *, kind: str, identifier: str, outer_ctx: Optional[ControlFlowContext] = None):
        """
        :param kind: from ControlFlowContext.Types
        :param outer_ctx:
        """
        self.kind = kind
        self.identifier = identifier
        self._outer_ctx = outer_ctx
        self._tf_control_flow_ctx = None  # type: Optional[_TFControlFlowCtx]
        self._loop_spatial_dim = None  # type: Optional[Dim]

    def __repr__(self):
        return "ControlFlowContext{%s}" % self.repr_inner()

    def repr_inner(self):
        """
        :rtype: str
        """
        return "/".join(ctx._repr_single() for ctx in self._abs_ctx_stack())

    def _repr_single(self):
        """
        :rtype: str
        """
        s = self.kind
        if self.is_loop() and self.loop_spatial_dim:
            try:
                with util.guard_infinite_recursion(ControlFlowContext._repr_single, self):
                    s += "(%s)" % self.loop_spatial_dim.short_repr()
            except util.InfiniteRecursionDetected as exc:
                # repr should always return and not throw errors
                s += "(%s for loop_spatial_dim?)" % exc
        return s

    def _abs_ctx_stack(self):
        """
        :rtype: list[ControlFlowContext]
        :return: chain of ctx, last is self
        """
        chain = []
        ctx = self
        while ctx:
            chain.append(ctx)
            ctx = ctx.outer_ctx
        chain.reverse()
        return chain

    @classmethod
    def abs_ctx_stack(cls, ctx):
        """
        :param ControlFlowContext|None ctx:
        :rtype: list[ControlFlowContext]
        """
        if ctx:
            return ctx._abs_ctx_stack()
        return []

    @classmethod
    def abs_ctx_stack_with_root(cls, ctx):
        """
        :param ControlFlowContext|None ctx:
        :rtype: list[ControlFlowContext|None]
        :return: chain of ctx, last is self, first is None
        """
        ls = [None]  # type: List[Optional[ControlFlowContext]]
        if ctx:
            ls += ctx._abs_ctx_stack()
        return ls

    @classmethod
    def is_parent_or_same(cls, parent, child):
        """
        :param ControlFlowContext|None parent:
        :param ControlFlowContext|None child:
        :rtype: bool
        """
        if parent == child:
            return True
        if not parent:
            return True  # parent is root
        if not child:
            return False  # child is root but parent is not
        while child:
            if child == parent:
                return True
            child = child.outer_ctx
        return False

    @classmethod
    def collect_parent_dims(cls, ctx):
        """
        :param ControlFlowContext|None ctx:
        :rtype: list[Dim]
        """
        dims = []
        for ctx_ in ControlFlowContext.abs_ctx_stack(ctx):
            if ctx_.is_loop() and ctx_.loop_spatial_dim:
                dims.append(ctx_.loop_spatial_dim)
        return dims

    def is_loop(self):
        """
        :rtype: bool
        """
        return self.kind == self.Types.Loop

    def is_cond(self):
        """
        :rtype: bool
        """
        return self.kind in {self.Types.CondTrue, self.Types.CondFalse}

    @property
    def outer_ctx(self):
        """
        :rtype: ControlFlowContext|None
        """
        return self._outer_ctx

    @property
    def tf_control_flow_ctx(self):
        """
        :rtype: tensorflow.python.ops.control_flow_ops.ControlFlowContext|None
        """
        return self._tf_control_flow_ctx

    @tf_control_flow_ctx.setter
    def tf_control_flow_ctx(self, ctx):
        """
        :param tensorflow.python.ops.control_flow_ops.ControlFlowContext ctx:
        """
        if self.is_loop():
            assert ctx.IsWhileContext()
        if self.is_cond():
            assert ctx.IsCondContext()
        self._tf_control_flow_ctx = ctx

    @property
    def loop_spatial_dim(self):
        """
        :rtype: Dim|None
        """
        assert self.is_loop()
        return self._loop_spatial_dim

    @loop_spatial_dim.setter
    def loop_spatial_dim(self, dim):
        """
        :param Dim dim:
        """
        assert self.is_loop()
        self._loop_spatial_dim = dim
