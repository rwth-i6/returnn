"""
Utilities to record and playback calls to `random`.
This is useful to be able to compare different backends even when they use random numbers.

It assumes that the order of `random` calls is deterministic and consistent between backends.

The logic becomes more complex when control flow is involved like conditionals or loops.
We are also storing the control flow context in the journal.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, List, Tuple
import numpy
from dataclasses import dataclass
from returnn.tensor import Tensor, Dim, ControlFlowContext
import returnn.frontend as rf


class RandomJournal:
    """random journal. see module docstring"""

    def __init__(self):
        self._entries: List[RandomJournalEntry] = []
        self._cur_entry_idx = 0
        self._graph_reader_nodes: List[Tuple[Tensor, rf.RunCtx]] = []

    def append(
        self,
        *,
        distribution: str,
        mean: Optional[Union[int, float, Tensor]] = None,
        stddev: Optional[Union[int, float, Tensor]] = None,
        bound: Optional[Union[int, float, Tensor]] = None,
        minval: Optional[Union[int, float, Tensor]] = None,
        maxval: Optional[Union[int, float, Tensor]] = None,
        seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
        static: Optional[bool] = None,
        out: Optional[Tensor[numpy.ndarray]],
    ):
        """append"""
        self._entries.append(
            RandomJournalEntry(
                out=out,
                control_flow_ctx=rf.get_current_control_flow_ctx(),
                run_ctx=rf.get_run_ctx(),
                distribution=distribution,
                mean=mean,
                stddev=stddev,
                bound=bound,
                minval=minval,
                maxval=maxval,
                seed=seed,
                static=static,
            )
        )

    def get_next(self, *, new_out_template: Optional[Tensor] = None) -> RandomJournalEntry:
        """read next"""
        assert self._cur_entry_idx < len(self._entries)
        entry = self._entries[self._cur_entry_idx]
        if new_out_template:
            assert new_out_template.dtype == entry.out.dtype, (
                f"random journal entry dtype mismatch,"
                f" expected {new_out_template}, got {entry.out} at index {self._cur_entry_idx}"
            )
            assert len(new_out_template.dims) == len(entry.out.dims), (
                f"random journal entry dims mismatch,"
                f" expected {new_out_template}, got {entry.out} at index {self._cur_entry_idx}"
            )
            for new_dim, old_dim in zip(new_out_template.dims, entry.out.dims):
                new_dim: Dim
                old_dim: Dim
                assert new_dim.dimension == old_dim.dimension, (
                    f"random journal entry dim mismatch,"
                    f" expected {new_out_template}, got {entry.out} at index {self._cur_entry_idx}"
                )
        self._cur_entry_idx += 1
        return entry

    def reached_end(self) -> bool:
        """reached end"""
        return self._cur_entry_idx >= len(self._entries)

    def add_graph_reader_node(self, out):
        """
        In graph mode, if reading (get_next), at cgraph construction time,
        register that we are reading from the journal
        now in the current context.
        This is used in :func:`get_recent_graph_reader_node_in_accessible_ctx`.
        """
        self._graph_reader_nodes.append((out, rf.get_run_ctx()))

    def get_graph_reader_idx(self) -> int:
        """current index"""
        return len(self._graph_reader_nodes)

    def get_recent_graph_reader_node_in_accessible_ctx(self) -> Optional[Tensor]:
        """
        From the graph reader nodes, return the most recent one which is in an accessible context.
        Accessible context means either the same, or a parent context.
        """
        cur_control_flow_ctx = rf.get_current_control_flow_ctx()
        cur_run_ctx = rf.get_run_ctx()
        for prev_out, prev_run_ctx in reversed(self._graph_reader_nodes):
            if prev_run_ctx != cur_run_ctx:
                return None
            if ControlFlowContext.is_parent_or_same(prev_out.control_flow_ctx, cur_control_flow_ctx):
                return prev_out
            consumers = rf.walk_tensor_consumers(
                prev_out,
                filter_outputs=lambda x: ControlFlowContext.is_parent_or_same(x.control_flow_ctx, cur_control_flow_ctx),
                ending_condition=lambda x: ControlFlowContext.is_parent_or_same(
                    x.control_flow_ctx, cur_control_flow_ctx
                ),
            )
            if not consumers:
                raise Exception(f"cannot handle {prev_out} in current {cur_control_flow_ctx}")
            return consumers[0]
        return None


@dataclass
class RandomJournalEntry:
    """entry"""

    out: Optional[Tensor[numpy.ndarray]]

    control_flow_ctx: Optional[ControlFlowContext]
    run_ctx: rf.RunCtx

    distribution: str
    mean: Optional[Union[int, float, Tensor]] = None
    stddev: Optional[Union[int, float, Tensor]] = None
    bound: Optional[Union[int, float, Tensor]] = None
    minval: Optional[Union[int, float, Tensor]] = None
    maxval: Optional[Union[int, float, Tensor]] = None
    seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None
    static: Optional[bool] = None
