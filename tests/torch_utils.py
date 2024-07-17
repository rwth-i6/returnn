"""
Utilities for PyTorch tests
"""

from __future__ import annotations
import os
import torch


def report_profile(prof: torch.profiler.profiler, check_events=(), *, _size_threshold=100) -> None:
    """
    Report profile
    """
    # Note: I tried prof.events(), prof.profiler.kineto_results.events(), prof._memory_profile().timeline,
    # but they all are not really giving me the information I want.
    # Either the Python stack is missing, or the memory information is incomplete,
    # or the Python/TorchOp events are missing.
    # The only complete information source seems to be prof.profiler.kineto_results.experimental_event_tree().

    import fnmatch

    # noinspection PyProtectedMember
    from torch.profiler._utils import traverse_dfs
    from torch._C._profiler import _EventType  # noqa

    _allocs = {}  # id -> dict with "size", "name"
    check_events = list(check_events)

    def _ev_visit(ev):
        # ev: torch._C._profiler._ProfilerEvent
        if ev.typed[0] == _EventType.Allocation:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_Allocation
            # ex.id/ex.allocation_id/ex.ptr redundant?
            if ex.allocation_id in _allocs:
                ev_name = "dealloc"  # deallocation
                assert _allocs[ex.allocation_id]["size"] == -ex.alloc_size
                name = _allocs[ex.allocation_id]["name"]
                del _allocs[ex.allocation_id]
            else:
                ev_name = "alloc"
                assert ex.alloc_size > 0
                assert ev.parent
                name = _ctx(ev.parent)
                _allocs[ex.allocation_id] = {"size": ex.alloc_size, "name": name}
            opts = {"id": ex.allocation_id, "name": name, "size": ex.alloc_size, "total_alloc": ex.total_allocated}
        elif ev.typed[0] == _EventType.TorchOp:
            ev_name = "torchop"
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_TorchOp
            opts = {"name": ex.name}
        elif ev.typed[0] == _EventType.PyCall:
            ev_name = "pycall"
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_PyCall
            ex0 = ex.caller  # torch._C._profiler._PyFrameState
            ex1 = ex.callsite  # torch._C._profiler._PyFrameState
            if _pycall_filter_fn(ex0.file_name) or _pycall_filter_fn(ex1.file_name):
                opts = {
                    "caller_loc": f"{ex0.file_name}:{ex0.line_number}",
                    "caller_name": ex0.function_name,
                    "callsite_name": ex1.function_name,
                }
            else:
                return
        else:
            return

        next_check = check_events[0] if check_events else None
        if next_check:
            next_check_name, next_check_opts = next_check
            if ev_name == next_check_name:
                for k, v in next_check_opts.items():
                    if isinstance(v, str) and "*" in v:
                        if not fnmatch.fnmatch(opts[k], v):
                            mismatch = f"Pattern mismatch: {opts[k]} vs {v}"
                            break
                    elif k == "total_alloc":
                        if abs(opts[k] - v) >= _size_threshold:
                            mismatch = f"Size mismatch: {opts[k]} vs {v}"
                            break
                    elif opts[k] != v:
                        mismatch = f"Value mismatch: {opts[k]} vs {v}"
                        break
                else:
                    mismatch = None
            else:
                mismatch = f"Different event: {ev_name} vs {next_check_name}"
        else:
            mismatch = "No check event"

        if ev_name in {"alloc", "dealloc"} and abs(opts["size"]) >= _size_threshold:
            assert not mismatch, f"Event not matched: {ev_name} {opts} to {next_check}: {mismatch}"

        if not mismatch:
            print(f"{ev_name} {opts} ✓")
            check_events.pop(0)
        else:
            print(f"({ev_name} {opts})")

    def _ctx(ev) -> str:
        stack = [None]
        parent = ev
        while parent and parent.typed[0] == _EventType.TorchOp:  # go to top torch op
            stack[-1] = parent.typed[1].name
            parent = parent.parent
        if not stack[-1] and parent.typed[0] == _EventType.PyCCall:
            stack[-1] = parent.typed[1].caller.function_name
            parent = parent.parent
        if not stack[-1]:
            stack.pop(-1)
        while parent:
            if parent.typed[0] == _EventType.PyCall:
                ex0 = parent.typed[1].caller  # torch._C._profiler._PyFrameState
                ex1 = parent.typed[1].callsite  # torch._C._profiler._PyFrameState
                if (
                    _pycall_filter_fn(ex1.file_name)
                    or (_pycall_filter_fn(ex0.file_name) and ex1.function_name == "backward")
                ) and ex1.function_name not in {"__torch_dispatch__"}:
                    stack.append(ex1.function_name)
            parent = parent.parent
        stack.reverse()
        return "/".join(stack) or "unknown"

    for ev_ in sorted(
        traverse_dfs(prof.profiler.kineto_results.experimental_event_tree()), key=lambda ev: ev.start_time_ns
    ):
        # ev: torch._C._profiler._ProfilerEvent
        _ev_visit(ev_)

    assert not _allocs, f"Remaining allocs: {_allocs}"
    assert not check_events, f"Remaining check events: {check_events}"


def _pycall_filter_fn(filename: str) -> bool:
    assert not filename.startswith("/")  # currently the case...
    if os.path.basename(filename) == os.path.basename(__file__):
        assert "/" not in filename  # currently the case...
        return True
    if filename.startswith("returnn/"):
        return True
    return False
