"""
Utilities for PyTorch tests
"""

from __future__ import annotations
from typing import Optional, Any, Sequence, Tuple, List, Dict
import torch


def report_profile(
    prof: torch.profiler.profiler,
    check_events: Optional[Sequence[Tuple[str, Dict[str, Any]]]] = None,
    *,
    allow_remaining_allocs: bool = False,
    _size_threshold: int = 100,
) -> None:
    """
    Report profile.

    Example usage::

        from torch.profiler import profile, ProfilerActivity

        with profile(
            activities=[ProfilerActivity.CPU], profile_memory=True, with_stack=True, record_shapes=True
        ) as prof:
            # code to profile
            ...

        report_profile(prof)

    :param prof: via torch.profiler.profile.
    :param check_events: if given, will check that the report matches the given events.
        Each entry is a tuple (event_name, event_opts).
        The event_name can be "alloc", "dealloc", "torchop", "pycall".
        The event_opts is a dict with the expected values.
        You can use "*" as a wildcard in the event_opts.
    :param allow_remaining_allocs: if True, will not raise an error if there are remaining allocations.
    :param _size_threshold: internal
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

    _printed_events = set()
    _events_stack_depth = {}
    _allocs = {}  # id -> dict with "size", "name"
    check_events = list(check_events) if check_events is not None else None

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
                name = _ev_ctx(ev.parent)
                _allocs[ex.allocation_id] = {"size": ex.alloc_size, "name": name}
            opts = {"id": ex.allocation_id, "name": name, "size": ex.alloc_size, "total_alloc": ex.total_allocated}
        elif ev.typed[0] == _EventType.TorchOp:
            ev_name = "torchop"
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_TorchOp
            opts = {"name": ex.name, "inputs": [_repr_tensor_metadata(i) for i in ex.inputs]}
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
        elif ev.typed[0] == _EventType.PyCCall:
            ev_name = "pyccall"
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_PyCCall
            ex0 = ex.caller  # torch._C._profiler._PyFrameState
            if _pycall_filter_fn(ex0.file_name):
                opts = {
                    "caller_loc": f"{ex0.file_name}:{ex0.line_number}",
                    "caller_name": ex0.function_name,
                }
            else:
                return
        else:
            return

        depth = 0
        if ev.parent:
            ev__ = ev.parent
            while ev__:
                if ev__ in _printed_events:
                    depth = _events_stack_depth[ev__] + 1
                    break
                if ev__ in _events_stack_depth:
                    depth = _events_stack_depth[ev__]
                    break
                ev__ = ev__.parent
            ev__ = ev
            while ev__:
                if ev__ in _events_stack_depth:
                    break
                _events_stack_depth[ev__] = depth
                ev__ = ev__.parent
        else:
            _events_stack_depth[ev] = 0
        _printed_events.add(ev)
        prefix = "  " * depth

        if check_events is None:
            print(f"{prefix}{ev_name} {opts}")
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
            print(f"{prefix}{ev_name} {opts} ✓")
            check_events.pop(0)
        else:
            print(f"{prefix}({ev_name} {opts})")

    for ev_ in sorted(
        traverse_dfs(prof.profiler.kineto_results.experimental_event_tree()), key=lambda ev: ev.start_time_ns
    ):
        # ev: torch._C._profiler._ProfilerEvent
        _ev_visit(ev_)

    if allow_remaining_allocs:
        for alloc_id, alloc in _allocs.items():
            print(f"Remaining alloc: {alloc_id} {alloc}")
    else:
        assert not _allocs, f"Remaining allocs: {_allocs}"
    assert not check_events, f"Remaining check events: {check_events}"


def get_remaining_allocs_from_profile(prof: torch.profiler.profiler) -> Dict[int, Dict[str, Any]]:
    """
    Get remaining allocs from profile.

    :param prof: via torch.profiler.profile.
    :return: allocs dict: id -> dict with "size", "name"
    """
    # noinspection PyProtectedMember
    from torch.profiler._utils import traverse_dfs
    from torch._C._profiler import _EventType  # noqa

    _allocs = {}  # id -> dict with "size", "name"

    for ev in sorted(
        traverse_dfs(prof.profiler.kineto_results.experimental_event_tree()), key=lambda ev: ev.start_time_ns
    ):
        # ev: torch._C._profiler._ProfilerEvent
        if ev.typed[0] == _EventType.Allocation:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_Allocation
            # ex.id/ex.allocation_id/ex.ptr redundant?
            if ex.allocation_id in _allocs:
                # expect deallocation
                assert _allocs[ex.allocation_id]["size"] == -ex.alloc_size
                del _allocs[ex.allocation_id]
            else:
                # allocation
                assert ex.alloc_size > 0
                assert ev.parent
                name = _ev_ctx(ev.parent)
                _allocs[ex.allocation_id] = {"size": ex.alloc_size, "name": name}

    return _allocs


def get_allocs_from_profile(prof: torch.profiler.profiler) -> List[Dict[str, Any]]:
    """
    Get allocs from profile.

    :param prof: via torch.profiler.profile.
    :return: allocs dict: id -> dict with "size", "name"
    """
    # noinspection PyProtectedMember
    from torch.profiler._utils import traverse_dfs
    from torch._C._profiler import _EventType  # noqa

    _allocs = []  # dict with "id", "size", "name"

    for ev in sorted(
        traverse_dfs(prof.profiler.kineto_results.experimental_event_tree()), key=lambda ev: ev.start_time_ns
    ):
        # ev: torch._C._profiler._ProfilerEvent
        if ev.typed[0] == _EventType.Allocation:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_Allocation
            if ex.alloc_size > 0:
                assert ev.parent
                name = _ev_ctx(ev.parent)
                _allocs.append({"id": ex.allocation_id, "size": ex.alloc_size, "name": name})

    return _allocs


def get_peak_alloc_from_profile(prof: torch.profiler.profiler) -> int:
    """
    Get remaining allocs from profile.

    :param prof: via torch.profiler.profile.
    :return: peak alloc size
    """
    # noinspection PyProtectedMember
    from torch.profiler._utils import traverse_dfs
    from torch._C._profiler import _EventType  # noqa

    _allocs = {}  # id -> dict with "size", "name"
    peak_alloc = 0

    for ev in sorted(
        traverse_dfs(prof.profiler.kineto_results.experimental_event_tree()), key=lambda ev: ev.start_time_ns
    ):
        # ev: torch._C._profiler._ProfilerEvent
        # ev: torch._C._profiler._ProfilerEvent
        if ev.typed[0] == _EventType.Allocation:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_Allocation
            # ex.id/ex.allocation_id/ex.ptr redundant?
            if ex.allocation_id in _allocs:
                # expect deallocation
                assert _allocs[ex.allocation_id]["size"] == -ex.alloc_size
                del _allocs[ex.allocation_id]
            else:
                # allocation
                assert ex.alloc_size > 0
                assert ev.parent
                name = _ev_ctx(ev.parent)
                _allocs[ex.allocation_id] = {"size": ex.alloc_size, "name": name}

                cur_total_alloc = sum(alloc["size"] for alloc in _allocs.values())
                if cur_total_alloc > peak_alloc:
                    peak_alloc = cur_total_alloc

    return peak_alloc


def _pycall_filter_fn(filename: str) -> bool:
    assert not filename.startswith("/")  # currently the case...
    if filename.startswith("test_"):
        assert "/" not in filename  # currently the case...
        return True
    if filename.startswith("returnn/"):
        return True
    return False


def _ev_ctx(ev) -> str:
    from torch._C._profiler import _EventType  # noqa

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


def _repr_tensor_metadata(x) -> Any:
    """
    :param x: torch._C._profiler._TensorMetadata or int
    :return: obj which has a nice __repr__/__str__
    """
    if isinstance(x, (list, tuple)):
        return type(x)(_repr_tensor_metadata(i) for i in x)
    if not type(x).__module__.startswith("torch"):
        return x
    assert x is not None
    return _ReprObj(f"<_T id={x.id} shape={x.sizes} dtype={x.dtype}>")


class _ReprObj:
    def __init__(self, obj: str):
        self.obj = obj

    def __repr__(self):
        return self.obj
