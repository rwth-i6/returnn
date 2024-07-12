"""
Test :mod:`returnn.torch.util`.
"""

from __future__ import annotations

import _setup_test_env  # noqa

import os
import sys
import unittest
import torch

from returnn.util import better_exchook
from returnn.torch.util.gradient_checkpoint import gradient_checkpoint_scope


def test_gradient_checkpoint_scope():
    # https://github.com/rwth-i6/returnn/issues/1552
    from copy import deepcopy
    from torch.profiler import profile, record_function, ProfilerActivity

    shape = (101, 103)

    class _Model(torch.nn.Module):
        def __init__(self, *, use_grad_ckpt: bool = False):
            super().__init__()
            self.var = torch.nn.Parameter(torch.randn(shape))
            self.input_var = torch.nn.Parameter(torch.randn(shape))
            self.opt = torch.optim.SGD(self.parameters(), lr=0.1)  # not common to have this here but ok for the test
            self.use_grad_ckpt = use_grad_ckpt

        @staticmethod
        def get_var_noise() -> torch.Tensor:
            return torch.randn(shape)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.use_grad_ckpt:
                return (self.var + self.get_var_noise()) * x

            with gradient_checkpoint_scope():
                v_ = self.var + self.get_var_noise()
            return v_ * x

        def demo_run(self):
            x = self.input_var
            y = self(x)
            loss = y.sum()  # dummy loss
            del x, y  # not needed anymore. makes test cleaner.
            loss.backward()
            del loss  # not needed anymore
            self.opt.step()
            self.opt.zero_grad()

    model = _Model(use_grad_ckpt=False)
    param_state = deepcopy(model.state_dict())
    rng_state = torch.get_rng_state()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, with_stack=True, record_shapes=True) as prof:
        with record_function("train_step_no_grad_ckpt"):
            model.demo_run()
    b = 4  # size single f32
    t = shape[0] * shape[1] * b  # size tensor
    r = rng_state.numel() * rng_state.element_size()
    _report_profile(
        prof,
        [
            # ignore private calls
            # ignore torchop
            # ignore allocs/deallocs <=b
            ("pycall", {"callsite_name": "demo_run"}),
            ("pycall", {"callsite_name": "forward"}),
            ("pycall", {"callsite_name": "get_var_noise"}),
            ("alloc", {"name": "*/get_var_noise/aten::randn", "size": t, "total_alloc": t}),
            ("alloc", {"name": "*/forward/aten::add", "size": t, "total_alloc": t * 2}),
            ("dealloc", {"name": "*/get_var_noise/aten::randn", "size": -t, "total_alloc": t}),
            ("alloc", {"name": "*/forward/aten::mul", "size": t, "total_alloc": t * 2}),
            ("dealloc", {"name": "*/forward/aten::mul", "size": -t, "total_alloc": t}),
            ("pycall", {"callsite_name": "backward"}),
            ("alloc", {"name": "*/backward/*MulBackward0", "size": t, "total_alloc": t * 2}),
            ("alloc", {"name": "*/backward/*MulBackward0", "size": t, "total_alloc": t * 3}),
            ("dealloc", {"name": "*/forward/aten::add", "size": -t, "total_alloc": t * 2}),
            ("torchop", {"name": "Optimizer.step#SGD.step"}),
            ("torchop", {"name": "Optimizer.zero_grad#SGD.zero_grad"}),
            ("dealloc", {"name": "*/backward/*MulBackward0", "size": -t, "total_alloc": t}),
            ("dealloc", {"name": "*/backward/*MulBackward0", "size": -t, "total_alloc": 0}),
        ],
    )
    param_post_state = deepcopy(model.state_dict())

    print("**** now with grad chkpt ****")
    model = _Model(use_grad_ckpt=True)
    model.load_state_dict(param_state)
    torch.set_rng_state(rng_state)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, with_stack=True, record_shapes=True) as prof:
        with record_function("train_step_grad_ckpt"):
            model.demo_run()
    _report_profile(
        prof,
        [
            ("pycall", {"callsite_name": "demo_run"}),
            ("pycall", {"callsite_name": "forward"}),
            ("pycall", {"callsite_name": "get_var_noise"}),
            ("pycall", {"callsite_name": "__torch_dispatch__"}),
            ("alloc", {"name": "*/get_rng_state", "size": r, "total_alloc": r}),
            ("alloc", {"name": "*/get_var_noise/aten::randn", "size": t, "total_alloc": t + r}),
            ("pycall", {"callsite_name": "record_op"}),
            ("alloc", {"name": "*/forward/aten::add", "size": t, "total_alloc": t * 2 + r}),
            ("pycall", {"callsite_name": "record_op"}),
            ("dealloc", {"name": "*/get_var_noise/aten::randn", "size": -t, "total_alloc": t + r}),
            ("pycall", {"callsite_name": "_pack_hook"}),
            ("alloc", {"name": "*/forward/aten::mul", "size": t, "total_alloc": t * 2 + r}),
            # Not sure that we can always rely on this order here in the test...
            ("pycall", {"callsite_name": "_tensor_del_hook"}),
            ("pycall", {"callsite_name": "exit_saved_tensors_hooks_scope"}),
            ("pycall", {"callsite_name": "_custom_saved_tensors_hooks_exit"}),
            ("pycall", {"callsite_name": "_unregister_custom_saved_tensors_hooks"}),
            ("dealloc", {"name": "*/forward/aten::add", "size": -t, "total_alloc": t + r}),  # !!
            ("dealloc", {"name": "*/forward/aten::mul", "size": -t, "total_alloc": r}),
            ("pycall", {"callsite_name": "backward"}),
            ("pycall", {"callsite_name": "_unpack_hook"}),
            ("pycall", {"callsite_name": "maybe_recompute"}),
            ("pycall", {"callsite_name": "get_rng_state"}),
            ("alloc", {"name": "*/get_rng_state", "size": r, "total_alloc": r * 2}),
            ("pycall", {"callsite_name": "set_rng_state"}),
            ("pycall", {"callsite_name": "recompute"}),
            (
                "alloc",
                {
                    "name": "*/backward/_unpack_hook/maybe_recompute/recompute/aten::randn",
                    "size": t,
                    "total_alloc": t + r * 2,
                },
            ),
            ("alloc", {"name": "*/backward/*/recompute/aten::add", "size": t, "total_alloc": t * 2 + r * 2}),
            ("dealloc", {"name": "*/recompute/aten::randn", "size": -t, "total_alloc": t + r * 2}),
            ("pycall", {"callsite_name": "set_rng_state"}),
            ("dealloc", {"name": "*/backward/*/get_rng_state", "size": -r, "total_alloc": t + r}),
            ("dealloc", {"name": "*/forward/*/get_rng_state", "size": -r, "total_alloc": t}),
            ("alloc", {"name": "*/backward/*MulBackward0", "size": t, "total_alloc": t * 2}),
            ("alloc", {"name": "*/backward/*MulBackward0", "size": t, "total_alloc": t * 3}),
            ("dealloc", {"name": "*/recompute/aten::add", "size": -t, "total_alloc": t * 2}),
            ("torchop", {"name": "Optimizer.step#SGD.step"}),
            ("torchop", {"name": "Optimizer.zero_grad#SGD.zero_grad"}),
            ("dealloc", {"name": "*/backward/*MulBackward0", "size": -t, "total_alloc": t}),
            ("dealloc", {"name": "*/backward/*MulBackward0", "size": -t, "total_alloc": 0}),
        ],
    )
    param_post_state_ = deepcopy(model.state_dict())
    assert set(param_post_state.keys()) == set(param_post_state_.keys())
    for k in param_post_state.keys():
        torch.testing.assert_allclose(param_post_state[k], param_post_state_[k])


def test_gradient_checkpoint_scope_twice():
    # https://github.com/rwth-i6/returnn/issues/1579
    shape = (101, 103)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.var = torch.nn.Parameter(torch.randn(shape))
            self.input_var = torch.nn.Parameter(torch.randn(shape))
            self.opt = torch.optim.SGD(self.parameters(), lr=0.1)  # not common to have this here but ok for the test

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.get_var() * x

        def get_var(self) -> torch.Tensor:
            with gradient_checkpoint_scope():
                return self.var + torch.randn(shape)

        def get_input(self) -> torch.Tensor:
            x = self.input_var
            with gradient_checkpoint_scope():
                return x + torch.randn(shape)

        def demo_run(self):
            self.opt.zero_grad()
            y = self(self.get_input())
            loss = y.sum()  # dummy loss
            del y  # not needed anymore
            loss.backward()
            del loss  # not needed anymore
            self.opt.step()

    orig_gradient_checkpoint_scope_tensor_del_hook = gradient_checkpoint_scope._tensor_del_hook
    try:
        # Overwrite this here to trigger the case where the tensor del hook will not do the cleanup.
        gradient_checkpoint_scope._tensor_del_hook = lambda self: None

        model = _Model()
        model.demo_run()
        model.demo_run()

    finally:
        gradient_checkpoint_scope._tensor_del_hook = orig_gradient_checkpoint_scope_tensor_del_hook


def _report_profile(prof: torch.profiler.profiler, check_events=(), *, _size_threshold=100):
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


if __name__ == "__main__":
    better_exchook.install()
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            if k.startswith("test_"):
                print("-" * 40)
                print("Executing: %s" % k)
                try:
                    v()
                except unittest.SkipTest as exc:
                    print("SkipTest:", exc)
                print("-" * 40)
        print("Finished all tests.")
    else:
        assert len(sys.argv) >= 2
        for arg in sys.argv[1:]:
            print("Executing: %s" % arg)
            if arg in globals():
                globals()[arg]()  # assume function and execute
            else:
                eval(arg)  # assume Python code and execute
