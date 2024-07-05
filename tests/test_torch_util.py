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
        def _get_var_noise() -> torch.Tensor:
            return torch.randn(shape)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if not self.use_grad_ckpt:
                return (self.var + self._get_var_noise()) * x

            with gradient_checkpoint_scope():
                v_ = self.var + self._get_var_noise()
            return v_ * x

        def demo_run(self):
            x = self.input_var
            y = self(x)
            loss = y.sum()  # dummy...
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()

    model = _Model(use_grad_ckpt=False)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, with_stack=True, record_shapes=True) as prof:
        with record_function("train_step_no_grad_ckpt"):
            model.demo_run()
    _report_profile(prof)
    # TODO... check?

    print("**** now with grad chkpt ****")
    model = _Model(use_grad_ckpt=True)
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, with_stack=True, record_shapes=True) as prof:
        with record_function("train_step_grad_ckpt"):
            model.demo_run()
    _report_profile(prof)
    # TODO check...


def _report_profile(prof: torch.profiler.profiler):
    # Note: I tried prof.events(), prof.profiler.kineto_results.events(), prof._memory_profile().timeline,
    # but they all are not really giving me the information I want.
    # Either the Python stack is missing, or the memory information is incomplete,
    # or the Python/TorchOp events are missing.
    # The only complete information source seems to be prof.profiler.kineto_results.experimental_event_tree().

    # TODO how to understand memory? i want to know: where is what allocated? and when deallocated? total consumption?
    # TODO also use sys.settrace to really check the flow of logic; not really needed anymore, have the trace already

    # noinspection PyProtectedMember
    from torch.profiler._utils import traverse_dfs
    from torch._C._profiler import _EventType  # noqa

    def _ev_repr(ev) -> str:
        if not ev:
            return "None"
        # ev: torch._C._profiler._ProfilerEvent
        if ev.typed[0] == _EventType.Allocation:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_Allocation
            # ex.allocation_id/ex.ptr redundant with ex.id?
            return (
                f"alloc id={ex.id} size={ex.alloc_size} total_alloc={ex.total_allocated}"
                f" parent={_ev_repr(ev.parent)}"
            )
        elif ev.typed[0] == _EventType.TorchOp:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_TorchOp
            return f"torchop {ex.name} parent={_ev_repr(ev.parent)}"
        elif ev.typed[0] == _EventType.PyCall:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_PyCall
            ex0 = ex.caller  # torch._C._profiler._PyFrameState
            ex1 = ex.callsite  # torch._C._profiler._PyFrameState
            if _pycall_filter_fn(ex0.file_name) or _pycall_filter_fn(ex1.file_name):
                return (
                    f"pycall {ex0.file_name}:{ex0.line_number} {ex0.function_name} -> {ex1.function_name},"
                    f" parent={_ev_repr(ev.parent)}"
                )
        return "Other"

    for ev_ in sorted(
        traverse_dfs(prof.profiler.kineto_results.experimental_event_tree()), key=lambda ev: ev.start_time_ns
    ):
        # ev: torch._C._profiler._ProfilerEvent
        s = _ev_repr(ev_)
        if s != "Other":
            print(s)


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
