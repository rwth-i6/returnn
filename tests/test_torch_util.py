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

    # Note: I tried prof.events(), prof.profiler.kineto_results.events(), prof._memory_profile().timeline,
    # but they all are not really giving me the information I want.
    # Either the Python stack is missing, or the memory information is incomplete,
    # or the Python/TorchOp events are missing.
    # The only complete information source seems to be prof.profiler.kineto_results.experimental_event_tree().

    from torch.profiler._utils import traverse_dfs
    from torch._C._profiler import _EventType  # noqa

    for ev in sorted(
        traverse_dfs(prof.profiler.kineto_results.experimental_event_tree()), key=lambda ev: ev.start_time_ns
    ):
        # ev: torch._C._profiler._ProfilerEvent
        # print(ev.start_time_ns, ev.id, ev.name, ev.typed)
        if ev.typed[0] == _EventType.Allocation:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_Allocation
            print("alloc", ex.allocation_id, ex.alloc_size, ex.id, ex.ptr, ex.total_allocated)
        elif ev.typed[0] == _EventType.TorchOp:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_TorchOp
            print("torchop", ex.name)
        elif ev.typed[0] == _EventType.PyCall:
            ex = ev.typed[1]  # torch._C._profiler._ExtraFields_PyCall
            ex0 = ex.caller  # torch._C._profiler._PyFrameState
            ex1 = ex.callsite  # torch._C._profiler._PyFrameState
            if os.path.basename(ex1.file_name) == os.path.basename(__file__) or os.path.basename(
                ex0.file_name
            ) == os.path.basename(__file__):
                print("pycall", ex0.function_name, ex0.file_name, ":", ex0.line_number, "->", ex1.function_name)

    # TODO how to understand memory? i want to know: where is what allocated? and when deallocated? total consumption?
    # TODO...
    # TODO now model = _Model(use_grad_ckpt=True) ...
    # TODO also use sys.settrace to really check the flow of logic


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
