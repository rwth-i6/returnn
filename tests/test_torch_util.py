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
    print(prof.events())
    for ev in prof.events():
        ev: torch.autograd.profiler.FunctionEvent
        # print(ev)
        print(
            ev.name,
            ev.id,
            ev.cpu_memory_usage,  # unclear what this is exactly. how much is freed? what is current total usage?
            [c.id for c in ev.cpu_children],
            ev.cpu_parent.id if ev.cpu_parent else None,
            ev.stack,  # why empty?
        )
    print(prof.profiler)
    print(prof.profiler.kineto_results)

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

    tot_mem_usage = 0
    for ev in prof.profiler.kineto_results.events():
        # ev: torch._C._autograd._KinetoEvent
        tot_mem_usage += ev.nbytes()
        print(
            ev.name(),
            ev.nbytes(),
            ev.stack(),
            ev.scope(),
            ev.sequence_nr(),
            ev.correlation_id(),
            tot_mem_usage,
        )
    """
    train_step_no_grad_ckpt 0 [] 7 -1 1 0
    aten::randn 0 [] 0 -1 2 0
    aten::empty 0 [] 0 -1 3 0
    [memory] 41612 [] 0 -1 0 41612    # new (noise): empty
    aten::normal_ 0 [] 0 -1 4 41612
    aten::add 0 [] 0 0 5 41612
    [memory] 41612 [] 0 -1 0 83224    # new (v_): add
    [memory] -41612 [] 0 -1 0 41612   # del (noise)
    aten::mul 0 [] 0 1 6 41612
    [memory] 41612 [] 0 -1 0 83224    # new (v_ * x): mul
    aten::sum 0 [] 0 2 7 83224
    aten::sum 0 [] 0 -1 8 83224
    [memory] 4 [] 0 -1 0 83228        # new (loss): sum
    aten::as_strided 0 [] 0 -1 9 83228
    aten::fill_ 0 [] 0 -1 10 83228
    aten::ones_like 0 [] 0 -1 11 83228
    aten::empty_like 0 [] 0 -1 12 83228
    aten::empty_strided 0 [] 0 -1 13 83228
    [memory] 4 [] 0 -1 0 83232
    aten::fill_ 0 [] 0 -1 14 83232
    autograd::engine::evaluate_function: SumBackward0 0 [] 0 2 15 83232
    SumBackward0 0 [] 1 2 16 83232
    aten::expand 0 [] 0 -1 17 83232
    aten::as_strided 0 [] 0 -1 18 83232
    autograd::engine::evaluate_function: MulBackward0 0 [] 0 1 19 83232
    MulBackward0 0 [] 1 1 20 83232
    aten::mul 0 [] 0 -1 21 83232
    [memory] 41612 [] 0 -1 0 124844
    aten::mul 0 [] 0 -1 22 124844
    [memory] 41612 [] 0 -1 0 166456
    [memory] -41612 [] 0 -1 0 124844
    autograd::engine::evaluate_function: torch::autograd::AccumulateGrad 0 [] 0 -1 23 124844
    torch::autograd::AccumulateGrad 0 [] 1 -1 24 124844
    aten::detach 0 [] 0 -1 25 124844
    detach 0 [] 0 -1 26 124844
    autograd::engine::evaluate_function: AddBackward0 0 [] 0 0 27 124844
    AddBackward0 0 [] 1 0 28 124844
    autograd::engine::evaluate_function: torch::autograd::AccumulateGrad 0 [] 0 -1 29 124844
    torch::autograd::AccumulateGrad 0 [] 1 -1 30 124844
    aten::detach 0 [] 0 -1 31 124844
    detach 0 [] 0 -1 32 124844
    [memory] -4 [] 0 -1 0 124840
    Optimizer.step#SGD.step 0 [] 7 -1 33 124840
    aten::add_ 0 [] 0 -1 34 124840
    aten::add_ 0 [] 0 -1 35 124840
    Optimizer.zero_grad#SGD.zero_grad 0 [] 7 -1 36 124840
    [memory] -41612 [] 0 -1 0 83228
    [memory] -41612 [] 0 -1 0 41616
    [memory] -41612 [] 0 -1 0 4
    [memory] -4 [] 0 -1 0 0
    """
    mem_prof = prof._memory_profile()
    for time, action, (tensor_key, version), size in mem_prof.timeline:
        print(f"{time=} {action=} {tensor_key=} {version=} {size=}")
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
