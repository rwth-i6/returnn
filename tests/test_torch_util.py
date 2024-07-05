"""
Test :mod:`returnn.torch.util`.
"""

from __future__ import annotations
import _setup_test_env  # noqa

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
            ev.cpu_memory_usage,
            [c.id for c in ev.cpu_children],
            ev.cpu_parent.id if ev.cpu_parent else None,
            ev.stack,
        )
    print(prof.profiler)
    print(prof.profiler.kineto_results)
    tot_mem_usage = 0
    for ev in prof.profiler.kineto_results.events():
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
