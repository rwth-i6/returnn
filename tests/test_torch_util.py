"""
Test :mod:`returnn.torch.util`.
"""

from __future__ import annotations

import _setup_test_env  # noqa

import os
import sys
import unittest
import torch

from torch_utils import report_profile

from returnn.util import better_exchook


@unittest.skipIf(torch.__version__ < (2,), "gradient_checkpoint_scope needs PyTorch >= 2.0")
def test_gradient_checkpoint_scope():
    # https://github.com/rwth-i6/returnn/issues/1552
    from copy import deepcopy
    from torch.profiler import profile, record_function, ProfilerActivity
    from returnn.torch.util.gradient_checkpoint import gradient_checkpoint_scope

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
    report_profile(
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
    report_profile(
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


@unittest.skipIf(torch.__version__ < (2,), "gradient_checkpoint_scope needs PyTorch >= 2.0")
def test_gradient_checkpoint_scope_twice():
    # https://github.com/rwth-i6/returnn/issues/1579

    from returnn.torch.util.gradient_checkpoint import gradient_checkpoint_scope

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


@unittest.skipIf(torch.__version__ < (2,), "gradient_checkpoint_scope needs PyTorch >= 2.0")
def test_saved_tensors_hooks_gc_segfault():
    # https://github.com/rwth-i6/returnn/issues/1581
    # https://github.com/pytorch/pytorch/issues/130734

    # noinspection PyProtectedMember
    from returnn.torch.util.gradient_checkpoint import _can_exit_saved_tensors_hooks_inside_hooks

    if not _can_exit_saved_tensors_hooks_inside_hooks():
        raise unittest.SkipTest("Not yet fixed.")

    shape = (101, 103)
    for i in range(10):
        print("**** iter", i)
        v = torch.nn.Parameter(torch.randn(shape))

        class _Handler:
            def __init__(self):
                self.scope = torch.autograd.graph.saved_tensors_hooks(self._pack_hook, self._unpack_hook)
                self.scope.__enter__()
                self.exited = False

            def _pack_hook(self, x):
                print(f"*** _pack_hook {self}")
                return x

            def _unpack_hook(self, x):
                print(f"*** _unpack_hook {self}")
                if not self.exited:
                    self.exited = True
                    print(
                        f"*** exit {self.scope},"
                        f" pack_hook {hex(id(self.scope.pack_hook))},"
                        f" unpack_hook {hex(id(self.scope.unpack_hook))}"
                    )
                    self.scope.__exit__()
                return x

        with torch.autograd.graph.saved_tensors_hooks(lambda x: x, lambda x: x):
            handler = _Handler()  # keep ref...  # noqa
            x = v * torch.randn(shape)
            x.sum().backward()


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
