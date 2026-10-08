"""
Test :mod:`returnn.torch.util`.
"""

from __future__ import annotations

import _setup_test_env  # noqa

from typing import Optional, Tuple
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


def test_debug_inf_nan():
    param = torch.nn.Parameter(torch.tensor([0.5349, 0.8094, -100, 0, -0.9890, 1, 1.3221, 0.8172, -0.7658, -0.7506]))

    def func():
        x = torch.tensor([0.5349, 0, 1.1103, -1.6898, -0.9890, 1, 1.3221, 0.8172, -0.7658, -0.7506])
        x = mod1(x)
        x = mod2(x)
        x = mod3(x) * param
        x = mod4(x)
        x = mod1(x)
        x = mod2(x)
        x = mod2(x)
        x = mod5(x)
        return x.sum()

    def mod1(x: torch.Tensor) -> torch.Tensor:
        return x * 2

    def mod2(x: torch.Tensor) -> torch.Tensor:
        return x.exp()

    def mod3(x: torch.Tensor) -> torch.Tensor:
        return x - 2

    def mod4(x: torch.Tensor) -> torch.Tensor:
        x.subtract_(-3.5)
        return x

    def mod5(x: torch.Tensor) -> torch.Tensor:
        return x / x

    x = func()
    print(x)
    print("inf/nan:", torch.isinf(x).any().item(), torch.isnan(x).any().item())

    from returnn.torch.util.debug_inf_nan import debug_inf_nan

    # Run directly, to just test that it goes through without exception.
    # For some reason, the detect_anomaly does not print the forward op?
    debug_inf_nan(func, with_grad=True, stop_reporting_after_first_inf_nan=False)

    from io import StringIO

    out = StringIO()
    debug_inf_nan(func, file=out, stop_reporting_after_first_inf_nan=False)
    assert "inf in aten.exp" in out.getvalue()
    assert "nan in aten.div" in out.getvalue()
    assert "mod5" in out.getvalue()
    assert os.path.basename(__file__) in out.getvalue()


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


def test_smoothed_ce_bwd_inductor_pattern():
    """
    The Inductor rewrite of the grad-level label-smoothed sparse-CE backward
    (:func:`returnn.torch.util.graph_capture._register_smoothed_ce_bwd_pattern`):
    numerics vs eager, and the pattern must FIRE on the canonical RF chain --
    it matches the plain aten emission, so this guards against
    decomposition drift across torch upgrades.
    """
    import tempfile

    if not torch.cuda.is_available():
        raise unittest.SkipTest("need CUDA (the rewrite targets the compiled GPU step)")
    # fresh cache: stale generated modules would mask whether the pattern fired
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="ce-pattern-test-")

    import returnn.frontend as rf
    from returnn.tensor import Dim, Tensor
    from returnn.torch.util import graph_capture

    rf.select_backend_torch()
    graph_capture._register_smoothed_ce_bwd_pattern()
    frames, classes = 700, 133
    time_dim = Dim(frames, name="time")
    vocab = Dim(classes, name="vocab")

    def step(logits_r, targets_r):
        logits = Tensor("logits", [time_dim, vocab], dtype="float32", raw_tensor=logits_r)
        targets = Tensor("targets", [time_dim], dtype="int32", sparse_dim=vocab, raw_tensor=targets_r)
        lp = rf.log_softmax(logits, axis=vocab)
        lp = rf.label_smoothed_log_prob_gradient(lp, 0.1, axis=vocab)
        ce = rf.cross_entropy(estimated=lp, target=targets, axis=vocab, estimated_type="log-probs")
        return rf.reduce_sum(ce, axis=ce.dims).raw_tensor

    logits_raw = torch.randn(frames, classes, device="cuda", requires_grad=True)
    targets_raw = torch.randint(0, classes, (frames,), dtype=torch.int32, device="cuda")
    loss_e = step(logits_raw, targets_raw)
    (g_e,) = torch.autograd.grad(loss_e, logits_raw)

    from functorch.compile import aot_function

    # noinspection PyProtectedMember
    from torch._inductor.compile_fx import compile_fx

    count_before = graph_capture._smoothed_ce_bwd_match_count
    cstep = aot_function(step, fw_compiler=compile_fx)  # like graph_capture (dynamo cannot trace RF)
    loss_c = cstep(logits_raw, targets_raw)
    (g_c,) = torch.autograd.grad(loss_c, logits_raw)
    assert torch.allclose(loss_e, loss_c, atol=1e-4)
    assert torch.allclose(g_e, g_c, atol=1e-5)
    assert graph_capture._smoothed_ce_bwd_match_count > count_before, "CE bwd pattern did not fire"


@unittest.skipIf(torch.__version__ < (2, 5), "compile_fx under aot_function: torch 2.0 segfaults, 2.5 works")
def test_inductor_fw_compiler_backends():
    """
    :func:`returnn.torch.util.graph_capture.inductor_fw_compiler` as the compiled train step uses it
    (aot_function, one inference-style graph), CPU:
    with Inductor's compile_fx, and with the eager ``nop`` of torch_cuda_graph opts "debug_aot_eager",
    whose already boxed result must not get the torch >= 2.11 compile_fx call shim.
    """
    from functorch.compile import aot_function, nop
    from returnn.torch.util.graph_capture import inductor_fw_compiler

    def _f(x, y):
        return torch.sin(x) * y + 1.0

    x, y = torch.randn(5), torch.randn(5)
    for backend in (None, nop):
        f_compiled = aot_function(_f, fw_compiler=inductor_fw_compiler(backend))
        torch.testing.assert_close(f_compiled(x, y), _f(x, y))


def test_all_reduce_sum_traces():
    """
    the differentiable all-reduce (the synchronized BatchNorm statistics go through it) traces under AOT autograd:
    the compiled step of torch_cuda_graph runs on fake tensors, which a direct collective call cannot take
    """
    import torch.distributed as dist
    from functorch.compile import aot_function, nop
    from returnn.torch.util.distributed import all_reduce_sum

    own_group = not dist.is_initialized()
    if own_group:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:

        def _loss(x_):
            return all_reduce_sum(x_ * 2.0).square().sum()

        x = torch.randn(5, requires_grad=True)
        (ref,) = torch.autograd.grad(_loss(x), x)
        traced = aot_function(_loss, fw_compiler=nop, bw_compiler=nop)
        (grad,) = torch.autograd.grad(traced(x), x)
        torch.testing.assert_close(grad, ref)
    finally:
        if own_group:
            dist.destroy_process_group()


def test_all_reduce_sum_eager_takes_the_custom_op():
    """
    on the default group an eager call goes through the same custom op as a traced step,
    forward and backward, so the tensor type does not decide between two implementations
    """
    import torch.distributed as dist
    from torch.utils._python_dispatch import TorchDispatchMode
    from returnn.torch.util.distributed import all_reduce_sum

    if not hasattr(torch.library, "custom_op"):
        raise unittest.SkipTest("torch without torch.library.custom_op")

    class _OpNames(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.names = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.names.append(str(func))
            return func(*args, **(kwargs or {}))

    own_group = not dist.is_initialized()
    if own_group:
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        x = torch.randn(5, requires_grad=True)
        with _OpNames() as ops:
            out = all_reduce_sum(x)
            (grad,) = torch.autograd.grad(out, x, grad_outputs=torch.ones(5))
        assert ops.names.count("returnn.all_reduce_sum.default") == 2, ops.names
        torch.testing.assert_close(out, x)
        torch.testing.assert_close(grad, torch.ones(5))
    finally:
        if own_group:
            dist.destroy_process_group()


def test_custom_op_string_annotations():
    """
    this module has ``from __future__ import annotations``, so the op signature below has string annotations,
    from which torch < 2.7 cannot infer the schema by itself
    """
    from returnn.torch.util.custom_op import custom_op

    if not hasattr(torch.library, "custom_op"):
        raise unittest.SkipTest("torch without torch.library.custom_op")

    @custom_op("returnn_test::scaled_sum_and_diff", mutates_args=())
    def _op(x: torch.Tensor, y: Optional[torch.Tensor], scale: float, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        y_ = y if y is not None else torch.zeros_like(x)
        return (x + y_) * scale * n, x - y_

    x, y = torch.tensor([1.0, 2.0]), torch.tensor([0.5, -1.0])
    out_sum, out_diff = _op(x, y, 2.0, 3)
    torch.testing.assert_close(out_sum, torch.tensor([9.0, 6.0]))
    torch.testing.assert_close(out_diff, torch.tensor([0.5, 3.0]))
    out_sum, out_diff = torch.ops.returnn_test.scaled_sum_and_diff(x, None, 1.0, 1)
    torch.testing.assert_close(out_sum, x)
    torch.testing.assert_close(out_diff, x)


def test_masked_select_bound():
    from returnn.torch.util.array_ import masked_select_bound

    generator = torch.Generator().manual_seed(7)
    x = torch.randn(3, 5, 4, generator=generator)
    mask = torch.rand(3, 5, generator=generator) > 0.5
    out, out_len = masked_select_bound(x, mask)
    num = int(mask.sum())
    assert int(out_len) == num
    assert out.shape == (15, 4)
    torch.testing.assert_close(out[:num], x[mask])
    assert (out[num:] == 0).all()
    # a tighter declared bound shrinks the output buffer
    out2, out_len2 = masked_select_bound(x, mask, bound=num)
    assert int(out_len2) == num
    assert out2.shape == (num, 4)
    torch.testing.assert_close(out2, x[mask])


def test_gpu_cpu_affinity_parse_cpulist():
    from returnn.torch.util.gpu_cpu_affinity import parse_cpulist

    assert parse_cpulist("0-11,24-35\n") == set(range(12)) | set(range(24, 36))
    assert parse_cpulist("72-143") == set(range(72, 144))
    assert parse_cpulist("3") == {3}
    assert parse_cpulist("") == set()


def test_gpu_cpu_affinity_sysfs_and_proc_lookup():
    """the local CPUs of a PCI device from sysfs, and the PCI id of a GPU by its UUID from the driver's proc dir"""
    import tempfile
    from returnn.torch.util.gpu_cpu_affinity import read_pci_device_local_cpus, find_pci_id_by_gpu_uuid

    with tempfile.TemporaryDirectory() as tmp:
        gpus = {
            "0000:1b:00.0": ("8ff8d0c7-8d30-8e55-0980-ac69fc03a6b8", "0-11"),
            "0001:01:00.0": ("0911978f", "72-143"),
        }
        for pci_id, (uuid, cpulist) in gpus.items():
            os.makedirs(os.path.join(tmp, "sys", pci_id))
            with open(os.path.join(tmp, "sys", pci_id, "local_cpulist"), "wt") as f:
                f.write(cpulist + "\n")
            os.makedirs(os.path.join(tmp, "proc", pci_id))
            with open(os.path.join(tmp, "proc", pci_id, "information"), "wt") as f:
                f.write(f"Model: \t\t NVIDIA H100\nGPU UUID: \t GPU-{uuid}\nBus Location: \t {pci_id}\n")
        assert read_pci_device_local_cpus("0001:01:00.0", sysfs_root=os.path.join(tmp, "sys")) == set(range(72, 144))
        proc = os.path.join(tmp, "proc")
        assert find_pci_id_by_gpu_uuid("8ff8d0c7-8d30-8e55-0980-ac69fc03a6b8", proc_root=proc) == "0000:1b:00.0"
        assert find_pci_id_by_gpu_uuid("GPU-0911978f", proc_root=proc) == "0001:01:00.0"
        try:
            find_pci_id_by_gpu_uuid("unknown", proc_root=proc)
        except RuntimeError as exc:
            assert "not found" in str(exc)
        else:
            raise AssertionError("unknown UUID must raise")


def test_gpu_cpu_affinity_select():
    """
    NUMA node when it covers the rank's share, else the socket, else nothing (unaligned cpuset);
    ranks sharing the set get disjoint slices when those still cover the share
    """
    from returnn.torch.util.gpu_cpu_affinity import select_gpu_local_cpus

    # GH200: 4 Grace CPUs of 72, one per GPU
    gh200 = [set(range(72 * i, 72 * (i + 1))) for i in range(4)]
    gh200_socket = {cpu: cpu // 72 for cpu in range(288)}.__getitem__
    for rank in range(4):
        assert select_gpu_local_cpus(gh200, rank, set(range(288)), cpu_socket=gh200_socket) == gh200[rank]

    # 2 sockets of 48 CPUs, 4 NUMA nodes of 12 each (NPS4), 2 GPUs per socket on nodes 0, 2, 4, 6
    socket = {cpu: cpu // 48 for cpu in range(96)}.__getitem__
    nodes = [set(range(12 * i, 12 * (i + 1))) for i in (0, 2, 4, 6)]
    all_cpus = set(range(96))
    # 4 ranks: share 24, a node has 12 -> socket (48), shared by 2 ranks -> slices of 24
    assert select_gpu_local_cpus(nodes, 0, all_cpus, cpu_socket=socket) == set(range(24))
    assert select_gpu_local_cpus(nodes, 1, all_cpus, cpu_socket=socket) == set(range(24, 48))
    assert select_gpu_local_cpus(nodes, 3, all_cpus, cpu_socket=socket) == set(range(72, 96))
    # 2 ranks on the same socket, whole node allowed: share 48, socket 0 is exactly that -> shared, no slices
    assert select_gpu_local_cpus(nodes[:2], 1, all_cpus, cpu_socket=socket) == set(range(48))
    # 8 ranks (2 per node): share 12, the node covers it, shared by 2 -> not sliceable (6 < 12), shared
    nodes8 = [n for n in nodes for _ in range(2)]
    assert select_gpu_local_cpus(nodes8, 3, all_cpus, cpu_socket=socket) == nodes[1]
    # a 24-CPU cpuset on the far socket: nothing local, nothing to pin
    assert select_gpu_local_cpus(nodes[:1], 0, set(range(48, 72)), cpu_socket=socket) is None
    # a 24-CPU cpuset half on each socket, 1 rank: share 24, the socket 0 part has 12 -> none
    assert select_gpu_local_cpus(nodes[:1], 0, set(range(36, 60)), cpu_socket=socket) is None
    # fewer allowed CPUs than ranks: the share is still one CPU, so never an empty set
    assert select_gpu_local_cpus(nodes, 0, {50, 51}, cpu_socket=socket) is None
    assert select_gpu_local_cpus(nodes, 2, {50, 51}, cpu_socket=socket) == {50, 51}
    # same cpuset, 2 ranks on different sockets: share 12, each socket part is exactly that
    assert select_gpu_local_cpus([nodes[0], nodes[2]], 0, set(range(36, 60)), cpu_socket=socket) == set(range(36, 48))
    assert select_gpu_local_cpus([nodes[0], nodes[2]], 1, set(range(36, 60)), cpu_socket=socket) == set(range(48, 60))


def test_gpu_cpu_affinity_from_config():
    """
    the startup glue: option off or a CPU device pins nothing;
    a single GPU pins its device index; distributed pins the local rank, sharing with the ranks on this host (here 1)
    """
    import socket
    from unittest import mock
    from returnn.config import Config
    import returnn.torch.distributed as dist_mod
    from returnn.torch.util.gpu_cpu_affinity import set_gpu_local_cpu_affinity_from_config

    assert "PT_DEVICE" not in os.environ
    with mock.patch("torch.cuda.is_available", return_value=True), mock.patch(
        "torch.cuda.current_device", return_value=0
    ), mock.patch("returnn.torch.util.gpu_cpu_affinity.set_gpu_local_cpu_affinity", return_value={0}) as set_affinity:
        set_gpu_local_cpu_affinity_from_config(Config({"device": "cuda", "gpu_local_cpu_affinity": False}))
        set_gpu_local_cpu_affinity_from_config(Config({"device": "cpu"}))
        set_affinity.assert_not_called()
        set_gpu_local_cpu_affinity_from_config(Config({"device": "cuda:1"}))
        set_affinity.assert_called_once_with(1, num_local_ranks=1)
        set_affinity.reset_mock()
        set_gpu_local_cpu_affinity_from_config(Config({"device": "gpu"}))  # bare cuda = the current device
        set_affinity.assert_called_once_with(0, num_local_ranks=1)
        set_affinity.reset_mock()

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        env = dict(
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(port),
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="1",
            LOCAL_WORLD_SIZE="2",
        )
        init_info_key = "_RETURNN_TORCH_DISTRIBUTED_INIT_INFO"
        assert init_info_key not in os.environ and not dist_mod._is_set_up
        try:
            with mock.patch.dict(os.environ, env):
                set_gpu_local_cpu_affinity_from_config(
                    Config({"device": "cuda", "torch_distributed": {"backend": "gloo"}})
                )
            set_affinity.assert_called_once_with(1, num_local_ranks=1)
        finally:
            os.environ.pop(init_info_key, None)
            dist_mod._is_set_up, dist_mod._ctx = False, None
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()


def test_gpu_cpu_affinity_set():
    """the real thing on a GPU node: the affinity shrinks to CPUs local to the device, within the allowed set"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    from returnn.torch.util.gpu_cpu_affinity import set_gpu_local_cpu_affinity, get_gpu_pci_id

    allowed = os.sched_getaffinity(0)
    try:
        pci_id = get_gpu_pci_id(0)
        assert os.path.isdir(f"/sys/bus/pci/devices/{pci_id}"), pci_id
        # as many ranks as visible GPUs, like a full-node job.
        # None only when even the socket is below the share (e.g. one visible GPU of a node, all CPUs allowed)
        n = torch.cuda.device_count()
        cpus = set_gpu_local_cpu_affinity(0, num_local_ranks=n)
        assert cpus is None or (cpus and cpus <= allowed)
        assert os.sched_getaffinity(0) == (cpus if cpus is not None else allowed)
        # the other ranks' sets are disjoint from this one or identical to it (shared), never overlapping otherwise
        for tid in os.listdir("/proc/self/task"):
            os.sched_setaffinity(int(tid), allowed)
        for rank in range(1, n):
            other = set_gpu_local_cpu_affinity(rank, num_local_ranks=n)
            assert other and cpus and (other == cpus or not (other & cpus)), (rank, cpus, other)
            for tid in os.listdir("/proc/self/task"):
                os.sched_setaffinity(int(tid), allowed)
        # the whole node for one rank: nothing local can cover that, so nothing is pinned
        for tid in os.listdir("/proc/self/task"):
            os.sched_setaffinity(int(tid), allowed)
        if cpus != allowed:
            assert set_gpu_local_cpu_affinity(0, num_local_ranks=1) is None
            assert os.sched_getaffinity(0) == allowed
        # more ranks than devices: nothing is pinned, no error
        assert set_gpu_local_cpu_affinity(0, num_local_ranks=n + 1) is None
        assert os.sched_getaffinity(0) == allowed
    finally:
        for tid in os.listdir("/proc/self/task"):
            os.sched_setaffinity(int(tid), allowed)


def test_depthwise_conv1d_triton_kernel_grad():
    """fwd and all grads of the Triton depthwise conv vs torch conv1d, small blocks force partial tiles"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    try:
        from returnn.torch.util import depthwise_conv_triton as m
    except ImportError as exc:
        raise unittest.SkipTest(f"triton not available ({exc})")

    dev = "cuda"
    gen = torch.Generator(device="cpu").manual_seed(11)
    f32, bf16 = torch.float32, torch.bfloat16
    small, small_blocks = (3, 37, 70), (16, 32, 16, 32)
    cases = [
        (small, 5, 2, 2, f32, f32, small_blocks),
        (small, 32, 15, 16, f32, f32, small_blocks),
        (small, 4, 0, 0, f32, f32, small_blocks),
        (small, 7, 4, 4, f32, f32, small_blocks),
        (small, 32, 15, 16, bf16, f32, None),
        ((920, 24, 1024), 32, 15, 16, bf16, bf16, None),
        ((3, 20, 70), 32, 15, 16, f32, f32, small_blocks),
        ((3, 20, 70), 32, 0, 16, f32, f32, small_blocks),
        ((920, 24, 1024), 32, 0, 16, bf16, bf16, None),
    ]
    for shape, width, pad_l, pad_r, x_dtype, w_dtype, blocks in cases:
        n_batch, n_time, n_chan = shape
        x = torch.randn(shape, generator=gen).to(dev, x_dtype).requires_grad_(True)
        w = (torch.randn(n_chan, width, generator=gen) * 0.3).to(dev, w_dtype).requires_grad_(True)
        bias = torch.randn(2 * n_chan, generator=gen).to(dev)[::2].requires_grad_(True)
        n_time_out = n_time + pad_l + pad_r - width + 1
        opts = {"blocks": blocks} if blocks else {}
        out = m.depthwise_conv1d(x, w, bias, pad_l=pad_l, n_time_out=n_time_out, **opts)
        d_out = torch.randn(n_batch, n_time_out, n_chan, generator=gen).to(dev, x_dtype)
        out.backward(d_out)
        grads = [t.grad.clone() for t in (x, w, bias)]
        for t in (x, w, bias):
            t.grad = None
        x_ref = torch.nn.functional.pad(x.float().transpose(1, 2), (pad_l, pad_r))
        ref = torch.nn.functional.conv1d(x_ref, w.float()[:, None, :], bias, groups=n_chan).transpose(1, 2)
        tight = {"rtol": 1e-4, "atol": 1e-4}
        tol = tight if x_dtype == f32 else {"rtol": 2e-2, "atol": 2e-2}
        assert out.shape == ref.shape and out.dtype == x_dtype, (width, pad_l, out.shape, out.dtype)
        torch.testing.assert_close(out.float(), ref, **tol)
        ref.backward(d_out.float())
        for g, t, t_tol in zip(grads, (x, w, bias), (tol, tol, tight)):
            assert g.dtype == t.dtype, (width, g.dtype, t.dtype)
            torch.testing.assert_close(g.float(), t.grad.float(), **t_tol)


def test_depthwise_conv1d_triton_guards():
    """a frozen filter skips the weight gradient kernel, second derivatives and invalid blocks or dtypes raise"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    try:
        from returnn.torch.util import depthwise_conv_triton as m
    except ImportError as exc:
        raise unittest.SkipTest(f"triton not available ({exc})")
    from unittest import mock

    x = torch.randn(2, 9, 8, device="cuda", requires_grad=True)
    w = torch.randn(8, 3, device="cuda")
    with mock.patch.object(m.kernels, "dw_bwd_dw") as dw_kernel:
        m.depthwise_conv1d(x, w, None, pad_l=1, n_time_out=9).sum().backward()
    assert not dw_kernel.mock_calls and x.grad is not None, dw_kernel.mock_calls
    w.requires_grad_(True)
    out = m.depthwise_conv1d(x, w, None, pad_l=1, n_time_out=9)
    (gx,) = torch.autograd.grad(out.square().sum(), x, create_graph=True)
    try:
        (gx.square().sum() + w.sum()).backward()
    except RuntimeError as exc:
        assert "once_differentiable" in str(exc), exc
    else:
        raise AssertionError("a second derivative through the conv must raise")
    for args, opts in (((x.double(), w.double()), {}), ((x, w), {"blocks": (0, 32, 16, 32)})):
        try:
            m.depthwise_conv1d(*args, None, pad_l=1, n_time_out=9, **opts)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"dtype {args[0].dtype} with {opts} must raise")


def test_depthwise_conv1d_triton_short_window_row_kernels():
    """a window no longer than the filter runs the row loops for the forward and the input gradient, not the tap loops"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    try:
        from returnn.torch.util import depthwise_conv_triton as m
    except ImportError as exc:
        raise unittest.SkipTest(f"triton not available ({exc})")
    from unittest import mock

    x = torch.randn(4, 24, 64, device="cuda", requires_grad=True)
    w = torch.randn(64, 32, device="cuda")
    with mock.patch.object(m.kernels, "dw_fwd") as fwd_taps, mock.patch.object(m.kernels, "dw_bwd_dx") as dx_taps:
        m.depthwise_conv1d(x, w, None, pad_l=0, n_time_out=9).sum().backward()
    assert not fwd_taps.mock_calls and not dx_taps.mock_calls, (fwd_taps.mock_calls, dx_taps.mock_calls)
    assert x.grad is not None


def test_depthwise_conv1d_triton_weight_grad_scratch_independent_of_rows():
    """the weight gradient sums over the rows inside each program, so its f32 scratch does not grow with the rows"""
    if not torch.cuda.is_available():
        raise unittest.SkipTest("needs CUDA")
    try:
        from returnn.torch.util import depthwise_conv_triton as m
    except ImportError as exc:
        raise unittest.SkipTest(f"triton not available ({exc})")

    import gc

    blocks = (m.kernels.BLOCK_R_CONV, m.kernels.BLOCK_C_CONV, m.kernels.BLOCK_R_DW, m.kernels.BLOCK_C_DW)
    peaks = []
    for n_batch in (500, 2000):
        x = torch.randn(n_batch, 24, 256, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(256, 32, device="cuda", dtype=torch.bfloat16)
        d_out = torch.randn(n_batch, 24, 256, device="cuda", dtype=torch.bfloat16)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_stats()["requested_bytes.all.current"]
        m._launch_bwd(x, w, d_out, has_bias=True, pad_l=15, blocks=blocks, need_dx=False, need_dw_db=True)
        torch.cuda.synchronize()
        peaks.append(torch.cuda.memory_stats()["requested_bytes.all.peak"] - base)
    assert peaks[0] == peaks[1], peaks


def test_ctc_fsa_cache_scoped_to_static_traceable_step():
    """
    The FSA cache serves the aux heads within one step, but never across static traceable steps:
    e.g. the warm run before the CUDA graph capture runs on the same targets buffer at the same version,
    and its FSA must not enter the graph.
    """
    import returnn.frontend as rf
    from returnn.torch.util import native_op

    targets = torch.tensor([[1, 2, 2, 3, 0], [2, 3, 0, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([4, 2], dtype=torch.int32)
    kwargs = dict(targets=targets, seq_lens=seq_lens, blank_idx=4)
    eager = native_op.get_ctc_fsa_fast_bw(**kwargs)
    assert native_op.get_ctc_fsa_fast_bw(**kwargs)[0] is eager[0], "eager: the heads share the FSA"
    steps = []
    for _ in range(2):  # e.g. the warm run, then the capture
        with rf.set_static_traceable_ctx():
            fsa = native_op.get_ctc_fsa_fast_bw(**kwargs)
            assert native_op.get_ctc_fsa_fast_bw(**kwargs)[0] is fsa[0], "within one step, the heads share the FSA"
        steps.append(fsa)
    assert steps[0][0] is not eager[0], "no eager FSA enters a step"
    assert steps[1][0] is not steps[0][0], "no FSA crosses steps"
    assert native_op.get_ctc_fsa_fast_bw(**kwargs)[0] is not steps[1][0], "no FSA leaks out of a step"


def test_ctc_fsa_cache_traced_step():
    """
    Like the compiled step of torch_cuda_graph: traced under static traceable,
    with the real targets buffer under an active fake mode.
    The heads share the trace-time FSA, and it never meets an eager FSA in either direction.
    """
    if not hasattr(torch.library, "register_fake"):
        raise unittest.SkipTest("torch.library.register_fake not available (torch < 2.4)")
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
    import returnn.frontend as rf
    from returnn.torch.util import native_op

    targets = torch.tensor([[1, 2, 2, 3, 0], [2, 3, 0, 0, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([4, 2], dtype=torch.int32)
    kwargs = dict(targets=targets, seq_lens=seq_lens, blank_idx=4)
    eager = native_op.get_ctc_fsa_fast_bw(**kwargs)
    with rf.set_static_traceable_ctx(), FakeTensorMode(allow_non_fake_inputs=True):
        traced = native_op.get_ctc_fsa_fast_bw(**kwargs)
        assert isinstance(traced[0], FakeTensor), "no eager FSA enters the trace"
        assert native_op.get_ctc_fsa_fast_bw(**kwargs)[0] is traced[0], "within the trace, the heads share the FSA"
    after = native_op.get_ctc_fsa_fast_bw(**kwargs)
    assert not isinstance(after[0], FakeTensor), "no trace-time FSA leaks out"
    torch.testing.assert_close(after, eager)
