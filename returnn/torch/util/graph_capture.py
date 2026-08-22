"""
Whole-train-step CUDA-graph capture for the torch engine ("graph replay training").

The whole train step -- extern data construction, the user train step function
(model forward incl e.g. SpecAugment), losses, and backward --
is captured once into a CUDA graph and then replayed per step,
eliminating all per-step Python/kernel-launch overhead of that region.
The optimizer step runs outside the graph on the static gradient buffers,
or in-graph with "capture_optimizer".

Config, e.g.::

    torch_cuda_graph = {
        "batch_size_bound": 200,        # max seqs per batch; smaller batches get zero-length padding seqs.
                                        # the batch dim is made STATIC (= this bound): always filled up to it
        "dim_capacity": {"data": 3000, "classes": 300},  # bound of the dynamic (time) dim per data key
        "packed_total_bound": {"data": 500_000},  # optional: tighter bound of the packed (gapped) total per key
        "partitioned": True,  # optional: fw/bwd-partitioned compile (min-cut remat) instead of one whole-step graph
        "activation_memory_budget": 0.9,  # optional, with "partitioned": save-vs-recompute knob (1.0 = save all)
        "warmup_steps": 2,              # eager steps before capture; 0 works too (see below)
        "capture_optimizer": True,      # grad clip + optimizer step in-graph (needs capturable optimizer)
        "compile": True,                # Inductor-codegen the whole step first, then capture that
        "capture": True,                # False (with compile): run the compiled step eagerly, no graph
    }

Requirements / current limitations (asserted):

- ``accum_grad_multiple_step == 1``, no grad scaler, no DDP / torch-distributed, no hot reloading.
- The batch dim is static (= ``batch_size_bound``);
  the varying real seq count shows up only as zero-length padding seqs.
  Ops normalizing by the batch-axis size
  (e.g. a plain mean over the batch dim, or a per-seq loss with default normalization)
  COUNT the padding seqs --
  normalize by lengths instead (e.g. ``custom_inv_norm_factor``, masked reductions).
- No eval epochs in the same process yet: the declared capacities live on the (global) template dims,
  so the eager eval path would build capacity-sized grids against normally-padded eval batches.
- The train step must be static traceable (:func:`rf.set_static_traceable` is enabled around it):
  static (bounded) shapes, bounded control flow (see e.g. :func:`rf.audio.specaugment`),
  no host reads of device values.
- Every batch must fit the declared bounds (asserted per step).

Mechanics (validated by standalone probes first, see the 2026 packed/CUDA-graph work):

- Static input buffers are the only graph inputs:
  per data key a capacity-padded data buffer and a device-resident seq-lens buffer,
  plus one device step scalar. Per step, the batch is copied in and the graph replayed.
- The extern data is rebuilt per step around these buffers with the template dims RESET,
  so at capture time every dim/layout cache misses and the whole layout chain
  (seq starts, masks, derived subsample lens, ...) is computed IN-graph from the lens buffers
  -- one captured graph then replays correctly across varying batch sizes and seq lengths.
- ``warmup_steps: 0`` (no eager step at all) works: the two things a warmup used to provide
  are handled explicitly -- the lazily created optimizer state + grads via
  :func:`GraphCapturedTrainStep._materialize_optimizer_state` (they are graph inputs of the
  in-graph optimizer step, so they must exist before the trace), and host-derived constants
  (e.g. the mel filterbank matrix) via creation outside all python dispatch modes in the torch
  backend's ``convert_to_tensor`` (created INSIDE the trace they become lifted inputs with a
  per-call H2D copy -- illegal under capture). Everything else (cuDNN/cuFFT plans, workspaces,
  autotune) is covered by the compiled warm run that precedes the capture.
- All eager steps before the capture run on a non-default CUDA stream:
  the first-ever backward binds autograd/grad-accumulator state to the stream it runs on,
  and a default-stream first backward makes any later capture-time backward fail
  (torch 2.7; reproduced in a minimal pure-torch example).
- Gradients live in static buffers, zeroed in-graph. They must never be freed
  (``p.grad = None`` would let the allocator hand the memory to other tensors,
  which the replay would then corrupt) -- thus also no ``optimizer.zero_grad()``.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Callable, Dict, List, Tuple
from contextlib import contextmanager
import gc
import os
import numpy
import torch

from returnn.util.basic import CollectionReadCheckCovered
from returnn.tensor import Tensor, TensorDict, Dim
import returnn.frontend as rf
from returnn.frontend.run_ctx import RunCtx, Loss

# noinspection PyProtectedMember
from ..data.extern_data import get_batch_dim_from_extern_data, _get_dyn_dims_from_extern_data

__all__ = ["GraphCapturedTrainStep", "graph_pools_reserved"]

# total bytes reserved by the current CUDA-graph private pool(s), set after capture
# (single active graph per engine; a recapture overwrites). For the engine memory log.
_graph_pools_reserved = 0


def graph_pools_reserved() -> int:
    """
    :return: bytes reserved by live CUDA-graph private pools --
        replay working memory, freed as tensors, thus INVISIBLE in (max_)memory_allocated
    """
    return _graph_pools_reserved


_inductor_workarounds_applied = False


def _patch_inductor_nan_asserts_nan_only() -> None:
    """
    Inductor's nan_asserts check isnan AND isinf on every buffer,
    but inf is legitimate here (mask fill values, -log 0 lattice scores from FastBaumWelch):
    only NaN indicates a defect.
    Replace the two emitters (torch 2.7 internals) with NaN-only versions
    that also put the buffer name into the assert message.
    """
    global _inductor_nan_asserts_patched
    if _inductor_nan_asserts_patched:
        return
    import sympy

    # noinspection PyProtectedMember
    from torch._inductor import ir

    # noinspection PyProtectedMember
    from torch._inductor.codegen.common import TensorArg

    # noinspection PyProtectedMember
    from torch._inductor.codegen.triton import TritonKernel

    # noinspection PyProtectedMember
    from torch._inductor.codegen.wrapper import PythonWrapperCodegen

    # noinspection PyProtectedMember
    from torch._inductor.virtualized import V

    def codegen_input_nan_asserts(self) -> None:
        """patched PythonWrapperCodegen method: nan-only input asserts (inf = legitimate mask values)"""
        self.prefix.writeline("# make sure graph inputs are not nan (inf is legitimate: mask values)")
        for name, buf in self.get_graph_inputs().items():
            if isinstance(buf, (sympy.Expr, ir.TorchBindObject)):
                continue
            self.prefix.writeline(f"assert not {name}.isnan().any().item(), {name!r}")

    def codegen_nan_check(self) -> None:
        """patched TritonKernel method: nan-only asserts, with the buffer name in the message"""
        wrapper = V.graph.wrapper_code
        _, call_args, arg_signatures, _ = self.args.python_argdefs()
        for arg, arg_signature in zip(call_args, arg_signatures):
            if isinstance(arg_signature, TensorArg):
                assert not V.graph.cpp_wrapper  # python wrapper only here
                if _nan_check_report_mode:
                    # report-only: NaN counts per buffer, no abort --
                    # the culprit is the buffer whose NaN pattern CHANGES at the failing call
                    # (expected masked-lane / pre-guard NaNs stay constant across calls)
                    wrapper.writeline(f"_n_ = {arg}.isnan().sum().item() if {arg}.is_floating_point() else 0")
                    wrapper.writeline(f"_n_ and print('NANREP', {arg!r}, _n_, flush=True)")
                    if arg in _nan_dump_buffer_names:
                        # overwritten every call: after a crash the file holds the failing call's value
                        wrapper.writeline(f"torch.save({arg}.clone(), 'nan-dump-{arg}.pt')")
                else:
                    wrapper.writeline(f"assert not {arg}.isnan().any().item(), {arg!r}")

    PythonWrapperCodegen.codegen_input_nan_asserts = codegen_input_nan_asserts
    TritonKernel.codegen_nan_check = codegen_nan_check
    _inductor_nan_asserts_patched = True


_inductor_nan_asserts_patched = False
_nan_check_report_mode = False
_nan_dump_buffer_names = ()


class _NanTraceMode:
    """
    TorchDispatchMode raising at the first aten op whose float output contains NaN
    (kernel granularity, catches everything in eager; a host sync per op -- one-step debug only).
    The flash varlen forward is allowlisted:
    its filler/tail rows are NaN by construction and cleaned right after (see the guard).
    Intentional -inf mask values are legal, so only NaN is checked.
    """

    def __new__(cls):
        # noinspection PyProtectedMember
        from torch.utils._python_dispatch import TorchDispatchMode

        class _Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                del types  # required by __torch_dispatch__
                out = func(*args, **(kwargs or {}))
                fname = str(func)
                # allocations are uninitialized by design; flash filler/tail rows are
                # NaN by construction (cleaned right after, see the guard);
                # alias/copy ops cannot CREATE NaN (detection moves to the first compute op)
                _skip = (
                    "flash_attention",
                    "empty",
                    "resize",
                    "detach",
                    "view",
                    "alias",
                    "transpose",
                    "permute",
                    "expand",
                    "clone",
                    "_to_copy",
                    "copy_",
                    "slice",
                    "select",
                    "squeeze",
                    "unsqueeze",
                    "reshape",
                    "contiguous",
                    "t.default",
                )
                if any(s in fname for s in _skip):
                    return out
                outs = out if isinstance(out, (tuple, list)) else [out]
                for t in outs:
                    if isinstance(t, torch.Tensor) and t.is_floating_point() and t.isnan().any():
                        shapes = [tuple(a.shape) for a in args if isinstance(a, torch.Tensor)]
                        raise RuntimeError(f"_NanTraceMode: NaN from {func} (arg shapes {shapes})")
                return out

        return _Mode()


def _patch_zero_init_generated_buffers() -> None:
    """
    Debug: zero-fill every buffer the generated code allocates.
    Bound-regime buffers have unwritten tails (beyond the actual totals) with arbitrary garbage;
    eager has them too, but Inductor fusion can move a reduction across the masking
    (then garbage can reach the outputs).
    Clean run with zero-fill vs broken without = uninitialized-read confirmed.
    One memset per allocation, debug only.
    The generated modules bind ``empty_strided_cuda`` from this symbol at load time,
    so the patch must be in place before compile/cache-load (it is: applied at compile setup).
    """
    global _zero_init_buffers_patched
    if _zero_init_buffers_patched:
        return
    # noinspection PyProtectedMember
    import torch._C._dynamo.guards as _guards

    # noinspection PyProtectedMember,PyUnresolvedReferences
    orig = _guards._empty_strided_cuda

    def _empty_strided_cuda_zeroed(*args, **kwargs) -> torch.Tensor:
        t = orig(*args, **kwargs)
        t.zero_()
        return t

    _guards._empty_strided_cuda = _empty_strided_cuda_zeroed
    _zero_init_buffers_patched = True


_zero_init_buffers_patched = False


def _apply_inductor_workarounds():
    """
    torch 2.7 Inductor issues hit by the compiled whole step:

    - aten.searchsorted lowers to an INLINE ops.bucketize; fused into a reduction's inner_fn,
      its read is a StarDep -> LoweringException ("StarDep does not have an index").
      Extern-kernel fallback instead (result realized, downstream reads are normal deps).
    - Inductor's own philox RNG derives per-call seeds host-side,
      which a CUDA-graph capture would FREEZE (dropout masks repeated across replays).
      fallback_random routes RNG through the aten kernels, whose generator state the capture
      registers -- each replay draws fresh randomness (and numerics match eager).
    - The addmm fusion pattern (add(mm(a, b), bias) -> addmm) checks shapes but not dtypes.
      Mixed dtypes are legal for the add (type promotion, e.g. bf16 matmul + f32 bias under AMP,
      which RF Linear produces: explicit dot + add, not F.linear),
      but not for addmm -> RuntimeError when the compiled code runs.
      Wrap the registered patterns' extra_check to require equal dtypes.
    - The flash varlen extern kernels require a contiguous last dim.
      Their registered sdpa_constraint misses the 3-dim (varlen, cu_seqlens) case:
      Inductor fuses away the explicit contiguous clones of q/k
      and materializes them head-interleaved (last-dim stride = num heads),
      -> RuntimeError when the compiled code runs.
      Constrain them to the fx (eager) strides, which were flash-legal.
    """
    global _inductor_workarounds_applied
    if _inductor_workarounds_applied:
        return

    # noinspection PyProtectedMember
    from torch._inductor import lowering, decomposition, config as inductor_config

    for overload in (torch.ops.aten.searchsorted.Tensor, torch.ops.aten.searchsorted.Scalar):
        lowering.lowerings.pop(overload, None)
        decomposition.decompositions.pop(overload, None)
        lowering.make_fallback(overload)
    inductor_config.fallback_random = True

    # noinspection PyProtectedMember
    for overload in (
        torch.ops.aten._flash_attention_forward.default,
        torch.ops.aten._flash_attention_backward.default,
    ):
        lowering.add_layout_constraint(overload, lowering.constrain_to_fx_strides)

    # noinspection PyProtectedMember
    import torch._inductor.fx_passes.post_grad as post_grad

    num_patched = 0
    for entries in post_grad.pass_patterns[2].patterns.values():
        for entry in entries:
            if getattr(getattr(entry, "handler", None), "__name__", None) != "addmm":
                continue
            orig_check = entry.extra_check

            def _addmm_equal_dtypes_check(match, _orig_check=orig_check):
                """the fused addmm requires equal dtypes, the matched add does not (promotion)"""
                mat1, mat2 = match.args
                nodes = [match.kwargs["inp"], mat1, mat2]
                vals = [n.meta.get("val") for n in nodes if isinstance(n, torch.fx.Node)]
                if len({v.dtype for v in vals if isinstance(v, torch.Tensor)}) > 1:
                    return False
                return _orig_check(match)

            entry.extra_check = _addmm_equal_dtypes_check
            num_patched += 1
    assert num_patched, "Inductor addmm fusion patterns not found (torch internals changed?)"

    _register_smoothed_ce_bwd_pattern()

    _inductor_workarounds_applied = True


def _register_smoothed_ce_bwd_pattern() -> None:
    """
    Rewrite the (grad-level label-smoothed) sparse-CE backward chain into its closed form.

    The chain is the PLAIN aten emission of ``-gather(log_probs, targets)`` backward
    (dense new_zeros + scatter_add accumulator) -> scale + shift*sum(abs)
    (:func:`scaled_gradient_ext`, i.e. :func:`label_smoothed_log_prob_gradient`)
    -> _log_softmax_backward_data. No marker op; a guard test asserts the match
    keeps firing across torch upgrades (decomposition drift fails loudly there).
    Inductor cannot do this itself: scatter is a fusion barrier, and the rewrite is algebraic
    (the accumulator has ONE nonzero per row, so its row reductions are analytic in the upstream).
    The closed form materializes NO dense intermediates:
    one elementwise kernel over [frames, classes] + one scatter_add into the OUTPUT
    (~2 GiB f32 per accumulator saved per CE head at the loq scale).
    """
    # noinspection PyProtectedMember
    import torch._inductor.fx_passes.post_grad as post_grad

    # noinspection PyProtectedMember
    from torch._inductor.pattern_matcher import CallFunction, Ignored, KeywordArg, register_graph_pattern

    aten = torch.ops.aten
    # The POST-GRAD aten emission of the generic gather backward + scaled_gradient_ext
    # + the DECOMPOSED log_softmax backward (dumped ground truth, see the project notes):
    # hand-built pattern, so the full() size list and the smoothing scalars are wildcards.
    full = CallFunction(
        aten.full.default, Ignored(), 0, dtype=Ignored(), layout=Ignored(), device=Ignored(), pin_memory=Ignored()
    )
    unsq = CallFunction(aten.unsqueeze.default, KeywordArg("neg_up"), 1)
    scat = CallFunction(aten.scatter_add.default, full, 1, KeywordArg("idx"), unsq, _users=2)
    mul = CallFunction(aten.mul.Tensor, scat, KeywordArg("factor"))
    abs1 = CallFunction(aten.abs.default, scat)
    sum1 = CallFunction(aten.sum.dim_IntList, abs1, Ignored(), True)
    mul1 = CallFunction(aten.mul.Tensor, sum1, KeywordArg("shift"))
    add = CallFunction(aten.add.Tensor, mul, mul1, _users=2)
    sum2 = CallFunction(aten.sum.dim_IntList, add, Ignored(), True)
    mul2 = CallFunction(aten.mul.Tensor, KeywordArg("exp_lp"), sum2)
    pat = CallFunction(aten.sub.Tensor, add, mul2)

    @register_graph_pattern(pat, pass_dict=post_grad.pass_patterns[1])
    def _smoothed_ce_bwd_repl(match, *, neg_up, idx, factor, shift, exp_lp):
        def repl(neg_up_, idx_, exp_lp_):
            """the closed-form replacement (traced)"""
            absu = torch.abs(neg_up_)
            num_classes = exp_lp_.shape[-1]
            sum_g = factor * neg_up_ + num_classes * shift * absu  # row sum of the dense grad, analytic
            dense = (shift * absu).unsqueeze(1) - exp_lp_ * sum_g.unsqueeze(1)
            return dense.scatter_add(1, idx_, (factor * neg_up_).unsqueeze(1))

        global _smoothed_ce_bwd_match_count
        _smoothed_ce_bwd_match_count += 1
        match.replace_by_example(repl, [neg_up, idx, exp_lp])


# incremented per pattern match at compile time; the guard test asserts it (see tests)
_smoothed_ce_bwd_match_count = 0


@contextmanager
def _allow_non_fake_inputs():
    """
    The AOT dispatcher runs its analysis under a strict FakeTensorMode,
    which rejects the closed-over real static buffers of this module.
    They are effectively graph constants read at runtime
    (the same static-buffer semantics the plain capture relies on),
    so relax the mode -- scoped around the trace+compile call, which runs on REAL inputs
    (fake inputs instead crash Inductor's runtime autotune on fake data ptrs).
    """
    # noinspection PyProtectedMember
    import torch._subclasses.fake_tensor as fake_tensor_mod

    orig_init = fake_tensor_mod.FakeTensorMode.__init__

    def permissive_init(self, *args, **kwargs):
        """FakeTensorMode.__init__ with allow_non_fake_inputs forced"""
        kwargs["allow_non_fake_inputs"] = True
        # noinspection PyArgumentList
        orig_init(self, *args, **kwargs)

    fake_tensor_mod.FakeTensorMode.__init__ = permissive_init
    try:
        yield
    finally:
        fake_tensor_mod.FakeTensorMode.__init__ = orig_init


class GraphCapturedTrainStep:
    """
    Orchestrates eager warmup steps, the one-time capture, and per-step replay.
    See the module docstring.
    """

    def __init__(
        self,
        *,
        opts: Dict[str, Any],
        extern_data_template: TensorDict,
        device: Union[str, torch.device],
        float_dtype: Optional[torch.dtype],
        params: List[torch.nn.Parameter],
        run_step: Callable[..., None],
        post_step: Optional[Callable[[], None]] = None,
        get_optimizer: Optional[Callable[[], torch.optim.Optimizer]] = None,
        get_buffers: Optional[Callable[[], List[torch.Tensor]]] = None,
        rf_params: Optional[List[rf.Parameter]] = None,
        packed_batch_size: Optional[Dict[str, int]] = None,
    ):
        """
        :param opts: the ``torch_cuda_graph`` config dict, see the module docstring
        :param extern_data_template:
        :param device:
        :param float_dtype:
        :param params: all model parameters. Their .grad becomes static buffers, zeroed in-graph.
        :param run_step: ``engine._run_step``-like: ``run_step(extern_data, step=step_tensor)``,
            initializing the train-step run ctx and running the user train step function.
        :param post_step: optional, captured IN-graph after the backward
            (opts "capture_optimizer": grad clip + optimizer step; must be capture-safe:
            capturable optimizer (device step counters), no host reads, constant/device-tensor lr).
        :param rf_params: the RF-level model params (required for opts "compile":
            the compiled step takes the param raw tensors as graph inputs via RF-level raw swap).
        :param packed_batch_size: the config option of the same name, when set and statically
            known (a dict/int, not a callable). Used ONLY to infer a missing per-key
            "packed_total_bound", see :func:`_get_data_buf`.
        """
        assert str(device).startswith("cuda"), f"torch_cuda_graph requires a cuda device, got {device!r}"
        opts = CollectionReadCheckCovered(opts)  # catch unknown (e.g. typo'd) option keys, see below
        self.batch_size_bound = int(opts["batch_size_bound"])
        self.dim_capacity: Dict[str, int] = dict(opts["dim_capacity"])
        # optional tighter bound of the packed (gapped) total per packed-collate key
        # (e.g. batch size + per-seq gap slack); the default -- every seq at full capacity --
        # can be far larger and the activations scale with it
        self.packed_total_bound: Dict[str, int] = dict(opts.get("packed_total_bound", {}))
        # only for inferring a missing packed_total_bound entry, see _get_data_buf
        self._packed_batch_size: Dict[str, int] = dict(packed_batch_size) if isinstance(packed_batch_size, dict) else {}
        self.warmup_steps = int(opts.get("warmup_steps", 2))
        # Run the eager warmup steps on a minimal dummy batch instead of the real one
        # (params + optimizer state are restored afterwards, so the model never sees them).
        # The warmup peaks far above the captured step (eager, no buffer-reuse planning),
        # and that peak, not the steady state, is what the job's GPU must fit.
        # Value: True = seq len 1 for every key. Deliberately extreme, NOT a tuned default:
        # the warmup exists for its SIDE EFFECTS (lazy allocs -- optimizer moments,
        # cuDNN/cuFFT plans -- must happen here, not inside the capture), so a front-end that
        # cannot take a length-1 seq (e.g. an stft/conv chain: empty-input branch, negative
        # "valid" out length) must fail LOUDLY here rather than silently warm up a branch the
        # traced graph never takes. Then set an int (frames per seq, all keys) or a per-key
        # dict, just above what the front-end needs (e.g. one stft window of audio).
        dummy_warmup_opt = opts.get("dummy_warmup", False)
        self.dummy_warmup = bool(dummy_warmup_opt)
        self._dummy_warmup_seq_len: Union[int, Dict[str, int]] = (
            1 if isinstance(dummy_warmup_opt, bool) else dummy_warmup_opt
        )
        self._get_optimizer = get_optimizer
        self._get_buffers = get_buffers
        self._pre_dummy_warmup_params: Optional[List[torch.Tensor]] = None
        self._pre_dummy_warmup_buffers: Optional[List[torch.Tensor]] = None
        self._device = torch.device(device)
        self._float_dtype = float_dtype
        self._extern_data_template = extern_data_template
        self._run_step = run_step
        self._post_step = post_step if opts.get("capture_optimizer", False) else None
        self._params = params
        self._compile = bool(opts.get("compile", False))
        self._capture_graph = bool(opts.get("capture", True))
        # partitioned mode: compile the loss via the joint AOT path instead of the single
        # inference-style whole-step graph: fw/bwd split by min-cut rematerialization,
        # grads via .backward() through the compiled autograd.Function (still captured in-graph).
        # This enables activation_memory_budget as a global save-vs-recompute knob.
        self._partitioned = bool(opts.get("partitioned", False))
        # 0..1 fraction of the save-everything activation memory (torch._functorch.config);
        # only meaningful with "partitioned"
        self._activation_memory_budget: Optional[float] = opts.get("activation_memory_budget")
        # with "partitioned" + budget: lift the min-cut recompute allowlist bans
        # (matmul-and-friends outputs are otherwise never recomputed, the budget saturates)
        self._aggressive_recomputation = bool(opts.get("aggressive_recomputation", False))
        # partitioned mode: sorted (data-buf keys, lens-buf keys) lifted to trace inputs; see _compiled_call_args
        self._partitioned_buf_keys: Optional[Tuple[List[str], List[str]]] = None
        if self._partitioned:
            assert self._compile, 'torch_cuda_graph: "partitioned" requires "compile"'
        # debug: Inductor generates a nan-assert after every kernel,
        # pinpointing the first kernel producing nan/inf inside the compiled program
        self._inductor_nan_asserts = bool(opts.get("inductor_nan_asserts", False))
        # debug: dump the AOT-traced FX graph (pre-Inductor) into this dir,
        # to inspect for wrongly baked constants (e.g. a capacity used as a length)
        self._dump_fx_dir: Optional[str] = opts.get("dump_fx_dir")
        # debug: record the CUDA allocator history and dump a snapshot
        # (cuda-memory-snapshot.pickle in cwd, for torch.cuda.memory viz)
        # on OOM during compile/capture -- shows what actually fills the memory at the peak
        self._dump_memory_snapshot = bool(opts.get("dump_memory_snapshot", False))
        # debug (compiled-eager, i.e. "capture": False): check the loss outputs every step,
        # on the first non-finite value dump that step's raw extern-data batch to cwd and raise
        # -- hands over the exact failing batch for offline eager-vs-compiled diffing
        self._debug_nan_dump_inputs = bool(opts.get("debug_nan_dump_inputs", False))
        # debug: zero-fill every buffer the generated code allocates (see the patch func)
        self._debug_zero_init_buffers = bool(opts.get("debug_zero_init_buffers", False))
        # debug: with inductor_nan_asserts, PRINT per-buffer NaN counts instead of asserting
        self._debug_nan_report = bool(opts.get("debug_nan_report", False))
        # debug: conservative Inductor codegen (no epilogue fusion, no pattern matcher) --
        # if the sporadic NaN vanishes, a miscompiled fusion is confirmed
        self._inductor_conservative = bool(opts.get("inductor_conservative", False))
        # debug: run the traced AOT graph with eager kernels (no Inductor at all)
        self._debug_aot_eager = bool(opts.get("debug_aot_eager", False))
        # debug: run step_core UNTRACED, plain eager, on the same bound buffers
        # (isolates bound-regime semantics from the AOT trace)
        self._debug_eager_bound = bool(opts.get("debug_eager_bound", False))
        # debug: with debug_nan_report, torch.save these generated-code buffers each call
        # (buffer names from a previous run's generated module; numbering is stable per graph)
        self._debug_dump_buffer_names = tuple(opts.get("debug_dump_buffer_names", ()))
        self._last_extern_data_raw: Optional[Dict[str, Union[torch.Tensor, numpy.ndarray]]] = None
        if self._dump_memory_snapshot:
            # noinspection PyProtectedMember
            torch.cuda.memory._record_memory_history(max_entries=200_000)
        assert self._capture_graph or self._compile, 'torch_cuda_graph: "capture": False requires "compile": True'
        self._compiled_fn: Optional[Callable[[List[torch.Tensor]], tuple]] = None
        self._compiled_n_calls = 0
        self._rf_params = rf_params
        if self._compile:
            assert rf_params is not None, "torch_cuda_graph compile: rf_params required (RF model)"
        # per loss: (name, traced Loss (for the flags), inv norm is a tensor output, else its static value)
        self._compiled_loss_meta: List[Tuple[str, Loss, bool, Optional[int]]] = []
        self._compiled_n_loss_outs = 0
        opts.assert_all_read()
        for p in self._params:
            p.grad = torch.zeros_like(p)  # static, never freed (the captured graph writes into these)

        self._batch_dim = get_batch_dim_from_extern_data(extern_data_template)
        self._batch_dim_staticized = self._batch_dim.dimension is None  # standard case: True
        if self._batch_dim_staticized:
            # The copy-in always fills the batch up to the bound (zero-length padding seqs),
            # so in this regime the batch dim IS static -- make it so.
            # (A process-wide template-dim mutation, like the declared capacities;
            # toggled off around the dynamic-shape paths, see set_bound_shapes_enabled.)
            self._batch_dim.size = self.batch_size_bound
            self._batch_dim.capacity = self.batch_size_bound
            self._batch_dim.dyn_size_ext = None
        assert self._batch_dim.dimension == self.batch_size_bound
        self._data_bufs: Dict[str, torch.Tensor] = {}
        self._pinned_bufs: Dict[str, torch.Tensor] = {}  # host staging, see _copy_in
        # guards the pinned staging: the next host write must wait for the previous async H2D
        self._copy_in_event: Optional[torch.cuda.Event] = None
        self._prev_copy_shapes: Dict[str, Tuple[int, ...]] = {}
        self._lens_bufs: Dict[str, torch.Tensor] = {}
        self._host_raws: Dict[str, Any] = {}  # non-capturable (string) keys, passed through host-side
        self._packed_opts: Dict[str, Dict[str, int]] = {}  # for packed-collate keys
        self._step_buf = torch.zeros((), dtype=torch.int64, device=self._device)
        self._step_t = Tensor("global_train_step", dims=(), dtype="int64")
        self._step_t.raw_tensor = self._step_buf

        self._eager_stream = torch.cuda.Stream()
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._ctx: Optional[RunCtx] = None
        self._n_eager = 0

        # declared capacities go on the template dims (persist across reset_eager;
        # capacity propagates through dim math, so derived dims are bounded too)
        self._cap_dims: List[Tuple[Dim, int]] = []
        for k, data in extern_data_template.data.items():
            for i, dim in enumerate(data.dims):
                if i >= 1 and dim.dimension is None and dim is not self._batch_dim:
                    assert k in self.dim_capacity, (
                        f"torch_cuda_graph: dim_capacity for data key {k!r} required (dyn dim {dim})"
                    )
                    dim.capacity = self.dim_capacity[k]
                    self._cap_dims.append((dim, self.dim_capacity[k]))

    def data_bound_sizes(self) -> Dict[str, int]:
        """
        :return: per extern-data key, the number of frames ONE captured step computes over,
            i.e. the static buffer extent: the packed total bound for packed keys,
            batch bound * capacity for the padded ones.
            The engine reports the unused part of this as the epoch's bound slack.
        """
        res = {}
        for k, buf in self._data_bufs.items():
            if k in self._packed_opts:
                res[k] = int(buf.shape[0])
            elif buf.ndim >= 2:
                res[k] = int(buf.shape[0]) * int(buf.shape[1])
        return res

    def set_bound_shapes_enabled(self, enabled: bool):
        """
        Toggle the process-wide bound-shape dim state:
        the static batch dim, the declared capacities, and the derived memoized capacities.
        The plain dynamic-shape paths (e.g. :func:`Engine.eval_model`) need it OFF --
        with it on, they would build capacity-sized grids and masks
        against normally-padded batches.
        The engine re-enables it at the start of each train epoch.
        """
        # noinspection PyProtectedMember
        from returnn.tensor import _dim_extra

        if enabled:
            if self._batch_dim_staticized:
                self._batch_dim.size = self.batch_size_bound
                self._batch_dim.capacity = self.batch_size_bound
                self._batch_dim.dyn_size_ext = None
            for dim, cap in self._cap_dims:
                dim.capacity = cap
        else:
            if self._batch_dim_staticized:
                self._batch_dim.size = None
                self._batch_dim.capacity = None
                self._batch_dim.dyn_size_ext = None
            for dim, _ in self._cap_dims:
                dim.capacity = None
            # derived dims memoized their capacity lazily; clear them too (they re-derive)
            # noinspection PyProtectedMember
            for dim in list(_dim_extra.derived_capacity_memoized_dims):
                dim.capacity = None
            _dim_extra.derived_capacity_memoized_dims.clear()

    def _get_data_buf(self, k: str, raw: torch.Tensor, packed: Optional[Dict[str, int]]) -> torch.Tensor:
        buf = self._data_bufs.get(k)
        if buf is not None:
            return buf
        data = self._extern_data_template.data[k]
        dtype = raw.dtype
        if dtype.is_floating_point and self._float_dtype:
            dtype = self._float_dtype
        if packed is not None:
            # packed collate: flat contiguous [total, ...feature];
            # bound = declared packed_total_bound, else batch bound * capacity
            self._packed_opts[k] = dict(packed)
            gap = int(packed.get("gap", 0))
            align = int(packed.get("align", 1))
            total = self.packed_total_bound.get(k)
            if total is None and k in self._packed_batch_size:
                # packed_batch_size budgets CONTENT; the LAYOUT additionally needs the per-seq
                # gap and align rounding: each seq occupies at most len + gap + align - 1 frames
                # (same derivation as _regap_total_bound in the packed backend). At the usual
                # gap 0 / align 1 this is exactly packed_batch_size, i.e. the rule everyone
                # already writes by hand; with a gap/align layout it is the term that is easy
                # to forget, and forgetting it only shows up as an async device-side assert.
                total = self._packed_batch_size[k] + self.batch_size_bound * (gap + align - 1)
            if total is None:
                # no content budget known (no packed_batch_size, or a callable one):
                # fall back to every seq at full capacity, which is correct but far larger
                total = self.batch_size_bound * self.dim_capacity[k]
            # early static sanity check of the DECLARED bound (before any step runs):
            # a single seq at full dim_capacity, laid out with the configured gap/align,
            # must fit -- a bound below that is certainly mis-configured (the per-batch
            # content check happens in pack(); this catches gross mistakes at startup).
            one_seq = -(-(self.dim_capacity[k] + gap) // align) * align
            assert total >= one_seq, (
                f"torch_cuda_graph: packed_total_bound[{k!r}] = {total} cannot hold one seq at"
                f" dim_capacity {self.dim_capacity[k]} with gap {gap} align {align}"
                f" (needs >= {one_seq}); raise packed_total_bound or lower dim_capacity"
            )
            shape = [total] + list(raw.shape[1:])
        else:
            assert all(d.dimension is not None for d in data.dims[2:]), (
                f"torch_cuda_graph: only dims[1] may be dynamic, got {data}"
            )
            has_dyn_spatial = len(data.dims) >= 2 and data.dims[1].dimension is None
            shape = [self.batch_size_bound]
            if has_dyn_spatial:
                shape.append(self.dim_capacity[k])
            shape += [d.dimension for d in data.dims[2 if has_dyn_spatial else 1 :]]
        buf = torch.zeros(shape, dtype=dtype, device=self._device)
        self._data_bufs[k] = buf
        return buf

    def _copy_in(self, extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]]):
        if self._copy_in_event is not None:
            # the pinned staging buffers are reused every step:
            # the host memcpy below must not overwrite them
            # while the previous step's async H2D from them is still in flight
            # (normally hidden by the per-step loss host-read, but NOT guaranteed --
            # a partially overwritten staging buffer corrupts the previous step's inputs)
            self._copy_in_event.synchronize()
        for k, data in self._extern_data_template.data.items():
            raw = extern_data_raw[k]
            if data.dtype == "string" or (isinstance(raw, numpy.ndarray) and raw.dtype.kind in "USO"):
                # strings (e.g. seq_tag) cannot be graph inputs; passed through host-side
                # (NOTE: under replay, the captured step never re-reads them --
                # a traceable train step must not consume their values)
                if raw.shape[0] < self.batch_size_bound:
                    # pad to the bound (the batch dim is static in this regime)
                    pad_shape = (self.batch_size_bound - raw.shape[0],) + raw.shape[1:]
                    raw = numpy.concatenate([raw, numpy.full(pad_shape, "", dtype=raw.dtype)])
                self._host_raws[k] = raw
                continue
            if isinstance(raw, numpy.ndarray):
                raw = torch.from_numpy(raw)
            packed = extern_data_raw.get(k + ":packed")
            buf = self._get_data_buf(k, raw, packed)
            if raw.dtype != buf.dtype:
                raw = raw.to(dtype=buf.dtype)
            if packed is not None:
                assert raw.shape[0] <= buf.shape[0], (
                    f"torch_cuda_graph: packed {k} total {raw.shape[0]} exceeds bound {buf.shape[0]}"
                )
                # the packed CONTENT bound (see pack(): bound - capacity*gap) sizes the traced
                # re-layout buffers; a batch above it would overflow them INSIDE the replay
                # (an async illegal access) -- reject it here, host-side, every step
                gap_k = int(self._packed_opts[k].get("gap", 0))
                align_k = int(self._packed_opts[k].get("align", 1))
                # the align term belongs here too: a seq occupies roundup(len + gap, align),
                # i.e. up to align-1 frames MORE than len + gap. Leaving it out made this check
                # pass for a batch the layout could not hold, and the overflow then surfaced as
                # an async device-side assert inside the replay instead of here.
                content_bound = buf.shape[0] - self.batch_size_bound * (gap_k + align_k - 1)
                assert raw.shape[0] <= content_bound, (
                    f"torch_cuda_graph: packed {k} content total {raw.shape[0]} exceeds the"
                    f" content bound {content_bound} (= packed_total_bound {buf.shape[0]}"
                    f" - batch_size_bound {self.batch_size_bound} * (gap {gap_k}"
                    f" + align {align_k} - 1)); raise packed_total_bound"
                )
            else:
                assert all(a <= b for a, b in zip(raw.shape, buf.shape)), (
                    f"torch_cuda_graph: {k} shape {tuple(raw.shape)} exceeds bounds {tuple(buf.shape)}"
                )
            prev_shape = self._prev_copy_shapes.get(k)
            if prev_shape is not None and any(n < p for n, p in zip(raw.shape, prev_shape)):
                # the batch extents shrank (common: sorted_reverse batching):
                # the region beyond the current extents still holds the PREVIOUS batch's data.
                # All consumers should mask it -- but stale plausible values there
                # are exactly the hardest corruption to notice if any masking is imperfect;
                # a memset is negligible vs the step. Zero it (simplest: the whole buffer).
                buf.zero_()
            self._prev_copy_shapes[k] = tuple(raw.shape)
            if raw.is_pinned():
                # already pinned (e.g. a pinned batch cache): direct true-async H2D, no staging memcpy
                buf[tuple(slice(0, s) for s in raw.shape)].copy_(raw, non_blocking=True)
            else:
                # pinned staging: a pageable .to(device, non_blocking=True) is a silent SYNCHRONOUS copy
                # (plus a temp device alloc);
                # host-memcpy into pinned,
                # then one true-async H2D of the actual region into the static buffer slice
                pin = self._pinned_bufs.get(k)
                if pin is None or pin.shape[0] < buf.shape[0]:
                    pin = torch.empty(buf.shape, dtype=buf.dtype, pin_memory=True)
                    self._pinned_bufs[k] = pin
                pin_slice = pin[tuple(slice(0, s) for s in raw.shape)]
                pin_slice.copy_(raw)
                buf[tuple(slice(0, s) for s in raw.shape)].copy_(pin_slice, non_blocking=True)
            size = extern_data_raw.get(k + ":seq_len")
            if size is not None:
                lens_buf = self._lens_bufs.get(k)
                lens_pin = self._pinned_bufs.get(k + ":seq_len")
                if lens_buf is None:
                    lens_buf = torch.zeros(self.batch_size_bound, dtype=torch.int32, device=self._device)
                    self._lens_bufs[k] = lens_buf
                    lens_pin = torch.zeros(self.batch_size_bound, dtype=torch.int32, pin_memory=True)
                    self._pinned_bufs[k + ":seq_len"] = lens_pin
                n = size.shape[0]
                assert n <= self.batch_size_bound, (
                    f"torch_cuda_graph: batch size {n} exceeds batch_size_bound {self.batch_size_bound}"
                )
                # NEVER truncate silently:
                # a seq longer than the declared capacity
                # would have its tail ignored by all capacity-sized masks/positions
                # -- degraded training with no structural error anywhere
                # (everything is self-consistently capacity-sized).
                # This is the one place that sees the true host lens for every key every step,
                # so the bound is enforced HERE, loudly.
                max_len = int(size.max()) if n > 0 else 0
                cap = self.dim_capacity.get(k)
                assert cap is None or max_len <= cap, (
                    f"torch_cuda_graph: seq len {max_len} of {k!r} exceeds dim_capacity {cap};"
                    f" raise dim_capacity (data beyond the capacity would be silently ignored)"
                )
                lens_pin[:n].copy_(size.to(dtype=torch.int32))
                lens_pin[n:].zero_()  # zero-length padding seqs
                lens_buf.copy_(lens_pin, non_blocking=True)
        if self._copy_in_event is None:
            self._copy_in_event = torch.cuda.Event()
        self._copy_in_event.record()

    def _build_extern_data(self) -> TensorDict:
        """
        Like :func:`returnn.torch.data.extern_data.raw_dict_to_extern_data`,
        but around the static buffers:
        capacity-padded data, device-resident seq lens, the static batch dim.
        The template dims are reset, so nothing from a previous step is cached
        (at capture time, the whole layout computes in-graph).
        """
        batch_dim = self._batch_dim
        for dim in _get_dyn_dims_from_extern_data(self._extern_data_template):
            dim.reset_eager()
        extern_data = TensorDict()
        for k, data in self._extern_data_template.data.items():
            data = data.copy_template()
            if k in self._host_raws:
                data.raw_tensor = self._host_raws[k]
                extern_data.data[k] = data
                continue
            buf = self._data_bufs[k]
            lens_buf = self._lens_bufs.get(k)
            if len(data.dims) >= 2 and data.dims[1].dimension is None:
                spatial = data.dims[1]
                assert lens_buf is not None, f"torch_cuda_graph: missing seq lens for {k}"
                if spatial.dyn_size_ext is None:
                    spatial.dyn_size_ext = Tensor(spatial.name or "time", dims=[batch_dim], dtype="int32")
                spatial.dyn_size_ext.dtype = "int32"
                spatial.dyn_size_ext.raw_tensor = lens_buf  # device-resident
            if k in self._packed_opts:
                opts = self._packed_opts[k]
                gap, align = opts["gap"], opts["align"]
                spatial = data.dims[1]
                data.dtype = str(buf.dtype).split(".")[-1]
                packed_dim = Dim(buf.shape[0], name=(spatial.name or "time") + ":packed")  # static bound
                inner = Tensor(k, dims=[packed_dim] + list(data.dims[2:]), dtype=data.dtype, sparse_dim=data.sparse_dim)
                inner.raw_tensor = buf
                packed_t = rf.pack_import(
                    inner, batch_dim=batch_dim, spatial_dim=spatial, packed_dim=packed_dim, feature_dim=data.feature_dim
                )
                if gap or align > 1:
                    # a declared packed_total_bound (caller guarantees it)
                    # is usually much tighter than the worst case
                    # -- every seq at full capacity (+gap), aligned --
                    # and the model activations scale with this bound
                    regap_bound = self.packed_total_bound.get(k)
                    if regap_bound is None:
                        regap_bound = self.batch_size_bound * (-(-(self.dim_capacity[k] + gap) // align) * align)
                    packed_t = rf.packed_regap(packed_t, gap, align=align, total_bound=regap_bound)
                data.raw_tensor = packed_t.raw_tensor
            else:
                data.dtype = str(buf.dtype).split(".")[-1]
                data.raw_tensor = buf
            extern_data.data[k] = data
        return extern_data

    @property
    def captures_optimizer(self) -> bool:
        """whether the optimizer step (incl grad clip) is captured in-graph (opts "capture_optimizer")"""
        return self._post_step is not None

    def _step(self) -> RunCtx:
        for p in self._params:
            p.grad.zero_()  # in-graph
        with rf.set_static_traceable_ctx():
            extern_data = self._build_extern_data()
            self._run_step(extern_data, step=self._step_t)
            ctx = rf.get_run_ctx()
            total_loss = ctx.total_loss()
        total_loss.raw_tensor.backward()
        if self._post_step is not None:
            self._post_step()  # in-graph: grad clip + optimizer step
        return ctx

    def _make_dummy_extern_data_raw(
        self, extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]]
    ) -> Dict[str, Union[torch.Tensor, numpy.ndarray]]:
        """
        :param extern_data_raw: a real batch, as template for keys/dtypes/devices
        :return: the same keys, but a single minimal seq -- for the dummy warmup (see the call site)
        """

        def _dummy_len(key: str) -> int:
            n = self._dummy_warmup_seq_len
            if isinstance(n, dict):
                assert key in n, f"dummy_warmup: no seq len for data key {key!r} in {n}"
                n = n[key]
            assert isinstance(n, int) and n >= 1, f"dummy_warmup: invalid seq len {n!r} for data key {key!r}"
            return n

        res = {}
        seq_lens = {}
        for k, v in extern_data_raw.items():
            if k.endswith(":seq_len"):
                v = v[:1] * 0 + _dummy_len(k[: -len(":seq_len")])
                seq_lens[k[: -len(":seq_len")]] = v
            res[k] = v
        for k, v in extern_data_raw.items():
            if k.endswith(":seq_len") or not isinstance(v, torch.Tensor):
                continue
            if k in seq_lens:  # data with a seq dim: one seq of the configured dummy len
                if k in self._packed_opts:  # packed collate: flat [total, ...]
                    shape = (_dummy_len(k),) + tuple(v.shape[1:])
                else:
                    shape = (1, _dummy_len(k)) + tuple(v.shape[2:])
                res[k] = torch.zeros(shape, dtype=v.dtype, device=v.device)
            elif v.ndim >= 1 and v.shape[0] == self._last_batch_n_seqs(extern_data_raw):
                res[k] = v[:1]
        if "num_seqs" in res:
            res["num_seqs"] = (
                type(res["num_seqs"])(1)
                if not isinstance(res["num_seqs"], torch.Tensor)
                else torch.ones_like(res["num_seqs"])
            )
        return res

    @staticmethod
    def _last_batch_n_seqs(extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]]) -> int:
        for k, v in extern_data_raw.items():
            if k.endswith(":seq_len") and hasattr(v, "shape"):
                return int(v.shape[0])
        return 1

    def _materialize_optimizer_state(self) -> None:
        """
        Create the optimizer's lazily-initialized state (e.g. AdamW moments, capturable
        step counters) plus the param grads WITHOUT any model step, enabling
        ``warmup_steps: 0``: the captured in-graph optimizer step reads/writes the state
        tensors as stable graph inputs, so they must exist BEFORE trace/capture
        (a lazy init during capture recording would allocate them inside the graph pool).
        Optimizer-agnostic: one ``step()`` with ALL-ZERO grads at lr 0 creates whatever
        state the optimizer wants, doubly neutral for the params (no gradient signal, and
        every update term scales with lr); the values the step wrote (step counters) are
        zeroed afterwards, only the EXISTENCE is kept.
        No-op when state already exists (real or dummy warmup ran).
        """
        opt = self._get_optimizer() if self._get_optimizer is not None else None
        if opt is None or opt.state:
            return
        with torch.no_grad():
            for p in self._params:
                if p.grad is None:
                    # also needed pre-capture: the partitioned capture zeroes + accumulates
                    # into pre-existing grads
                    p.grad = torch.zeros_like(p)
            saved_lrs = []
            for g in opt.param_groups:
                lr = g["lr"]
                if isinstance(lr, torch.Tensor):  # capturable: device-tensor lr
                    saved_lrs.append(lr.clone())
                    lr.fill_(0)
                else:
                    saved_lrs.append(lr)
                    g["lr"] = 0.0
            opt.step()
            for g, lr in zip(opt.param_groups, saved_lrs):
                if isinstance(g["lr"], torch.Tensor):
                    g["lr"].copy_(lr)
                else:
                    g["lr"] = lr
            for state in opt.state.values():
                for v in state.values():
                    if isinstance(v, torch.Tensor):
                        v.zero_()

    def _restore_after_dummy_warmup(self):
        """
        Undo the dummy warmup steps: restore the module buffers (and, for optimizers whose
        update is not provably lr-multiplicative, the parameters -- see the dispatch site:
        for SGD/Adam/AdamW the dummy steps ran with lr 0, so the params never moved and no
        snapshot was taken), and zero the optimizer state, so only the EXISTENCE of the
        lazily created state (e.g. AdamW moments) is kept, not its dummy-batch values.

        Buffers need explicit restore in EVERY mode: running statistics (batch norm and
        friends) are updated in the forward pass, not by the optimizer, so neither
        restoring params nor a zero learning rate protects them.
        """
        with torch.no_grad():
            if self._pre_dummy_warmup_params is not None:
                for p, p_orig in zip(self._params, self._pre_dummy_warmup_params):
                    p.copy_(p_orig)
            for p in self._params:
                if p.grad is not None:
                    p.grad.zero_()
            if self._pre_dummy_warmup_buffers is not None:
                for b, b_orig in zip(self._get_buffers(), self._pre_dummy_warmup_buffers):
                    b.copy_(b_orig)
        self._pre_dummy_warmup_params = None
        self._pre_dummy_warmup_buffers = None
        opt = self._get_optimizer() if self._get_optimizer is not None else None
        if opt is not None:
            for state in opt.state.values():
                for v in state.values():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        v.zero_()
                    elif isinstance(v, torch.Tensor):  # step counters
                        v.zero_()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def _warmup_step_dynamic(
        self, extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]], *, global_train_step: int
    ) -> RunCtx:
        """
        One eager warmup step at the batch's ACTUAL shapes, not the bound-sized buffers:
        eager execution at the bounds peaks far above both the dynamic eager step
        and the Inductor-planned captured graph
        (bound-sized activations without any buffer-reuse planning;
        the AOT trace itself runs on fake tensors and costs no GPU memory).
        Kernels/cudnn warm up at approximately the right shapes;
        dim/layout caches built here are cleared again
        when the bound shapes get re-enabled.
        """
        from returnn.torch.data import extern_data as extern_data_util

        self.set_bound_shapes_enabled(False)
        try:
            # release the previous step's cached blocks:
            # each warmup batch has different shapes,
            # and the caching allocator cannot merge smaller cached blocks to serve bigger tensors,
            # so the cache from step N would stack on top of step N+1's allocations
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            for dim in _get_dyn_dims_from_extern_data(self._extern_data_template):
                dim.reset_eager()
            extern_data = extern_data_util.raw_dict_to_extern_data(
                extern_data_raw, extern_data_template=self._extern_data_template, device=self._device
            )
            for p in self._params:
                p.grad.zero_()
            # plain int step, like the normal eager engine path
            # (the device step tensor would mix devices with the cpu-resident dyn sizes here)
            self._run_step(extern_data, step=global_train_step)
            ctx = rf.get_run_ctx()
            total_loss = ctx.total_loss()
            total_loss.raw_tensor.backward()
            if self._post_step is not None:
                self._post_step()
            return ctx
        finally:
            self.set_bound_shapes_enabled(True)

    def _make_compiled_step(self) -> Callable[[List[torch.Tensor]], tuple]:
        """
        Build the Inductor-compiled whole step:
        ``compiled(param_raws) -> (per-loss summed / tensor inv-norm raws..., grads...)``.
        AOT tracing + Inductor codegen only -- no Dynamo, no torch.compile
        (a static-traceable step is aten-trace-safe by construction).
        Gradients are computed INSIDE the traced function (torch.autograd.grad, no .backward),
        all outputs detached: the compiled program is one inference-style graph.
        """
        from functorch.compile import aot_function

        # noinspection PyProtectedMember
        from torch._inductor.compile_fx import compile_fx

        # torch 2.7: a REAL flex HOP compile earlier in the process
        # (e.g. FlexAttention in the eager warmup steps) poisons flex compiles under fake tracing
        # (dynamo skip error on torch._library.utils.is_builtin inside can_auto_functionalize,
        # then the packed flex paths would decline mid-trace).
        # Resetting dynamo clears the stale compile state; harmless otherwise
        # (this path uses no dynamo itself, other compiled artifacts just recompile lazily).
        # noinspection PyProtectedMember
        torch._dynamo.reset()
        _apply_inductor_workarounds()
        if self._inductor_nan_asserts:
            # noinspection PyProtectedMember
            import torch._inductor.config as inductor_config

            inductor_config.nan_asserts = True
            if self._debug_nan_report:
                global _nan_check_report_mode, _nan_dump_buffer_names
                _nan_check_report_mode = True
                _nan_dump_buffer_names = self._debug_dump_buffer_names
            _patch_inductor_nan_asserts_nan_only()
        if self._debug_zero_init_buffers:
            _patch_zero_init_generated_buffers()
        if self._inductor_conservative:
            # noinspection PyProtectedMember
            import torch._inductor.config as inductor_config

            inductor_config.epilogue_fusion = False
            inductor_config.pattern_matcher = False
        rf_params = self._rf_params
        orig_raws = [p.raw_tensor for p in rf_params]
        trainable = [r.requires_grad for r in orig_raws]

        def step_core(raws):
            """the whole train step on the given param raws -> (loss raws..., grads...), see above"""
            for p_, t in zip(rf_params, raws):
                p_.raw_tensor = t
            try:
                with rf.set_static_traceable_ctx():
                    extern_data = self._build_extern_data()
                    self._run_step(extern_data, step=self._step_t)
                    ctx = rf.get_run_ctx()
                    total = ctx.total_loss()
                    loss_meta = []
                    outs = []
                    for name, loss in ctx.losses.items():
                        outs.append(loss.get_summed_loss().raw_tensor)
                        inv = loss.get_inv_norm_factor()
                        inv_is_tensor = isinstance(inv, Tensor)
                        if inv_is_tensor:
                            outs.append(inv.raw_tensor)
                        loss_meta.append((name, loss, inv_is_tensor, None if inv_is_tensor else inv))
                    self._compiled_loss_meta = loss_meta
                    self._compiled_n_loss_outs = len(outs)
                    if self._partitioned:
                        # joint AOT path: the total is the differentiable output
                        # (the partitioner traces and splits its backward);
                        # the per-loss outputs are detached metadata
                        return (total.raw_tensor,) + tuple(t.detach() for t in outs)
                    train_raws = [t for t, tr in zip(raws, trainable) if tr]
                    grads = torch.autograd.grad(total.raw_tensor, train_raws, allow_unused=True)
                    grads = [g if g is not None else torch.zeros_like(t) for g, t in zip(grads, train_raws)]
                return tuple(t.detach() for t in outs + grads)
            finally:
                for p_, r0 in zip(rf_params, orig_raws):
                    p_.raw_tensor = r0

        if self._debug_eager_bound:
            # plain eager execution of the untraced step on the bound buffers:
            # everything else (copy-in, static shapes, fillers, post-step) identical
            if not self._debug_nan_dump_inputs:
                return step_core

            def eager_bound_step(raws: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
                """on a non-finite loss: restore params+RNG, replay the step under the NaN tracer
                (raises at the exact op producing the first NaN)"""
                saved_params = [p.detach().clone() for p in self._params]
                saved_rng = torch.cuda.get_rng_state()
                outs = step_core(raws)
                # check ALL outs (losses AND grads): NaN grads poison the params via the
                # in-step optimizer, and the LOSS only turns non-finite one step later --
                # detecting at the grad step lets the replay trace the producing backward op
                bad = any(
                    bool(torch.logical_not(torch.isfinite(t)).any().item())
                    for t in outs
                    if isinstance(t, torch.Tensor) and t.is_floating_point()
                )
                if not bad:
                    return outs
                print("debug eager-bound: non-finite loss; REPLAYING the step under the NaN tracer", flush=True)
                with torch.no_grad():
                    for p, s in zip(self._params, saved_params):
                        p.copy_(s)
                torch.cuda.set_rng_state(saved_rng)
                with _NanTraceMode():
                    step_core(raws)
                raise Exception("debug eager-bound: replay did not reproduce the non-finite loss")

            return eager_bound_step
        backend = compile_fx
        if self._debug_aot_eager:
            # run the traced AOT graph with EAGER kernels (no Inductor codegen):
            # NaN persisting = the aten decompositions in the trace differ from plain eager;
            # clean = Inductor codegen at fault
            from functorch.compile import nop

            backend = nop
        if self._dump_fx_dir:
            assert not self._partitioned, "torch_cuda_graph: dump_fx_dir with partitioned not supported"
            dump_dir = self._dump_fx_dir
            inner_backend = backend

            def fw_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]) -> Callable:
                """dump the AOT fw graph (grads are traced inside it, so this is the whole step), then compile"""
                os.makedirs(dump_dir, exist_ok=True)
                with open(os.path.join(dump_dir, "aot_fw_graph.py"), "wt", encoding="utf-8") as f:
                    f.write(gm.print_readable(print_output=False))
                return inner_backend(gm, example_inputs)

            return aot_function(step_core, fw_compiler=fw_compiler)
        if self._partitioned:
            # loss-only joint graph: min-cut partitions fw/bwd,
            # activation_memory_budget decides save-vs-recompute globally
            # noinspection PyProtectedMember
            import torch._functorch.config as functorch_config

            # noinspection PyProtectedMember
            from torch._functorch.partitioners import min_cut_rematerialization_partition

            # noinspection PyProtectedMember
            import torch._inductor.config as inductor_config_

            # comprehensive STRIDE padding would re-pad the halves' output strides
            # (saved activations are not user-visible outputs here, so their layout is free);
            # without torch.compile's fw->bwd stride negotiation (absent under raw aot_function)
            # the bwd input stride asserts then fail.
            # Shape padding (pad_mm, the joint-graph pass in partition_fn) stays on.
            inductor_config_.comprehensive_padding = False
            if self._activation_memory_budget is not None:
                functorch_config.activation_memory_budget = float(self._activation_memory_budget)
            if self._aggressive_recomputation:
                functorch_config.aggressive_recomputation = True
                # aggressive_recomputation lifts all ban heuristics EXCEPT reductions;
                # the big loq savers (log-softmax outputs, normalizations) are reductions -- lift that too
                functorch_config.ban_recompute_reductions = False

            inner_backend_ = backend
            if inner_backend_ is compile_fx:
                # compile each half with compile_fx_inner, like torch.compile does after partitioning.
                # The full compile_fx wraps the half in ANOTHER aot_module_simplified layer,
                # whose positional-args call frame PINS all inputs for the entire call:
                # the generated code's progressive input freeing (args.clear + per-arg del)
                # then never frees anything,
                # and the bwd runs with the whole saved set resident
                # (measured: all 46.45 GiB saved still active at 90% through the bwd; OOM at bs200k).
                # compile_fx_inner returns the boxed CompiledFxGraph directly, no extra layer.
                # noinspection PyProtectedMember
                from torch._inductor.compile_fx import compile_fx_inner

                inner_backend_ = compile_fx_inner

            def fw_compiler_logged(gm, example_inputs):
                """log the saved-activation set (what the budget solver kept), then compile"""
                out_args = [n for n in gm.graph.nodes if n.op == "output"][0].args[0]
                total_bytes = 0
                items = []
                for a in out_args[self._compiled_n_loss_outs + 1 :]:
                    v = getattr(a, "meta", {}).get("val") if a is not None else None
                    if v is not None and hasattr(v, "numel"):
                        nb = v.numel() * v.element_size()
                        total_bytes += nb
                        items.append((nb, tuple(v.shape), tuple(v.stride()), str(v.dtype)))
                print(
                    f"torch_cuda_graph partitioned fw: {len(items)} saved activations,"
                    f" {total_bytes / 2**30:.2f} GiB (traced sizes)",
                    flush=True,
                )
                for nb, shape, stride, dtype in sorted(items, reverse=True)[:15]:
                    print(f"  saved {nb / 2**20:9.1f} MiB {dtype} {shape} stride {stride}", flush=True)
                return inner_backend_(gm, example_inputs)

            backend = fw_compiler_logged
            # no make_boxed_compiler here: compile_fx_inner's CompiledFxGraph is already boxed;
            # make_boxed_compiler would re-wrap it positionally and break the calling convention
            # (TypeError: CompiledFxGraph.__call__ takes 2 positional arguments)

            def partition_fn(gm, joint_inputs, **kwargs):
                """
                Like the torch.compile path: the Inductor joint-graph passes
                (incl. pad_mm shape padding) run on the JOINT graph before partitioning,
                so fw and bwd split a graph whose metas already carry the padded layouts
                and the saved-activation strides agree across the graph boundary.
                (Splitting the unpadded joint instead, with compile_fx padding only
                within the fw, hits the compiled bwd's input stride asserts;
                the torch.compile fw->bwd stride negotiation does not function
                under raw aot_function -- repro-verified, torch 2.7.)
                The halves are then compiled with compile_fx_inner (no joint-pass re-run), like torch.compile.
                """
                # noinspection PyProtectedMember
                from torch._inductor.fx_passes.joint_graph import joint_graph_passes

                joint_graph_passes(gm)
                fw_module, bw_module = min_cut_rematerialization_partition(gm, joint_inputs, **kwargs)
                return fw_module, bw_module

            # the full compile_fx applied the Inductor decomp table in its inner aot layer;
            # with compile_fx_inner halves the OUTER trace must decompose
            # (else e.g. aten.floor_divide reaches lowering: "both a fallback and a decomp")
            # noinspection PyProtectedMember
            from torch._inductor.decomposition import select_decomp_table

            # data/lens buffers + step tensor as explicit trace inputs (see _compiled_call_args)
            data_keys = sorted(self._data_bufs)
            lens_keys = sorted(self._lens_bufs)
            self._partitioned_buf_keys = (data_keys, lens_keys)
            n_params = len(orig_raws)

            def step_core_buf_inputs(all_raws):
                """step_core with the closure buffers swapped for the passed trace inputs"""
                bufs = all_raws[n_params:]
                data_ph = bufs[: len(data_keys)]
                lens_ph = bufs[len(data_keys) : len(data_keys) + len(lens_keys)]
                saved = (dict(self._data_bufs), dict(self._lens_bufs), self._step_t.raw_tensor)
                try:
                    for k, t in zip(data_keys, data_ph):
                        self._data_bufs[k] = t
                    for k, t in zip(lens_keys, lens_ph):
                        self._lens_bufs[k] = t
                    self._step_t.raw_tensor = bufs[-1]
                    return step_core(all_raws[:n_params])
                finally:
                    self._data_bufs.update(saved[0])
                    self._lens_bufs.update(saved[1])
                    self._step_t.raw_tensor = saved[2]

            return aot_function(
                step_core_buf_inputs,
                fw_compiler=backend,
                bw_compiler=inner_backend_,
                partition_fn=partition_fn,
                decompositions=select_decomp_table(),
            )
        # NOTE: partition_fn / activation_memory_budget are IRRELEVANT in this (default) mode:
        # step_core computes the grads itself, so this is ONE inference-style graph,
        # never fw/bwd-partitioned; buffer lifetimes are Inductor memory planning.
        # (For the partitioned alternative see opts "partitioned".)
        if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 12):
            # torch >= 2.12: compile_fx's compat wrapper declares _boxed_call=True
            # but re-wraps an already-boxed args list, so the generated runner sees [[args]];
            # call it star-unpacked instead, while the shim stays boxed towards aot_function.
            _compile_fx_raw = backend

            def _compile_fx_call_unboxed(gm, example_inputs):
                """compile via compile_fx, call the result star-unpacked"""
                compiled = _compile_fx_raw(gm, example_inputs)

                def _call(args):
                    return compiled(*args)

                _call._boxed_call = True
                return _call

            backend = _compile_fx_call_unboxed
            # torch >= 2.12 also lifts closed-over tensors into runtime args of the generated code
            # instead of baking them as graph constants, and raw aot_function does not supply them;
            # pass the buffers as explicit trace inputs, like the partitioned mode above.
            data_keys = sorted(self._data_bufs)
            lens_keys = sorted(self._lens_bufs)
            self._partitioned_buf_keys = (data_keys, lens_keys)
            n_params = len(orig_raws)

            def step_core_buf_inputs_v212(all_raws):
                """step_core with the closure buffers swapped for the passed trace inputs"""
                bufs = all_raws[n_params:]
                data_ph = bufs[: len(data_keys)]
                lens_ph = bufs[len(data_keys) : len(data_keys) + len(lens_keys)]
                saved = (dict(self._data_bufs), dict(self._lens_bufs), self._step_t.raw_tensor)
                try:
                    for k, t in zip(data_keys, data_ph):
                        self._data_bufs[k] = t
                    for k, t in zip(lens_keys, lens_ph):
                        self._lens_bufs[k] = t
                    self._step_t.raw_tensor = bufs[-1]
                    return step_core(all_raws[:n_params])
                finally:
                    self._data_bufs.update(saved[0])
                    self._lens_bufs.update(saved[1])
                    self._step_t.raw_tensor = saved[2]

            return aot_function(step_core_buf_inputs_v212, fw_compiler=backend)
        return aot_function(step_core, fw_compiler=backend)

    def _compiled_call_args(self, raws: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        The compiled step's runtime inputs: the param raws,
        plus (partitioned mode, or any mode on torch >= 2.12) the data/lens buffers
        and the step tensor.
        These are closure state of step_core.
        aot_function bakes closed-over tensors as graph CONSTANTS,
        and compile_fx_inner constant-folds them
        (measured: the seq lens frozen at their trace-time values -> wrong losses).
        The full-compile_fx backend hid this by re-lifting closure tensors in its inner aot layer.
        The buffer objects are created once and refreshed in place, so identity is stable.
        """
        if self._partitioned_buf_keys is None:
            return raws
        data_keys, lens_keys = self._partitioned_buf_keys
        return (
            list(raws)
            + [self._data_bufs[k] for k in data_keys]
            + [self._lens_bufs[k] for k in lens_keys]
            + [self._step_buf]
        )

    def _ensure_compiled(self) -> Callable[[List[torch.Tensor]], tuple]:
        """build the compiled step once; the first call traces + Inductor-compiles + autotunes"""
        if self._compiled_fn is None:
            self._compiled_fn = self._make_compiled_step()
            # drop the warmup steps' result ctx before the first compiled run:
            # its retained tensors (~0.4 GiB) plus cached free blocks
            # count against a first-run peak
            # that is short by only a few hundred MiB at the loq bs200k bound
            self._ctx = None
            gc.collect()
            torch.cuda.empty_cache()
            raws = [p.raw_tensor for p in self._rf_params]
            with _allow_non_fake_inputs():
                outs = self._compiled_fn(self._compiled_call_args(raws))
                if self._partitioned:
                    # the bwd graph compiles on the first backward
                    for p in self._params:
                        p.grad.zero_()
                    outs[0].backward()
        return self._compiled_fn

    def _bind_grads(self, raws: List[torch.Tensor], outs: tuple):
        """rebind .grad to the compiled grad outputs (fresh full grads -- no accumulation, no zeroing)"""
        grad_outs = outs[self._compiled_n_loss_outs :]
        train_raws = [r for r in raws if r.requires_grad]
        assert len(train_raws) == len(grad_outs)
        for r, g in zip(train_raws, grad_outs):
            r.grad = g

    def _capture_compiled(self, graph: torch.cuda.CUDAGraph):
        """
        Trace + compile + autotune on real inputs (outside capture), then capture:
        compiled step, gradient rebinding, optional in-graph optimizer.
        The compiled outputs are capture-pool allocations with stable addresses,
        refreshed in place by each replay.
        """
        compiled = self._ensure_compiled()
        raws = [p.raw_tensor for p in self._rf_params]
        self._log_misaligned_inputs(raws)
        # plain warm run (in partitioned mode incl. backward: autotune + workspaces)
        outs = compiled(self._compiled_call_args(raws))
        if self._partitioned:
            for p in self._params:
                p.grad.zero_()
            outs[0].backward()
        torch.cuda.synchronize()
        # release the warm runs' cached blocks BEFORE capture:
        # capture allocates from the separate graph pool and cannot reuse them,
        # and DURING capture the allocator cannot cudaFree cached blocks either
        # (illegal under capture, so the usual free-and-retry rescue is disabled)
        # -- without this release, capture needs the step footprint TWICE
        del outs
        torch.cuda.empty_cache()
        with torch.cuda.graph(graph):
            if self._partitioned:
                # in-graph: backward ACCUMULATES into the static grads -> zero first
                for p in self._params:
                    p.grad.zero_()
            outs = compiled(self._compiled_call_args(raws))
            if self._partitioned:
                outs[0].backward()
            else:
                # host-side during capture: the grads are stable graph outputs
                self._bind_grads(raws, outs)
            if self._post_step is not None:
                self._post_step()  # in-graph: grad clip + optimizer step
        self._ctx = self._build_result_ctx(outs)

    def _build_result_ctx(self, outs: tuple) -> RunCtx:
        """
        Result-facing run ctx: scalar losses wrapping the compiled outputs
        (under capture these are stable graph outputs, refreshed in place by each replay).
        """
        ctx = RunCtx(stage="train_step", train_flag=True, step=self._step_t)
        # partitioned mode: outs[0] is the differentiable total, the loss outputs follow
        i = 1 if self._partitioned else 0
        for name, loss_traced, inv_is_tensor, inv_static in self._compiled_loss_meta:
            summed = Tensor(name, dims=(), dtype=str(outs[i].dtype).split(".")[-1])
            summed.raw_tensor = outs[i]
            i += 1
            inv = Tensor(name + ":inv_norm", dims=(), dtype="int64")
            if inv_is_tensor:
                inv.dtype = str(outs[i].dtype).split(".")[-1]
                inv.raw_tensor = outs[i]
                i += 1
            else:
                inv.raw_tensor = torch.tensor(inv_static, dtype=torch.int64)
            ctx.losses[name] = Loss(
                loss=summed,
                name=name,
                scale=loss_traced.scale,
                as_error=loss_traced.as_error,
                use_normalized_loss=loss_traced.use_normalized_loss,
                use_flatten_frames=loss_traced.use_flatten_frames,
                custom_inv_norm_factor=inv,
            )
        assert i == self._compiled_n_loss_outs + (1 if self._partitioned else 0)
        return ctx

    def _log_misaligned_inputs(self, raws: List[torch.Tensor]) -> None:
        """
        One-time diagnostic: Inductor CLONES every graph input whose data pointer
        is not 16-byte aligned (``copy_misaligned_inputs``), per call --
        both a memory and a per-step-time cost. Log the offenders loudly.
        """
        bad = [(i, tuple(r.shape), r.data_ptr() % 16) for i, r in enumerate(raws) if r.data_ptr() % 16 != 0]
        if bad:
            print(
                f"torch_cuda_graph WARNING: {len(bad)} misaligned compiled-graph inputs"
                f" (Inductor clones each per call): {bad[:10]}"
            )

    def _run_compiled_eager(self) -> RunCtx:
        """
        opts "capture": False: launch the compiled bound-shaped program eagerly (no graph).
        Outputs are fresh tensors each call -> grads/ctx rebound per step.
        """
        raws = [p.raw_tensor for p in self._rf_params]
        if self._compiled_n_calls == 0:
            self._log_misaligned_inputs(raws)
        self._compiled_n_calls += 1
        if self._partitioned:
            for p in self._params:
                p.grad.zero_()
        outs = self._compiled_fn(self._compiled_call_args(raws))
        if self._debug_nan_dump_inputs:
            self._check_finite_dump_inputs(outs)
        if self._partitioned:
            outs[0].backward()
        else:
            self._bind_grads(raws, outs)
        if self._post_step is not None:
            self._post_step()  # grad clip + optimizer step (plain eager here)
        self._ctx = self._build_result_ctx(outs)
        return self._ctx

    def run_train_step(
        self, extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]], *, global_train_step: int
    ) -> RunCtx:
        """
        Run one train step: copy the batch into the static buffers, then
        eager warmup / one-time capture (optionally of the Inductor-compiled step) / graph replay.
        Backward is included; the optimizer step is the caller's job (on the static grads),
        unless "capture_optimizer" puts it in-graph.

        :return: the run ctx holding the losses. Under replay this is the capture-time ctx;
            its loss tensors are the static outputs, refreshed by the replay.
        """
        if self._debug_nan_dump_inputs:
            self._last_extern_data_raw = extern_data_raw
        self._copy_in(extern_data_raw)
        self._step_buf.fill_(global_train_step)
        if self._graph is not None:
            self._graph.replay()
            return self._ctx
        if self._compiled_fn is not None:  # "capture": False mode, post-warmup
            return self._run_compiled_eager()
        if self._n_eager < self.warmup_steps:
            self._n_eager += 1
            warmup_raw = extern_data_raw
            dummy_saved_lrs = None
            dummy_lr_opt = None
            if self.dummy_warmup:
                # The eager warmup exists only to materialize lazy state (optimizer moment buffers,
                # cudnn/cublas handles/workspaces). What it computes is irrelevant, but eager keeps
                # every intermediate and every autograd-saved tensor alive with no reuse planning,
                # so at real shapes it peaks far above the planned captured step (measured on loq
                # base: 61.6GB warmup vs 9.0GB replay), and THAT peak sizes the job's GPU.
                # A minimal batch makes it collapse to ~the param/optimizer footprint.
                # The params/optimizer state are restored afterwards (see _restore_after_dummy_warmup),
                # so the garbage gradients of these steps never reach the model.
                warmup_raw = self._make_dummy_extern_data_raw(extern_data_raw)
                if self._pre_dummy_warmup_buffers is None and self._get_buffers is not None:
                    # running stats (batch norm etc.) are updated in the FORWARD pass,
                    # so an lr of 0 cannot protect them: snapshot + restore (small, stats only)
                    self._pre_dummy_warmup_buffers = [b.detach().clone() for b in self._get_buffers()]
                opt = self._get_optimizer() if self._get_optimizer is not None else None
                if opt is not None and type(opt).__name__ in ("SGD", "Adam", "AdamW"):
                    # lr 0 makes the param update EXACTLY zero for these optimizers
                    # (every update term scales with lr, incl. the decoupled weight decay),
                    # so no param snapshot is needed -- a full extra param copy would be
                    # real memory at the warmup peak for large models. lr restored below;
                    # the moments/step counters the dummy step wrote are zeroed afterwards
                    # (_restore_after_dummy_warmup), only their EXISTENCE is kept.
                    dummy_lr_opt = opt
                    dummy_saved_lrs = []
                    for g in opt.param_groups:
                        lr = g["lr"]
                        if isinstance(lr, torch.Tensor):  # capturable: device-tensor lr
                            dummy_saved_lrs.append(lr.clone())
                            lr.fill_(0)
                        else:
                            dummy_saved_lrs.append(lr)
                            g["lr"] = 0.0
                elif self._pre_dummy_warmup_params is None:
                    # unknown optimizer: its update may not scale with lr -> full param snapshot
                    self._pre_dummy_warmup_params = [p.detach().clone() for p in self._params]
            # eager warmup on a non-default stream, see the module docstring
            self._eager_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._eager_stream):
                self._ctx = self._warmup_step_dynamic(warmup_raw, global_train_step=global_train_step)
            torch.cuda.current_stream().wait_stream(self._eager_stream)
            if dummy_saved_lrs is not None:
                for g, lr in zip(dummy_lr_opt.param_groups, dummy_saved_lrs):
                    if isinstance(g["lr"], torch.Tensor):
                        g["lr"].copy_(lr)
                    else:
                        g["lr"] = lr
            if self.dummy_warmup and self._n_eager >= self.warmup_steps:
                self._restore_after_dummy_warmup()
            return self._ctx
        if not self._compile:
            # side-stream warmup (kernel/cudnn warmup; each _step call is cold w.r.t. dim/layout caches)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(s)
        # with warmup_steps 0 there was no real optimizer step yet: create the lazy state
        # (and the param grads) explicitly -- no-op if a warmup step already did
        self._materialize_optimizer_state()
        torch.cuda.synchronize()
        # release the warmup's cached blocks before the compiled program / capture
        # allocates its own (bound-sized, differently-shaped) pool -- see _warmup_step_dynamic
        torch.cuda.empty_cache()
        try:
            if self._compile and not self._capture_graph:
                self._ensure_compiled()
                return self._run_compiled_eager()
            graph = torch.cuda.CUDAGraph()
            if self._compile:
                # the compiled step warms itself (trace + compile + autotune, pre-capture)
                self._capture_compiled(graph)
            else:
                with torch.cuda.graph(graph):
                    # cold capture: this _step call recomputes every dim/layout cache IN-graph
                    self._ctx = self._step()
        except torch.OutOfMemoryError:
            self._write_memory_snapshot()
            raise
        self._graph = graph
        self._log_graph_pool_size(graph)
        # The graph pool's blocks were LIVE allocations during the capture (thus inside
        # max_memory_allocated) and are freed-but-retained afterwards (thus outside it).
        # Reset the peak here so the reported usage (allocated + pool, see the engine)
        # counts the pool exactly once, in every epoch.
        torch.cuda.reset_peak_memory_stats()
        # capture only RECORDS the kernels; replay now to actually compute this batch
        graph.replay()
        return self._ctx

    @staticmethod
    def _log_graph_pool_size(graph: torch.cuda.CUDAGraph) -> None:
        """
        One-time report of the graph's PRIVATE memory pool: its blocks are freed as tensors
        after capture (they leave max_memory_allocated) but stay reserved for replay, so the
        allocated stat alone under-reports the resident footprint by the whole pool
        (measured loq base: 9GB allocated vs ~46GB resident).
        Also registers the size for the engine's per-step memory log (allocated + pool).
        """
        pool_id = graph.pool()
        segs = torch.cuda.memory_snapshot()
        # segment_pool_id absent (older torch) -> .get() None never matches -> tot 0, no report
        tot = sum(s.get("total_size", 0) for s in segs if s.get("segment_pool_id") == pool_id)
        live = sum(s.get("allocated_size", 0) for s in segs if s.get("segment_pool_id") == pool_id)
        global _graph_pools_reserved
        _graph_pools_reserved = tot
        if tot:
            print(
                f"torch_cuda_graph: private pool {tot / 2**30:.2f} GiB reserved"
                f" ({live / 2**30:.2f} GiB live outputs/buffers)"
                f" -- the pool is INVISIBLE in (max_)memory_allocated",
                flush=True,
            )

    def _check_finite_dump_inputs(self, outs: tuple) -> None:
        """see the debug_nan_dump_inputs option; syncs every step, debug only"""
        bad = None
        for i, o in enumerate(outs[: self._compiled_n_loss_outs]):
            if isinstance(o, torch.Tensor) and not torch.isfinite(o).all():
                bad = i
                break
        if bad is None:
            return
        fn = f"nan-step-inputs-call{self._compiled_n_calls}.pt"
        torch.save(
            {
                "call_idx": self._compiled_n_calls,
                "bad_loss_out_idx": bad,
                "loss_outs": [
                    o.detach().cpu() if isinstance(o, torch.Tensor) else o for o in outs[: self._compiled_n_loss_outs]
                ],
                "extern_data_raw": {
                    k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in (self._last_extern_data_raw or {}).items()
                },
            },
            fn,
        )
        raise Exception(
            f"torch_cuda_graph debug_nan_dump_inputs:"
            f" non-finite loss output {bad} at compiled call {self._compiled_n_calls},"
            f" batch dumped to {fn}"
        )

    def _write_memory_snapshot(self) -> None:
        """dump the recorded CUDA allocator history (see the dump_memory_snapshot option)"""
        if not self._dump_memory_snapshot:
            return
        import logging

        fn = "cuda-memory-snapshot.pickle"
        # noinspection PyProtectedMember
        torch.cuda.memory._dump_snapshot(fn)
        logging.getLogger("returnn").warning("torch_cuda_graph: CUDA memory snapshot dumped to %s", fn)
