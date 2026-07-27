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
        "warmup_steps": 2,              # eager steps before capture
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
import numpy
import torch

from returnn.util.basic import CollectionReadCheckCovered
from returnn.tensor import Tensor, TensorDict, Dim
import returnn.frontend as rf
from returnn.frontend.run_ctx import RunCtx, Loss

# noinspection PyProtectedMember
from ..data.extern_data import get_batch_dim_from_extern_data, _get_dyn_dims_from_extern_data


__all__ = ["GraphCapturedTrainStep"]


_inductor_workarounds_applied = False


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
    _inductor_workarounds_applied = True


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
        rf_params: Optional[List[rf.Parameter]] = None,
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
        """
        assert str(device).startswith("cuda"), f"torch_cuda_graph requires a cuda device, got {device!r}"
        opts = CollectionReadCheckCovered(opts)  # catch unknown (e.g. typo'd) option keys, see below
        self.batch_size_bound = int(opts["batch_size_bound"])
        self.dim_capacity: Dict[str, int] = dict(opts["dim_capacity"])
        self.warmup_steps = int(opts.get("warmup_steps", 2))
        self._device = torch.device(device)
        self._float_dtype = float_dtype
        self._extern_data_template = extern_data_template
        self._run_step = run_step
        self._post_step = post_step if opts.get("capture_optimizer", False) else None
        self._params = params
        self._compile = bool(opts.get("compile", False))
        self._capture_graph = bool(opts.get("capture", True))
        assert self._capture_graph or self._compile, 'torch_cuda_graph: "capture": False requires "compile": True'
        self._compiled_fn: Optional[Callable[[List[torch.Tensor]], tuple]] = None
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
        if self._batch_dim.dimension is None:
            # The copy-in always fills the batch up to the bound (zero-length padding seqs),
            # so in this regime the batch dim IS static -- make it so.
            # (A process-wide template-dim mutation, like the declared capacities.)
            self._batch_dim.size = self.batch_size_bound
            self._batch_dim.capacity = self.batch_size_bound
            self._batch_dim.dyn_size_ext = None
        assert self._batch_dim.dimension == self.batch_size_bound
        self._data_bufs: Dict[str, torch.Tensor] = {}
        self._pinned_bufs: Dict[str, torch.Tensor] = {}  # host staging, see _copy_in
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
        for k, data in extern_data_template.data.items():
            for i, dim in enumerate(data.dims):
                if i >= 1 and dim.dimension is None and dim is not self._batch_dim:
                    assert k in self.dim_capacity, (
                        f"torch_cuda_graph: dim_capacity for data key {k!r} required (dyn dim {dim})"
                    )
                    dim.capacity = self.dim_capacity[k]

    def _get_data_buf(self, k: str, raw: torch.Tensor, packed: Optional[Dict[str, int]]) -> torch.Tensor:
        buf = self._data_bufs.get(k)
        if buf is not None:
            return buf
        data = self._extern_data_template.data[k]
        dtype = raw.dtype
        if dtype.is_floating_point and self._float_dtype:
            dtype = self._float_dtype
        if packed is not None:
            # packed collate: flat contiguous [total, ...feature]; bound = batch bound * capacity
            self._packed_opts[k] = dict(packed)
            shape = [self.batch_size_bound * self.dim_capacity[k]] + list(raw.shape[1:])
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
            else:
                assert all(a <= b for a, b in zip(raw.shape, buf.shape)), (
                    f"torch_cuda_graph: {k} shape {tuple(raw.shape)} exceeds bounds {tuple(buf.shape)}"
                )
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
                lens_pin[:n].copy_(size.to(dtype=torch.int32))
                lens_pin[n:].zero_()  # zero-length padding seqs
                lens_buf.copy_(lens_pin, non_blocking=True)

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

        _apply_inductor_workarounds()
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
                    train_raws = [t for t, tr in zip(raws, trainable) if tr]
                    grads = torch.autograd.grad(total.raw_tensor, train_raws, allow_unused=True)
                    grads = [g if g is not None else torch.zeros_like(t) for g, t in zip(grads, train_raws)]
                    self._compiled_loss_meta = loss_meta
                    self._compiled_n_loss_outs = len(outs)
                return tuple(t.detach() for t in outs + grads)
            finally:
                for p_, r0 in zip(rf_params, orig_raws):
                    p_.raw_tensor = r0

        return aot_function(step_core, fw_compiler=compile_fx)

    def _ensure_compiled(self) -> Callable[[List[torch.Tensor]], tuple]:
        """build the compiled step once; the first call traces + Inductor-compiles + autotunes"""
        if self._compiled_fn is None:
            self._compiled_fn = self._make_compiled_step()
            raws = [p.raw_tensor for p in self._rf_params]
            with _allow_non_fake_inputs():
                self._compiled_fn(raws)
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
        compiled(raws)  # plain warm run
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            outs = compiled(raws)
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
        i = 0
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
        assert i == self._compiled_n_loss_outs
        return ctx

    def _run_compiled_eager(self) -> RunCtx:
        """
        opts "capture": False: launch the compiled bound-shaped program eagerly (no graph).
        Outputs are fresh tensors each call -> grads/ctx rebound per step.
        """
        raws = [p.raw_tensor for p in self._rf_params]
        outs = self._compiled_fn(raws)
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
        self._copy_in(extern_data_raw)
        self._step_buf.fill_(global_train_step)
        if self._graph is not None:
            self._graph.replay()
            return self._ctx
        if self._compiled_fn is not None:  # "capture": False mode, post-warmup
            return self._run_compiled_eager()
        if self._n_eager < self.warmup_steps:
            self._n_eager += 1
            # eager warmup on a non-default stream, see the module docstring
            self._eager_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._eager_stream):
                self._ctx = self._step()
            torch.cuda.current_stream().wait_stream(self._eager_stream)
            return self._ctx
        if not self._compile:
            # side-stream warmup (kernel/cudnn warmup; each _step call is cold w.r.t. dim/layout caches)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
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
        self._graph = graph
        # capture only RECORDS the kernels; replay now to actually compute this batch
        graph.replay()
        return self._ctx
