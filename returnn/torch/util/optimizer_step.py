"""
Separately compiled and CUDA-graph-captured optimizer step.

The eager optimizer step launches many small kernels per param
(e.g. Muon: momentum, Newton-Schulz, update, for every matrix),
so it is often dominated by host-side kernel launch overhead, not by the arithmetic.
Here, the existing ``optimizer.step()`` is traced (AOT, no Dynamo) and compiled with Inductor
(fuses the ops), like the model step in :mod:`returnn.torch.util.graph_capture`,
and the compiled step is then captured into a CUDA graph and replayed (removes the per-kernel host launches).

This is independent of the model step: it runs after the normal grad computation,
the grad reduction (e.g. ``reduce_type`` "grad_explicit" in distributed training)
and the grad clipping. So it also works where the whole-step graph with "capture_optimizer"
(see :mod:`returnn.torch.util.graph_capture`) cannot be used,
because something (like the grad reduction) must run between the grads and the update.

Config, e.g.::

    torch_optimizer_step = {
        "compile": True,  # trace + Inductor-compile the optimizer step (default True)
        "capture": True,  # CUDA-graph capture + replay the (compiled) step (default True)
        "dynamic_state_keys": ["step"],  # Python scalars in the optimizer state to keep as device tensors
    }

The trace runs the unchanged ``optimizer.step()`` on explicit inputs:
per param group the lr, per param (with grad) the param, its grad and its tensor state entries.
The in-place updates of params and state are input mutations of the traced function,
kept in the compiled graph (Inductor writes them in place).
Tensor-valued ``alpha`` / ``value`` arguments (e.g. ``p.add_(u, alpha=-lr)`` with a tensor lr)
are rewritten to explicit multiplications, see :class:`_TensorScalarArgsMode`,
so optimizers written for a Python-float lr (like Muon) work unchanged.

Every call of :func:`OptimizerStep.step` is exactly one real update.
The phases are: one plain eager step (creates the lazy optimizer state),
then one compiled step (the trace itself runs on fake tensors, then compile, autotune, update),
then the capture (records only) directly followed by a replay (updates),
and from then on only replays.

What would otherwise be frozen into the graph as a constant is kept as a device tensor, updated in place:

- The learning rate (each param group ``"lr"``).
- The optimizer state entries in ``dynamic_state_keys``, when they are Python scalars
  (e.g. a Python-int step counter ``state["step"] += 1`` of a custom optimizer).
  Any other Python scalar in the optimizer state is an error.

The checkpoint keeps the ordinary Python scalars for both (see :func:`OptimizerStep.state_dict_to_host_scalars`).

The graph records the addresses of all params, grads, state tensors and lrs.
When any of them is replaced (e.g. the grads were set to None, or the optimizer state was loaded),
the graph is recaptured (checked on every step). The compiled step takes all of them as inputs,
so it is only retraced when their shapes, dtypes or the state layout change.

Requirements (asserted): one CUDA device for all params, no closure, no grad scaler.
The set of params with grads must stay fixed.
Inductor compiles with ``emulate_precision_casts``, so low-precision intermediates are rounded as in eager.
Compiled arithmetic is still not guaranteed to be bitwise identical to the eager step.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Callable, Dict, List, Tuple
from collections import defaultdict
import torch

# noinspection PyProtectedMember
from torch.overrides import TorchFunctionMode

from returnn.log import log
from returnn.util.basic import CollectionReadCheckCovered
from .capture_lock import cuda_graph_capture

__all__ = ["OptimizerStep"]


class OptimizerStep:
    """
    Executes ``optimizer.step()``: eager, then compiled, then as a replayed CUDA graph.
    See the module docstring.
    """

    def __init__(self, *, optimizer: torch.optim.Optimizer, opts: Dict[str, Any]):
        """
        :param optimizer:
        :param opts: the ``torch_optimizer_step`` config dict, see the module docstring
        """
        opts = CollectionReadCheckCovered(opts)
        self._compile = bool(opts.get("compile", True))
        self._capture = bool(opts.get("capture", True))
        self._dynamic_state_keys = tuple(opts.get("dynamic_state_keys", ()))
        opts.assert_all_read()
        assert all(isinstance(k, str) for k in self._dynamic_state_keys), (
            f"torch_optimizer_step: dynamic_state_keys must be str, got {self._dynamic_state_keys!r}"
        )
        self._optimizer = optimizer
        devices = {p.device for group in optimizer.param_groups for p in group["params"]}
        assert len(devices) == 1, f"torch_optimizer_step: params on multiple devices {devices}"
        (self._device,) = devices
        assert self._device.type == "cuda" or not self._capture, (
            f"torch_optimizer_step: capture requires a CUDA device, got {self._device}"
        )
        self._eager_step_done = False
        self._compiled_fn: Optional[Callable[[List[torch.Tensor]], Any]] = None
        self._compiled_layout: Optional[Tuple[Any, ...]] = None  # see _layout
        self._compiled_warm = False  # the compiled step ran once (compile + autotune)
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._graph_inputs: Optional[List[torch.Tensor]] = None
        self._graph_signature: Optional[Tuple[int, ...]] = None
        self._num_traces = 0
        self._num_captures = 0
        # id -> Python type, of the device scalars created by _make_dynamic
        self._dynamic_scalars: Dict[int, type] = {}
        self._dynamic_scalars_refs: List[torch.Tensor] = []

    def step(self):
        """one optimizer update, like ``optimizer.step()``"""
        if self._graph is not None:
            if self._signature(self._inputs()) == self._graph_signature:
                self._graph.replay()
                return
            print("torch_optimizer_step: params, grads or optimizer state replaced, recapture", file=log.v3)
            self.invalidate()
        if not self._eager_step_done:
            # The lazy state is created by the first step, as the optimizer does it,
            # as a plain eager step (a real update).
            with _TensorScalarArgsMode():
                self._optimizer.step()
            self._eager_step_done = True
            return
        self._make_dynamic()
        if not self._compile:
            if not self._capture or not self._compiled_warm:
                # capture: one eager run before, for lazy init (e.g. cuBLAS workspaces) outside the capture
                with _TensorScalarArgsMode():
                    self._optimizer.step()
                self._compiled_warm = True
                return
            self._capture_graph(self._inputs())
            self._graph.replay()  # the capture only records, this is the update
            return
        inputs = self._inputs()
        layout = self._layout()
        if self._compiled_fn is None or layout != self._compiled_layout:
            self._compiled_fn = self._make_compiled_fn()
            self._compiled_layout = layout
            self._compiled_warm = False
        if not self._compiled_warm:
            # the first call traces (fake tensors, no update), compiles and autotunes, then updates;
            # all outside of any capture
            # noinspection PyProtectedMember
            import torch._inductor.config as inductor_config

            # Round low-precision intermediates as eager does: e.g. Muon's bf16 Newton-Schulz
            # otherwise stays in fp32 inside the fused kernels (measured param diff vs eager 7e-4, with this 1.5e-8)
            with inductor_config.patch(emulate_precision_casts=True):
                self._compiled_fn(inputs)
            self._compiled_warm = True
            return
        if not self._capture:
            self._compiled_fn(inputs)
            return
        self._capture_graph(inputs)
        self._graph.replay()  # the capture only records, this is the update

    def invalidate(self):
        """
        Drop the captured graph, e.g. after the optimizer state was loaded.
        The next step recaptures (and retraces, if the layout changed).
        """
        self._graph = None
        self._graph_inputs = None
        self._graph_signature = None

    def state_dict_to_host_scalars(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        :param state_dict: from ``optimizer.state_dict()``
        :return: shallow copy where the device scalars which were Python scalars originally
            (the lr, the ``dynamic_state_keys`` entries) are Python scalars again
        """
        res = dict(state_dict)
        res["param_groups"] = [
            {**g, "lr": float(g["lr"].item())} if id(g["lr"]) in self._dynamic_scalars else g
            for g in state_dict["param_groups"]
        ]
        res["state"] = {
            k: {
                k_: self._dynamic_scalars[id(v)](v.item()) if id(v) in self._dynamic_scalars else v
                for k_, v in s.items()
            }
            for k, s in state_dict["state"].items()
        }
        return res

    def _make_dynamic(self):
        """lr and the dynamic_state_keys to device tensors, in place in the optimizer"""
        for group in self._optimizer.param_groups:
            if not isinstance(group["lr"], torch.Tensor):
                group["lr"] = self._make_dynamic_scalar(float(group["lr"]))
        for state in self._optimizer.state.values():
            for k, v in list(state.items()):
                if isinstance(v, torch.Tensor):
                    if v.device != self._device:
                        raise ValueError(
                            f"torch_optimizer_step: optimizer state {k!r} on {v.device}, not on {self._device},"
                            " it would not be updated by the graph"
                            " (for the torch optimizers, use capturable=True)"
                        )
                    continue
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                if k not in self._dynamic_state_keys:
                    raise ValueError(
                        f"torch_optimizer_step: optimizer state {k!r} is a Python scalar {v!r},"
                        " which would be frozen into the compiled/captured step;"
                        f" add it to dynamic_state_keys (currently {self._dynamic_state_keys!r})"
                    )
                state[k] = self._make_dynamic_scalar(v)

    def _make_dynamic_scalar(self, value: Union[int, float]) -> torch.Tensor:
        """
        :return: device scalar tensor, registered to be converted back to its Python type for the checkpoint
        """
        dtype = torch.int64 if isinstance(value, int) else torch.float32
        t = torch.tensor(value, dtype=dtype, device=self._device)
        # by identity: the optimizer state_dict() refers to the same tensor objects
        self._dynamic_scalars[id(t)] = type(value)
        self._dynamic_scalars_refs.append(t)  # keeps the id unique while registered
        return t

    def _params_with_grad(self, group: Dict[str, Any]) -> List[torch.Tensor]:
        return [p for p in group["params"] if p.grad is not None]

    def _state_tensor_keys(self, p: torch.Tensor) -> List[str]:
        state = self._optimizer.state.get(p, {})
        return sorted(k for k, v in state.items() if isinstance(v, torch.Tensor))

    def _inputs(self) -> List[torch.Tensor]:
        """
        :return: the explicit inputs of the compiled step:
            per group: lr, then per param with grad: param, grad, tensor state entries (sorted keys)
        """
        res = []
        state = self._optimizer.state
        for group in self._optimizer.param_groups:
            res.append(group["lr"])
            for p in self._params_with_grad(group):
                res += [p, p.grad]
                res += [state[p][k] for k in self._state_tensor_keys(p)]
        return res

    def _layout(self) -> Tuple[Any, ...]:
        """
        :return: what the trace depends on besides the input addresses:
            the input structure, shapes, strides, dtypes, and the non-tensor state values
        """
        res = []
        state = self._optimizer.state
        for group in self._optimizer.param_groups:
            params = self._params_with_grad(group)
            res.append(len(params))
            for p in params:
                s = state.get(p, {})
                keys = self._state_tensor_keys(p)
                res.append(tuple(keys))
                res.append(tuple(sorted((k, v) for k, v in s.items() if k not in keys)))
        for t in self._inputs():
            res.append((tuple(t.shape), tuple(t.stride()), t.dtype, t.device))
        return tuple(res)

    @staticmethod
    def _signature(inputs: List[torch.Tensor]) -> Tuple[int, ...]:
        """addresses of everything the captured step reads or writes"""
        return tuple(t.data_ptr() for t in inputs)

    def _make_compiled_fn(self) -> Callable[[List[torch.Tensor]], Any]:
        """
        Trace the unchanged ``optimizer.step()`` on the explicit inputs (see :func:`_inputs`),
        with AOT tracing + Inductor codegen, no Dynamo, as the model step
        (see :func:`returnn.torch.util.graph_capture.GraphCapturedTrainStep._make_compiled_step`).
        """
        from functorch.compile import aot_function
        from .graph_capture import inductor_fw_compiler

        self._num_traces += 1
        print(f"torch_optimizer_step: compiling the optimizer step (trace {self._num_traces})...", file=log.v3)
        opt = self._optimizer
        groups = opt.param_groups
        # the real params per group, and the state keys per param, in the order of _inputs
        structure = [[(p, self._state_tensor_keys(p)) for p in self._params_with_grad(group)] for group in groups]

        def step_core(inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
            """optimizer.step() with lr, params, grads and state swapped for the given inputs"""
            saved_lrs = [group["lr"] for group in groups]
            saved_params = [group["params"] for group in groups]
            saved_state = opt.state
            it = iter(inputs)
            new_state = defaultdict(dict)
            try:
                for group, group_structure in zip(groups, structure):
                    group["lr"] = next(it)
                    new_params = []
                    for p, keys in group_structure:
                        p_, grad = next(it), next(it)
                        p_.grad = grad
                        s = dict(saved_state[p])  # non-tensor entries as they are
                        for k in keys:
                            s[k] = next(it)
                        new_state[p_] = s
                        new_params.append(p_)
                    group["params"] = new_params
                opt.state = new_state
                with _TensorScalarArgsMode():
                    opt.step()
            finally:
                for group, lr, params in zip(groups, saved_lrs, saved_params):
                    group["lr"] = lr
                    group["params"] = params
                opt.state = saved_state
            return ()  # all effects are input mutations

        # keep_inference_input_mutations: the updates stay in the compiled graph (in place),
        # instead of copies back into the inputs after it
        return aot_function(step_core, fw_compiler=inductor_fw_compiler(), keep_inference_input_mutations=True)

    def _capture_graph(self, inputs: List[torch.Tensor]):
        graph = torch.cuda.CUDAGraph()
        with cuda_graph_capture(graph):
            if self._compile:
                self._compiled_fn(inputs)
            else:
                with _TensorScalarArgsMode():
                    self._optimizer.step()
        self._graph = graph
        self._graph_inputs = inputs  # keeps the recorded addresses alive
        self._graph_signature = self._signature(inputs)
        self._num_captures += 1
        print(f"torch_optimizer_step: captured the optimizer step (capture {self._num_captures})", file=log.v3)


class _TensorScalarArgsMode(TorchFunctionMode):
    """
    Rewrites tensor-valued ``alpha`` / ``value`` arguments, which the in-place tensor methods only accept
    as Python numbers, to explicit multiplications, e.g. ``p.add_(u, alpha=lr)`` -> ``p.add_(u * lr)``.
    With a device-tensor lr (see :class:`OptimizerStep`), optimizers written for a Python-float lr then work unchanged.
    """

    _Alpha = {"add", "add_", "sub", "sub_"}
    _Value = {
        "addcmul": torch.Tensor.add,
        "addcmul_": torch.Tensor.add_,
        "addcdiv": torch.Tensor.add,
        "addcdiv_": torch.Tensor.add_,
    }

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        name = getattr(func, "__name__", None)
        if name in self._Alpha and isinstance(kwargs.get("alpha"), torch.Tensor):
            alpha = kwargs.pop("alpha")
            args = (args[0], args[1] * alpha) + tuple(args[2:])
        elif name in self._Value and isinstance(kwargs.get("value"), torch.Tensor):
            value = kwargs.pop("value")
            self_, tensor1, tensor2 = args[:3]
            update = tensor1 * tensor2 if name.startswith("addcmul") else tensor1 / tensor2
            func, args = self._Value[name], (self_, update * value)
        return func(*args, **kwargs)
