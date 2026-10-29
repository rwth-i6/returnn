"""
Exports a RF/PT module via torch.export and compiles it into an AOTInductor package (.pt2),
which can be loaded in C++ via torch::inductor::AOTIModelPackageLoader, e.g. by RASR.

Same conventions as torch_export_to_onnx.py:
the config needs `get_model()`, `forward_step()`, `extern_data` and `model_outputs`,
and sequence lengths are extra inputs/outputs named "<name>:size<axis>".

The package only has positional inputs and outputs, ordered by --input_names and --output_names.
--out_io_spec writes a JSON file with index, shape, dtype and dynamic axes of all inputs and outputs.
--io_spec_names renames entries there, e.g. to the names RASR expects.

All dynamic dims, including batch, are exported as dynamic (torch.export.Dim.DYNAMIC by default).
--dynamic_dims sets options per dim (key: dim name or "<input name>:<axis>"):
"min", "max", "example" (size used for tracing), and "factor"/"offset" to get `factor * base + offset`.
--static_batch exports with a fixed batch size of 1.
torch.export traces with batch size >= 2, which can break batch size 1. --check tests this.

--state_info is a JSON list of dicts with "name", "input", "output", "state_kind" and "layer",
and adds a "states" entry to the I/O spec. State inputs and outputs must be consecutive and in the same order.

The package expects contiguous inputs, strides are not checked.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy
import torch

import _setup_returnn_env  # noqa
import returnn.__main__ as rnn
import returnn.frontend as rf
import returnn.util.basic as util
from returnn.config import Config
from returnn.log import log
from returnn.tensor import Dim, Tensor, TensorDict, batch_dim
from returnn.tensor.utils import tensor_dict_fill_random_numpy_
from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_
from returnn.torch.frontend.bridge import RFModuleAsPTModule


config = None  # type: Optional[Config]


def init(config_filename: str, checkpoint: str, log_verbosity: int, device: str):
    """
    :param config_filename: Filename to config file.
    :param checkpoint: Filename to the trained model.
    :param log_verbosity: 5 for all seqs (default: 4)
    :param device:
    """
    assert os.path.exists(checkpoint), "The specified checkpoint doesn't exist."
    rnn.init_better_exchook()
    assert os.path.exists(config_filename), "The specified config doesn't exist."
    print("Using config file %r." % config_filename)
    rnn.init_config(
        config_filename=config_filename,
        extra_updates={
            "log": None,
            "log_verbosity": log_verbosity,
            "task": __file__,  # just extra info for the config
            "device": device,
        },
    )
    global config
    config = rnn.config
    rnn.init_log()
    print("RETURNN frontend module to AOTInductor conversion.", file=log.v1)
    rnn.returnn_greeting()
    config.typed_dict.setdefault("backend", "torch")
    rnn.init_backend_engine()
    assert util.BackendEngine.is_torch_selected(), "For now only the torch backend is supported."
    rnn.init_faulthandler()


class ForwardModule(torch.nn.Module):
    """
    forward_step as module with positional inputs and outputs.
    """

    def __init__(
        self,
        *,
        pt_module: torch.nn.Module,
        forward_model: Any,
        forward_step: Callable,
        extern_data: TensorDict,
        model_outputs: TensorDict,
        input_names: List[str],
        input_dtypes: Dict[str, torch.dtype],
        output_names: Optional[List[str]] = None,
        epoch: int,
        step: int,
    ):
        """
        :param pt_module: module which holds the parameters
        :param forward_model: model passed to forward_step (RF module or `pt_module`)
        :param forward_step:
        :param extern_data: template
        :param model_outputs:
        :param input_names:
        :param input_dtypes: dtypes as expected by forward_step
        :param output_names:
        :param epoch:
        :param step:
        """
        super().__init__()
        self.model = pt_module
        self.forward_model = forward_model
        self.forward_step_func = forward_step
        self.extern_data = extern_data
        self.model_outputs = model_outputs
        self.input_names = input_names
        self.input_dtypes = input_dtypes
        self.output_names = output_names
        self.epoch = epoch
        self.step = step

    def forward(self, *inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        :return: outputs, ordered by output_names
        """
        outputs = self.forward_dict(*inputs)
        return tuple(outputs[name] for name in self.output_names)

    def forward_dict(self, *inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :return: all outputs marked in forward_step
        """
        rf.init_forward_step_run_ctx(expected_outputs=self.model_outputs, step=self.step, epoch=self.epoch)
        data = {name: x.to(self.input_dtypes[name]) for name, x in zip(self.input_names, inputs)}
        extern_data = self.extern_data.copy_template()
        extern_data.assign_from_raw_tensor_dict_(data, with_scalar_dyn_sizes=False, duplicate_dims_are_excluded=True)
        _assign_scalar_dyn_dims(extern_data)
        self.forward_step_func(model=self.forward_model, extern_data=extern_data)
        rf.get_run_ctx().check_outputs_complete()
        return rf.get_run_ctx().outputs.as_raw_tensor_dict(include_scalar_dyn_sizes=False)


def _assign_scalar_dyn_dims(extern_data: TensorDict):
    """
    Like in tools/torch_export_to_onnx.py, but works with the symbolic shapes of torch.export.
    """
    for key, value in extern_data.data.items():
        if value.raw_tensor is None:
            continue
        for axis, dim in enumerate(value.dims):
            if dim.dyn_size_ext is None:
                continue
            if dim.dyn_size_ext.dims != ():
                continue
            if dim.dyn_size_ext.raw_tensor is not None:
                continue
            dim.dyn_size_ext.raw_tensor = torch.full(
                (),
                value.raw_tensor.shape[axis],
                dtype=getattr(torch, dim.dyn_size_ext.dtype),
                device=rf.get_default_dim_size_device(),
            )


def _split_size_name(raw_name: str) -> Optional[Tuple[str, int]]:
    """
    :return: (key, axis) for "<key>:size<axis>", else None
    """
    m = re.fullmatch(r"(.+):size(\d+)", raw_name)
    return (m.group(1), int(m.group(2))) if m else None


def _raw_input_dims(extern_data: TensorDict, raw_name: str) -> Sequence[Dim]:
    """
    :return: dims of the raw input tensor
    """
    if raw_name in extern_data.data:
        return extern_data.data[raw_name].dims
    key_axis = _split_size_name(raw_name)
    assert key_axis and key_axis[0] in extern_data.data, f"unknown input {raw_name!r}"
    return extern_data.data[key_axis[0]].dims[key_axis[1]].dyn_size_ext.dims


def _raw_input_dtype(extern_data: TensorDict, raw_name: str) -> torch.dtype:
    """
    :return: dtype as expected by forward_step
    """
    if raw_name in extern_data.data:
        return getattr(torch, extern_data.data[raw_name].dtype)
    key, axis = _split_size_name(raw_name)
    return getattr(torch, extern_data.data[key].dims[axis].dyn_size_ext.dtype)


class DynamicDims:
    """
    Settings for the dynamic dims, see module docstring.
    """

    def __init__(self, settings: Dict[str, Dict[str, int]], *, static_batch: bool = False):
        assert not (static_batch and "batch" in settings), "batch dim settings with static batch"
        self.settings = settings
        self.static_batch = static_batch
        self.torch_dims: Dict[Dim, Any] = {}

    def get_settings(self, dim: Dim, raw_name: str, axis: int) -> Optional[Dict[str, int]]:
        """
        :return: settings, or None if static
        """
        if dim.is_batch_dim():
            return None if self.static_batch else self.settings.get("batch", {})
        for key in (dim.name, f"{raw_name}:{axis}"):
            if key in self.settings:
                return self.settings[key]
        return {}

    def get_example_size(self, dim: Dim, raw_name: str, axis: int, *, size_offset: int = 0) -> int:
        """
        :param dim:
        :param raw_name:
        :param axis:
        :param size_offset: added to the example size
        :return: size in the example inputs
        """
        settings = self.get_settings(dim, raw_name, axis)
        if settings is None:  # static batch dim
            return 1
        base = settings.get("example", 2 if dim.is_batch_dim() else 100) + size_offset
        return settings.get("factor", 1) * base + settings.get("offset", 0)

    def get_torch_dim(self, dim: Dim, raw_name: str, axis: int):
        """
        :return: dim for torch.export, or None if static
        """
        settings = self.get_settings(dim, raw_name, axis)
        if settings is None:
            return None
        if not settings:
            return torch.export.Dim.DYNAMIC
        if dim not in self.torch_dims:
            base_name = re.sub(r"\W", "_", dim.name or f"{raw_name}_{axis}")
            factor, offset = settings.get("factor", 1), settings.get("offset", 0)
            kwargs = {k: settings[k] for k in ("min", "max") if k in settings}
            if factor == 1 and offset == 0:
                self.torch_dims[dim] = torch.export.Dim(base_name, **kwargs)
            else:
                self.torch_dims[dim] = factor * torch.export.Dim(f"{base_name}_base", **kwargs) + offset
        return self.torch_dims[dim]


def make_example_inputs(
    extern_data: TensorDict,
    dynamic_dims: DynamicDims,
    *,
    size_dtype: torch.dtype,
    device: str,
    size_offset: int = 0,
    batch_size: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Random inputs, where each dynamic dim has its example size.

    :param extern_data:
    :param dynamic_dims:
    :param size_dtype: dtype of the size inputs
    :param device:
    :param size_offset: added to the example sizes
    :param batch_size: overrides the batch size
    :return: raw tensor dict
    """
    extern_data.reset_content()
    sizes: Dict[Dim, int] = {}
    for name, tensor in extern_data.data.items():
        for axis, dim in enumerate(tensor.dims):
            if (dim.is_batch_dim() or dim.is_dynamic()) and dim not in sizes:
                if dim.is_batch_dim() and batch_size is not None:
                    sizes[dim] = batch_size
                else:
                    sizes[dim] = dynamic_dims.get_example_size(dim, name, axis, size_offset=size_offset)
    # Set sizes explicitly, batch first. The random fill keeps them.
    batch_size = next((size for dim, size in sizes.items() if dim.is_batch_dim()), 1)
    for dim, size in sorted(sizes.items(), key=lambda item: not item[0].is_batch_dim()):
        if dim.dyn_size_ext is None:
            if dim.is_batch_dim():
                dim.dyn_size_ext = Tensor("batch", dims=[], dtype="int32")
            else:
                dim.dyn_size_ext = Tensor(dim.name or "time", dims=[batch_dim], dtype="int32")
        shape = [batch_size if d.is_batch_dim() else d.dimension for d in dim.dyn_size_ext.dims]
        dim.dyn_size_ext.raw_tensor = numpy.full(shape, size, dtype=dim.dyn_size_ext.dtype)
    tensor_dict_fill_random_numpy_(extern_data)
    tensor_dict_numpy_to_torch_(extern_data)
    raw = extern_data.as_raw_tensor_dict(include_scalar_dyn_sizes=False, exclude_duplicate_dims=True)
    return {k: (v.to(size_dtype) if _split_size_name(k) else v).to(device) for k, v in raw.items()}


def check_outputs(ref_outputs, outputs, output_names: List[str], desc: str, *, rtol: float, atol: float):
    """
    Raises an exception if the outputs differ from the reference.
    """
    assert len(ref_outputs) == len(outputs), f"{desc}: got {len(outputs)} outputs, expected {len(ref_outputs)}"
    for name, ref, out in zip(output_names, ref_outputs, outputs):
        if ref.shape != out.shape or ref.dtype != out.dtype:
            raise RuntimeError(
                f"{desc}: output {name!r} has shape {tuple(out.shape)}/{out.dtype}, "
                f"expected {tuple(ref.shape)}/{ref.dtype}"
            )
        if ref.is_floating_point():
            max_diff = (ref - out).abs().max().item() if ref.numel() > 0 else 0.0
            if not torch.allclose(ref, out, rtol=rtol, atol=atol):
                raise RuntimeError(f"{desc}: output {name!r} differs, max abs diff {max_diff}")
            print(f"{desc}: output {name!r} max abs diff {max_diff}", file=log.v3)
        elif not torch.equal(ref, out):
            raise RuntimeError(f"{desc}: output {name!r} differs")


def _bound_to_int(bound) -> Optional[int]:
    """
    :return: bound of a value range, None if unbounded
    """
    try:
        return int(bound)
    except (TypeError, ValueError, OverflowError, AttributeError):
        return None


def make_io_spec(
    *,
    exported: torch.export.ExportedProgram,
    input_names: List[str],
    output_names: List[str],
    io_spec_names: Dict[str, str],
    state_info: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    :return: I/O spec. Indices are positions in the flat inputs/outputs, dynamic axes have size -1 in "shape".
    """
    spec_name_by_raw_name = {raw_name: spec_name for spec_name, raw_name in io_spec_names.items()}
    unknown = set(spec_name_by_raw_name) - set(input_names) - set(output_names)
    assert not unknown, f"io_spec_names refers to unknown inputs/outputs {sorted(unknown)}"

    def _describe(raw_name: str, index: int, fake_tensor) -> Dict[str, Any]:
        shape, dynamic_axes = [], {}
        for axis, size in enumerate(fake_tensor.shape):
            if isinstance(size, torch.SymInt):
                shape.append(-1)
                expr = size.node.expr
                axis_info: Dict[str, Any] = {"expr": str(expr)}
                if expr in exported.range_constraints:
                    value_range = exported.range_constraints[expr]
                    axis_info["min"] = _bound_to_int(value_range.lower)
                    axis_info["max"] = _bound_to_int(value_range.upper)
                dynamic_axes[str(axis)] = axis_info
            else:
                shape.append(int(size))
        return {
            "index": index,
            "kind": "tensor",
            "name": raw_name,
            "shape": shape,
            "dtype": str(fake_tensor.dtype),
            "dynamic_axes": dynamic_axes,
        }

    placeholders = {node.name: node for node in exported.graph.nodes if node.op == "placeholder"}
    user_inputs = exported.graph_signature.user_inputs
    assert len(user_inputs) == len(input_names), (user_inputs, input_names)
    inputs = [
        _describe(raw_name, index, placeholders[node_name].meta["val"])
        for index, (raw_name, node_name) in enumerate(zip(input_names, user_inputs))
    ]
    output_node = next(node for node in exported.graph.nodes if node.op == "output")
    output_vals = [node.meta["val"] for node in output_node.args[0]]
    outputs = [_describe(raw_name, index, val) for index, (raw_name, val) in enumerate(zip(output_names, output_vals))]

    spec_inputs = {spec_name_by_raw_name.get(entry["name"], entry["name"]): entry for entry in inputs}
    spec_outputs = {spec_name_by_raw_name.get(entry["name"], entry["name"]): entry for entry in outputs}

    if state_info:
        assert "states" not in spec_inputs and "states" not in spec_outputs, "'states' is a reserved name"
        state_input_indices = [input_names.index(state["input"]) for state in state_info]
        state_output_indices = [output_names.index(state["output"]) for state in state_info]
        for desc, indices in (("inputs", state_input_indices), ("outputs", state_output_indices)):
            first = indices[0]
            assert indices == list(range(first, first + len(indices))), (
                f"state {desc} must be consecutive and in the order of state_info, got positions {indices}"
            )
        input_entries, output_entries = [], []
        for state_index, state in enumerate(state_info):
            state_input = inputs[state_input_indices[state_index]]
            state_output = outputs[state_output_indices[state_index]]
            assert state_input["dtype"] == state_output["dtype"], f"dtype mismatch for state {state['name']!r}"
            input_entries.append(
                {
                    "name": state["name"],
                    "state_index": state_index,
                    "state_kind": state["state_kind"],
                    "layer": state["layer"],
                    "shape": state_input["shape"],
                    "dtype": state_input["dtype"],
                }
            )
            output_entries.append(
                {"state_index": state_index, "shape": state_output["shape"], "dtype": state_output["dtype"]}
            )
        spec_inputs["states"] = {"arg_index": state_input_indices[0], "entries": input_entries}
        spec_outputs["states"] = {"output_index": state_output_indices[0], "entries": output_entries}

    return {
        "inputs": spec_inputs,
        "outputs": spec_outputs,
        "flat_inputs": inputs,
        "flat_outputs": outputs,
        "range_constraints": {
            str(symbol): {"min": _bound_to_int(value_range.lower), "max": _bound_to_int(value_range.upper)}
            for symbol, value_range in exported.range_constraints.items()
        },
    }


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "config",
        type=str,
        help="Filename to config file. Must have `get_model()` and `forward_step()`.",
    )
    parser.add_argument("checkpoint", type=str, help="PyTorch checkpoint (.pt) as saved by RETURNN.")
    parser.add_argument("--out_aoti_package", type=str, help="Filename of the AOTInductor package (.pt2).")
    parser.add_argument("--out_exported_program", type=str, help="Filename of the serialized ExportedProgram (.pt2).")
    parser.add_argument("--out_io_spec", type=str, help="Filename of the JSON I/O specification.")
    parser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' (default) or 'gpu'.")
    parser.add_argument("--input_names", type=str, help="Comma-separated list of input names, defines the order.")
    parser.add_argument("--output_names", type=str, help="Comma-separated list of output names, defines the order.")
    parser.add_argument("--io_spec_names", type=str, help="JSON dict: I/O spec name -> input/output name.")
    parser.add_argument("--dynamic_dims", type=str, help="JSON dict: dim name -> dict with dim settings.")
    parser.add_argument("--state_info", type=str, help="JSON list of state descriptions.")
    parser.add_argument("--static_batch", action="store_true", help="Export with static batch size 1.")
    parser.add_argument("--size_dtype", type=str, default="int64", help="dtype of the size inputs (default: int64).")
    parser.add_argument("--check", action="store_true", help="Compare exported/compiled model to the eager model.")
    parser.add_argument("--check_rtol", type=float, default=1e-4, help="Relative tolerance for --check.")
    parser.add_argument("--check_atol", type=float, default=1e-4, help="Absolute tolerance for --check.")
    args = parser.parse_args()
    assert args.out_aoti_package or args.out_exported_program, "need --out_aoti_package or --out_exported_program"

    assert tuple(int(x) for x in torch.__version__.split(".")[:2]) >= (2, 6), (
        f"needs PyTorch >= 2.6, got {torch.__version__}"
    )
    device = {"gpu": "cuda"}.get(args.device, args.device)
    init(config_filename=args.config, checkpoint=args.checkpoint, log_verbosity=args.verbosity, device=device)

    io_spec_names: Dict[str, str] = json.loads(args.io_spec_names) if args.io_spec_names else {}
    dynamic_dims = DynamicDims(
        json.loads(args.dynamic_dims) if args.dynamic_dims else {}, static_batch=args.static_batch
    )
    state_info: List[Dict[str, Any]] = json.loads(args.state_info) if args.state_info else []
    size_dtype = getattr(torch, args.size_dtype)

    model_outputs_dict = config.typed_value("model_outputs")
    assert model_outputs_dict is not None, (
        "The specified config needs to have explicit model outputs. Please define `model_outputs` in your config."
    )
    model_outputs = TensorDict()
    model_outputs.update(model_outputs_dict, auto_convert=True)

    loaded_checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
    epoch = loaded_checkpoint["epoch"]
    step = loaded_checkpoint["step"]

    rf.init_forward_step_run_ctx(expected_outputs=model_outputs, step=step, epoch=epoch)
    rf.set_random_seed(42)

    get_model_func = config.typed_value("get_model")
    assert get_model_func, "get_model() isn't specified in the config passed as a parameter."
    model = get_model_func(epoch=epoch, step=step, **util.get_fwd_compat_kwargs())
    forward_step_func = config.typed_value("forward_step")
    assert forward_step_func is not None, "forward_step() must be defined in the config."

    if isinstance(model, rf.Module):
        pt_module = RFModuleAsPTModule(model)
    else:
        assert isinstance(model, torch.nn.Module), (
            "The module returned by get_model() isn't a returnn.frontend.Module or a torch.nn.Module."
        )
        pt_module = model
    pt_module.load_state_dict(loaded_checkpoint["model"])
    pt_module.to(device)
    pt_module.eval()

    extern_data_dict = config.typed_value("extern_data")
    extern_data = TensorDict()
    extern_data.update(extern_data_dict, auto_convert=True)
    for k, v in list(extern_data.data.items()):
        if not v.available_for_inference:
            del extern_data.data[k]

    def _make_inputs(size_offset: int = 0, batch_size: Optional[int] = None) -> Dict[str, torch.Tensor]:
        return make_example_inputs(
            extern_data,
            dynamic_dims,
            size_dtype=size_dtype,
            device=device,
            size_offset=size_offset,
            batch_size=batch_size,
        )

    example_inputs = _make_inputs()
    if args.input_names:
        input_names = args.input_names.split(",")
        assert set(input_names) == set(example_inputs.keys()), (
            f"mismatch between input_names {input_names} and extern_data keys {list(example_inputs.keys())}"
        )
    else:
        input_names = list(example_inputs.keys())
    example_args = tuple(example_inputs[name] for name in input_names)

    fwd_module = ForwardModule(
        pt_module=pt_module,
        forward_model=model,
        forward_step=forward_step_func,
        extern_data=extern_data.copy_template(),
        model_outputs=model_outputs,
        input_names=input_names,
        input_dtypes={name: _raw_input_dtype(extern_data, name) for name in input_names},
        epoch=epoch,
        step=step,
    )
    with torch.no_grad():
        available_outputs = fwd_module.forward_dict(*example_args)
    if args.output_names:
        output_names = args.output_names.split(",")
        unknown = [name for name in output_names if name not in available_outputs]
        assert not unknown, f"unknown output names {unknown}, available: {list(available_outputs.keys())}"
    else:
        output_names = list(available_outputs.keys())
    fwd_module.output_names = output_names

    dynamic_shapes = []
    for name in input_names:
        axes = {}
        for axis, dim in enumerate(_raw_input_dims(extern_data, name)):
            if dim.is_batch_dim() or dim.is_dynamic():
                torch_dim = dynamic_dims.get_torch_dim(dim, name, axis)
                if torch_dim is not None:
                    axes[axis] = torch_dim
        dynamic_shapes.append(axes or None)

    print("*** Input names:", input_names, file=log.v1)
    print("*** Output names:", output_names, file=log.v1)
    print(
        "*** Input dims:",
        {name: [str(d) for d in _raw_input_dims(extern_data, name)] for name in input_names},
        file=log.v1,
    )
    print("*** Dynamic shapes:", dict(zip(input_names, dynamic_shapes)), file=log.v1)

    with torch.no_grad():
        # dynamic_shapes of the *inputs arg
        exported = torch.export.export(fwd_module, args=example_args, dynamic_shapes={"inputs": tuple(dynamic_shapes)})
    print("*** Range constraints:", exported.range_constraints, file=log.v1)

    if args.out_exported_program:
        torch.export.save(exported, args.out_exported_program)
        print("*** Saved exported program:", args.out_exported_program, file=log.v1)

    aoti_model = None
    if args.out_aoti_package:
        torch._inductor.aoti_compile_and_package(exported, package_path=args.out_aoti_package)
        print("*** Saved AOTInductor package:", args.out_aoti_package, file=log.v1)
        aoti_model = torch._inductor.aoti_load_package(args.out_aoti_package)

    if args.check:
        # Also check other sizes and batch size 1, to catch wrong specializations.
        check_models: List[Tuple[str, Callable]] = [("exported program", exported.module())]
        if aoti_model is not None:
            check_models.append(("AOTInductor package", aoti_model))
        check_input_settings = [(0, None), (7, None)] + ([] if args.static_batch else [(0, 1)])
        for size_offset, batch_size in check_input_settings:
            check_inputs = _make_inputs(size_offset, batch_size)
            check_args = tuple(check_inputs[name] for name in input_names)
            with torch.no_grad():
                ref_outputs = fwd_module(*check_args)
                for model_name, check_model in check_models:
                    desc = f"{model_name} (size offset {size_offset}, batch size {batch_size or 'default'})"
                    outputs = check_model(*check_args)
                    check_outputs(ref_outputs, outputs, output_names, desc, rtol=args.check_rtol, atol=args.check_atol)
                    print(f"*** Check of {desc} passed.", file=log.v1)

    if args.out_io_spec:
        io_spec = make_io_spec(
            exported=exported,
            input_names=input_names,
            output_names=output_names,
            io_spec_names=io_spec_names,
            state_info=state_info,
        )
        io_spec.update(
            {
                "checkpoint": os.path.realpath(args.checkpoint),
                "exported_program": args.out_exported_program,
                "aot_package": args.out_aoti_package,
            }
        )
        with open(args.out_io_spec, "wt") as f:
            json.dump(io_spec, f, indent=2)
        print("*** Saved I/O specification:", args.out_io_spec, file=log.v1)


if __name__ == "__main__":
    main()
