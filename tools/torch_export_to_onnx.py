"""
Converts a RF/PT module to ONNX.

For that, it uses `get_model(*, epoch, step, **_other_kwargs)` which must be available in the config
and creates dummy data to be forwarded to the model.

Since `get_model(...)` can return either a torch.nn.Module or a rf.Module, both cases are handled.

`extern_data` defines the inputs.
  - It strips away all entries with available_for_inference=False.
  - Dynamic dimensions with dyn_size_ext being a scalar are excluded.
  - If dynamic dims occur multiple times, only the first occurence is kept.
    Note that the order depends on the order of the dict, as it is defined (not sorted alphabetically).

`model_outputs` defines the output dimensions.
  - Dynamic dimensions with dyn_size_ext being a scalar are excluded.

For RASR, we usually expect that the input and output are always of the format [B, T, D] (batch, time, dim).
We distinguish three cases on RASR side:

- Input without sequence length are provided, output also without sequence lengths:
    Define the input time dimension in `extern_data` to have `dyn_size_ext` being a scalar,
    and same for `model_outputs`.

- Input with sequence length are provided, output without sequence lengths,
  meaning that RASR expects that the outputs have the same sequence length as the inputs:
    Define the output time dimension in `model_outputs` to have `dyn_size_ext` being a scalar.

- Input with sequence length are provided, output with sequence lengths:
    The input and output time dimensions are as usual, having `dyn_size_ext` of shape [B].
"""

from __future__ import annotations
import torch
from typing import Callable, Optional, Dict, List
import argparse
import os
from random import random

import _setup_returnn_env  # noqa
from returnn.config import Config
from returnn.log import log
from returnn.tensor import TensorDict

# noinspection PyProtectedMember
from returnn.torch.frontend.bridge import RFModuleAsPTModule
import returnn.frontend as rf
import returnn.util.basic as util
from returnn.tensor.utils import tensor_dict_fill_random_numpy_
from returnn.torch.data.tensor_utils import tensor_dict_numpy_to_torch_
import returnn.__main__ as rnn


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
    rnn.init_thread_join_hack()
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
    print("RETURNN frontend module to ONNX conversion.", file=log.v1)
    rnn.returnn_greeting()
    config.typed_dict.setdefault("backend", "torch")
    rnn.init_backend_engine()
    assert util.BackendEngine.is_torch_selected(), "For now only the torch backend is supported."
    rnn.init_faulthandler()


class ForwardModulePT(torch.nn.Module):
    """
    Wrapper of a PyTorch module that's meant to call forward_step from the config when called.
    """

    def __init__(self, pt_module: torch.nn.Module, forward_step: Callable, extern_data: TensorDict):
        """
        :param pt_module: RF module as obtained from the config.
        :param forward_step: forward_step function as obtained from the config.
        :param extern_data:
        """
        super().__init__()

        self.model = pt_module
        self.forward_step_func = forward_step
        self.extern_data = extern_data

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Wrapper to forward_step from the config.
        """
        extern_data = self.extern_data.copy_template()
        extern_data.assign_from_raw_tensor_dict_(data, with_scalar_dyn_sizes=False, duplicate_dims_are_excluded=True)
        self.forward_step_func(model=self.model, extern_data=extern_data)
        _check_matching_outputs()
        return rf.get_run_ctx().outputs.as_raw_tensor_dict(include_scalar_dyn_sizes=False)


class ForwardModuleRF(RFModuleAsPTModule):
    """
    Wrapper of a RETURNN frontend module that's meant to call forward_step from the config when called.
    """

    def __init__(self, rf_module: rf.Module, forward_step: Callable, extern_data: TensorDict):
        """
        :param rf_module: RF module as obtained from the config.
        :param forward_step: forward_step function as obtained from the config.
        :param extern_data:
        """
        super().__init__(rf_module)

        self.forward_step_func = forward_step
        self.extern_data = extern_data

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Wrapper to forward_step from the config.
        """
        extern_data = self.extern_data.copy_template()
        extern_data.assign_from_raw_tensor_dict_(data, with_scalar_dyn_sizes=False, duplicate_dims_are_excluded=True)
        self.forward_step_func(model=self.rf_module, extern_data=extern_data)
        _check_matching_outputs()
        return rf.get_run_ctx().outputs.as_raw_tensor_dict(include_scalar_dyn_sizes=False)


def _check_matching_outputs():
    rf.get_run_ctx().check_outputs_complete()
    model_outputs_raw_keys = set(_get_model_outputs_raw_keys())
    outputs_raw = rf.get_run_ctx().outputs.as_raw_tensor_dict(include_scalar_dyn_sizes=False)
    outputs_raw_keys = set(outputs_raw.keys())
    assert model_outputs_raw_keys == outputs_raw_keys, (
        f"Model outputs raw keys and output raw keys from forward_step don't match.\n"
        f"Model outputs raw keys: {sorted(model_outputs_raw_keys)}\n"
        f"Output raw keys: {sorted(outputs_raw_keys)}"
    )
    assert all(v is not None for k, v in outputs_raw.items()), (
        f"Output raw keys from forward_step contain None values.\n"
        f"Output raw keys with None: {list(k for k, v in outputs_raw.items() if v is None)}"
    )


def _get_model_outputs_raw_keys() -> List[str]:
    model_outputs = rf.get_run_ctx().expected_outputs
    model_outputs_raw_keys = []
    for k, v in model_outputs.data.items():
        model_outputs_raw_keys.append(k)
        for i, dim in enumerate(v.dims):
            if dim.dyn_size_ext and dim.dyn_size_ext.dims:
                model_outputs_raw_keys.append(f"{k}:size{i}")
    return model_outputs_raw_keys


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "config",
        type=str,
        help="Filename to config file. Must have `get_model()` and `forward_step()`. Can optionally have `export()`.",
    )
    parser.add_argument("checkpoint", type=str, help="Checkpoint to RF module, considering the backend.")
    parser.add_argument("out_onnx_filename", type=str, help="Filename of the final ONNX model.")
    parser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' (default) or 'gpu'.")
    parser.add_argument("--input_names", type=str, help="Comma-separated list of input names.")
    parser.add_argument("--output_names", type=str, help="Comma-separated list of output names.")
    args = parser.parse_args()

    init(config_filename=args.config, checkpoint=args.checkpoint, log_verbosity=args.verbosity, device=args.device)

    model_outputs_dict = config.typed_value("model_outputs")
    assert (
        model_outputs_dict is not None
    ), "The specified config needs to have explicit model outputs. Please define `model_outputs` in your config."
    model_outputs = TensorDict()
    model_outputs.update(model_outputs_dict, auto_convert=True)

    loaded_checkpoint = torch.load(args.checkpoint, map_location=torch.device(args.device))
    epoch = loaded_checkpoint["epoch"]
    step = loaded_checkpoint["step"]

    rf.init_forward_step_run_ctx(expected_outputs=model_outputs, step=step, epoch=epoch)
    rf.set_random_seed(42)

    get_model_func = config.typed_value("get_model")
    assert get_model_func, "get_model() isn't specified in the config passed as a parameter."
    sentinel_kw = {"__fwd_compatible_random_arg_%i" % int(random() * 100): None}
    model = get_model_func(epoch=epoch, step=step, **sentinel_kw)

    is_rf_module = isinstance(model, rf.Module)
    is_pt_module = isinstance(model, torch.nn.Module)
    assert (
        is_rf_module or is_pt_module
    ), "The module returned by get_model() isn't a returnn.frontend.Module or a torch.nn.Module."

    export_func = config.typed_value("export") or torch.onnx.export
    forward_step_func = config.typed_value("forward_step")
    assert forward_step_func is not None, "forward_step() must be defined in the config."

    extern_data_dict = config.typed_value("extern_data")
    extern_data = TensorDict()
    extern_data.update(extern_data_dict, auto_convert=True)
    extern_data.reset_content()
    for k, v in list(extern_data.data.items()):
        if not v.available_for_inference:
            del extern_data.data[k]

    tensor_dict_fill_random_numpy_(extern_data)
    tensor_dict_numpy_to_torch_(extern_data)
    extern_data_raw = extern_data.as_raw_tensor_dict(include_scalar_dyn_sizes=False, exclude_duplicate_dims=True)
    model_outputs_raw_keys = _get_model_outputs_raw_keys()

    if is_pt_module:
        model.load_state_dict(loaded_checkpoint["model"])
        model.eval()
        pt_model_fwd = ForwardModulePT(model, forward_step_func, extern_data)
    elif is_rf_module:
        pt_model_fwd = ForwardModuleRF(model, forward_step_func, extern_data)
        pt_model_fwd.load_state_dict(loaded_checkpoint["model"])
        pt_model_fwd.eval()
    else:
        assert False, "PT/RF module?"  # should not get here

    dynamic_axes = {}
    for k, v in list(extern_data.data.items()) + list(model_outputs.data.items()):
        dynamic_axes[k] = {i: dim.name for i, dim in enumerate(v.dims) if dim.is_dynamic()}
        for i, dim in enumerate(v.dims):
            if dim.dyn_size_ext and dim.dyn_size_ext.dims == ():
                continue
            if dim.dyn_size_ext:
                dynamic_axes[f"{k}:size{i}"] = {
                    j: dim_.name for j, dim_ in enumerate(dim.dyn_size_ext.dims) if dim_.is_dynamic()
                }

    if args.input_names:
        input_names = args.input_names.split(",")
        assert set(extern_data_raw.keys()) == set(
            input_names
        ), f"missmatch between input_names {input_names} and extern_data keys {list(extern_data_raw.keys())}"
    else:
        input_names = list(extern_data_raw.keys())
    if args.output_names:
        output_names = args.output_names.split(",")
        assert set(model_outputs_raw_keys) == set(
            output_names
        ), f"missmatch between output_names {output_names} and model_outputs keys {model_outputs_raw_keys}"
    else:
        output_names = model_outputs_raw_keys

    print("*** Input names:", input_names)
    print("*** Output names:", output_names)
    print("*** Dynamic axes:", dynamic_axes)

    export_func(
        pt_model_fwd,
        (extern_data_raw, {}),
        f=args.out_onnx_filename,
        verbose=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


if __name__ == "__main__":
    main()
