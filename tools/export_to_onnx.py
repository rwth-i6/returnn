"""
Converts a module from a config to ONNX. For that, it uses get_model() which must be available in the config
and creates dummy data to be forwarded to the model.

Since get_model() can return either a torch.nn.Module or a rf.Module, both cases must be taken into account.
"""

from __future__ import annotations
import torch
from typing import Callable, Optional, Dict
import argparse
import os
import numpy

from returnn.config import Config
from returnn.log import log
from returnn.tensor import Dim, Tensor, TensorDict

# noinspection PyProtectedMember
from returnn.torch.frontend.bridge import _RFModuleAsPTModule
import returnn.frontend as rf
import returnn.util.basic as util
from returnn.torch.data.tensor_utils import tensor_numpy_to_torch_
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
        extern_data.assign_from_raw_tensor_dict_(data)
        self.forward_step_func(model=self.model, extern_data=extern_data)
        return rf.get_run_ctx().outputs.as_raw_tensor_dict()


class ForwardModuleRF(_RFModuleAsPTModule):
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
        extern_data.assign_from_raw_tensor_dict_(data)
        self.forward_step_func(model=self.rf_module, extern_data=extern_data)
        return rf.get_run_ctx().outputs.as_raw_tensor_dict()


def main():
    """
    Main entry point
    """
    parser = argparse.ArgumentParser(description="Converts a RF/PT module to ONNX.")
    parser.add_argument(
        "config", type=str, help="Filename to config file. Must have get_model(). Can optionally have export()."
    )
    parser.add_argument("checkpoint", type=str, help="Checkpoint to RF module, considering the backend.")
    parser.add_argument("out_onnx_filename", type=str, help="Filename of the final ONNX model.")
    parser.add_argument("--verbosity", default=4, type=int, help="5 for all seqs (default: 4)")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' (default) or 'gpu'.")
    args = parser.parse_args()

    init(config_filename=args.config, checkpoint=args.checkpoint, log_verbosity=args.verbosity, device=args.device)
    rf.init_forward_step_run_ctx()
    rf.set_random_seed(42)

    get_model_func = config.typed_value("get_model")
    assert get_model_func, "get_model() isn't specified in the config passed as a parameter."
    model = get_model_func()
    loaded_checkpoint = torch.load(args.checkpoint)

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

    for v in extern_data.data.values():
        _reset_tensor(v)
    rnd = numpy.random.RandomState(42)
    for v in extern_data.data.values():
        _fill_random(v, rnd=rnd)
    for v in extern_data.data.values():
        tensor_numpy_to_torch_(v)
    extern_data_raw = extern_data.as_raw_tensor_dict()

    if is_pt_module:
        model.load_state_dict(loaded_checkpoint["model"])
        model.eval()
        pt_model_fwd = ForwardModulePT(model, forward_step_func, extern_data)
    else:
        pt_model_fwd = ForwardModuleRF(model, forward_step_func, extern_data)
        pt_model_fwd.load_state_dict(loaded_checkpoint["model"])
        pt_model_fwd.eval()

    model_outputs_dict = config.typed_value("model_outputs")
    model_outputs = TensorDict()
    model_outputs.update(model_outputs_dict, auto_convert=True)
    model_outputs_raw_keys = []
    for k, v in model_outputs.data.items():
        model_outputs_raw_keys.append(k)
        for i, dim in enumerate(v.dims):
            if dim.is_batch_dim() or dim.is_dynamic():
                model_outputs_raw_keys.append(f"{k}:size{i}")

    dynamic_axes = {}
    for k, v in list(extern_data.data.items()) + list(model_outputs.data.items()):
        dynamic_axes[k] = {i: dim.name for i, dim in enumerate(v.dims) if dim.is_dynamic() or dim.is_batch_dim()}
        for i, dim in enumerate(v.dims):
            if dim.dyn_size_ext:
                dynamic_axes[f"{k}:size{i}"] = {
                    j: dim_.name
                    for j, dim_ in enumerate(dim.dyn_size_ext.dims)
                    if dim_.is_dynamic() or dim_.is_batch_dim()
                }

    export_func(
        pt_model_fwd,
        (extern_data_raw, {}),
        f=args.out_onnx_filename,
        verbose=True,
        input_names=list(extern_data_raw.keys()),
        output_names=model_outputs_raw_keys,
        dynamic_axes=dynamic_axes,
    )


def _reset_tensor(x: Tensor):
    """reset"""
    x.batch = None
    x.raw_tensor = None
    for dim in x.dims:
        dim.batch = None
        if dim.dyn_size_ext:
            _reset_tensor(dim.dyn_size_ext)


def _fill_random(
    x: Tensor,
    *,
    min_val: int = 0,
    max_val: Optional[int] = None,
    rnd: numpy.random.RandomState,
    dyn_dim_max_sizes: Optional[Dict[Dim, int]] = None,
) -> bool:
    """fill. return whether sth was filled"""
    if dyn_dim_max_sizes is None:
        dyn_dim_max_sizes = {}
    filled = False
    while True:
        have_unfilled = False
        filled_this_round = False

        for dim in x.dims:
            if dim.is_batch_dim() and not dim.dyn_size_ext:
                dim.dyn_size_ext = Tensor("batch", [], dtype="int32")
            if not dim.dyn_size_ext:
                continue
            if _fill_random(
                dim.dyn_size_ext,
                min_val=2,
                max_val=dyn_dim_max_sizes.get(dim, None),
                rnd=rnd,
                dyn_dim_max_sizes=dyn_dim_max_sizes,
            ):
                if dim in dyn_dim_max_sizes:
                    # Make sure at least one of the dyn sizes matches the max size.
                    i = rnd.randint(0, dim.dyn_size_ext.raw_tensor.size)
                    dim.dyn_size_ext.raw_tensor.flat[i] = dyn_dim_max_sizes[dim]
                filled = True
                filled_this_round = True
            if dim.dyn_size_ext.raw_tensor is None:
                have_unfilled = True
            elif not isinstance(dim.dyn_size_ext.raw_tensor, numpy.ndarray):
                have_unfilled = True

        if have_unfilled:
            assert filled_this_round, f"should have filled something, {x}"

        if not have_unfilled:
            break

    if x.raw_tensor is not None:
        if not isinstance(x.raw_tensor, numpy.ndarray):
            x.raw_tensor = None

    if x.raw_tensor is None:
        shape = [d.get_dim_value() for d in x.dims]
        if x.dtype.startswith("int"):
            if max_val is None:
                max_val = rnd.randint(5, 20)
            if x.sparse_dim and x.sparse_dim.dimension is not None:
                max_val = x.sparse_dim.dimension
            x.raw_tensor = rnd.randint(min_val, max_val, size=shape, dtype=x.dtype)
        elif x.dtype.startswith("float"):
            x.raw_tensor = rnd.normal(0.0, 1.0, size=shape).astype(x.dtype)
        elif x.dtype.startswith("complex"):
            real = rnd.normal(0.0, 1.0, size=shape)
            imag = rnd.normal(0.0, 1.0, size=shape)
            x.raw_tensor = (real + 1j * imag).astype(x.dtype)
        else:
            raise NotImplementedError(f"not implemented for {x} dtype {x.dtype}")
        filled = True

    assert isinstance(x.raw_tensor, numpy.ndarray)

    return filled


if __name__ == "__main__":
    main()
