"""
Converts a module from a config to ONNX. For that, it uses get_model() which must be available in the config
and creates dummy data to be forwarded to the model.

Since get_model() can return either a torch.nn.Module or a rf.Module, both cases must be taken into account.
"""


import torch
from typing import Callable, Optional, Sequence
import argparse
import os
import numpy

from returnn.config import Config
from returnn.log import log
from returnn.tensor import Dim, Tensor, TensorDict

# noinspection PyProtectedMember
from returnn.torch.frontend.bridge import _RFModuleAsPTModule, _RFTensorAsPTTensor
import returnn.frontend as rf
import returnn.util.basic as util
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
    rnn.init_backend_engine()
    assert util.BackendEngine.is_torch_selected(), "For now only the torch backend is supported."
    rnn.init_faulthandler()


class ForwardModulePT(torch.nn.Module):
    """
    Wrapper of a PyTorch module that's meant to call forward_step from the config when called.
    """

    def __init__(self, pt_module: torch.nn.Module, forward_step: Callable):
        """
        :param pt_module: RF module as obtained from the config.
        :param forward_step: forward_step function as obtained from the config.
        """
        super().__init__()

        self.model = pt_module
        self.forward_step_func = forward_step

    def __call__(self, data: _RFTensorAsPTTensor):
        """
        Wrapper to forward_step from the config.
        """
        extern_data = TensorDict()
        extern_data.update({"data": data.rf_tensor}, auto_convert=True)
        self.forward_step_func(model=self.model, extern_data=extern_data)
        # debug_raw_tensor_dict = rf.get_run_ctx().outputs.as_raw_tensor_dict()
        # return debug_raw_tensor_dict  # doesnt work, as there's more than one output in the dict (output:size0, etc)
        return rf.get_run_ctx().outputs.data["output"].raw_tensor  # works


class ForwardModuleRF(_RFModuleAsPTModule):
    """
    Wrapper of a RETURNN frontend module that's meant to call forward_step from the config when called.
    """

    def __init__(self, rf_module: rf.Module, forward_step: Callable):
        """
        :param rf_module: RF module as obtained from the config.
        :param forward_step: forward_step function as obtained from the config.
        """
        super().__init__(rf_module)

        self.forward_step_func = forward_step

    def __call__(self, data: _RFTensorAsPTTensor):
        """
        Wrapper to forward_step from the config.
        """
        extern_data = TensorDict()
        extern_data.update({"data": data.rf_tensor}, auto_convert=True)
        self.forward_step_func(model=self.rf_module, extern_data=extern_data)
        # debug_raw_tensor_dict = rf.get_run_ctx().outputs.as_raw_tensor_dict()
        # return debug_raw_tensor_dict  # doesnt work, as there's more than one output in the dict (output:size0, etc)
        return rf.get_run_ctx().outputs.data["output"].raw_tensor  # works


def fill_batch_time_dims(dims: Sequence[Dim]):
    """
    Creates random capacities for batch and time dimensions.
    This step is prior to creating a random tensor.

    :param dims: Input dimensions extracted from extern_data. This argument is modified in-place.
    """
    rnd = numpy.random.RandomState(42)
    initial_dims = []
    initial_raw_dims = []

    # Handle batch dim(s)
    # TODO: more refined logic
    # TODO: what if some dim.kind is undefined? E.g. in rf-demo all dims would be batch if not by d.name.startswith...
    local_batch_dims = [
        d
        for d in dims
        if not d.is_spatial_dim()
        and not d.is_feature_dim()
        and not d.name.startswith("time")
        and not d.name.startswith("in")
    ]
    for local_batch_dim in local_batch_dims:
        raw_tensor = rnd.randint(5, 20, size=initial_raw_dims, dtype="int32")
        local_batch_dim.dyn_size_ext = Tensor(
            name=local_batch_dim.name, dims=initial_dims, dtype="int32", raw_tensor=raw_tensor
        )
        initial_dims.append(local_batch_dim)
        initial_raw_dims.append(raw_tensor.size)

    # Handle time dim(s) in a similar way to batch dim(s)
    # TODO: more refined logic, also same problem as above
    local_time_dims = [d for d in dims if d.name.startswith("time")]
    for local_time_dim in local_time_dims:
        raw_tensor = rnd.randint(5, 20, size=initial_raw_dims, dtype="int32")
        local_time_dim.dyn_size_ext = Tensor(
            name=local_time_dim.name, dims=initial_dims, dtype="int32", raw_tensor=raw_tensor
        )
        initial_dims.append(local_time_dim)
        initial_raw_dims.append(raw_tensor.size)


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
    extern_data_aux = TensorDict()
    extern_data_aux.update(extern_data_dict, auto_convert=True)
    dims = extern_data_aux["data"].dims

    fill_batch_time_dims(dims)

    dtype = extern_data_aux["data"].dtype
    dummy_tensor = rf.random(dims=dims, dtype=dtype, distribution="uniform")
    # dummy_tensor = _RFTensorAsPTTensor(dummy_tensor)
    dummy_final_tensor = dummy_tensor.raw_tensor.as_subclass(_RFTensorAsPTTensor)
    dummy_final_tensor.rf_tensor = dummy_tensor
    if is_pt_module:
        model.load_state_dict(loaded_checkpoint["model"])
        model.eval()
        pt_model_fwd = ForwardModulePT(model, forward_step_func)
        # dummy_tensor = dummy_tensor.raw_tensor
    else:
        pt_model_fwd = ForwardModuleRF(model, forward_step_func)
        pt_model_fwd.load_state_dict(loaded_checkpoint["model"])
        pt_model_fwd.eval()

    # extern_data_raw = extern_data.as_raw_tensor_dict()
    # dummy_tensor = extern_data_raw["data"]

    export_func(
        pt_model_fwd,
        (dummy_final_tensor,),
        f=args.out_onnx_filename,
        verbose=True,
        input_names=["data"],  # , "data_len"],
        output_names=["classes"],
        dynamic_axes={
            "data": {0: "batch", 1: "time"},  # TODO: automatically infer dynamic axes
            # "data_len": {0: "batch"},
            # "classes": {0: "batch", 1: "time"},
        },
    )


if __name__ == "__main__":
    main()
