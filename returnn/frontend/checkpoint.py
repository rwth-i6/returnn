"""
Backend-neutral checkpoint I/O for RF models.

The format is plain NumPy (``.npz``), keyed by the RF parameter names
(:func:`rf.Module.named_parameters`), so the same file can be read by any backend,
and a PyTorch checkpoint converts into it one to one.
That is what a parameter-level comparison between the backends needs.

Only the file format and the name/shape bookkeeping live here.
Reading a parameter's value and writing it back is backend specific
(eager backends read the array directly, TF graph mode needs a session),
so each backend provides its own ``get_model_params`` / ``set_model_params``.
"""

from __future__ import annotations
from typing import Optional, Dict, Sequence

import numpy

import returnn.frontend as rf
from returnn.log import log


__all__ = ["save_params", "load_params", "load_torch_checkpoint", "check_params_complete"]


def save_params(
    values: Dict[str, numpy.ndarray], filename: str, *, step: Optional[int] = None, epoch: Optional[int] = None
):
    """
    :param values: parameters, keyed by their RF name
    :param filename: ``.npz`` file to write
    :param step: stored alongside, for the engine
    :param epoch: stored alongside, for the engine
    """
    meta = {}
    if step is not None:
        meta["_step"] = numpy.array(step)
    if epoch is not None:
        meta["_epoch"] = numpy.array(epoch)
    numpy.savez(filename, **values, **meta)
    print(f"Saved checkpoint {filename} ({len(values)} params)", file=log.v3)


def load_params(filename: str) -> Dict[str, numpy.ndarray]:
    """
    :param filename: ``.npz`` file, as written by :func:`save_params`
    :return: the parameters, without the meta entries (which start with an underscore)
    """
    with numpy.load(filename) as data:
        return {name: data[name] for name in data.files if not name.startswith("_")}


def load_torch_checkpoint(filename: str) -> Dict[str, numpy.ndarray]:
    """
    Read a PyTorch checkpoint written by the PyTorch engine, as plain NumPy arrays.

    The PyTorch engine stores ``{"model": state_dict, "epoch": ..., "step": ...}``,
    and the state dict is keyed by the same RF parameter names the other backends use
    (``rf_module_to_pt_module`` keeps them), so the values transfer one to one.

    Needs PyTorch installed -- reading a PyTorch checkpoint does.

    :param filename: a ``.pt`` file
    :return: the parameters, keyed by RF parameter name
    """
    import torch

    data = torch.load(filename, map_location="cpu", weights_only=False)
    state_dict = data["model"] if isinstance(data, dict) and "model" in data else data
    return {name: value.detach().cpu().numpy() for name, value in state_dict.items()}


def check_params_complete(
    model: rf.Module, params: Dict[str, numpy.ndarray], *, allow_missing: bool = False
) -> Sequence[str]:
    """
    :param model:
    :param params: as from :func:`load_params`
    :param allow_missing: do not fail on parameters absent from params
    :return: the names to set, i.e. the model parameters present in params
    :raise ValueError: on a mismatch between model and params
    """
    model_params = dict(model.named_parameters())
    missing = [name for name in model_params if name not in params]
    if missing and not allow_missing:
        raise ValueError(f"checkpoint is missing: {missing}")
    unexpected = [name for name in params if name not in model_params]
    if unexpected:
        raise ValueError(f"checkpoint has parameters which the model does not: {unexpected}")
    for name, param in model_params.items():
        if name not in params:
            continue
        expected = tuple(d.get_dim_value() for d in param.dims)
        if tuple(params[name].shape) != expected:
            raise ValueError(f"checkpoint {name} has shape {params[name].shape}, model expects {expected}")
    return [name for name in model_params if name in params]
