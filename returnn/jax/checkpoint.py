"""
Checkpoints for the JAX backend.

The format is plain NumPy (``.npz``), keyed by the RF parameter names,
i.e. deliberately backend-neutral:
the same file can be read by any backend, and a PyTorch checkpoint can be converted into it,
which is what a parameter-level comparison between the backends needs.
"""

from __future__ import annotations
from typing import Optional, Dict

import numpy
import jax.numpy as jnp

from returnn.tensor import Tensor
import returnn.frontend as rf
from returnn.log import log


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "load_torch_checkpoint",
    "get_model_params",
    "set_model_params",
]


def get_model_params(model: rf.Module) -> Dict[str, numpy.ndarray]:
    """
    :param model:
    :return: all parameters of the model as numpy arrays, keyed by their RF name
    """
    return {name: numpy.asarray(param.raw_tensor) for name, param in model.named_parameters()}


def set_model_params(model: rf.Module, params: Dict[str, numpy.ndarray], *, allow_missing: bool = False) -> None:
    """
    :param model:
    :param params: as from :func:`get_model_params` or :func:`load_checkpoint`
    :param allow_missing: do not fail on parameters absent from params
    """
    model_params = dict(model.named_parameters())
    missing = [name for name in model_params if name not in params]
    if missing and not allow_missing:
        raise ValueError(f"set_model_params: missing in checkpoint: {missing}")
    unexpected = [name for name in params if name not in model_params]
    if unexpected:
        raise ValueError(f"set_model_params: not in model: {unexpected}")
    for name, param in model_params.items():
        if name not in params:
            continue
        value = params[name]
        expected = tuple(d.get_dim_value() for d in param.dims)
        if tuple(value.shape) != expected:
            raise ValueError(f"set_model_params: {name} has shape {value.shape}, expected {expected}")
        param.assign(Tensor(name, dims=param.dims, dtype=param.dtype, raw_tensor=jnp.asarray(value)))


def save_checkpoint(model: rf.Module, filename: str, *, step: Optional[int] = None, epoch: Optional[int] = None):
    """
    :param model:
    :param filename: ``.npz`` file to write
    :param step: stored alongside, for the engine
    :param epoch: stored alongside, for the engine
    """
    values = get_model_params(model)
    meta = {}
    if step is not None:
        meta["_step"] = numpy.array(step)
    if epoch is not None:
        meta["_epoch"] = numpy.array(epoch)
    numpy.savez(filename, **values, **meta)
    print(f"Saved JAX checkpoint {filename} ({len(values)} params)", file=log.v3)


def load_checkpoint(filename: str) -> Dict[str, numpy.ndarray]:
    """
    :param filename: ``.npz`` file, as written by :func:`save_checkpoint`
    :return: the parameters, without the meta entries (which start with an underscore)
    """
    with numpy.load(filename) as data:
        return {name: data[name] for name in data.files if not name.startswith("_")}


def load_torch_checkpoint(filename: str) -> Dict[str, numpy.ndarray]:
    """
    Read a PyTorch checkpoint written by the PyTorch engine, as plain NumPy arrays.

    The PyTorch engine stores ``{"model": state_dict, "epoch": ..., "step": ...}``,
    and the state dict is keyed by the same RF parameter names this backend uses
    (``rf_module_to_pt_module`` keeps them), so the values transfer one to one.

    Needs PyTorch installed -- reading a PyTorch checkpoint does.
    JAX-side training does not; the format written here is plain ``.npz``.

    :param filename: a ``.pt`` file
    :return: the parameters, keyed by RF parameter name
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            f"load_torch_checkpoint({filename!r}) needs PyTorch installed, to read the checkpoint."
        ) from None

    obj = torch.load(filename, map_location="cpu", weights_only=False)
    state_dict = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    return {name: value.detach().cpu().numpy() for name, value in state_dict.items()}
