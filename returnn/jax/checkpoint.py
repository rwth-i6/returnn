"""
Checkpoints for the JAX backend: Orbax with OCDBT storage.
A checkpoint is a DIRECTORY (postfix ``.orbax``), not a file.

OCDBT is required, not preferred: 18 inodes per checkpoint against 513 without it (169 arrays),
i.e. ~3.6k vs ~103k over a 100-epoch training with optimizer state.

Keys are the RF parameter names, so a PyTorch checkpoint converts across one to one
(see :func:`load_torch_checkpoint`).
"""

from __future__ import annotations
from typing import Optional, Any, Dict

import os
import numpy
import jax
import jax.numpy as jnp

from returnn.tensor import Tensor
import returnn.frontend as rf
from returnn.log import log


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "save_opt_state",
    "load_opt_state",
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
        print(f"set_model_params: not in model, ignored: {unexpected}", file=log.v3)
    for name, param in model_params.items():
        if name not in params:
            continue
        value = params[name]
        expected = tuple(d.get_dim_value() for d in param.dims)
        if tuple(value.shape) != expected:
            raise ValueError(f"set_model_params: {name} has shape {value.shape}, expected {expected}")
        param.assign(Tensor(name, dims=param.dims, dtype=param.dtype, raw_tensor=jnp.asarray(value)))


def _checkpointer():
    """
    :return: an Orbax checkpointer writing OCDBT (its default)

    Imported lazily: ``load_torch_checkpoint`` and the NumPy helpers do not need Orbax.
    """
    import orbax.checkpoint as ocp

    return ocp.PyTreeCheckpointer()


def save_checkpoint(model: rf.Module, filename: str, *, step: Optional[int] = None, epoch: Optional[int] = None):
    """
    :param model:
    :param filename: ``.orbax`` DIRECTORY to write
    :param step: stored alongside, for the engine
    :param epoch: stored alongside, for the engine
    """
    values = get_model_params(model)
    tree = dict(values)
    if step is not None:
        tree["_step"] = numpy.array(step)
    if epoch is not None:
        tree["_epoch"] = numpy.array(epoch)
    _save_tree(tree, filename)
    print(f"Saved JAX checkpoint {filename} ({len(values)} params)", file=log.v3)


def load_checkpoint(filename: str) -> Dict[str, numpy.ndarray]:
    """
    :param filename: ``.orbax`` directory, as written by :func:`save_checkpoint`
    :return: the parameters, without the meta entries (which start with an underscore)
    """
    tree = _checkpointer().restore(os.path.abspath(filename))
    return {name: numpy.asarray(value) for name, value in tree.items() if not name.startswith("_")}


def save_opt_state(opt_state: Any, filename: str):
    """
    Save the optimizer state, so that training continues exactly.

    Orbax keeps the pytree structure, so this does not depend on the leaf ORDER
    matching between the run that wrote it and the run that reads it.

    :param opt_state:
    :param filename: ``.orbax`` DIRECTORY to write
    """
    _save_tree(opt_state, filename)
    print(f"Saved JAX optimizer state {filename} ({len(jax.tree_util.tree_leaves(opt_state))} arrays)", file=log.v3)


def load_opt_state(opt_state: Any, filename: str) -> Any:
    """
    :param opt_state: a freshly initialized state, giving the structure to restore into.
        A mismatch is an error: the optimizer or the trained parameter set changed.
    :param filename: as written by :func:`save_opt_state`
    :return: the state from the file
    """
    return _checkpointer().restore(os.path.abspath(filename), item=opt_state)


def _save_tree(tree: Any, filename: str) -> None:
    """
    :param tree: pytree of arrays
    :param filename: directory to write; an existing one is replaced

    Orbax needs an absolute path, and refuses to write over an existing directory unless told to.
    """
    _checkpointer().save(os.path.abspath(filename), tree, force=True)


def load_torch_checkpoint(filename: str) -> Dict[str, numpy.ndarray]:
    """
    Read a PyTorch checkpoint written by the PyTorch engine, as plain NumPy arrays.

    The PyTorch engine stores ``{"model": state_dict, "epoch": ..., "step": ...}``,
    and the state dict is keyed by the same RF parameter names this backend uses
    (``rf_module_to_pt_module`` keeps them), so the values transfer one to one.

    Needs PyTorch installed -- reading a PyTorch checkpoint does.
    JAX-side training does not; the format written here is Orbax (see the module docstring).

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
