"""
Reading and writing the parameters of an RF model on the TF backend.

The file format and the name bookkeeping are backend neutral and live in
:mod:`returnn.frontend.checkpoint`. What is specific here is that TF runs in graph mode:
a parameter's value is only available through a session, and setting one is an assign op
which has to be run.
"""

from __future__ import annotations
from typing import Optional, Dict

import numpy

import returnn.frontend as rf
import returnn.tf.compat as tf_compat
from returnn.frontend.checkpoint import save_params, load_params, load_torch_checkpoint, check_params_complete
from returnn.tf.frontend_low_level import TFBackend


__all__ = [
    "get_model_params",
    "set_model_params",
    "save_checkpoint",
    "load_checkpoint",
    "load_torch_checkpoint",
]


def get_model_params(model: rf.Module, session: tf_compat.v1.Session) -> Dict[str, numpy.ndarray]:
    """
    :param model:
    :param session:
    :return: all parameters of the model as numpy arrays, keyed by their RF name
    """
    variables = {name: TFBackend.get_parameter_variable(param) for name, param in model.named_parameters()}
    return session.run(variables)


def set_model_params(
    model: rf.Module,
    params: Dict[str, numpy.ndarray],
    session: tf_compat.v1.Session,
    *,
    allow_missing: bool = False,
):
    """
    :param model:
    :param params: as from :func:`get_model_params` or :func:`load_checkpoint`
    :param session:
    :param allow_missing: do not fail on parameters absent from params
    """
    names = check_params_complete(model, params, allow_missing=allow_missing)
    model_params = dict(model.named_parameters())
    ops = []
    feed_dict = {}
    for name in names:
        var = TFBackend.get_parameter_variable(model_params[name])
        # via a placeholder, so the values do not end up as constants in the graph
        placeholder = tf_compat.v1.placeholder(var.dtype.base_dtype, var.shape, name=f"assign_param/{name}")
        ops.append(tf_compat.v1.assign(var, placeholder))
        feed_dict[placeholder] = params[name]
    session.run(ops, feed_dict=feed_dict)


def save_checkpoint(
    model: rf.Module,
    session: tf_compat.v1.Session,
    filename: str,
    *,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
):
    """
    :param model:
    :param session:
    :param filename: ``.npz`` file to write
    :param step:
    :param epoch:
    """
    save_params(get_model_params(model, session), filename, step=step, epoch=epoch)


def load_checkpoint(model: rf.Module, session: tf_compat.v1.Session, filename: str, *, allow_missing: bool = False):
    """
    :param model:
    :param session:
    :param filename: ``.npz`` file, as written by :func:`save_checkpoint`,
        or by any other backend -- the keys are the RF parameter names
    :param allow_missing:
    """
    set_model_params(model, load_params(filename), session, allow_missing=allow_missing)


def load_torch_checkpoint_into_model(
    model: rf.Module, session: tf_compat.v1.Session, filename: str, *, allow_missing: bool = False
):
    """
    :param model:
    :param session:
    :param filename: a ``.pt`` file written by the PyTorch engine
    :param allow_missing:
    """
    set_model_params(model, load_torch_checkpoint(filename), session, allow_missing=allow_missing)
