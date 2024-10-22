"""
Helper for any type of PyTorch exceptions
"""

from __future__ import annotations

from typing import Optional, Union, Tuple

import torch
import numpy as np

from returnn.log import log
import returnn.frontend as rf
from returnn.tensor import TensorDict


def help_on_torch_exception(
    exc: Exception,
    *,
    step_idx: Optional[int] = None,
    extern_data: Optional[TensorDict] = None,
    model: Union[rf.Module, torch.nn.Module],
    always_direct_print: bool = False,
):
    """
    Gather some information which might be helpful for debugging a PyTorch exception.
    """
    from returnn.util.better_exchook import get_func_from_code_object, iter_traceback

    print(f"{type(exc).__name__}: {exc}", file=log.v1)

    exc_ext = [f"Step idx: {step_idx}"]
    if extern_data:
        exc_ext.append("Extern data:")
        if "seq_tag" in extern_data:
            exc_ext.append(f"  Seq tags: {extern_data['seq_tag'].raw_tensor}")
        covered_dim_tags = set()
        for data_key, data in extern_data.data.items():
            info, v_minmax = _help_data_or_array(data.raw_tensor)
            exc_ext.append(f"  {data_key}: {info}, {data}")
            if data.sparse:
                if v_minmax[0] < 0 or v_minmax[1] >= data.dim:
                    exc_ext.append(f"  WARNING, invalid label for data sparse dim {data.sparse_dim}")
            for dim in data.dims:
                if dim in covered_dim_tags:
                    continue
                covered_dim_tags.add(dim)
                if not dim.dyn_size_ext:
                    continue
                info, _ = _help_data_or_array(dim.dyn_size_ext.raw_tensor)
                exc_ext.append(f"    dim {dim.short_repr()} size: {info}")

    # Extend exception message by module call stack.
    exc_ext.append("Module call stack:")
    module_names_by_id = {}  # id -> name
    count_frames = 0
    for name, mod in model.named_modules():
        if id(mod) not in module_names_by_id:
            module_names_by_id[id(mod)] = name or "(root)"
    for frame in iter_traceback(exc.__traceback__):
        if frame.f_code.co_nlocals == 0:
            continue
        frame_self = frame.f_locals.get("self")
        if isinstance(frame_self, (torch.nn.Module, rf.Module)):
            func = get_func_from_code_object(frame.f_code, frame=frame)
            if func and func.__name__ and func.__name__.startswith("_") and not func.__name__.startswith("__"):
                continue
            func_name = (func and func.__qualname__) or type(frame_self).__name__
            exc_ext.append(f"({func_name}) {module_names_by_id.get(id(frame_self), '(unknown)')}")
            count_frames += 1
    if not count_frames:
        exc_ext.append("(No module call frames.)")

    if len(exc.args) == 1 and isinstance(exc.args[0], str) and not always_direct_print:
        exc.args = ("\n".join([exc.args[0], ""] + exc_ext),)
    else:
        for msg in exc_ext:
            print(msg, file=log.v3)


def _help_data_or_array(
    value: Union[torch.Tensor, np.ndarray, bool, object]
) -> Tuple[str, Tuple[Union[int, float], Union[int, float]]]:
    """
    :param value:
    :return: (info,(min,max))
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.dtype == torch.bfloat16:
            value = value.float()
        value = value.numpy()
    v_minmax = -1, -1
    if isinstance(value, np.ndarray):
        info = "shape %s, dtype %s" % (value.shape, value.dtype)
        if value.dtype.kind in "biuf":
            if value.size > 1:
                v_minmax = np.min(value), np.max(value)
                info += ", min/max %s/%s" % v_minmax
                if value.dtype.kind == "f":
                    info += ", mean/stddev %s/%s" % (np.mean(value), np.std(value))
                if value.ndim <= 1:
                    info += " (%s)" % np.array2string(value)
            elif value.size == 1:
                info += " (%s)" % np.array2string(value)
            else:
                info += ", EMPTY"
    elif isinstance(value, (np.floating, np.integer, np.bool_, float, int, bool, str, bytes)):
        info = "%s(%s)" % (type(value).__name__, value)
    elif value is None:
        info = "None"
    else:
        info = "type %r" % type(value)
    return info, v_minmax
