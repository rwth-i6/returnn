"""
Data pipeline for the JAX backend.

It reuses RETURNN's existing backend-agnostic batching
(:func:`Dataset.generate_batches` -> :class:`BatchSetGenerator` -> :func:`batch_to_raw_dict`),
which is the same path the TF engine takes,
so no batching, padding or chunking logic is duplicated here.
Only the last step is JAX-specific: putting the NumPy arrays into a :class:`TensorDict` of JAX arrays.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Iterator, List, Sequence

import numpy
import jax.numpy as jnp

from returnn.tensor import Tensor, Dim, TensorDict
from returnn.datasets.basic import Dataset
from returnn.engine.batch import batch_to_raw_dict


__all__ = ["raw_dict_to_extern_data", "iter_dataset_batches"]


def raw_dict_to_extern_data(
    extern_data: TensorDict, raw: Dict[str, Any], *, device: Optional[str] = None
) -> TensorDict:
    """
    The dims of the template are REUSED and filled in, not replaced by fresh ones per batch,
    exactly as the PyTorch pipeline does it. Two reasons, both found the hard way:
    RF compares dims by identity, so the model's dims and these have to be the same objects;
    and RETURNN's legacy axis attributes (``Tensor.time_dim_axis``, which recipe code reaches
    through ``get_time_dim_tag``) are derived from the batch dim being the config's batch dim.

    :param extern_data: templates (dims, dtypes), as from the config
    :param raw: as from :func:`batch_to_raw_dict`: the data arrays plus "<key>_seq_lens" and "batch_dim"
    :param device: where to put the arrays, None for the JAX default
    :return: a TensorDict holding JAX arrays, with the dynamic dims' sizes filled in
    """
    batch_dim_value = int(raw["batch_dim"])
    batch_dim_ = _batch_dim_of(extern_data)
    # Reset FIRST, and the batch dim with them: reset_eager also drops Dim's cached size max,
    # which otherwise keeps reporting the first batch's value for every later batch.
    for dim in _dyn_dims_of(extern_data):
        dim.reset_eager()
    if batch_dim_.size is None:
        if batch_dim_.dyn_size_ext is None:
            batch_dim_.dyn_size_ext = Tensor(batch_dim_.name or "batch", dims=[], dtype="int32")
        # scalar: deliberately NOT device-committed, see _to_jax
        batch_dim_.dyn_size_ext.raw_tensor = jnp.asarray(batch_dim_value, dtype=jnp.int32)
    out = TensorDict()
    for key, template in extern_data.data.items():
        if key not in raw:
            continue
        data = template.copy_template()
        for i, dim in enumerate(data.dims):
            if i == 0 or not dim.is_dynamic():
                continue
            seq_lens = raw[f"{key}_seq_lens"]
            if dim.dyn_size_ext is None:
                dim.dyn_size_ext = Tensor(dim.name or "time", dims=[batch_dim_], dtype="int32")
            dim.dyn_size_ext.raw_tensor = _to_jax(seq_lens, dtype=dim.dyn_size_ext.dtype, device=device)
            # capacity = the padded extent, which is what every shape in the step is built from
            dim.capacity = int(numpy.max(seq_lens)) if len(seq_lens) else 0
        if template.dtype == "string":
            # JAX has no string arrays; keep them as NumPy, which RF dispatches to its NumPy backend.
            # That keeps entries like seq_tag available to the step function, just not on the device.
            data.raw_tensor = numpy.asarray(raw[key])
        else:
            data.raw_tensor = _to_jax(raw[key], dtype=template.dtype, device=device)
        out.data[key] = data
    return out


def _batch_dim_of(extern_data: TensorDict) -> Dim:
    """
    :param extern_data: templates
    :return: the batch dim, which every entry has as its first dim
    """
    dims = {data.dims[0] for data in extern_data.data.values() if data.dims}
    assert len(dims) == 1, f"extern_data entries do not share one batch dim: {dims}"
    return next(iter(dims))


def _dyn_dims_of(extern_data: TensorDict) -> List[Dim]:
    """
    :param extern_data: templates
    :return: the dynamic dims, the batch dim included -- it varies per batch just as much,
        and its cached size max has to be dropped with the others
    """
    res = []
    for data in extern_data.data.values():
        for dim in data.dims:
            if dim.is_dynamic() and not any(dim is d for d in res):
                res.append(dim)
    return res


def iter_dataset_batches(
    dataset: Dataset,
    *,
    extern_data: TensorDict,
    batch_size: int,
    max_seqs: int = -1,
    epoch: int = 1,
    data_keys: Optional[Sequence[str]] = None,
    device: Optional[str] = None,
    with_complete_frac: bool = False,
    **batch_opts,
) -> Iterator[TensorDict]:
    """
    Iterate one epoch of the dataset as batches of JAX tensors.

    :param dataset:
    :param extern_data: templates, defining what to read and with which dims
    :param batch_size: max frames per batch
    :param max_seqs: max seqs per batch
    :param epoch: passed to init_seq_order, so the dataset's seq_ordering (laplace, random, ...) applies
    :param data_keys: which entries to read, default all of extern_data
    :param device:
    :param with_complete_frac: yield ``(batch, complete_frac)`` instead of just the batch,
        where complete_frac is how much of the epoch is done after this batch,
        or None when the dataset cannot say it accurately enough (it feeds the LR schedule)
    :param batch_opts: further options for Dataset.generate_batches (max_seq_length, seq_drop, ...)
    :return: one TensorDict per batch
    """
    if data_keys is None:
        data_keys = sorted(extern_data.data.keys())
    dataset.init_seq_order(epoch=epoch)
    batches = dataset.generate_batches(
        recurrent_net=True, batch_size=batch_size, max_seqs=max_seqs, used_data_keys=set(data_keys), **batch_opts
    )
    while batches.has_more():
        (batch,) = batches.peek_next_n(1)
        raw = batch_to_raw_dict(batch, dataset=dataset, extern_data=extern_data, data_keys=data_keys)
        out = raw_dict_to_extern_data(extern_data, raw, device=device)
        if with_complete_frac:
            yield out, dataset.get_complete_frac(batch.end_seq - 1, allow_only_lr_suitable=True)
        else:
            yield out
        batches.advance(1)


def _to_jax(value: numpy.ndarray, *, dtype: str, device: Optional[str]):
    """
    :param value:
    :param dtype: the declared dtype, which the raw array must match (RF checks this)
    :param device:
    :return: the value as a JAX array
    """
    raw = jnp.asarray(numpy.asarray(value, dtype=dtype))
    if device:
        import jax

        # noinspection PyProtectedMember
        from returnn.jax.frontend._backend import _device_from_str

        raw = jax.device_put(raw, _device_from_str(device))
    return raw
