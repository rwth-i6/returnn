"""
Data pipeline for the JAX backend.

It reuses RETURNN's existing backend-agnostic batching
(:func:`Dataset.generate_batches` -> :class:`BatchSetGenerator` -> :func:`batch_to_raw_dict`),
which is the same path the TF engine takes,
so no batching, padding or chunking logic is duplicated here.
Only the last step is JAX-specific: putting the NumPy arrays into a :class:`TensorDict` of JAX arrays.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Iterator, List, Sequence, Union

import numpy
import jax.numpy as jnp

from returnn.tensor import Tensor, Dim, TensorDict
from returnn.datasets.basic import Dataset
from returnn.engine.batch import batch_to_raw_dict


__all__ = [
    "raw_dict_to_extern_data",
    "batch_to_jax_raws",
    "fill_extern_data",
    "reset_extern_data_dims",
    "pad_raws_to_bucket",
    "iter_dataset_batches",
]


def raw_dict_to_extern_data(
    extern_data: TensorDict, raw: Dict[str, Any], *, device: Optional[str] = None, time_multiple: int = 0
) -> TensorDict:
    """
    One batch, from the NumPy arrays of the shared batching layer to a TensorDict of JAX arrays.
    This is :func:`batch_to_jax_raws` followed by :func:`fill_extern_data`;
    the compiled step needs those two halves separately (only the second one is traceable).

    :param extern_data: templates (dims, dtypes), as from the config
    :param raw: as from :func:`batch_to_raw_dict`: the data arrays plus "<key>_seq_lens" and "batch_dim"
    :param device: where to put the arrays, None for the JAX default
    :param time_multiple: see :func:`batch_to_jax_raws`
    :return: a TensorDict holding JAX arrays, with the dynamic dims' sizes filled in
    """
    return fill_extern_data(
        extern_data, batch_to_jax_raws(raw, extern_data=extern_data, device=device, time_multiple=time_multiple)
    )


def batch_to_jax_raws(
    raw: Dict[str, Any],
    *,
    extern_data: TensorDict,
    device: Optional[str] = None,
    time_multiple: Union[int, Dict[str, int]] = 0,
) -> Dict[str, Any]:
    """
    Host side of the data path: the arrays of one batch as JAX arrays, keys unchanged.
    Kept separate from :func:`fill_extern_data` because this half is what a compiled step
    cannot contain (it reads and converts host values), while the other half is traceable.

    :param raw: as from :func:`batch_to_raw_dict`
    :param extern_data: templates, for the dtypes
    :param device: where to put the arrays, None for the JAX default
    :param time_multiple: if >1, pad the time axis up to a multiple of this.
        A compiled step is specialized per input shape, so rounding the padded extent up
        bounds how many variants get compiled -- at the price of computing on the padding.
        PER DATA KEY (``{"audio": 16000, "text": 8}``), because the multiple is in the unit of the
        axis it pads: 16000 audio samples and 16000 target labels are not remotely the same thing.
        A bare int applies to every key, which is only unambiguous with a single dynamic dim --
        see :func:`returnn.jax.engine._check_time_multiple`.
    :return: the data arrays plus "<key>_seq_lens" and "batch_dim", as JAX arrays
    """
    out: Dict[str, Any] = {}
    for key, template in extern_data.data.items():
        if key not in raw:
            continue
        value = raw[key]
        seq_lens = raw.get(f"{key}_seq_lens")
        if template.dtype == "string":
            # JAX has no string arrays; keep them as NumPy, which RF dispatches to its NumPy backend.
            # That keeps entries like seq_tag available to the step function, just not on the device.
            out[key] = numpy.asarray(value)
        else:
            multiple = time_multiple.get(key, 0) if isinstance(time_multiple, dict) else time_multiple
            if seq_lens is not None and multiple > 1 and value.ndim > 1:
                value = _pad_time(value, multiple=multiple)
            out[key] = _to_jax(value, dtype=template.dtype, device=device)
        if seq_lens is not None:
            out[f"{key}_seq_lens"] = _to_jax(seq_lens, dtype="int32", device=device)
    # packed keys carry a flat <key>:packed buffer instead of the padded array,
    # under a name extern_data does not have (see staticize_raws).
    # Their seq lens come along here too: the loop above transfers lens only for keys it sees,
    # and it never sees a packed key.
    for key, value in raw.items():
        if not key.endswith(":packed"):
            continue
        base = key[: -len(":packed")]
        out[key] = _to_jax(value, dtype=extern_data.data[base].dtype, device=device)
        lens = raw.get(f"{base}_seq_lens")
        if lens is not None:
            out[f"{base}_seq_lens"] = _to_jax(lens, dtype="int32", device=device)
    # scalar: deliberately NOT device-committed, see _to_jax
    out["batch_dim"] = jnp.asarray(int(raw["batch_dim"]), dtype=jnp.int32)
    return out


def fill_extern_data(extern_data: TensorDict, raws: Dict[str, Any]) -> TensorDict:
    """
    Fill the templates with one batch of raw arrays.

    Traceable: no host read of any value happens here, so this also runs inside ``jax.jit``,
    where the arrays are tracers. The one thing that must stay static is every SHAPE,
    and each dynamic dim gets its ``capacity`` from the raw array it is filled from --
    the padded extent, which is what every shape in the step is then built from.

    The dims of the template are REUSED and filled in, not replaced by fresh ones per batch,
    exactly as the PyTorch pipeline does it. Two reasons, both found the hard way:
    RF compares dims by identity, so the model's dims and these have to be the same objects;
    and RETURNN's legacy axis attributes (``Tensor.time_dim_axis``, which recipe code reaches
    through ``get_time_dim_tag``) are derived from the batch dim being the config's batch dim.

    :param extern_data: templates (dims, dtypes), as from the config
    :param raws: as from :func:`batch_to_jax_raws`
    :return: a TensorDict holding those arrays, with the dynamic dims' sizes filled in
    """
    batch_dim_ = _batch_dim_of(extern_data)
    # Reset FIRST, and the batch dim with them: reset_eager also drops Dim's cached size max,
    # which otherwise keeps reporting the first batch's value for every later batch.
    reset_extern_data_dims(extern_data)
    out = TensorDict()
    for key, template in extern_data.data.items():
        if key not in raws:
            continue
        data = template.copy_template()
        raw = raws[key]
        for i, dim in enumerate(data.dims):
            if not dim.is_dynamic():
                continue
            if i == 0:
                if batch_dim_.dyn_size_ext is None:
                    batch_dim_.dyn_size_ext = Tensor(batch_dim_.name or "batch", dims=[], dtype="int32")
                batch_dim_.dyn_size_ext.raw_tensor = raws["batch_dim"]
                batch_dim_.capacity = raw.shape[0]
                continue
            if dim.dyn_size_ext is None:
                dim.dyn_size_ext = Tensor(dim.name or "time", dims=[batch_dim_], dtype="int32")
            dim.dyn_size_ext.raw_tensor = raws[f"{key}_seq_lens"]
            dim.capacity = raw.shape[i]
        data.raw_tensor = raw
        out.data[key] = data
    return out


def pad_raws_to_bucket(raws: Dict[str, Any], *, extern_data: TensorDict, bucket: Dict[str, int]) -> Dict[str, Any]:
    """
    Pad one batch up to a declared bucket: the batch axis to ``bucket["batch_dim"]`` sequences,
    and every dynamic axis to the extent the bucket declares for its key.

    The added sequences get length 0, so every mask in the step excludes them, and the batch dim's
    VALUE stays the true number of sequences -- only its capacity becomes the bucket's.
    That is what makes a padded batch axis correct rather than merely shaped right.

    :param raws: as from :func:`batch_to_jax_raws`
    :param extern_data: templates
    :param bucket: ``batch_dim`` plus one entry per data key that has a dynamic axis
    :return: raws with exactly the bucket's shapes
    """
    num_seqs = int(bucket["batch_dim"])
    out = dict(raws)
    for key, template in extern_data.data.items():
        if key not in raws or template.dtype == "string":
            continue
        value = raws[key]
        pad_width = [(0, num_seqs - value.shape[0])]
        for axis, dim in enumerate(template.dims[1:], start=1):
            target = int(bucket[key]) if dim.is_dynamic() else value.shape[axis]
            pad_width.append((0, target - value.shape[axis]))
        if any(before or after for before, after in pad_width):
            out[key] = jnp.pad(value, pad_width)
        lens_key = f"{key}_seq_lens"
        if lens_key in raws:
            # length 0 for the added sequences: they exist in the buffer and nowhere else
            out[lens_key] = jnp.pad(raws[lens_key], [(0, num_seqs - raws[lens_key].shape[0])])
    return out


def reset_extern_data_dims(extern_data: TensorDict):
    """
    Drop the raw sizes of all dynamic dims of the templates, the batch dim included.

    Needed after a compiled step: what the dims hold then are tracers of a finished trace,
    which any later use would fail on.

    :param extern_data: templates
    """
    for dim in _dyn_dims_of(extern_data):
        dim.reset_eager()


def _pad_time(value: numpy.ndarray, *, multiple: int) -> numpy.ndarray:
    """
    :param value: [B,T,...]
    :param multiple:
    :return: the same, with T padded up to a multiple. The padding is zeros, and the seq lens
        do not change, so it is masked out wherever the seq lens are respected.
    """
    time = value.shape[1]
    padded = -(-time // multiple) * multiple
    if padded == time:
        return value
    return numpy.pad(value, [(0, 0), (0, padded - time)] + [(0, 0)] * (value.ndim - 2))


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
    as_raws: bool = False,
    time_multiple: Union[int, Dict[str, int]] = 0,
    static_shapes: Optional[Dict[str, Any]] = None,
    packed_keys: Sequence[str] = (),
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
    :param as_raws: yield the raw arrays (see :func:`batch_to_jax_raws`) instead of a TensorDict.
        That is what a compiled step takes: filling the templates is part of the traced function.
    :param time_multiple: see :func:`batch_to_jax_raws`
    :param static_shapes: bring every batch to these declared shapes on the host,
        see :func:`staticize_raws` -- what a compiled step needs, instead of buckets
    :param packed_keys: keys to store packed, with ``static_shapes``
    :param batch_opts: further options for Dataset.generate_batches (max_seq_length, seq_drop, ...)
    :return: one TensorDict (or raw dict, see ``as_raws``) per batch
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
        if static_shapes is not None:
            raw = staticize_raws(raw, extern_data=extern_data, opts=static_shapes, packed_keys=packed_keys)
        out = batch_to_jax_raws(raw, extern_data=extern_data, device=device, time_multiple=time_multiple)
        if not as_raws:
            out = fill_extern_data(extern_data, out)
        if with_complete_frac:
            yield out, dataset.get_complete_frac(batch.end_seq - 1, allow_only_lr_suitable=True)
        else:
            yield out
        batches.advance(1)


def staticize_raws(
    raw: Dict[str, Any],
    *,
    extern_data: TensorDict,
    opts: Dict[str, Any],
    packed_keys: Sequence[str] = (),
) -> Dict[str, Any]:
    """
    Bring one batch to the DECLARED shapes, on the host, before any transfer.

    A compiled step needs ONE input signature.
    Buckets get there by rounding each batch up to the nearest declared shape;
    bounds get there directly, as the TF and PyTorch engines do it.
    Packed keys have no padded array at all: the seqs are concatenated into a flat buffer,
    which is why this runs here and not in the step --
    feeding ``[batch_size_bound, capacity, ...]`` would transfer mostly padding.

    :param raw: as from :func:`batch_to_raw_dict`, NumPy
    :param extern_data: templates
    :param opts: ``batch_size_bound``, and optionally ``dim_capacity`` / ``packed_total_bound``
    :param packed_keys: stored packed, as a flat ``<key>:packed`` entry
    :return: raws with the declared shapes
    """
    batch_bound = opts["batch_size_bound"]
    dim_capacity = opts.get("dim_capacity") or {}
    total_bounds = opts.get("packed_total_bound") or {}
    out: Dict[str, Any] = {"batch_dim": raw["batch_dim"]}
    for key, template in extern_data.data.items():
        if key not in raw:
            continue
        value = raw[key]
        lens = raw.get(f"{key}_seq_lens")
        if lens is None or template.dtype == "string":
            out[key] = value
            continue
        n = lens.shape[0]
        assert n <= batch_bound, f"batch has {n} seqs > batch_size_bound {batch_bound}"
        out[f"{key}_seq_lens"] = numpy.pad(lens, (0, batch_bound - n))
        if key in packed_keys:
            total_bound = total_bounds.get(key)
            assert isinstance(total_bound, int), f"jax_static_shapes: no packed_total_bound for packed key {key!r}"
            total = int(lens.sum())
            assert total <= total_bound, f"{key}: packed total {total} > declared bound {total_bound}"
            flat = numpy.zeros((total_bound,) + value.shape[2:], dtype=value.dtype)
            pos = 0
            for i in range(n):
                flat[pos : pos + lens[i]] = value[i, : lens[i]]
                pos += int(lens[i])
            out[f"{key}:packed"] = flat
            continue
        cap = dim_capacity.get(key)
        assert cap is not None, (
            f"jax_static_shapes: key {key!r} has a dynamic spatial dim but neither"
            f" dim_capacity[{key!r}] nor packing (packed_tensors) -- set one"
        )
        assert value.shape[1] <= cap, f"{key}: seq len {value.shape[1]} > dim_capacity {cap}"
        pad = [(0, batch_bound - value.shape[0]), (0, cap - value.shape[1])]
        pad += [(0, 0)] * (value.ndim - 2)
        out[key] = numpy.pad(value, pad)
    return out


def _to_jax(value: numpy.ndarray, *, dtype: str, device: Optional[str]):
    """
    :param value:
    :param dtype: the declared dtype, which the raw array must match (RF checks this)
    :param device:
    :return: the value as a JAX array
    """
    arr = numpy.asarray(value, dtype=dtype)
    if device:
        import jax

        # noinspection PyProtectedMember
        from returnn.jax.frontend._backend import _device_from_str

        # jnp.asarray would stage it twice, and TRACES PER SHAPE
        # (1.7 ms repeated vs 29.9 ms fresh),
        # and the shapes here are fresh almost every batch.
        return jax.device_put(arr, _device_from_str(device))
    return jnp.asarray(arr)
