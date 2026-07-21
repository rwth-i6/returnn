"""
From raw dict to extern_data tensor dict.
"""

from __future__ import annotations
from typing import Optional, Any, Union, Dict, List, Sequence
import numpy
import torch
from returnn.tensor import Tensor, TensorDict, Dim
import returnn.frontend as rf


def extern_data_template_from_config_opts(extern_data_dict: Dict[str, Any]) -> TensorDict:
    """
    :param extern_data_dict: as you would specify in the config
    :return: extern data tensor dict
    """
    extern_data = TensorDict()
    extern_data.update(extern_data_dict, auto_convert=True)
    if "seq_tag" not in extern_data.data:
        batch_dim = get_batch_dim_from_extern_data(extern_data)
        extern_data.data["seq_tag"] = Tensor(name="seq_tag", dtype="string", dims=[batch_dim])
    return extern_data


def raw_dict_to_extern_data(
    extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]],
    *,
    extern_data_template: TensorDict,
    device: Union[str, torch.device],
    float_dtype: Optional[Union[str, torch.dtype]] = None,
    with_eval_targets: bool = False,
) -> TensorDict:
    """
    :param extern_data_raw: This comes out of the DataLoader, via our collate_batch.
    :param extern_data_template: Specified via `extern_data` in the config.
    :param device: E.g. the GPU.
    :param float_dtype:
    :param with_eval_targets: if False, we skip all tensors with ``available_for_inference=False``.
    :return: tensor dict, like extern_data_template, but with raw tensors set to Torch tensors, on the right device.
    """
    if isinstance(float_dtype, str):
        float_dtype = getattr(torch, float_dtype)
        assert isinstance(float_dtype, torch.dtype)
    assert isinstance(extern_data_raw, dict) and extern_data_raw
    batch_dim = get_batch_dim_from_extern_data(extern_data_template)
    for dim in _get_dyn_dims_from_extern_data(extern_data_template):
        dim.reset_eager()  # they will be reset below
    if batch_dim.size is None and batch_dim.dyn_size_ext is None:
        batch_dim.dyn_size_ext = Tensor(batch_dim.name or "batch", dims=[], dtype="int32")
    extern_data = TensorDict()
    for k, data in extern_data_template.data.items():
        if not with_eval_targets and not data.available_for_inference:
            continue
        data = data.copy_template()
        # packing is inferred from the collate meta marker (see collate_batch), not the config
        if extern_data_raw.get(k + ":packed"):
            _set_packed_extern_data(
                data, k, extern_data_raw, batch_dim=batch_dim, device=device, float_dtype=float_dtype
            )
            extern_data.data[k] = data
            continue
        raw_tensor = extern_data_raw[k]
        assert len(raw_tensor.shape) == data.batch_ndim, f"ndim mismatch for {k}: {raw_tensor.shape} vs {data}"
        for i, dim in enumerate(data.dims):
            if dim.dimension is not None:
                assert dim.dimension == raw_tensor.shape[i], (
                    f"shape mismatch for {k}: {raw_tensor.shape} vs {data.batch_shape}"
                )
        if isinstance(raw_tensor, torch.Tensor):
            if raw_tensor.dtype.is_floating_point and float_dtype:
                raw_tensor = raw_tensor.to(dtype=float_dtype)
            data.dtype = str(raw_tensor.dtype).split(".")[-1]  # just overwrite for now...
            data.raw_tensor = raw_tensor.to(device, non_blocking=True)
        elif isinstance(raw_tensor, numpy.ndarray):
            data.raw_tensor = raw_tensor  # leave it as it is
        else:
            raise TypeError(f"Unexpected type {type(raw_tensor)} for {k} in extern_data_raw.")

        if batch_dim.dyn_size_ext is not None and batch_dim.dyn_size_ext.raw_tensor is None:
            batch_dim.dyn_size_ext.raw_tensor = torch.tensor(extern_data_raw[k].shape[0], dtype=torch.int32)

        # This has certain assumptions on the dataset, the data pipeline and collate_batch.
        # Namely, we expect that we get the batch dim in the first dim (see collate_batch).
        # We also expect that the sequence lengths are in the second dim, if it is dynamic.
        if (
            len(data.dims) >= 2
            and data.dims[1].size is None
            and (data.dims[1].dyn_size_ext is None or data.dims[1].dyn_size_ext.raw_tensor is None)
        ):
            assert k + ":seq_len" in extern_data_raw, (
                f"extern_data {data}, dyn spatial dim, missing {k}:seq_len in raw dict, check dataset or collate_batch"
            )
            size = extern_data_raw[k + ":seq_len"]
            # Sequence lengths have to be on CPU for the later call to rnn.pack_padded_sequence
            assert size.device.type == "cpu"
            size_dtype = str(size.dtype).split(".")[-1]
            if data.dims[1].dyn_size_ext is None:
                data.dims[1].dyn_size_ext = Tensor(data.dims[1].name or "time", dims=[batch_dim], dtype=size_dtype)
            data.dims[1].dyn_size_ext.dtype = size_dtype
            data.dims[1].dyn_size_ext.raw_tensor = size

        extern_data.data[k] = data

    return extern_data


def _set_packed_extern_data(
    data: Tensor,
    key: str,
    extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]],
    *,
    batch_dim: Dim,
    device: Union[str, torch.device],
    float_dtype: Optional[torch.dtype],
) -> None:
    """
    Fill data (a template copy) with a packed (concatenated, unpadded) raw tensor,
    from :func:`returnn.torch.data.pipeline.collate_batch`.
    ``extern_data_raw[key]`` is the flat [total, ...feature] buffer,
    ``extern_data_raw[key + ":seq_len"]`` the per-seq sizes.
    """
    raw_tensor = extern_data_raw[key]
    assert isinstance(raw_tensor, torch.Tensor), f"packed extern_data expects a torch tensor for {key}"
    opts = extern_data_raw[key + ":packed"]
    gap, align = opts["gap"], opts["align"]
    size = extern_data_raw[key + ":seq_len"]
    assert size.device.type == "cpu"
    n_seqs = int(size.shape[0])
    if batch_dim.dyn_size_ext is not None and batch_dim.dyn_size_ext.raw_tensor is None:
        batch_dim.dyn_size_ext.raw_tensor = torch.tensor(n_seqs, dtype=torch.int32)
    spatial = data.dims[1]
    size_dtype = str(size.dtype).split(".")[-1]
    if spatial.dyn_size_ext is None:
        spatial.dyn_size_ext = Tensor(spatial.name or "time", dims=[batch_dim], dtype=size_dtype)
    spatial.dyn_size_ext.dtype = size_dtype
    spatial.dyn_size_ext.raw_tensor = size
    if raw_tensor.dtype.is_floating_point and float_dtype:
        raw_tensor = raw_tensor.to(dtype=float_dtype)
    inner_dtype = str(raw_tensor.dtype).split(".")[-1]
    raw_tensor = raw_tensor.to(device, non_blocking=True)
    packed_dim = Dim(int(raw_tensor.shape[0]), name=(spatial.name or "time") + ":packed")
    inner = Tensor(key, dims=[packed_dim] + list(data.dims[2:]), dtype=inner_dtype, sparse_dim=data.sparse_dim)
    inner.raw_tensor = raw_tensor
    data.dtype = inner_dtype  # match the (possibly float_dtype-cast) buffer, like the padded path
    packed_t = rf.pack_import(
        inner, batch_dim=batch_dim, spatial_dim=spatial, packed_dim=packed_dim, feature_dim=data.feature_dim
    )
    if gap or align > 1:
        packed_t = rf.packed_regap(packed_t, gap, align=align)  # on device (raw_tensor already on device)
    data.raw_tensor = packed_t.raw_tensor


def raw_dict_can_split_batch(
    extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]],
    *,
    num_splits: int = 2,
) -> bool:
    """
    :param extern_data_raw: This comes out of the DataLoader, via our collate_batch.
    :param num_splits: into how many parts
    :return:
    """
    packed_keys = _packed_keys(extern_data_raw)
    if packed_keys:
        # the batch is the number of sequences (seq_len length), not dim 0 (= total frames)
        n_seqs = int(extern_data_raw[packed_keys[0] + ":seq_len"].shape[0])
        return n_seqs >= num_splits
    some_value = next(iter(extern_data_raw.values()))
    assert isinstance(some_value, (torch.Tensor, numpy.ndarray))
    batch_size = int(some_value.shape[0])
    return batch_size >= num_splits


def raw_dict_split_batch(
    extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]],
    *,
    splits: Union[Sequence[int], int],
) -> List[Dict[str, Union[torch.Tensor, numpy.ndarray]]]:
    """
    :param extern_data_raw: This comes out of the DataLoader, via our collate_batch.
    :param splits: either the list of resulting batch dims, or the num of splits
    :return: splitted extern_data_raw
    """
    if _packed_keys(extern_data_raw):
        return _raw_dict_split_batch_packed(extern_data_raw, splits=splits)
    some_value = next(iter(extern_data_raw.values()))
    assert isinstance(some_value, (torch.Tensor, numpy.ndarray))
    batch_size = int(some_value.shape[0])
    if isinstance(splits, int):
        splits = _make_split_seq_from_num_splits(splits, batch_size)
    assert isinstance(splits, (tuple, list)) and sum(splits) == batch_size
    res: List[Dict[str, Union[torch.Tensor, numpy.ndarray]]] = [{} for _ in range(len(splits))]
    for k, v in extern_data_raw.items():
        if not isinstance(v, (torch.Tensor, numpy.ndarray)):
            raise TypeError(f"got invalid value of type ({type(v).__name__}) for key {k!r}")
        offset = 0
        for i, split_size in enumerate(splits):
            res[i][k] = v[offset : offset + split_size] if v.ndim > 0 else v
            offset += split_size
    for res_ in res:
        res_: Dict[str, Union[torch.Tensor, numpy.ndarray]]
        for k, v in res_.items():
            if "%s:seq_len" % k in res_:
                max_len = torch.max(res_["%s:seq_len" % k])
                res_[k] = v[:, :max_len]
    return res


def _packed_keys(extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]]) -> List[str]:
    """:return: the data keys stored packed (marked by ``<key>:packed`` from collate_batch)"""
    suffix = ":packed"
    return [k[: -len(suffix)] for k in extern_data_raw if k.endswith(suffix)]


def _raw_dict_split_batch_packed(
    extern_data_raw: Dict[str, Union[torch.Tensor, numpy.ndarray]],
    *,
    splits: Union[Sequence[int], int],
) -> List[Dict[str, Union[torch.Tensor, numpy.ndarray]]]:
    """
    Split a packed raw dict along the sequences.
    Each packed data key is sliced at its OWN frame boundaries (cumsum of its seq_len),
    the seq_len and any per-seq key are sliced by the seq index,
    scalars and meta markers are copied.
    """
    packed = set(_packed_keys(extern_data_raw))
    n_seqs = int(extern_data_raw[next(iter(packed)) + ":seq_len"].shape[0])
    if isinstance(splits, int):
        splits = _make_split_seq_from_num_splits(splits, n_seqs)
    assert isinstance(splits, (tuple, list)) and sum(splits) == n_seqs
    # per packed key: exclusive frame offsets (a leading 0), so a seq range maps to a frame slice
    frame_offsets = {}
    for pk in packed:
        seq_len = extern_data_raw[pk + ":seq_len"]
        cumsum = torch.zeros(n_seqs + 1, dtype=torch.long)
        cumsum[1:] = torch.cumsum(seq_len.to(torch.long), dim=0)
        frame_offsets[pk] = cumsum
    res: List[Dict[str, Union[torch.Tensor, numpy.ndarray]]] = [{} for _ in range(len(splits))]
    seq_off = 0
    for i, n in enumerate(splits):
        seq_lo, seq_hi = seq_off, seq_off + n
        for k, v in extern_data_raw.items():
            if not isinstance(v, (torch.Tensor, numpy.ndarray)):
                res[i][k] = v  # scalar / meta marker (e.g. the ":packed" bool)
            elif k.endswith(":seq_len") and k[: -len(":seq_len")] in packed:
                res[i][k] = v[seq_lo:seq_hi]
            elif k in packed:
                cumsum = frame_offsets[k]
                res[i][k] = v[int(cumsum[seq_lo]) : int(cumsum[seq_hi])]  # frame slice
            elif v.ndim > 0:
                res[i][k] = v[seq_lo:seq_hi]  # static / non-packed per-seq key: by seq index
            else:
                res[i][k] = v
        seq_off = seq_hi
    return res


def _make_split_seq_from_num_splits(num_splits: int, total: int) -> List[int]:
    assert num_splits <= total
    remaining = total
    split_size = -(-total // num_splits)  # ceildiv
    res = []
    for i in range(num_splits):
        res.append(min(remaining, split_size))
        remaining -= res[-1]
    assert remaining == 0 and len(res) == num_splits and sum(res) == total
    return res


def _get_dyn_dims_from_extern_data(extern_data: TensorDict) -> List[Dim]:
    visited = set()
    res = []
    for k, v in extern_data.data.items():
        for dim in v.dims:
            if dim not in visited and dim.size is None:
                visited.add(dim)
                res.append(dim)
    return res


def get_batch_dim_from_extern_data(extern_data: TensorDict) -> Dim:
    """
    We expect that the batch dim is the first dim in any of the tensors.
    See collate_batch.

    We allow that the batch dim is not necessarily the global batch_dim object.
    We also allow that this is not even marked as batch dim (is_batch_dim() can be False).
    """
    batch_dim = next(iter(extern_data.data.values())).dims[0]
    return batch_dim
