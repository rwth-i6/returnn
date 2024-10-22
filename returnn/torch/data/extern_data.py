"""
From raw dict to extern_data tensor dict.
"""

from __future__ import annotations
from typing import Optional, Any, Union, Dict, List, Sequence
import numpy
import torch
from returnn.tensor import Tensor, TensorDict, Dim


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
) -> TensorDict:
    """
    :param extern_data_raw: This comes out of the DataLoader, via our collate_batch.
    :param extern_data_template: Specified via `extern_data` in the config.
    :param device: E.g. the GPU.
    :param float_dtype:
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
        data = data.copy_template()
        raw_tensor = extern_data_raw[k]
        assert len(raw_tensor.shape) == data.batch_ndim, f"ndim mismatch for {k}: {raw_tensor.shape} vs {data}"
        for i, dim in enumerate(data.dims):
            if dim.dimension is not None:
                assert (
                    dim.dimension == raw_tensor.shape[i]
                ), f"shape mismatch for {k}: {raw_tensor.shape} vs {data.batch_shape}"
        if isinstance(raw_tensor, torch.Tensor):
            if raw_tensor.dtype.is_floating_point and float_dtype:
                raw_tensor = raw_tensor.to(dtype=float_dtype)
            data.dtype = str(raw_tensor.dtype).split(".")[-1]  # just overwrite for now...
            data.raw_tensor = raw_tensor.to(device)
        elif isinstance(raw_tensor, numpy.ndarray):
            data.raw_tensor = raw_tensor  # leave it as it is
        else:
            raise TypeError(f"Unexpected type {type(raw_tensor)} for {k} in extern_data_raw.")

        if batch_dim.dyn_size_ext and batch_dim.dyn_size_ext.raw_tensor is None:
            batch_dim.dyn_size_ext.raw_tensor = torch.tensor(extern_data_raw[k].shape[0], dtype=torch.int32)

        # This has certain assumptions on the dataset, the data pipeline and collate_batch.
        # Namely, we expect that we get the batch dim in the first dim (see collate_batch).
        # We also expect that the sequence lengths are in the second dim, if it is dynamic.
        if (
            len(data.dims) >= 2
            and data.dims[1].size is None
            and (not data.dims[1].dyn_size_ext or data.dims[1].dyn_size_ext.raw_tensor is None)
        ):
            assert k + ":seq_len" in extern_data_raw, (
                f"extern_data {data}, dyn spatial dim, missing {k}:seq_len in raw dict, "
                f"check dataset or collate_batch"
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
            res[i][k] = v[offset : offset + split_size]
            offset += split_size
    for res_ in res:
        res_: Dict[str, Union[torch.Tensor, numpy.ndarray]]
        for k, v in res_.items():
            if "%s:seq_len" % k in res_:
                max_len = torch.max(res_["%s:seq_len" % k])
                res_[k] = v[:, :max_len]
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
