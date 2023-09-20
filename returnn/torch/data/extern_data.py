"""
From raw dict to extern_data tensor dict.
"""

from __future__ import annotations
from typing import Any, Union, Dict, List
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
) -> TensorDict:
    """
    :param extern_data_raw: This comes out of the DataLoader.
    :param extern_data_template: Specified via `extern_data` in the config.
    :param device: E.g. the GPU.
    :return: tensor dict, like extern_data_template, but with raw tensors set to Torch tensors, on the right device.
    """
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
