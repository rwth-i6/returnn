"""
Dict of tensors.

For example the extern_data in the user config is such a dict,
describing the data coming from the dataset (after batch preparation).

We also might have model_outputs in the user config.
(https://github.com/rwth-i6/returnn/issues/1166)
"""

from __future__ import annotations
from typing import Optional, Union, Any, Dict, Sequence
from .tensor import Tensor


_TensorT = Union[Tensor, Dict[str, Any]]
_DataAutoConvertT = Union[Dict[str, _TensorT], Sequence[_TensorT]]
_DataStrictT = Union[Dict[str, Tensor], Sequence[Tensor]]


class TensorDict:
    """dict of tensors"""

    def __init__(self, data: Optional[_DataStrictT] = None):
        self.data = {}  # type: Dict[str, Tensor]
        if data:
            self.update(data)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

    def update(self, data: Union[_DataAutoConvertT, TensorDict], *, auto_convert: bool = False):
        """update"""
        if isinstance(data, TensorDict):
            for key, value in data.data.items():
                self.data[key] = value.copy()
        elif isinstance(data, dict):
            for key, value in data.items():
                if auto_convert:
                    value = _convert_to_tensor(value, name=key)
                else:
                    assert isinstance(value, Tensor)
                self.data[key] = value.copy()
        elif isinstance(data, (list, tuple)):
            for value in data:
                if auto_convert:
                    value = _convert_to_tensor(value)
                else:
                    assert isinstance(value, Tensor)
                self.data[value.name] = value.copy()
        else:
            raise TypeError(f"invalid `data` type: {type(data)}")

    def __getitem__(self, item: str) -> Tensor:
        return self.data[item]

    def copy_template(self) -> TensorDict:
        """copy template"""
        return TensorDict({k: v.copy_template() for k, v in self.data.items()})

    def as_raw_tensor_dict(self, *, include_const_sizes: bool = False) -> Dict[str, Any]:
        """
        :return: dict of raw tensors, including any sequence lengths / dynamic sizes
        """
        out = {}
        for key, value in self.data.items():
            assert key not in out
            out[key] = value.raw_tensor
            for i, dim in enumerate(value.dims):
                key_ = f"{key}:size{i}"
                assert key_ not in out
                if dim.is_batch_dim() and (not dim.dyn_size_ext or dim.dyn_size_ext.raw_tensor is None):
                    out[key_] = dim.get_dim_value()
                elif dim.dyn_size_ext:
                    out[key_] = dim.dyn_size_ext.raw_tensor
                elif dim.size is not None:
                    if include_const_sizes:
                        out[key_] = dim.size
                else:
                    raise Exception(f"cannot handle dim: {dim}")
        return out

    def assign_from_raw_tensor_dict_(self, raw_tensor_dict: Dict[str, Any]):
        """
        :param raw_tensor_dict: dict of raw tensors, including any sequence lengths / dynamic sizes
        """
        for key, value in self.data.items():
            assert key in raw_tensor_dict
            value.raw_tensor = raw_tensor_dict[key]
            for i, dim in enumerate(value.dims):
                key_ = f"{key}:size{i}"
                if dim.is_batch_dim() and not dim.dyn_size_ext:
                    dim.dyn_size_ext = Tensor("batch", [], dtype="int32")
                if dim.dyn_size_ext:
                    assert key_ in raw_tensor_dict
                    dim.dyn_size_ext.raw_tensor = raw_tensor_dict[key_]
                else:
                    if key_ in raw_tensor_dict:
                        assert dim.size == raw_tensor_dict[key_]


def _convert_to_tensor(opts: _TensorT, *, name: Optional[str] = None) -> Tensor:
    """
    :param opts:
    """
    if isinstance(opts, Tensor):
        return opts
    assert isinstance(opts, dict)
    opts = opts.copy()
    if name:
        opts["name"] = name
    else:
        assert "name" in opts, f"missing `name` in Tensor {opts!r}"
    return Tensor(**opts)
