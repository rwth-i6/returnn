"""
Dict of tensors.

For example the extern_data in the user config is such a dict,
describing the data coming from the dataset (after batch preparation).

We also might have model_outputs in the user config.
(https://github.com/rwth-i6/returnn/issues/1166)
"""

from __future__ import annotations
from typing import Optional, Union, Any, Type, Dict, Sequence, List
from .tensor import Tensor
from .dim import Dim


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

    def __contains__(self, item: str) -> bool:
        return item in self.data

    def __getitem__(self, item: str) -> Tensor:
        return self.data[item]

    def reset_content(self):
        """reset content, i.e. all raw_tensor's to None, including dyn_size_ext of dim tags"""
        dims = []
        dims_set = set()
        for value in self.data.values():
            value.reset()
            for dim in value.dims:
                if dim not in dims_set:
                    dims_set.add(dim)
                    dims.append(dim)
        for dim in dims:
            dim.reset_batch_and_raw()

    def copy_template(self) -> TensorDict:
        """copy template"""
        return TensorDict({k: v.copy_template() for k, v in self.data.items()})

    def as_raw_tensor_dict(
        self,
        *,
        include_const_sizes: bool = False,
        include_scalar_dyn_sizes: bool = True,
        exclude_duplicate_dims: bool = False,
        expected_value_type: Union[Type, Sequence[Type]] = object,
    ) -> Dict[str, Any]:
        """
        :return: dict of raw tensors, including any sequence lengths / dynamic sizes
        """
        assert not (include_const_sizes and not include_scalar_dyn_sizes)
        visited_dims = set()
        out = {}
        for key, value in self.data.items():
            assert key not in out
            assert isinstance(value.raw_tensor, expected_value_type), (
                f"key {key} {value}: unexpected {type(value.raw_tensor)}, expected {expected_value_type}"
            )
            out[key] = value.raw_tensor
            for i, dim in enumerate(value.dims):
                if exclude_duplicate_dims and dim in visited_dims:
                    continue
                key_ = f"{key}:size{i}"
                assert key_ not in out
                if dim.is_batch_dim() and (dim.dyn_size_ext is None or dim.dyn_size_ext.raw_tensor is None):
                    if include_scalar_dyn_sizes:
                        dim_value = dim.get_dim_value()
                        assert isinstance(dim_value, expected_value_type), (
                            f"key {key_} {dim}: unexpected {type(dim_value)}, expected {expected_value_type}"
                        )
                        out[key_] = dim_value
                elif dim.dyn_size_ext is not None:
                    if include_scalar_dyn_sizes or dim.dyn_size_ext.dims:
                        assert isinstance(dim.dyn_size_ext.raw_tensor, expected_value_type), (
                            f"key {key_} {dim} {dim.dyn_size_ext}:"
                            f" unexpected {type(dim.dyn_size_ext.raw_tensor)}, expected {expected_value_type}"
                        )
                        out[key_] = dim.dyn_size_ext.raw_tensor
                elif dim.size is not None:
                    if include_scalar_dyn_sizes and include_const_sizes:
                        assert isinstance(dim.size, expected_value_type), (
                            f"key {key_} {dim}: unexpected {type(dim.size)}, expected {expected_value_type}"
                        )
                        out[key_] = dim.size
                else:
                    raise Exception(f"cannot handle dim: {dim}")
                visited_dims.add(dim)
        return out

    def assign_from_raw_tensor_dict_(
        self,
        raw_tensor_dict: Dict[str, Any],
        *,
        with_scalar_dyn_sizes: bool = True,
        duplicate_dims_are_excluded: bool = False,
    ):
        """
        :param raw_tensor_dict: dict of raw tensors, including any sequence lengths / dynamic sizes
        :param with_scalar_dyn_sizes: `include_scalar_dyn_sizes` was used in :func:`as_raw_tensor_dict`
        :param duplicate_dims_are_excluded: `exclude_duplicate_dims` was used in :func:`as_raw_tensor_dict`
        """
        visited_dims = set()
        for key, value in self.data.items():
            assert key in raw_tensor_dict, f"key {key} not in raw_tensor_dict {list(raw_tensor_dict.keys())}"
            value.raw_tensor = raw_tensor_dict[key]
            for i, dim in enumerate(value.dims):
                dim: Dim
                if duplicate_dims_are_excluded and dim in visited_dims:
                    continue
                key_ = f"{key}:size{i}"
                dim.reset_raw(only_self=True)
                if dim.is_batch_dim() and dim.dyn_size_ext is None:
                    dim.dyn_size_ext = Tensor("batch", [], dtype="int32")
                if dim.dyn_size_ext is not None:
                    if not with_scalar_dyn_sizes and not dim.dyn_size_ext.dims:
                        pass
                    else:
                        assert key_ in raw_tensor_dict, f"keys: f{raw_tensor_dict.keys()}"
                        dim.dyn_size_ext.raw_tensor = raw_tensor_dict[key_]
                else:
                    if key_ in raw_tensor_dict:
                        assert dim.size == raw_tensor_dict[key_]
                visited_dims.add(dim)

    def all_dims(self) -> List[Dim]:
        """
        :return: list of dims
        """
        visited_dims = set()
        out = []
        for key, value in self.data.items():
            for dim in value.dims:
                if dim in visited_dims:
                    continue
                out.append(dim)
                visited_dims.add(dim)
        return out


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
