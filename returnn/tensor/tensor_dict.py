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

    def update(self, data: _DataAutoConvertT, *, auto_convert: bool = False):
        """update"""
        if isinstance(data, dict):
            for key, value in data.items():
                if auto_convert:
                    value = _convert_to_tensor(value, name=key)
                else:
                    assert isinstance(value, Tensor)
                self.data[key] = value
        elif isinstance(data, (list, tuple)):
            for value in data:
                if auto_convert:
                    value = _convert_to_tensor(value)
                else:
                    assert isinstance(value, Tensor)
                self.data[value.name] = value
        else:
            raise TypeError(f"invalid `data` type: {type(data)}")

    def __getitem__(self, item: str) -> Tensor:
        return self.data[item]


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
