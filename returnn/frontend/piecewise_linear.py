"""
Piecewise linear function
"""

from __future__ import annotations
from typing import Union, Dict
import numpy as np
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


__all__ = ["PiecewiseLinear"]


class PiecewiseLinear(rf.Module):
    """
    Piecewise linear function.
    """

    def __init__(self, points: Dict[Union[int, float], Union[float, Tensor]]):
        """
        :param points: dict of key -> value pairs.
        """
        super().__init__()
        if not points:
            raise ValueError(f"{self}: points must not be empty")
        self._points_sorted = sorted(points.items())
        self.points_dim = Dim(len(self._points_sorted), name="pcw_schd_pieces")
        # Note: Use rf.Parameter to work around deepcopy issue. https://github.com/rwth-i6/returnn/issues/1541
        self._keys = rf.Parameter(
            rf.convert_to_tensor(
                np.array([k for k, _ in self._points_sorted], dtype=rf.get_default_float_dtype()),
                dims=[self.points_dim],
            ),
            auxiliary=True,
        )
        self._values = rf.Parameter(
            rf.stack([rf.convert_to_tensor(v) for _, v in self._points_sorted], out_dim=self.points_dim)[0],
            auxiliary=True,
        )

    def __call__(self, x: Tensor) -> Tensor:
        """
        :param x: (x_dims...) -> value in keys
        :return: y: (x_dims...,y_dims...) -> value in values
        """
        index = rf.search_sorted(self._keys, x, axis=self.points_dim)
        index = rf.clip_by_value(index, 1, self.points_dim.dimension - 1)
        x_start = rf.gather(self._keys, indices=index - 1)
        x_end = rf.gather(self._keys, indices=index)
        x_frac = (x - x_start) / (x_end - x_start)
        x_frac = rf.clip_by_value(x_frac, 0.0, 1.0)
        y_start = rf.gather(self._values, indices=index - 1)
        y_end = rf.gather(self._values, indices=index)
        return rf.lerp(y_start, y_end, x_frac)
