"""
Reference implementation of edit distance.
"""

from typing import Union, Sequence
import numpy
from returnn.tensor import Tensor, Dim


def edit_distance_ref(a: Tensor, a_spatial_dim: Dim, b: Tensor, b_spatial_dim: Dim) -> Tensor:
    """
    Reference implementation for edit distance.
    """
    batch_dim = a.dims[0]
    assert a.dims == (batch_dim, a_spatial_dim) and b.dims == (batch_dim, b_spatial_dim)
    res = []
    for i in range(batch_dim.dimension):
        assert a_spatial_dim.dyn_size[i] <= a.raw_tensor.size(1)
        assert b_spatial_dim.dyn_size[i] <= b.raw_tensor.size(1)
        res.append(
            edit_distance_ref_np_b1(
                a.raw_tensor[i, : a_spatial_dim.dyn_size[i]], b.raw_tensor[i, : b_spatial_dim.dyn_size[i]]
            )
        )
    return Tensor("edit_dist", dims=[batch_dim], dtype="int32", raw_tensor=numpy.array(res, dtype=numpy.int32))


def edit_distance_ref_np(
    a: numpy.ndarray, a_seq_lens: numpy.ndarray, b: numpy.ndarray, b_seq_lens: numpy.ndarray
) -> numpy.ndarray:
    """
    Reference implementation for edit distance.
    """
    batch_size = a.shape[0]
    assert a.shape == (batch_size, a_seq_lens.max()) and b.shape == (batch_size, b_seq_lens.max())
    res = []
    for i in range(batch_size):
        assert a_seq_lens[i] <= a.shape[1]
        assert b_seq_lens[i] <= b.shape[1]
        res.append(edit_distance_ref_np_b1(a[i, : a_seq_lens[i]], b[i, : b_seq_lens[i]]))
    return numpy.array(res, dtype=numpy.int32)


def edit_distance_ref_np_b1(a: Union[Sequence[int], numpy.ndarray], b: Union[Sequence[int], numpy.ndarray]) -> int:
    """
    Reference implementation for edit distance, single batch.
    """
    n = len(a) + 1
    m = len(b) + 1
    d = numpy.zeros((n, m), dtype=numpy.int32)
    for i in range(n):
        d[i, 0] = i
    for j in range(m):
        d[0, j] = j
    for j in range(1, m):
        for i in range(1, n):
            if a[i - 1] == b[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(
                    d[i - 1, j] + 1,  # deletion
                    d[i, j - 1] + 1,  # insertion
                    d[i - 1, j - 1] + 1,  # substitution
                )
    return int(d[n - 1, m - 1])
