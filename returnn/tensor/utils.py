"""
Some helper utils.
"""


from __future__ import annotations
from typing import Optional, Union, Dict
import numpy
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim


def tensor_dict_fill_random_numpy_(
    tensor_dict: TensorDict,
    *,
    rnd: Union[int, numpy.random.RandomState] = 42,
    dyn_dim_max_sizes: Optional[Dict[Dim, int]] = None,
    dyn_dim_min_sizes: Optional[Dict[Dim, int]] = None,
):
    """
    Random fill with NumPy arrays.

    :param tensor_dict:
    :param rnd:
    :param dyn_dim_max_sizes: you can specify max sizes for dim tags with dynamic sizes.
        The fill random code makes sure that there is at least one entry where we reach the max size,
        so that the dim value will be the max size.
    :param dyn_dim_min_sizes:
    """
    if not isinstance(rnd, numpy.random.RandomState):
        rnd = numpy.random.RandomState(rnd)
    for v in tensor_dict.data.values():
        tensor_fill_random_numpy_(v, rnd=rnd, dyn_dim_max_sizes=dyn_dim_max_sizes, dyn_dim_min_sizes=dyn_dim_min_sizes)


def tensor_fill_random_numpy_(
    x: Tensor,
    *,
    min_val: int = 0,
    max_val: Optional[int] = None,
    rnd: numpy.random.RandomState,
    dyn_dim_max_sizes: Optional[Dict[Dim, int]] = None,
    dyn_dim_min_sizes: Optional[Dict[Dim, int]] = None,
) -> bool:
    """fill. return whether sth was filled"""
    if dyn_dim_max_sizes is None:
        dyn_dim_max_sizes = {}
    if dyn_dim_min_sizes is None:
        dyn_dim_min_sizes = {}
    filled = False
    while True:
        have_unfilled = False
        filled_this_round = False

        for dim in x.dims:
            if dim.is_batch_dim() and not dim.dyn_size_ext:
                dim.dyn_size_ext = Tensor("batch", [], dtype="int32")
            if dim.is_dynamic() and not dim.dyn_size_ext:
                dim.dyn_size_ext = Tensor(dim.name or "time", dims=[batch_dim], dtype="int32")
            if not dim.dyn_size_ext:
                continue
            if tensor_fill_random_numpy_(
                dim.dyn_size_ext,
                min_val=dyn_dim_min_sizes.get(dim, 2),
                max_val=dyn_dim_max_sizes.get(dim, None),
                rnd=rnd,
                dyn_dim_max_sizes=dyn_dim_max_sizes,
            ):
                if dim in dyn_dim_max_sizes:
                    # Make sure at least one of the dyn sizes matches the max size.
                    i = rnd.randint(0, dim.dyn_size_ext.raw_tensor.size)
                    dim.dyn_size_ext.raw_tensor.flat[i] = dyn_dim_max_sizes[dim]
                    if dim in dyn_dim_min_sizes:
                        j = rnd.randint(0, dim.dyn_size_ext.raw_tensor.size - 1)
                        if j >= i:
                            j += 1
                        dim.dyn_size_ext.raw_tensor.flat[j] = dyn_dim_min_sizes[dim]
                elif dim in dyn_dim_min_sizes:
                    raise Exception(f"also define {dim} in dyn_dim_max_sizes, not just dyn_dim_min_sizes")
                filled = True
                filled_this_round = True
            if dim.dyn_size_ext.raw_tensor is None:
                have_unfilled = True
            elif not isinstance(dim.dyn_size_ext.raw_tensor, numpy.ndarray):
                have_unfilled = True

        if have_unfilled:
            assert filled_this_round, f"should have filled something, {x}"

        if not have_unfilled:
            break

    if x.raw_tensor is not None:
        if not isinstance(x.raw_tensor, numpy.ndarray):
            x.raw_tensor = None

    if x.raw_tensor is None:
        shape = [d.get_dim_value() for d in x.dims]
        if x.dtype.startswith("int"):
            if max_val is None:
                max_val = rnd.randint(5, 20)
            if x.sparse_dim and x.sparse_dim.dimension is not None:
                max_val = x.sparse_dim.dimension
            x.raw_tensor = rnd.randint(min_val, max_val, size=shape, dtype=x.dtype)
        elif x.dtype == "bool":
            x.raw_tensor = rnd.randint(0, 2, size=shape, dtype=x.dtype)
        elif x.dtype.startswith("float"):
            x.raw_tensor = rnd.normal(0.0, 1.0, size=shape).astype(x.dtype)
        elif x.dtype.startswith("complex"):
            real = rnd.normal(0.0, 1.0, size=shape)
            imag = rnd.normal(0.0, 1.0, size=shape)
            x.raw_tensor = (real + 1j * imag).astype(x.dtype)
        else:
            raise NotImplementedError(f"not implemented for {x} dtype {x.dtype}")
        filled = True

    assert isinstance(x.raw_tensor, numpy.ndarray)

    return filled
