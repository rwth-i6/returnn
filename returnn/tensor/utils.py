"""
Some helper utils.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Dict, List, Tuple
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
            if dim.is_batch_dim() and dim.dyn_size_ext is None:
                dim.dyn_size_ext = Tensor("batch", [], dtype="int32")
            if dim.is_dynamic() and dim.dyn_size_ext is None:
                dim.dyn_size_ext = Tensor(dim.name or "time", dims=[batch_dim], dtype="int32")
            if dim.dyn_size_ext is None:
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
        elif x.dtype == "bfloat16":
            # Numpy does not support bfloat16, will later be casted to bfloat16
            x.raw_tensor = rnd.normal(0.0, 1.0, size=shape).astype("float32")
        elif x.dtype.startswith("complex"):
            real = rnd.normal(0.0, 1.0, size=shape)
            imag = rnd.normal(0.0, 1.0, size=shape)
            x.raw_tensor = (real + 1j * imag).astype(x.dtype)
        else:
            raise NotImplementedError(f"not implemented for {x} dtype {x.dtype}")
        filled = True

    assert isinstance(x.raw_tensor, numpy.ndarray)

    return filled


def tensor_dict_dims_random_seq_len_min_max(
    tensor_dict: TensorDict,
    seq_lens: Union[None, int, Tuple[int, int], Dict[Union[str, Dim, None], Union[int, Tuple[int, int]]]] = None,
) -> Tuple[List[Dim], Dict[Dim, Tuple[int, int]]]:
    """
    This is specifically intended to prepare the list of all dynamic dims from the tensor dict
    and the seq_len_min_max for :func:`get_random_seq_lens_for_dyn_dims`.

    :param tensor_dict:
    :param seq_lens: either fixed seq len, or take randint. per data key, or per dim, or same for all
    :return: dims, seq_len_min_max
    """
    if seq_lens is None:
        seq_lens = {}
    if not isinstance(seq_lens, dict):
        seq_lens = {None: seq_lens}
    seq_lens: Dict[Union[str, Dim, None], Union[int, Tuple[int, int]]]

    # Collect all dyn dim tags, including derived_from_op ones.
    # The order will be sorted such that derived_from_op roots come first.
    visited_dims = set()
    dims = []
    seq_len_min_max = {}  # Also collect seq_len_min_max.
    for k, v in tensor_dict.data.items():
        for dim in v.dims:
            if dim.is_dynamic() and dim not in visited_dims and not dim.is_batch_dim():
                queue = [dim]
                offset = len(dims)
                while queue:
                    dim = queue.pop(0)
                    if not dim.is_dynamic():
                        continue
                    if dim in visited_dims:
                        continue
                    visited_dims.add(dim)
                    dims.insert(offset, dim)
                    dim.reset_batch_and_raw()
                    if dim.derived_from_op:
                        queue.extend(dim.derived_from_op.inputs)
                    else:
                        # Need to specify seq_len_min_max.
                        if dim in seq_lens or k in seq_lens or None in seq_lens:
                            if dim in seq_lens:
                                size = seq_lens[dim]
                            elif k in seq_lens:
                                size = seq_lens[k]
                            else:
                                size = seq_lens[None]
                            if isinstance(size, int):
                                size = (size, size)
                            else:
                                assert (
                                    isinstance(size, tuple)
                                    and len(size) == 2
                                    and all(isinstance(s, int) for s in size)
                                    and 0 <= size[0] <= size[1]
                                ), f"invalid size {size!r} in seq lens {seq_lens}"
                        else:
                            if v.shape in {(None,), (None, 1)} and v.dtype.startswith("float"):
                                # Assume raw audio data samples, take longer seq lens by default, assume 16khz.
                                size = (1_000, 8_000)
                            else:
                                size = (5, 15)
                        seq_len_min_max[dim] = size

    return dims, seq_len_min_max


def get_random_seq_lens_for_dyn_dims(
    dims: Sequence[Dim],
    seq_len_min_max: Dict[Dim, Tuple[int, int]],
    *,
    batch_size: int = 1,
    rnd: Union[int, numpy.random.RandomState] = 1337,
) -> Dict[Dim, numpy.ndarray]:
    """
    Make random seq lens for dims.

    Note that dim tags are not actually modified here,
    as we need to have this in a safe way,
    which might run in parallel to the main thread.

    :param dims: Note that the order matter, as we use complete_dyn_size() (or equivalent).
    :param seq_len_min_max:
    :param batch_size:
    :param rnd:
    """
    if not isinstance(rnd, numpy.random.RandomState):
        rnd = numpy.random.RandomState(rnd)

    gen_dims = {}
    for dim in dims:
        if dim not in gen_dims:
            if dim.derived_from_op:
                # If we get a KeyError for the following, the order of dims is invalid.
                values = [gen_dims[dim_] for dim_ in dim.derived_from_op.inputs]
                kind = dim.derived_from_op.kind
                a = values[0]
                for b in values[1:]:
                    if kind == "add":
                        a = numpy.maximum(a + b, 0)
                    elif kind == "sub":
                        a = numpy.maximum(a - b, 0)
                    elif kind == "mul":
                        a = a * b
                    elif kind in ("floordiv", "truediv"):  # truediv assumes there is no remainder
                        a = a // b
                    elif kind == "ceildiv":
                        a = -(-a // b)
                    else:
                        raise ValueError("unknown op kind %r" % kind)
                gen_dims[dim] = a
                continue

            min_, max_ = seq_len_min_max[dim]
            gen_dims[dim] = rnd.randint(min_, max_ + 1, size=[batch_size], dtype=numpy.int32)

    return gen_dims
