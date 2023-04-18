"""
Dot / matmul
"""


from __future__ import annotations
from typing import Sequence, Union, TypeVar
from returnn.tensor import Tensor, Dim

T = TypeVar("T")

__all__ = ["matmul", "dot"]


# noinspection PyShadowingNames
def matmul(a: Tensor[T], b: Tensor[T], *, reduce: Union[Dim, Sequence[Dim]], use_mask: bool = True) -> Tensor[T]:
    """
    This performs a batched matmul of two sources a and b
    (non-batched matmul and dot product are special cases).
    The underlying operation is a batched matmul (shared..., I, J) * (shared..., J, K) -> (shared..., I, K).
    The inputs a and b are transformed internally into the required shapes in the following way:
    The axis J is specified via the Dim given as 'reduce'. If multiple reduce Dims are given the corresponding axes
    are merged into one before the matmul via a reshape. All other matching Dims in a and b will be treated as
    batch dimensions ('shared...'). Dims unique to a and b define the axes I and K, respectively. (Multiple or no
    unique axes in a and b are supported too.)

    Depending on which Dims exist in a, b and reduce this dot operation can be used to compute scaling, scalar
    product, outer product, matrix-vector multiplication, matrix-matrix multiplication etc. (all possibly batched).

    :param a:
    :param b:
    :param reduce: Dims over which to perform the product, have to be present in both a and b
    :param use_mask: If the reduction is over dynamic axes, to get the correct sum reduction,
        we need to apply masking to one of the inputs. This is done automatically.
        By disabling this flag, this would be disabled.
    :return: result of dot product, Dim order: common axes as sorted in a, unique axes of a (in order),
        unique axes of b (in order)
    """
    # noinspection PyProtectedMember
    return a._raw_backend.matmul(a=a, b=b, reduce=reduce, use_mask=use_mask)


# alias for some older code
dot = matmul
