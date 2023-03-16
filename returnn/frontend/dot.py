"""
Dot / matmul
"""


from __future__ import annotations
from typing import Sequence, Union, TypeVar
from returnn.tensor import Tensor, Dim
from ._backend import get_backend_by_tensor

T = TypeVar("T")


# noinspection PyShadowingNames
def dot(a: Tensor[T], b: Tensor[T], *, reduce: Union[Dim, Sequence[Dim]]) -> Tensor[T]:
    """
    This performs a dot-product of two sources a and b.
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
    :return: result of dot product, Dim order: common axes as sorted in a, unique axes of a (in order),
        unique axes of b (in order)
    """
    rf = get_backend_by_tensor(a)
    return rf.dot(a=a, b=b, reduce=reduce)
