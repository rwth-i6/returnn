"""
Random number generator utilities
"""

from __future__ import annotations
from typing import Optional, Union, Sequence
import numpy
from returnn.tensor import Tensor, Dim
from ._backend import global_backend as _global_backend

__all__ = ["random"]


def random(
    *,
    dims: Sequence[Dim],
    dtype: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    distribution: str,
    mean: Optional[Union[int, float, Tensor]] = None,
    stddev: Optional[Union[int, float, Tensor]] = None,
    bound: Optional[Union[int, float, Tensor]] = None,
    minval: Optional[Union[int, float, Tensor]] = None,
    maxval: Optional[Union[int, float, Tensor]] = None,
    seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
    algorithm: Optional[str] = None,
    explicit_state: Optional[Tensor] = None,
    auto_update_state: Optional[bool] = None,
    static: Optional[bool] = None,
) -> Tensor:
    """
    Generates random numbers from uniform or normal or truncated normal distribution.

    This uses the TensorFlow stateless random ops internally, i.e. all the state handling is explicit.
    The state var can be explicitly provided and initialized via :class:`RandomStateInitLayer`,
    or when not provided it will be automatically created.

    There are two possible distinct use cases:

    - For any randomness in the model, e.g. dropout. So each ``session.run`` step will produce a new random number
      and advance the random state.
    - To initialize parameters via the config, using :class:`VariableLayer` with the ``init_by_layer`` option.
      This will only be called once when initializing the parameters.
      For this use case, we do not want to keep a random state var.
      You can just pass ``static=False``.
      Alternatively you could also pass the output of a :class:`RandomStateInitLayer` as ``state``.

    :param Sequence[Dim] dims:
    :param str dtype:
    :param sparse_dim:
    :param str distribution: "uniform", "normal" or "truncated_normal"
    :param int|float|Tensor|None mean:
    :param int|float|Tensor|None stddev:
    :param int|float|Tensor|None bound: for uniform, defining the range [-bound, bound)
    :param int|float|Tensor|None minval: for uniform
    :param int|float|Tensor|None maxval: for uniform
    :param int|list[int]|numpy.ndarray|None seed: If not given, uses self.network.random.randint,
      i.e. then it is controlled by the global seed setting, and every layer would get its own seed.
      If you specify it explicitly, make sure every :class:`RandomLayer` uses a different seed,
      otherwise you would get the same random numbers everywhere.
    :param str|tf.random.Algorithm|None algorithm: see :class:`RandomStateInitLayer`
    :param Tensor|None explicit_state: You can pass the state explicitly here.
      If not given, will be created automatically, and updated automatically.
      You could pass a :class:`VariableLayer` with initial value via :class:`RandomStateInitLayer`,
      or directly a :class:`RandomStateInitLayer`.
      If auto_update_state is True, it must be a variable,
      and every time a new random number is created, this variable is updated.
      Otherwise (default), it will not be updated automatically.
    :param bool|None auto_update_state: only used when you pass an explicit state
    :param bool|None static: if no state at all should be used. it just relies on the seed then.
    :return: layer
    """
    return _global_backend.random(
        dims=dims,
        dtype=dtype,
        sparse_dim=sparse_dim,
        distribution=distribution,
        mean=mean,
        stddev=stddev,
        bound=bound,
        minval=minval,
        maxval=maxval,
        seed=seed,
        algorithm=algorithm,
        explicit_state=explicit_state,
        auto_update_state=auto_update_state,
        static=static,
    )
