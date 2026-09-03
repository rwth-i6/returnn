"""
Mixed precision (AMP): compute in a reduced float dtype while the parameters stay float32.

This exists because most backends have no autocast of their own.
PyTorch does (``torch.autocast`` intercepts at the aten level, see the ``torch_amp`` config option),
so the PyTorch backend does NOT use this; JAX and TF have nothing comparable,
and there the casts have to be placed by us.

Where the casts happen:

- **In the backend**, for the primitive ops: matmul and conv take the compute dtype,
  softmax / log_softmax / reductions / losses take float32.
- **In the RF modules**, for anything the backend never sees as one op.
  :class:`rf.LayerNorm` for example is a composition of reduce + rsqrt in RF,
  where PyTorch's autocast would see a single ``layer_norm`` and run it in float32.

The op categories follow the idea of PyTorch's autocast lists -- matmul-like ops in the reduced
dtype, numerically sensitive ops (normalization, exp/log-like, reductions, losses) in float32,
everything else in whatever dtype its inputs have -- but deliberately not their exact contents:
the granularity differs (RF primitives vs aten ops), and an exact match is not the goal.

The parameters are not touched: they stay float32 and are cast where they are USED,
which is inside the matmul/conv that consumes them. So the optimizer, its state
and the gradients are float32, as with PyTorch AMP.
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, TypeVar
from contextlib import contextmanager
from dataclasses import dataclass

from returnn.tensor import Tensor
from .array_ import cast
from .dtype import is_float_dtype


__all__ = [
    "AmpPolicy",
    "get_amp_policy",
    "set_amp_policy",
    "set_amp_policy_ctx",
    "amp_cast_compute",
    "amp_cast_float32",
]


@dataclass(frozen=True)
class AmpPolicy:
    """
    What mixed precision means for a run. See the module docstring.
    """

    compute_dtype: str  # e.g. "bfloat16"

    def __post_init__(self):
        if not is_float_dtype(self.compute_dtype):
            raise ValueError(f"AmpPolicy: compute_dtype {self.compute_dtype!r} is not a float dtype")


_amp_policy: Optional[AmpPolicy] = None


def get_amp_policy() -> Optional[AmpPolicy]:
    """
    :return: the active policy, or None when not in mixed precision
    """
    return _amp_policy


def set_amp_policy(policy: Optional[Union[AmpPolicy, str]]):
    """
    :param policy: the policy, or just the compute dtype (e.g. "bfloat16"), or None to disable
    """
    global _amp_policy
    if isinstance(policy, str):
        policy = AmpPolicy(compute_dtype=policy)
    assert policy is None or isinstance(policy, AmpPolicy)
    _amp_policy = policy


@contextmanager
def set_amp_policy_ctx(policy: Optional[Union[AmpPolicy, str]]):
    """
    :param policy: see :func:`set_amp_policy`
    """
    global _amp_policy
    old = _amp_policy
    try:
        set_amp_policy(policy)
        yield
    finally:
        _amp_policy = old


T = TypeVar("T", bound=Optional[Tensor])


def amp_cast_compute(*tensors: T) -> Union[T, Sequence[T]]:
    """
    Cast float tensors to the policy's compute dtype, for the matmul-like ops. No-op without a policy.

    :param tensors: None entries and non-float tensors are passed through
    :return: the tensor, or the tuple of them when more than one was given
    """
    policy = _amp_policy
    res = tensors if policy is None else tuple(_cast(x, policy.compute_dtype) for x in tensors)
    return res[0] if len(res) == 1 else res


def amp_cast_float32(*tensors: T) -> Union[T, Sequence[T]]:
    """
    Cast float tensors to float32, for the numerically sensitive ops. No-op without a policy.

    :param tensors: None entries and non-float tensors are passed through
    :return: the tensor, or the tuple of them when more than one was given
    """
    policy = _amp_policy
    res = tensors if policy is None else tuple(_cast(x, "float32") for x in tensors)
    return res[0] if len(res) == 1 else res


def _cast(x: Optional[Tensor], dtype: str) -> Optional[Tensor]:
    """
    :param x:
    :param dtype:
    :return: x in that dtype, if it is a float tensor at all
    """
    if x is None or x.dtype == dtype or not is_float_dtype(x.dtype):
        return x
    return cast(x, dtype)
