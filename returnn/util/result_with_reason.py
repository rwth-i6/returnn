"""
ResultWithReason class.
"""

from __future__ import annotations

from typing import TypeVar, Generic, Optional
from dataclasses import dataclass


T = TypeVar("T")


@dataclass
class ResultWithReason(Generic[T]):
    """
    This is a wrapper class for a result value, which can also have a reason.
    """

    result: T
    reason: Optional[str] = None
