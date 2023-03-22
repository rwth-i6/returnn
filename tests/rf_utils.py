"""
RETURNN frontend (returnn.frontend) utils
"""

from __future__ import annotations
from typing import Callable
import returnn.frontend as rf


def run_model(get_model: Callable[[], rf.Module]):
    """run"""
    get_model  # noqa  # TODO
    raise NotImplementedError  # TODO


def run_model_torch(get_model: Callable[[], rf.Module]):
    """run"""
    get_model  # noqa  # TODO
    raise NotImplementedError  # TODO


def run_model_net_dict_tf(get_model: Callable[[], rf.Module]):
    """run"""
    get_model  # noqa  # TODO
    raise NotImplementedError  # TODO
