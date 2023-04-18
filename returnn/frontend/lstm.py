"""
Provides the :class:`LSTM` module.
"""

from __future__ import annotations

from typing import Tuple, TypeVar

import returnn.frontend as rf
from returnn.frontend import State
from returnn.tensor import Tensor, Dim


T = TypeVar("T")

__all__ = ["LSTM"]


class LSTM(rf.Module):
    """
    LSTM module.
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        *,
        with_bias: bool = True,
    ):
        """
        Code to initialize the LSTM module.
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.ff_weights = rf.Parameter((self.out_dim * 4, self.in_dim))  # type: Tensor[T]
        self.ff_weights.initial = rf.init.Glorot()
        self.recurrent_weights = rf.Parameter((self.out_dim * 4, self.out_dim))  # type: Tensor[T]
        self.recurrent_weights.initial = rf.init.Glorot()

        self.ff_biases = None
        self.recurrent_biases = None
        if with_bias:
            self.ff_biases = rf.Parameter((self.out_dim * 4,))  # type: Tensor[T]
            self.ff_biases.initial = 0.0
            self.recurrent_biases = rf.Parameter((self.out_dim * 4,))  # type: Tensor[T]
            self.recurrent_biases.initial = 0.0

    def __call__(self, source: Tensor[T], state: State) -> Tuple[Tensor, State]:
        """
        Forward call of the LSTM.

        :param source: Tensor of size ``[*, in_dim]``.
        :param state: State of the LSTM. Contains two :class:`Tensor`: ``state.h`` as the hidden state,
            and ``state.c`` as the cell state. Both are of shape ``[out_dim]``.
        :return: Output of forward as a :class:`Tensor` of size ``[*, out_dim]``, and next LSTM state.
        """
        if not state.h or not state.c:
            raise ValueError(f"{self}: state {state} needs attributes ``h`` (hidden) and ``c`` (cell).")
        if self.in_dim not in source.dims_set:
            raise ValueError(f"{self}: input {source} does not have in_dim {self.in_dim}")

        # noinspection PyProtectedMember
        result, new_state = source._raw_backend.lstm(
            source=source,
            state=state,
            ff_weights=self.ff_weights,
            ff_biases=self.ff_biases,
            rec_weights=self.recurrent_weights,
            rec_biases=self.recurrent_biases,
            spatial_dim=self.in_dim,
            out_dim=self.out_dim,
        )

        return result, new_state
