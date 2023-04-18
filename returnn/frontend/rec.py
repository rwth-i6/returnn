"""
Provides the :class:`LSTM` module.
"""

from __future__ import annotations

from typing import Tuple

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


__all__ = ["LSTM", "LstmState"]


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

        self.ff_weight = rf.Parameter((4 * self.out_dim, self.in_dim))
        self.ff_weight.initial = rf.init.Glorot()
        self.rec_weight = rf.Parameter((4 * self.out_dim, self.out_dim))
        self.rec_weight.initial = rf.init.Glorot()

        self.bias = None
        if with_bias:
            self.bias = rf.Parameter((4 * self.out_dim,))
            self.bias.initial = 0.0

    def __call__(self, source: Tensor, *, state: LstmState, spatial_dim: Dim) -> Tuple[Tensor, LstmState]:
        """
        Forward call of the LSTM.

        :param source: Tensor of size {...,in_dim} if spatial_dim is single_step_dim else {...,spatial_dim,in_dim}.
        :param state: State of the LSTM. Both h and c are of shape {...,out_dim}.
        :return: output of shape {...,out_dim} if spatial_dim is single_step_dim else {...,spatial_dim,out_dim},
            and new state of the LSTM.
        """
        if not state.h or not state.c:
            raise ValueError(f"{self}: state {state} needs attributes ``h`` (hidden) and ``c`` (cell).")
        if self.in_dim not in source.dims_set:
            raise ValueError(f"{self}: input {source} does not have in_dim {self.in_dim}")

        # noinspection PyProtectedMember
        result, new_state = source._raw_backend.lstm(
            source=source,
            state_c=state.c,
            state_h=state.h,
            ff_weight=self.ff_weight,
            rec_weight=self.rec_weight,
            bias=self.bias,
            spatial_dim=spatial_dim,
            in_dim=self.in_dim,
            out_dim=self.out_dim,
        )
        new_state = LstmState(*new_state)

        return result, new_state


class LstmState(rf.State):
    """LSTM state"""

    def __init__(self, h: Tensor, c: Tensor):
        super().__init__()
        self.h = h
        self.c = c
