"""
Provides the :class:`LSTM` module.
"""

from __future__ import annotations

from typing import Tuple, Sequence

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim


__all__ = ["LSTM", "LstmState", "ZoneoutLSTM"]


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
        if state.h is None or state.c is None:
            raise ValueError(f"{self}: state {state} needs attributes ``h`` (hidden) and ``c`` (cell).")
        if self.in_dim not in source.dims:
            raise ValueError(f"{self}: input {source} does not have in_dim {self.in_dim}")

        # noinspection PyProtectedMember
        result, (new_state_h, new_state_c) = source._raw_backend.lstm(
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
        new_state = LstmState(h=new_state_h, c=new_state_c)
        result.feature_dim = self.out_dim
        new_state.h.feature_dim = self.out_dim
        new_state.c.feature_dim = self.out_dim

        return result, new_state

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> LstmState:
        """initial state"""
        return LstmState(
            h=rf.zeros(list(batch_dims) + [self.out_dim], feature_dim=self.out_dim),
            c=rf.zeros(list(batch_dims) + [self.out_dim], feature_dim=self.out_dim),
        )


class LstmState(rf.State):
    """LSTM state"""

    def __init__(self, *_args, h: Tensor = None, c: Tensor = None):
        super().__init__(*_args)
        if not _args:
            self.h = h
            self.c = c


class ZoneoutLSTM(LSTM):
    """
    Zoneout LSTM module.
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        *,
        with_bias: bool = True,
        zoneout_factor_cell: float = 0.0,
        zoneout_factor_output: float = 0.0,
        use_zoneout_output: bool = True,
        forget_bias: float = 0.0,
        parts_order: str = "ifco",
    ):
        """
        :param in_dim:
        :param out_dim:
        :param with_bias:
        :param zoneout_factor_cell: 0.0 is disabled. reasonable is 0.15.
        :param zoneout_factor_output: 0.0 is disabled. reasonable is 0.05.
        :param use_zoneout_output: True is like the original paper. False is like older RETURNN versions.
        :param forget_bias: 1.0 is default in RETURNN/TF ZoneoutLSTM.
            0.0 is default in :class:`LSTM`, or RETURNN NativeLSTM, PyTorch LSTM, etc.
        :param parts_order:
            i: input gate.
            f: forget gate.
            o: output gate.
            c|g|j: input.
            icfo: like RETURNN/TF ZoneoutLSTM.
            ifco: PyTorch (cuDNN) weights, standard for :class:`LSTM`.
            cifo: RETURNN NativeLstm2 weights.
        """
        super().__init__(
            in_dim,
            out_dim,
            with_bias=with_bias,
        )
        self.zoneout_factor_cell = zoneout_factor_cell
        self.zoneout_factor_output = zoneout_factor_output
        self.use_zoneout_output = use_zoneout_output
        self.forget_bias = forget_bias
        self.parts_order = parts_order.replace("c", "j").replace("g", "j")
        self.dropout_broadcast = rf.dropout_broadcast_default()
        assert len(self.parts_order) == 4 and set(self.parts_order) == set("ijfo")

    def _inner_step(self, x: Tensor, *, state: LstmState) -> Tuple[Tensor, LstmState]:
        prev_c = state.c
        prev_h = state.h

        # Apply vanilla LSTM
        rec = rf.dot(prev_h, self.rec_weight, reduce=self.out_dim)
        x = x + rec
        parts = rf.split(x, axis=4 * self.out_dim, out_dims=[self.out_dim] * 4)
        parts = {k: v for k, v in zip(self.parts_order, parts)}
        i, j, f, o = parts["i"], parts["j"], parts["f"], parts["o"]

        new_c = rf.sigmoid(f + self.forget_bias) * prev_c + rf.sigmoid(i) * rf.tanh(j)
        new_h = rf.sigmoid(o) * rf.tanh(new_c)
        output = new_h

        # Now the ZoneoutLSTM part, which is optional (zoneout_factor_cell > 0 or zoneout_factor_output > 0).
        c = _zoneout(
            prev=prev_c,
            cur=new_c,
            factor=self.zoneout_factor_cell,
            out_dim=self.out_dim,
            dropout_broadcast=self.dropout_broadcast,
        )
        h = _zoneout(
            prev=prev_h,
            cur=new_h,
            factor=self.zoneout_factor_output,
            out_dim=self.out_dim,
            dropout_broadcast=self.dropout_broadcast,
        )
        new_state = LstmState(c=c, h=h)

        if self.use_zoneout_output:  # really the default, sane and original behavior
            output = h

        output.feature_dim = self.out_dim
        new_state.h.feature_dim = self.out_dim
        new_state.c.feature_dim = self.out_dim
        return output, new_state

    def __call__(self, source: Tensor, *, state: LstmState, spatial_dim: Dim) -> Tuple[Tensor, LstmState]:
        """
        Forward call of the LSTM.

        :param source: Tensor of size {...,in_dim} if spatial_dim is single_step_dim else {...,spatial_dim,in_dim}.
        :param state: State of the LSTM. Both h and c are of shape {...,out_dim}.
        :return: output of shape {...,out_dim} if spatial_dim is single_step_dim else {...,spatial_dim,out_dim},
            and new state of the LSTM.
        """
        if state.h is None or state.c is None:
            raise ValueError(f"{self}: state {state} needs attributes ``h`` (hidden) and ``c`` (cell).")
        if self.in_dim not in source.dims:
            raise ValueError(f"{self}: input {source} does not have in_dim {self.in_dim}")

        x = rf.dot(source, self.ff_weight, reduce=self.in_dim)
        if self.bias is not None:
            x = x + self.bias

        if spatial_dim == single_step_dim:
            return self._inner_step(x, state=state)

        batch_dims = source.remaining_dims((spatial_dim, self.in_dim))
        output, new_state, _ = rf.scan(
            spatial_dim=spatial_dim,
            initial=state,
            xs=x,
            ys=Tensor("lstm-out", dims=batch_dims + [self.out_dim], dtype=source.dtype, feature_dim=self.out_dim),
            body=lambda x_, s: self._inner_step(x_, state=s),
        )
        return output, new_state


def _zoneout(*, prev: Tensor, cur: Tensor, factor: float, out_dim: Dim, dropout_broadcast: bool) -> Tensor:
    if factor == 0.0:
        return cur
    return rf.cond(
        rf.get_run_ctx().is_train_flag_enabled(func=ZoneoutLSTM.__call__),
        lambda: (1 - factor) * rf.dropout(cur - prev, factor, axis=dropout_broadcast and out_dim) + prev,
        lambda: (1 - factor) * cur + factor * prev,
    )
