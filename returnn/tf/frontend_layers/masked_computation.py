"""
Masked computation. Wrap :class:`MaskedComputationLayer` in RETURNN.

https://github.com/rwth-i6/returnn_common/issues/23

"""

from __future__ import annotations

from returnn.tensor import Tensor
import returnn.frontend as rf
import returnn.tf.frontend_layers as rfl
from . import _utils


__all__ = ["MaskedComputation", "MaskedComputationModule"]


class MaskedComputation:
    """
    This is expected to be inside a :class:`Loop`.

    Usage example::

        loop = nn.Loop(...)
        loop.state.y = ...  # some initial output
        loop.state.h = ...  # some initial state
        with loop:

          mask = ...  # dtype bool, shape [batch] or whatever, for current (fast) frame
          with nn.MaskedComputation(mask=mask):
            loop.state.y, loop.state.h = slow_rnn(x, loop.state.h)
          y = loop.state.y  # access from outside

    This is equivalent to::

        loop = nn.Loop(...)
        loop.state.y = ...  # some initial output
        loop.state.h = ...  # some initial state
        with loop:

          mask = ...  # dtype bool, shape [batch] or whatever, for current frame
          y_, h_ = slow_rnn(x, loop.state.h)
          loop.state.y = nest.map(lambda a, b: nn.where(cond=mask, x=a, y=b), y_, loop.state.y)
          loop.state.h = nest.map(lambda a, b: nn.where(cond=mask, x=a, y=b), h_, loop.state.h)
          y = loop.state.y

    In pseudocode, non-batched (mask is just a scalar bool), it would look like::

        y = ...  # some initial output
        h = ...  # some initial state
        while True:

          mask = ...  # bool
          if mask:
            y, h = slow_rnn(x, h)

    """

    def __init__(self, mask: Tensor, *, name: str = "masked_computation"):
        """
        :param Tensor mask: bool, shape [batch]
        """
        self.mask = mask
        self.name = name
        self.layer_module = MaskedComputationModule(masked_computation=self)
        self.name_ctx = rfl.Layer(module=self.layer_module, suggested_name=name, parent=rfl.Layer.current_ctx())
        self.name_ctx.custom_layer_name_scope = ""
        self.name_ctx.is_subnet = True

    def __enter__(self) -> MaskedComputation:
        self.name_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if not exc_type:
                # Make sure there is an "output" layer. (Similar as for Module with subnetwork.)
                if "output" not in self.name_ctx.children:
                    last_child = self.name_ctx.get_recent_tensor(only_same_control_flow=True)
                    if last_child is not None:
                        _utils.copy(last_child, name=self.name_ctx.get_child("output"))
                    else:
                        _utils.constant(value=0, name=self.name_ctx.get_child("output"))  # unused
        finally:
            self.name_ctx.__exit__(exc_type, exc_val, exc_tb)
        if not exc_type:
            self.layer_module()  # create the rec layer itself


class MaskedComputationModule(rf.Module):
    """
    This is for internal use by :class:`MaskedComputation`.
    """

    def __init__(self, masked_computation: MaskedComputation):
        super().__init__()
        self.masked_computation = masked_computation

    def __call__(self) -> Tensor:
        """
        Makes layer dict for this loop, i.e. a RecLayer.
        """
        name_ctx = self.masked_computation.name_ctx
        out = name_ctx.children["output"].tensor
        loop = rfl.Layer.inner_loop()
        assert loop, f"{self}: need to be inside loop"  # not implemented otherwise
        return rfl.make_layer(
            {
                "class": "masked_computation",
                "mask": self.masked_computation.mask,
                "in_spatial_dim": loop.loop_spatial_dim,
                "unit": {"class": "subnetwork", "from": [], "subnetwork": name_ctx.make_net()},
            },
            name=name_ctx,
            predefined_out_data=out,
        )
