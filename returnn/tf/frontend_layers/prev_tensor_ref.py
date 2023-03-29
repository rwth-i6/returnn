"""
prev tensor ref for loop, i.e. RecLayer, specifically "prev:..." layer references
"""

from __future__ import annotations
from returnn.tensor import Tensor
from .. import frontend_layers as rfl


__all__ = ["PrevTensorRef"]


class PrevTensorRef(Tensor):
    """
    Refers to a layer from the previous loop iteration.
    """

    @classmethod
    def get_prev_ref(cls, *, cur_layer_name_ctx: rfl.Layer, initial: Tensor) -> PrevTensorRef:
        """
        Create prev ref.
        """
        parent_name_ctx = cur_layer_name_ctx.parent
        prev_tensor_name_ctx = parent_name_ctx.get_child(f"prev:{cur_layer_name_ctx.name}")
        if prev_tensor_name_ctx.tensor:
            prev_tensor_ref = prev_tensor_name_ctx.tensor
            assert isinstance(prev_tensor_ref, PrevTensorRef)
            assert prev_tensor_ref.cur_layer_name_ctx is cur_layer_name_ctx
        else:
            prev_tensor_ref = PrevTensorRef(
                name_ctx=prev_tensor_name_ctx, cur_layer_name_ctx=cur_layer_name_ctx, data=initial
            )
            assert prev_tensor_name_ctx.tensor is prev_tensor_ref
        return prev_tensor_ref

    def __init__(self, *, name_ctx: rfl.Layer, cur_layer_name_ctx: rfl.Layer, data: Tensor):
        # At the time we instantiate this, cur_layer_name_ctx.tensor probably does not exist yet.
        super().__init__(**data.get_kwargs())
        self.raw_tensor = name_ctx
        self.cur_layer_name_ctx = cur_layer_name_ctx
        self.raw_tensor.layer_extra_dependencies.append(self.cur_layer_name_ctx)

    def assign_new_cur_tensor_name_ctx(self, cur_tensor_name_ctx: rfl.Layer):
        """
        Changes self.name_ctx to new name_ctx.
        """
        self.raw_tensor.layer_extra_dependencies.remove(self.cur_layer_name_ctx)
        prev_layer_name = f"prev:{cur_tensor_name_ctx.name}"
        assert prev_layer_name not in cur_tensor_name_ctx.parent.children
        prev_layer_name_ctx = cur_tensor_name_ctx.parent.get_child(prev_layer_name)
        prev_layer_name_ctx.move_tensor_here(self)
        assert self.raw_tensor is prev_layer_name_ctx
        self.cur_layer_name_ctx = cur_tensor_name_ctx
        self.raw_tensor.layer_extra_dependencies.append(self.cur_layer_name_ctx)
