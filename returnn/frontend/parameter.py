"""
Parameter / variable
"""

from __future__ import annotations
from typing import Optional, Union, TypeVar, Sequence
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import global_backend as _global_backend


__all__ = ["Parameter"]


T = TypeVar("T")


class Parameter(Tensor[T]):
    """
    This represents a (potential trainable) parameter,
    aka ``tf.Variable`` in TensorFlow,
    wrapping to ``VariableLayer`` in RETURNN.
    """

    def __init__(
        self,
        dims_or_tensor: Union[None, Sequence[Dim], Tensor] = None,
        dtype: Optional[str] = None,
        *,
        dims: Optional[Sequence[Dim]] = None,
        sparse_dim: Optional[Dim] = None,
        trainable: Optional[bool] = None,
        auxiliary: bool = False,
        non_critical_for_restore: bool = False,
        weight_decay: Optional[float] = 0.0,
        initial: Optional[rf.init.ParamInitType] = None,
        raw_tensor: Optional[T] = None,
        device: Optional[str] = None,
    ):
        """
        :param dims_or_tensor:
        :param dims:
        :param dtype:
        :param sparse_dim:
        :param trainable: if True, and optimizer would do updates to this parameter in training mode
        :param auxiliary: if True, this indicates that this parameter should not be transformed by transformations
          such as weight normalization. One example are running statistics, as used for batch normalization.
          This usually implies that the parameter is not trainable, i.e. not to be updated by the optimizer,
          but usually has some custom update.
          This flag is not passed on to RETURNN but just used here for returnn-common logic.
        :param non_critical_for_restore: if True, this parameter is not critical for restoring a model.
        :param weight_decay:
        :param initial:
        :param raw_tensor:
        :param device:
        """
        if dims_or_tensor is None:
            if dims is None:
                raise ValueError("rf.Parameter: dims must be set.")
        elif isinstance(dims_or_tensor, (tuple, list)):
            if dims is not None:
                raise ValueError("rf.Parameter: dims is set twice.")
            dims = dims_or_tensor
        elif isinstance(dims_or_tensor, Tensor):
            if initial is not None:
                raise ValueError("rf.Parameter: initial is set twice.")
            initial = dims_or_tensor
            if dims is None:
                dims = initial.dims
            if dtype is None:
                dtype = initial.dtype
            if sparse_dim is None:
                sparse_dim = initial.sparse_dim
        else:
            raise TypeError(f"rf.Parameter: invalid type for dims_or_tensor: {type(dims_or_tensor)}")
        if not all(isinstance(dim, Dim) for dim in dims):
            raise TypeError(f"rf.Parameter: shape {dims} must be a sequence of Dim")
        if len(dims) != len(set((d, d.match_priority) for d in dims)):
            raise ValueError(f"rf.Parameter: shape {dims} dims must be unique")
        super(Parameter, self).__init__(
            "parameter",
            dims=dims,
            dtype=dtype or (rf.get_default_float_dtype() if not sparse_dim else rf.get_default_array_index_dtype()),
            sparse_dim=sparse_dim,
        )
        if raw_tensor is not None:
            self.raw_tensor = raw_tensor
        else:
            self.raw_tensor = _global_backend.create_parameter_raw(self, device=device)
        self._trainable = None  # type: Optional[bool]
        self._auxiliary = auxiliary
        self._non_critical_for_restore = non_critical_for_restore
        self._weight_decay = weight_decay
        self._initial = None  # type: Optional[rf.init.ParamInitType]

        self.trainable = trainable  # use setter
        self.initial = initial  # use setter

    def __copy__(self):
        # Should return new copy. https://github.com/rwth-i6/returnn_common/pull/215#issuecomment-1269651064
        # Note that the values are *not* copied, but rather it will use the same param init scheme.
        res = type(self)(
            dims=self.dims,
            dtype=self.dtype,
            trainable=self.trainable,
            auxiliary=self.auxiliary,
            non_critical_for_restore=self.non_critical_for_restore,
            weight_decay=self.weight_decay,
        )
        res.initial = self.initial
        return res

    def __deepcopy__(self, memo=None):
        # Should return new copy. https://github.com/rwth-i6/returnn_common/pull/215#issuecomment-1269651064
        # Note that the values are *not* copied, but rather it will use the same param init scheme.
        from copy import deepcopy

        res = self.__copy__()
        if isinstance(self.initial, rf.init.ParamInit):
            res.initial = deepcopy(self.initial, memo=memo)  # noqa
        else:
            res.initial = self.initial
        return res

    @property
    def initial(self) -> Optional[rf.init.ParamInitType]:
        """initial value of the parameter"""
        return self._initial

    @initial.setter
    def initial(self, value: Optional[rf.init.ParamInitType]):
        # Keep the original ParamInit, so that copies of the Parameter would have a different initial random value.
        # https://github.com/rwth-i6/returnn_common/issues/216
        self._initial = value

        if isinstance(value, rf.init.ParamInit):
            value = value(dims=self.dims, dtype=self.dtype, device=self.device)
        self._raw_backend.set_parameter_initial_value(self, value)

    def assign(self, value: Union[Tensor, rf.RawTensorTypes]):
        """
        Assign new value to this parameter.
        This will also update the allocated raw tensor inplace.

        For graph-based backends, handling the control flow is up to the backend,
        e.g.~making sure it is being executed in the right order,
        in the right control flow context, and at all.
        There is no op or anything like that returned here which the user needs to take care of.
        So the user can think of it just as imperative eager-style code.
        """
        self._raw_backend.parameter_assign(
            self, rf.convert_to_tensor(value, _backend=self._raw_backend, device=self.device), op="assign"
        )

    def assign_add(self, value: Union[Tensor, rf.RawTensorTypes]):
        """
        Add value to this parameter.
        This will also update the raw tensor.
        See :func:`assign`.
        """
        self._raw_backend.parameter_assign(
            self, rf.convert_to_tensor(value, _backend=self._raw_backend, device=self.device), op="add"
        )

    def assign_key(
        self,
        axis: Union[Dim, Sequence[Dim]],
        key: rf.ItemKeyType,
        key_dim: Optional[Union[Dim, Sequence[Optional[Dim]]]],
        value: Union[Tensor, rf.RawTensorTypes],
    ):
        """
        Basically var[key] = value, if axis is the first axis, or otherwise accordingly.
        Note that the __setitem__ API is not supported because it depends on the order of axes,
        but this here is the equivalent function.
        See :func:`assign`.
        """
        self._raw_backend.parameter_assign_key(
            self,
            axis=axis,
            key=key,
            key_dim=key_dim,
            value=rf.convert_to_tensor(value, _backend=self._raw_backend, device=self.device),
            op="assign",
        )

    def to(self, *, device: Optional[str] = None, dtype: Optional[str] = None):
        """
        Move the parameter to the specified device, and/or the specified dtype.

        Note: This is an inplace operation.
        raw_tensor might point to a new raw tensor (or parameter) afterward.
        """
        self._raw_backend.parameter_move_to(self, device=device, dtype=dtype)

    @property
    def weight_decay(self) -> float:
        """
        Weight decay, which is equivalent to L2 loss on the parameters for SGD.
        On RETURNN side, whether this is handled separately or is part of the main loss,
        can be controlled via the ``decouple_constraints`` config option.
        https://github.com/rwth-i6/returnn_common/issues/59#issuecomment-1073913421
        """
        return self._weight_decay or 0.0

    @weight_decay.setter
    def weight_decay(self, value: Optional[float]):
        self._weight_decay = value

    @property
    def trainable(self) -> Optional[bool]:
        """trainable"""
        return self._trainable

    @trainable.setter
    def trainable(self, trainable: Optional[bool]):
        self._trainable = trainable
        if trainable is None:
            if self.auxiliary:
                trainable = False
            elif self.dtype.startswith("int"):
                trainable = False
            else:
                trainable = True
        self._raw_backend.set_parameter_trainable(self, trainable)

    @property
    def auxiliary(self) -> bool:
        """auxiliary"""
        return self._auxiliary

    @auxiliary.setter
    def auxiliary(self, value: bool):
        self._auxiliary = value

    @property
    def non_critical_for_restore(self) -> bool:
        """non_critical_for_restore"""
        return self._non_critical_for_restore

    @non_critical_for_restore.setter
    def non_critical_for_restore(self, value: bool):
        self._non_critical_for_restore = value
