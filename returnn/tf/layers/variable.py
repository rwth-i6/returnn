"""
VariableLayer and related
"""

from __future__ import annotations
from typing import Optional, Sequence, List, Callable, TypeVar
import contextlib
import tensorflow as tf
import tensorflow.python.ops.resource_variable_ops as tf_resource_variable_ops
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tensor import Tensor, Dim
from .base import LayerBase


T = TypeVar("T")


class VariableLayer(LayerBase):
    """
    Represents a variable. Can add batch/time dimension if wanted. Can be trainable.
    See defaults.
    """

    layer_class = "variable"

    def __init__(
        self,
        shape,
        dtype="float32",
        add_batch_axis=False,
        add_time_axis=False,
        trainable=True,
        non_critical_for_restore=False,
        init=None,
        init_by_layer=None,
        param_name=None,
        **kwargs,
    ):
        """
        :param tuple[int|Dim]|list[int|Dim] shape:
        :param str dtype:
        :param bool add_batch_axis:
        :param bool add_time_axis:
        :param bool trainable:
        :param bool non_critical_for_restore:
        :param str|float|int|None init: see :func:`returnn.tf.util.basic.get_initializer`. 0 by default.
          Alternatively, you can also use option `init_by_layer`.
        :param LayerBase|None init_by_layer:
        :param str|None param_name: self.name (layer name) by default
        """
        shape  # noqa  # used in get_out_data_from_opts
        super(VariableLayer, self).__init__(trainable=trainable, **kwargs)
        assert not self.sources, "%s: does not expect any sources" % self
        self.init_by_layer = init_by_layer
        dim_tags = list(self.output.dim_tags)
        if add_batch_axis:
            assert dim_tags[0].is_batch_dim()
            dim_tags = dim_tags[1:]
        if add_time_axis:
            assert dim_tags[0].dimension == 1
            dim_tags = dim_tags[1:]
        shape_ = [d.dimension for d in dim_tags]
        assert all(shape_), self.output  # all static
        with self.var_creation_scope():
            if init_by_layer is None:
                if init is None:
                    init = 0
                initializer = tf_util.get_initializer(
                    init, dtype=dtype, seed=self.network.random.randint(2**31), eval_local_ns={"layer": self}
                )
            else:
                assert init_by_layer is not None
                out_data_base = Tensor(name=self.output.name, dim_tags=dim_tags, dtype=dtype)
                initializer = init_by_layer.output.copy_compatible_to(out_data_base).placeholder
                shape_ = None  # get_variable requires shape to be not defined when the initializer is another tensor
            self.var = self.add_param(
                tf_compat.v1.get_variable(
                    name=param_name or self.name,
                    shape=shape_,
                    dtype=dtype,
                    initializer=initializer,
                    trainable=trainable,
                ),
                axes_split_info=[d.axis_split_info() for d in dim_tags],
                non_critical_for_restore=non_critical_for_restore,
            )
            self.wrapped_var = None  # type: Optional[_WrappedVariable]
            out = self.var
            if add_time_axis:
                out = tf.expand_dims(out, axis=0)
            if add_batch_axis:
                # Unbroadcast to not confuse some other layers
                batch_dim = self.output.get_batch_dim()
                out = tf_util.expand_dims_unbroadcast(out, axis=0, dim=batch_dim)
            if isinstance(out, tf_resource_variable_ops.ResourceVariable):
                self.wrapped_var = _WrappedVariable(variable_layer=self, variable=out)
                out = self.wrapped_var
                tf_util.set_param_axes_split_info(out, axes_split_info=[d.axis_split_info() for d in dim_tags])
        self.output.placeholder = out

    def get_dep_layers(self):
        """
        :rtype: list[LayerBase]
        """
        deps = super(VariableLayer, self).get_dep_layers()
        if self.init_by_layer:
            deps.append(self.init_by_layer)
        return deps

    @classmethod
    def transform_config_dict(cls, d, network, get_layer):
        """
        :param dict[str] d: will modify inplace
        :param returnn.tf.network.TFNetwork network:
        :param ((str) -> LayerBase) get_layer: function to get or construct another layer
        """
        # Overwrite default behavior for default sources.
        # Here: none by default.
        d.setdefault("from", [])
        super(VariableLayer, cls).transform_config_dict(d, network=network, get_layer=get_layer)
        if d.get("init_by_layer", None):
            d["init_by_layer"] = get_layer(d["init_by_layer"])

    @classmethod
    def get_out_data_from_opts(
        cls, name, network, shape, dtype="float32", add_batch_axis=False, add_time_axis=False, **kwargs
    ):
        """
        :param str name:
        :param returnn.tf.network.TFNetwork network:
        :param tuple[int|Dim]|list[int|Dim] shape:
        :param str dtype:
        :param bool add_batch_axis:
        :param bool add_time_axis:
        :rtype: Tensor
        """
        assert isinstance(shape, (list, tuple))
        assert len(shape) == 0 or all(shape)
        dim_tags = []
        for i, d in enumerate(shape):
            if isinstance(d, Dim):
                assert d.dimension is not None, "%r: need static dims but got %r" % (name, d)
            elif isinstance(d, int):
                d = Dim(
                    kind=Dim.Types.Spatial if i < len(shape) - 1 else Dim.Types.Feature,
                    description="%s:static:%i" % (name, i),
                    auto_generated=True,
                    dimension=d,
                )
            else:
                raise TypeError("Layer %r: invalid type %s in shape %r" % (name, type(d), shape))
            dim_tags.append(d)
        if add_time_axis:
            dim_tags.insert(
                0, Dim(kind=Dim.Types.Time, description="%s:dummy-time" % name, dimension=1, auto_generated=True)
            )
        if add_batch_axis:
            dim_tags.insert(
                0, Dim(kind=Dim.Types.Batch, description="batch", batch=network.get_global_batch_info(), dimension=None)
            )
        return Tensor(
            name="%s_output" % name,
            dim_tags=dim_tags,
            dtype=dtype,
            batch=network.get_global_batch_info() if add_batch_axis else None,
        )


class _WrappedVariable(tf_resource_variable_ops.BaseResourceVariable):
    """
    Similar as TF _WrappedVariable, this wraps an underlying variable.
    We make sure that reads and assignments are in the right order.
    If there is no assignment (via :class:`VariableAssignLayer`),
    no extra logic will be done, and reads will just be direct.

    The code is based on TF _WrappedVariable and adopted.
    """

    def __init__(
        self,
        *,
        variable: tf_resource_variable_ops.BaseResourceVariable,
        variable_layer: VariableLayer,
    ):
        assert isinstance(variable, tf_resource_variable_ops.BaseResourceVariable), f"{variable!r} is {type(variable)}"
        handle: tf.Tensor = variable.handle
        dtype: tf.DType = variable.dtype
        shape: tf.TensorShape = variable.shape
        unique_id = variable._unique_id
        handle_name: str = getattr(handle, "name", "")  # if EagerTensor, raises AttributeError, thus we get ""
        super(_WrappedVariable, self).__init__(
            handle=handle,
            shape=shape,
            handle_name=handle_name,
            unique_id=unique_id,
            dtype=dtype,
            graph_element=None,
        )
        self._variable: tf_resource_variable_ops.BaseResourceVariable = variable
        self._variable_layer: VariableLayer = variable_layer
        self._assign_ops: List[tf.Operation] = []
        self._all_ops: List[tf.Operation] = []

    @property
    def name(self):
        """name"""
        if self._in_graph_mode:
            return self.op.name
        else:
            return "UnreadVariable"

    @property
    def op(self):
        """The op for this variable."""
        if not self._all_ops:
            return self._handle.op
        return self._get_recent_op(self._all_ops)

    @staticmethod
    def _get_recent_op(coll: List[tf.Operation]) -> Optional[tf.Operation]:
        if not coll:
            return None
        recent_op = None
        for op in reversed(coll):
            recent_op = tf_util.op_in_right_control_flow_context(op)
            if recent_op is not None:
                break
        if recent_op is not coll[-1]:
            coll.append(recent_op)
        return recent_op

    def _wrap_func(self, func: Callable[[], T], *, is_assign_op: bool) -> T:
        recent_op = self._get_recent_op(self._all_ops if is_assign_op else self._assign_ops)
        with tf.control_dependencies([recent_op]) if recent_op else contextlib.nullcontext():
            res = func()
        if isinstance(res, tf.Operation):
            op = res
        else:
            assert isinstance(res.op, tf.Operation)
            op = res.op
        if is_assign_op:
            self._assign_ops.append(op)
        self._all_ops.append(op)
        return res

    def value(self):
        """value"""
        return self._read_variable_op()

    def read_value(self):
        """read value"""
        return self._read_variable_op()

    @property
    def _graph_element(self):
        return self._read_variable_op()

    @_graph_element.setter
    def _graph_element(self, value):
        pass  # ignore

    def _read_variable_op(self, *args, **kwargs):
        assert not args and not kwargs
        return self._wrap_func(
            lambda: tf_resource_variable_ops.read_variable_op(self._handle, self._dtype), is_assign_op=False
        )

    def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
        """assign sub"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).assign_sub(delta, use_locking, name, read_value), is_assign_op=True
        )

    def assign_add(self, delta, use_locking=None, name=None, read_value=True):
        """assign add"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).assign_add(delta, use_locking, name, read_value), is_assign_op=True
        )

    def assign(self, value, use_locking=None, name=None, read_value=True):
        """assign"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).assign(value, use_locking, name, read_value), is_assign_op=True
        )

    def scatter_sub(self, sparse_delta, use_locking=False, name=None):
        """scatter sub"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_sub(sparse_delta, use_locking, name), is_assign_op=True
        )

    def scatter_add(self, sparse_delta, use_locking=False, name=None):
        """scatter add"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_add(sparse_delta, use_locking, name), is_assign_op=True
        )

    def scatter_max(self, sparse_delta, use_locking=False, name=None):
        """scatter max"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_max(sparse_delta, use_locking, name), is_assign_op=True
        )

    def scatter_min(self, sparse_delta, use_locking=False, name=None):
        """scatter min"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_min(sparse_delta, use_locking, name), is_assign_op=True
        )

    def scatter_mul(self, sparse_delta, use_locking=False, name=None):
        """scatter mul"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_mul(sparse_delta, use_locking, name), is_assign_op=True
        )

    def scatter_div(self, sparse_delta, use_locking=False, name=None):
        """scatter div"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_div(sparse_delta, use_locking, name), is_assign_op=True
        )

    def scatter_update(self, sparse_delta, use_locking=False, name=None):
        """scatter update"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_update(sparse_delta, use_locking, name), is_assign_op=True
        )

    def batch_scatter_update(self, sparse_delta, use_locking=False, name=None):
        """batch scatter update"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).batch_scatter_update(sparse_delta, use_locking, name),
            is_assign_op=True,
        )

    def scatter_nd_sub(self, indices, updates, name=None):
        """scatter nd sub"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_nd_sub(indices, updates, name), is_assign_op=True
        )

    def scatter_nd_add(self, indices, updates, name=None):
        """scatter nd add"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_nd_add(indices, updates, name), is_assign_op=True
        )

    def scatter_nd_update(self, indices, updates, name=None):
        """scatter nd update"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_nd_update(indices, updates, name), is_assign_op=True
        )

    def scatter_nd_max(self, indices, updates, name=None):
        """scatter nd max"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_nd_max(indices, updates, name), is_assign_op=True
        )

    def scatter_nd_min(self, indices, updates, name=None):
        """scatter nd min"""
        return self._wrap_func(
            lambda: super(_WrappedVariable, self).scatter_nd_min(indices, updates, name), is_assign_op=True
        )


class VariableAssignLayer(LayerBase):
    """
    Assigns a new value to a variable.
    """

    layer_class = "variable_assign"

    def __init__(
        self,
        var: LayerBase,
        value: LayerBase,
        control_dependencies: Optional[Sequence[LayerBase]] = None,
        op: str = "assign",
        **kwargs,
    ):
        """
        :param var:
        :param value:
        :param control_dependencies:
        :param op: "assign" or "add"
        """
        super().__init__(**kwargs)
        self.var = var
        self.value = value
        self.control_dependencies = list(control_dependencies) if control_dependencies else []
        deps = [src.output.placeholder.op for src in self.control_dependencies]

        while not isinstance(var, VariableLayer):
            if isinstance(var, VariableAssignLayer):
                deps.append(var.output.placeholder.op)
                var = var.var
            elif isinstance(var, VariableReadLayer):
                deps.append(var.output.placeholder.op)
                var = var.var
            else:
                raise TypeError(f"{self}: invalid var {var!r}")
        assert isinstance(var, VariableLayer), f"{self}: var must be a VariableLayer, got {var}"
        assert var.wrapped_var is not None, f"{self}: var must be wrapped, got {var}"
        self.tf_var: _WrappedVariable = var.wrapped_var

        value_data = value.output.copy_compatible_to(self.var.output)
        with tf.control_dependencies(deps) if deps else contextlib.nullcontext():
            if op == "assign":
                op_ = self.tf_var.assign(value_data.placeholder, read_value=False)
            elif op == "add":
                op_ = self.tf_var.assign_add(value_data.placeholder, read_value=False)
            else:
                raise ValueError(f"{self}: invalid op {op!r}")
        # op_ is only defined in graph-mode. in eager-mode, it's not relevant.
        with tf.control_dependencies([op_]) if op_ is not None else contextlib.nullcontext():
            self.output.placeholder = tf.zeros((), dtype="int32")

    def get_dep_layers(self) -> List[LayerBase]:
        """deps"""
        return super().get_dep_layers() + [self.var, self.value] + self.control_dependencies

    @classmethod
    def transform_config_dict(cls, d, network, get_layer):
        """transform"""
        d.setdefault("from", [])
        super().transform_config_dict(d, network=network, get_layer=get_layer)
        d["var"] = get_layer(d["var"])
        d["value"] = get_layer(d["value"])
        if d.get("control_dependencies"):
            d["control_dependencies"] = [get_layer(layer) for layer in d["control_dependencies"]]

    @classmethod
    def get_out_data_from_opts(cls, name: str, var: LayerBase, **kwargs):
        """out"""
        return Tensor(name, dims=(), dtype="int32")  # dummy, will be just the op


class VariableReadLayer(LayerBase):
    """
    Read a variable (currently expected from VariableLayer).
    Supports control dependencies to exactly specify when it should be read.
    """

    layer_class = "variable_read"

    def __init__(self, var: LayerBase, control_dependencies: Optional[Sequence[LayerBase]] = None, **kwargs):
        """
        :param var: e.g. VariableLayer
        :param control_dependencies: to control what ops must run before the var is read (e.g. assign ops)
        """
        super().__init__(**kwargs)
        self.var = var
        self.control_dependencies = list(control_dependencies) if control_dependencies else []
        deps = [src.output.placeholder.op for src in self.control_dependencies]

        while not isinstance(var, VariableLayer):
            if isinstance(var, VariableAssignLayer):
                deps.append(var.output.placeholder.op)
                var = var.var
            elif isinstance(var, VariableReadLayer):
                deps.append(var.output.placeholder.op)
                var = var.var
            else:
                raise TypeError(f"{self}: invalid var {var!r}")
        assert isinstance(var, VariableLayer), f"{self}: var must be a VariableLayer, got {var}"
        self.tf_var = var.var
        assert isinstance(self.tf_var, tf.Variable), f"{self}: var must be a tf.Variable, got {self.tf_var}"
        with tf.control_dependencies(deps) if deps else contextlib.nullcontext():
            self.output.placeholder = self.tf_var.read_value()

    def get_dep_layers(self) -> List[LayerBase]:
        """deps"""
        return super().get_dep_layers() + [self.var] + self.control_dependencies

    @classmethod
    def transform_config_dict(cls, d, network, get_layer):
        """transform"""
        d.setdefault("from", [])
        super().transform_config_dict(d, network=network, get_layer=get_layer)
        d["var"] = get_layer(d["var"])
        if d.get("control_dependencies"):
            d["control_dependencies"] = [get_layer(layer) for layer in d["control_dependencies"]]

    @classmethod
    def get_out_data_from_opts(cls, name: str, var: LayerBase, **kwargs):
        """out"""
        return var.output.copy_template(name="%s_output" % name)
