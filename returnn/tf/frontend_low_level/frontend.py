"""
Frontend for exposing TensorFlow-specific functionality.
"""

from __future__ import annotations
from typing import Union, Sequence
import tensorflow as tf

from returnn.util.basic import NotSpecified
from returnn.frontend_api import Frontend, RawTensorTypes
from returnn.tensor import Tensor, Dim
from returnn.tf.util import basic as tf_util
from ._internal_frontend import TFInternalFrontend

_TT = Tensor[tf.Tensor]


class TFFrontend(Frontend[tf.Tensor]):
    """
    TensorFlow low-level frontend, operating on tf.Tensor
    """

    RawTensorType = tf.Tensor
    is_tensorflow = True
    _internal_frontend = TFInternalFrontend

    @staticmethod
    def convert_to_tensor(value: Union[_TT, tf.Tensor, RawTensorTypes]) -> _TT:
        """
        :param value:
        :return: tensor
        """
        if isinstance(value, Tensor):
            return value
        value = tf.convert_to_tensor(value)
        assert isinstance(value, tf.Tensor)
        assert value.shape.as_list() == [], f"scalar expected, got {value}"
        return Tensor("const", raw_tensor=value, dims=[], dtype=value.dtype.base_dtype.name)

    @staticmethod
    def range_over_dim(dim: Dim) -> _TT:
        """
        :param dim:
        :return: range over dim
        """
        out = Tensor(name=dim.description or "range_over_dim", dims=[dim], sparse_dim=dim)
        dim_value = dim.get_dim_value()
        out.raw_tensor = tf.range(0, dim_value, dtype=out.dtype)
        return out

    @staticmethod
    def reduce(source: _TT, *, mode: str, axis: Union[Dim, Sequence[Dim]], use_time_mask: bool = NotSpecified) -> _TT:
        """Reduce"""
        x = source
        axes = x.get_axes_from_description(axis)
        if use_time_mask in (None, NotSpecified):
            use_time_mask = any(x.has_dynamic_size(a) for a in axes)
        out_data = x.copy_template()
        dim_tags = [dim_tag for i, dim_tag in enumerate(x.dim_tags) if i not in axes]
        out_data = out_data.copy_template_new_dim_tags(dim_tags)
        sparse_out = mode.lower().startswith("arg")
        if sparse_out:
            assert len(axes) == 1
            out_data.sparse_dim = x.dim_tags[axes[0]]
            out_data.dtype = "int32"
        assert isinstance(use_time_mask, bool)
        mode = mode.lower()
        if mode == "avg":  # alias
            mode = "mean"
        reduce_abs_funcs = {
            name: getattr(tf, "reduce_%s" % name) for name in ["max", "min", "sum", "logsumexp", "any", "all"]
        }
        reduce_rel_func = {"mean": tf.reduce_mean}
        arg_funcs = {name: getattr(tf, name) for name in ["argmax", "argmin"]}
        funcs = dict(list(reduce_abs_funcs.items()) + list(reduce_rel_func.items()) + list(arg_funcs.items()))
        assert mode in funcs, "invalid mode %r. choose from: %r" % (mode, funcs)
        f = funcs[mode]
        x_ = x.placeholder
        # Check if we should ignore some frames, e.g. via masking.
        correction_factor = None
        if use_time_mask and any(x.has_dynamic_size(a) for a in axes):
            if x.batch_dim_axis in axes and x.time_dim_axis in axes and len(axes) == 2:
                assert mode not in arg_funcs, "unexpected arg reduce for multiple axes"
                # Flattening.
                axes = [a if (a < x.time_dim_axis) else (a - 1) for a in axes if a != x.time_dim_axis]
                x = x.copy_time_flattened()
                x_ = x.placeholder

            else:
                # Fhe fastest and simplest way is masking.
                for axis in axes:
                    if axis == x.batch_dim_axis:
                        continue
                    if not x.has_dynamic_size(axis):
                        continue
                    mask = x.get_sequence_mask_broadcast(axis=axis)

                    zeros = tf.zeros((), dtype=x.placeholder.dtype)
                    # Cannot call x.placeholder.dtype.{min,max} in case input is e.g. a bool
                    if x.placeholder.dtype.is_floating or x.placeholder.dtype.is_integer:
                        if f in (tf.reduce_mean, tf.reduce_sum):
                            replacement_value = zeros
                        elif f in (tf.reduce_max, tf.reduce_logsumexp, tf.argmax):
                            replacement_value = zeros + x.placeholder.dtype.min
                        elif f in (tf.reduce_min, tf.argmin):
                            replacement_value = zeros + x.placeholder.dtype.max
                        else:
                            raise ValueError("unexpected reduce function %r" % f)
                    elif x.placeholder.dtype.is_bool:
                        if f in (tf.reduce_any,):
                            replacement_value = zeros
                        elif f in (tf.reduce_all,):
                            replacement_value = tf.ones((), dtype=x.placeholder.dtype)
                        else:
                            raise ValueError("unexpected reduce function %r" % f)
                    else:
                        raise TypeError("reduce: unexpected input type %r from input %s" % (x.placeholder.dtype, x))

                    x_ = tf_util.where_bc(mask, x_, replacement_value, name="x_masked_axis_%i" % axis)
                    if f == tf.reduce_mean:
                        tag = x.dim_tags[axis]
                        assert tag.dyn_size_ext is not None  # checked above
                        size_all = tf.shape(x.placeholder)[axis]
                        size_actual = tag.dyn_size_ext
                        while any(d not in out_data.dim_tags for d in size_actual.dim_tags):
                            # We have some axis (e.g. B) which is not in the output.
                            # We need to remove this.
                            # https://github.com/rwth-i6/returnn/issues/1242
                            i, d = [(i, d) for i, d in enumerate(size_actual.dim_tags) if d not in out_data.dim_tags][0]
                            assert not d.is_dynamic()  # not implemented
                            size_all *= d.get_dim_value()
                            s = tf.reduce_sum(size_actual.placeholder, axis=i)
                            size_actual = size_actual.copy_template_excluding_axis(i)
                            size_actual.placeholder = s
                        seq_len_bc = size_actual.copy_compatible_to(
                            out_data, check_sparse=False, check_dtype=False
                        ).placeholder
                        seq_len_bc = tf.maximum(seq_len_bc, 1)  # avoid nan
                        correction_factor_ = tf.cast(size_all, tf.float32) / tf.cast(seq_len_bc, tf.float32)
                        correction_factor = tf_util.optional_mul(correction_factor, correction_factor_)
        if mode in arg_funcs:
            assert len(axes) == 1, "For argmax/argmin, only one reduction axis is supported"
            y = f(x_, axis=axes[0], output_type=tf.int32)
        else:
            y = f(x_, axis=axes)
            y = tf_util.optional_mul(y, correction_factor)
        out_data.raw_tensor = y
        return out_data
