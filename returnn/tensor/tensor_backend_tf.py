"""
TensorFlow backend for :class:`Tensor`.
"""

from __future__ import annotations
import tensorflow as tf
from returnn.tf import compat as tf_compat
from .tensor_backend import TensorBackend, _dispatch_table
from . import tensor as _t


class TensorBackendTF(TensorBackend[tf.Tensor]):
    """
    TensorFlow backend for :class:`Tensor`.
    """

    def create_placeholder(self, tensor: _t.Tensor) -> tf.Tensor:
        """
        :return: tf.placeholder in TF
        """
        with tf.name_scope("extern_data/placeholders/%s/" % tensor.name):
            return tf_compat.v1.placeholder(**tensor.get_placeholder_kwargs(with_batch=True))

    def runtime_sanity_checks(self, tensor: _t.Tensor) -> tf.Operation:
        """
        Runtime checks
        """
        checks = []
        with tf.name_scope("runtime_sanity_check"):
            shape = tf.shape(tensor.placeholder)
            # noinspection PyShadowingNames
            batch_dim = shape[tensor.batch_dim_axis] if tensor.have_batch_axis() else 1
            rank = tf.rank(tensor.placeholder)
            data = ["Data.get_runtime_sanity_check_op:", str(tensor), "shape", shape]
            for i, tag in enumerate(tensor.dim_tags):
                if tag.dyn_size is not None:
                    data += ["dyn_size[%i] (%s)" % (i, tag), tag.dyn_size, ".shape", tf.shape(tag.dyn_size)]
            checks += [tf.Assert(tf.equal(rank, tensor.batch_ndim), data + ["-> invalid rank"])]
            if tensor.have_batch_axis():
                batch_dim_via_info = tensor.get_batch_dim()
                checks += [
                    tf.Assert(
                        tf.equal(batch_dim, batch_dim_via_info),
                        data + ["-> invalid batch dim info", batch_dim_via_info],
                    )
                ]
            for i in range(tensor.batch_ndim):
                if tensor.batch_shape[i] is not None:
                    checks += [
                        tf.Assert(tf.equal(shape[i], tensor.batch_shape[i]), data + ["-> invalid shape[%i]" % i])
                    ]
                dyn_size_ext = tensor.dim_tags[i].dyn_size_ext
                if dyn_size_ext and dyn_size_ext.placeholder is not None:
                    dyn_size = dyn_size_ext.placeholder
                    if dyn_size_ext.have_batch_axis() and tensor.have_batch_axis():
                        checks += [
                            tf.Assert(
                                tf.equal(tf.shape(dyn_size)[dyn_size_ext.batch_dim_axis], batch_dim),
                                data + ["-> invalid axis %i tag dyn size batch dim" % i],
                            )
                        ]
                    checks += [
                        tf.Assert(
                            # Note: in almost all cases, we have equality here.
                            # However, not strictly in all cases, e.g. DecideLayer, maybe some others...
                            # But that should not be more than 1 less.
                            tf.logical_or(
                                tf.logical_and(
                                    tf.less_equal(tf.reduce_max(dyn_size), shape[i]),
                                    tf.greater_equal(tf.reduce_max(dyn_size), shape[i] - 1),
                                ),
                                # In other rare cases, this might be a broadcast dim
                                # (e.g. as initial values of att weights for a rec loop).
                                tf.equal(1, shape[i]),
                            ),
                            data + ["-> invalid shape[%i] or max(dyn_size[%i])" % (i, i)],
                        )
                    ]
                    checks += [dyn_size_ext.get_runtime_sanity_check_op()]
        return tf.group(*checks)


tensor_backend_tf = TensorBackendTF()
_dispatch_table[tf.Tensor] = tensor_backend_tf
