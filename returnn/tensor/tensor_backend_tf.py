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


tensor_backend_tf = TensorBackendTF()
_dispatch_table[tf.Tensor] = tensor_backend_tf
