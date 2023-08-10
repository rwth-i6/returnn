"""
Operations on strings.
"""


from __future__ import annotations
import numpy


def str_to_numpy_array(s: str) -> numpy.ndarray:
    """
    Canonical way to make a Numpy array for a Python string.

    For :class:`returnn.tensor.Tensor` instances on Numpy arrays,
    the dtype logic assumes this behavior.

    :param s: string
    :return: numpy array, `dtype.kind == "U"`
    """
    return numpy.array(s)
