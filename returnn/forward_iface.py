"""
Defines the interface for the "forward" task,
which can be used for recognition, alignment, search, etc.

https://github.com/rwth-i6/returnn/issues/1336
"""

from __future__ import annotations
from returnn.tensor import TensorDict


class ForwardCallbackIface:
    """
    Callback interface for the forward task.

    Define `forward_callback` in your config to an instance or class of this.

    https://github.com/rwth-i6/returnn/issues/1336
    """

    def init(self, *, model):
        """
        Run at the beginning.
        """

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        """
        Called for each sequence, or entry in the dataset.
        This does not have the batch dim anymore.
        The values in `outputs` are Numpy arrays.

        :param seq_tag:
        :param outputs:
        """

    def finish(self):
        """
        Run at the end.
        """
