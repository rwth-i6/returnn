"""
RETURNN frontend (returnn.frontend) tests
"""


import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


def _demo_run(model: rf.Module):
    model  # noqa  # TODO
    raise NotImplementedError  # TODO


def test_simple_net_linear():
    class _Net(rf.Module):
        def __init__(self, in_dim: Dim, out_dim: Dim):
            super().__init__()
            self.linear = rf.Linear(in_dim, out_dim)

        def __call__(self, x: Tensor) -> Tensor:
            """
            Forward
            """
            return self.linear(x)

    _demo_run(_Net(Dim(7, name="in"), Dim(13, name="out")))
