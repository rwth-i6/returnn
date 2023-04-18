"""
RETURNN frontend (returnn.frontend) tests
"""

from __future__ import annotations
import _setup_test_env  # noqa
from returnn.tensor import Tensor, Dim, TensorDict, batch_dim
from rf_utils import run_model


def test_conformer():
    # This test needs a huge stack size currently, due to the way RETURNN layer construction works currently.
    # On RETURNN side, there is the option flat_net_construction to solve this,
    # however, it's experimental and also does not work for this case.
    # https://github.com/rwth-i6/returnn/issues/957
    # https://stackoverflow.com/a/16248113/133374
    import resource
    import sys

    try:
        resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
    except Exception as exc:
        print(f"resource.setrlimit {type(exc).__name__}: {exc}")
    sys.setrecursionlimit(10**6)

    time_dim = Dim(Tensor("time", [batch_dim], dtype="int32"))
    in_dim = Dim(7, name="in")
    extern_data = TensorDict(
        {
            "data": Tensor("data", [batch_dim, time_dim, in_dim], dtype="float32"),
        }
    )

    from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

    # noinspection PyShadowingNames
    def _forward_step(*, model: ConformerEncoder, extern_data: TensorDict):
        out, out_spatial_dim = model(extern_data["data"], in_spatial_dim=time_dim)
        out.mark_as_default_output(shape=(batch_dim, out_spatial_dim, model.out_dim))

    run_model(
        extern_data,
        lambda *, epoch, step: ConformerEncoder(
            in_dim,
            Dim(14, name="out"),
            ff_dim=Dim(17, name="ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2")],
                filter_sizes=[(3, 3), (3, 3)],
                pool_sizes=[(2, 1), (2, 1)],
            ),
            num_heads=2,
            num_layers=2,
        ),
        _forward_step,
    )
