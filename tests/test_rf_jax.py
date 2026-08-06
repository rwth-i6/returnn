"""
Tests for the RF JAX backend (:class:`returnn.jax.frontend.JaxBackend`).

Reference values come from the Torch backend on the same inputs,
in the same process, so the comparison covers the RF semantics and not just JAX itself.
"""

from __future__ import annotations
import os

# Keep the whole suite on CPU, and off the GPU memory of whatever else runs on the node.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy
import numpy.testing
import pytest

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402  # after the importorskip


def _rf_jax():
    rf.select_backend("jax")


def _make(name: str, arr: numpy.ndarray, dims) -> Tensor:
    return Tensor(name, dims=dims, dtype=arr.dtype.name, raw_tensor=jnp.asarray(arr))


def test_select_backend_and_dispatch():
    _rf_jax()
    assert rf.get_selected_backend() == "jax"

    # noinspection PyProtectedMember
    from returnn.frontend._backend import get_backend_by_raw_tensor_type
    from returnn.jax.frontend import JaxBackend

    x = jnp.ones((2, 3), dtype=jnp.float32)
    assert get_backend_by_raw_tensor_type(type(x)) is JaxBackend

    # tracers must dispatch too: they have jax.Array as a virtual base only
    seen = []

    def _f(y):
        seen.append(get_backend_by_raw_tensor_type(type(y)))
        return y

    jax.jit(_f)(x)
    assert seen == [JaxBackend]


def test_dtypes_and_x64():
    _rf_jax()
    # x64 must be on, else int64/float64 silently become 32 bit and disagree with Tensor.dtype
    for dtype in ["float32", "float64", "int32", "int64", "bool", "bfloat16"]:
        raw = jnp.zeros((2,), dtype=jnp.dtype(dtype))
        x = Tensor("x", dims=[Dim(2, name="d")], dtype=dtype, raw_tensor=raw)
        assert x.raw_tensor.dtype.name == dtype
        assert x.dtype == dtype


def test_basic_ops_vs_torch():
    batch, time, feat = Dim(3, name="batch"), Dim(5, name="time"), Dim(4, name="feat")
    dims = [batch, time, feat]
    rnd = numpy.random.RandomState(42)
    a_np = rnd.normal(size=(3, 5, 4)).astype("float32")
    b_np = rnd.normal(size=(3, 5, 4)).astype("float32")

    def _run(make) -> dict:
        a, b = make("a", a_np, dims), make("b", b_np, dims)
        out = {
            "add": a + b,
            "sub": a - b,
            "mul": a * b,
            "div": a / b,
            "pow": rf.abs(a) ** 2.0,
            "maximum": rf.maximum(a, b),
            "neg": -a,
            "tanh": rf.tanh(a),
            "relu": rf.relu(a),
            "gelu": rf.gelu(a),
            "sigmoid": rf.sigmoid(a),
            "exp": rf.exp(a),
            "square": rf.square(a),
            "where": rf.where(a > b, a, b),
            "reduce_sum": rf.reduce_sum(a, axis=feat, use_mask=False),
            "reduce_sum_multi": rf.reduce_sum(a, axis=[time, feat], use_mask=False),
            "reduce_max": rf.reduce_max(a, axis=feat, use_mask=False),
            "reduce_mean": rf.reduce_mean(a, axis=feat, use_mask=False),
            "logsumexp": rf.reduce_logsumexp(a, axis=feat, use_mask=False),
            "argmax": rf.reduce_argmax(a, axis=feat, use_mask=False),
            "range": rf.range_over_dim(time),
            "cast": rf.cast(a, "float64"),
            "compare": a > b,
        }
        return {k: v.raw_tensor for k, v in out.items()}

    rf.select_backend_torch()
    import torch

    ref = _run(lambda name, arr, d: Tensor(name, dims=d, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr)))
    ref = {k: v.detach().cpu().numpy() for k, v in ref.items()}

    _rf_jax()
    got = {k: numpy.asarray(v) for k, v in _run(_make).items()}

    assert set(got) == set(ref)
    for key in sorted(ref):
        numpy.testing.assert_allclose(got[key], ref[key], rtol=1e-6, atol=1e-6, err_msg=f"op {key} differs")


def test_transpose_and_expand():
    _rf_jax()
    batch, time, feat = Dim(3, name="batch"), Dim(5, name="time"), Dim(4, name="feat")
    a_np = numpy.random.RandomState(7).normal(size=(3, 5, 4)).astype("float32")
    a = _make("a", a_np, [batch, time, feat])

    raw = a.copy_compatible_to_dims_raw([feat, batch, time])
    numpy.testing.assert_allclose(numpy.asarray(raw), a_np.transpose(2, 0, 1))

    # broadcast a dim in: [batch,time,feat] -> [batch,time,feat] compatible with an extra dim
    other = Dim(2, name="other")
    bc = a.copy_compatible_to_dims_raw([batch, time, feat, other])
    assert tuple(bc.shape) == (3, 5, 4, 1)


def _copy_params_from(src_mod, dst_mod):
    """
    Assign the parameters of one module to another, going through numpy.

    Cross-backend parameter transfer, i.e. the mechanism the PT-checkpoint parity check will need,
    and the reason the tests do not try to match RNG streams between the backends.
    """
    src = dict(src_mod.named_parameters())
    dst = dict(dst_mod.named_parameters())
    assert set(src) == set(dst), f"different parameter sets: {sorted(src)} vs {sorted(dst)}"
    for name, p_dst in dst.items():
        p_src = src[name]
        assert p_src.dims == p_dst.dims, f"{name}: {p_src.dims} vs {p_dst.dims}"
        value = p_src.copy_compatible_to_dims_raw(p_dst.dims).detach().cpu().numpy()
        p_dst.assign(Tensor(name, dims=p_dst.dims, dtype=p_dst.dtype, raw_tensor=jnp.asarray(value)))


def test_linear_and_norms_vs_torch():
    import torch

    batch, time, in_dim, out_dim = Dim(3, name="batch"), Dim(5, name="time"), Dim(4, name="in"), Dim(6, name="out")
    x_np = numpy.random.RandomState(11).normal(size=(3, 5, 4)).astype("float32")

    def _build():
        rf.set_random_seed(31)
        return rf.Linear(in_dim, out_dim), rf.LayerNorm(in_dim), rf.RMSNorm(in_dim)

    def _fwd(mods, x):
        linear, ln, rms = mods
        return {
            "linear": linear(x),
            "layer_norm": ln(x),
            "rms_norm": rms(x),
            "softmax": rf.softmax(linear(x), axis=out_dim),
            "log_softmax": rf.log_softmax(linear(x), axis=out_dim),
        }

    rf.select_backend_torch()
    mods_pt = _build()
    x_pt = Tensor("x", dims=[batch, time, in_dim], dtype="float32", raw_tensor=torch.from_numpy(x_np))
    ref = {k: v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy() for k, v in _fwd(mods_pt, x_pt).items()}

    _rf_jax()
    mods_jax = _build()
    for mod_pt, mod_jax in zip(mods_pt, mods_jax):
        _copy_params_from(mod_pt, mod_jax)
    x_jax = _make("x", x_np, [batch, time, in_dim])
    got = {k: numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)) for k, v in _fwd(mods_jax, x_jax).items()}

    assert set(got) == set(ref)
    for key in sorted(ref):
        numpy.testing.assert_allclose(got[key], ref[key], rtol=1e-5, atol=1e-6, err_msg=f"{key} differs")


def test_random_and_param_init():
    _rf_jax()
    d = Dim(1000, name="d")
    for distribution, kwargs, (lo, hi) in [
        ("uniform", {"minval": -1.0, "maxval": 1.0}, (-1.0, 1.0)),
        ("normal", {"mean": 0.0, "stddev": 1.0}, (-6.0, 6.0)),
        ("truncated_normal", {"mean": 0.0, "stddev": 1.0}, (-2.0, 2.0)),
    ]:
        x = rf.random(dims=[d], dtype="float32", distribution=distribution, **kwargs)
        raw = numpy.asarray(x.raw_tensor)
        assert raw.shape == (1000,) and raw.dtype == numpy.float32
        assert lo <= raw.min() and raw.max() <= hi, f"{distribution}: [{raw.min()}, {raw.max()}] outside [{lo}, {hi}]"
        assert raw.std() > 0.1, f"{distribution}: degenerate"

    # same seed -> same draw; the global stream advances between draws
    a = rf.random(dims=[d], dtype="float32", distribution="normal", static=True, seed=7)
    b = rf.random(dims=[d], dtype="float32", distribution="normal", static=True, seed=7)
    numpy.testing.assert_array_equal(numpy.asarray(a.raw_tensor), numpy.asarray(b.raw_tensor))
    rf.set_random_seed(3)
    c = rf.random(dims=[d], dtype="float32", distribution="normal")
    e = rf.random(dims=[d], dtype="float32", distribution="normal")
    assert not numpy.allclose(numpy.asarray(c.raw_tensor), numpy.asarray(e.raw_tensor))

    # a parameter materializes its initial value through the backend
    param = rf.Parameter([d], dtype="float32")
    param.initial = 1.5
    numpy.testing.assert_allclose(numpy.asarray(param.raw_tensor), numpy.full((1000,), 1.5, dtype="float32"))
    param.assign(rf.convert_to_tensor(2.5, dtype="float32"))
    numpy.testing.assert_allclose(numpy.asarray(param.raw_tensor), numpy.full((1000,), 2.5, dtype="float32"))


def test_matmul_vs_torch():
    import torch

    batch, time, a_dim, b_dim = Dim(3, name="batch"), Dim(5, name="time"), Dim(4, name="a"), Dim(6, name="b")
    rnd = numpy.random.RandomState(13)
    x_np = rnd.normal(size=(3, 5, 4)).astype("float32")
    w_np = rnd.normal(size=(4, 6)).astype("float32")
    y_np = rnd.normal(size=(3, 5, 6)).astype("float32")

    def _run(make):
        x = make("x", x_np, [batch, time, a_dim])
        w = make("w", w_np, [a_dim, b_dim])
        y = make("y", y_np, [batch, time, b_dim])
        return {
            # weight matmul: no common dims
            "linear": rf.matmul(x, w, reduce=a_dim),
            # batched, common dims on both sides
            "batched": rf.matmul(x, y, reduce=time),
            # reduce over two dims at once
            "two_reduce": rf.matmul(x, x, reduce=[time, a_dim]),
        }

    rf.select_backend_torch()
    ref = {
        k: (v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy(), v.dims)
        for k, v in _run(
            lambda name, arr, d: Tensor(name, dims=d, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr))
        ).items()
    }

    _rf_jax()
    got = {k: (numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)), v.dims) for k, v in _run(_make).items()}

    for key in sorted(ref):
        ref_raw, ref_dims = ref[key]
        got_raw, got_dims = got[key]
        assert got_dims == ref_dims, f"{key}: dims {got_dims} vs {ref_dims}"
        numpy.testing.assert_allclose(got_raw, ref_raw, rtol=1e-5, atol=1e-6, err_msg=f"matmul {key} differs")


def _jax_step_and_grad(params, step_fn):
    """
    Differentiate an RF computation under JAX: the shape every JAX engine step will have.

    JAX has no tape, so the loss must be a function OF the parameters:
    they go in as an explicit pytree, get bound into the rf.Parameter objects for the duration
    of the call, and jax.value_and_grad differentiates the whole thing.
    Which parameters are differentiated is decided here, from the list passed in,
    since JAX arrays carry no requires_grad.

    :param params: rf.Parameter list, in the order of the returned gradients
    :param step_fn: builds the scalar loss, takes no arguments
    :return: (loss, gradients)
    """
    orig = [p.raw_tensor for p in params]

    def _loss(raws):
        for p, raw in zip(params, raws):
            p.raw_tensor = raw
        try:
            return step_fn().raw_tensor
        finally:
            for p, raw in zip(params, orig):
                p.raw_tensor = raw

    return _loss, jax.value_and_grad(_loss)


def _model_and_loss(x, in_dim, out_dim, *, grad_scale=None):
    """builds Linear + LayerNorm and a scalar loss; optionally scales the gradient in between"""
    rf.set_random_seed(31)
    linear, norm = rf.Linear(in_dim, out_dim), rf.LayerNorm(in_dim)

    def _loss():
        h = norm(x)
        if grad_scale is not None:
            h = rf.scaled_gradient(h, grad_scale)
        y = linear(h)
        return rf.reduce_sum(y * y, axis=y.dims)

    return (linear, norm), _loss


@pytest.mark.parametrize("grad_scale", [None, 0.5, -1.0])
def test_gradients_vs_torch(grad_scale):
    import torch

    batch, time, in_dim, out_dim = Dim(3, name="batch"), Dim(5, name="time"), Dim(4, name="in"), Dim(6, name="out")
    x_np = numpy.random.RandomState(17).normal(size=(3, 5, 4)).astype("float32")

    rf.select_backend_torch()
    x_pt = Tensor("x", dims=[batch, time, in_dim], dtype="float32", raw_tensor=torch.from_numpy(x_np))
    mods_pt, loss_fn_pt = _model_and_loss(x_pt, in_dim, out_dim, grad_scale=grad_scale)
    params_pt = [p for mod in mods_pt for _, p in mod.named_parameters()]
    loss_pt = loss_fn_pt()
    grads_pt = torch.autograd.grad(loss_pt.raw_tensor, [p.raw_tensor for p in params_pt])
    ref_loss = float(loss_pt.raw_tensor.detach())
    ref_grads = [g.detach().cpu().numpy() for g in grads_pt]

    _rf_jax()
    x_jax = _make("x", x_np, [batch, time, in_dim])
    mods_jax, loss_fn_jax = _model_and_loss(x_jax, in_dim, out_dim, grad_scale=grad_scale)
    for mod_pt, mod_jax in zip(mods_pt, mods_jax):
        _copy_params_from(mod_pt, mod_jax)
    params_jax = [p for mod in mods_jax for _, p in mod.named_parameters()]
    assert len(params_jax) == len(params_pt)

    raw_fn, value_and_grad = _jax_step_and_grad(params_jax, loss_fn_jax)
    raws = [p.raw_tensor for p in params_jax]

    def _check(loss, grads, what):
        numpy.testing.assert_allclose(float(loss), ref_loss, rtol=1e-5, err_msg=f"loss differs {what}")
        for p, g, g_ref in zip(params_jax, grads, ref_grads):
            numpy.testing.assert_allclose(
                numpy.asarray(g), g_ref, rtol=1e-5, atol=1e-6, err_msg=f"grad of {p.name} differs {what}"
            )

    _check(*value_and_grad(raws), "eager")
    # the same step under jit: the RF code is traced there,
    # so anything wrongly baked static (a host-read shape, a cached artifact) shows up here and not above
    _check(*jax.jit(jax.value_and_grad(raw_fn))(raws), "under jit")


def test_scaled_gradient_ext_vs_torch():
    import torch

    batch, feat = Dim(3, name="batch"), Dim(4, name="feat")
    x_np = numpy.random.RandomState(23).normal(size=(3, 4)).astype("float32")

    def _run(x):
        # shift weighted by the summed absolute gradient over feat, i.e. the label-smoothing style transform
        y = rf.scaled_gradient_ext(x, scale=2.0, shift=-0.1, scale_shift_by_sum_over_axis=feat)
        return rf.reduce_sum(y * y, axis=y.dims)

    rf.select_backend_torch()
    x_pt_raw = torch.from_numpy(x_np).requires_grad_(True)
    loss_pt = _run(Tensor("x", dims=[batch, feat], dtype="float32", raw_tensor=x_pt_raw))
    (grad_pt,) = torch.autograd.grad(loss_pt.raw_tensor, x_pt_raw)

    _rf_jax()

    def _loss(x_raw):
        return _run(Tensor("x", dims=[batch, feat], dtype="float32", raw_tensor=x_raw)).raw_tensor

    loss_jax, grad_jax = jax.value_and_grad(_loss)(jnp.asarray(x_np))
    numpy.testing.assert_allclose(float(loss_jax), float(loss_pt.raw_tensor.detach()), rtol=1e-6)
    numpy.testing.assert_allclose(
        numpy.asarray(grad_jax), grad_pt.detach().cpu().numpy(), rtol=1e-5, atol=1e-6, err_msg="grad transform differs"
    )


def test_device():
    _rf_jax()
    x = _make("x", numpy.zeros((2,), dtype="float32"), [Dim(2, name="d")])
    assert x.device == "cpu"
    assert rf.copy_to_device(x, "cpu").device == "cpu"
