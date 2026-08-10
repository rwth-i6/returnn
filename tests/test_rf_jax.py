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

from returnn.util.basic import BehaviorVersion  # noqa: E402

# Both backends must see the SAME behavior version, else RF's version-dependent defaults differ
# and the comparison measures that instead of the backend:
# conv/pool masking is on from 23, attention dropout stops broadcasting at 19,
# TransformerDecoder ties its logits to the input embedding from 20.
# select_backend_jax requires >= 29 (a new backend has no legacy configs to stay compatible with),
# so pin the same minimum here, before the torch side builds anything.
BehaviorVersion.set_min_behavior_version(29)


def _rf_jax():
    rf.select_backend("jax")
    # On GPU, JAX computes float32 matmuls in TF32 by default while torch does not
    # (torch.backends.cuda.matmul.allow_tf32 is False by default),
    # which alone makes grads differ by ~1e-3 relative -- more than the difference these tests look for.
    # Same reasoning as the TF32 toggles in the packed GPU tests.
    jax.config.update("jax_default_matmul_precision", "highest")


def _no_tf32_torch():
    """torch side of the same: cuDNN convolutions default to TF32, matmuls do not"""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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

    _no_tf32_torch()
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
        # dims cannot be compared by identity here: each module construction makes its own internal Dims
        # (filter-dim0, conv_dummy_in, ...). Both modules come from the same code, so the dim ORDER matches,
        # and comparing the sizes in that order is the real check.
        src_shape = [d.dimension for d in p_src.dims]
        dst_shape = [d.dimension for d in p_dst.dims]
        assert src_shape == dst_shape, f"{name}: shape {src_shape} vs {dst_shape} ({p_src.dims} vs {p_dst.dims})"
        value = p_src.copy_compatible_to_dims_raw(p_src.dims).detach().cpu().numpy()
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

    _no_tf32_torch()
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

    _no_tf32_torch()
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

    _no_tf32_torch()
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

    _no_tf32_torch()
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


def test_shape_ops_vs_torch():
    import torch

    batch, time, feat = Dim(3, name="batch"), Dim(6, name="time"), Dim(4, name="feat")
    half_a, half_b = Dim(2, name="half_a"), Dim(2, name="half_b")
    merged = Dim(18, name="batch_time")
    split_t = [Dim(2, name="t_outer"), Dim(3, name="t_inner")]
    sliced = Dim(2, name="sliced")
    # every Dim must be shared between the two backend runs: RF compares dims by identity, not by name
    one_dim, feat2, stacked, expanded = (
        Dim(1, name="one"),
        Dim(8, name="feat2"),
        Dim(2, name="stacked"),
        Dim(2, name="expanded"),
    )
    rnd = numpy.random.RandomState(29)
    x_np = rnd.normal(size=(3, 6, 4)).astype("float32")
    y_np = rnd.normal(size=(3, 6, 4)).astype("float32")

    def _run(make):
        x = make("x", x_np, [batch, time, feat])
        y = make("y", y_np, [batch, time, feat])
        first, second = rf.split(x, axis=feat, out_dims=[half_a, half_b])
        out = {
            # merge_dims / concat / stack return (tensor, dim) at the rf level
            "merge": rf.merge_dims(x, dims=[batch, time], out_dim=merged)[0],
            "split_dims": rf.split_dims(x, axis=time, dims=split_t),
            "split0": first,
            "split1": second,
            "concat": rf.concat((x, feat), (y, feat), out_dim=feat2)[0],
            "stack": rf.stack([x, y], out_dim=stacked)[0],
            "unstack0": rf.unstack(x, axis=batch)[0],
            "unstack2": rf.unstack(x, axis=batch)[2],
            "expand": rf.expand_dim(x, dim=expanded),
            "squeeze": rf.squeeze(rf.expand_dim(x, dim=one_dim), axis=one_dim),
            "slice": rf.slice(x, axis=time, start=1, size=2, out_dim=sliced)[0],
            "cumsum": rf.cumsum(x, spatial_dim=time),
            "flip": rf.reverse_sequence(x, axis=time, handle_dynamic_dims=False),
            "clip": rf.clip_by_value(x, -0.5, 0.5),
            "lerp": rf.lerp(x, y, 0.3),
            "is_finite": rf.is_finite(x / 0.0),
            "is_inf": rf.is_infinite(x / 0.0),
            "full": rf.full(dims=[batch, feat], fill_value=2.5, dtype="float32"),
        }
        return {k: (v.copy_compatible_to_dims_raw(v.dims), v.dims) for k, v in out.items()}

    _no_tf32_torch()
    rf.select_backend_torch()
    ref = {
        k: (v.detach().cpu().numpy(), dims)
        for k, (v, dims) in _run(
            lambda name, arr, d: Tensor(name, dims=d, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr))
        ).items()
    }

    _rf_jax()
    got = {k: (numpy.asarray(v), dims) for k, (v, dims) in _run(_make).items()}

    assert set(got) == set(ref)
    for key in sorted(ref):
        ref_raw, ref_dims = ref[key]
        got_raw, got_dims = got[key]
        assert got_dims == ref_dims, f"{key}: dims {got_dims} vs {ref_dims}"
        numpy.testing.assert_allclose(got_raw, ref_raw, rtol=1e-6, atol=1e-6, err_msg=f"op {key} differs")


def _make_dyn_time(batch: Dim, seq_lens, name="time") -> Dim:
    """a dynamic time dim with the given seq lens, on the currently selected backend"""
    lens = rf.convert_to_tensor(numpy.array(seq_lens, dtype="int32"), dims=[batch], dtype="int32")
    return Dim(lens, name=name)


def test_masked_reduce_vs_torch():
    import torch

    batch = Dim(3, name="batch")
    feat = Dim(4, name="feat")
    seq_lens = [6, 3, 5]
    x_np = numpy.random.RandomState(31).normal(size=(3, 6, 4)).astype("float32")

    def _run(make):
        time = _make_dyn_time(batch, seq_lens)
        x = make("x", x_np, [batch, time, feat])
        return {
            "sum": rf.reduce_sum(x, axis=time),
            "mean": rf.reduce_mean(x, axis=time),
            "max": rf.reduce_max(x, axis=time),
            "min": rf.reduce_min(x, axis=time),
            "logsumexp": rf.reduce_logsumexp(x, axis=time),
            "argmax": rf.reduce_argmax(x, axis=time),
        }

    _no_tf32_torch()
    rf.select_backend_torch()
    ref = {
        k: v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy()
        for k, v in _run(
            lambda name, arr, d: Tensor(name, dims=d, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr))
        ).items()
    }

    _rf_jax()
    got = {k: numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)) for k, v in _run(_make).items()}

    for key in sorted(ref):
        numpy.testing.assert_allclose(got[key], ref[key], rtol=1e-5, atol=1e-6, err_msg=f"masked reduce {key} differs")

    # the masking must actually matter: an unmasked reduce over the padded frames gives something else
    _rf_jax()
    time = _make_dyn_time(batch, seq_lens)
    x = _make("x", x_np, [batch, time, feat])
    unmasked = numpy.asarray(rf.reduce_sum(x, axis=time, use_mask=False).copy_compatible_to_dims_raw([batch, feat]))
    assert not numpy.allclose(unmasked, ref["sum"]), "masking had no effect, so the test proves nothing"


def test_reduce_over_padded_batch():
    """
    A batch dim with a static capacity (what tracing needs) and a dynamic size is padded like
    any other axis, so a reduce over it has to mask. That the batch axis is special here is a
    property of the legacy ``Tensor.get_sequence_mask_broadcast`` only, not of the mask itself.
    """
    import jax.numpy as jnp

    _rf_jax()
    batch = Dim(None, name="batch", kind=Dim.Types.Batch)
    batch.dyn_size_ext = Tensor("batch_size", dims=(), dtype="int32", raw_tensor=jnp.asarray(3, dtype="int32"))
    batch.capacity = 5  # two padded entries, which must not reach any of the reductions
    feat = Dim(4, name="feat")
    values = numpy.arange(20, dtype="float32").reshape(5, 4)
    x = Tensor("x", dims=(batch, feat), dtype="float32", raw_tensor=jnp.asarray(values))

    assert batch.need_masking()
    for mode, ref in [
        ("sum", values[:3].sum(axis=0)),
        ("mean", values[:3].mean(axis=0)),
        ("max", values[:3].max(axis=0)),
        ("min", values[:3].min(axis=0)),
    ]:
        out = rf.reduce(x, mode=mode, axis=batch)
        numpy.testing.assert_allclose(numpy.asarray(out.raw_tensor), ref, rtol=1e-6, err_msg=mode)


def test_gather_scatter_pad_vs_torch():
    import torch

    batch, time, feat = Dim(3, name="batch"), Dim(5, name="time"), Dim(4, name="feat")
    vocab = Dim(7, name="vocab")
    padded_time = Dim(8, name="padded_time")
    rnd = numpy.random.RandomState(37)
    x_np = rnd.normal(size=(3, 5, 4)).astype("float32")
    emb_np = rnd.normal(size=(7, 4)).astype("float32")
    ids_np = rnd.randint(0, 7, size=(3, 5)).astype("int32")
    pos_np = rnd.randint(0, 5, size=(3, 5)).astype("int32")
    seg_np = numpy.array([[0, 0, 1, 1, 2], [2, 2, 0, 1, 1], [1, 0, 0, 2, 2]], dtype="int32")
    segments = Dim(3, name="segments")

    def _run(make):
        x = make("x", x_np, [batch, time, feat])
        emb = make("emb", emb_np, [vocab, feat])
        ids = Tensor("ids", dims=[batch, time], dtype="int32", sparse_dim=vocab, raw_tensor=make("i", ids_np, None))
        pos = Tensor("pos", dims=[batch, time], dtype="int32", sparse_dim=time, raw_tensor=make("p", pos_np, None))
        seg = Tensor("seg", dims=[batch, time], dtype="int32", sparse_dim=segments, raw_tensor=make("s", seg_np, None))
        return {
            # embedding lookup: indices carry no dim of the source
            "embed": rf.gather(emb, indices=ids, axis=vocab),
            # per-batch reordering: indices share the batch dim with the source
            "reorder": rf.gather(x, indices=pos, axis=time),
            # single index
            "first": rf.gather(x, indices=0, axis=time),
            "scatter_sum": rf.scatter(x, indices=seg, indices_dim=time, mode="sum", fill_value=0, out_dim=segments),
            "pad": rf.pad(
                x, axes=[time], padding=[(1, 2)], out_dims=[padded_time], handle_dynamic_dims=False, value=0.5
            )[0],
        }

    def _raw_torch(name, arr, dims):
        return (
            torch.from_numpy(arr)
            if dims is None
            else Tensor(name, dims=dims, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr))
        )

    def _raw_jax(name, arr, dims):
        return jnp.asarray(arr) if dims is None else _make(name, arr, dims)

    _no_tf32_torch()
    rf.select_backend_torch()
    ref = {
        k: (v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy(), v.dims) for k, v in _run(_raw_torch).items()
    }

    _rf_jax()
    got = {k: (numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)), v.dims) for k, v in _run(_raw_jax).items()}

    assert set(got) == set(ref)
    for key in sorted(ref):
        ref_raw, ref_dims = ref[key]
        got_raw, got_dims = got[key]
        assert got_dims == ref_dims, f"{key}: dims {got_dims} vs {ref_dims}"
        numpy.testing.assert_allclose(got_raw, ref_raw, rtol=1e-6, atol=1e-6, err_msg=f"op {key} differs")


def test_masked_select_vs_torch():
    """
    masked_select, and rf.pack_padded on top of it, which the real train step uses
    to pack the targets before the CE loss.
    """
    import torch

    batch = Dim(3, name="batch")
    feat = Dim(4, name="feat")
    seq_lens = [5, 2, 4]
    x_np = numpy.random.RandomState(17).normal(size=(3, 5, 4)).astype("float32")

    def _run(make, make_dyn_time):
        time_dim = make_dyn_time(batch, seq_lens)
        x = make("x", x_np, [batch, time_dim, feat])
        packed, pack_dim = rf.pack_padded(x, dims=[batch, time_dim])
        assert packed.dims[1:] == (feat,)
        return packed.copy_compatible_to_dims_raw(packed.dims), int(pack_dim.get_dim_value())

    rf.select_backend_torch()
    ref, ref_size = _run(
        lambda name, arr, d: Tensor(name, dims=d, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr)),
        lambda b, lens: Dim(
            Tensor("l", dims=[b], dtype="int32", raw_tensor=torch.tensor(lens, dtype=torch.int32)), name="time"
        ),
    )
    ref = ref.detach().cpu().numpy()

    _rf_jax()
    got, got_size = _run(_make, _make_dyn_time)
    got = numpy.asarray(got)

    assert got_size == ref_size == sum(seq_lens), f"{got_size} vs {ref_size}, expected {sum(seq_lens)}"
    numpy.testing.assert_allclose(got, ref, rtol=0, atol=0)
    # and it really is the unpadded content, in order
    expected = numpy.concatenate([x_np[i, :length] for i, length in enumerate(seq_lens)], axis=0)
    numpy.testing.assert_allclose(got, expected, rtol=0, atol=0)


def test_random_uniform_per_element_bounds():
    """
    SpecAugment draws the number and the position of its masks with PER-SEQ bounds,
    i.e. minval / maxval are tensors, not numbers.
    """
    _rf_jax()
    batch = Dim(4, name="batch")
    minval = _make("min", numpy.array([0, 2, 5, 7], dtype="int32"), [batch])
    maxval = _make("max", numpy.array([1, 4, 6, 20], dtype="int32"), [batch])
    for _ in range(5):
        out = rf.random_uniform([batch], minval=minval, maxval=maxval, dtype="int32")
        raw = numpy.asarray(out.raw_tensor)
        assert raw.dtype == numpy.int32 and raw.shape == (4,)
        assert (raw >= numpy.asarray(minval.raw_tensor)).all() and (raw < numpy.asarray(maxval.raw_tensor)).all(), raw


def test_amp_policy_dtypes():
    """
    Mixed precision: which op runs in which dtype, and that the parameters stay float32.
    See returnn.frontend.amp -- matmul/conv in the compute dtype, the sensitive ops in float32.
    """
    _rf_jax()
    in_dim, out_dim = Dim(4, name="in"), Dim(3, name="out")
    batch, time_dim = Dim(2, name="batch"), Dim(5, name="time")
    x_np = numpy.random.RandomState(23).normal(size=(2, 5, 4)).astype("float32")

    linear = rf.Linear(in_dim, out_dim)
    norm = rf.LayerNorm(out_dim)
    x = _make("x", x_np, [batch, time_dim, in_dim])

    with rf.set_amp_policy_ctx("bfloat16"):
        assert rf.get_amp_policy().compute_dtype == "bfloat16"
        y = linear(x)
        assert y.dtype == "bfloat16", y  # matmul: the compute dtype
        normed = norm(y)
        assert normed.dtype == "bfloat16", normed  # statistics in float32, result back to the input dtype
        probs = rf.softmax(y, axis=out_dim)
        assert probs.dtype == "float32", probs  # normalization: float32
        summed = rf.reduce_sum(y, axis=time_dim)
        assert summed.dtype == "float32", summed  # accumulation: float32
        # the parameters themselves are untouched, they are cast where they are USED
        assert linear.weight.dtype == "float32" and norm.scale.dtype == "float32"
        # elementwise ops are not in any list: they follow their inputs
        assert (y * 2.0).dtype == "bfloat16"

    # and without a policy nothing is cast at all
    assert rf.get_amp_policy() is None
    assert linear(x).dtype == "float32"

    # the values still agree with the float32 computation, within bf16 resolution
    with rf.set_amp_policy_ctx("bfloat16"):
        got = numpy.asarray(linear(x).copy_compatible_to_dims_raw([batch, time_dim, out_dim]).astype(jnp.float32))
    ref = numpy.asarray(linear(x).copy_compatible_to_dims_raw([batch, time_dim, out_dim]))
    numpy.testing.assert_allclose(got, ref, rtol=0.05, atol=0.05)


def test_conv_pool_vs_torch():
    import torch

    batch, time, in_dim, out_dim = Dim(3, name="batch"), Dim(12, name="time"), Dim(4, name="in"), Dim(6, name="out")
    freq, out2d = Dim(8, name="freq"), Dim(5, name="out2d")
    x_np = numpy.random.RandomState(41).normal(size=(3, 12, 4)).astype("float32")
    x2d_np = numpy.random.RandomState(43).normal(size=(3, 12, 8, 4)).astype("float32")

    def _build():
        rf.set_random_seed(31)
        return (
            rf.Conv1d(in_dim, out_dim, filter_size=3, padding="same"),
            rf.Conv1d(in_dim, out_dim, filter_size=3, padding="valid", strides=2),
            # depthwise, as in the Conformer conv block
            rf.Conv1d(in_dim, in_dim, filter_size=5, padding="same", groups=in_dim.dimension),
            rf.Conv2d(in_dim, out2d, filter_size=(3, 3), padding="same", strides=(2, 1)),
        )

    def _fwd(mods, x, x2d):
        c_same, c_valid, c_depth, c_2d = mods
        outs = {}
        outs["conv_same"] = c_same(x, in_spatial_dim=time)[0]
        outs["conv_valid_stride"] = c_valid(x, in_spatial_dim=time)[0]
        outs["conv_depthwise"] = c_depth(x, in_spatial_dim=time)[0]
        outs["conv2d"] = c_2d(x2d, in_spatial_dims=[time, freq])[0]
        outs["max_pool"] = rf.max_pool1d(x, pool_size=2, strides=2, padding="valid", in_spatial_dim=time)[0]
        outs["avg_pool"] = rf.pool1d(x, mode="avg", pool_size=3, strides=2, padding="same", in_spatial_dim=time)[0]
        return outs

    _no_tf32_torch()
    rf.select_backend_torch()
    mods_pt = _build()
    x_pt = Tensor("x", dims=[batch, time, in_dim], dtype="float32", raw_tensor=torch.from_numpy(x_np))
    x2d_pt = Tensor("x2d", dims=[batch, time, freq, in_dim], dtype="float32", raw_tensor=torch.from_numpy(x2d_np))
    ref = {
        k: (v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy(), v.dims)
        for k, v in _fwd(mods_pt, x_pt, x2d_pt).items()
    }

    _rf_jax()
    mods_jax = _build()
    for mod_pt, mod_jax in zip(mods_pt, mods_jax):
        _copy_params_from(mod_pt, mod_jax)
    x_jax = _make("x", x_np, [batch, time, in_dim])
    x2d_jax = _make("x2d", x2d_np, [batch, time, freq, in_dim])
    got = {
        k: (numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)), v.dims)
        for k, v in _fwd(mods_jax, x_jax, x2d_jax).items()
    }

    assert set(got) == set(ref)
    for key in sorted(ref):
        ref_raw, ref_dims = ref[key]
        got_raw, got_dims = got[key]
        assert got_dims == ref_dims, f"{key}: dims {got_dims} vs {ref_dims}"
        numpy.testing.assert_allclose(got_raw, ref_raw, rtol=1e-5, atol=1e-5, err_msg=f"{key} differs")


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("train_flag", [False, True])
def test_batch_norm_vs_torch(use_mask: bool, train_flag: bool):
    """
    BatchNorm, which the real Conformer uses in its conv block (and on the features).
    Both paths matter and they are different code: with masking, rf.BatchNorm does it with generic
    RF ops; without, it calls the backend's own ``batch_norm``.
    In train mode the current batch's statistics normalize AND the running ones get updated,
    so the test checks the updated running stats as well.
    """
    import torch

    batch = Dim(3, name="batch")
    in_dim = Dim(5, name="in")
    seq_lens = [7, 4, 2]
    x_np = numpy.random.RandomState(3).normal(size=(3, 7, 5)).astype("float32")
    init = {
        "gamma": numpy.random.RandomState(4).normal(size=(5,)).astype("float32"),
        "beta": numpy.random.RandomState(5).normal(size=(5,)).astype("float32"),
        "running_mean": numpy.random.RandomState(6).normal(size=(5,)).astype("float32") * 0.1,
        "running_variance": numpy.abs(numpy.random.RandomState(7).normal(size=(5,)).astype("float32")) + 0.5,
    }

    def _run(make, make_dyn_time):
        rf.init_train_step_run_ctx(train_flag=train_flag, step=0, epoch=1)
        bn = rf.BatchNorm(in_dim, use_mask=use_mask)
        for name, value in init.items():
            param = getattr(bn, name)
            param.assign(make(name, value, [in_dim]))
        time_dim = make_dyn_time(batch, seq_lens)
        out = bn(make("x", x_np, [batch, time_dim, in_dim]))
        return {
            "out": out.copy_compatible_to_dims_raw(out.dims),
            "running_mean": bn.running_mean.raw_tensor,
            "running_variance": bn.running_variance.raw_tensor,
        }

    _no_tf32_torch()
    rf.select_backend_torch()
    ref = {
        k: v.detach().cpu().numpy()
        for k, v in _run(
            lambda name, arr, d: Tensor(name, dims=d, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr.copy())),
            lambda b, lens: Dim(
                Tensor("l", dims=[b], dtype="int32", raw_tensor=torch.tensor(lens, dtype=torch.int32)), name="time"
            ),
        ).items()
    }

    _rf_jax()
    try:
        got = {k: numpy.asarray(v) for k, v in _run(_make, _make_dyn_time).items()}
    finally:
        # do not leave a train flag behind: the later comparisons run modules whose dropout
        # would then be live (and drawn from two independent RNG streams)
        rf.init_forward_step_run_ctx()

    for key in ref:
        numpy.testing.assert_allclose(got[key], ref[key], rtol=1e-5, atol=1e-6, err_msg=key)
    if train_flag:
        # the update actually happened, so the comparison above is not vacuous
        assert not numpy.allclose(got["running_mean"], init["running_mean"]), "running stats were not updated"
    else:
        numpy.testing.assert_allclose(got["running_mean"], init["running_mean"], rtol=0, atol=0)


def test_conformer_subsample_vs_torch():
    """the real frontend of the target model: ConformerConvSubsample as configured in test_rf_packed"""
    import torch
    from returnn.frontend.encoder.conformer import ConformerConvSubsample

    batch, time, in_dim = Dim(3, name="batch"), Dim(20, name="time"), Dim(8, name="feat")
    out_dims = [Dim(4, name="conv1"), Dim(4, name="conv2"), Dim(4, name="conv3")]
    x_np = numpy.random.RandomState(47).normal(size=(3, 20, 8)).astype("float32")

    def _build():
        rf.set_random_seed(31)
        return ConformerConvSubsample(
            in_dim,
            out_dims=out_dims,
            filter_sizes=[(3, 3), (3, 3), (3, 3)],
            pool_sizes=[(1, 2)],
            strides=[(1, 1), (3, 1), (2, 1)],
        )

    _no_tf32_torch()
    rf.select_backend_torch()
    mod_pt = _build()
    x_pt = Tensor("x", dims=[batch, time, in_dim], dtype="float32", raw_tensor=torch.from_numpy(x_np))
    out_pt, spatial_pt = mod_pt(x_pt, in_spatial_dim=time)
    ref = out_pt.copy_compatible_to_dims_raw(out_pt.dims).detach().cpu().numpy()

    _rf_jax()
    mod_jax = _build()
    _copy_params_from(mod_pt, mod_jax)
    x_jax = _make("x", x_np, [batch, time, in_dim])
    out_jax, spatial_jax = mod_jax(x_jax, in_spatial_dim=time)

    assert out_jax.dims == out_pt.dims, f"dims {out_jax.dims} vs {out_pt.dims}"
    assert spatial_jax == spatial_pt
    numpy.testing.assert_allclose(
        numpy.asarray(out_jax.copy_compatible_to_dims_raw(out_jax.dims)), ref, rtol=1e-5, atol=1e-5
    )


def test_attention_vs_torch():
    """the attentions of the target model: Conformer rel-pos self-att, decoder rotary causal self-att, cross-att"""
    import torch
    from returnn.frontend.attention import RelPosSelfAttention, RotaryPosCausalSelfAttention, CrossAttention

    batch, time, kv_time = Dim(3, name="batch"), Dim(7, name="time"), Dim(5, name="kv_time")
    model_dim, enc_dim = Dim(8, name="model"), Dim(8, name="enc")
    rnd = numpy.random.RandomState(53)
    x_np = rnd.normal(size=(3, 7, 8)).astype("float32")
    enc_np = rnd.normal(size=(3, 5, 8)).astype("float32")

    def _build():
        rf.set_random_seed(31)
        # att_dropout=0: the two backends have unrelated RNG streams, so any dropout would differ
        return (
            rf.SelfAttention(
                model_dim, model_dim, key_dim_total=model_dim, value_dim_total=model_dim, num_heads=2, att_dropout=0.0
            ),
            RelPosSelfAttention(
                model_dim, model_dim, key_dim_total=model_dim, value_dim_total=model_dim, num_heads=2, att_dropout=0.0
            ),
            RotaryPosCausalSelfAttention(
                model_dim,
                model_dim,
                key_dim_total=model_dim,
                value_dim_total=model_dim,
                num_heads=2,
                with_bias=False,
                att_dropout=0.0,
            ),
            CrossAttention(
                encoder_dim=enc_dim,
                query_in_dim=model_dim,
                proj_dim=model_dim,
                key_dim_total=model_dim,
                value_dim_total=model_dim,
                num_heads=2,
                att_dropout=0.0,
            ),
        )

    def _fwd(mods, x, enc):
        self_att, rel_pos_att, rope_att, cross_att = mods
        return {
            "self_att": self_att(x, axis=time),
            "rel_pos_att": rel_pos_att(x, axis=time),
            # causal self-att returns (output, state)
            "rope_causal_att": rope_att(x, axis=time)[0],
            "cross_att": cross_att(x, cross_att.transform_encoder(enc, axis=kv_time)),
        }

    _no_tf32_torch()
    rf.select_backend_torch()
    mods_pt = _build()
    x_pt = Tensor("x", dims=[batch, time, model_dim], dtype="float32", raw_tensor=torch.from_numpy(x_np))
    enc_pt = Tensor("enc", dims=[batch, kv_time, enc_dim], dtype="float32", raw_tensor=torch.from_numpy(enc_np))
    ref = {
        k: (v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy(), v.dims)
        for k, v in _fwd(mods_pt, x_pt, enc_pt).items()
    }

    _rf_jax()
    mods_jax = _build()
    for mod_pt, mod_jax in zip(mods_pt, mods_jax):
        _copy_params_from(mod_pt, mod_jax)
    x_jax = _make("x", x_np, [batch, time, model_dim])
    enc_jax = _make("enc", enc_np, [batch, kv_time, enc_dim])
    got = {
        k: (numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)), v.dims)
        for k, v in _fwd(mods_jax, x_jax, enc_jax).items()
    }

    assert set(got) == set(ref)
    for key in sorted(ref):
        ref_raw, ref_dims = ref[key]
        got_raw, got_dims = got[key]
        assert [d.dimension for d in got_dims] == [d.dimension for d in ref_dims], f"{key}: {got_dims} vs {ref_dims}"
        numpy.testing.assert_allclose(got_raw, ref_raw, rtol=1e-5, atol=1e-5, err_msg=f"{key} differs")


def test_losses_vs_torch():
    import torch

    batch = Dim(3, name="batch")
    vocab = Dim(7, name="vocab")
    seq_lens, target_lens = [9, 6, 8], [3, 2, 4]
    rnd = numpy.random.RandomState(59)
    logits_np = rnd.normal(size=(3, 9, 7)).astype("float32")
    # targets stay below the blank (the last index), as CTC expects
    targets_np = rnd.randint(0, 6, size=(3, 4)).astype("int32")
    probs_np = rnd.uniform(size=(3, 9, 7)).astype("float32")
    probs_np /= probs_np.sum(axis=-1, keepdims=True)

    def _run(make):
        time = _make_dyn_time(batch, seq_lens, name="time")
        targets_spatial = _make_dyn_time(batch, target_lens, name="targets_time")
        logits = make("logits", logits_np, [batch, time, vocab])
        logits.feature_dim = vocab
        probs = make("probs", probs_np, [batch, time, vocab])
        labels = Tensor(
            "labels",
            dims=[batch, targets_spatial],
            dtype="int32",
            sparse_dim=vocab,
            raw_tensor=make("t", targets_np, None),
        )
        sparse_ce_targets = Tensor(
            "ce_targets",
            dims=[batch, time],
            dtype="int32",
            sparse_dim=vocab,
            raw_tensor=make("c", targets_np[:, :1].repeat(9, axis=1), None),
        )
        return {
            "ce_sparse": rf.cross_entropy(
                estimated=logits, target=sparse_ce_targets, axis=vocab, estimated_type="logits"
            ),
            "ce_dense": rf.cross_entropy(estimated=logits, target=probs, axis=vocab, estimated_type="logits"),
            "ctc": rf.ctc_loss(
                logits=logits,
                targets=labels,
                input_spatial_dim=time,
                targets_spatial_dim=targets_spatial,
                blank_index=6,
            ),
        }

    def _raw_torch(name, arr, dims):
        if dims is None:
            return torch.from_numpy(arr)
        return Tensor(name, dims=dims, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr))

    def _raw_jax(name, arr, dims):
        return jnp.asarray(arr) if dims is None else _make(name, arr, dims)

    _no_tf32_torch()
    rf.select_backend_torch()
    ref = {k: v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy() for k, v in _run(_raw_torch).items()}

    _rf_jax()
    got = {k: numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)) for k, v in _run(_raw_jax).items()}

    for key in sorted(ref):
        numpy.testing.assert_allclose(got[key], ref[key], rtol=1e-4, atol=1e-4, err_msg=f"loss {key} differs")
    assert numpy.all(got["ctc"] > 0), "CTC loss should be positive"


def test_full_model_vs_torch():
    """
    The model of tests/test_rf_packed.py::test_full_model_packed_traced_program_replay, padded:
    Conformer encoder + TransformerDecoder + aux CTC head, losses CTC + CE.
    Forward AND every parameter gradient against torch, eager and under jax.jit.
    """
    import torch
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from returnn.frontend.decoder.transformer import TransformerDecoder, FeedForwardGated

    n_batch, t_max, s_max = 3, 32, 6
    batch = Dim(n_batch, name="batch")
    in_dim, vocab_dim, wb_vocab_dim = Dim(8, name="feat"), Dim(11, name="vocab"), Dim(12, name="vocab_wb")
    enc_dim = Dim(32, name="enc")
    seq_lens, tgt_lens = [32, 22, 15], [6, 2, 2]
    rnd = numpy.random.RandomState(61)
    x_np = rnd.normal(size=(n_batch, t_max, 8)).astype("float32")
    tgt_np = rnd.randint(0, 11, size=(n_batch, s_max)).astype("int32")

    def _build():
        rf.set_random_seed(31)
        encoder = ConformerEncoder(
            in_dim,
            enc_dim,
            ff_dim=Dim(24, name="enc-ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(4, name="conv1"), Dim(4, name="conv2"), Dim(4, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=2,
                conv_norm_opts={"use_mask": True},
            ),
            num_layers=2,
            dropout=0.0,
            att_dropout=0.0,
        )
        decoder = TransformerDecoder(
            enc_dim,
            vocab_dim,
            Dim(32, name="dec"),
            num_layers=2,
            num_heads=2,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            dropout=0.0,
            att_dropout=0.0,
            # explicit, because their defaults depend on the GLOBAL behavior version,
            # which the torch forward pass raises before the JAX model is built --
            # the two models would then differ (share_embedding ties logits to the input embedding)
            share_embedding=False,
            input_embedding_scale=1.0,
        )
        aux_logits = rf.Linear(enc_dim, wb_vocab_dim)
        return encoder, decoder, aux_logits

    def _step(mods, x, targets, time_dim, tgt_time):
        """the whole model: encoder -> aux CTC + decoder -> CE; returns the summed loss"""
        encoder, decoder, aux_logits = mods
        enc_out, enc_spatial = encoder(x, in_spatial_dim=time_dim)
        log_probs = rf.log_softmax(aux_logits(enc_out), axis=wb_vocab_dim)
        ctc = rf.ctc_loss(
            logits=log_probs,
            logits_normalized=True,
            targets=targets,
            input_spatial_dim=enc_spatial,
            targets_spatial_dim=tgt_time,
            blank_index=wb_vocab_dim.dimension - 1,
        )
        enc_state = decoder.transform_encoder(enc_out, axis=enc_spatial)
        logits, _ = decoder(
            targets,
            spatial_dim=tgt_time,
            state=decoder.default_initial_state(batch_dims=[batch]),
            encoder=enc_state,
        )
        ce = rf.cross_entropy(estimated=logits, target=targets, axis=vocab_dim, estimated_type="logits")
        return rf.reduce_sum(ctc, axis=ctc.dims) + rf.reduce_sum(ce, axis=ce.dims)

    def _dims(make_lens):
        # capacities: under jit every shape must be static, and the dims derived from these
        # (subsampled time, attention kv) inherit the capacity rather than reading a traced max
        time_dim = Dim(make_lens(seq_lens, "time_lens"), name="time", capacity=t_max)
        tgt_time = Dim(make_lens(tgt_lens, "tgt_lens"), name="tgt_time", capacity=s_max)
        return time_dim, tgt_time

    # --- torch reference
    # No train flag, set explicitly rather than inherited: the encoder's input_dropout defaults to 0.1,
    # and under a train flag the two backends would draw it from their own RNG streams.
    rf.init_forward_step_run_ctx()
    _no_tf32_torch()
    rf.select_backend_torch()
    mods_pt = _build()
    time_pt, tgt_time_pt = _dims(
        lambda lens, name: Tensor(name, dims=[batch], dtype="int32", raw_tensor=torch.tensor(lens, dtype=torch.int32))
    )
    x_pt = Tensor("x", dims=[batch, time_pt, in_dim], dtype="float32", raw_tensor=torch.from_numpy(x_np))
    tgt_pt = Tensor(
        "targets", dims=[batch, tgt_time_pt], dtype="int32", sparse_dim=vocab_dim, raw_tensor=torch.from_numpy(tgt_np)
    )
    loss_pt = _step(mods_pt, x_pt, tgt_pt, time_pt, tgt_time_pt)
    params_pt = [
        (f"{prefix}.{name}", p)
        for prefix, mod in zip(("enc", "dec", "aux"), mods_pt)
        for name, p in mod.named_parameters()
    ]
    # torch itself decides what is differentiable (non-float params are not), same as the packed test does
    trainable = [p.raw_tensor.requires_grad for _, p in params_pt]
    grads_pt = torch.autograd.grad(
        loss_pt.raw_tensor, [p.raw_tensor for (_, p), tr in zip(params_pt, trainable) if tr], allow_unused=True
    )
    ref_loss = float(loss_pt.raw_tensor.detach())
    ref_grads = [g.detach().cpu().numpy() if g is not None else None for g in grads_pt]

    # --- jax
    _rf_jax()
    mods_jax = _build()
    for mod_pt, mod_jax in zip(mods_pt, mods_jax):
        _copy_params_from(mod_pt, mod_jax)
    time_jax, tgt_time_jax = _dims(
        lambda lens, name: Tensor(name, dims=[batch], dtype="int32", raw_tensor=jnp.asarray(lens, dtype=jnp.int32))
    )
    x_jax = _make("x", x_np, [batch, time_jax, in_dim])
    tgt_jax = Tensor(
        "targets", dims=[batch, tgt_time_jax], dtype="int32", sparse_dim=vocab_dim, raw_tensor=jnp.asarray(tgt_np)
    )
    params_jax = [p for prefix, mod in zip(("enc", "dec", "aux"), mods_jax) for _, p in mod.named_parameters()]
    train_params = [p for p, tr in zip(params_jax, trainable) if tr]
    names = [n for (n, _), tr in zip(params_pt, trainable) if tr]

    raw_fn, value_and_grad = _jax_step_and_grad(
        train_params, lambda: _step(mods_jax, x_jax, tgt_jax, time_jax, tgt_time_jax)
    )
    raws = [p.raw_tensor for p in train_params]

    def _check(loss, grads, what):
        numpy.testing.assert_allclose(float(loss), ref_loss, rtol=1e-4, err_msg=f"loss differs {what}")
        for name, g, g_ref in zip(names, grads, ref_grads):
            if g_ref is None:
                continue  # unused in torch, nothing to compare against
            numpy.testing.assert_allclose(
                numpy.asarray(g), g_ref, rtol=1e-3, atol=1e-4, err_msg=f"grad of {name} differs {what}"
            )

    _check(*value_and_grad(raws), "eager")
    # NOT jitted here, deliberately: under jit every shape must be static, so every dynamic dim
    # would have to report a capacity instead of a traced max over the seq lens.
    # The input dims carry one (see _dims above), but dims DERIVED from them do not:
    # the subsampled time dim and the attention kv dim (bounded_by) lose it,
    # which is exactly the bound-shape plumbing the PyTorch graph-capture path needed
    # (set_bound_shapes_enabled + derived-capacity memoization).
    # Making that work for JAX belongs to the packed / bound-shape step, not here.
    # test_gradients_vs_torch already covers jit for a model with static dims.


def test_train_steps_and_checkpoint():
    """
    The training mechanics: optax updater + checkpoint I/O.

    Trains the Linear+LayerNorm model of test_gradients_vs_torch for a few steps on a fixed batch,
    which must reduce the loss, and round-trips a checkpoint through the file system.
    """
    import tempfile
    from returnn.jax.updater import Updater
    from returnn.jax.checkpoint import save_checkpoint, load_checkpoint, set_model_params, get_model_params

    _rf_jax()
    batch, time, in_dim, out_dim = Dim(3, name="batch"), Dim(5, name="time"), Dim(4, name="in"), Dim(6, name="out")
    x_np = numpy.random.RandomState(67).normal(size=(3, 5, 4)).astype("float32")
    x = _make("x", x_np, [batch, time, in_dim])
    mods, loss_fn = _model_and_loss(x, in_dim, out_dim)
    params = [p for mod in mods for _, p in mod.named_parameters()]

    raw_fn, value_and_grad = _jax_step_and_grad(params, loss_fn)
    step_fn = jax.jit(jax.value_and_grad(raw_fn))

    updater = Updater(optimizer_opts={"class": "adamw", "epsilon": 1e-8, "weight_decay": 0.0})
    raws = [p.raw_tensor for p in params]
    opt_state = updater.init(raws)

    losses = []
    for _ in range(20):
        loss, grads = step_fn(raws)
        losses.append(float(loss))
        raws, opt_state = updater.step(params=raws, grads=grads, opt_state=opt_state, learning_rate=0.05)
    for p, raw in zip(params, raws):
        p.raw_tensor = raw

    assert losses[-1] < losses[0] * 0.5, f"loss did not go down: {losses[0]} -> {losses[-1]}"
    assert all(numpy.isfinite(losses)), losses

    # checkpoint round-trip: write, perturb the model, load back, and the values must match again
    linear = mods[0]
    before = get_model_params(linear)
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = f"{tmp_dir}/model.orbax"
        save_checkpoint(linear, filename, step=20, epoch=1)
        for _, p in linear.named_parameters():
            p.assign(rf.zeros(p.dims, dtype=p.dtype))
        assert not numpy.allclose(get_model_params(linear)["weight"], before["weight"])
        set_model_params(linear, load_checkpoint(filename))
    after = get_model_params(linear)
    assert set(after) == set(before)
    for name in before:
        numpy.testing.assert_array_equal(after[name], before[name], err_msg=f"{name} did not round-trip")


def test_dataset_batches_to_jax():
    """
    The data path: a RETURNN dataset, through the shared backend-agnostic batching,
    into JAX tensors with the right dims and seq lens.
    """
    from returnn.datasets.generating import StaticDataset
    from returnn.tensor import TensorDict, batch_dim
    from returnn.jax.data import iter_dataset_batches

    _rf_jax()
    n_feat, n_classes = 5, 4
    seq_lens = [7, 3, 11, 5, 2, 9]
    rnd = numpy.random.RandomState(71)
    seqs = [
        {
            "data": rnd.normal(size=(t, n_feat)).astype("float32"),
            "classes": rnd.randint(0, n_classes, size=(t,)).astype("int32"),
        }
        for t in seq_lens
    ]
    dataset = StaticDataset(data=seqs, output_dim={"data": (n_feat, 2), "classes": (n_classes, 1)})

    time_dim = Dim(None, name="time")
    feat_dim, classes_dim = Dim(n_feat, name="feat"), Dim(n_classes, name="classes")
    extern_data = TensorDict(
        {
            "data": Tensor("data", dims=[batch_dim, time_dim, feat_dim], dtype="float32"),
            "classes": Tensor("classes", dims=[batch_dim, time_dim], dtype="int32", sparse_dim=classes_dim),
        }
    )

    # Everything is checked INSIDE the loop: the dims are the template's own and get filled in per
    # batch (as in the PyTorch pipeline), so a batch's dims are only valid until the next one.
    num_batches, total_seqs, total_frames = 0, 0, 0
    seen = {}
    for batch in iter_dataset_batches(dataset, extern_data=extern_data, batch_size=20, max_seqs=3):
        data, classes = batch.data["data"], batch.data["classes"]
        assert isinstance(data.raw_tensor, jax.Array) and isinstance(classes.raw_tensor, jax.Array)
        assert data.dtype == "float32" and classes.dtype == "int32"
        b_dim, t_dim, f_dim = data.dims
        assert b_dim is batch_dim and f_dim == feat_dim
        assert t_dim is time_dim, "the dynamic dim must be the template's own, not a fresh one"
        assert t_dim.is_dynamic() and t_dim.capacity == data.raw_tensor.shape[1]
        lens = numpy.asarray(t_dim.dyn_size_ext.raw_tensor)
        assert lens.max() == data.raw_tensor.shape[1], f"padded extent {data.raw_tensor.shape} vs lens {lens}"
        assert data.raw_tensor.shape[0] == len(lens) == int(batch_dim.dyn_size_ext.raw_tensor)
        # the padded frames must be zero, and the real ones must match the dataset
        raw = numpy.asarray(data.raw_tensor)
        for i, length in enumerate(lens):
            numpy.testing.assert_array_equal(raw[i, length:], 0.0, err_msg="padding is not zero")
            seen[tuple(raw[i, :length].flatten().tolist())] = True
        num_batches += 1
        total_seqs += len(lens)
        total_frames += int(lens.sum())
    assert num_batches > 1, f"{num_batches} batches, need several with DIFFERENT sizes for the check below"
    assert total_seqs == len(seq_lens), f"{total_seqs} seqs over all batches, expected {len(seq_lens)}"
    assert total_frames == sum(seq_lens), f"{total_frames} frames, expected {sum(seq_lens)}"
    # every seq appeared, with its original content
    for seq in seqs:
        assert tuple(seq["data"].flatten().tolist()) in seen, "a sequence went missing"

    # A second pass over a SMALLER dataset: the dims are shared and must not report the
    # previous pass's sizes. Dim caches its size max, so filling a dim in without resetting it
    # first leaves every later batch (e.g. the dev set after training) with a stale extent.
    small = StaticDataset(data=seqs[:2], output_dim={"data": (n_feat, 2), "classes": (n_classes, 1)})
    for batch in iter_dataset_batches(small, extern_data=extern_data, batch_size=20, max_seqs=3):
        data = batch.data["data"]
        b_dim, t_dim, _ = data.dims
        assert b_dim.get_dim_value() == data.raw_tensor.shape[0], "stale batch dim"
        assert t_dim.get_dim_value() == data.raw_tensor.shape[1], "stale time dim"


_EngineTestNumFeat, _EngineTestNumClasses = 5, 4


def _simple_train_setup(tmp_dir: str, *, dropout: float = 0.0, batch_norm: bool = False, **extra_config_opts):
    """
    A minimal but complete training config, as :mod:`returnn.__main__` would set it up.

    :param tmp_dir: for the model and the learning-rate file
    :param dropout: if set, the step draws random numbers, which is what the RNG stream tests need
    :param batch_norm: if set, the model normalizes its input with rf.BatchNorm, whose running
        statistics are auxiliary parameters written by the step itself, not by the optimizer
    :param extra_config_opts: added to the config
    :return: (config, func creating the dataset)
    """
    from returnn.config import Config, set_global_config
    from returnn.datasets.generating import StaticDataset
    from returnn.tensor import batch_dim
    from returnn.util.basic import BackendEngine

    _rf_jax()
    n_feat, n_classes = _EngineTestNumFeat, _EngineTestNumClasses
    rnd = numpy.random.RandomState(73)
    seqs = [
        {
            "data": rnd.normal(size=(t, n_feat)).astype("float32"),
            "classes": rnd.randint(0, n_classes, size=(t,)).astype("int32"),
        }
        for t in [7, 3, 11, 5, 2, 9, 6, 8]
    ]

    def _make_dataset():
        return StaticDataset(data=seqs, output_dim={"data": (n_feat, 2), "classes": (n_classes, 1)})

    # the dims are declared once and shared by extern_data and the model, as a real config does
    time_dim = Dim(None, name="time")
    in_dim = Dim(n_feat, name="in")
    classes_dim = Dim(n_classes, name="classes")

    def get_model(*, epoch: int, step: int, **_kwargs):
        """model, as the config API defines it"""

        class _Model(rf.Module):
            def __init__(self):
                super().__init__()
                self.linear = rf.Linear(in_dim, classes_dim)
                self.norm = rf.BatchNorm(in_dim) if batch_norm else None
                self.in_dim, self.out_dim = in_dim, classes_dim

        return _Model()

    def train_step(*, model, extern_data, **_kwargs):
        """train step, as the config API defines it"""
        data = extern_data["data"]
        targets = extern_data["classes"]
        time_dim = data.dims[1]
        if dropout:
            data = rf.dropout(data, dropout, axis=data.dims[1:])
        if model.norm is not None:
            data = model.norm(data)
        logits = model.linear(data)
        loss = rf.cross_entropy(estimated=logits, target=targets, axis=model.out_dim, estimated_type="logits")
        rf.get_run_ctx().mark_as_loss(loss, "ce", custom_inv_norm_factor=time_dim.get_size_tensor())

    config = Config(
        {
            "backend": "jax",
            "extern_data": {
                "data": {"dims": [batch_dim, time_dim, in_dim], "dtype": "float32"},
                "classes": {"dims": [batch_dim, time_dim], "sparse_dim": classes_dim, "dtype": "int32"},
            },
            "get_model": get_model,
            "train_step": train_step,
            "batch_size": 20,
            "max_seqs": 3,
            "num_epochs": 2,
            "learning_rate": 0.05,
            "optimizer": {"class": "adamw", "weight_decay": 0.0},
            "model": f"{tmp_dir}/model",
            "learning_rate_file": f"{tmp_dir}/lr",
            **extra_config_opts,
        }
    )
    # the engine machinery reads the global config (as returnn.__main__.init_config sets it up),
    # and selects the backend from it
    set_global_config(config)
    BackendEngine.select_engine(config=config)
    assert BackendEngine.is_jax_selected()
    return config, _make_dataset


def test_engine_train_from_config():
    """
    The engine end to end: a config with get_model / train_step, a dataset, two epochs of training,
    checkpoints written per epoch, and the train score going down.
    """
    import tempfile
    from returnn.jax.engine import Engine
    from returnn.jax.checkpoint import load_checkpoint

    with tempfile.TemporaryDirectory() as tmp_dir:
        config, make_dataset = _simple_train_setup(tmp_dir)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset(), dev_data=make_dataset())
        engine.train()

        # a checkpoint per epoch, holding the model's parameters
        for epoch in (1, 2):
            params = load_checkpoint(f"{tmp_dir}/model.{epoch:03d}.orbax")
            assert set(params) == {"linear.weight", "linear.bias"}, sorted(params)
            assert params["linear.weight"].shape == (_EngineTestNumFeat, _EngineTestNumClasses)
        # and training moved the parameters
        assert not numpy.allclose(
            load_checkpoint(f"{tmp_dir}/model.001.orbax")["linear.weight"],
            load_checkpoint(f"{tmp_dir}/model.002.orbax")["linear.weight"],
        ), "the parameters did not change between epochs"

        scores = engine.learning_rate_control.epoch_data
        train_scores = [scores[ep].error["train_loss_ce"] for ep in (1, 2)]
        assert train_scores[1] < train_scores[0], f"train score did not improve: {train_scores}"
        assert "dev_loss_ce" in scores[1].error, sorted(scores[1].error)


@pytest.mark.parametrize("time_multiple", [0, 4])
def test_engine_train_jit(time_multiple: int):
    """
    The compiled step (``jax_jit``) trains the same as the eager one -- also with the time axis
    padded up to a multiple, which is what keeps the number of compiled variants small
    (a compiled step is specialized per input shape).
    """
    import tempfile
    from returnn.jax.engine import Engine

    def _run(**extra):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config, make_dataset = _simple_train_setup(tmp_dir, **extra)
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=make_dataset(), dev_data=make_dataset())
            engine.train()
            scores = engine.learning_rate_control.epoch_data
            return (
                {name: numpy.asarray(param.raw_tensor) for name, param in engine.model.named_parameters()},
                [scores[ep].error["train_loss_ce"] for ep in (1, 2)],
            )

    eager_params, eager_scores = _run()
    jit_params, jit_scores = _run(jax_jit={"time_multiple": time_multiple} if time_multiple else True)
    numpy.testing.assert_allclose(jit_scores, eager_scores, rtol=1e-5)
    for name, value in eager_params.items():
        numpy.testing.assert_allclose(jit_params[name], value, rtol=1e-4, atol=1e-6, err_msg=name)


def test_checkpoint_is_ocdbt_and_compact():
    """
    Checkpoints must use OCDBT: 18 inodes vs 513 at 169 arrays, ~3.6k vs ~103k over 100 epochs.
    A silent fallback to the plain layout breaks nothing, it just exhausts inodes days later.
    """
    import tempfile
    from returnn.jax.checkpoint import save_checkpoint, load_checkpoint

    _rf_jax()
    in_dim, out_dim = Dim(7, name="in"), Dim(5, name="out")
    model = rf.Linear(in_dim, out_dim)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = f"{tmp_dir}/model.001.orbax"
        save_checkpoint(model, path, step=3, epoch=1)

        assert os.path.isdir(path), "an Orbax checkpoint is a directory"
        assert os.path.exists(f"{path}/manifest.ocdbt"), sorted(os.listdir(path))
        # the metadata file the engine's existence check keys on
        assert os.path.exists(f"{path}/_CHECKPOINT_METADATA"), sorted(os.listdir(path))
        inodes = sum(len(dirs) + len(files) for _root, dirs, files in os.walk(path)) + 1
        assert inodes < 40, f"{inodes} inodes for a 2-array checkpoint -- OCDBT not in use?"

        restored = load_checkpoint(path)
        assert set(restored) == {"weight", "bias"}, sorted(restored)  # no _step / _epoch
        for name, param in model.named_parameters():
            numpy.testing.assert_allclose(restored[name], numpy.asarray(param.raw_tensor), rtol=0, atol=0)


def test_get_existing_models_ignores_empty_checkpoint_dir():
    """
    An empty checkpoint directory is not a checkpoint.
    Pre-created job outputs and runs killed mid-save both leave one,
    and keying on the directory would resume a fresh run from an unreadable "last" epoch.
    """
    import tempfile
    from returnn.config import Config
    from returnn.engine.base import EngineBase

    _rf_jax()
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = Config({"backend": "jax", "model": f"{tmp_dir}/model", "num_epochs": 3, "task": "train"})
        os.makedirs(f"{tmp_dir}/model.001.orbax")
        os.makedirs(f"{tmp_dir}/model.001.opt.orbax")
        assert EngineBase.get_existing_models(config) == {}


def test_engine_batch_norm_running_stats():
    """
    rf.BatchNorm's running statistics are written BY THE STEP, and stay out of the optimizer.

    ``rf.Parameter.trainable`` reports them as None, not False (the resolution lives in the setter),
    so splitting on it hands them to the optimizer, where weight decay shrinks them,
    and drops the step's own updates. Train does not notice; only eval degrades, epoch by epoch.
    """
    import tempfile
    from returnn.jax.engine import Engine

    with tempfile.TemporaryDirectory() as tmp_dir:
        config, make_dataset = _simple_train_setup(tmp_dir, batch_norm=True, num_epochs=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset(), dev_data=make_dataset())

        params = dict(engine.model.named_parameters())
        # noinspection PyProtectedMember
        other = {engine._param_names[i] for i in engine._other_param_idx}
        assert other == {"norm.running_mean", "norm.running_variance"}

        engine.train()

        mean = numpy.asarray(params["norm.running_mean"].raw_tensor)
        variance = numpy.asarray(params["norm.running_variance"].raw_tensor)
        # Not the values (standard-normal data lands near (0, 1) either way) but their variation:
        # a weight-decayed constant stays uniform, a real estimate differs per feature.
        assert not numpy.allclose(mean, 0.0) and not numpy.allclose(variance, 1.0)
        assert mean.std() > 0.0 and variance.std() > 0.0


def test_pad_raws_to_bucket_changes_nothing():
    """
    Padding the batch axis with zero-length sequences is only safe if every masked op ignores them.
    If it does not, training would be silently wrong rather than broken, so this compares the
    masked results of a batch against the same batch padded up to a bucket -- no RNG involved.
    """
    import jax.numpy as jnp
    from returnn.tensor import batch_dim, TensorDict

    # noinspection PyProtectedMember
    from returnn.jax.data import fill_extern_data, pad_raws_to_bucket, reset_extern_data_dims

    _rf_jax()
    time_dim = Dim(None, name="time")
    feat = Dim(3, name="feat")
    extern_data = TensorDict()
    extern_data.data["data"] = Tensor("data", dims=[batch_dim, time_dim, feat], dtype="float32")

    values = numpy.arange(2 * 4 * 3, dtype="float32").reshape(2, 4, 3)
    raws = {
        "data": jnp.asarray(values),
        "data_seq_lens": jnp.asarray([4, 2], dtype="int32"),
        "batch_dim": jnp.asarray(2, dtype="int32"),
    }
    bucket = {"batch_dim": 5, "data": 7}
    padded = pad_raws_to_bucket(raws, extern_data=extern_data, bucket=bucket)
    assert padded["data"].shape == (5, 7, 3) and padded["data_seq_lens"].shape == (5,)
    assert list(numpy.asarray(padded["data_seq_lens"])) == [4, 2, 0, 0, 0]

    results = []
    for name, one in [("plain", raws), ("padded", padded)]:
        data = fill_extern_data(extern_data, one)["data"]
        results.append(
            {
                "sum": numpy.asarray(rf.reduce_sum(data, axis=data.dims).raw_tensor),
                "mean": numpy.asarray(rf.reduce_mean(data, axis=data.dims).raw_tensor),
                "max": numpy.asarray(rf.reduce_max(data, axis=data.dims).raw_tensor),
            }
        )
        reset_extern_data_dims(extern_data)
        del name
    for key in results[0]:
        numpy.testing.assert_allclose(results[1][key], results[0][key], rtol=1e-6, err_msg=key)


def test_jit_time_multiple_validation():
    """
    ``time_multiple`` is in the unit of the axis it pads. One number for keys whose axes are
    different dims silently pads a label sequence by an audio-sample granularity -- measured once
    at 16000, where the decoder self-attention became a 152 GiB buffer and the compile OOMed.
    And keys sharing a dim must pad to the same extent, or the dim needs two capacities.
    """
    from returnn.tensor import batch_dim, TensorDict

    # noinspection PyProtectedMember
    from returnn.jax.engine import _check_time_multiple

    _rf_jax()
    audio_time, text_time = Dim(None, name="audio_time"), Dim(None, name="text_time")
    feat = Dim(4, name="feat")
    separate = TensorDict()
    separate.data["audio"] = Tensor("audio", dims=[batch_dim, audio_time, feat], dtype="float32")
    separate.data["text"] = Tensor("text", dims=[batch_dim, text_time], dtype="int32")
    shared = TensorDict()
    shared.data["data"] = Tensor("data", dims=[batch_dim, audio_time, feat], dtype="float32")
    shared.data["classes"] = Tensor("classes", dims=[batch_dim, audio_time], dtype="int32")

    with pytest.raises(NotImplementedError, match="different dims and different units"):
        _check_time_multiple(16_000, extern_data=separate)
    _check_time_multiple({"audio": 16_000, "text": 8}, extern_data=separate)  # per key: fine
    _check_time_multiple(0, extern_data=separate)  # off: nothing to check

    _check_time_multiple(4, extern_data=shared)  # one dim, so one number is unambiguous
    with pytest.raises(NotImplementedError, match="sharing one dim"):
        _check_time_multiple({"data": 4, "classes": 8}, extern_data=shared)
    with pytest.raises(NotImplementedError, match="unknown data keys"):
        _check_time_multiple({"nope": 4}, extern_data=shared)


def test_engine_jit_donates_buffers():
    """
    The compiled step donates the parameters, the optimizer state and the RNG key -- the arguments
    it returns a new version of. XLA then writes the new values into those buffers instead of
    allocating a second set, which is what keeps the peak at one copy of each.

    Donation DELETES the input buffers, so this also checks the engine's side of that contract:
    it must hold exactly one reference to each and replace it right after the step.
    """
    import tempfile
    import jax
    from returnn.jax.engine import Engine

    with tempfile.TemporaryDirectory() as tmp_dir:
        config, make_dataset = _simple_train_setup(tmp_dir, jax_jit=True, num_epochs=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset(), dev_data=make_dataset())
        # noinspection PyProtectedMember
        params_before = [engine._params[i].raw_tensor for i in engine._train_param_idx]
        # noinspection PyProtectedMember
        opt_state_before = [
            leaf for leaf in jax.tree_util.tree_leaves(engine._opt_state) if hasattr(leaf, "is_deleted")
        ]
        assert params_before and opt_state_before
        engine.train()

        assert all(raw.is_deleted() for raw in params_before), "the parameter buffers were not donated"
        assert all(leaf.is_deleted() for leaf in opt_state_before), "the optimizer state was not donated"
        # and what the engine holds now is live, and is what training produced
        for name, param in engine.model.named_parameters():
            assert not param.raw_tensor.is_deleted(), name
            assert numpy.all(numpy.isfinite(numpy.asarray(param.raw_tensor))), name


def test_engine_jit_rng_advances():
    """
    The RNG stream goes through the compiled step as a value, in and out.
    If it did not -- if the step read the backend's global key -- that key would be captured once,
    when the step is traced, and every step would then draw the very same dropout mask.
    """
    import tempfile
    import jax.numpy as jnp
    from returnn.jax.engine import Engine

    # noinspection PyProtectedMember
    from returnn.jax.frontend._backend import JaxBackend

    with tempfile.TemporaryDirectory() as tmp_dir:
        config, make_dataset = _simple_train_setup(tmp_dir, dropout=0.5, jax_jit=True)

        def _step_with(key):
            """
            :param key: the RNG key to run one step with
            :return: (loss, the key the step returns)

            A fresh engine per call: the step donates the parameters and the optimizer state,
            so their buffers are gone afterwards. The model is seeded, so every engine here
            starts from the very same parameters.
            """
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=make_dataset(), dev_data=make_dataset())
            # noinspection PyProtectedMember
            batch_raws, _ = next(iter(engine._iter_batches(make_dataset(), train=True)))
            # string data cannot be an argument of a compiled function, so the step does not get it
            # noinspection PyProtectedMember
            assert "seq_tag" in batch_raws and "seq_tag" not in engine._step_raws(batch_raws)
            # noinspection PyProtectedMember
            # (train raws, other raws, opt state, rng key, loss, losses, grad norm)
            _, _, _, key_out, loss, _, _ = engine._train_step(
                [engine._params[i].raw_tensor for i in engine._train_param_idx],
                [engine._params[i].raw_tensor for i in engine._other_param_idx],
                engine._step_raws(batch_raws),
                engine._opt_state,
                key,
                jnp.asarray(0.05, dtype=jnp.float32),
                jnp.asarray(0, dtype=jnp.int32),
                1,
            )
            return float(loss), key_out

        # everything but the key held fixed, so any difference in the loss comes from the RNG
        key_1 = JaxBackend._get_rng_key_()
        loss_1, key_2 = _step_with(key_1)
        loss_2, _ = _step_with(key_2)
        loss_3, _ = _step_with(key_1)
        assert loss_1 == loss_3, "the same key gave a different result"
        assert loss_1 != loss_2, "the RNG stream did not advance across steps"


def test_engine_sets_rf_default_device():
    """
    RF puts the tensors IT creates (constants, ranges, random, the correction factor of a masked
    reduce) on ``rf.get_default_device()``, while JAX puts everything else on its own default.
    If the engine leaves the first one unset, those two disagree on any machine with an accelerator
    and JAX rejects the computation -- which is invisible on a cpu-only box, hence this test.
    """
    import tempfile
    import jax
    from returnn.jax.engine import Engine

    with tempfile.TemporaryDirectory() as tmp_dir:
        config, make_dataset = _simple_train_setup(tmp_dir, device="cpu")
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset())
        assert rf.get_default_device() == "cpu"

    with tempfile.TemporaryDirectory() as tmp_dir:
        # unset in the config: resolved to what JAX itself would use, never left open
        config, make_dataset = _simple_train_setup(tmp_dir)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset())
        assert rf.get_default_device() == jax.devices()[0].platform


def test_engine_continue_from_checkpoint():
    """
    A second run in the same model dir continues from the existing checkpoint
    instead of silently starting over and overwriting it.
    """
    import tempfile
    from returnn.jax.engine import Engine
    from returnn.jax.checkpoint import load_checkpoint

    with tempfile.TemporaryDirectory() as tmp_dir:
        config, make_dataset = _simple_train_setup(tmp_dir, num_epochs=1)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset(), dev_data=make_dataset())
        engine.train()
        assert engine.epoch == 2  # after the last trained epoch
        saved = load_checkpoint(f"{tmp_dir}/model.001.orbax")

        config, make_dataset = _simple_train_setup(tmp_dir, num_epochs=2)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset(), dev_data=make_dataset())
        assert engine.epoch == 2, "did not continue after the existing checkpoint"
        for name, param in engine.get_model().named_parameters():
            numpy.testing.assert_allclose(numpy.asarray(param.raw_tensor), saved[name], rtol=0, atol=0)


def test_engine_dynamic_learning_rate():
    """
    The config's ``dynamic_learning_rate`` decides the learning rate of each STEP
    (that is where the piecewise-linear schedules of the real setups live),
    on top of the epoch-level rate from the learning-rate control.
    """
    import tempfile
    from returnn.jax.engine import Engine

    calls = []

    def _dyn_lr(*, global_train_step: int, epoch: int, epoch_continuous, learning_rate: float, **_kwargs):
        """dynamic_learning_rate, as the config API defines it"""
        calls.append((global_train_step, epoch, epoch_continuous, learning_rate))
        return 0.0  # freezes the parameters, which is what the test checks

    with tempfile.TemporaryDirectory() as tmp_dir:
        config, make_dataset = _simple_train_setup(tmp_dir, num_epochs=1, dynamic_learning_rate=_dyn_lr)
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset())
        before = {name: numpy.asarray(p.raw_tensor) for name, p in engine.get_model().named_parameters()}
        engine.train()
        after = {name: numpy.asarray(p.raw_tensor) for name, p in engine.get_model().named_parameters()}

    assert calls, "dynamic_learning_rate was never called"
    # The first call is the one at the START of the epoch, which is what records
    # :meta:effective_learning_rate in the learning_rates file (the PyTorch engine does the same).
    # Being at the start, its epoch_continuous is exactly 0, which the per-step calls never are.
    init_call, step_calls = calls[0], calls[1:]
    assert (init_call[0], init_call[2], init_call[3]) == (0, 0, 0.05), init_call
    steps = [c[0] for c in step_calls]
    assert steps == list(range(len(step_calls))), f"global_train_step not consecutive from 0: {steps}"
    assert all(c[1] == 1 for c in calls), calls  # epoch
    for _, _, epoch_continuous, learning_rate in step_calls:
        assert epoch_continuous is not None and 0.0 < epoch_continuous <= 1.0, calls
        assert learning_rate == 0.05, calls  # the epoch-level rate the function gets to modify
    for name, value in before.items():
        numpy.testing.assert_allclose(after[name], value, rtol=0, atol=0, err_msg=f"{name} moved at lr 0")


def test_updater_weight_decay_blacklist():
    """
    ``weight_decay_modules_blacklist`` (and the bias rule) decide which parameters decay,
    as in the PyTorch updater. Checked through the update itself, on zero gradients,
    where adamw's whole update IS the decay.
    """
    import jax.numpy as jnp
    from returnn.jax.updater import Updater

    _rf_jax()
    in_dim, out_dim = Dim(3, name="in"), Dim(2, name="out")

    class _Model(rf.Module):
        def __init__(self):
            super().__init__()
            self.emb = rf.Embedding(in_dim, out_dim)
            self.linear = rf.Linear(out_dim, out_dim)

    model = _Model()
    for _, param in model.named_parameters():
        ones = jnp.ones(param.batch_shape, dtype=param.dtype)
        param.assign(Tensor("v", dims=param.dims, dtype=param.dtype, raw_tensor=ones))
    names = [name for name, _ in model.named_parameters()]
    params = [param.raw_tensor for _, param in model.named_parameters()]
    assert set(names) == {"emb.weight", "linear.weight", "linear.bias"}, names

    updater = Updater(
        optimizer_opts={
            "class": "adamw",
            "weight_decay": 0.5,
            "weight_decay_modules_blacklist": ["rf.Embedding"],
        },
        model=model,
        param_names=names,
    )
    new_params, _ = updater.step(
        params=params, grads=[jnp.zeros_like(p) for p in params], opt_state=updater.init(params), learning_rate=1.0
    )
    got = {name: float(numpy.asarray(p).flat[0]) for name, p in zip(names, new_params)}
    # decayed: linear.weight. not decayed: emb.weight (blacklisted module), linear.bias (a bias)
    assert got["linear.weight"] == 0.5, got
    assert got["emb.weight"] == 1.0, got
    assert got["linear.bias"] == 1.0, got


def test_engine_train_amp():
    """
    ``jax_amp`` end to end: the step computes in bfloat16, while the checkpoint
    (parameters and optimizer state) stays float32, as with PyTorch AMP.
    """
    import tempfile
    from returnn.jax.engine import Engine
    from returnn.jax.checkpoint import load_checkpoint

    with tempfile.TemporaryDirectory() as tmp_dir:
        config, make_dataset = _simple_train_setup(tmp_dir, num_epochs=1, jax_amp="bfloat16")
        engine = Engine(config=config)
        engine.init_train_from_config(train_data=make_dataset(), dev_data=make_dataset())
        engine.train()

        params = load_checkpoint(f"{tmp_dir}/model.001.orbax")
        for name, value in params.items():
            assert value.dtype == numpy.float32, f"{name} is {value.dtype}, expected float32"
        scores = engine.learning_rate_control.epoch_data[1].error
        assert numpy.isfinite(scores["train_loss_ce"]), scores
    # the policy is scoped to the step, so nothing outside the engine is left in bfloat16
    assert rf.get_amp_policy() is None


def test_engine_unsupported_config_opts():
    """
    Options which other engines implement and this one does not are rejected, not ignored.
    """
    import tempfile
    import pytest
    from returnn.jax.engine import Engine

    with tempfile.TemporaryDirectory() as tmp_dir:
        for opts in [
            {"accum_grad_multiple_step": 2},
            {"preload_from_files": {"base": {"filename": "/dev/null"}}, "jax_distributed": {"reduce_type": "grad"}},
            # a config copied from a PyTorch setup: the torch_ names are rejected too,
            # rather than silently doing nothing
            {"torch_amp": "bfloat16"},
            {"chunking": "200:100"},
            {"forward_step": lambda **_kwargs: None},
        ]:
            config, make_dataset = _simple_train_setup(tmp_dir, **opts)
            dataset = make_dataset()
            engine = Engine(config=config)
            with pytest.raises(NotImplementedError) as exc:
                engine.init_train_from_config(train_data=dataset)
            for key in opts:
                assert key in str(exc.value), f"{key} not in {exc.value}"

        # the no-op values of those options do not trip the check
        config, make_dataset = _simple_train_setup(tmp_dir, accum_grad_multiple_step=1, save_interval=1)
        Engine(config=config).init_train_from_config(train_data=make_dataset())


def test_stft_and_logmel_vs_torch():
    """
    stft, and the log-mel feature extraction the real recipe uses
    (`rf.audio.log_mel_filterbank_from_raw`), which is what a production config feeds the model.
    """
    import torch

    batch, time_dim = Dim(2, name="batch"), Dim(4000, name="samples")
    audio_np = numpy.random.RandomState(79).normal(size=(2, 4000)).astype("float32") * 0.1

    def _run(make):
        audio = make("audio", audio_np, [batch, time_dim])
        out = {}
        # rf.stft derives the out dims itself and returns them
        stft, _, _ = rf.stft(audio, in_spatial_dim=time_dim, frame_step=160, frame_length=400, fft_length=400)
        out["stft_abs"] = rf.abs(stft)
        feat_dim = Dim(80, name="logmel")
        mel, mel_spatial = rf.audio.log_mel_filterbank_from_raw(
            audio, in_spatial_dim=time_dim, out_dim=feat_dim, sampling_rate=16_000
        )
        out["log_mel"] = mel
        return out

    _no_tf32_torch()
    rf.select_backend_torch()
    ref = {
        k: v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy()
        for k, v in _run(
            lambda name, arr, d: Tensor(name, dims=d, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr))
        ).items()
    }

    _rf_jax()
    got = {k: numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)) for k, v in _run(_make).items()}

    for key in sorted(ref):
        assert got[key].shape == ref[key].shape, f"{key}: {got[key].shape} vs {ref[key].shape}"
        numpy.testing.assert_allclose(got[key], ref[key], rtol=1e-4, atol=1e-4, err_msg=f"{key} differs")


def test_top_k_vs_torch():
    import torch

    batch, time_dim, vocab = Dim(3, name="batch"), Dim(5, name="time"), Dim(7, name="vocab")
    k_dim = Dim(3, name="k")
    scores_np = numpy.random.RandomState(83).normal(size=(3, 5, 7)).astype("float32")

    def _run(make):
        scores = make("scores", scores_np, [batch, time_dim, vocab])
        values, indices, _ = rf.top_k(scores, axis=vocab, k=3, k_dim=k_dim)
        # over two axes at once, as beam search does (beam x vocab)
        values2, (idx_time, idx_vocab), _ = rf.top_k(scores, axis=[time_dim, vocab], k=3, k_dim=k_dim)
        return {
            "values": values,
            "indices": indices,
            "values_2d": values2,
            "idx_time": idx_time,
            "idx_vocab": idx_vocab,
        }

    _no_tf32_torch()
    rf.select_backend_torch()
    ref = {
        k: v.copy_compatible_to_dims_raw(v.dims).detach().cpu().numpy()
        for k, v in _run(
            lambda name, arr, d: Tensor(name, dims=d, dtype=arr.dtype.name, raw_tensor=torch.from_numpy(arr))
        ).items()
    }

    _rf_jax()
    got = {k: numpy.asarray(v.copy_compatible_to_dims_raw(v.dims)) for k, v in _run(_make).items()}

    for key in sorted(ref):
        numpy.testing.assert_allclose(got[key], ref[key], rtol=1e-6, err_msg=f"top_k {key} differs")


def test_edit_distance_vs_reference():
    """
    edit_distance, used as an error metric in RF code (nn_rf/encoder/layered.py, nn_rf/text_augment.py).
    Compared against an independent Python implementation, per sequence and at its own lengths.
    """
    _rf_jax()
    batch = Dim(4, name="batch")
    a_lens, b_lens = [5, 3, 0, 6], [4, 3, 2, 6]
    rnd = numpy.random.RandomState(89)
    a_np = rnd.randint(0, 5, size=(4, 6)).astype("int32")
    b_np = rnd.randint(0, 5, size=(4, 6)).astype("int32")
    # one pair identical, so a zero distance is covered too
    b_np[1, : b_lens[1]] = a_np[1, : a_lens[1]][: b_lens[1]]

    def _ref(x, y):
        """plain Levenshtein, the definition"""
        prev = list(range(len(y) + 1))
        for i, xi in enumerate(x, start=1):
            cur = [i]
            for j, yj in enumerate(y, start=1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (xi != yj)))
            prev = cur
        return prev[-1]

    a_time = _make_dyn_time(batch, a_lens, name="a_time")
    b_time = _make_dyn_time(batch, b_lens, name="b_time")
    vocab = Dim(5, name="vocab")
    a = Tensor("a", dims=[batch, a_time], dtype="int32", sparse_dim=vocab, raw_tensor=jnp.asarray(a_np))
    b = Tensor("b", dims=[batch, b_time], dtype="int32", sparse_dim=vocab, raw_tensor=jnp.asarray(b_np))
    got = numpy.asarray(rf.edit_distance(a, a_time, b, b_time).raw_tensor)

    expected = [_ref(list(a_np[i, : a_lens[i]]), list(b_np[i, : b_lens[i]])) for i in range(4)]
    numpy.testing.assert_array_equal(got, expected, err_msg=f"edit distances {got} vs {expected}")
    assert expected[1] == 0, "the identical pair should have distance 0"


def test_torch_checkpoint_import_parity():
    """
    The decisive cross-backend check, on the target model:
    take a PyTorch checkpoint, load its parameters into the JAX model, and the outputs must match.

    Unlike the other comparisons, nothing is copied in memory here --
    it goes through a real checkpoint file, i.e. the path a production PT baseline would take.
    """
    import tempfile
    import torch
    from returnn.torch.frontend.bridge import rf_module_to_pt_module
    from returnn.jax.checkpoint import load_torch_checkpoint, set_model_params
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )

    batch, in_dim, enc_dim = Dim(2, name="batch"), Dim(8, name="feat"), Dim(32, name="enc")
    seq_lens = [16, 11]
    x_np = numpy.random.RandomState(97).normal(size=(2, 16, 8)).astype("float32")

    def _build():
        rf.set_random_seed(31)
        return ConformerEncoder(
            in_dim,
            enc_dim,
            ff_dim=Dim(24, name="enc-ff"),
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(4, name="conv1"), Dim(4, name="conv2")],
                filter_sizes=[(3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1)],
            ),
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=2,
                conv_norm_opts={"use_mask": True},
            ),
            num_layers=1,
            dropout=0.0,
            att_dropout=0.0,
        )

    # Inference semantics, and set explicitly: without a run ctx of its own this test would inherit
    # whatever a previous test left behind, and under a train flag the encoder's input_dropout (0.1 by
    # default) is live -- two independent RNG streams, so the comparison could never hold.
    rf.init_forward_step_run_ctx()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # --- torch side: build, give the params non-trivial values, save a checkpoint as the engine does
        _no_tf32_torch()
        rf.select_backend_torch()
        model_pt = _build()
        pt_module = rf_module_to_pt_module(model_pt)
        with torch.no_grad():
            for param in pt_module.parameters():
                param.copy_(torch.randn_like(param) * 0.1)
        filename = f"{tmp_dir}/model.001.pt"
        torch.save({"model": pt_module.state_dict(), "epoch": 1, "step": 0}, filename)

        time_pt = Dim(
            Tensor("l", dims=[batch], dtype="int32", raw_tensor=torch.tensor(seq_lens, dtype=torch.int32)), name="time"
        )
        x_pt = Tensor("x", dims=[batch, time_pt, in_dim], dtype="float32", raw_tensor=torch.from_numpy(x_np))
        out_pt, spatial_pt = model_pt(x_pt, in_spatial_dim=time_pt)
        ref = out_pt.copy_compatible_to_dims_raw(out_pt.dims).detach().cpu().numpy()
        ref_lens = spatial_pt.dyn_size_ext.raw_tensor.detach().cpu().numpy()

        # --- jax side: build the same model, load the checkpoint, run
        _rf_jax()
        model_jax = _build()
        params = load_torch_checkpoint(filename)
        assert set(params) == {name for name, _ in model_jax.named_parameters()}, (
            f"parameter names differ: {sorted(set(params) ^ {n for n, _ in model_jax.named_parameters()})}"
        )
        set_model_params(model_jax, params)

        time_jax = Dim(
            Tensor("l", dims=[batch], dtype="int32", raw_tensor=jnp.asarray(seq_lens, dtype=jnp.int32)), name="time"
        )
        x_jax = _make("x", x_np, [batch, time_jax, in_dim])
        out_jax, spatial_jax = model_jax(x_jax, in_spatial_dim=time_jax)
        got = numpy.asarray(out_jax.copy_compatible_to_dims_raw(out_jax.dims))
        got_lens = numpy.asarray(spatial_jax.dyn_size_ext.raw_tensor)

    numpy.testing.assert_array_equal(got_lens, ref_lens, err_msg="subsampled seq lens differ")
    assert got.shape == ref.shape, f"{got.shape} vs {ref.shape}"
    numpy.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5, err_msg="outputs differ after checkpoint import")


def test_device():
    # device-agnostic on purpose: the default is cpu on a CPU-only run and cuda:0 on a GPU node
    _rf_jax()
    x = _make("x", numpy.zeros((2,), dtype="float32"), [Dim(2, name="d")])
    default = x.device
    assert default == "cpu" or default.startswith("cuda:"), f"unexpected device name {default!r}"
    # the RF device naming must round-trip through copy_to_device
    assert rf.copy_to_device(x, default).device == default
    if default != "cpu":
        assert numpy.asarray(rf.copy_to_device(x, "cpu").raw_tensor).shape == (2,)
