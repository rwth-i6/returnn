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


def test_device():
    _rf_jax()
    x = _make("x", numpy.zeros((2,), dtype="float32"), [Dim(2, name="d")])
    assert x.device == "cpu"
    assert rf.copy_to_device(x, "cpu").device == "cpu"
