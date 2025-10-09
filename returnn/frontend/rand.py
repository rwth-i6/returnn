"""
Random number generator utilities.

Note on the global seed:

In the TF engine, we have this in ``_init_network``:

.. code-block:: python

    tf_random_seed = 42
    net_random_seed = epoch
    if self.config.opt_typed_value("random_seed", None):
        seed = self.config.int("random_seed", None)
        net_random_seed = (epoch * 3 + seed * 5 + 7) % (2**31)
        tf_random_seed = (net_random_seed * 2 + 3) % (2**31)
    tf_compat.v1.set_random_seed(tf_random_seed)

``_init_network`` is only called in the beginning
and then only when needed (e.g. different model due to pretrain).

The ``net_random_seed`` is used inside the :class:`TFNetwork` for:

.. code-block:: python

    self.random = numpy.random.RandomState(rnd_seed)

This in turns is used for param init in the TF layers, like:

.. code-block:: python

    fwd_weights_initializer = get_initializer(
        forward_weights_init, seed=self.network.random.randint(2**31), eval_local_ns={"layer": self}
    )

It is also used by the :class:`RandomLayer` when ``static=True``:

.. code-block:: python

    # static is True
    if seed is None:
        seed = self.network.random.randint(2**31, size=[32], dtype="uint32")

So, it means, with ``static=True``, the random numbers are always the same in each execution step.
To reflect that in an eager-based backend,
we need to reset the static-random-state in the beginning of each step.

"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Dict
import numpy
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from ._backend import global_backend as _global_backend


__all__ = [
    "set_random_seed",
    "get_random_state",
    "set_random_state",
    "reset_step_random_state",
    "get_static_step_based_seed",
    "random",
    "random_uniform",
    "random_normal",
    "random_truncated_normal",
    "random_choice_without_replacement",
    "random_choice_with_replacement",
]


def set_random_seed(seed: int):
    """
    This initializes the random state of the backend
    and also the step-based random state
    (see :func:`get_static_step_based_seed`, only used when ``static=True`` in :func:`random`).

    Call this at the beginning of the program
    (after the RF backend was selected),
    or when the model and computation graph is supposed to be reinitialized
    or at the beginning of each epoch.

    :param seed: should depend on epoch or step
    """
    _global_backend.set_random_seed(seed)
    global _step_rnd_seed
    _step_rnd_seed = (seed * 5393 + 1187) % (2**31)
    reset_step_random_state()


def get_random_state() -> Dict[str, bytes]:
    """
    :return: current random state, for serialization, to be able to restore it later
    """
    return _global_backend.get_random_state()


def set_random_state(state: Dict[str, bytes]):
    """
    Recovers the random state.

    There are many potential cases where we cannot recover the state
    (e.g. different backend version, different hardware, ...),
    In this case, a run without interruption is not the same as a run with interruption.

    We still assume that :func:`set_random_seed` was called before in any case.

    :param state: as returned by :func:`get_random_state`
    """
    # Always call this first.
    _global_backend.set_random_state(state)


_step_rnd_seed = 42
_step_rnd = numpy.random.RandomState(_step_rnd_seed)


def reset_step_random_state():
    """
    When ``static=True`` is used in :func:`random`,
    the random state is reset to the beginning of the step.
    So this should be called in the beginning of each step.
    Also see the module docstring.
    """
    _step_rnd.seed(_step_rnd_seed)


def get_static_step_based_seed(*, size=None) -> Union[int, numpy.ndarray]:
    """
    This is intended as a static seed for :func:`random` when ``static=True`` is used.

    :return: from the static step-based random state, get a seed
    """
    return _step_rnd.randint(2**31, size=size)


def random(
    dims: Sequence[Dim],
    *,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
    distribution: str,
    mean: Optional[Union[int, float, Tensor]] = None,
    stddev: Optional[Union[int, float, Tensor]] = None,
    bound: Optional[Union[int, float, Tensor]] = None,
    minval: Optional[Union[int, float, Tensor]] = None,
    maxval: Optional[Union[int, float, Tensor]] = None,
    seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
    algorithm: Optional[str] = None,
    explicit_state: Optional[Tensor] = None,
    auto_update_state: Optional[bool] = None,
    static: Optional[bool] = None,
    out: Optional[Tensor] = None,
) -> Tensor:
    """
    Generates random numbers from uniform or normal or truncated normal distribution.

    There will be no gradients to mean, stddev, bound, minval, maxval!

    In case of TensorFlow:
    This uses the TensorFlow stateless random ops internally, i.e. all the state handling is explicit.
    The state var can be explicitly provided and initialized via :class:`RandomStateInitLayer`,
    or when not provided it will be automatically created.

    There are two possible distinct use cases:

    - For any randomness in the model, e.g. dropout. So each ``session.run`` step will produce a new random number
      and advance the random state.
    - To initialize parameters via the config, using :class:`VariableLayer` with the ``init_by_layer`` option.
      This will only be called once when initializing the parameters.
      For this use case, we do not want to keep a random state var.
      You can just pass ``static=False``.
      Alternatively you could also pass the output of a :class:`RandomStateInitLayer` as ``state``.

    :param dims:
    :param dtype:
    :param device:
    :param sparse_dim:
    :param feature_dim:
    :param str distribution: "uniform", "normal" or "truncated_normal"
    :param int|float|Tensor|None mean:
    :param int|float|Tensor|None stddev:
    :param int|float|Tensor|None bound: for uniform, defining the range [-bound, bound)
    :param int|float|Tensor|None minval: for uniform, inclusive
    :param int|float|Tensor|None maxval: for uniform, exclusive
    :param int|list[int]|numpy.ndarray|None seed:
        Only for the case ``static=True``.
        If not given, uses self.network.random.randint,
        i.e. then it is controlled by the global seed setting, and every layer would get its own seed.
        If you specify it explicitly, make sure every :class:`RandomLayer` uses a different seed,
        otherwise you would get the same random numbers everywhere.
    :param str|tf.random.Algorithm|None algorithm: see :class:`RandomStateInitLayer`
    :param Tensor|None explicit_state: You can pass the state explicitly here.
        If not given, will be created automatically, and updated automatically.
        You could pass a :class:`VariableLayer` with initial value via :class:`RandomStateInitLayer`,
        or directly a :class:`RandomStateInitLayer`.
        If auto_update_state is True, it must be a variable,
        and every time a new random number is created, this variable is updated.
        Otherwise (default), it will not be updated automatically.
    :param bool|None auto_update_state: only used when you pass an explicit state
    :param bool|None static: if no state at all should be used. it just relies on the seed then.
    :param out: if given, will directly write into it, if possible by backend
    :return: random values
    """
    if explicit_state is None:
        if static is None:
            static = False
        assert isinstance(static, bool)
        if static:
            if seed is None:
                seed = get_static_step_based_seed()
    if dtype is None:
        if sparse_dim:
            dtype = rf.get_default_array_index_dtype()
        else:
            dtype = rf.get_default_float_dtype()
    return _global_backend.random(
        dims=dims,
        dtype=dtype,
        device=device,
        sparse_dim=sparse_dim,
        feature_dim=feature_dim,
        distribution=distribution,
        mean=mean,
        stddev=stddev,
        bound=bound,
        minval=minval,
        maxval=maxval,
        seed=seed,
        algorithm=algorithm,
        explicit_state=explicit_state,
        auto_update_state=auto_update_state,
        static=static,
        out=out,
    )


def random_uniform(
    dims: Sequence[Dim],
    *,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
    minval: Union[int, float, Tensor] = 0,
    maxval: Union[int, float, Tensor] = 1,
    seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
    algorithm: Optional[str] = None,
    explicit_state: Optional[Tensor] = None,
    auto_update_state: Optional[bool] = None,
    static: Optional[bool] = None,
    out: Optional[Tensor] = None,
):
    """
    See :func:`random`. :func:`random` with ``distribution="uniform"``.
    ``maxval`` is exclusive.
    """
    return random(
        dims=dims,
        dtype=dtype,
        device=device,
        sparse_dim=sparse_dim,
        feature_dim=feature_dim,
        distribution="uniform",
        minval=minval,
        maxval=maxval,
        seed=seed,
        algorithm=algorithm,
        explicit_state=explicit_state,
        auto_update_state=auto_update_state,
        static=static,
        out=out,
    )


def random_normal(
    dims: Sequence[Dim],
    *,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
    mean: Optional[Union[int, float, Tensor]] = 0.0,
    stddev: Optional[Union[int, float, Tensor]] = 1.0,
    seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
    algorithm: Optional[str] = None,
    explicit_state: Optional[Tensor] = None,
    auto_update_state: Optional[bool] = None,
    static: Optional[bool] = None,
    out: Optional[Tensor] = None,
):
    """
    See :func:`random`. :func:`random` with ``distribution="normal"``.
    """
    return random(
        dims=dims,
        dtype=dtype,
        device=device,
        sparse_dim=sparse_dim,
        feature_dim=feature_dim,
        distribution="normal",
        mean=mean,
        stddev=stddev,
        seed=seed,
        algorithm=algorithm,
        explicit_state=explicit_state,
        auto_update_state=auto_update_state,
        static=static,
        out=out,
    )


def random_truncated_normal(
    dims: Sequence[Dim],
    *,
    dtype: Optional[str] = None,
    device: Optional[str] = None,
    sparse_dim: Optional[Dim] = None,
    feature_dim: Optional[Dim] = None,
    mean: Optional[Union[int, float, Tensor]] = 0.0,
    stddev: Optional[Union[int, float, Tensor]] = 1.0,
    minval: Union[int, float, Tensor] = None,
    maxval: Union[int, float, Tensor] = None,
    seed: Optional[Union[int, Sequence[int], numpy.ndarray]] = None,
    algorithm: Optional[str] = None,
    explicit_state: Optional[Tensor] = None,
    auto_update_state: Optional[bool] = None,
    static: Optional[bool] = None,
    out: Optional[Tensor] = None,
):
    """
    See :func:`random`. :func:`random` with ``distribution="truncated_normal"``.
    """
    return random(
        dims=dims,
        dtype=dtype,
        device=device,
        sparse_dim=sparse_dim,
        feature_dim=feature_dim,
        distribution="truncated_normal",
        mean=mean,
        stddev=stddev,
        minval=minval,
        maxval=maxval,
        seed=seed,
        algorithm=algorithm,
        explicit_state=explicit_state,
        auto_update_state=auto_update_state,
        static=static,
        out=out,
    )


def random_choice_without_replacement(
    *,
    log_probs: Tensor,
    axis: Union[Dim, Sequence[Dim]],
    num_samples_dim: Dim,
    noise_scale: Union[float, Tensor] = 1.0,
) -> Union[Tensor, Sequence[Tensor]]:
    """
    Randomly sample without replacement.

    :param log_probs: {log_probs_dims..., axis}
    :param axis: same as in :func:`top_k`
    :param num_samples_dim: how many samples to draw
    :param noise_scale: scale the noise. with scale=0, you get :func:`top_k`.
    :return: random indices shape {log_probs_dims..., num_samples_dim} -> axis.
        if axis was a sequence, will return a sequence of tensors.
    """
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    scores_random_sample = -rf.log(
        -rf.log(random_uniform(log_probs.dims, dtype=log_probs.dtype, device=log_probs.device))
    )
    if not isinstance(noise_scale, (int, float)) or noise_scale != 1.0:
        scores_random_sample *= noise_scale
    scores = log_probs + scores_random_sample
    _, indices, _ = rf.top_k(scores, k_dim=num_samples_dim, axis=axis)
    return indices


def random_choice_with_replacement(dims: Sequence[Dim], *, probs: Tensor, axis: Dim) -> Tensor:
    """
    Randomly sample with replacement.

    :param dims: {common_dims..., new_dims...}. Defines how many samples to draw. Defines the output shape.
    :param probs: {common_dims..., axis}
    :param axis: feature axis, where to sample from
    :return: random indices shape dims -> axis.
    """
    # noinspection PyProtectedMember
    return probs._raw_backend.random_choice_with_replacement(dims=dims, probs=probs, axis=axis)
