"""
Run context

We can either be in param-init stage,
or in the main training loop,
or forwarding loop.
"""

from __future__ import annotations
from typing import Optional, Union, Any, Sequence, Dict
import logging
from dataclasses import dataclass
from returnn.tensor import Tensor, Dim, TensorDict
import returnn.frontend as rf
from . import _backend


__all__ = ["RunCtx", "Loss", "get_run_ctx", "init_train_step_run_ctx", "init_forward_step_run_ctx"]


_run_ctx = None  # type: Optional[RunCtx]
_init_run_ctx = None  # type: Optional[RunCtx]


def reset_run_ctx():
    """
    If we get out of a train step or forward step.
    """
    global _run_ctx
    _run_ctx = None


def init_train_step_run_ctx(*, train_flag: Union[bool, Tensor]):
    """
    Call this at the beginning of a new train step.
    """
    global _run_ctx
    _run_ctx = RunCtx(stage="train_step", train_flag=train_flag)


def init_forward_step_run_ctx(expected_outputs: Optional[TensorDict] = None):
    """
    Call this at the beginning of a new forward step.
    """
    global _run_ctx
    _run_ctx = RunCtx(stage="forward_step", expected_outputs=expected_outputs)


def get_run_ctx() -> RunCtx:
    """
    :return: current run context, see :class:`RunCtx`
    """
    global _run_ctx, _init_run_ctx
    if _run_ctx is None:
        if _init_run_ctx is None:
            _init_run_ctx = RunCtx(stage="init")
        return _init_run_ctx
    return _run_ctx


class RunCtx:
    """
    We can either be in param-init stage,
    or in the main training (or eval) loop,
    or forwarding loop (doing recog, beam search, dumping whatever, ...).

    In training/eval, we expect that some loss is being defined via mark_as_loss().
    In forwarding, we expect that some output is being defined via mark_as_output().
    """

    def __init__(
        self, *, stage: str, train_flag: Union[bool, Tensor] = False, expected_outputs: Optional[TensorDict] = None
    ):
        """
        :param stage:
            - "init"
            - "train_step", also for eval, for mark_as_loss and get_total_loss
            - "forward_step", for mark_as_output
        """
        self._stage = stage
        self._train_flag = train_flag
        self.losses = {}  # type: Dict[str, Loss]
        self.outputs = TensorDict()
        self.expected_outputs = expected_outputs

    @property
    def stage(self) -> str:
        """
        :return: "init", "train_step", "forward_step"
        """
        return self._stage

    @property
    def train_flag(self) -> Union[bool, Tensor]:
        """
        :return: whether we are in training mode, i.e. the model is updated,
            and we are supposed to use dropout and similar mechanisms.
            In a graph-based backend, this can be dynamic.
        """
        return self._train_flag

    def mark_as_loss(
        self,
        loss: Union[Tensor, Any],
        name: str,
        *,
        scale: float = 1.0,
        as_error: bool = False,
        use_normalized_loss: bool = False,
        use_flatten_frames: bool = True,
        custom_inv_norm_factor: Optional[Tensor] = None,
    ) -> None:
        """
        Mark the given loss tensor as a loss.
        This has the effect that it is specially handled by RETURNN.
        Specifically, the optimizer can use it in training,
        and it is used for reporting per batch or per epoch,
        and for learning rate scheduling.

        This currently uses :class:`AsIsLoss` in RETURNN
        but this is an implementation detail and might change.

        :param loss: it should not be reduced - this will be done later, and this is important to properly
            do accumulation taking different seq lengths into account.
            A :class:`Tensor` is usually expected, but a raw tensor is also possible.
        :param name: name of the loss. this name is used for reporting by RETURNN, and also for LR scheduling.
        :param scale: scale the loss by this factor for the training optimizer
          (but not for any reporting). setting to 0.0 has the effect that this loss is not used by the optimizer.
        :param as_error: if True, this loss is reported as an error instead of a loss,
          and not used by the training optimizer.
          This is by convention sth like the frame-error or edit-distance, and usually not differentiable anyway.
        :param bool use_flatten_frames: If True, will use :func:`returnn.tf.util.basic.flatten_with_seq_len_mask`,
          i.e. a "packed" sequence with the padded frames removed, and accumulates over that.
          This can be more efficient, also because it will further optimize incoming computations
          and e.g. skip softmax computations right before on the padded frames.
          This can also avoid issues with inf/nan in some cases.
          If False, it will mask the loss to 0 in the padded frames and accumulate over that.
          Typically, setting this to True (default) is both more efficient and better.
        :param bool use_normalized_loss: the loss used in optimization will be normalized.
          E.g. if the overall normalization is sum(loss)/sum(num_frames), this is also what the optimizer will use,
          otherwise the optimizer will just use sum(loss).
        :param custom_inv_norm_factor:
          The standard inv norm factor is sum(target_seq_len) if the target has a time-axis,
          or sum(output_seq_len) if there is no target and the output has a time-axis,
          or 1 otherwise. (See :func:`Loss.init` for details.)
          This is used for proper normalization of accumulated loss/error per epoch
          and also proper normalization per batch for reporting,
          no matter if use_normalized_loss is True or False.
          If you want to change this norm factor, you can set this.
          Basically, for all reporting, it uses sum(loss) / sum(custom_inv_norm_factor).
        """
        assert self.stage == "train_step"
        if not isinstance(loss, Tensor):
            assert isinstance(loss, _backend.global_backend.RawTensorType)
            loss = rf.convert_to_tensor(loss)
        assert name not in self.losses
        self.losses[name] = Loss(
            loss=loss,
            name=name,
            scale=scale,
            as_error=as_error,
            use_normalized_loss=use_normalized_loss,
            use_flatten_frames=use_flatten_frames,
            custom_inv_norm_factor=custom_inv_norm_factor,
        )

    def mark_as_output(self, tensor: Union[Tensor, Any], name: str, *, dims: Optional[Sequence[int]] = None) -> None:
        """
        Mark this as an output.
        This has the effect that RETURNN will in any case construct the corresponding layer.
        Also see :func:`mark_as_default_output`.

        This is intended mostly for forwarding, or exporting the model (TF graph, TFLite, ONNX, etc).
        You must specify a shape to have the output shape (order of dims) well-defined
        (if not specified, we check if some defaults are possible, like BTF, or BF).

        :param tensor:
        :param name:
        :param dims: this specifies the order of the dims of the output, such that it is well-defined
            for some external application.
            If not specified, we try to infer BTF or BF as default, if that works, otherwise it will be an error.
        """
        assert self.stage == "forward_step"

        if self.expected_outputs is not None:
            assert (
                name in self.expected_outputs.data
            ), f"mark_as_output: unexpected output {name!r}, we expect outputs: {self.expected_outputs}"
        expected_output = self.expected_outputs.data[name] if self.expected_outputs else None
        if dims is None and expected_output:
            dims = expected_output.dims
        if dims is not None and expected_output:
            assert (
                expected_output.dims == dims
            ), f"mark_as_output: {name!r} dims mismatch from expected output, given {dims}, expected {expected_output}"

        if not isinstance(tensor, Tensor):
            tensor = _output_tensor_from_raw(tensor, dims=dims, name=name)
            # In case it was not specified, just accept whatever order we got.
            dims = tensor.dims

        if dims is None:
            # We try some reasonable defaults, specifically: BTF or BF.
            dims = _default_dim_order(tensor)
        assert set(dims) == set(tensor.dims), f"mark_as_output: tensor {tensor} does not have the dims {dims}"
        tensor = tensor.copy_transpose(dims, allow_int=False)
        tensor = tensor.copy(name=name)
        assert name not in self.outputs.data
        self.outputs.data[name] = tensor

        if expected_output:
            # Perform sanity checks using the expected output.
            # The expected output usually comes from `model_outputs` from the user config.
            # The dimensions of `expected_output` and `tensor` should match,
            # but we can't directly check for expected_output.dims == tensor.dims
            # because not all dims can be known in advance in `expected_output`,
            # e.g. dynamic dims.
            # Thus, we allow undefined dims in the expected output,
            # and ignore them when checking for equality.
            # The most important thing for the user is to define what dims are dynamic and what dims are static.
            # This is also necessary for ONNX export.
            assert len(expected_output.dims) == len(tensor.dims), (
                f"mark_as_output: lengths of expected output {expected_output.dims}"
                f" and actual output {tensor.dims} don't match."
            )
            for expected_dim, actual_dim in zip(expected_output.dims, tensor.dims):
                expected_dim: Dim
                actual_dim: Dim
                if not expected_dim.is_dim_known():
                    assert actual_dim.is_dynamic(), (
                        f"mark_as_output: expected dim {expected_dim} doesn't have a known value."
                        f" Matching actual dim assumed to be dynamic, but got non-dynamic dim {actual_dim}."
                    )
                elif expected_dim.is_dynamic():
                    assert actual_dim.is_dynamic(), (
                        f"mark_as_output: expected dim {expected_dim} is dynamic."
                        f" Matching actual dim assumed to be dynamic, but got non-dynamic dim {actual_dim}."
                    )
                elif expected_dim.is_static():
                    assert expected_dim.is_static() and actual_dim.dimension == expected_dim.dimension, (
                        f"mark_as_output: expected dim {expected_dim} is static."
                        f" Matching actual dim assumed to be the same static dim value, but got {actual_dim}."
                    )
                else:
                    assert False, f"mark_as_output: unexpected expected dim {expected_dim}."
            assert expected_output.dtype == tensor.dtype, (
                f"mark_as_output: {name!r} dtype mismatch from expected output,"
                f" given {tensor.dtype}, expected {expected_output.dtype}"
            )
            assert expected_output.sparse_dim == tensor.sparse_dim, (
                f"mark_as_output: {name!r} sparse_dim mismatch from expected output,"
                f" given {tensor.sparse_dim}, expected {expected_output.sparse_dim}"
            )

    def mark_as_default_output(self, tensor: Union[Tensor, Any], *, shape: Optional[Sequence[Dim]] = None) -> None:
        """
        Calls mark_as_output(tensor, "output", shape=shape).

        Mark this as the default output.
        See :func:`Frontend.mark_as_default_output` for more details.

        :param tensor:
        :param shape:
        """
        self.mark_as_output(tensor, "output", dims=shape)

    def check_outputs_complete(self):
        """
        If expected outputs are given, check that all expected outputs are present.
        """
        if self.expected_outputs:
            assert set(self.expected_outputs.data.keys()) == set(self.outputs.data.keys()), (
                f"check_outputs_complete: expected outputs {self.expected_outputs} do not match"
                f" actual outputs {self.outputs}"
            )
            # We don't need to check the dims, dtype, etc, as this is already done in mark_as_output.

    def total_loss(self) -> Union[Tensor, float]:
        """
        :return: total loss, as it is used for backpropagation
        """
        assert self.stage == "train_step"
        assert self.losses, "call rf.get_run_ctx().mark_as_loss(...)"
        loss = 0.0
        for name, loss_obj in self.losses.items():
            if loss_obj.scale == 0.0 or loss_obj.as_error:
                continue
            loss += loss_obj.get_scaled_reduced_loss()
        return loss


@dataclass
class Loss:
    """
    Loss via :func:`RunCtx.mark_as_loss`.

    We collect all relevant information here.
    """

    loss: Tensor
    name: str

    scale: float = 1.0
    as_error: bool = False
    use_normalized_loss: bool = False  # for the gradient / total loss
    use_flatten_frames: bool = True
    custom_inv_norm_factor: Optional[Tensor] = None

    _summed_loss_cached: Optional[Tensor] = None
    _mean_loss_cached: Optional[Tensor] = None

    def get_summed_loss(self) -> Tensor:
        """
        :return: sum of loss (scalar)
        """
        if not self.loss.dims:
            return self.loss
        if self._summed_loss_cached is not None:
            return self._summed_loss_cached
        if self._mean_loss_cached is not None:
            return self._mean_loss_cached / self.get_inv_norm_factor()
        self._summed_loss_cached = rf.reduce_sum(self.loss, axis=self.loss.dims)
        return self._summed_loss_cached

    def get_mean_loss(self) -> Tensor:
        """
        :return: sum of loss (scalar)
        """
        if self._mean_loss_cached is not None:
            return self._mean_loss_cached
        if self.custom_inv_norm_factor:
            loss = self.get_summed_loss()
            loss /= rf.cast(self.custom_inv_norm_factor, dtype=loss.dtype)
            return loss
        if not self.loss.dims:
            return self.loss
        self._mean_loss_cached = rf.reduce_mean(self.loss, axis=self.loss.dims)
        return self._mean_loss_cached

    def get_inv_norm_factor(self) -> Union[int, Tensor]:
        """
        :return: inverse norm factor (scalar)
        """
        if self.custom_inv_norm_factor:
            return self.custom_inv_norm_factor
        return self.loss.num_elements()

    def get_scaled_reduced_loss(self) -> Tensor:
        """
        :return: scaled reduced loss (scalar), as it is supposed to be used for calculating the train gradient
        """
        if self.use_normalized_loss:
            loss = self.get_mean_loss()
        else:
            loss = self.get_summed_loss()
        return loss * self.scale


def _default_dim_order(tensor: Tensor) -> Sequence[Dim]:
    """
    See if some reasonable default dim order like BTF or BF is possible.

    :param tensor:
    :return:
    """
    rem_dims = list(tensor.dims)
    dims = []
    if tensor.have_batch_axis():
        rem_dims.remove(tensor.get_batch_dim_tag())
        dims.append(tensor.get_batch_dim_tag())
    if tensor.have_time_axis():
        rem_dims.remove(tensor.get_time_dim_tag())
        dims.append(tensor.get_time_dim_tag())
    dyn_dims = [d for d in rem_dims if d.is_dynamic()]
    if len(dyn_dims) > 1:
        raise Exception(
            f"Cannot infer order of dims automatically for output {tensor}. Please specify `dims` explicitly."
        )
    elif len(dyn_dims) == 1:
        rem_dims.remove(dyn_dims[0])
        dims.append(dyn_dims[0])
    if len(rem_dims) > 1:
        raise Exception(
            f"Cannot infer order of dims automatically for output {tensor}. Please specify `dims` explicitly."
        )
    elif len(rem_dims) == 1:
        dims.append(rem_dims[0])
    return dims


def _output_tensor_from_raw(raw_tensor, *, dims: Sequence[Dim], name: str) -> Tensor:
    assert isinstance(raw_tensor, _backend.global_backend.RawTensorType)
    tensor = rf.convert_to_tensor(raw_tensor, dims=dims)
    # This is called when the user passed some raw tensor directly to mark_as_output.
    # So e.g. in case of pure PyTorch code.
    # In this case, it is common that the user does not explicitly specify the dyn sizes of the dim tags.
    # We accept this, but for ONNX export and maybe other cases,
    # we automatically fill in the dyn sizes from the raw tensor.
    # In case the dim tag has scalar size, this is non-ambiguous,
    # however, otherwise, it is ambiguous, so we print a warning that we always use the highest dim size.
    for axis, dim in enumerate(tensor.dims):
        if dim.dyn_size_ext and dim.dyn_size_ext.raw_tensor is None:
            dim_value_raw = _backend.global_backend.get_shape_tuple_raw(raw_tensor)[axis]
            dim_value = rf.convert_to_tensor(dim_value_raw, dims=())
            dim_value = rf.cast(dim_value, dtype=dim.dyn_size_ext.dtype)
            if dim.dyn_size_ext.dims:
                dyn_size = rf.full(dims=dim.dyn_size_ext.dims, fill_value=dim_value)
                logging.warning(
                    f"Output {name!r} {tensor}: Cannot infer dynamic size for dim {dim}. "
                    f"Using {dyn_size} as fallback."
                )
            else:
                dyn_size = dim_value
            dim.dyn_size_ext.raw_tensor = dyn_size.raw_tensor
    return tensor
