"""
Helpers to debug nan/inf values in tensors.
E.g., you get nan/inf values in the loss, and you want to know where it comes from.
There could be multiple potential sources:

- The parameters are already broken (nan/inf).
    Then some prev step caused this.
    For this, we might want to add another option which performs a check before we update params,
    so that updating params will never break them unnoticed.
- The gradients are broken (nan/inf).
    There are some PyTorch utilities to check this.
    This is currently not the focus here.
- Some part of the (forward) computation results in nan/inf.
    Currently, this is the focus here.
    We want to know where this happens.

We could run the forward pass again in different modes:

- Python tracing, and inspecting all local variables which are tensors.
    (Probably slow).
- PyTorch JIT tracing to compute the loss. This will give us the computation graph.
    We can run this computation graph again and inspect all the intermediate values,
    and then see where the nan/inf values come from.
- PyTorch profiling.

Note, one problem is non-determinism in the computation via e.g. dropout.
So the method might not be totally reliable.
Also, there might be inf/nan values which are ok, expected, and not a problem
(e.g. masking the logits for attention).
So we don't stop on the first occurrence but just report all of them.
"""

from __future__ import annotations

import sys
from typing import Optional, Union, Callable, TextIO
from io import TextIOBase
import traceback
from types import FrameType
import torch
import tree

# noinspection PyProtectedMember
from torch.utils._python_dispatch import TorchDispatchMode

# noinspection PyProtectedMember
from torch._dispatch.python import no_python_dispatcher


def debug_inf_nan(
    func: Callable[[], Optional[torch.Tensor]],
    *,
    with_grad: bool = False,
    report_every_op_call: bool = True,
    stop_reporting_after_first_inf_nan: bool = True,
    file: Optional[Union[TextIO, TextIOBase]] = None,
):
    """
    Debug the function.

    :param func: will be called like func(). if `with_grad`, we expect some loss tensor as return,
        and we will call `loss = func(); loss.backward()`.
    :param with_grad: whether to compute and debug gradients for inf/nan.
    :param report_every_op_call: whether to report every op call.
    :param stop_reporting_after_first_inf_nan: whether to stop reporting after the first inf/nan.
    :param file: where to write the output to. Default is stdout.
    """

    if file is None:
        file = sys.stdout

    # noinspection PyUnresolvedReferences,PyProtectedMember
    cur_frame: FrameType = sys._getframe()
    trace_ops = _TraceOps(
        root_frame=cur_frame,
        file=file,
        report_every_op_call=report_every_op_call,
        stop_reporting_after_first_inf_nan=stop_reporting_after_first_inf_nan,
    )

    if with_grad:
        with torch.autograd.detect_anomaly():
            with trace_ops:  # currently only for forward (but we might want to trace the backward too)
                loss = func()
            file.flush()  # the backward detect_anomaly might screw up the output otherwise
            try:
                loss.backward()
            except RuntimeError as exc:
                print(f"Caught RuntimeError in backward: {exc}", file=file)

    else:  # without grad
        with trace_ops:
            func()


# For efficiency, and to be less spammy
_TraceFuncNameBlacklist = {
    "aten::empty.memory_format",
    "aten::zeros_like",
    "aten::ones_like",
    "aten::full",
    "aten::scalar_tensor",  # when we deliberately create a scalar inf tensor
    "aten::_local_scalar_dense",
    "aten::where.self",  # when we intentionally mask with inf
    "aten::detach",
    "aten::_to_copy",
    "aten::clone",
    "aten::stack",
    "aten::view",
    "aten::_unsafe_view",
    "aten::permute",
    "aten::t",
    "aten::split_with_sizes",
    "aten::slice.Tensor",
    "aten::select.int",
    "aten::max_pool2d_with_indices",
}


class _TraceOps(TorchDispatchMode):
    def __init__(
        self,
        *,
        root_frame: FrameType,
        file: Union[TextIO, TextIOBase],
        report_every_op_call: bool = True,
        stop_reporting_after_first_inf_nan: bool = True,
    ):
        super().__init__()
        self.root_frame = root_frame
        self.file = file
        self.enabled = True
        self.report_every_op_call = report_every_op_call
        self.stop_reporting_after_first_inf_nan = stop_reporting_after_first_inf_nan

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not self.enabled or func.name() in _TraceFuncNameBlacklist:
            return func(*args, **kwargs)
        if self.report_every_op_call:
            print(f"--- op {func.name()}", file=self.file)
        out = func(*args, **kwargs)
        for out_ in tree.flatten(out):
            if isinstance(out_, torch.Tensor):
                with no_python_dispatcher():
                    got_nan_inf_t = torch.stack([torch.isnan(out_).any(), torch.isinf(out_).any()]).cpu()
                    got_nan = got_nan_inf_t[0].item()
                    got_inf = got_nan_inf_t[1].item()
                    if got_nan or got_inf:
                        s = "/".join([s_ for s_, b in [("nan", got_nan), ("inf", got_inf)] if b])
                        print(f"--> {s} in {func}: {out_}", file=self.file)
                        traceback.print_list(
                            _extract_stack_up_to(skip_top_num_frames=1, root_frame=self.root_frame), file=self.file
                        )
                        if self.stop_reporting_after_first_inf_nan:
                            self.enabled = False
        return out


def _walk_stack_up_to(f: FrameType, *, root_frame: FrameType):
    while f is not None and f != root_frame:
        yield f, f.f_lineno
        f = f.f_back


def _extract_stack_up_to(*, skip_top_num_frames: int = 0, root_frame: FrameType):
    # noinspection PyUnresolvedReferences,PyProtectedMember
    frame = sys._getframe()
    skip_top_num_frames += 1  # skip this function
    for _ in range(skip_top_num_frames):
        frame = frame.f_back
    stack = traceback.StackSummary.extract(_walk_stack_up_to(frame, root_frame=root_frame))
    stack.reverse()
    return stack
