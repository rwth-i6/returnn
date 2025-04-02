"""
Lion optimizer <https://arxiv.org/abs/2302.06675>

Code adapted from https://github.com/lucidrains/lion-pytorch/,
which is adapted from https://github.com/google/automl/blob/master/lion/lion_pytorch.py.
"""

from __future__ import annotations
from typing import Optional, Tuple, Callable
import inspect
import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """
    Lion (Evolved Sign Momentum (Evo_l_ved S_i_gn M_o_me_n_tum)) optimizer <https://arxiv.org/abs/2302.06675>
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
        use_triton: Optional[bool] = None,
        decoupled_weight_decay: bool = False,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        self._init_lr = lr
        self.decoupled_wd = decoupled_weight_decay

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

        if use_triton is None:
            use_triton = bool(triton_update_fn)
        self.use_triton = use_triton

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """update step"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                beta1, beta2 = group["betas"]
                grad, lr, wd, state, decoupled_wd, init_lr = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    self.state[p],
                    self.decoupled_wd,
                    self._init_lr,
                )

                # maybe decoupled weight decay

                if decoupled_wd:
                    wd /= init_lr

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                if self.use_triton and p.is_cuda:
                    triton_update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)
                else:
                    update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss


# update functions


def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
    """
    Lion update function
    """
    # stepweight decay

    p.data.mul_(1.0 - lr * wd)

    # weight update

    update = exp_avg.clone().mul_(beta1).add(grad, alpha=1.0 - beta1).sign_()
    p.add_(update, alpha=-lr)

    # decay the momentum running average coefficient

    exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)


try:
    # noinspection PyPackageRequirements
    import triton

    # noinspection PyPackageRequirements
    import triton.language as tl
except ImportError as e:
    triton = None
    tl = None


# restore_value is not available in older versions of triton
if triton and "restore_value" in inspect.signature(triton.autotune).parameters:
    # triton cuda kernel

    # noinspection PyPep8Naming,PyArgumentList
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
            triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        ],
        key=["n_elements"],
        restore_value=["p_ptr", "exp_avg_ptr"],
    )
    @triton.jit
    def _triton_update_fn_kernel(
        p_ptr,
        grad_ptr,
        exp_avg_ptr,
        lr,
        wd,
        beta1,
        beta2,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)

        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        mask = offsets < n_elements

        # offsetted pointers

        offset_p_ptr = p_ptr + offsets
        offset_grad_ptr = grad_ptr + offsets
        offset_exp_avg_ptr = exp_avg_ptr + offsets

        # load

        p = tl.load(offset_p_ptr, mask=mask)
        grad = tl.load(offset_grad_ptr, mask=mask)
        exp_avg = tl.load(offset_exp_avg_ptr, mask=mask)

        # stepweight decay

        p = p * (1 - lr * wd)

        # diff between momentum running average and grad

        diff = exp_avg - grad

        # weight update

        update = diff * beta1 + grad

        # torch.sign

        can_update = update != 0
        update_sign = tl.where(update > 0, -lr, lr)

        p = p + update_sign * can_update

        # decay the momentum running average coefficient

        exp_avg = diff * beta2 + grad

        # store new params and momentum running average coefficient

        tl.store(offset_p_ptr, p, mask=mask)
        tl.store(offset_exp_avg_ptr, exp_avg, mask=mask)

    def triton_update_fn(
        p: torch.Tensor, grad: torch.Tensor, exp_avg: torch.Tensor, lr: float, wd: float, beta1: float, beta2: float
    ):
        """
        Lion update function using triton kernel
        """
        assert all([t.is_cuda for t in (p, grad, exp_avg)])
        n_elements = p.numel()

        def _grid(meta):
            return tuple((triton.cdiv(n_elements, meta["BLOCK_SIZE"]),))

        _triton_update_fn_kernel[_grid](p, grad, exp_avg, lr, wd, beta1, beta2, n_elements)

else:
    triton_update_fn = None
