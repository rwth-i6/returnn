"""
AMUSE optimizer <https://arxiv.org/html/2605.22432>

Code adapted from https://github.com/kjeiun/amuse/

AMUSE is a schedule-free optimizer: the same y/x/z averaging schedule
(with a warmup-coupled beta1 ramp) is wrapped around one of several inner update rules,
selected via ``update_type``:

- ``"muon"``: Muon momentum with Newton-Schulz orthogonalization,
  for matrix hidden-layer weights (ndim >= 2).
  4D parameters are flattened to a matrix before the orthogonalization,
  while 3D parameters (e.g. Conv1d kernels) are orthogonalized batch-wise
  over the leading dim, matching the upstream implementation.
  Exclude ndim > 2 params via the params_filter if they should use the fallback instead.
- ``"adamw"``: AdamW-style second-moment normalization,
  for embeddings, output heads, biases and other parameters.
- ``"sgd"``: plain gradient update on z.

One AMUSE instance applies a single update type to all its parameters.
The usual AMUSE setup (Muon on hidden matrices, AdamW-style on the rest)
is expressed with :class:`returnn.torch.optim.multi.MultiOptimizer`::

    from returnn.torch.optim.multi import make_hidden_matrix_filter

    optimizer = {
        "class": "multi",
        "optimizers": [
            {
                "class": "amuse",
                "update_type": "muon",
                "params_filter": make_hidden_matrix_filter(),
                "momentum": 0.95,
                "warmup_steps": 10_000,
            },
            {
                "class": "amuse",
                "update_type": "adamw",
                "learning_rate_multiplier": 0.015,
                "warmup_steps": 10_000,
            },
        ],
    }
    learning_rate = 0.02

The optimizer follows the schedule-free ``train()``/``eval()`` convention:
during training, the params hold the training iterate y,
``eval()`` converts them to the averaged weights x (used for evaluation and checkpoints),
and ``train()`` converts back.
The RETURNN engine calls these automatically at the train epoch boundaries
(see :func:`returnn.torch.updater.Updater.set_optimizer_training_mode`).
BatchNorm running statistics are collected under y during training,
so after switching to x the engine forwards some train batches in train mode without gradient
before evaluation and checkpoint saving, as the reference implementation does
(config ``schedule_free_batchnorm_refresh_batches``, default 50, 0 disables it).

The learning rate of each param group (as set externally, e.g. by the RETURNN LR schedule)
is used as the base learning rate, and AMUSE applies its internal warmup factor
``min(1, t / warmup_steps)`` on top.
The warmup is required by the schedule-free averaging (the z/x averaging weights
are a function of the per-step learning rate), so do not disable it.
If an external schedule already contains a warmup, the two warmups multiply.

The schedule state of each param group (``k``, ``weight_sum``, ``ckp1``, ``beta1``, ``c_warmup``)
is kept as float64 scalar tensors on the device of the params, updated in place,
and the learning rate may be a device tensor as well.
So the step reads nothing on the host and can be captured in a CUDA graph
(``torch_cuda_graph`` with ``"capture_optimizer"``),
with :func:`AMUSE.init_state` creating the per-param state before the capture.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim.optimizer import Optimizer

UPDATE_TYPES = {"muon", "adamw", "sgd"}
AUX_UPDATE_TYPES = {"adamw", "sgd"}

_DEFAULT_LR_BY_UPDATE_TYPE = {"muon": 0.02, "adamw": 3e-4, "sgd": 1.0}


@torch.no_grad()
def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Approximate the orthogonalization of ``grad`` (semi-orthogonal matrix with the same "direction")
    via a quintic Newton-Schulz iteration, computed in bfloat16.

    :param grad: matrix of shape [..., m, n]
    :param steps: number of Newton-Schulz iterations
    :return: orthogonalized matrix, same shape as ``grad``
    """
    assert grad.ndim >= 2
    a, b, c = 3.4445, -4.7750, 2.0315

    x = grad.bfloat16()
    transposed = False
    if grad.size(-2) > grad.size(-1):
        x = x.mT
        transposed = True

    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        gram = x @ x.mT
        poly = b * gram + c * (gram @ gram)
        x = a * x + poly @ x

    if transposed:
        x = x.mT
    return x


@torch.no_grad()
def muon_update(
    grad: torch.Tensor,
    momentum: torch.Tensor,
    beta: float = 0.95,
    aux_update_type: str = "adamw",
    nesterov: bool = True,
) -> torch.Tensor:
    """
    Compute the Muon update direction: momentum, Newton-Schulz orthogonalization, scaling.

    :param grad: gradient
    :param momentum: momentum buffer, updated inplace
    :param beta: momentum factor
    :param aux_update_type: how the auxiliary (non-Muon) params are trained, "adamw" or "sgd".
        This selects the scaling of the orthogonalized update.
    :param nesterov: whether to use Nesterov momentum
    :return: update direction (to be applied with the negative learning rate)
    """
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum

    if update.ndim == 4:
        update = update.reshape(len(update), -1)

    update = zeropower_via_newtonschulz5(update)

    if aux_update_type == "adamw":
        # Scaling used in the AdamW-aux AMUSE setting.
        # Based on the last two dims, i.e. per orthogonalized matrix (3D params are batches of matrices).
        update *= 0.2 * max(update.size(-2), update.size(-1)) ** 0.5
    elif aux_update_type == "sgd":
        # Muon default scaling used when auxiliary layers are trained by SGD.
        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    else:
        raise ValueError(f"Invalid AMUSE aux_update_type: {aux_update_type}. Expected one of {{'adamw', 'sgd'}}.")

    return update


class AMUSE(Optimizer):
    """
    AMUSE optimizer, one update type per instance (see the module docstring).

    State convention:

    - p stores y while training.
    - ``eval()`` converts y -> x using the current beta1.
    - ``train()`` converts x -> y using the current beta1.
    - ``state["z"]`` stores the anchor z.

    Hyperparameters:

    - beta1: initial y/x interpolation. During warmup beta1 is constant.
    - rho: controls how quickly beta1 approaches 1 after warmup.
      Higher rho pushes beta1 toward 1 faster, so y moves closer to x
      faster. Lower rho keeps y farther from x for longer.
    - r: polynomial power for the z/x averaging weights.
    - weight_decay: decoupled decay. Applied to z for the Muon and SGD update types,
      and added to the update for the AdamW update type.
      The RETURNN updater applies its default parameter-group split for it
      (no decay on biases and blacklisted modules).
    - weight_decay_at_y: optional decay applied while p is still y.
    """

    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        *,
        update_type: str = "adamw",
        momentum: float = 0.95,
        aux_update_type: str = "adamw",
        beta2: float = 0.999,
        eps: float = 1e-10,
        weight_decay: float = 0.0,
        weight_decay_at_y: float = 0.0,
        beta1: float = 0.9,
        weight_lr_power: float = 2.0,
        warmup_steps: int = 0,
        rho: float = 1.0,
        r: float = 0.0,
    ):
        """
        :param params: params or param groups
        :param lr: base learning rate. In RETURNN, this is set and scheduled externally
            (``learning_rate`` config option, optionally with a per-group learning_rate_multiplier).
        :param update_type: "muon", "adamw" or "sgd", see the module docstring
        :param momentum: Muon momentum factor (update_type "muon" only)
        :param aux_update_type: for update_type "muon": how the remaining params are trained
            ("adamw" or "sgd"), selects the Muon update scaling
        :param beta2: second-moment factor (update_type "adamw" only)
        :param eps: epsilon (update_type "adamw" only)
        :param weight_decay: decoupled weight decay
        :param weight_decay_at_y: optional decay applied while p is still y
        :param beta1: initial y/x interpolation
        :param weight_lr_power: exponent on the per-step lr in the z/x averaging weights
        :param warmup_steps: internal lr warmup, required > 0
        :param rho: beta1 ramp speed after warmup, in [0, 1]. 0 keeps beta1 constant at ``beta1``.
        :param r: polynomial power for the z/x averaging weights
        """
        if warmup_steps != int(warmup_steps):
            raise ValueError(f"AMUSE warmup_steps must be an integer, got {warmup_steps}.")
        if warmup_steps <= 0:
            raise ValueError("AMUSE requires warmup_steps > 0.")
        if not 0.0 < beta1 < 1.0:
            raise ValueError(f"AMUSE beta1 must be in (0, 1), got {beta1}.")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"AMUSE rho must be in [0, 1], got {rho}.")
        if update_type not in UPDATE_TYPES:
            raise ValueError(f"Invalid AMUSE update_type: {update_type}. Expected one of {{'muon', 'adamw', 'sgd'}}.")
        if aux_update_type not in AUX_UPDATE_TYPES:
            raise ValueError(f"Invalid AMUSE aux_update_type: {aux_update_type}. Expected one of {{'adamw', 'sgd'}}.")

        self.update_type = update_type
        self.aux_update_type = aux_update_type
        self.weight_decay_at_y = weight_decay_at_y
        self.beta1_init = float(beta1)
        self.weight_lr_power = weight_lr_power
        self.warmup_steps = int(warmup_steps)
        self.rho = float(rho)
        self.r = r
        self.train_mode = False

        if lr is None:
            lr = _DEFAULT_LR_BY_UPDATE_TYPE[update_type]
        defaults = {"lr": lr, "weight_decay": weight_decay}
        if update_type == "muon":
            defaults["momentum"] = momentum
        elif update_type == "adamw":
            defaults["beta2"] = beta2
            defaults["eps"] = eps

        super().__init__(params, defaults=defaults)
        for group in self.param_groups:
            for legacy_key in ("use_muon", "aux_update_type"):
                if legacy_key in group:
                    raise ValueError(
                        f"AMUSE: param group key {legacy_key!r} is no longer supported."
                        " AMUSE applies one update type per instance now (update_type constructor argument)."
                        " Compose multiple update types over different param subsets"
                        " via returnn.torch.optim.multi.MultiOptimizer."
                    )
            if group.get("update_type", self.update_type) != self.update_type:
                raise ValueError(
                    f"AMUSE: param group update_type {group['update_type']!r} differs from"
                    f" the instance update_type {self.update_type!r}. Per-group update types are"
                    " no longer supported, compose multiple AMUSE instances"
                    " via returnn.torch.optim.multi.MultiOptimizer."
                )
            if self.update_type == "muon":
                for p in group["params"]:
                    if p.ndim < 2:
                        raise ValueError(
                            f"AMUSE with update_type 'muon' requires matrix params (ndim >= 2),"
                            f" got a param of shape {tuple(p.shape)}."
                            " Restrict the params via params_filter"
                            " and train the rest with another update type"
                            " via returnn.torch.optim.multi.MultiOptimizer."
                        )
            self._init_schedule_state(group)

    _pickle_attrs = (
        "update_type",
        "aux_update_type",
        "weight_decay_at_y",
        "beta1_init",
        "weight_lr_power",
        "warmup_steps",
        "rho",
        "r",
        "train_mode",
    )

    def __getstate__(self):
        # the base class covers only defaults/state/param_groups, which would drop these on pickle/deepcopy
        state = super().__getstate__()
        state.update({name: getattr(self, name) for name in self._pickle_attrs})
        return state

    def _init_schedule_state(self, group: Dict[str, Any]):
        """
        Put the schedule state of the param group as float64 scalar tensors on the device of its params.
        Values already in the group are kept (e.g. Python numbers from a checkpoint of an earlier version).

        :param group: param group, modified in place
        """
        device = group["params"][0].device if group["params"] else None
        defaults = {
            "k": 0.0,
            "weight_sum": 0.0,
            "ckp1": 1.0,
            "beta1": self.beta1_init,
            "c_warmup": 1.0 / self.warmup_steps,
        }
        for key, default in defaults.items():
            value = group.get(key, default)
            if isinstance(value, torch.Tensor):
                group[key] = value.to(device=device, dtype=torch.float64)
            else:
                group[key] = torch.tensor(float(value), dtype=torch.float64, device=device)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load the state, see :class:`torch.optim.Optimizer`.
        The schedule state of each param group then is on the device of its params again,
        also when the checkpoint holds it as Python numbers.
        """
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            self._init_schedule_state(group)

    def _init_param_state(self, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        :param p: param with grad
        :return: its state, created if not there yet (the anchor z as a copy of the param, and the moments)
        """
        state = self.state[p]
        if "z" not in state:
            state["z"] = torch.clone(p, memory_format=torch.preserve_format)
        if self.update_type == "muon" and "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        elif self.update_type == "adamw" and "exp_avg_sq" not in state:
            state["exp_avg_sq"] = torch.zeros_like(p)
        return state

    @torch.no_grad()
    def init_state(self):
        """
        Create the state of all params with grad as the first :func:`step` would, without a step,
        for the optimizer step captured in a CUDA graph,
        see :func:`returnn.torch.updater.init_optimizer_state`.
        """
        for group in self.param_groups:
            self._init_schedule_state(group)
            for p in group["params"]:
                if p.grad is not None:
                    self._init_param_state(p)

    def _beta1_and_c_warmup(self, group: Dict[str, Any], t: torch.Tensor, ckp1: torch.Tensor):
        """
        The beta1 of this step and the new anchor of its ramp, as tensor ops on the schedule state
        (no Python branch on their values).
        c_warmup is the ckp1 of the warmup boundary step,
        or the first later ckp1 below 1 if that one is degenerate (e.g. an lr of 0 at the boundary).

        :param group: param group with the schedule state
        :param t: step number, starting at 1
        :param ckp1: z-to-x averaging weight of this step
        :return: (beta1, c_warmup)
        """
        c_warmup = group["c_warmup"]
        c_valid = (c_warmup > 0.0) & (c_warmup < 1.0)
        after_warmup = (t > self.warmup_steps) & (ckp1 < 1.0)
        new_c_warmup = torch.where((t == self.warmup_steps) | (after_warmup & ~c_valid), ckp1, c_warmup)
        ramp = after_warmup & c_valid
        # placeholders where there is no ramp, so the unused lanes stay finite
        half = torch.full_like(ckp1, 0.5)
        ckp1_ = torch.where(ramp, ckp1, half)
        c_warmup_ = torch.where(ramp, c_warmup, half)
        s_t = (ckp1_ * (1.0 - c_warmup_)) / (c_warmup_ * (1.0 - ckp1_))
        beta1_ramp = 1.0 - (s_t**self.rho) * (1.0 - self.beta1_init)
        beta1 = torch.where(ramp, beta1_ramp, torch.full_like(ckp1, self.beta1_init))
        return beta1, new_c_warmup

    @torch.no_grad()
    def eval(self):
        """
        Switch the params from the training iterate y to the averaged weights x,
        for evaluation and checkpoint saving. No-op if already in eval mode.
        """
        if self.train_mode:
            for group in self.param_groups:
                weight = 1.0 - 1.0 / group["beta1"]
                for p in group["params"]:
                    state = self.state.get(p)
                    if state and "z" in state:
                        p.lerp_(end=state["z"], weight=weight.to(p.dtype))
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        """
        Switch the params from the averaged weights x back to the training iterate y.
        No-op if already in train mode.
        """
        if not self.train_mode:
            for group in self.param_groups:
                weight = 1.0 - group["beta1"]
                for p in group["params"]:
                    state = self.state.get(p)
                    if state and "z" in state:
                        p.lerp_(end=state["z"], weight=weight.to(p.dtype))
        self.train_mode = True

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform one optimization step.

        :param closure: optional closure to reevaluate the model and return the loss
        """
        if not self.train_mode:
            raise Exception(
                "Optimizer was not in train mode when step is called. "
                "Please insert .train() and .eval() calls on the optimizer."
            )
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # schedule of this step, tensor ops on the device state (lr as a Python number or a device tensor)
            t = group["k"] + 1
            lr = group["lr"] * torch.clamp(t / self.warmup_steps, max=1.0)

            # ckp1 is the new z-to-x averaging weight c_t
            weight = (t**self.r) * (lr**self.weight_lr_power)
            weight_sum = group["weight_sum"] + weight
            ckp1 = torch.where(weight_sum > 0, weight / weight_sum, torch.ones_like(weight_sum))
            beta1, c_warmup = self._beta1_and_c_warmup(group, t, ckp1)
            group["k"].copy_(t)
            group["weight_sum"].copy_(weight_sum)
            group["ckp1"].copy_(ckp1)
            group["beta1"].copy_(beta1)
            group["c_warmup"].copy_(c_warmup)

            wd = group.get("weight_decay", 0.0)
            if self.update_type == "adamw":
                beta2 = group.get("beta2", 0.999)
                eps = group.get("eps", 1e-10)
                bias_correction2 = 1.0 - beta2**t

            # lerp weights in the param dtype, older torch versions take no other in lerp_
            lerp_weights_by_dtype = {}
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self._init_param_state(p)
                z = state["z"]
                if p.dtype not in lerp_weights_by_dtype:
                    lerp_weights_by_dtype[p.dtype] = [w.to(p.dtype) for w in (1.0 - 1.0 / beta1, ckp1, 1.0 - beta1)]
                to_x, to_ckp1, to_y = lerp_weights_by_dtype[p.dtype]

                if self.weight_decay_at_y != 0.0:
                    z.addcmul_(p, lr * self.weight_decay_at_y, value=-1)
                    p.addcmul_(p, lr * self.weight_decay_at_y * (1.0 - beta1), value=-1)

                # y_t -> x_t, then update z, then rebuild y_{t+1}.
                p.lerp_(end=z, weight=to_x)
                if self.update_type == "muon":
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=group.get("momentum", 0.95),
                        aux_update_type=self.aux_update_type,
                        nesterov=True,
                    )
                    if wd != 0.0:
                        z.mul_(1.0 - lr * wd)
                    z.addcmul_(update.reshape(p.shape), lr, value=-1)
                elif self.update_type == "adamw":
                    v = state["exp_avg_sq"]
                    grad = p.grad
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = v.div(bias_correction2).sqrt_().add_(eps)
                    update = grad / denom
                    if wd != 0.0:
                        update = update.add(z, alpha=wd)
                    z.addcmul_(update, lr, value=-1)
                elif self.update_type == "sgd":
                    if wd != 0.0:
                        z.mul_(1.0 - lr * wd)
                    z.addcmul_(p.grad, lr, value=-1)
                else:
                    raise ValueError(f"Invalid AMUSE update_type: {self.update_type}")
                p.lerp_(end=z, weight=to_ckp1)
                p.lerp_(end=z, weight=to_y)

        return loss
