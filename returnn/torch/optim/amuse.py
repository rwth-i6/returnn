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

The learning rate of each param group (as set externally, e.g. by the RETURNN LR schedule)
is used as the base learning rate, and AMUSE applies its internal warmup factor
``min(1, t / warmup_steps)`` on top.
The warmup is required by the schedule-free averaging (the z/x averaging weights
are a function of the per-step learning rate), so do not disable it.
If an external schedule already contains a warmup, the two warmups multiply.

AMUSE reads the learning rate from the param group on the host in each step
(including ``.item()`` on device lr tensors),
so captured optimizer steps (``torch_cuda_graph`` with ``"capture_optimizer"``) are not supported.
"""

from __future__ import annotations

from typing import Optional

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
    - ``state["z"]`` stores the anchor z.
    - ``state["beta1"]`` stores the beta1 used to encode the param's current y,
      ``state["step"]`` the number of updates the param received.
    - ``eval()`` converts y -> x using the param's ``state["beta1"]``.
    - ``train()`` converts x -> y using the current group beta1 and records it in ``state["beta1"]``.

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
            group.setdefault("k", 0)
            group.setdefault("weight_sum", 0.0)
            group.setdefault("beta1", self.beta1_init)

    def _compute_beta1(self, group, t, ckp1):
        if t <= self.warmup_steps:
            if t == self.warmup_steps:
                group["c_warmup"] = ckp1
            return self.beta1_init

        if ckp1 >= 1.0:
            return self.beta1_init
        c_warmup = group.get("c_warmup", 1.0 / self.warmup_steps)
        if not 0.0 < c_warmup < 1.0:
            group["c_warmup"] = ckp1
            return self.beta1_init
        s_t = (ckp1 * (1.0 - c_warmup)) / (c_warmup * (1.0 - ckp1))
        return 1.0 - (s_t**self.rho) * (1.0 - self.beta1_init)

    def _get_z(self, p):
        state = self.state[p]
        z = state.get("z")
        if z is None:
            z = state["z"] = torch.clone(p, memory_format=torch.preserve_format)
        return z

    def _apply_weight_decay_at_y(self, p, z, lr, beta1):
        if self.weight_decay_at_y == 0.0:
            return
        z.sub_(p, alpha=lr * self.weight_decay_at_y)
        p.sub_(p, alpha=lr * self.weight_decay_at_y * (1.0 - beta1))

    @torch.no_grad()
    def eval(self):
        """
        Switch the params from the training iterate y to the averaged weights x,
        for evaluation and checkpoint saving. No-op if already in eval mode.
        """
        if self.train_mode:
            for group in self.param_groups:
                group_beta1 = group.get("beta1", self.beta1_init)
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        beta1 = state.get("beta1", group_beta1)
                        p.lerp_(end=state["z"], weight=1.0 - 1.0 / beta1)
        self.train_mode = False

    @torch.no_grad()
    def train(self):
        """
        Switch the params from the averaged weights x back to the training iterate y.
        No-op if already in train mode.
        """
        if not self.train_mode:
            for group in self.param_groups:
                beta1 = group.get("beta1", self.beta1_init)
                for p in group["params"]:
                    state = self.state[p]
                    if "z" in state:
                        p.lerp_(end=state["z"], weight=1.0 - beta1)
                        state["beta1"] = beta1
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
            base_lr = group["lr"]
            if isinstance(base_lr, torch.Tensor):
                base_lr = base_lr.item()
            k = group["k"]

            t = k + 1
            sched = min(1.0, t / self.warmup_steps)
            lr = base_lr * sched

            # ckp1 is the new z-to-x averaging weight c_t.
            weight = (t**self.r) * (lr**self.weight_lr_power)
            future_weight_sum = group.get("weight_sum", 0.0) + weight
            ckp1 = weight / future_weight_sum if future_weight_sum > 0 else 1.0
            group["ckp1"] = ckp1
            group["weight_sum"] = future_weight_sum

            beta1_prev_group = group["beta1"]
            beta1 = self._compute_beta1(group, t, ckp1)
            group["beta1"] = beta1
            wd = group.get("weight_decay", 0.0)
            beta_m = group.get("momentum", 0.95)
            beta2 = group.get("beta2", 0.999)
            eps = group.get("eps", 1e-10)

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "z" in state:
                    # y was encoded with the group beta1 of the param's last update,
                    # legacy state without these entries was updated in every group step so far
                    step = state.get("step", k) + 1
                    beta1_prev = state.get("beta1", beta1_prev_group)
                else:
                    step = 1
                    beta1_prev = beta1_prev_group
                state["step"] = step
                z = self._get_z(p)
                self._apply_weight_decay_at_y(p, z, lr, beta1_prev)

                # y_t -> x_t with the beta1 that encoded y_t, then update z, then rebuild y_{t+1}.
                p.lerp_(end=z, weight=1.0 - 1.0 / beta1_prev)
                if self.update_type == "muon":
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(
                        p.grad,
                        state["momentum_buffer"],
                        beta=beta_m,
                        aux_update_type=self.aux_update_type,
                        nesterov=True,
                    )
                    if wd != 0.0:
                        z.mul_(1.0 - lr * wd)
                    z.add_(update.reshape(p.shape), alpha=-lr)
                elif self.update_type == "adamw":
                    if "exp_avg_sq" not in state:
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    v = state["exp_avg_sq"]
                    grad = p.grad
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = v.div(1.0 - beta2**step).sqrt_().add_(eps)
                    update = grad / denom
                    if wd != 0.0:
                        update = update.add(z, alpha=wd)
                    z.add_(update, alpha=-lr)
                elif self.update_type == "sgd":
                    if wd != 0.0:
                        z.mul_(1.0 - lr * wd)
                    z.add_(p.grad, alpha=-lr)
                else:
                    raise ValueError(f"Invalid AMUSE update_type: {self.update_type}")
                p.lerp_(end=z, weight=ckp1)
                p.lerp_(end=z, weight=1.0 - beta1)
                state["beta1"] = beta1

            group["k"] = k + 1

        return loss
