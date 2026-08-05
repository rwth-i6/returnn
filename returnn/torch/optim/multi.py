"""
:class:`MultiOptimizer`: compose multiple optimizers over disjoint parameter subsets.

Some optimizers only handle a subset of the model parameters by design,
for example :class:`torch.optim.Muon` (PyTorch >= 2.10), which only accepts 2D parameters
and is meant to be combined with e.g. :class:`torch.optim.AdamW`
for biases, normalization parameters, embeddings and output heads.
In plain PyTorch, you would create two optimizers and step both.
RETURNN manages a single optimizer object,
so this module provides a composite optimizer which wraps multiple sub-optimizers
and behaves like a single one.

Example RETURNN config::

    from returnn.torch.optim.multi import make_hidden_matrix_filter

    optimizer = {
        "class": "multi",
        "optimizers": [
            {
                "class": "muon",
                "params_filter": make_hidden_matrix_filter(max_ndim=2),
                "momentum": 0.95,
                "adjust_lr_fn": "match_rms_adamw",
                "weight_decay": 1e-2,
            },
            {
                # No params_filter: the last entry is the catch-all for all remaining params.
                "class": "adamw",
                "learning_rate_multiplier": 0.015,
                "weight_decay": 1e-2,
                "epsilon": 1e-8,
            },
        ],
    }
    learning_rate = 0.02

Each entry of ``optimizers`` describes one sub-optimizer:

- ``class``: resolved like the top-level optimizer class (torch.optim name, module path, class).
- ``params_filter``: callable deciding which parameters this sub-optimizer gets, with signature
  ``params_filter(*, full_param_name: str, param: torch.nn.Parameter, module: torch.nn.Module,
  rf_module, **kwargs) -> bool``.
  The filters are evaluated once at optimizer creation, in declaration order, first match wins.
  Only the last entry may omit the filter and then acts as the catch-all.
  If parameters remain unassigned and there is no catch-all, optimizer creation fails.
- ``learning_rate_multiplier``: relative factor on the global scheduled learning rate for this
  sub-optimizer. Absolute learning rates are not allowed, consistent with the single-optimizer case.
- All further keys are passed to the sub-optimizer constructor.
  ``weight_decay`` uses the same default parameter-group split
  (no decay on biases and blacklisted modules such as LayerNorm and Embedding)
  as in the single-optimizer case, applied per sub-optimizer.
  ``param_groups_custom`` is not supported inside sub-optimizer entries.

The composite is constructed by the RETURNN updater
(see :func:`returnn.torch.updater.Updater._create_optimizer`),
which evaluates the filters (it has access to the parameter names and modules)
and creates the sub-optimizers.
"""

from __future__ import annotations

import functools
from collections.abc import MutableMapping
from typing import Any, Callable, Dict, Iterator, Optional, Sequence

import torch


class MultiOptimizer(torch.optim.Optimizer):
    """
    Composite optimizer wrapping multiple sub-optimizers over disjoint parameter subsets.

    It implements the standard :class:`torch.optim.Optimizer` interface:
    ``param_groups`` is the concatenation of the sub-optimizers' param groups
    (the same dict objects, so in-place updates such as the learning rate schedule
    propagate to the sub-optimizers),
    ``state`` is a live view delegating to the owning sub-optimizer per parameter,
    ``step``/``zero_grad`` are forwarded to all sub-optimizers,
    and ``state_dict``/``load_state_dict`` use the standard flattened format
    (parameter indices counted over the concatenated param groups),
    so optimizer checkpoints are saved and loaded like for any other optimizer.
    Hook registration (``register_step_pre_hook`` etc.) works via the base class.

    The sub-optimizers must cover disjoint parameter sets, which is validated at construction.
    ``add_param_group`` after construction is not supported,
    the parameter assignment is fixed at creation time.

    ``train()``/``eval()`` are forwarded to sub-optimizers which define them
    (schedule-free optimizers such as :class:`returnn.torch.optim.amuse.AMUSE`).

    This class is not constructed via the generic ``optim_class(param_groups, **opts)`` path,
    the RETURNN updater constructs the sub-optimizers and passes them here.
    Subclasses must keep the keyword-only ``sub_optimizers`` constructor argument.
    """

    def __init__(self, *, sub_optimizers: Sequence[torch.optim.Optimizer]):
        """
        :param sub_optimizers: the already constructed sub-optimizers.
            Their param groups must cover disjoint parameter sets.
        """
        sub_optimizers = list(sub_optimizers)
        assert sub_optimizers, "MultiOptimizer: need at least one sub-optimizer"
        for sub in sub_optimizers:
            assert isinstance(sub, torch.optim.Optimizer), f"MultiOptimizer: invalid sub-optimizer {sub!r}"
        param_owner_by_id: Dict[int, int] = {}
        for sub_idx, sub in enumerate(sub_optimizers):
            for group in sub.param_groups:
                for param in group["params"]:
                    if id(param) in param_owner_by_id:
                        raise ValueError(
                            f"MultiOptimizer: param of shape {tuple(param.shape)} is in both"
                            f" sub-optimizer {param_owner_by_id[id(param)]} and sub-optimizer {sub_idx},"
                            " the sub-optimizers must cover disjoint parameter sets"
                        )
                    param_owner_by_id[id(param)] = sub_idx
        self.sub_optimizers = sub_optimizers
        self._in_init = True
        # The base class appends the given group dicts as-is (no copy),
        # so self.param_groups shares the group dicts with the sub-optimizers,
        # and in-place updates (e.g. of "lr") are seen by them.
        # This also sets up the standard base-class machinery (hook containers etc.).
        super().__init__([group for sub in sub_optimizers for group in sub.param_groups], defaults={})
        self._in_init = False
        # Replace the (empty) base-class state container by a live view over the sub-optimizers.
        self.state = _MultiOptimizerStateView(sub_optimizers)

    def __repr__(self):
        return "%s(\n%s\n)" % (
            self.__class__.__name__,
            ",\n".join(repr(sub) for sub in self.sub_optimizers),
        )

    def __reduce__(self):
        # The base class __getstate__ only covers defaults/state/param_groups,
        # which would drop the sub-optimizers on pickle/deepcopy.
        # Reconstruct via the constructor instead, which rebuilds all views consistently.
        return _reconstruct_multi_optimizer, (type(self), self.sub_optimizers)

    def step(self, closure=None):
        """
        Perform one step with each sub-optimizer.
        """
        assert closure is None, "MultiOptimizer: step with closure not supported"
        for sub in self.sub_optimizers:
            sub.step()

    def zero_grad(self, set_to_none: Optional[bool] = None):
        """
        Reset the gradients of all sub-optimizers.

        :param set_to_none: forwarded to the sub-optimizers if given.
            If not given, each sub-optimizer applies its own default,
            which differs across PyTorch versions (False up to 1.13, True since 2.0).
        """
        for sub in self.sub_optimizers:
            if set_to_none is None:
                sub.zero_grad()
            else:
                sub.zero_grad(set_to_none=set_to_none)

    def add_param_group(self, param_group: Dict[str, Any]):
        """
        Not supported after construction: the parameter assignment is fixed at creation time.
        """
        if getattr(self, "_in_init", False):
            super().add_param_group(param_group)
            return
        raise NotImplementedError("MultiOptimizer: add_param_group not supported")

    def train(self):
        """
        Set train mode on all sub-optimizers which support it (schedule-free optimizers).
        """
        for sub in self.sub_optimizers:
            func = getattr(sub, "train", None)
            if callable(func):
                func()

    def eval(self):
        """
        Set eval mode on all sub-optimizers which support it (schedule-free optimizers).
        """
        for sub in self.sub_optimizers:
            func = getattr(sub, "eval", None)
            if callable(func):
                func()

    def state_dict(self) -> Dict[str, Any]:
        """
        :return: state in the standard flattened :class:`torch.optim.Optimizer` format,
            i.e. ``{"state": {param_idx: ...}, "param_groups": [...]}``,
            where the parameter indices count over the concatenated param groups.
            Non-parameter state entries (state keys which are not parameters)
            are not supported.
        """
        for pre_hook in getattr(self, "_optimizer_state_dict_pre_hooks", {}).values():
            pre_hook(self)
        merged_state = {}
        merged_groups = []
        offset = 0
        for sub_idx, sub in enumerate(self.sub_optimizers):
            sub_state_dict = sub.state_dict()
            num_params = sum(len(group["params"]) for group in sub_state_dict["param_groups"])
            for param_idx, param_state in sub_state_dict["state"].items():
                if not isinstance(param_idx, int) or not 0 <= param_idx < num_params:
                    raise NotImplementedError(
                        f"MultiOptimizer: non-parameter state key {param_idx!r}"
                        f" of sub-optimizer {sub_idx} ({type(sub).__name__}) not supported"
                    )
                merged_state[param_idx + offset] = param_state
            for group in sub_state_dict["param_groups"]:
                group = dict(group)
                group["params"] = [param_idx + offset for param_idx in group["params"]]
                merged_groups.append(group)
            offset += num_params
        state_dict = {"state": merged_state, "param_groups": merged_groups}
        for post_hook in getattr(self, "_optimizer_state_dict_post_hooks", {}).values():
            hook_result = post_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """
        Load a state dict in the format of :func:`state_dict`.
        """
        for pre_hook in getattr(self, "_optimizer_load_state_dict_pre_hooks", {}).values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result
        groups = state_dict["param_groups"]
        state = state_dict["state"]
        num_groups_per_sub = [len(sub.param_groups) for sub in self.sub_optimizers]
        assert len(groups) == sum(num_groups_per_sub), (
            f"MultiOptimizer: loaded state dict has {len(groups)} param groups,"
            f" but sub-optimizers have {num_groups_per_sub} groups"
        )
        group_idx = 0
        all_global_param_indices = set()
        for sub, num_groups in zip(self.sub_optimizers, num_groups_per_sub):
            sub_groups = groups[group_idx : group_idx + num_groups]
            group_idx += num_groups
            global_param_indices = [param_idx for group in sub_groups for param_idx in group["params"]]
            all_global_param_indices.update(global_param_indices)
            assert len(global_param_indices) == len(set(global_param_indices)), (
                "MultiOptimizer: duplicate param indices in loaded state dict"
            )
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(global_param_indices)}
            local_groups = []
            for group in sub_groups:
                group = dict(group)
                group["params"] = [global_to_local[param_idx] for param_idx in group["params"]]
                local_groups.append(group)
            local_state = {
                global_to_local[param_idx]: param_state
                for param_idx, param_state in state.items()
                if param_idx in global_to_local
            }
            sub.load_state_dict({"state": local_state, "param_groups": local_groups})
        unconsumed_state_keys = [key for key in state if key not in all_global_param_indices]
        if unconsumed_state_keys:
            raise NotImplementedError(
                f"MultiOptimizer: state keys {unconsumed_state_keys!r} in the loaded state dict"
                " do not correspond to any parameter index, non-parameter state is not supported"
            )
        # The sub-optimizers' load_state_dict rebuilt their param group dicts,
        # so re-establish the shared-dict aliasing of our concatenated view.
        # Otherwise external in-place updates (e.g. the RETURNN LR schedule)
        # would go to stale group dicts which the sub-optimizers no longer read.
        self.param_groups = [group for sub in self.sub_optimizers for group in sub.param_groups]
        for post_hook in getattr(self, "_optimizer_load_state_dict_post_hooks", {}).values():
            post_hook(self)


def _reconstruct_multi_optimizer(cls, sub_optimizers: Sequence[torch.optim.Optimizer]) -> MultiOptimizer:
    """
    Reconstruct a :class:`MultiOptimizer` (or subclass) from its sub-optimizers, for pickle/deepcopy.
    """
    return cls(sub_optimizers=sub_optimizers)


class _MultiOptimizerStateView(MutableMapping):
    """
    Live view over the per-parameter state of the sub-optimizers,
    following the :class:`torch.optim.Optimizer` ``state`` container conventions:
    accessing the state of a known parameter creates an empty entry if there is none yet
    (like the defaultdict the base class uses),
    and mutations are delegated to the owning sub-optimizer.
    """

    def __init__(self, sub_optimizers: Sequence[torch.optim.Optimizer]):
        self._sub_optimizers = list(sub_optimizers)
        self._sub_by_param_id: Dict[int, torch.optim.Optimizer] = {}
        for sub in self._sub_optimizers:
            for group in sub.param_groups:
                for param in group["params"]:
                    self._sub_by_param_id[id(param)] = sub

    def _owning_sub(self, param: torch.nn.Parameter) -> Optional[torch.optim.Optimizer]:
        return self._sub_by_param_id.get(id(param))

    def __getitem__(self, param: torch.nn.Parameter) -> Dict[str, Any]:
        sub = self._owning_sub(param)
        if sub is None:
            raise KeyError(f"MultiOptimizer state: param of shape {tuple(param.shape)} not in any sub-optimizer")
        return sub.state[param]

    def __contains__(self, param) -> bool:
        # Do not go through __getitem__ here (the MutableMapping default),
        # it would create an empty entry in the sub-optimizer's defaultdict.
        sub = self._owning_sub(param)
        return sub is not None and param in sub.state

    def get(self, param, default=None):
        """
        :return: the state of the param, or the default, without creating an entry
        """
        sub = self._owning_sub(param)
        if sub is None or param not in sub.state:
            return default
        return sub.state[param]

    def __setitem__(self, param: torch.nn.Parameter, value: Dict[str, Any]):
        sub = self._owning_sub(param)
        if sub is None:
            raise KeyError(f"MultiOptimizer state: param of shape {tuple(param.shape)} not in any sub-optimizer")
        sub.state[param] = value

    def __delitem__(self, param: torch.nn.Parameter):
        sub = self._owning_sub(param)
        if sub is None:
            raise KeyError(f"MultiOptimizer state: param of shape {tuple(param.shape)} not in any sub-optimizer")
        del sub.state[param]

    def __iter__(self) -> Iterator[torch.nn.Parameter]:
        for sub in self._sub_optimizers:
            yield from sub.state

    def __len__(self) -> int:
        return sum(len(sub.state) for sub in self._sub_optimizers)

    def clear(self):
        """
        Clear the state of all sub-optimizers.
        """
        for sub in self._sub_optimizers:
            sub.state.clear()


_DefaultExcludeNameSubstrings = ("embed", "logit", "lm_head", "head")


def _hidden_matrix_filter(
    *,
    full_param_name: str,
    param: torch.nn.Parameter,
    exclude_name_substrings: Sequence[str],
    min_ndim: int,
    max_ndim: Optional[int],
    **_kwargs,
) -> bool:
    if param.dim() < min_ndim:
        return False
    if max_ndim is not None and param.dim() > max_ndim:
        return False
    lname = full_param_name.lower()
    return not any(s in lname for s in exclude_name_substrings)


def make_hidden_matrix_filter(
    *,
    exclude_name_substrings: Sequence[str] = _DefaultExcludeNameSubstrings,
    min_ndim: int = 2,
    max_ndim: Optional[int] = None,
) -> Callable[..., bool]:
    """
    Make a picklable ``params_filter`` which selects the hidden matrix weights,
    i.e. parameters with ``min_ndim <= ndim (<= max_ndim)``
    whose name does not contain any of the given substrings.
    This is the usual convention for Muon-style optimizers.
    The matrix hidden weights are selected, while embeddings, output heads,
    biases and normalization parameters fall through to the next sub-optimizer.

    Use ``max_ndim=2`` together with :class:`torch.optim.Muon`, which only accepts 2D parameters.
    Note that name-based exclusion depends on your model's parameter naming.
    If it does not fit (e.g. an embedding not containing "embed" in its name),
    write a custom filter, e.g. based on the owning module type.

    :param exclude_name_substrings: case-insensitive substrings of parameter names to exclude
    :param min_ndim: minimum number of dims (default 2, i.e. exclude biases and other vectors)
    :param max_ndim: optional maximum number of dims
    :return: params_filter callable
    """
    return functools.partial(
        _hidden_matrix_filter,
        exclude_name_substrings=tuple(s.lower() for s in exclude_name_substrings),
        min_ndim=min_ndim,
        max_ndim=max_ndim,
    )
