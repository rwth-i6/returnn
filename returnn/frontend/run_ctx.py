"""
Run context

We can either be in param-init stage,
or in the main training loop,
or forwarding loop.
"""

from __future__ import annotations


__all__ = ["RunCtx", "get_current_run_ctx"]


def get_current_run_ctx() -> RunCtx:
    """
    :return: current run context, see :class:`RunCtx`
    """
    pass  # TODO...


class RunCtx:
    """
    We can either be in param-init stage,
    or in the main training loop,
    or forwarding loop.

    In training, we expect that some loss is being defined via mark_as_loss().
    In forwarding, we expect that some output is being defined via mark_as_output().
    """

    # TODO ...
