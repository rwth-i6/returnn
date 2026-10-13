"""
Packed-tensors batch configuration (``packed_tensors``, see :mod:`returnn.frontend._packed_backend`),
framework-agnostic: used by the torch data pipeline and the TF RF engine.
"""

from __future__ import annotations
from typing import Optional, Dict, Any

__all__ = ["packed_batch_config", "packed_batch_key_opts"]


def packed_batch_config() -> Optional[Dict[str, Any]]:
    """
    :return: the ``packed_tensors`` config dict, or None if packing is off.
        ``packed_tensors`` is ``True`` (all defaults: dense, gap 0, align 1)
        or a dict with the global ``gap``/``align`` and optional per-key overrides
        under the reserved ``per_key`` sub-dict::

            packed_tensors = {"gap": 120, "align": 6, "per_key": {"data": {"gap": 240}}}

        The defaults are resolved per key by :func:`packed_batch_key_opts`,
        so this only validates the keys and passes the dict through (``True`` -> ``{}``).
    """
    from returnn.config import get_global_config

    config = get_global_config(raise_exception=False)
    if config is None:
        return None
    opt = config.typed_value("packed_tensors", None)
    if opt is None:
        opt = config.bool("packed_tensors", False)
    if not opt:
        return None
    if opt is True:
        return {}
    assert isinstance(opt, dict), f"packed_tensors: expected bool or dict, got {opt!r}"
    allowed = {"gap", "align", "per_key"}
    assert set(opt).issubset(allowed), f"packed_tensors: unexpected keys {set(opt) - allowed}, allowed {allowed}"
    return opt


def packed_batch_key_opts(packing: Dict[str, Any], key: str) -> Optional[Dict[str, int]]:
    """
    :return: the ``{"gap", "align"}`` for the given data key, per-key override else global default;
        None if the key opts out of packing (``per_key: {<key>: {"packed": False}}`` -> padded),
        e.g. targets that the train step consumes padded while the audio is packed
    """
    per = packing.get("per_key", {}).get(key, {})
    if not per.get("packed", True):
        return None
    return {
        "gap": int(per.get("gap", packing.get("gap", 0))),
        "align": int(per.get("align", packing.get("align", 1))),
    }
