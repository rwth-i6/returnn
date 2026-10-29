"""
Packed (ragged / varlen) tensor storage:
the sequences are concatenated without padding,
while the tensor keeps its original (virtual) dims,
so the model code is unchanged.

See :mod:`returnn.frontend._packed_backend`.
"""

from __future__ import annotations

from ._packed_backend import pack, pack_import, unpack, is_packed
from ._packed_backend import regap as packed_regap
from ._packed_backend import flat_content as packed_flat_content
from ._packed_backend import conform as packed_conform

__all__ = ["pack", "pack_import", "unpack", "packed_regap", "packed_flat_content", "packed_conform", "is_packed"]
