"""
Checkpoint serialization.
"""

from typing import Any
import os
import time
import torch
from returnn.log import log


def save_checkpoint(obj: Any, filename: str):
    """
    Save a checkpoint atomically, retrying transient write failures up to three times.

    Each attempt starts with a fresh temporary file. The previous checkpoint is
    replaced only after a successful save; the final error propagates to the caller.
    """
    tmp_filename = filename + ".tmp_write"
    for attempt in range(3):
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)
        try:
            torch.save(obj, tmp_filename)
        except (OSError, RuntimeError) as exc:
            if attempt == 2:
                raise
            print(f"Checkpoint write failed for {filename}: {exc}. Retrying ({attempt + 1}/2).", file=log.v3)
            time.sleep(attempt + 1)
        else:
            os.rename(tmp_filename, filename)
            return
