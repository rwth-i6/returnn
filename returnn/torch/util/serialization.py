"""
Checkpoint serialization.
"""

from typing import Any
import errno
import os
import time
import torch
from returnn.log import log


def save_checkpoint(obj: Any, filename: str):
    """
    Save a checkpoint atomically, retrying transient write failures every ten seconds.

    Like TFNetwork.save_params_to_file, retry EBUSY, EDQUOT, EIO and ENOSPC until
    the filesystem recovers. PyTorch reports stream writer failures as RuntimeError.
    Other errors propagate immediately. Each attempt starts with a fresh temporary
    file, and the previous checkpoint is replaced only after a successful save.
    """
    tmp_filename = filename + ".tmp_write"
    while True:
        if os.path.exists(tmp_filename):
            os.unlink(tmp_filename)
        try:
            torch.save(obj, tmp_filename)
        except (OSError, RuntimeError) as exc:
            if isinstance(exc, OSError):
                if exc.errno not in (errno.EBUSY, errno.EDQUOT, errno.EIO, errno.ENOSPC):
                    raise
            elif "PytorchStreamWriter" not in str(exc):
                raise
            print(f"Checkpoint write failed for {filename}: {exc}. Retrying in 10 secs.", file=log.v3)
            time.sleep(10)
        else:
            os.rename(tmp_filename, filename)
            return
