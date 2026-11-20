"""
Test :file:`tools/torch_avg_checkpoints.py`.
"""

from __future__ import annotations

import _setup_test_env  # noqa

import os
import sys
import tempfile
import torch

sys.path += ["tools"]

from torch_avg_checkpoints import merge_checkpoints


def test_merge_checkpoints_shared_tensor():
    """
    Every key gets the mean over the checkpoints, also keys that store one tensor together, as tied weights do.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        in_ckpts = []
        for epoch, value in enumerate([1.0, 2.0, 6.0], start=1):
            shared = torch.full((2, 3), value)
            model = {"embedding": shared, "output": shared, "bias": torch.full((3,), value)}
            in_ckpts.append(os.path.join(tmp_dir, f"epoch.{epoch}.pt"))
            torch.save({"model": model, "epoch": epoch, "step": epoch}, in_ckpts[-1])
        out_ckpt = os.path.join(tmp_dir, "avg.pt")
        merge_checkpoints(in_ckpts, out_ckpt)
        out_model = torch.load(out_ckpt)["model"]

    assert torch.equal(out_model["embedding"], torch.full((2, 3), 3.0))
    assert torch.equal(out_model["output"], torch.full((2, 3), 3.0))
    assert torch.equal(out_model["bias"], torch.full((3,), 3.0))
