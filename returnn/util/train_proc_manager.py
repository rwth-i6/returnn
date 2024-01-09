"""
Start RETURNN training in a subprocess
and if it crashes, try to restart it under certain conditions.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

from returnn.config import Config
from returnn.util import basic as util
from returnn.engine.base import EngineBase


def maybe_start_train_proc_manager(*, config: Config):
    """
    Under certain conditions, start the train proc manager.
    """
    if os.environ.get("__RETURNN_PROC_MANAGED") == "1":
        print("Running in managed mode.")
        return
    if config.value("task", "train") != "train":
        return
    if config.value("start_epoch", "auto") != "auto":
        return

    main_proc_manager(config=config)


def main_proc_manager(*, config: Config):
    """
    Start this in the parent.
    """
    os.environ["__RETURNN_PROC_MANAGED"] = "1"

    print("RETURNN starting up, version %s, train proc manager" % util.describe_returnn_version())

    # Setup BackendEngine. This is that other utility functions know the right model file extension.
    # _select_rf_backend to avoid importing Torch/TF/anything here.
    util.BackendEngine.select_engine(config=config, _select_rf_backend=False)

    total_start_time = time.time()
    num_starts = 0
    last_model_epoch = -1
    while True:
        models = EngineBase.get_existing_models(config, for_training=True)
        cur_model_epoch = max(models) if models else 0
        print("Most recent trained model:", models.get(cur_model_epoch))

        if last_model_epoch is not None:
            print("Most recent trained model before RETURNN run:", last_model_epoch)
            assert last_model_epoch <= cur_model_epoch
            print(f"-> trained successfully {cur_model_epoch - last_model_epoch} epoch(s)")
            if cur_model_epoch == last_model_epoch:
                print("-> break")
                break
            print("Try again, restart RETURNN...")
        else:
            print("Run RETURNN...")

        sys.stdout.flush()
        last_model_epoch = cur_model_epoch
        num_starts += 1
        start_time = time.time()
        proc = subprocess.Popen(sys.argv, executable=sys.executable)
        proc.wait()
        print("RETURNN runtime:", util.hms(time.time() - start_time))
        print("RETURNN return code:", proc.returncode)
        if proc.returncode == 0:
            break

    print("Total RETURNN num starts:", num_starts)
    print("Total RETURNN runtime:", util.hms(time.time() - total_start_time))
