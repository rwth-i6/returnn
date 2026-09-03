"""
Watch for a stalled train step, and dump stacks when it happens.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import multiprocessing
from typing import Optional

from returnn.util.debug import install_subproc_faulthandler


def watch_stall(*, timeout: float, native: bool = True, repeat: int = 3):
    """
    Start a subproc which dumps this process's stacks once it stops making progress.

    A hang inside a CUDA kernel or a collective is invisible in the Python stack:
    it bottoms out at whatever host call waits for the device (e.g. a ``.item()`` sync),
    which does not say what the device is doing.
    The C/C++ frames are the informative part,
    and reading them needs an external tracer, hence a separate process rather than a thread.

    Deliberately never kills anything:
    a stall is often just a slow step,
    and a watchdog that ends the job would destroy the state one wants to inspect.

    :param timeout: seconds without progress before dumping
    :param native: include the C/C++ frames (py-spy --native)
    :param repeat: how many dumps; a second one shows whether it moved at all in between
    :return: heartbeat, whose ``.value`` the caller sets to ``time.time()`` on every step
    """
    heartbeat = multiprocessing.get_context("spawn").Value("d", time.time())
    proc = multiprocessing.get_context("spawn").Process(
        target=_watch_stall_main,
        args=(os.getpid(), heartbeat, float(timeout), bool(native), int(repeat)),
        name="watch_stall",
        daemon=True,
    )
    proc.start()
    return heartbeat


def _find_py_spy() -> Optional[str]:
    # explicit path first: py-spy is often installed outside the env used for training
    cand = os.environ.get("RETURNN_PY_SPY")
    if cand and os.path.exists(cand):
        return cand
    return shutil.which("py-spy")


def _watch_stall_main(pid: int, heartbeat, timeout: float, native: bool, repeat: int):
    if sys.platform == "linux":
        try:
            with open("/proc/self/comm", "w") as f:
                f.write("watch stall")
        except OSError:
            pass

    install_subproc_faulthandler()

    def _print(*args):
        print("STALL:", *args)
        sys.stdout.flush()

    py_spy = _find_py_spy()
    dumps = 0
    while dumps < repeat:
        time.sleep(min(timeout, 30.0))
        if not _alive(pid):
            return
        idle = time.time() - heartbeat.value
        if idle < timeout:
            continue
        _print(f"no progress for {idle:.0f}s in pid {pid} (timeout {timeout:.0f}s), dumping stacks")
        if not py_spy:
            _print("py-spy not found; set RETURNN_PY_SPY=/path/to/py-spy or pip install py-spy")
            return
        cmd = [py_spy, "dump", "--pid", str(pid)] + (["--native"] if native else [])
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            _print(f"py-spy dump (idle {idle:.0f}s):\n{out.stdout}{out.stderr}")
        except (subprocess.TimeoutExpired, OSError) as exc:
            _print(f"py-spy failed: {exc}")
            return
        dumps += 1
        # a stall usually persists; space the dumps out so they show whether anything moved
        time.sleep(timeout)
    _print("dump limit reached, not watching further")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
