"""
Manager with watchdog
"""

from __future__ import annotations
from typing import Optional
import os
import signal
import time
import threading
import multiprocessing

# noinspection PyProtectedMember
from multiprocessing.context import BaseContext
from multiprocessing.managers import SyncManager


def create_manager(*, ctx: Optional[BaseContext] = None) -> SyncManager:
    """
    Basically the same as :func:`multiprocessing.Manager` or :func:`ctx.Manager`,
    but with a watch dog thread that will kill the manager process if the parent process dies.
    """
    print(f"main (parent) pid is {os.getpid()}")
    if ctx is None:
        ctx = multiprocessing.get_context()
    m = SyncManager(ctx=ctx)
    m.start(initializer=_manager_init_watch_dog, initargs=(os.getpid(),))
    return m


def _manager_init_watch_dog(parent_pid: int) -> None:
    """
    This is the initializer for the manager. It is called when the manager process starts.
    """
    thread = threading.Thread(target=_manager_die_if_parent_dies, args=(parent_pid,), name="manager_watch_dog_thread")
    thread.daemon = True
    thread.start()


def _manager_die_if_parent_dies(parent_pid: int) -> None:
    """
    This is the function that is called when the manager process starts.
    It checks if the parent process is still alive.
    If the parent process dies, it will kill the manager process.
    """
    while True:
        new_parent_pid = os.getppid()
        if new_parent_pid != parent_pid:
            print(
                f"multiprocessing.Manager process ({os.getpid()}:"
                f" parent process {parent_pid} is dead,"
                f" new parent pid is {new_parent_pid},"
                f" killing myself"
            )
            os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.1)
