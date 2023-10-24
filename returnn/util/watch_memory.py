"""
Watch memory usage over time.
"""

from __future__ import annotations
from typing import Dict
import time
from collections import defaultdict
from threading import Thread
import psutil


def watch_memory():
    global _watch_memory_thread
    if _watch_memory_thread:
        return
    _watch_memory_thread = Thread(target=_watch_memory_thread_main, name="watch_memory", daemon=True)
    _watch_memory_thread.start()


_watch_memory_thread = None


def _watch_memory_thread_main():
    prefix = "MEMORY:"
    cur_proc = psutil.Process()
    procs = []
    mem_per_pid = {}

    while True:
        change = False
        procs_ = [cur_proc] + cur_proc.children(recursive=True)
        for p in procs:
            if p not in procs_:
                print(prefix, f"proc {_format_proc(p)} exited, old:", _format_mem_info(mem_per_pid[p.pid]))
                mem_per_pid.pop(p.pid, None)
                change = True
        procs = procs_

        for p in procs:
            old_mem_info = mem_per_pid.get(p.pid, None)
            try:
                mem_info = get_mem_info(p)
            except psutil.NoSuchProcess:  # race condition, can happen
                if old_mem_info:
                    print(prefix, f"proc {_format_proc(p)} exited, old:", _format_mem_info(old_mem_info))
                    mem_per_pid.pop(p.pid, None)
                    change = True
                continue
            proc_prefix = "main" if p == cur_proc else "sub"
            if not old_mem_info:
                print(prefix, f"{proc_prefix} proc {_format_proc(p)} initial:", _format_mem_info(mem_info))
                mem_per_pid[p.pid] = mem_info
                change = True
            elif mem_info["rss"] > old_mem_info["rss"] and _format_mem_size(old_mem_info["rss"]) != _format_mem_size(
                mem_info["rss"]
            ):
                print(prefix, f"{proc_prefix} proc {_format_proc(p)} increased RSS:", _format_mem_info(mem_info))
                # keep old info otherwise, such that the update check works
                mem_per_pid[p.pid] = mem_info
                change = True

        if change:
            res = {"pss": 0, "uss": 0}
            for mem_info in mem_per_pid.values():
                for k in res.keys():
                    res[k] += mem_info[k]
            print(prefix, f"total ({len(mem_per_pid)} procs):", _format_mem_info(res))

        time.sleep(5)


def _format_proc(proc: psutil.Process) -> str:
    return "%s(%s)" % (proc.name(), proc.pid)


def _format_mem_info(info: Dict[str, int]) -> str:
    return " ".join("%s=%s" % (k, _format_mem_size(v)) for (k, v) in info.items())


def _format_mem_size(c: int) -> str:
    if c < 1024:
        return "%iB" % c
    S = "KMG"
    i = 0
    while i < len(S) - 1:
        if c < 0.8 * 1024 ** (i + 2):
            break
        i += 1
    f = float(c) / (1024 ** (i + 1))
    return "%.1f%sB" % (f, S[i])


def get_mem_info(proc: psutil.Process) -> Dict[str, int]:
    """
    Code from:
    https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
    """
    res = defaultdict(int)
    for mmap in proc.memory_maps():
        res["rss"] += mmap.rss
        res["pss"] += mmap.pss
        res["uss"] += mmap.private_clean + mmap.private_dirty
        res["shared"] += mmap.shared_clean + mmap.shared_dirty
    return res
