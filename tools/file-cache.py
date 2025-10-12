#!/usr/bin/env python3

"""
File cache utility.
"""

from __future__ import annotations

import _setup_returnn_env  # noqa
import os
import shutil
import argparse
from returnn.util import file_cache
from returnn.util.basic import get_temp_dir, human_bytes_size


def main():
    """
    Main entry.
    """
    argparser = argparse.ArgumentParser(description="Dump something from dataset.")
    argparser.add_argument("--temp-dir", help="Temp directory to use.", default=None)
    argparser.add_argument("--cache-dir", help="Cache directory to use.", default=None)
    argparser.add_argument("--require-free-space", type=_parse_size, default=0)
    argparser.add_argument("action", help="Action to perform.", choices=["info", "cleanup"])
    args = argparser.parse_args()

    if args.temp_dir:
        os.environ["TMPDIR"] = args.temp_dir
    # logic from get_temp_dir:
    for envname in ["TMPDIR", "TEMP", "TMP"]:
        dirname = os.getenv(envname)
        print(f"env {envname}: {dirname}")
        if dirname:
            break
    print("temp dir:", get_temp_dir(with_username=False))

    cache = file_cache.FileCache(**({"cache_directory": args.cache_dir} if args.cache_dir else {}))
    print("cache dir:", cache.cache_directory)

    disk_usage = shutil.disk_usage(cache.cache_directory)
    print(
        f"Cache dir, disk usage: total {human_bytes_size(disk_usage.total)},"
        f" used {human_bytes_size(disk_usage.used)},"
        f" free {human_bytes_size(disk_usage.free)}"
    )

    if args.action == "cleanup":
        print("Cleanup.")
        print(f"Require free space: {human_bytes_size(args.require_free_space)}")
        res = cache.cleanup(need_at_least_free_space_size=args.require_free_space)
        print(res)


def _parse_size(size_str: str) -> int:
    """
    Parse size string like "10G" or "500M" to integer bytes.
    """
    size_str = size_str.strip().upper()
    if size_str.endswith("G"):
        return int(float(size_str[:-1]) * (1024**3))
    if size_str.endswith("M"):
        return int(float(size_str[:-1]) * (1024**2))
    if size_str.endswith("K"):
        return int(float(size_str[:-1]) * 1024)
    return int(size_str)


if __name__ == "__main__":
    main()
