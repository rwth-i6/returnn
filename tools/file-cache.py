#!/usr/bin/env python3

"""
File cache utility.
"""

from __future__ import annotations

import _setup_returnn_env  # noqa
import sys
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
    argparser.add_argument("--cleanup-require-free-space", type=_parse_size, default=0)
    argparser.add_argument(
        "--cleanup-files-always-older-than-days", type=float, help="Cleanup files always older than this many days."
    )
    argparser.add_argument(
        "--cleanup-files-wanted-older-than-days", type=float, help="Cleanup files wanted older than this many days."
    )
    argparser.add_argument(
        "--cleanup-disk-usage-wanted-free-ratio", type=float, help="Cleanup to reach this free disk space ratio."
    )
    argparser.add_argument(
        "--cleanup-disk-usage-wanted-multiplier", type=float, help="Cleanup to free this multiplier of required space."
    )

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

    cache = file_cache.FileCache(
        **({"cache_directory": args.cache_dir} if args.cache_dir else {}),
        **(
            {"cleanup_files_always_older_than_days": args.cleanup_files_always_older_than_days}
            if args.cleanup_files_always_older_than_days is not None
            else {}
        ),
        **(
            {"cleanup_files_wanted_older_than_days": args.cleanup_files_wanted_older_than_days}
            if args.cleanup_files_wanted_older_than_days is not None
            else {}
        ),
        **(
            {"cleanup_disk_usage_wanted_free_ratio": args.cleanup_disk_usage_wanted_free_ratio}
            if args.cleanup_disk_usage_wanted_free_ratio is not None
            else {}
        ),
        **(
            {"cleanup_disk_usage_wanted_multiplier": args.cleanup_disk_usage_wanted_multiplier}
            if args.cleanup_disk_usage_wanted_multiplier is not None
            else {}
        ),
    )
    print("cache dir:", cache.cache_directory)
    if not os.path.exists(cache.cache_directory):
        print("Cache dir does not exist.")
        sys.exit(1)

    disk_usage = shutil.disk_usage(cache.cache_directory)
    print(
        f"Cache dir, disk usage: total {human_bytes_size(disk_usage.total)},"
        f" used {human_bytes_size(disk_usage.used)},"
        f" free {human_bytes_size(disk_usage.free)}"
    )

    if args.action == "cleanup":
        print("Cleanup.")
        print(f"Require free space: {human_bytes_size(args.cleanup_require_free_space)}")
        res = cache.cleanup(need_at_least_free_space_size=args.cleanup_require_free_space)
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
