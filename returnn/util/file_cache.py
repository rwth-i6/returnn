"""
File cache.

Copies files from a remote filesystem (e.g. NFS)
to a local filesystem (e.g. /var/tmp) to speed up access.

See https://github.com/rwth-i6/returnn/issues/1519 for initial discussion.

Main class is :class:`FileCache`.
"""

from typing import Any, Collection, Iterable, List, Optional, Tuple, Union
import errno
import os
import pathlib
import time
import shutil
import dataclasses
from dataclasses import dataclass
from collections import defaultdict
from contextlib import contextmanager
import json
from threading import Thread, Event
from returnn.config import Config, get_global_config
from .basic import expand_env_vars, LockFile, human_bytes_size


__all__ = ["FileCache", "CachedFile", "get_instance"]


class FileCache:
    """
    File cache.

    Copies files from a remote filesystem (e.g. NFS)
    to a local filesystem (e.g. ``/var/tmp``) to speed up access.

    Some assumptions we depend on:

    - When a cached file is available,
      and its size matches the original file
      and its mtime is not older than the original file,
      we can use it.
    - We will update the cached file mtime frequently (every second) via a background thread
      of used cached files, to mark that they are used.
      (We would maybe want to use atime, but we don't expect that atime can be relied on.)
      Note that updating mtime might influence the behavior of some external tools.
    - :func:`os.utime` will update mtime, and mtime is somewhat accurate (up to 10 secs maybe),
      mtime compares to time.time().
    - :func:`shutil.disk_usage` can be relied on.

    See https://github.com/rwth-i6/returnn/issues/1519 for initial discussion.
    """

    def __init__(
        self,
        *,
        cache_directory: str = "$TMPDIR/$USER/returnn/file_cache",
        cleanup_files_always_older_than_days: float = 31.0,
        cleanup_files_wanted_older_than_days: float = 1.0,
        cleanup_disk_usage_wanted_free_ratio: float = 0.2,  # try to free at least 20% disk space
        cleanup_disk_usage_wanted_multiplier: float = 2.0,  # try to free 2x the required space for a file
        num_tries: int = 3,  # retry twice by default
    ):
        """
        :param cache_directory: directory where to cache files.
            Uses :func:`expand_env_vars` to expand environment variables.
        :param cleanup_files_always_older_than_days: always cleanup files older than this.
        :param cleanup_files_wanted_older_than_days: if cleanup_disk_usage_wanted_free_ratio not reached,
            cleanup files older than this.
        :param cleanup_disk_usage_wanted_free_ratio: try to free at least this ratio of disk space.
        :param cleanup_disk_usage_wanted_multiplier: when making space for a new file, try to free at
            least this times as much space.
        :param num_tries: how many times to try caching a file before giving up
        """
        self.cache_directory = expand_env_vars(cache_directory)
        self._cleanup_files_always_older_than_days = cleanup_files_always_older_than_days
        self._cleanup_files_wanted_older_than_days = cleanup_files_wanted_older_than_days
        self._cleanup_disk_usage_wanted_free_ratio = cleanup_disk_usage_wanted_free_ratio
        assert cleanup_disk_usage_wanted_multiplier >= 1.0
        self._cleanup_disk_usage_wanted_multiplier = cleanup_disk_usage_wanted_multiplier
        self._touch_files_thread = _TouchFilesThread(cache_base_dir=self.cache_directory)
        assert num_tries > 0
        self._num_tries = num_tries

    # Note on lock_timeout: It will check whether a potentially existing lock file is older than this timeout,
    # and if so, then it would delete the existing lock file, assuming it is from a crashed previous run.
    # We are always keeping the lock file mtime updated via the _touch_files_thread (every second),
    # so it should never be older than this timeout.
    # If there is a Python exception anywhere here, we will always properly release the lock.
    # Only if the process dies (e.g. killed, segfault or so), the lock file might be left-over,
    # and another process might need to wait for this timeout.
    # We don't expect that this must be configured, so let's just use a reasonable default.
    # This should be more than the _touch_files_thread interval (1 sec).
    _lock_timeout = 20

    def __del__(self):
        self._touch_files_thread.stop.set()

    def get_file(self, src_filename: str) -> str:
        """
        Get cached file.
        This will copy the file to the cache directory if it is not already there.
        This will also make sure that the file is not removed from the cache directory
        via the _touch_files_thread
        until you call :func:`release_file`.

        :param src_filename: source file to copy (if it is not already in the cache).
        :return: cached file path (in the cache directory)
        """
        dst_filename = self._get_dst_filename(src_filename)
        last_error = None
        for try_nr in range(self._num_tries):
            if try_nr > 0:
                print(
                    f"FileCache: Ignoring error while copying {dst_filename}: {type(last_error).__name__}: {last_error}"
                )
                time.sleep(1)
            try:
                self._copy_file_if_needed(src_filename, dst_filename)
                break
            except OSError as e:
                if e.errno == errno.ENOSPC:
                    last_error = e
                else:
                    raise e
        if last_error is not None:
            raise last_error
        info_file = self._get_info_filename(dst_filename)
        os.utime(dst_filename, None)
        os.utime(info_file, None)
        # protect info file from tempreaper, which looks at the mtime
        self._touch_files_thread.files_extend([dst_filename, info_file])
        return dst_filename

    def release_files(self, filenames: Union[str, Iterable[str]]):
        """
        Release cached files.
        This just says that we are not using the files anymore for now.
        They will be kept in the cache directory for now,
        and might be removed when the cache directory is cleaned up.

        :param filenames: files to release (paths in the cache directory)
        """
        if isinstance(filenames, str):
            filenames = [filenames]
        self._touch_files_thread.files_remove(fn_ for fn in filenames for fn_ in [fn, self._get_info_filename(fn)])

    def cleanup(self, *, need_at_least_free_space_size: int = 0):
        """
        Cleanup cache directory.
        """
        if not os.path.exists(self.cache_directory):
            return
        disk_usage = shutil.disk_usage(self.cache_directory)
        want_free_space_size = max(
            int(self._cleanup_disk_usage_wanted_multiplier * need_at_least_free_space_size),
            int(self._cleanup_disk_usage_wanted_free_ratio * disk_usage.total),
        )
        cleanup_timestamp_file = self.cache_directory + "/.recent_full_cleanup"
        try:
            last_full_cleanup = os.stat(cleanup_timestamp_file).st_mtime
        except FileNotFoundError:
            last_full_cleanup = float("-inf")
        # Get current time now, so that cur_time - mtime is pessimistic,
        # and does not count the time for the cleanup itself.
        cur_time = time.time()
        # If we have enough free space, and we did a full cleanup recently, we don't need to do anything.
        if want_free_space_size <= disk_usage.free and cur_time - last_full_cleanup < 60 * 10:
            return
        # immediately update the file's timestamp to reduce racyness between worker processes
        # Path().touch() also creates the file if it doesn't exist yet
        pathlib.Path(cleanup_timestamp_file).touch(exist_ok=True)
        # Do a full cleanup, i.e. iterate through all files in cache directory and check their mtime.
        all_files = []  # mtime, neg size (better for sorting), filename
        for root, dirs, files in os.walk(self.cache_directory):
            for rel_fn in files:
                fn = root + "/" + rel_fn
                if fn == cleanup_timestamp_file:
                    continue
                elif self._is_info_filename(fn):
                    # skip keepalive files, they are processed together with the file they guard
                    continue
                try:
                    f_stat = os.stat(fn)
                except Exception as exc:
                    print(f"FileCache: Error while stat {fn}: {type(exc).__name__}: {exc}")
                    continue
                else:
                    all_files.append((f_stat.st_mtime, -f_stat.st_blocks * 512, fn))
        all_files.sort()
        cur_expected_free = disk_usage.free
        reached_more_recent_files = False
        cur_used_time_threshold = self._lock_timeout * 0.5  # Used files mtime should be updated every second.
        total_cache_files_size = sum(-neg_size for _, neg_size, _ in all_files)
        total_cur_used_cache_files_size = sum(
            -neg_size for mtime, neg_size, fn in all_files if cur_time - mtime <= cur_used_time_threshold
        )
        report_size_str = (
            f"Total size cached files: {human_bytes_size(total_cache_files_size)},"
            f" currently used: {human_bytes_size(total_cur_used_cache_files_size)}"
        )
        for mtime, neg_size, fn in all_files:
            size = -neg_size
            delete_reason = None
            if cur_time - mtime > self._cleanup_files_always_older_than_days * 60 * 60 * 24:
                delete_reason = f"File is {(cur_time - mtime) / 60 / 60 / 24:.1f} days old"
            else:
                reached_more_recent_files = True
            if not delete_reason and need_at_least_free_space_size > cur_expected_free:
                # Still must delete some files.
                if cur_time - mtime > cur_used_time_threshold:
                    delete_reason = f"Still need more space, file is {(cur_time - mtime) / 60 / 60:.1f} hours old"
                else:
                    raise Exception(
                        f"We cannot free enough space on {self.cache_directory}.\n"
                        f"Needed: {human_bytes_size(need_at_least_free_space_size)},\n"
                        f"currently available: {human_bytes_size(cur_expected_free)},\n"
                        f"oldest file is still too recent: {fn}.\n"
                        f"{report_size_str}"
                    )
            if not delete_reason and want_free_space_size > cur_expected_free:
                if cur_time - mtime > self._cleanup_files_wanted_older_than_days * 60 * 60 * 24:
                    delete_reason = f"Still want more space, file is {(cur_time - mtime) / 60 / 60:.1f} hours old"
                else:
                    # All further files are even more recent, so we would neither cleanup them,
                    # so we can also just stop now.
                    break

            if delete_reason:
                cur_expected_free += size
                print(
                    f"FileCache: Delete file {fn}, size {human_bytes_size(size)}. {delete_reason}."
                    f" After deletion, have {human_bytes_size(cur_expected_free)} free space."
                )
                try:
                    os.remove(fn)
                except Exception as exc:
                    print(f"FileCache: Error while removing {fn}: {type(exc).__name__}: {exc}")
                    cur_expected_free -= size
                try:
                    os.remove(self._get_info_filename(fn))
                except Exception as exc:
                    print(f"FileCache: Ignoring error file removing info file of {fn}: {type(exc).__name__}: {exc}")

            if reached_more_recent_files and want_free_space_size <= cur_expected_free:
                # Have enough free space now.
                break

        if need_at_least_free_space_size > cur_expected_free:
            raise Exception(
                f"We cannot free enough space on {self.cache_directory}.\n"
                f"Needed: {human_bytes_size(need_at_least_free_space_size)},\n"
                f"currently available: {human_bytes_size(cur_expected_free)}.\n"
                f"{report_size_str}"
            )

        # Cleanup empty dirs.
        for root, dirs, files in os.walk(self.cache_directory, topdown=False):
            if files:
                continue
            try:
                if cur_time - os.stat(root).st_mtime <= cur_used_time_threshold:  # still in use?
                    continue
            except Exception as exc:
                print(f"FileCache: Error while stat dir {root}: {type(exc).__name__}: {exc}")
                continue
            try:
                # Recheck existence of dirs, because they might have been deleted by us.
                if any(os.path.exists(root + "/" + d) for d in dirs):
                    continue
            except Exception as exc:
                print(f"FileCache: Error while checking sub dirs in {root}: {type(exc).__name__}: {exc}")
                continue
            try:
                # We can delete this empty dir.
                print(f"FileCache: Remove empty dir {root}")
                os.rmdir(root)
            except Exception as exc:
                print(f"FileCache: Error while removing empty dir {root}: {type(exc).__name__}: {exc}")

    def handle_cached_files_in_config(self, config: Any) -> Tuple[Any, List[str]]:
        """
        :param config: some config, e.g. dict, or any nested structure
        :return: modified config, where all :class:`CachedFile` instances are replaced by the cached file path,
            and the list of cached files which are used.
        """
        import tree

        res_files = []

        def _handle_value(value):
            if isinstance(value, CachedFile):
                res = self.get_file(value.filename)
                res_files.append(res)
                return res
            return value

        return tree.map_structure(_handle_value, config), res_files

    def _get_dst_filename(self, src_filename: str) -> str:
        """
        Get the destination filename in the cache directory.
        """
        assert src_filename.startswith("/")
        dst_file_name = self.cache_directory + src_filename
        return dst_file_name

    @staticmethod
    def _get_info_filename(filename: str) -> str:
        """:return: the name of the corresponding info file to `filename`."""
        return f"{filename}.returnn-info"

    @staticmethod
    def _is_info_filename(filename: str) -> bool:
        """:return: whether `filename` points to a info file."""
        return filename.endswith(".returnn-info")

    def _copy_file_if_needed(self, src_filename: str, dst_filename: str):
        """
        Copy the file to the cache directory.
        """
        # Create dirs.
        dst_dir = os.path.dirname(dst_filename)
        os.makedirs(dst_dir, exist_ok=True)

        # Copy the file, while holding a lock. See comment on lock_timeout above.
        with LockFile(
            directory=dst_dir, name=os.path.basename(dst_filename) + ".lock", lock_timeout=self._lock_timeout
        ) as lock, self._touch_files_thread.files_added_context(lock.lockfile):
            # Maybe it was copied in the meantime, while waiting for the lock.
            if self._check_existing_copied_file_maybe_cleanup(src_filename, dst_filename):
                print(f"FileCache: using existing file {dst_filename}")
                return

            print(f"FileCache: Copy file {src_filename} to cache")

            # Make sure we have enough disk space, st_size +1 due to _copy_with_prealloc
            self.cleanup(need_at_least_free_space_size=os.stat(src_filename).st_size + 1)

            dst_tmp_filename = dst_filename + ".copy"
            if os.path.exists(dst_tmp_filename):
                # The minimum age should be at least the lock_timeout.
                # (But leave a bit of room for variance in timing in the sanity check below.)
                dst_tmp_file_age = time.time() - os.stat(dst_tmp_filename).st_mtime
                assert dst_tmp_file_age > self._lock_timeout * 0.8, (
                    f"FileCache: Expected left-over temp copy file {dst_tmp_filename}"
                    f" from crashed previous copy attempt"
                    f" to be older than {self._lock_timeout * 0.8:.1f}s but it is {dst_tmp_file_age} seconds old"
                )

            with self._touch_files_thread.files_added_context(dst_dir):
                # save mtime before the copy process to have it pessimistic
                orig_mtime_ns = os.stat(src_filename).st_mtime_ns
                FileInfo(mtime_ns=orig_mtime_ns).save(self._get_info_filename(dst_filename))

                _copy_with_prealloc(src_filename, dst_tmp_filename)
                os.rename(dst_tmp_filename, dst_filename)

    @staticmethod
    def _check_existing_copied_file_maybe_cleanup(src_filename: str, dst_filename: str) -> bool:
        """
        Check if the file is in the cache directory.
        """
        if not os.path.exists(dst_filename):
            return False
        src_stat = os.stat(src_filename)
        dst_stat = os.stat(dst_filename)
        try:
            last_known_mtime_ns = FileInfo.load(FileCache._get_info_filename(dst_filename)).mtime_ns
        except FileNotFoundError:
            # for existing setups where files may be cached, but the info file has
            # not been written to disk yet, we re-fetch the file
            last_known_mtime_ns = None
        if (
            src_stat.st_size != dst_stat.st_size
            or last_known_mtime_ns is None
            or src_stat.st_mtime_ns > last_known_mtime_ns
        ):
            os.remove(dst_filename)
            return False
        return True


def get_instance(config: Optional[Config] = None) -> FileCache:
    """
    Returns a file cache instance potentially initialized by the global config.

    Uses defaults if no global config is set.
    """
    config = config or get_global_config(return_empty_if_none=True)
    kwargs = config.typed_value("file_cache_opts") or {}
    return FileCache(**kwargs)


def _copy_with_prealloc(src: str, dst: str):
    """
    Copies the file at `src` to `dst` preallocating the space at `dst` before the
    copy to reduce the chance of race conditions w/ free-disk-space checks occuring.

    Note the function preallocates `size + 1` to allow detecting incompletely copied
    files by a mismatch in the file size, should the copy process be interrupted. The
    additional byte is then truncated away after copying.

    In practice this function is used to copy to a temporary file first, so the
    +1-size trick is technically not necessary -- but it also does not hurt leaving
    it in.
    """
    file_size = os.stat(src).st_size
    with open(dst, "wb") as dst_file:
        if file_size > 0:
            # Prealloc size + 1, see docstring for why.
            #
            # See also `_check_existing_copied_file_maybe_cleanup`.
            if os.name == "posix":
                os.posix_fallocate(dst_file.fileno(), 0, file_size + 1)
            else:
                dst_file.seek(file_size)
                dst_file.write(b"\0")
                dst_file.seek(0)
        with open(src, "rb") as src_file:
            if os.name == "posix":
                os.posix_fadvise(src_file.fileno(), 0, file_size, os.POSIX_FADV_SEQUENTIAL)
                os.posix_fadvise(dst_file.fileno(), 0, file_size, os.POSIX_FADV_SEQUENTIAL)
            shutil.copyfileobj(src_file, dst_file)
        dst_file.truncate(file_size)


@dataclass
class CachedFile:
    """
    Represents some file to be cached in a user config.
    See :func:`FileCache.handle_cached_files_in_config`.
    """

    filename: str  # original filename


@dataclass(frozen=True)
class FileInfo:
    """
    Represents meta information about cached files.

    We currently save the last known mtime of the source file.
    """

    mtime_ns: int

    def save(self, filename: str):
        """Saves this instance to `filename`."""
        with open(filename, "wt") as f:
            f.write(json.dumps(dataclasses.asdict(self)))

    @classmethod
    def load(cls, filename: str) -> "FileInfo":
        """Loads a previously saved FileInfo from the file at `filename`."""
        with open(filename, "rt") as f:
            data = json.load(f)
        return cls(**data)


class _TouchFilesThread(Thread):
    def __init__(self, *, interval: float = 1.0, cache_base_dir: str):
        super().__init__(daemon=True)
        self.stop = Event()
        self.files = defaultdict(int)  # usage counter
        self.interval = interval
        self.cache_base_dir = cache_base_dir
        self._is_started = False  # careful: `_started` is already a member of the base class

    def run(self):
        """thread main loop"""
        while True:
            all_files = {}  # dict to have order deterministic
            for filename in self.files:
                all_files[filename] = True
                all_files.update({k: True for k in _all_parent_dirs(filename, base_dir=self.cache_base_dir)})
            for filename in all_files:
                os.utime(filename, None)
            if self.stop.wait(self.interval):
                return

    def start_once(self):
        """reentrant variant of start() that can safely be called multiple times"""
        if self._is_started:
            return
        self._is_started = True
        self.start()

    def files_extend(self, to_add: Union[str, Iterable[str]]):
        """append"""
        if isinstance(to_add, str):
            to_add = [to_add]
        assert isinstance(to_add, Iterable)
        self.start_once()
        for file in to_add:
            self.files[file] += 1

    def files_remove(self, to_remove: Union[str, Iterable[str]]):
        """remove"""
        if isinstance(to_remove, str):
            to_remove = [to_remove]
        assert isinstance(to_remove, Iterable)
        for filename in to_remove:
            self.files[filename] -= 1
            if self.files[filename] <= 0:
                del self.files[filename]

    @contextmanager
    def files_added_context(self, files: Collection[str]):
        """temporarily add files, and remove them afterwards again."""
        self.files_extend(files)
        try:
            yield
        finally:
            self.files_remove(files)


def _all_parent_dirs(filename: str, *, base_dir: str) -> List[str]:
    assert filename.startswith(base_dir + "/")
    dirs = []
    while True:
        filename = os.path.dirname(filename)
        if filename == base_dir:
            break
        dirs.append(filename)
    return dirs
