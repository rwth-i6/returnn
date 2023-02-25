"""
Used by setup.py.
"""

from __future__ import annotations
from pprint import pprint
import os
import sys


_my_dir = os.path.dirname(os.path.abspath(__file__))
# Use realpath to resolve any symlinks. We want the real root-dir, to be able to check the Git revision.
_root_dir = os.path.dirname(os.path.realpath(_my_dir))


def debug_print_file(fn):
    """
    :param str fn:
    """
    print("%s:" % fn)
    if not os.path.exists(fn):
        print("<does not exist>")
        return
    if os.path.isdir(fn):
        print("<dir:>")
        pprint(sorted(os.listdir(fn)))
        return
    print(open(fn).read())


def parse_pkg_info(fn):
    """
    :param str fn:
    :return: dict with info written by distutils. e.g. ``res["Version"]`` is the version.
    :rtype: dict[str,str]
    """
    res = {}
    for ln in open(fn).read().splitlines():
        if not ln or not ln[:1].strip():
            continue
        key, value = ln.split(": ", 1)
        res[key] = value
    return res


def git_head_version(git_dir=_root_dir, long=False):
    """
    :param str git_dir:
    :param bool long: see :func:`get_version_str`
    :rtype: str
    """
    from returnn.util.basic import git_commit_date, git_commit_rev, git_is_dirty

    commit_date = git_commit_date(git_dir=git_dir)  # like "20190202.154527"
    version = "1.%s" % commit_date  # distutils.version.StrictVersion compatible
    if long:
        # Keep SemVer compatible.
        rev = git_commit_rev(git_dir=git_dir)
        version += "+git.%s" % rev
        if git_is_dirty(git_dir=git_dir):
            version += ".dirty"
    return version


def get_version_str(verbose=False, verbose_error=False, fallback=None, long=False):
    """
    :param bool verbose: print exactly how we end up with some version
    :param bool verbose_error: print only any potential errors
    :param str|None fallback:
    :param bool long:
      False: Always distutils.version.StrictVersion compatible. just like "1.20190202.154527".
      True: Will also add the revision string, like "1.20180724.141845+git.7865d01".
        The format might change in the future.
        We will keep it `SemVer <https://semver.org/>`__ compatible.
        I.e. the string before the `"+"` will be the short version.
        We always make sure that there is a `"+"` in the string.
    :rtype: str
    """
    # Earlier we checked PKG-INFO, via parse_pkg_info. Both in the root-dir and in my-dir.
    # Now we should always have _setup_info_generated.py, copied by our own setup.
    # Do not use PKG-INFO at all anymore (for now), as it would only have the short version.
    # Only check _setup_info_generated in the current dir, not in the root-dir,
    # because we want to only use it if this was installed via a package.
    # Otherwise, we want the current Git version.
    if os.path.exists("%s/_setup_info_generated.py" % _my_dir):
        # noinspection PyUnresolvedReferences
        from . import _setup_info_generated as info

        if verbose:
            print("Found _setup_info_generated.py, long version %r, version %r." % (info.long_version, info.version))
        if long:
            assert "+" in info.long_version
            return info.long_version
        return info.version

    info_in_root_filename = "%s/_setup_info_generated.py" % _root_dir
    if os.path.exists(info_in_root_filename):
        # The root dir might not be in sys.path, so just load directly.
        code = compile(open(info_in_root_filename).read(), info_in_root_filename, "exec")
        info = {}
        eval(code, info)
        version = info["version"]
        long_version = info["long_version"]
        if verbose:
            print("Found %r in root, long version %r, version %r." % (info_in_root_filename, long_version, version))
        if long:
            assert "+" in long_version
            return long_version
        return version

    if os.path.exists("%s/.git" % _root_dir):
        try:
            version = git_head_version(git_dir=_root_dir, long=long)
            if verbose:
                print("Version via Git:", version)
            if long:
                assert "+" in version
            return version
        except Exception as exc:
            if verbose or verbose_error:
                print("Exception while getting Git version:", exc)
                sys.excepthook(*sys.exc_info())
            if not fallback:
                raise  # no fallback

    if fallback:
        if verbose:
            print("Version via fallback:", fallback)
        if long:
            assert "+" in fallback
        return fallback
    raise Exception("Cannot get RETURNN version.")
