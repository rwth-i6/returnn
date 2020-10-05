
"""
Used by setup.py.
"""

from __future__ import print_function
from pprint import pprint
import os
import sys
import shutil


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


def get_version_str(verbose=False, fallback=None, long=False):
  """
  :param bool verbose:
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
  # Earlier we checked PKG-INFO, via parse_pkg_info. Both in the root-dir as well as in my-dir.
  # Now we should always have _setup_info_generated.py, copied by our own setup.
  # Do not use PKG-INFO at all anymore (for now), as it would only have the short version.
  # Only check _setup_info_generated in the current dir, not in the root-dir,
  # because we want to only use it if this was installed via a package.
  # Otherwise we want the current Git version.
  if os.path.exists("%s/_setup_info_generated.py" % _my_dir):
    # noinspection PyUnresolvedReferences
    from . import _setup_info_generated as info
    if verbose:
      print("Found _setup_info_generated.py, long version %r, version %r." % (info.long_version, info.version))
    if long:
      assert "+" in info.long_version
      return info.long_version
    return info.version

  if os.path.exists("%s/.git" % _root_dir):
    try:
      version = git_head_version(git_dir=_root_dir, long=long)
      if verbose:
        print("Version via Git:", version)
      if long:
        assert "+" in version
      return version
    except Exception as exc:
      if verbose:
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


def main():
  """
  Setup main entry
  """
  # Do not use current time as fallback for the version anymore,
  # as this would result in a version which can be bigger than what we actually have,
  # so this would not be useful at all.
  long_version = get_version_str(verbose=True, fallback="1.0.0+setup-fallback-version", long=True)
  version = long_version[:long_version.index("+")]

  if os.environ.get("DEBUG", "") == "1":
    debug_print_file(".")
    debug_print_file("PKG-INFO")
    debug_print_file("pip-egg-info")
    debug_print_file("pip-egg-info/returnn.egg-info")
    debug_print_file("pip-egg-info/returnn.egg-info/SOURCES.txt")  # like MANIFEST

  if os.path.exists("PKG-INFO"):
    if os.path.exists("MANIFEST"):
      print("package_data, found PKG-INFO and MANIFEST")
      package_data = open("MANIFEST").read().splitlines() + ["PKG-INFO"]
    else:
      print("package_data, found PKG-INFO, no MANIFEST, use *")
      # Currently the setup will ignore all other data except in returnn/.
      # At least make the version available.
      shutil.copy("PKG-INFO", "returnn/")
      shutil.copy("_setup_info_generated.py", "returnn/")
      # Just using package_data = ["*"] would only take files from current dir.
      package_data = []
      for root, dirs, files in os.walk('.'):
        for file in files:
          package_data.append(os.path.join(root, file))
  else:
    print("dummy package_data, does not matter, likely you are running sdist")
    with open("_setup_info_generated.py", "w") as f:
      f.write("version = %r\n" % version)
      f.write("long_version = %r\n" % long_version)
    package_data = ["MANIFEST", "_setup_info_generated.py"]

  from distutils.core import setup
  setup(
    name='returnn',
    version=version,
    packages=['returnn'],
    include_package_data=True,
    package_data={'returnn': package_data},  # filtered via MANIFEST.in
    description='The RWTH extensible training framework for universal recurrent neural networks',
    author='Albert Zeyer',
    author_email='albzey@gmail.com',
    url='https://github.com/rwth-i6/returnn/',
    license='RETURNN license',
    long_description=open('README.rst').read(),
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
      'Development Status :: 5 - Production/Stable',
      'Environment :: Console',
      'Environment :: GPU',
      'Environment :: GPU :: NVIDIA CUDA',
      'Intended Audience :: Developers',
      'Intended Audience :: Education',
      'Intended Audience :: Science/Research',
      'License :: Other/Proprietary License',
      'Operating System :: MacOS :: MacOS X',
      'Operating System :: Microsoft :: Windows',
      'Operating System :: POSIX',
      'Operating System :: Unix',
      'Programming Language :: Python',
      'Programming Language :: Python :: 3',
      'Topic :: Scientific/Engineering',
      'Topic :: Scientific/Engineering :: Artificial Intelligence',
      'Topic :: Software Development :: Libraries :: Python Modules',
    ]
  )


if __name__ == "__main__":
  main()
