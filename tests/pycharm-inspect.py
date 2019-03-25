#!/usr/bin/env python3

"""
Will use the PyCharm code inspection.

See here:
  https://github.com/albertz/pycharm-inspect
  https://stackoverflow.com/questions/55323910/pycharm-code-style-check-via-command-line
"""

import os
import sys
import subprocess
import tempfile
from glob import glob
import argparse

my_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(my_dir)
sys.path.insert(0, base_dir)
os.chdir(base_dir)

import better_exchook  # noqa
better_exchook.install()


def check_pycharm_dir(pycharm_dir):
  """
  :param str pycharm_dir:
  """
  assert os.path.isdir(pycharm_dir)
  assert os.path.exists("%s/bin/inspect.sh" % pycharm_dir)


def install_pycharm():
  """
  :return: pycharm dir
  :rtype: str
  """
  # travis_fold: https://github.com/travis-ci/travis-ci/issues/1065
  print("travis_fold:start:script.install")
  print("Install PyCharm...")
  pycharm_dir = "%s/pycharm" % tempfile.mkdtemp()
  subprocess.check_call([my_dir + "/install_pycharm.sh"], cwd=os.path.dirname(pycharm_dir), stderr=subprocess.STDOUT)
  check_pycharm_dir(pycharm_dir)
  print("travis_fold:end:script.install")
  return pycharm_dir


def prepare_src_dir(files=None):
  """
  New clean source dir, where we symlink only the relevant src files.

  :param list[str]|None files:
  :return: src dir
  :rtype: str
  """
  print("travis_fold:start:script.prepare")
  print("Prepare project source files...")
  if not files:
    files = sorted(glob(base_dir + "/*.py"))
  src_tmp_dir = "%s/returnn" % tempfile.mkdtemp()
  os.mkdir(src_tmp_dir)
  for fn in files:
    os.symlink(fn, "%s/%s" % (src_tmp_dir, os.path.basename(fn)))
  print("All source files:")
  subprocess.check_call(["ls", src_tmp_dir])
  print("travis_fold:end:script.prepare")
  return src_tmp_dir


def run_inspect(pycharm_dir, src_dir):
  """
  :param str pycharm_dir:
  :param str src_dir:
  :return: list of xml files
  :rtype: list[str]
  """
  out_tmp_dir = tempfile.mkdtemp()

  print("travis_fold:start:script.inspect")
  # Note: Will not run if PyCharm is already running.
  # Maybe we can find some workaround for this?
  # See here: https://stackoverflow.com/questions/55339010/run-pycharm-inspect-sh-even-if-pycharm-is-already-running
  # And here: https://github.com/albertz/pycharm-inspect
  # Also: https://stackoverflow.com/questions/55323910/pycharm-code-style-check-via-command-line
  cmd = [
    "%s/bin/inspect.sh" % pycharm_dir,
    src_dir,
    "%s/PyCharm-inspection-profile.xml" % my_dir,
    out_tmp_dir,
    "-v2"]
  print("$ %s" % " ".join(cmd))
  subprocess.check_call(cmd, stderr=subprocess.STDOUT)
  print("travis_fold:end:script.inspect")

  fs = list(glob(out_tmp_dir + "/*.xml"))
  assert fs
  return fs


def report_inspect_xml(fn):
  """
  :param str fn:
  """
  print("travis_fold:start:inspect.%s" % os.path.basename(fn))
  print("File %s:" % fn)
  # Example PyPackageRequirementsInspection.xml:
  """
  <problems is_local_tool="true">
  <problem>
    <file>file://$PROJECT_DIR$/TFUtil.py</file>
    <line>1</line>
    <module>returnn</module>
    <entry_point TYPE="file" FQNAME="file://$PROJECT_DIR$/TFUtil.py" />
    <problem_class severity="WARNING" attribute_key="WARNING_ATTRIBUTES">Package requirements</problem_class>
    <description>Package requirements 'h5py', 'theano==0.9' are not satisfied</description>
  </problem>
  </problems>
  """
  inspect_class = os.path.splitext(os.path.basename(fn))[0]  # e.g. "PyPackageRequirementsInspection"
  import xml.etree.ElementTree as et
  root = et.parse(fn).getroot()
  assert isinstance(root, et.Element)
  assert root.tag == "problems"
  for problem in root.findall("./problem"):
    assert isinstance(problem, et.Element)
    assert problem.tag == "problem"
    filename = problem.find("./file").text.strip()
    if filename.startswith("file://$PROJECT_DIR$/"):
      filename = filename[len("file://$PROJECT_DIR$/"):]
    line = int(problem.find("./line").text.strip())
    problem_severity = problem.find("./problem_class").attrib["severity"]
    description = problem.find("./description").text.strip()
    print("%s:%i: %s %s: %s" % (filename, line, problem_severity, inspect_class, description))
  print("travis_fold:end:inspect.%s" % os.path.basename(fn))


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--xml")
  arg_parser.add_argument("--pycharm")
  arg_parser.add_argument("--files", nargs="*")
  args = arg_parser.parse_args()

  if args.xml:
    report_inspect_xml(args.xml)
    return

  if args.pycharm:
    pycharm_dir = args.pycharm
    check_pycharm_dir(pycharm_dir)
  else:
    pycharm_dir = install_pycharm()
  src_dir = prepare_src_dir(files=args.files)
  fs = run_inspect(pycharm_dir=pycharm_dir, src_dir=src_dir)
  for fn in fs:
    # noinspection PyBroadException
    try:
      report_inspect_xml(fn)
    except Exception:
      sys.excepthook(*sys.exc_info())
      # Go on.


if __name__ == "__main__":
  main()
