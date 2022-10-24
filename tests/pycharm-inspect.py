#!/usr/bin/env python3

"""
Will use the PyCharm code inspection.

See here:
  https://github.com/albertz/pycharm-inspect
  https://stackoverflow.com/questions/55323910/pycharm-code-style-check-via-command-line
  https://youtrack.jetbrains.com/issue/PY-34863
  https://youtrack.jetbrains.com/issue/PY-34864
"""

import os
import sys
import re
import time
import shutil
import subprocess
import tempfile
import typing
from glob import glob
import argparse
from xml.dom import minidom
from xml.etree import ElementTree

my_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(my_dir)
sys.path.insert(0, root_dir)
os.chdir(root_dir)

from returnn.util import better_exchook  # noqa
from returnn.util.basic import pip_install, which_pip, pip_check_is_installed, hms  # noqa

travis_env = os.environ.get("TRAVIS") == "true"
github_env = os.environ.get("GITHUB_ACTIONS") == "true"

gray_color = "black"  # black is usually gray
if github_env:
  gray_color = "white"  # black is black, on black background. so just use white


class _StdoutTextFold:
  def __init__(self, name):
    """
    :param str name:
    """
    self.name = name
    self.start_time = time.time()

    if github_env:
      # https://github.community/t/has-github-action-somthing-like-travis-fold/16841
      if not folds:  # nested folds not supported, https://github.com/actions/toolkit/issues/112
        print("::group::%s" % name)

    if travis_env:
      # travis_fold: https://github.com/travis-ci/travis-ci/issues/1065
      print("travis_fold:start:%s" % name)

    sys.stdout.flush()

  def finish(self):
    """
    End fold.
    """
    elapsed_time = time.time() - self.start_time
    print("%s: Elapsed time: %s" % (self.name, hms(elapsed_time)))

    if travis_env:
      print("travis_fold:end:%s" % folds[-1])

    if github_env:
      if len(folds) == 1:
        print("::endgroup::")

    sys.stdout.flush()


folds = []  # type: typing.List[_StdoutTextFold]


def fold_start(name):
  """
  :param str name:
  """
  folds.append(_StdoutTextFold(name))


def fold_end():
  """
  Ends the fold.
  """
  assert folds
  folds[-1].finish()
  folds.pop(-1)


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
  fold_start("script.install")
  print("travis_fold:start:script.install")
  install_dir = tempfile.mkdtemp()
  pycharm_dir = "%s/pycharm" % install_dir
  print("Install PyCharm into:", pycharm_dir)
  sys.stdout.flush()

  pycharm_version = (2020, 2)
  name = "pycharm-community-%i.%i" % pycharm_version
  fn = "%s.tar.gz" % name

  subprocess.check_call(
    ["wget", "--progress=dot:mega",
     "-c", "https://download.jetbrains.com/python/%s" % fn],
    cwd=install_dir,
    stderr=subprocess.STDOUT)
  tar_out = subprocess.check_output(
    ["tar", "-xzvf", fn],
    cwd=install_dir,
    stderr=subprocess.STDOUT)
  print((b"\n".join(tar_out.splitlines()[-10:])).decode("utf8"))
  assert os.path.isdir("%s/%s" % (install_dir, name))
  os.remove("%s/%s" % (install_dir, fn))
  os.rename("%s/%s" % (install_dir, name), pycharm_dir)
  check_pycharm_dir(pycharm_dir)

  fold_end()
  return pycharm_dir


def get_version_str_from_pycharm(pycharm_dir):
  """
  :param str pycharm_dir:
  :return: e.g. "CE2018.3"
  :rtype: str
  """
  import re
  import json
  if os.path.exists("%s/product-info.json" % pycharm_dir):
    d = json.load(open("%s/product-info.json" % pycharm_dir))
    name = d["dataDirectoryName"]
    assert isinstance(name, str)
    assert name.startswith("PyCharm")
    return name[len("PyCharm"):]
  # This works on PyCharm 2019.
  code = open("%s/bin/pycharm.sh" % pycharm_dir).read()
  m = re.search("-Didea\\.paths\\.selector=PyCharm(\\S+) ", code)
  assert m, "pycharm %r not as expected" % pycharm_dir
  return m.group(1)


def parse_pycharm_version(version_str):
  """
  :param str version_str: e.g. "CE2018.3"
  :rtype: ((int,int),str)
  :return: e.g. (2018,3),"CE"
  """
  name = ""
  if version_str.startswith("CE"):
    name = "CE"
    version_str = version_str[2:]
  assert version_str.startswith("2")
  version_str_parts = version_str.split(".")
  assert len(version_str_parts) == 2, "version %r" % version_str
  return tuple([int(p) for p in version_str_parts]), name


def create_stub_dir(pycharm_dir, stub_dir, pycharm_major_version):
  """
  :param str pycharm_dir:
  :param str stub_dir:
  :param int pycharm_major_version:
  """
  fold_start("script.create_python_stubs")
  print("Generating Python stubs via helpers/generator3.py...")
  if pycharm_major_version >= 2020:
    generator_path = "%s/plugins/python-ce/helpers/generator3/__main__.py" % pycharm_dir
    assert os.path.exists(generator_path)
    cmd = [sys.executable, generator_path, "-d", stub_dir]
    # The stdout can sometimes be very long. Thus we pipe and filter it a bit.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = proc.communicate()
    if proc.returncode != 0:
      raise subprocess.CalledProcessError(returncode=proc.returncode, cmd=cmd, output=stdout)
    for line in stdout.splitlines():
      line = line.decode("utf8")
      if len(line) < 240:
        print(line)
      else:
        print(line[:240] + "...")
  elif pycharm_major_version <= 2019:
    generator_path = "%s/helpers/generator3.py" % pycharm_dir
    assert os.path.exists(generator_path)
    subprocess.check_call([sys.executable, generator_path, "-d", stub_dir, "-b"])
    print("Collecting further native modules...")
    sys.stdout.flush()
    mod_names = []
    for line in subprocess.check_output([
          sys.executable, generator_path, "-L"]).decode("utf8").splitlines()[1:]:
      # First line is version, so we skipped those.
      # Then we get sth like "<module name> <other things>...".
      assert isinstance(line, str)
      mod_name = line.split()[0]
      # There are duplicates. Ignore.
      if mod_name not in mod_names:
        mod_names.append(mod_name)
    for mod_name in mod_names:
      print("Generate for %r." % mod_name)
      sys.stdout.flush()
      # Ignore errors here.
      subprocess.call([sys.executable, generator_path, "-d", stub_dir, mod_name])
  fold_end()


_use_stub_zip = False


def setup_pycharm_python_interpreter(pycharm_dir):
  """
  Unfortunately, the headless PyCharm bin/inspect will use the global PyCharm settings,
  and requires that we have a Python interpreter set up,
  with the same name as we use in our `.idea` settings, which we will link in :func:`prepare_src_dir`.
  See here: https://youtrack.jetbrains.com/issue/PY-34864

  Our current way to work around this: We create (or extend) the file
  ``~/.PyCharm<VERSION>/config/options/jdk.table.xml`` such that it has the right Python interpreter.

  :param str pycharm_dir:
  """
  fold_start("script.setup_pycharm_python_interpreter")
  print("Setup PyCharm Python interpreter... (jdk.table.xml)")
  print("Current Python:", sys.executable, sys.version, sys.version_info)
  name = "Python 3 (.../bin/python3)"  # used in our PyCharm.idea. this should match.
  pycharm_version_str = get_version_str_from_pycharm(pycharm_dir)
  pycharm_version, pycharm_version_name = parse_pycharm_version(pycharm_version_str)
  if sys.platform == "darwin":
    pycharm_config_dir = os.path.expanduser("~/Library/Preferences/PyCharm%s" % pycharm_version_str)
    pycharm_system_dir = os.path.expanduser("~/Library/Caches/PyCharm%s" % pycharm_version_str)
  else:  # assume Linux/Unix
    if pycharm_version[0] >= 2020:
      pycharm_config_dir = os.path.expanduser("~/.config/JetBrains/PyCharm%s" % pycharm_version_str)
      pycharm_system_dir = os.path.expanduser("~/.cache/JetBrains/PyCharm%s" % pycharm_version_str)
    else:  # <= 2020
      pycharm_config_dir = os.path.expanduser("~/.PyCharm%s/config" % pycharm_version_str)
      pycharm_system_dir = os.path.expanduser("~/.PyCharm%s/system" % pycharm_version_str)

  # I just zipped the stubs from my current installation on Linux.
  # Maybe we can also reuse these stubs for other PyCharm versions, or even other Python versions.
  if _use_stub_zip:
    stub_base_name = "pycharm2018.3-python3.6-stubs"
    stub_fn = "%s/python_stubs/%s.zip" % (pycharm_system_dir, stub_base_name)
    stub_dir = "%s/python_stubs/%s" % (pycharm_system_dir, stub_base_name)
    os.makedirs(os.path.dirname(stub_fn), exist_ok=True)
    if os.path.exists(stub_dir):
      print("Python stubs dir exists already:", stub_dir)
    else:
      if not os.path.exists(stub_fn):
        subprocess.check_call([
          "wget",
          "https://www-i6.informatik.rwth-aachen.de/web/Software/returnn/%s.zip" % stub_base_name],
          cwd=os.path.dirname(stub_fn))
      assert os.path.exists(stub_fn)
      subprocess.check_call(
        ["unzip", "%s.zip" % stub_base_name, "-d", stub_base_name],
        cwd=os.path.dirname(stub_fn))
      assert os.path.isdir(stub_dir)
  else:
    fold_start("script.opt_install_further_py_deps")
    if not pip_check_is_installed("tensorflow") and not pip_check_is_installed("tensorflow-gpu"):
      pip_install("tensorflow")
    # Note: Horovod will usually fail to install in this env.
    for pkg in ["typing", "librosa==0.8.1", "PySoundFile", "nltk", "matplotlib", "mpi4py", "pycodestyle"]:
      if not pip_check_is_installed(pkg):
        try:
          pip_install(pkg)
        except subprocess.CalledProcessError as exc:
          print("Pip install failed:", exc)
          print("Ignore...")
    fold_end()

    stub_dir = "%s/python_stubs/python%s-generated" % (
      pycharm_system_dir, "%i.%i.%i" % sys.version_info[:3])
    if os.path.exists(stub_dir):
      print("Python stubs already exists, not recreating (%s)" % stub_dir)
    else:
      print("Generate stub dir:", stub_dir)
      os.makedirs(stub_dir)
      create_stub_dir(pycharm_dir=pycharm_dir, stub_dir=stub_dir, pycharm_major_version=pycharm_version[0])

  jdk_table_fn = "%s/options/jdk.table.xml" % pycharm_config_dir
  print("Filename:", jdk_table_fn)
  os.makedirs(os.path.dirname(jdk_table_fn), exist_ok=True)

  if os.path.exists(jdk_table_fn):
    print("Loading existing jdk.table.xml.")
    et = ElementTree.parse(jdk_table_fn)
    root = et.getroot()
    assert isinstance(root, ElementTree.Element)
    jdk_collection = root.find("./component")
    assert isinstance(jdk_collection, ElementTree.Element)
    assert jdk_collection.tag == "component" and jdk_collection.attrib["name"] == "ProjectJdkTable"
  else:
    print("Creating new jdk.table.xml.")
    root = ElementTree.Element("application")
    et = ElementTree.ElementTree(root)
    jdk_collection = ElementTree.SubElement(root, "component", name="ProjectJdkTable")
    assert isinstance(jdk_collection, ElementTree.Element)

  existing_jdk = jdk_collection.find("./jdk/name[@value='%s']/.." % name)
  if existing_jdk:
    print("Found existing Python interpreter %r. Remove and recreate." % name)
    assert isinstance(existing_jdk, ElementTree.Element)
    assert existing_jdk.find("./name").attrib["value"] == name
    jdk_collection.remove(existing_jdk)

  # Example content:
  """
  <application>
  <component name="ProjectJdkTable">
    <jdk version="2">
      <name value="Python 2.7.3 (/usr/bin/python2.7)" />
      <type value="Python SDK" />
      <version value="Python 2.7.12" />
      <homePath value="/usr/bin/python2.7" />
      <roots>
        <classPath>
          <root type="composite">
            <root url="file:///usr/bin" type="simple" />
            ...
          </root>
        </classPath>
        <sourcePath>
          <root type="composite" />
        </sourcePath>
      </roots>
      <additional />
    </jdk>
  </component>
  </application>
  """

  jdk_entry = ElementTree.SubElement(jdk_collection, "jdk", version="2")
  ElementTree.SubElement(jdk_entry, "name", value=name)
  ElementTree.SubElement(jdk_entry, "type", value="Python SDK")
  ElementTree.SubElement(jdk_entry, "version", value="Python %i.%i.%i" % sys.version_info[:3])
  ElementTree.SubElement(jdk_entry, "homePath", value=sys.executable)
  paths_root = ElementTree.SubElement(jdk_entry, "roots")
  classes_paths = ElementTree.SubElement(ElementTree.SubElement(paths_root, "classPath"), "root", type="composite")
  relevant_paths = list(sys.path)
  if root_dir in relevant_paths:
    relevant_paths.remove(root_dir)
  if my_dir in relevant_paths:
    relevant_paths.remove(my_dir)
  relevant_paths.extend([
    stub_dir,
    "$APPLICATION_HOME_DIR$/helpers/python-skeletons",
    "$APPLICATION_HOME_DIR$/helpers/typeshed/stdlib/3",
    "$APPLICATION_HOME_DIR$/helpers/typeshed/stdlib/2and3",
    "$APPLICATION_HOME_DIR$/helpers/typeshed/third_party/3",
    "$APPLICATION_HOME_DIR$/helpers/typeshed/third_party/2and3"
  ])
  # Maybe also add Python stubs path? How to generate them?
  for path in relevant_paths:
    ElementTree.SubElement(classes_paths, "root", url="file://%s" % path, type="simple")
  ElementTree.SubElement(ElementTree.SubElement(paths_root, "sourcePath"), "root", type="composite")
  ElementTree.SubElement(jdk_entry, "additional")

  print("Save XML.")
  et.write(jdk_table_fn, encoding="UTF-8")

  fold_start("script.jdk_table")
  print("XML content:")
  rough_string = ElementTree.tostring(root, 'utf-8')
  print(minidom.parseString(rough_string).toprettyxml(indent="  "))
  fold_end()

  fold_end()


def read_spelling_dict():
  """
  :rtype: list[str]
  """
  return open("%s/spelling.dic" % my_dir).read().splitlines()


def create_spelling_dict_xml(src_dir):
  """
  Need to create this on-the-fly for the current user.
  """
  # Example:
  """
  <component name="ProjectDictionaryState">
  <dictionary name="az">
    <words>
      <w>dtype</w>
      <w>idxs</w>
      <w>keepdims</w>
      ...
    </words>
  </dictionary>
  </component>
  """
  from returnn.util.basic import get_login_username
  user_name = get_login_username()
  root = ElementTree.Element("component", name="ProjectDictionaryState")
  dict_ = ElementTree.SubElement(root, "dictionary", name=user_name)
  words = ElementTree.SubElement(dict_, "words")
  for w in read_spelling_dict():
    ElementTree.SubElement(words, "w").text = w
  et = ElementTree.ElementTree(root)
  print("Save XML.")
  xml_filename = "%s/.idea/dictionaries/%s.xml" % (src_dir, user_name)
  os.makedirs(os.path.dirname(xml_filename), exist_ok=True)
  et.write(xml_filename, encoding="UTF-8")


def prepare_src_dir(files=None):
  """
  New clean source dir, where we symlink only the relevant src files.

  :param list[str]|None files:
  :return: src dir
  :rtype: str
  """
  fold_start("script.prepare")
  print("Prepare project source files...")
  if not files:
    files = ["returnn", "tools", "demos", "rnn.py", "setup.py", "__init__.py"]
  src_tmp_dir = "%s/returnn" % tempfile.mkdtemp()
  os.mkdir(src_tmp_dir)
  shutil.copytree("%s/PyCharm.idea" % my_dir, "%s/.idea" % src_tmp_dir, symlinks=True)
  for fn in files:
    fn = "%s/%s" % (root_dir, fn)
    dst = "%s/%s" % (src_tmp_dir, os.path.basename(fn))
    if os.path.isdir(fn):
      shutil.copytree(fn, dst, symlinks=True)
    else:
      shutil.copy(fn, dst)
  create_spelling_dict_xml(src_tmp_dir)
  print("All source files:")
  sys.stdout.flush()
  subprocess.check_call(["ls", "-la", src_tmp_dir])
  fold_end()
  return src_tmp_dir


def run_inspect(pycharm_dir, src_dir, skip_pycharm_inspect=False):
  """
  :param str pycharm_dir:
  :param str src_dir:
  :param bool skip_pycharm_inspect:
  :return: dir of xml files
  :rtype: str
  """
  out_tmp_dir = tempfile.mkdtemp()

  fold_start("script.inspect")
  if not skip_pycharm_inspect:
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

  # PyCharm does not do PEP8 code style checks by itself but uses the (bundled) pycodestyle tool.
  # https://youtrack.jetbrains.com/issue/PY-43901
  # Do that now.
  root = ElementTree.Element("problems")
  from lint_common import find_all_py_source_files
  for py_src_file in find_all_py_source_files():
    ignore_codes = "E121,E123,E126,E226,E24,E704,W503,W504"  # PyCharm defaults
    indent_size = 2  # default for RETURNN
    if py_src_file.endswith("/better_exchook.py"):
      indent_size = 4
    cmd = [
      "pycodestyle",
      py_src_file,
      "--ignore=%s" % ignore_codes,
      "--indent-size=%i" % indent_size,
      "--max-line-length=120"]
    print("$ %s" % " ".join(cmd))
    sys.stdout.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout, _ = proc.communicate()
    # We do not check returncode, as this is always non-zero if there is any warning.
    for line in stdout.decode("utf8").splitlines():
      # Example line: demos/demo-record-and-push-to-webserver.py:48:1: E302 expected 2 blank lines, found 1
      m = re.match("^(.*):([0-9]+):([0-9]+): ([EW][0-9]+) (.+)$", line)
      assert m, "unexpected line %r" % line
      fn_, line_nr, col_nr, warn_id, description = m.groups()
      assert fn_ == py_src_file, "unexpected line %r" % line
      line_nr, col_nr = int(line_nr), int(col_nr)
      description = "%s: %s" % (warn_id, description)
      prob = ElementTree.SubElement(root, "problem")
      # Note: We do not aim to have this complete. This is just enough such that report_inspect_xml can read it.
      ElementTree.SubElement(prob, "file").text = "file://$PROJECT_DIR$/%s" % py_src_file
      ElementTree.SubElement(prob, "line").text = str(line_nr)
      ElementTree.SubElement(prob, "offset").text = str(col_nr)
      ElementTree.SubElement(prob, "problem_class", severity="WEAK WARNING", id=warn_id).text = description
      ElementTree.SubElement(prob, "description").text = description
  et = ElementTree.ElementTree(root)
  et.write("%s/Pep8CodeStyle.xml" % out_tmp_dir, encoding="UTF-8")

  fold_end()
  return out_tmp_dir


def report_inspect_xml(fn):
  """
  :param str fn:
  :return: list of (filename, line, problem_severity, inspect_class, description)
  :rtype: list[(str,int,str,str,str)]
  """
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
  root = ElementTree.parse(fn).getroot()
  assert isinstance(root, ElementTree.Element)
  assert root.tag == "problems"
  result = []
  for problem in root.findall("./problem"):
    assert isinstance(problem, ElementTree.Element)
    assert problem.tag == "problem"
    filename = problem.find("./file").text.strip()
    if filename.startswith("file://$PROJECT_DIR$/"):
      filename = filename[len("file://$PROJECT_DIR$/"):]
    line = int(problem.find("./line").text.strip())
    problem_severity = problem.find("./problem_class").attrib["severity"]
    description = problem.find("./description").text.strip()

    # Do some filtering for false positives. This is ugly, but the other solution would be to ignore all of them.
    possible_false_positive = False
    if inspect_class == "PyArgumentListInspection" and "'d0' unfilled" in description:  # Numpy false positive
      possible_false_positive = True
    if inspect_class == "PyArgumentListInspection" and "'d1' unfilled" in description:  # Numpy false positive
      possible_false_positive = True
    if inspect_class == "PyArgumentListInspection" and "'self' unfilled" in description:  # Numpy false positive
      possible_false_positive = True
    if inspect_class == "PyStringFormatInspection" and "Unexpected type None" in description:
      possible_false_positive = True
    if possible_false_positive:
      problem_severity = "POSSIBLE-FALSE %s" % problem_severity

    result.append((filename, line, problem_severity, inspect_class, description))

  return result


def report_inspect_dir(inspect_xml_dir,
                       inspect_class_blacklist=None, inspect_class_not_counted=None,
                       ignore_count_for_files=()):
  """
  :param str inspect_xml_dir:
  :param set[str]|None inspect_class_blacklist:
  :param set[str]|None inspect_class_not_counted:
  :param set[str]|tuple[str]|None ignore_count_for_files:
  :return: count of reports
  :rtype: int
  """
  if os.path.isfile(inspect_xml_dir):
    assert inspect_xml_dir.endswith(".xml")
    inspect_xml_files = [inspect_xml_dir]
  else:
    assert os.path.isdir(inspect_xml_dir)
    inspect_xml_files = list(glob(inspect_xml_dir + "/*.xml"))
    assert inspect_xml_files

  inspections = []
  for fn in inspect_xml_files:
    inspections.extend(report_inspect_xml(fn))
  inspections.sort()
  inspections.append((None, None, None, None, None))  # final marker

  # copy
  inspect_class_blacklist = set(inspect_class_blacklist or ())
  inspect_class_not_counted = set(inspect_class_not_counted or ())

  # maybe update inspect_class_not_counted
  from lint_common import find_all_py_source_files
  returnn_py_source_files = set(find_all_py_source_files())
  all_files = set()
  relevant_inspections_for_file = set()
  explicitly_ignored_files = ignore_count_for_files
  ignore_count_for_files = set(ignore_count_for_files)
  for filename, line, problem_severity, inspect_class, description in inspections:
    all_files.add(filename)
    if filename not in returnn_py_source_files:
      continue
    if inspect_class in inspect_class_blacklist:
      continue
    if inspect_class in inspect_class_not_counted:
      continue
    if problem_severity.startswith("POSSIBLE-FALSE "):
      continue
    relevant_inspections_for_file.add(filename)
  for filename in all_files:
    if filename not in relevant_inspections_for_file:
      ignore_count_for_files.add(filename)

  print("Reporting individual files. We skip all files which have no warnings at all.")
  color = better_exchook.Color()
  total_relevant_count = 0
  file_count = None
  last_filename = None
  for filename, line, problem_severity, inspect_class, description in inspections:
    if filename and filename not in returnn_py_source_files:
      continue  # for now, to not spam Travis too much
    if inspect_class in inspect_class_blacklist:
      continue

    if filename != last_filename:
      if last_filename:
        if last_filename in explicitly_ignored_files:
          msg = color.color("This file is on the ignore list.", color=gray_color)
        elif last_filename not in returnn_py_source_files:
          msg = color.color("This file is not part of the official RETURNN Python source code.", color=gray_color)
        elif last_filename in ignore_count_for_files:
          msg = color.color("The inspection reports for this file are all non critical.", color=gray_color)
        else:
          msg = color.color("The inspection reports for this file are fatal!", color="red")
        print(msg)
        fold_end()
      if filename:
        file_msg = color.color(
          "File: %s" % filename, color=gray_color if filename in ignore_count_for_files else "red")
        if github_env:
          fold_start(file_msg)
        else:
          fold_start("inspect.%s" % filename)
          print(file_msg)
      last_filename = filename
      file_count = 0
    if not filename:
      continue
    if filename in ignore_count_for_files and file_count >= 10:
      if file_count == 10:
        print("... (further warnings skipped)")
      file_count += 1
      continue

    msg = "%s:%i: %s %s: %s" % (filename, line, problem_severity, inspect_class, description)
    msg_counted = True
    if inspect_class in inspect_class_not_counted:
      msg_counted = False
    if problem_severity.startswith("POSSIBLE-FALSE "):
      msg_counted = False
    if msg_counted:
      print(color.color(msg, color="red"))
      if filename not in ignore_count_for_files:
        total_relevant_count += 1
    else:
      print(color.color(msg, color=gray_color))
    file_count += 1

  print("Total relevant inspection reports:", total_relevant_count)
  return total_relevant_count


def main():
  """
  Main entry point for this script.
  """
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--xml")
  arg_parser.add_argument("--pycharm")
  arg_parser.add_argument("--setup_pycharm_only", action="store_true")
  arg_parser.add_argument("--skip_setup_pycharm", action="store_true")
  arg_parser.add_argument("--skip_pycharm_inspect", action="store_true", help="only PEP8")
  arg_parser.add_argument("--files", nargs="*")
  args = arg_parser.parse_args()

  from lint_common import ignore_count_for_files
  inspect_kwargs = dict(
    inspect_class_blacklist={
    },
    inspect_class_not_counted={
      # Here we disable more than what you would do in the IDE.
      # The aim is that any left over warnings are always indeed important and should be fixed.

      # False alarms.
      "PyTypeCheckerInspection",  # too much false alarms: https://youtrack.jetbrains.com/issue/PY-34893

      # Not critical.
      "SpellCheckingInspection",  # way too much for now...
      "PyClassHasNoInitInspection",  # not relevant?
      "PyMethodMayBeStaticInspection",  # not critical
    },
    ignore_count_for_files=ignore_count_for_files)

  if args.xml:
    if report_inspect_dir(args.xml, **inspect_kwargs) > 0:
      sys.exit(1)
    return

  if args.pycharm:
    pycharm_dir = args.pycharm
    check_pycharm_dir(pycharm_dir)
  else:
    pycharm_dir = install_pycharm()

  if not args.skip_setup_pycharm and not args.skip_pycharm_inspect:
    setup_pycharm_python_interpreter(pycharm_dir=pycharm_dir)
  if args.setup_pycharm_only:
    return

  src_dir = prepare_src_dir(files=args.files)
  res_dir = run_inspect(pycharm_dir=pycharm_dir, src_dir=src_dir, skip_pycharm_inspect=args.skip_pycharm_inspect)
  if report_inspect_dir(res_dir, **inspect_kwargs) > 0:
    sys.exit(1)


if __name__ == "__main__":
  better_exchook.install()
  main()
