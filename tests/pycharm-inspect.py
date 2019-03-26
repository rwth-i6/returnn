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
  os.symlink("%s/PyCharm.idea" % my_dir, "%s/.idea" % src_tmp_dir)
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
  :return: dir of xml files
  :rtype: str
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
  import xml.etree.ElementTree as ElementTree
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
    result.append((filename, line, problem_severity, inspect_class, description))
  return result


def report_inspect_dir(path, inspect_class_whitelist=None, inspect_class_blacklist=None, ignore_count_for_files=()):
  """
  :param str path:
  :param set[str]|None inspect_class_whitelist:
  :param set[str]|None inspect_class_blacklist:
  :param set[str]|tuple[str]|None ignore_count_for_files:
  :return: count of reports
  :rtype: int
  """
  if os.path.isfile(path):
    assert path.endswith(".xml")
    fs = [path]
  else:
    assert os.path.isdir(path)
    fs = list(glob(path + "/*.xml"))
    assert fs

  inspections = []
  for fn in fs:
    inspections.extend(report_inspect_xml(fn))
  inspections.sort()
  inspections.append((None, None, None, None, None))  # final marker

  color = better_exchook.Color()
  total_count = 0
  file_count = None
  last_filename = None
  for filename, line, problem_severity, inspect_class, description in inspections:
    if inspect_class_whitelist is not None and inspect_class not in inspect_class_whitelist and inspect_class:
      continue
    if inspect_class_blacklist is not None and inspect_class in inspect_class_blacklist:
      continue

    if filename != last_filename:
      if last_filename:
        if file_count == 0:
          print("The inspection reports for this file are currently ignored.")
        else:
          print(color.color("The inspection reports for this file are fatal!", color="red"))
        print("travis_fold:end:inspect.%s" % last_filename)
      if filename:
        print("travis_fold:start:inspect.%s" % filename)
        print(color.color(
          "File: %s" % filename, color="black" if filename in ignore_count_for_files else "red"))
        last_filename = filename
        file_count = 0
    if not filename:
      continue

    print("%s:%i: %s %s: %s" % (filename, line, problem_severity, inspect_class, description))
    if filename not in ignore_count_for_files:
      total_count += 1
      file_count += 1

  return total_count


def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--xml")
  arg_parser.add_argument("--pycharm")
  arg_parser.add_argument("--files", nargs="*")
  args = arg_parser.parse_args()

  inspect_kwargs = dict(
    inspect_class_blacklist={
      "PyInterpreterInspection",  # TODO how to select this in PyCharm.idea?
      "SpellCheckingInspection",  # way too much for now... TODO this should be fixed later, probably in PyCharm.idea
      "PyClassHasNoInitInspection",  # not relevant?
    },
    # Proceed like this: Fix all warnings for some file, then remove it from this list.
    # I commented out the files which really should not have warnings (mostly the TF backend + shared files).
    ignore_count_for_files={
      'ActivationFunctions.py',
      'BestPathDecoder.py',
      'BundleFile.py',
      'CTC.py',
      # 'CachedDataset.py',
      'CachedDataset2.py',
      # 'Config.py',
      'CustomLSTMFunctions.py',
      # 'Dataset.py',
      # 'Debug.py',
      # 'DebugHelpers.py',
      'Device.py',
      'Engine.py',
      # 'EngineBatch.py',
      'EngineTask.py',
      # 'EngineUtil.py',
      'External.py',
      # 'Fsa.py',
      'FunctionLoader.py',
      # 'GeneratingDataset.py',
      'HDFDataset.py',
      # 'HyperParamTuning.py',
      'Inv.py',
      # 'LearningRateControl.py',
      # 'LmDataset.py',
      # 'Log.py',
      # 'MetaDataset.py',
      # 'MultiBatchBeam.py',
      # 'NativeOp.py',
      'Network.py',
      'NetworkBaseLayer.py',
      'NetworkCNNLayer.py',
      'NetworkCopyUtils.py',
      'NetworkCtcLayer.py',
      'NetworkDescription.py',
      'NetworkHiddenLayer.py',
      'NetworkLayer.py',
      'NetworkLstmLayer.py',
      'NetworkOutputLayer.py',
      'NetworkRecurrentLayer.py',
      'NetworkStream.py',
      'NetworkTwoDLayer.py',
      'NormalizationData.py',
      # 'NumpyDumpDataset.py',
      'OpBLSTM.py',
      'OpInvAlign.py',
      'OpLSTM.py',
      'OpLSTMCell.py',
      'OpLSTMCustom.py',
      'OpLSTMRec.py',
      'OpNumpyAlign.py',
      # 'Pretrain.py',
      # 'RawWavDataset.py',
      'RecurrentTransform.py',
      'Server.py',
      # 'SprintCache.py',
      # 'SprintControl.py',
      # 'SprintDataset.py',
      # 'SprintErrorSignals.py',
      # 'SprintExternInterface.py',
      # 'SprintInterface.py',
      'StereoDataset.py',
      # 'TFDataPipeline.py',
      # 'TFEngine.py',
      # 'TFKenLM.py',
      # 'TFNativeOp.py',
      # 'TFNetwork.py',
      # 'TFNetworkLayer.py',
      'TFNetworkNeuralTransducer.py',
      # 'TFNetworkRecLayer.py',
      'TFNetworkSegModLayer.py',
      'TFNetworkSigProcLayer.py',
      # 'TFUpdater.py',
      # 'TFUtil.py',
      'TaskSystem.py',
      'TaskSystem_example.py',
      'TheanoUtil.py',
      'TorchWrapper.py',
      'TwoStateBestPathDecoder.py',
      'TwoStateHMMOp.py',
      'Updater.py',
      # 'Util.py',
      # '__init__.py',
      # 'better_exchook.py',
      # 'rnn.py',
    })

  if args.xml:
    if report_inspect_dir(args.xml, **inspect_kwargs) > 0:
      sys.exit(1)
    return

  if args.pycharm:
    pycharm_dir = args.pycharm
    check_pycharm_dir(pycharm_dir)
  else:
    pycharm_dir = install_pycharm()
  src_dir = prepare_src_dir(files=args.files)
  res_dir = run_inspect(pycharm_dir=pycharm_dir, src_dir=src_dir)
  if report_inspect_dir(res_dir, **inspect_kwargs) > 0:
    sys.exit(1)


if __name__ == "__main__":
  main()
