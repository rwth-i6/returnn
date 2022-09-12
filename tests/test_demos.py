
from __future__ import print_function

import _setup_test_env  # noqa
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import re
import os
import sys
from glob import glob
import unittest
from nose.tools import assert_less, assert_in
from returnn.util import better_exchook
from returnn.util.basic import which_pip


py = sys.executable
print("Python:", py)

os.chdir((os.path.dirname(__file__) or ".") + "/..")
assert os.path.exists("rnn.py")


def build_env(env_update=None):
  """
  :param dict[str,str]|None env_update:
  :return: env dict for Popen
  :rtype: dict[str,str]
  """
  env_update_ = os.environ.copy()
  if env_update:
    env_update_.update(env_update)
  return env_update_


def run(*args, env_update=None):
  args = list(args)
  print("run:", args)
  # RETURNN by default outputs on stderr, so just merge both together
  p = Popen(args, stdout=PIPE, stderr=STDOUT, env=build_env(env_update=env_update))
  out, _ = p.communicate()
  if p.returncode != 0:
    print("Return code is %i" % p.returncode)
    print("std out/err:\n---\n%s\n---\n" % out.decode("utf8"))
    raise CalledProcessError(cmd=args, returncode=p.returncode, output=out)
  return out.decode("utf8")


def run_and_parse_last_fer(*args, **kwargs):
  out = run(*args, **kwargs)
  parsed_fer = None
  for line in out.splitlines():
    # example: epoch 5 score: 0.0231807245472 elapsed: 0:00:04 dev: score 0.0137521058997 error 0.00268961807423
    m = re.match("epoch [0-9]+ score: .* dev: .* error ([0-9.]+)\\s?", line)
    if not m:
      # example: dev: score 0.03350000149202181 error 0.009919877954075871
      m = re.match("dev: score .* error ([0-9.]+)\\s?", line)
    if not m:
      continue
    parsed_fer = float(m.group(1))
  err_msg = "ERROR: No epoch dev errors found in output"
  assert parsed_fer is not None, "%s.\nOutput:\n\n%s\n\n%s." % (err_msg, out, err_msg)
  return parsed_fer


def run_config_get_fer(config_filename, env_update=None):
  cleanup_tmp_models(config_filename)
  fer = run_and_parse_last_fer(
    py, "rnn.py", config_filename, "++log_verbosity", "5", env_update=env_update)
  print("FER:", fer)
  cleanup_tmp_models(config_filename)
  return fer


def cleanup_tmp_models(config_filename):
  assert os.path.exists(config_filename)
  from returnn.config import Config
  config = Config()
  config.load_file(config_filename)
  model_filename = config.value('model', '')
  assert model_filename
  # Remove existing models
  assert model_filename.startswith("/tmp/")
  for f in glob(model_filename + ".*"):
    os.remove(f)


def test_demo_tf_task12ax():
  fer = run_config_get_fer("demos/demo-tf-native-lstm.12ax.config")
  assert_less(fer, 0.01)


def test_demo_tf_task12ax_no_test_env():
  fer = run_config_get_fer("demos/demo-tf-native-lstm2.12ax.config", env_update={"RETURNN_TEST": ""})
  assert_less(fer, 0.01)


def test_demo_iter_dataset_task12ax():
  cleanup_tmp_models("demos/demo-theano-task12ax.config")
  # While the config is for Theano, it should not matter for demo-iter-dataset
  out = run(py, "demos/demo-iter-dataset.py", "demos/demo-theano-task12ax.config")
  assert_in("Epoch 5.", out.splitlines())


def test_demo_returnn_as_framework():
  print("Prepare.")
  import subprocess
  import shutil
  from glob import glob
  from returnn.util.basic import get_login_username
  # echo via subprocess, because this stdout as well as the other will always be visible.
  subprocess.check_call(["echo", "travis_fold:start:test_demo_returnn_as_framework"])
  assert os.path.exists("setup.py")
  if glob("dist/*.tar.gz"):
    # we want it unique below
    for fn in glob("dist/*.tar.gz"):
      os.remove(fn)
  if os.path.exists("MANIFEST"):
    os.remove("MANIFEST")  # auto-generated. will be recreated
  if os.path.exists("docs/crnn"):
    os.remove("docs/crnn")  # this is auto-generated, and confuses setup.py sdist
  if os.path.exists("docs/returnn"):
    os.remove("docs/returnn")  # this is auto-generated, and confuses setup.py sdist
  tmp_model_dir = "/tmp/%s/returnn-demo-as-framework" % get_login_username()
  if os.path.exists(tmp_model_dir):
    shutil.rmtree(tmp_model_dir, ignore_errors=True)
  print("setup.py sdist, to create package.")
  subprocess.check_call([py, "setup.py", "sdist"])
  dist_fns = glob("dist/*.tar.gz")
  assert len(dist_fns) == 1
  dist_fn = os.path.abspath(dist_fns[0])
  pip_path = which_pip()
  print("Pip install Returnn.")
  in_virtual_env = hasattr(sys, 'real_prefix')  # https://stackoverflow.com/questions/1871549/
  cmd = [py, pip_path, "install"]
  if not in_virtual_env:
    cmd += ["--user"]
  cmd += ["-v", dist_fn]
  print("$ %s" % " ".join(cmd))
  subprocess.check_call(cmd, cwd="/")
  print("Running demo now.")
  subprocess.check_call([py, "demo-returnn-as-framework.py"], cwd="demos")
  print("Success.")
  subprocess.check_call(["echo", "travis_fold:end:test_demo_returnn_as_framework"])


def test_demo_sprint_interface():
  import subprocess
  # echo via subprocess, because this stdout as well as the other will always be visible.
  subprocess.check_call(["echo", "travis_fold:start:test_demo_sprint_interface"])
  subprocess.check_call([py, os.path.abspath("demos/demo-sprint-interface.py")], cwd="/")
  subprocess.check_call(["echo", "travis_fold:end:test_demo_sprint_interface"])


def test_returnn_as_framework_TaskSystem():
  import subprocess
  # echo via subprocess, because this stdout as well as the other will always be visible.
  subprocess.check_call(["echo", "travis_fold:start:test_returnn_as_framework_TaskSystem"])
  subprocess.check_call([py, os.path.abspath("tests/returnn-as-framework.py"), "test_TaskSystem_Pickler()"], cwd="/")
  subprocess.check_call(["echo", "travis_fold:end:test_returnn_as_framework_TaskSystem"])


def test_returnn_as_framework_old_style_crnn_TFUtil():
  import subprocess
  # echo via subprocess, because this stdout as well as the other will always be visible.
  subprocess.check_call(["echo", "travis_fold:start:test_returnn_as_framework_old_style_crnn_TFUtil"])
  subprocess.check_call([
    py, os.path.abspath("tests/returnn-as-framework.py"),
    "--old-style", "--returnn-package-name", "crnn",
    "test_old_style_import_crnn_TFUtil()"], cwd="/")
  subprocess.check_call(["echo", "travis_fold:end:test_returnn_as_framework_old_style_crnn_TFUtil"])


def test_returnn_as_framework_old_style_TFUtil():
  import subprocess
  # echo via subprocess, because this stdout as well as the other will always be visible.
  subprocess.check_call(["echo", "travis_fold:start:test_returnn_as_framework_old_style_TFUtil"])
  subprocess.check_call([
    py, os.path.abspath("tests/returnn-as-framework.py"), "--old-style", "test_old_style_import_TFUtil()"], cwd="/")
  subprocess.check_call(["echo", "travis_fold:end:test_returnn_as_framework_old_style_TFUtil"])


if __name__ == '__main__':
  better_exchook.install()
  import sys
  fns = sys.argv[1:]
  if not fns:
    fns = [arg for arg in globals() if arg.startswith("test_")]
  for arg in fns:
    f = globals()[arg]
    print("-" * 20)
    print("Run:", f)
    print("-" * 20)
    try:
      f()
    except unittest.SkipTest as exc:
      print("(SkipTest: %s)" % exc)
    print("-" * 20)
    print("Ok.")
    print("-" * 20)
