
from __future__ import print_function

import _setup_test_env  # noqa
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import re
import os
import sys
from glob import glob
import unittest
from nose.tools import assert_less
from returnn.util import better_exchook


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
  theano_flags = {key: value for (key, value)
                  in [s.split("=", 1) for s in os.environ.get("THEANO_FLAGS", "").split(",") if s]}
  # First set some sane default for compile dir.
  theano_flags.setdefault("compiledir_format",
                          "compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s")
  # Extend compile dir for our tests in a subprocess.
  # This is important because there might be other tests where we import Theano in this
  # process and any subprocess would block because the parent process has locked the compile-dir.
  theano_flags["compiledir_format"] += "--nosetests"
  # Nose-tests will set mode=FAST_COMPILE. We don't want this for our tests as it is way too slow.
  theano_flags["mode"] = "FAST_RUN"
  env_update_ = os.environ.copy()
  env_update_["THEANO_FLAGS"] = ",".join(["%s=%s" % (key, value) for (key, value) in theano_flags.items()])
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


def test_demo_theano_task12ax():
  fer = run_config_get_fer("demos/demo-theano-task12ax.config")
  assert_less(fer, 0.01)


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
