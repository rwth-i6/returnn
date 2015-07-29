
from subprocess import Popen, PIPE, STDOUT
import re
import os
import sys
from glob import glob
from nose.tools import assert_less, assert_in


def build_env():
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
  env_update = os.environ.copy()
  env_update["THEANO_FLAGS"] = ",".join(["%s=%s" % (key, value) for (key, value) in theano_flags.items()])
  #print >>sys.stderr, theano_flags
  return env_update


def run(*args):
  args = list(args)
  assert os.path.exists(args[0]), "not found: %s" % args[0]
  args[0] = os.path.abspath(args[0])
  # crnn by default outputs on stderr, so just merge both together
  p = Popen(args, stdout=PIPE, stderr=STDOUT, env=build_env())
  out, _ = p.communicate()
  assert p.returncode == 0, "Return code is %i, std out/err: %s\n" % (p.returncode, out)
  return out


def run_and_parse_last_fer(*args):
  out = run(*args)
  parsed_fer = None
  for l in out.splitlines():
    # example: epoch 5 score: 0.0231807245472 elapsed: 0:00:04 dev: score 0.0137521058997 error 0.00268961807423
    m = re.match("epoch [0-9]+ score: .* dev: .* error ([0-9.]+)\\s?", l)
    if not m:
      continue
    parsed_fer = float(m.group(1))
  assert parsed_fer is not None, "No epoch dev errors found in output: %s\n" % out
  return parsed_fer


def run_config_get_fer(config_filename):
  cleanup_tmp_models(config_filename)
  fer = run_and_parse_last_fer("rnn.py", config_filename)
  cleanup_tmp_models(config_filename)
  return fer


def cleanup_tmp_models(config_filename):
  assert os.path.exists(config_filename)
  from Config import Config
  config = Config()
  config.load_file(config_filename)
  model_filename = config.value('model', '')
  assert model_filename
  # Remove existing models
  assert model_filename.startswith("/tmp/")
  for f in glob(model_filename + ".*"):
    os.remove(f)


class TestDemos(object):

  @classmethod
  def setup_class(cls):
    os.chdir((os.path.dirname(__file__) or ".") + "/..")
    assert os.path.exists("rnn.py")

  def test_demo_task12ax(self):
    fer = run_config_get_fer("demos/demo-task12ax.config")
    assert_less(fer, 0.01)

  def test_demo_iter_dataset_task12ax(self):
    cleanup_tmp_models("demos/demo-task12ax.config")
    out = run("demos/demo-iter-dataset.py", "demos/demo-task12ax.config")
    assert_in("Epoch 5.", out.splitlines())
