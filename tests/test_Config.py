
import sys
sys.path += ["."]  # Python 3 hack

import unittest
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_greater, assert_true, assert_false
from pprint import pprint
from returnn.config import Config
from returnn.util.basic import PY3
from returnn.util import better_exchook
better_exchook.replace_traceback_format_tb()
if PY3:
  from io import StringIO
else:
  # noinspection PyUnresolvedReferences,PyCompatibility
  from StringIO import StringIO


def test_old_format():
  config = Config()
  config.load_file(StringIO("""
  # comment
  num_inputs 3
  hidden_type forward,lstm
  """))

  assert_true(config.has("num_inputs"))
  assert_true(config.has("hidden_type"))
  assert_equal(config.int("num_inputs", -1), 3)
  assert_equal(config.value("hidden_type", "x"), "forward,lstm")
  assert_equal(config.value("hidden_type", "x", index=0), "forward")
  assert_equal(config.value("hidden_type", "x", index=1), "lstm")
  assert_equal(config.list("hidden_type", ["x"]), ["forward", "lstm"])

  assert_false(config.is_typed("num_inputs"))
  assert_false(config.is_typed("hidden_type"))


def test_json_format():
  config = Config()
  config.load_file(StringIO("""
  {
  // comment
  "num_inputs": 3,
  "hidden_type": ["forward", "lstm"]
  }
  """))

  assert_true(config.has("num_inputs"))
  assert_true(config.has("hidden_type"))
  assert_equal(config.int("num_inputs", -1), 3)
  assert_equal(config.value("hidden_type", "x"), "forward,lstm")
  assert_equal(config.value("hidden_type", "x", index=0), "forward")
  assert_equal(config.value("hidden_type", "x", index=1), "lstm")
  assert_equal(config.list("hidden_type", ["x"]), ["forward", "lstm"])

  assert_true(config.is_typed("num_inputs"))
  assert_true(config.is_typed("hidden_type"))
  assert_is_instance(config.typed_value("num_inputs"), int)
  assert_is_instance(config.typed_value("hidden_type"), list)
  assert_equal(config.typed_value("hidden_type"), ["forward", "lstm"])


def test_py_config():
  config = Config()
  config.load_file(StringIO("""#!rnn.py
# comment
num_inputs = 3
hidden_type = ["forward", "lstm"]
  """))

  assert_true(config.has("num_inputs"))
  assert_true(config.has("hidden_type"))
  assert_equal(config.int("num_inputs", -1), 3)
  assert_equal(config.value("hidden_type", "x"), "forward,lstm")
  assert_equal(config.value("hidden_type", "x", index=0), "forward")
  assert_equal(config.value("hidden_type", "x", index=1), "lstm")
  assert_equal(config.list("hidden_type", ["x"]), ["forward", "lstm"])

  assert_true(config.is_typed("num_inputs"))
  assert_true(config.is_typed("hidden_type"))
  assert_is_instance(config.typed_value("num_inputs"), int)
  assert_is_instance(config.typed_value("hidden_type"), list)
  assert_equal(config.typed_value("hidden_type"), ["forward", "lstm"])


def test_rnn_init_config_py_global_var():
  import returnn.__main__ as rnn
  import tempfile
  with tempfile.NamedTemporaryFile(mode="w", suffix=".config", prefix="test_rnn_initConfig") as cfgfile:
    cfgfile.write("""#!rnn.py
task = config.value("task", "train")

test_value = 42

def test_func():
  return task

    """)
    cfgfile.flush()
    rnn.init_config(command_line_options=[cfgfile.name, "--task", "search"])

  assert isinstance(rnn.config, Config)
  pprint(rnn.config.dict)
  pprint(rnn.config.typed_dict)
  assert rnn.config.has("task")
  assert rnn.config.has("test_value")
  assert rnn.config.has("test_func")
  assert_equal(rnn.config.value("task", None), "search")
  assert rnn.config.is_typed("test_value")
  assert_equal(rnn.config.typed_value("test_value"), 42)
  assert rnn.config.is_typed("test_func")
  # So far it's fine.
  # Now something a bit strange.
  # Earlier, this failed, because the command-line overwrote this.
  assert rnn.config.is_typed("task")
  test_func = rnn.config.typed_dict["test_func"]
  assert callable(test_func)
  assert_equal(test_func(), "search")


def test_rnn_init_config_py_cmd_type():
  import returnn.__main__ as rnn
  import tempfile
  with tempfile.NamedTemporaryFile(mode="w", suffix=".config", prefix="test_rnn_initConfig") as cfgfile:
    cfgfile.write("""#!rnn.py
max_seq_length = {'bpe': 75}

def test_func():
  return max_seq_length

    """)
    cfgfile.flush()
    rnn.init_config(command_line_options=[cfgfile.name, "++max_seq_length", "0"])

  assert isinstance(rnn.config, Config)
  assert rnn.config.has("max_seq_length")
  assert rnn.config.has("test_func")
  assert rnn.config.is_typed("max_seq_length")
  assert rnn.config.is_typed("test_func")
  test_func = rnn.config.typed_dict["test_func"]
  assert callable(test_func)
  assert_equal(test_func(), 0)


if __name__ == "__main__":
  better_exchook.install()
  if len(sys.argv) <= 1:
    for k, v in sorted(globals().items()):
      if k.startswith("test_"):
        print("-" * 40)
        print("Executing: %s" % k)
        try:
          v()
        except unittest.SkipTest as exc:
          print("SkipTest:", exc)
        print("-" * 40)
    print("Finished all tests.")
  else:
    assert len(sys.argv) >= 2
    for arg in sys.argv[1:]:
      print("Executing: %s" % arg)
      if arg in globals():
        globals()[arg]()  # assume function and execute
      else:
        eval(arg)  # assume Python code and execute
