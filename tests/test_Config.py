
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_greater, assert_true, assert_false
from Config import Config
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

