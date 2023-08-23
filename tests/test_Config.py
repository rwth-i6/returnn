import sys
import _setup_test_env  # noqa
import unittest
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_greater, assert_true, assert_false
from pprint import pprint
from returnn.config import *
from returnn.util import better_exchook
from io import StringIO
import textwrap


def test_old_format():
    config = Config()
    config.load_file(
        StringIO(
            """
  # comment
  num_inputs 3
  hidden_type forward,lstm
  """
        )
    )

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
    config.load_file(
        StringIO(
            """
  {
  // comment
  "num_inputs": 3,
  "hidden_type": ["forward", "lstm"]
  }
  """
        )
    )

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
    config.load_file(
        StringIO(
            """#!rnn.py
# comment
num_inputs = 3
hidden_type = ["forward", "lstm"]
  """
        )
    )

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
        cfgfile.write(
            """#!rnn.py
task = config.value("task", "train")

test_value = 42

def test_func():
  return task

    """
        )
        cfgfile.flush()
        rnn.init_config(command_line_options=[cfgfile.name, "--task", "search"])

    assert isinstance(rnn.config, Config)
    rnn.config.typed_dict.pop("__builtins__", None)  # not needed, too verbose for pprint
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
        cfgfile.write(
            """#!rnn.py
max_seq_length = {'bpe': 75}

def test_func():
  return max_seq_length

    """
        )
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


def test_config_py_ext():
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="test_rnn_initConfig") as cfgfile:
        cfgfile.write(
            """
def test_func():
  return config.value("task", "train")
    """
        )
        cfgfile.flush()
        config = Config()
        config.load_file(cfgfile.name)  # should determine format by suffix ".py"

    config.typed_dict.pop("__builtins__", None)  # not needed, too verbose for pprint
    pprint(config.dict)
    pprint(config.typed_dict)
    assert config.has("test_func")
    assert config.is_typed("test_func")
    test_func = config.typed_dict["test_func"]
    assert callable(test_func)
    assert_equal(test_func(), "train")
    config.set("task", "search")
    assert_equal(test_func(), "search")


def test_config_py_old_returnn_imports():
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", prefix="test_rnn_initConfig") as cfgfile:
        cfgfile.write(
            """
# These are some common imports found in older config files.
from Pretrain import WrapEpochValue
from TFUtil import where_bc
import TFUtil
import returnn.TFUtil

import TFHorovod
hvd = TFHorovod.get_ctx(config=config)  # should return None if no Horovod context

from returnn.Util import describe_returnn_version
returnn_version = describe_returnn_version()

tf_version_tuple = returnn.TFUtil.tf_version_tuple()

    """
        )
        cfgfile.flush()
        config = Config()
        config.load_file(cfgfile.name)  # should determine format by suffix ".py"

    config.typed_dict.pop("__builtins__", None)  # not needed, too verbose for pprint
    pprint(config.typed_dict)

    import returnn.util.basic as util
    import returnn.tf.util.basic as tf_util

    assert config.typed_dict["where_bc"] is tf_util.where_bc
    assert config.typed_dict["TFUtil"].where_bc is tf_util.where_bc
    assert config.typed_dict["hvd"] is None
    assert config.typed_dict["tf_version_tuple"] == tf_util.tf_version_tuple()
    assert config.typed_dict["returnn_version"] == util.describe_returnn_version()
    assert config.typed_dict["returnn"].TFUtil is tf_util


def test_pickle_config():
    import pickle
    import io

    config = Config()
    config.load_file(
        StringIO(
            textwrap.dedent(
                """\
                #!returnn.py

                def my_custom_func():
                    return 42

                class CustomClass:
                    x = 43

                    def __init__(self):
                        super().__init__()
                        CustomClass.x = 44

                    def get_value(self):
                        return CustomClass.x
                """
            )
        )
    )
    f = config.typed_dict["my_custom_func"]
    obj = config.typed_dict["CustomClass"]()

    sio = io.BytesIO()
    # noinspection PyUnresolvedReferences
    p = pickle._Pickler(sio)  # better for debugging
    with global_config_ctx(config):
        p.dump(config)

    config_ = pickle.loads(sio.getvalue())
    f_ = config_.typed_dict["my_custom_func"]
    assert f is not f_  # it should really be a copy, via Config.__getstate__ logic
    assert f() == f_() == 42
    obj_ = config_.typed_dict["CustomClass"]()
    assert type(obj) is not type(obj_)
    assert obj.get_value() == obj_.get_value() == 44


def test_config_pickle_function():
    # Having some function inside the config, there are cases when we need to pickle it,
    # e.g. when it is the post_process function of a dataset,
    # and then used as the PyTorch dataset -- see test_PTDataset.py, test_HDFDataset_pickle.
    # Currently, it fails with:
    #   PicklingError: Can't pickle <function speed_pert_...>:
    #     import of module '__returnn_config__' failed
    # Functions are pickled by reference, storing the module name and function name.
    import pickle

    config = Config()
    config.load_file(
        StringIO(
            textwrap.dedent(
                """\
                #!returnn.py

                def my_custom_func():
                    return 42
                """
            )
        )
    )
    with global_config_ctx(config):
        f = config.typed_dict["my_custom_func"]
        f_ = pickle.loads(pickle.dumps(f))
        assert f_ is f
        assert f_() == 42


def _config_pickle_proc_main(config, f):
    assert isinstance(config, Config)
    assert get_global_config() is config
    assert callable(f)
    f()


def test_config_pickle_function_multi_proc():
    # Same as test_config_pickle_function but via multiprocessing,
    # so across process boundaries.
    import multiprocessing

    _mp = multiprocessing.get_context("spawn")

    config = Config()
    config.load_file(
        StringIO(
            textwrap.dedent(
                """\
                #!returnn.py

                def my_custom_func():
                    import sys
                    sys.exit(42)
                """
            )
        )
    )
    with global_config_ctx(config):
        f = config.typed_dict["my_custom_func"]
        proc = _mp.Process(target=_config_pickle_proc_main, args=(config, f))
        proc.start()
        proc.join()
        assert proc.exitcode == 42


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
