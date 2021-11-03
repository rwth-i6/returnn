"""
Setups the environment for tests.
In the test code, you would have this::

    import _setup_test_env  # noqa

Also see :mod:`_setup_returnn_env`.
See :func:`setup` below for details.
"""


from __future__ import print_function


def setup():
  """
  Calls necessary setups.
  """
  import logging
  import os

  # Enable all logging, up to debug level.
  logging.basicConfig(level=logging.DEBUG, format='%(message)s')

  # Disable extensive TF debug verbosity. Must come before the first TF import.
  logging.getLogger('tensorflow').disabled = True
  # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
  # logging.getLogger("tensorflow").setLevel(logging.INFO)

  # Get us some further useful debug messages (in some cases, e.g. CUDA).
  # For example: https://github.com/tensorflow/tensorflow/issues/24496
  # os.environ["CUDNN_LOGINFO_DBG"] = "1"
  # os.environ["CUDNN_LOGDEST_DBG"] = "stdout"
  # The following might fix (workaround): Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
  # (https://github.com/tensorflow/tensorflow/issues/24496).
  # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

  os.environ.setdefault("RETURNN_TEST", "1")

  import _setup_returnn_env  # noqa

  import returnn.util.basic as util
  util.init_thread_join_hack()

  from returnn.util import better_exchook
  better_exchook.install()
  better_exchook.replace_traceback_format_tb()

  from returnn.log import log
  log.initialize(verbosity=[5], propagate=True)

  # TF is optional.
  # Note that importing TF still has a small side effect:
  # BackendEngine._get_default_engine() will return TF by default, if TF is already loaded.
  # For most tests, this does not matter.
  try:
    import tensorflow as tf
  except ImportError:
    tf = None

  if tf:
    import returnn.tf.util.basic as tf_util
    tf_util.debug_register_better_repr()

  import returnn.util.debug as debug
  debug.install_lib_sig_segfault()

  try:
    import faulthandler
    # Enable after libSigSegfault, so that we have both,
    # because faulthandler will also call the original sig handler.
    faulthandler.enable()
  except ImportError:
    print("no faulthandler")

  _try_hook_into_tests()


def _try_hook_into_tests():
  """
  Hook into nosetests or other unittest based frameworks.

  The hook will throw exceptions such that a debugger like PyCharm can inspect them easily.
  This will only be done if there is just a single test case.

  This code might be a bit experimental.
  It should work though. But if it does not, we can also skip this.
  Currently any exception here would be fatal though, as we expect this to work.

  Also see: https://youtrack.jetbrains.com/issue/PY-9848
  """
  # Check if this is run inside a debugger. Skip if this is not the case.
  import sys
  get_trace = getattr(sys, "gettrace", None)
  if not get_trace:
    return
  if get_trace() is None:
    return
  # get TestProgram instance from stack...
  from unittest import TestProgram
  from returnn.util.better_exchook import get_current_frame
  frame = get_current_frame()
  if not frame:
    # This will not always work. Just silently accept this. This should be rare.
    return
  test_program = None
  while frame:
    local_self = frame.f_locals.get("self")
    if isinstance(local_self, TestProgram):
      test_program = local_self
      break
    frame = frame.f_back
  if not test_program:
    # Ok, this is not run as test, so fine, nothing to do then.
    return
  test_names = getattr(test_program, "testNames")
  if not test_names:
    # Unexpected, but just silently ignore.
    return
  if len(test_names) >= 2 or ":" not in test_names[0]:
    # Multiple tests are being run. Do not hook into this.
    # We only want to install the hook if there is only a single test case.
    return

  # Ok, try to install our plugin.
  class _ReraiseExceptionTestHookPlugin:
    @staticmethod
    def _reraise_exception(test, err):
      exc_class, exc, tb = err
      print("Test %s, exception %s %s, reraise now." % (test, exc_class.__name__, exc))
      raise exc

    handleFailure = _reraise_exception
    handleError = _reraise_exception

  config = getattr(test_program, "config")
  config.plugins.addPlugin(_ReraiseExceptionTestHookPlugin())


setup()
