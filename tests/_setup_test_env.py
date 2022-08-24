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
  import sys

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
  if sys.excepthook != sys.__excepthook__:
    prev_sys_excepthook = sys.excepthook

    def _chained_excepthook(exctype, value, traceback):
      better_exchook.better_exchook(exctype, value, traceback)
      prev_sys_excepthook(exctype, value, traceback)

    sys.excepthook = _chained_excepthook
  else:
    sys.excepthook = better_exchook.better_exchook
  better_exchook.replace_traceback_format_tb()

  from returnn.log import log
  # No propagate, use stdout directly.
  log.initialize(verbosity=[5], propagate=False)

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
    # noinspection PyUnresolvedReferences
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
  import sys
  import types
  get_trace = getattr(sys, "gettrace", None)
  in_debugger = False
  if get_trace and get_trace() is not None:
    in_debugger = True

  # get TestProgram instance from stack...
  from unittest import TestProgram
  from returnn.util.better_exchook import get_current_frame
  from returnn.util.better_exchook import get_func_str_from_code_object
  top_frame = get_current_frame()
  if not top_frame:
    # This will not always work. Just silently accept this. This should be rare.
    return

  test_program = None
  frame = top_frame
  while frame:
    local_self = frame.f_locals.get("self")
    if isinstance(local_self, TestProgram):
      test_program = local_self
      break
    frame = frame.f_back

  test_names = None
  if test_program:  # nosetest, unittest
    test_names = getattr(test_program, "testNames")

  test_session = None
  try:
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    import pytest
  except ImportError:
    pass
  else:
    frame = top_frame
    while frame:
      local_self = frame.f_locals.get("self")
      if isinstance(local_self, pytest.Session):
        test_session = local_self
        break
      frame = frame.f_back
    if test_session and not test_names:
      test_names = test_session.config.args

  if not test_names:
    # Unexpected, but just silently ignore.
    return
  if len(test_names) >= 2 or ":" not in test_names[0]:
    # Multiple tests are being run. Do not hook into this.
    # We only want to install the hook if there is only a single test case.
    return

  # Skip this if we are not in a debugger.
  if test_program and in_debugger:  # nosetest, unittest

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

  if test_session and in_debugger:
    items = []

    def _custom_pytest_runtestloop(session):
      print("test env hook pytest_runtestloop")
      assert len(session.items) == len(test_names) == 1
      items.extend(session.items)

    def _custom_pytest_sessionfinish(session, exitstatus):
      session, exitstatus  # noqa  # not used
      print("test env hook pytest_sessionfinish")

    class _CustomPlugin:
      # noinspection PyShadowingNames
      def pytest_unconfigure(self, config):
        """hook for pytest_unconfigure."""
        print("test env hook pytest_unconfigure")
        # This will get called (potentially) multiple times via config._ensure_unconfigure in the `finally` block
        # in certain stages of the pytest call stack.
        frame_ = get_current_frame()
        while frame_:
          assert isinstance(frame_, types.FrameType)
          # If pytest_cmdline_main is in the call stack trace, we are not yet in the lowest stack.
          if get_func_str_from_code_object(frame_.f_code) == "pytest_cmdline_main":
            # We assume there is yet another lower `finally` block in the call stack
            # which calls to config._ensure_unconfigure.
            # However, it would not call pytest_unconfigure again when the configured flag is already set to False.
            # So we again call config._do_configure to set the flag to True again.
            # We don't want to run any of the other plugin hooks anymore, so unregister them all (except us).
            for plugin in test_session.config.pluginmanager.get_plugins():
              if plugin != self:
                test_session.config.pluginmanager.unregister(plugin)
            # noinspection PyProtectedMember
            config._do_configure()  # causes this to run again
            return
          frame_ = frame_.f_back
        print("test env hook pytest_unconfigure, final call")
        config.add_cleanup(self._custom_final_cleanup)

      @staticmethod
      def _custom_final_cleanup():
        print("test env hook pytest config cleanup")
        print("Now calling the test:", items[0])
        items[0].obj()

    # Overwrite these hooks completely - we don't want to run the tests at this point,
    # as all exceptions would be caught here.
    test_session.config.hook.pytest_runtestloop = _custom_pytest_runtestloop
    test_session.config.hook.pytest_sessionfinish = _custom_pytest_sessionfinish
    # For other things, using a plugin to register the hooks is probably the more clean way.
    test_session.config.pluginmanager.register(_CustomPlugin())


setup()
