"""
Setups the environment for tests.
In the test code, you would have this::

    import _setup_test_env  # noqa

Also see :mod:`_setup_returnn_env`.
See :func:`setup` below for details.
"""


def setup():
  """
  Calls necessary setups.
  """

  # Disable extensive TF debug verbosity. Must come before the first TF import.
  import logging
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

  import _setup_returnn_env  # noqa

  import returnn.util.basic as util
  util.init_thread_join_hack()

  from returnn.util import better_exchook
  better_exchook.install()
  better_exchook.replace_traceback_format_tb()

  from returnn.log import log
  log.initialize(verbosity=[5])

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


setup()
