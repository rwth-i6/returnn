
"""
Provides the main class for logging, :class:`Log`, and some helpers.
"""

from __future__ import print_function

import logging
import os
import sys
try:
  import StringIO
except ImportError:
  # noinspection PyPep8Naming
  import io as StringIO
import threading
from threading import RLock
import contextlib
import string
import typing

PY3 = sys.version_info[0] >= 3


class Stream:
  """
  Simple stream wrapper, which provides :func:`write` and :func:`flush`.
  """

  # noinspection PyShadowingNames
  def __init__(self, log, lvl):
    """
    :type log: logging.Logger
    :type lvl: int
    """
    self.buf = StringIO.StringIO()
    self.log = log
    self.lvl = lvl
    self.lock = RLock()

  def write(self, msg):
    """
    :param str msg:
    """
    with self.lock:
      if msg == '\n':
        self.flush()
      else:
        self.buf.write(msg)

  def flush(self):
    """
    Flush, i.e. writes to the log.
    """
    with self.lock:
      self.buf.flush()
      self.log.log(self.lvl, self.buf.getvalue())
      self.buf.truncate(0)
      # truncate does not change the current position.
      # In Python 2.7, it incorrectly does. See: http://bugs.python.org/issue30250
      self.buf.seek(0)


class Log:
  """
  The main logging class.
  """

  def __init__(self):
    self.initialized = False
    self.filename = None  # type: typing.Optional[str]
    self.v = None  # type: typing.Optional[typing.List[logging.Logger]]
    self.verbose = None  # type: typing.Optional[typing.List[bool]]
    self.error = None  # type: typing.Optional[Stream]
    self.v0 = None  # type: typing.Optional[Stream]
    self.v1 = None  # type: typing.Optional[Stream]
    self.v2 = None  # type: typing.Optional[Stream]
    self.v3 = None  # type: typing.Optional[Stream]
    self.v4 = None  # type: typing.Optional[Stream]
    self.v5 = None  # type: typing.Optional[Stream]

  def initialize(self, logs=None, verbosity=None, formatter=None):
    """
    :param list[str] logs:
    :param list[int] verbosity:
    :param list[str] formatter: 'default', 'timed', 'raw' or 'verbose'
    """
    if formatter is None:
      formatter = []
    if verbosity is None:
      verbosity = []
    if logs is None:
      logs = []
    self.initialized = True
    fmt = {
      'default': logging.Formatter('%(message)s'),
      'timed': logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S.%MS'),
      'raw': logging.Formatter('%(message)s'),
      'verbose': logging.Formatter('%(levelname)s - %(asctime)s %(message)s', datefmt='%Y-%m-%d,%H:%M:%S.%MS')
    }
    self.v = [logging.getLogger('v' + str(v)) for v in range(6)]
    for logger in self.v:
      # Reset handler list, in case we have initialized some earlier (e.g. multiple log.initialize() calls).
      logger.handlers = []
    if 'stdout' not in logs:
      logs.append('stdout')
    if len(formatter) == 1:
      # if only one format provided, use it for all logs
      formatter = [formatter[0]] * len(logs)
    for i in range(len(logs)):
      t = logs[i]
      v = 3
      if i < len(verbosity):
        v = verbosity[i]
      elif len(verbosity) == 1:
        v = verbosity[0]
      assert v <= 5, "invalid verbosity: " + str(v)
      f = fmt['default'] if i >= len(formatter) or formatter[i] not in fmt else fmt[formatter[i]]
      if t == 'stdout':
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
      elif t.startswith("|"):  # pipe-format
        proc_cmd = t[1:].strip()
        from subprocess import Popen, PIPE
        proc = Popen(proc_cmd, shell=True, stdin=PIPE)
        handler = logging.StreamHandler(proc.stdin)
        handler.setLevel(logging.DEBUG)
      elif os.path.isdir(os.path.dirname(t)):
        if "$" in t:
          from returnn.util.basic import get_utc_start_time_filename_part
          t = string.Template(t).substitute(date=get_utc_start_time_filename_part())
        self.filename = t
        handler = logging.FileHandler(t)
        handler.setLevel(logging.DEBUG)
      else:
        assert False, "invalid log target %r" % t
      handler.setFormatter(f)
      for j in range(v + 1):
        if handler not in self.v[j].handlers:
          self.v[j].addHandler(handler)
    self.verbose = [True] * 6
    null = logging.FileHandler(os.devnull)
    for i in range(len(self.v)):
      self.v[i].setLevel(logging.DEBUG)
      if not self.v[i].handlers:
        self.verbose[i] = False
        self.v[i].addHandler(null)
    self.error = Stream(self.v[0], logging.CRITICAL)
    self.v0 = Stream(self.v[0], logging.ERROR)
    self.v1 = Stream(self.v[1], logging.INFO)
    self.v2 = Stream(self.v[2], logging.INFO)
    self.v3 = Stream(self.v[3], logging.DEBUG)
    self.v4 = Stream(self.v[4], logging.DEBUG)
    self.v5 = Stream(self.v[5], logging.DEBUG)

  def init_by_config(self, config):
    """
    :param Config.Config config:
    """
    logs = config.list('log', [])
    log_verbosity = config.int_list('log_verbosity', [])
    log_format = config.list('log_format', [])
    if config.is_true("use_horovod"):
      import returnn.tf.horovod
      hvd = returnn.tf.horovod.get_ctx(config=config)
      new_logs = []
      for fn in logs:
        fn_prefix, fn_ext = os.path.splitext(fn)
        fn_ext = ".horovod-%i-%i%s" % (hvd.rank(), hvd.size(), fn_ext)
        new_logs.append(fn_prefix + fn_ext)
      logs = new_logs
    self.initialize(logs=logs, verbosity=log_verbosity, formatter=log_format)


log = Log()

# Some external code (e.g. pyzmq) will initialize the logging system
# via logging.basicConfig(). That will set a handler to the root logger,
# if there is none. This default handler usually prints to stderr.
# Because all our custom loggers are childs of the root logger,
# this will have the effect that everything gets logged twice.
# By adding a dummy handler to the root logger, we will avoid that
# it adds any other default handlers.
logging.getLogger().addHandler(logging.NullHandler())


class StreamThreadLocal(threading.local):
  """
  This will just buffer everything, thread-locally, and not forward it to any stream.
  The idea is that multiple tasks will run in multiple threads and you want to catch all the logging/stdout
  of each to not clutter the output, and also you want to keep it separate for each.
  """

  def __init__(self):
    self.buf = StringIO.StringIO()

  def write(self, msg):
    """
    :param str msg:
    """
    self.buf.write(msg)

  def flush(self):
    """
    Ignored.
    """


class StreamDummy:
  """
  This will just discard any data.
  """

  def write(self, msg):
    """
    Ignored.

    :param str msg:
    """
    pass

  def flush(self):
    """
    Ignored.
    """


@contextlib.contextmanager
def wrap_log_streams(alternative_stream, also_sys_stdout=False, tf_log_verbosity=None):
  """
  :param StreamThreadLocal|StreamDummy alternative_stream:
  :param bool also_sys_stdout: wrap sys.stdout as well
  :param int|str|None tf_log_verbosity: e.g. "WARNING"
  :return: context manager which yields (original info stream v1, alternative_stream)
  """
  v_attrib_keys = ["v%i" % i for i in range(6)] + ["error"]
  # Store original values.
  orig_v_list = log.v
  orig_v_attribs = {key: getattr(log, key) for key in v_attrib_keys}
  orig_stdout = sys.stdout
  log.v = [alternative_stream] * len(orig_v_list)
  for key in v_attrib_keys:
    setattr(log, key, alternative_stream)
  if also_sys_stdout:
    sys.stdout = alternative_stream
  orig_tf_log_verbosity = None
  if tf_log_verbosity is not None:
    import returnn.tf.compat as tf_compat
    orig_tf_log_verbosity = tf_compat.v1.logging.get_verbosity()
    tf_compat.v1.logging.set_verbosity(tf_log_verbosity)
  try:
    yield orig_v_attribs["v1"], alternative_stream
  finally:
    # Restore original values.
    log.v = orig_v_list
    for key, value in orig_v_attribs.items():
      setattr(log, key, value)
    if also_sys_stdout:
      sys.stdout = orig_stdout
    if tf_log_verbosity is not None:
      import returnn.tf.compat as tf_compat
      tf_compat.v1.logging.set_verbosity(orig_tf_log_verbosity)
