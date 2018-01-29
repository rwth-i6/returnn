
from __future__ import print_function

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2014"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]
__license__ = "GPL"
__version__ = "0.9"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"

import sys
PY3 = sys.version_info[0] >= 3

if PY3:
  import builtins
  unicode = str
  long = int
else:
  import __builtin__ as builtins
  unicode = builtins.unicode
  long = builtins.long


class Config:
  def __init__(self):
    self.dict = {}; """ :type: dict[str, list[str]] """
    self.typed_dict = {}; """ :type: dict[str] """  # could be loaded via JSON or so
    self.network_topology_json = None; """ :type: str | None """
    self.files = []

  def load_file(self, f):
    """
    Reads the configuration parameters from a file and adds them to the inner set of parameters
    :param string|io.TextIOBase|io.StringIO f:
    """
    if isinstance(f, str):
      import os
      assert os.path.isfile(f), "config file not found: %r" % f
      self.files.append(f)
      filename = f
      content = open(filename).read()
    else:
      # assume stream-like
      filename = "<config string>"
      content = f.read()
    content = content.strip()
    if content.startswith("#!"):  # assume Python
      from Util import custom_exec
      # Operate inplace on ourselves.
      # Also, we want that it's available as the globals() dict, so that defined functions behave well
      # (they would loose the local context otherwise).
      user_ns = self.typed_dict
      # Always overwrite:
      user_ns.update({"config": self, "__file__": filename, "__name__": "__crnn_config__"})
      custom_exec(content, filename, user_ns, user_ns)
      return
    if content.startswith("{"):  # assume JSON
      from Util import load_json
      json_content = load_json(content=content)
      assert isinstance(json_content, dict)
      self.update(json_content)
      return
    # old line-based format
    for line in content.splitlines():
      if "#" in line:  # Strip away comment.
        line = line[:line.index("#")]
      line = line.strip()
      if not line:
        continue
      line = line.split(None, 1)
      assert len(line) == 2, "unable to parse config line: %r" % line
      self.add_line(key=line[0], value=line[1])

  @classmethod
  def get_config_file_type(cls, f):
    """
    :param str f: file path
    :return: "py", "js" or "txt"
    :rtype: str
    """
    with open(f, "r") as f:
      start = f.read(3)
    if start.startswith("#!"):
      return "py"
    if start.startswith("{"):
      return "js"
    return "txt"

  def parse_cmd_args(self, args):
    """
    :param list[str]|tuple[str] args:
    """
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-a", "--activation", dest="activation",
                      help="[STRING/LIST] Activation functions: logistic, tanh, softsign, relu, identity, zero, one, maxout.")
    parser.add_option("-b", "--batch_size", dest="batch_size",
                      help="[INTEGER/TUPLE] Maximal number of frames per batch (optional: shift of batching window).")
    parser.add_option("-c", "--chunking", dest="chunking",
                      help="[INTEGER/TUPLE] Maximal number of frames per sequence (optional: shift of chunking window).")
    parser.add_option("-d", "--description", dest="description", help="[STRING] Description of experiment.")
    parser.add_option("-e", "--epoch", dest="epoch", help="[INTEGER] Starting epoch.")
    parser.add_option("-E", "--eval", dest="eval", help="[STRING] eval file path")
    parser.add_option("-f", "--gate_factors", dest="gate_factors",
                      help="[none/local/global] Enables pooled (local) or separate (global) coefficients on gates.")
    parser.add_option("-g", "--lreg", dest="lreg", help="[FLOAT] L1 or L2 regularization.")
    parser.add_option("-i", "--save_interval", dest="save_interval",
                      help="[INTEGER] Number of epochs until a new model will be saved.")
    parser.add_option("-j", "--dropout", dest="dropout", help="[FLOAT] Dropout probability (0 to disable).")
    # parser.add_option("-k", "--multiprocessing", dest = "multiprocessing", help = "[BOOLEAN] Enable multi threaded processing (required when using multiple devices).")
    parser.add_option("-k", "--output_file", dest="output_file",
                      help="[STRING] Path to target file for network output.")
    parser.add_option("-l", "--log", dest="log", help="[STRING] Log file path.")
    parser.add_option("-L", "--load", dest="load", help="[STRING] load model file path.")
    parser.add_option("-m", "--momentum", dest="momentum",
                      help="[FLOAT] Momentum term in gradient descent optimization.")
    parser.add_option("-n", "--num_epochs", dest="num_epochs",
                      help="[INTEGER] Number of epochs that should be trained.")
    parser.add_option("-o", "--order", dest="order", help="[default/sorted/random] Ordering of sequences.")
    parser.add_option("-p", "--loss", dest="loss", help="[loglik/sse/ctc] Objective function to be optimized.")
    parser.add_option("-q", "--cache", dest="cache",
                      help="[INTEGER] Cache size in bytes (supports notation for kilo (K), mega (M) and gigabtye (G)).")
    parser.add_option("-r", "--learning_rate", dest="learning_rate",
                      help="[FLOAT] Learning rate in gradient descent optimization.")
    parser.add_option("-s", "--hidden_sizes", dest="hidden_sizes",
                      help="[INTEGER/LIST] Number of units in hidden layers.")
    parser.add_option("-t", "--truncate", dest="truncate",
                      help="[INTEGER] Truncates sequence in BPTT routine after specified number of timesteps (-1 to disable).")
    parser.add_option("-u", "--device", dest="device",
                      help="[STRING/LIST] CPU and GPU devices that should be used (example: gpu0,cpu[1-6] or gpu,cpu*).")
    parser.add_option("-v", "--verbose", dest="log_verbosity", help="[INTEGER] Verbosity level from 0 - 5.")
    parser.add_option("-w", "--window", dest="window", help="[INTEGER] Width of sliding window over sequence.")
    parser.add_option("-x", "--task", dest="task", help="[train/forward/analyze] Task of the current program call.")
    parser.add_option("-y", "--hidden_type", dest="hidden_type",
                      help="[VALUE/LIST] Hidden layer types: forward, recurrent, lstm.")
    parser.add_option("-z", "--max_sequences", dest="max_seqs", help="[INTEGER] Maximal number of sequences per batch.")
    parser.add_option("--config", dest="load_config", help="[STRING] load config")
    (options, args) = parser.parse_args(list(args))
    options = vars(options)
    for opt in options.keys():
      if options[opt] is not None:
        if opt == "load_config":
          self.load_file(options[opt])
        else:
          self.add_line(opt, options[opt])
    assert len(args) % 2 == 0, "expect (++key, value) config tuples in remaining args: %r" % args
    for i in range(0, len(args), 2):
      key, value = args[i:i + 2]
      assert key[0:2] == "++", "expect key prefixed with '++' in (%r, %r)" % (key, value)
      if value[:2] == "+-":
        value = value[1:]  # otherwise we never could specify things like "++threshold -0.1"
      self.add_line(key=key[2:], value=value)

  def add_line(self, key, value):
    """
    Adds one specific configuration (key,value) pair to the inner set of parameters
    :type key: str
    :type value: str
    """
    if key in self.typed_dict:
      # This is a special case. We overwrite a config value which was typed before.
      # E.g. this could have been loaded via a Python config file.
      # We want to keep the entry in self.typed_dict because there might be functions/lambdas inside
      # the config which require the global variable to be available.
      # See :func:`test_rnn_initConfig_py_global_var`.
      value_type = type(self.typed_dict[key])
      if value_type == str:
        pass  # keep as-is
      else:
        value = eval(value)
      self.typed_dict[key] = value
      return
    if value.find(',') > 0:
      value = value.split(',')
    else:
      value = [value]
    if key == 'include':
      for f in value:
        self.load_file(f)
    else:
      self.dict[key] = value

  def has(self, key):
    """
    Returns whether the given key is present in the inner set of parameters
    :type key: string
    :rtype: boolean
    :returns True if and only if the given key is in the inner set of parameters
    """
    if key in self.typed_dict:
      return True
    return key in self.dict

  def is_typed(self, key):
    """
    :type key: string
    :rtype: boolean
    :returns True if and only if the value of the given key has a specified data type
    """
    return key in self.typed_dict

  def is_true(self, key, default=False):
    """
    :param str key:
    :param bool default:
    :return: bool(value) if it is set or default
    :rtype: bool
    """
    if self.is_typed(key):
      return bool(self.typed_dict[key])
    return self.bool(key, default=default)

  def is_of_type(self, key, types):
    """
    :param str key:
    :param type|tuple[type] types: for isinstance() check
    :return: whether is_typed(key) is True and isinstance(value, types) is True
    :rtype: bool
    """
    if key in self.typed_dict:
      return isinstance(self.typed_dict[key], types)
    return False

  def get_of_type(self, key, types, default=None):
    """
    :param str key:
    :param type|list[type]|T types: for isinstance() check
    :param T|None default:
    :return: if is_of_type(key, types) is True, returns the value, otherwise default
    :rtype: T
    """
    if self.is_of_type(key, types):
      return self.typed_dict[key]
    return default

  def set(self, key, value):
    """
    :type key: str
    :type value: list[str] | str | int | float | bool | None
    """
    self.typed_dict[key] = value

  def update(self, dikt):
    """
    :type dikt: dict
    """
    for key, value in dikt.items():
      self.set(key, value)

  def _hack_value_reading_debug(self):
    orig_value_func = self.value
    def wrapped_value_func(*args, **kwargs):
      res = orig_value_func(*args, **kwargs)
      print("Config.value(%s) -> %r" % (", ".join(list(map(repr, args)) + ["%s=%r" for (k, v) in kwargs.items()]), res))
      return res
    self.value = wrapped_value_func

  def value(self, key, default, index=None, list_join_str=","):
    """
    :type key: str
    :type default: T
    :type index: int | None
    :rtype: str | T
    """
    if key in self.typed_dict:
      l = self.typed_dict[key]
      if index is None:
        if isinstance(l, (list, tuple)):
          return list_join_str.join([str(v) for v in l])
        elif l is None:
          return default
        else:
          return str(l)
      else:
        return str(l[index])
    if key in self.dict:
      l = self.dict[key]
      if index is None:
        return list_join_str.join(l)
      else:
        return l[index]
    return default

  def typed_value(self, key, default=None, index=None):
    """
    :type key: str
    :type default: T
    :type index: int | None
    :rtype: T | object
    """
    value = self.typed_dict.get(key, default)
    if index is not None:
      assert isinstance(index, int)
      if isinstance(value, (list, tuple)):
        value = value[index]
      else:
        assert index == 0
    return value

  def opt_typed_value(self, key, default=None):
    """
    :param str key:
    :param T|None default:
    :rtype: T|object|str|None
    """
    if key in self.typed_dict:
      return self.typed_dict[key]
    return self.value(key, default)

  def int(self, key, default, index=0):
    """
    Parses the value of the given key as integer, returning default if not existent
    :type key: str
    :type default: T
    :type index: int
    :rtype: int | T
    """
    if key in self.typed_dict:
      value = self.typed_value(key, default=default, index=index)
      if value is not None:
        assert isinstance(value, int)
      return value
    return int(self.value(key, default, index))

  def bool(self, key, default, index=0):
    """
    Parses the value of the given key as boolean, returning default if not existent
    :type key: str
    :type default: T
    :type index: bool
    :rtype: bool | T
    """
    if key in self.typed_dict:
      value = self.typed_value(key, default=default, index=index)
      if isinstance(value, int):
        value = bool(value)
      if value is not None:
        assert isinstance(value, bool)
      return value
    if key not in self.dict:
      return default
    v = str(self.value(key, None, index))
    if not v:
      return default
    from Util import to_bool
    return to_bool(v)

  def bool_or_other(self, key, default, index=0):
    if key in self.typed_dict:
      return self.typed_value(key, default=default, index=index)
    if key not in self.dict:
      return default
    v = str(self.value(key, None, index))
    if not v:
      return default
    from Util import to_bool
    try:
      return to_bool(v)
    except ValueError:
      return v

  def float(self, key, default, index=0):
    """
    Parses the value of the given key as float, returning default if not existent
    :type key: str
    :type default: T
    :type index: int
    :rtype: float | T
    """
    if key in self.typed_dict:
      value = self.typed_value(key, default=default, index=index)
      if value is not None:
        if isinstance(value, (str, unicode)):
          # Special case for float as str. We automatically cast this case.
          # This is also to handle special values such as "inf".
          value = float(value)
        assert isinstance(value, (int, float))
      return value
    return float(self.value(key, default, index))

  def list(self, key, default=None):
    """
    :type key: str
    :type default: T
    :rtype: list[str] | T
    """
    if default is None:
      default = []
    if key in self.typed_dict:
      value = self.typed_value(key, default=default)
      if value is None:
        return default
      if not isinstance(value, (tuple,list)):
        value = [value]
      return list(value)
    if key not in self.dict:
      return default
    return self.dict[key]

  def int_list(self, key, default=None):
    """
    :type key: str
    :type default: T
    :rtype: list[int] | T
    """
    if default is None:
      default = []
    if key in self.typed_dict:
      value = self.typed_value(key, default=default)
      if value is None:
        return default
      if not isinstance(value, (tuple,list)):
        value = [value]
      for x in value:
        assert isinstance(x, int)
      return list(value)
    return [int(x) for x in self.list(key, default)]

  def float_list(self, key, default=None):
    """
    :type key: str
    :type default: T
    :rtype: list[float] | T
    """
    if default is None:
      default = []
    if key in self.typed_dict:
      value = self.typed_value(key, default=default)
      if value is None:
        return default
      if not isinstance(value, (tuple,list)):
        value = [value]
      for x in value:
        assert isinstance(x, (float,int))
      return list(value)
    return [float(x) for x in self.list(key, default)]

  def int_pair(self, key, default=None):
    if default is None:
      default = (0, 0)
    if not self.has(key):
      return default
    if key in self.typed_dict:
      value = self.typed_value(key, default=default)
      if not isinstance(value, (tuple,list)):
        value = (value, value)
      assert len(value) == 2
      for x in value:
        assert isinstance(x, int)
      return tuple(value)
    value = self.value(key, '')
    if ':' in value:
      return int(value.split(':')[0]), int(value.split(':')[1])
    else:
      return int(value), int(value)


def get_global_config(raise_exception=True):
  """
  :param bool raise_exception: if no global config is found, raise an exception, otherwise return None
  :rtype: Config|None
  """
  import TaskSystem
  import Device
  if not TaskSystem.isMainProcess:
    # We expect that we are a Device subprocess.
    assert Device.asyncChildGlobalDevice is not None
    return Device.asyncChildGlobalDevice.config
  # We are the main process.
  import sys
  main_mod = sys.modules["__main__"]  # should be rnn.py
  if isinstance(getattr(main_mod, "config", None), Config):
    return main_mod.config
  # Maybe __main__ is not rnn.py, or config not yet loaded.
  # Anyway, try directly. (E.g. for SprintInterface.)
  import rnn
  if isinstance(rnn.config, Config):
    return rnn.config
  if raise_exception:
    raise Exception("No global config found.")
  return None
