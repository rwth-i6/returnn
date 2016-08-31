#! /usr/bin/python2.7

__author__ = "Patrick Doetsch"
__copyright__ = "Copyright 2014"
__credits__ = ["Patrick Doetsch", "Paul Voigtlaender"]
__license__ = "GPL"
__version__ = "0.9"
__maintainer__ = "Patrick Doetsch"
__email__ = "doetsch@i6.informatik.rwth-aachen.de"


class Config:
  def __init__(self):
    self.dict = {}; """ :type: dict[str, list[str]] """
    self.typed_dict = {}; """ :type: dict[str] """  # could be loaded via JSON or so
    self.network_topology_json = None; """ :type: str | None """

  def load_file(self, f):
    """
    Reads the configuration parameters from a file and adds them to the inner set of parameters
    :type f: string
    """
    if isinstance(f, str):
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

  def add_line(self, key, value):
    """
    Adds one specific configuration (key,value) pair to the inner set of parameters
    :type key: string
    :type value: object
    """
    if value.find(',') > 0:
      value = value.split(',')
    else:
      value = [value]
    if key in self.typed_dict:
      del self.typed_dict[key]
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

  def set(self, key, value):
    """
    :type key: str
    :type value: list[str] | str | int | float | bool
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
        if isinstance(l, (list,tuple)):
          return list_join_str.join([str(v) for v in l])
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
    :rtype: str | T
    """
    value = self.typed_dict.get(key, default)
    if index is not None:
      assert isinstance(index, int)
      if isinstance(value, (list,tuple)):
        value = value[index]
      else:
        assert index == 0
    return value

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


def get_global_config():
  """
  :rtype: Config
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
  assert isinstance(rnn.config, Config)  # no other option anymore
  return rnn.config
