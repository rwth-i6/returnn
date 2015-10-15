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
    if isinstance(f, str):
      content = open(f).read()
    else:
      # assume stream-like
      content = f.read()
    content = content.strip()
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
    if value.find(',') > 0:
      value = value.split(',')
    else:
      value = [value]
    self.dict[key] = value

  def has(self, key):
    if key in self.typed_dict:
      return True
    return key in self.dict

  def is_typed(self, key):
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

  def value(self, key, default, index=None):
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
          return ",".join([str(v) for v in l])
        else:
          return str(l)
      else:
        return str(l[index])
    if key in self.dict:
      l = self.dict[key]
      if index is None:
        return ",".join(l)
      else:
        return l[index]
    return default

  def typed_value(self, key, default=None, index=None):
    value = self.typed_dict.get(key, default)
    if index is not None:
      assert isinstance(index, int)
      if isinstance(value, (list,tuple)):
        value = value[index]
      else:
        assert index == 0
    return value

  def int(self, key, default, index=0):
    if key in self.typed_dict:
      value = self.typed_value(key, default=default, index=index)
      if value is not None:
        assert isinstance(value, int)
      return value
    return int(self.value(key, default, index))

  def bool(self, key, default, index=0):
    if key in self.typed_dict:
      value = self.typed_value(key, default=default, index=index)
      if isinstance(value, int):
        value = bool(value)
      if value is not None:
        assert isinstance(value, bool)
      return value
    if key not in self.dict:
      return default
    v = str(self.value(key, None, index)).lower()
    if v == "true" or v == 'True' or v == "1":
      return True
    if v == "false" or v == 'False' or v == "0":
      return False
    assert v == "", "invalid bool value for %s: %s" % (key, v)
    return default

  def float(self, key, default, index=0):
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
