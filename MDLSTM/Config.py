#!/usr/bin/env python

class Config:
  def __init__(self):
    self.dict = {}
  
  def load_file(self, filename):
    for line in open(filename).readlines():
      if len(line.strip()) == 0 or line.strip()[0] == '#':
        continue
      line = line.strip().split()
      assert len(line) == 2
      key = line[0]
      value = line[1]
      if value.find(',') > 0:
        value = value.split(',')
      else:
        value = [value]
      self.dict[key] = value
      
  def has(self, key):
    return self.dict.has_key(key)
  
  def value(self, key, default, index=0, force_key=False):
    if force_key:
      assert self.has(key)
    if not self.dict.has_key(key):
      return default
    return self.dict[key][index]

  def int(self, key, default, index=0, force_key=False):
    if force_key:
      assert self.has(key)
    return int(self.value(key, default, index))

  def bool(self, key, default, index=0, force_key=False):
    if force_key:
      assert self.has(key)
    v = str(self.value(key, default, index)).lower()    
    if v == "true" or v == "1":
      return True
    if v == "false" or v == "0":
      return False
    return default

  def float(self, key, default, index=0, force_key=False):
    if force_key:
      assert self.has(key)
    return float(self.value(key, default, index))

  def list(self, key, default=[], force_key=False):
    if force_key:
      assert self.has(key)
    if not self.dict.has_key(key):
      return default
    return self.dict[key]
  
  def int_list(self, key, default=[], force_key=False):
    if force_key:
      assert self.has(key)
    return [int(x) for x in self.list(key, default)]

  def float_list(self, key, default=[], force_key=False):
    if force_key:
      assert self.has(key)
    return [float(x) for x in self.list(key, default)]
  
  def int_pair(self, key, default=(0,0), force_key=False):
    if force_key:
      assert self.has(key)
    if not self.has(key): return default
    value = self.value(key, '')
    if ':' in value:
      return int(value.split(':')[0]),int(value.split(':')[1])
    else:
      return int(value), int(value)
