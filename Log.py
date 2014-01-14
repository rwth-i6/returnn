import logging
import os

import StringIO
class Stream():
  def __init__(self, log, lvl):
    self.buf = StringIO.StringIO()
    self.log = log
    self.lvl = lvl

  def write(self, msg):
    if msg == '\n':
      self.flush()
    else:
      self.buf.write(msg)
    
  def flush(self):
    self.buf.flush()
    self.log.log(self.lvl, self.buf.getvalue())
    self.buf.truncate(0)

class Log:
  def initialize(self, logs = [], verbosity = [], formatter = []):
    fmt = { 'default' : logging.Formatter('%(message)s'),
            'timed' : logging.Formatter('%(asctime)s %(message)s', datefmt = '%Y-%m-%d,%H:%M:%S.%MS'),
            'raw''' : logging.Formatter('%(message)s'),
            'verbose': logging.Formatter('%(levelname)s - %(asctime)s %(message)s', datefmt = '%Y-%m-%d,%H:%M:%S.%MS')
          }
    self.v = [ logging.getLogger('v' + str(v)) for v in xrange(6) ]
    if not 'stdout' in logs:
      logs.append('stdout')
    for i in xrange(len(logs)):
      t = logs[i]
      v = 3
      if i < len(verbosity):
        v = verbosity[i]
      elif len(verbosity) == 1:
        v = verbosity[0]
      assert v <= 5, "invalid verbosity: " + str(v)
      f = fmt['default'] if i >= len(formatter) or not fmt.has_key(formatter[i]) else fmt[formatter[i]]
      if t == 'stdout':
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
      elif os.path.isdir(os.path.dirname(t)):
        handler = logging.FileHandler(t)
        handler.setLevel(logging.DEBUG)
      else: assert False, "invalid log target"
      handler.setFormatter(f)
      for j in xrange(v + 1):
        if not handler in self.v[j].handlers:
          self.v[j].addHandler(handler)
    self.verbose = [ True ] * 6
    null = logging.FileHandler(os.devnull)
    for i in xrange(len(self.v)):
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
  def write(self, msg): 
    self.info(msg)

log = Log()