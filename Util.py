import subprocess
import h5py
from scipy.io.netcdf import NetCDFFile
from collections import deque
import inspect

def cmd(cmd):
  """
  :type cmd: list[str] | str
  :rtype: list[str]
  """
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, close_fds=True)
  result = [ tag.strip() for tag in p.communicate()[0].split('\n')[:-1]]
  p.stdout.close()
  return result

def hdf5_dimension(filename, dimension):
  fin = h5py.File(filename, "r")
  res = fin.attrs[dimension]
  fin.close()
  return res

def hdf5_strings(handle, name, data):
  S=max([len(d) for d in data])
  dset = handle.create_dataset(name, (len(data),), dtype="S"+str(S))
  dset[...] = data

def strtoact(act):
  """
  :param str act: activation function name
  :rtype: theano.Op
  """
  import theano.tensor as T
  activations = { 'logistic' : T.nnet.sigmoid,
                  'tanh' : T.tanh,
                  'relu': lambda z : (T.sgn(z) + 1) * z * 0.5,
                  'identity' : lambda z : z,
                  'one' : lambda z : 1,
                  'zero' : lambda z : 0,
                  'softsign': lambda z : z / (1.0 + abs(z)),
                  'softsquare': lambda z : 1 / (1.0 + z * z),
                  'maxout': lambda z : T.max(z, axis = 0),
                  'sin' : T.sin,
                  'cos' : T.cos }
  assert activations.has_key(act), "invalid activation function: " + act
  return activations[act]

def terminal_size(): # this will probably work on linux only
  import os, sys
  if not os.isatty(sys.stdout.fileno()):
    return -1, -1
  env = os.environ
  def ioctl_GWINSZ(fd):
    try:
      import fcntl, termios, struct, os
      cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,'1234'))
    except:
        return
    return cr
  cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
  if not cr:
    try:
        fd = os.open(os.ctermid(), os.O_RDONLY)
        cr = ioctl_GWINSZ(fd)
        os.close(fd)
    except:
        pass
  if not cr:
    cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
  return int(cr[1]), int(cr[0])

def hms(s):
  m, s = divmod(s, 60)
  h, m = divmod(m, 60)
  return "%d:%02d:%02d"%(h,m,s)

def progress_bar(complete = 1.0, prefix = "", suffix = ""):
  import sys
  terminal_width, _ = terminal_size()
  if terminal_width == -1: return
  if complete == 1.0:
    sys.stdout.write("\r%s"%(terminal_width * ' '))
    sys.stdout.flush()
    sys.stdout.write("\r")
    sys.stdout.flush()
    return
  progress = "%.02f%%" % (complete * 100)
  if prefix != "": prefix = prefix + " "
  if suffix != "": suffix = " " + suffix
  ntotal = terminal_width - len(progress) - len(prefix) - len(suffix) - 4
  bars = '|' * int(complete * ntotal)
  spaces = ' ' * (ntotal - int(complete * ntotal))
  bar = bars + spaces
  sys.stdout.write("\r%s" % prefix + "[" + bar[:len(bar)/2] + " " + progress + " " + bar[len(bar)/2:] + "]" + suffix)
  sys.stdout.flush()

def betterRepr(o):
  """
  The main difference: this one is deterministic.
  The orig dict.__repr__ has the order undefined for dict or set.
  For big dicts/sets/lists, add "," at the end to make textual diffs nicer.
  """
  if isinstance(o, list):
    return "[\n%s]" % "".join(map(lambda v: betterRepr(v) + ",\n", o))
  if isinstance(o, deque):
    return "deque([\n%s])" % "".join(map(lambda v: betterRepr(v) + ",\n", o))
  if isinstance(o, tuple):
    if len(o) == 1: return "(%s,)" % o[0]
    return "(%s)" % ", ".join(map(betterRepr, o))
  if isinstance(o, dict):
    return "{\n%s}" % "".join(map(lambda (k,v): betterRepr(k) + ": " + betterRepr(v) + ",\n", sorted(o.iteritems())))
  if isinstance(o, set):
    return "set([\n%s])" % "".join(map(lambda v: betterRepr(v) + ",\n", o))
  # fallback
  return repr(o)

def simpleObjRepr(obj):
  """
  All self.__init__ args.
  """
  return obj.__class__.__name__ + "(%s)" % \
                                  ", ".join(["%s=%s" % (arg, betterRepr(getattr(obj, arg)))
                                             for arg in inspect.getargspec(obj.__init__).args[1:]])

class ObjAsDict:
  def __init__(self, obj):
    self.__obj = obj

  def __getitem__(self, item):
    try:
      return getattr(self.__obj, item)
    except AttributeError, e:
      raise KeyError(e)
