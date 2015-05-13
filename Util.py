import subprocess
import h5py
from collections import deque
import inspect
import os
import shlex


def cmd(cmd):
  """
  :type cmd: list[str] | str
  :rtype: list[str]
  """
  p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True, close_fds=True)
  result = [ tag.strip() for tag in p.communicate()[0].split('\n')[:-1]]
  p.stdout.close()
  return result


def eval_shell_env(token):
  if token.startswith("$"):
    return os.environ.get(token[1:], "")
  return token

def eval_shell_str(s):
  """
  :type s: str
  :rtype: list[str]

  Parses `s` as shell like arguments (via shlex.split) and evaluates shell environment variables (eval_shell_env).
  """
  tokens = []
  for token in shlex.split(s):
    if token.startswith("$"):
      tokens += eval_shell_str(eval_shell_env(token))
    else:
      tokens += [token]
  return tokens

def hdf5_dimension(filename, dimension):
  fin = h5py.File(filename, "r")
  if '/' in dimension:
    res = fin['/'.join(dimension.split('/')[:-1])].attrs[dimension.split('/')[-1]]
  else:
    res = fin.attrs[dimension]
  fin.close()
  return res

def hdf5_group(filename, dimension):
  fin = h5py.File(filename, "r")
  res = { k : fin[dimension].attrs[k] for k in fin[dimension].attrs }
  fin.close()
  return res

def hdf5_shape(filename, dimension):
  fin = h5py.File(filename, "r")
  res = fin[dimension].shape
  fin.close()
  return res


def hdf5_strings(handle, name, data):
  S=max([len(d) for d in data])
  dset = handle.create_dataset(name, (len(data),), dtype="S"+str(S))
  dset[...] = data


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
  return "%d:%02d:%02d" % (h, m, s)


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
    if len(o) == 1:
      return "(%s,)" % o[0]
    return "(%s)" % ", ".join(map(betterRepr, o))
  if isinstance(o, dict):
    l = [betterRepr(k) + ": " + betterRepr(v) for (k,v) in sorted(o.iteritems())]
    if sum([len(v) for v in l]) >= 40:
      return "{\n%s}" % "".join([v + ",\n" for v in l])
    else:
      return "{%s}" % ", ".join(l)
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


def obj_diff_str(self, other):
  s = []
  for attrib in sorted(set(other.__dict__).union(other.__dict__.keys())):
    if attrib not in self.__dict__ or attrib not in other.__dict__:
      s += ["attrib %r not on both" % attrib]
      continue
    value_self = getattr(self, attrib)
    value_other = getattr(other, attrib)
    if isinstance(value_self, list):
      if not isinstance(value_other, list):
        s += ["attrib %r self is list but other is %r" % (attrib, type(value_other))]
      elif len(value_self) != len(value_other):
        s += ["attrib %r list differ. len self: %i, len other: %i" % (attrib, len(value_self), len(value_other))]
      else:
        for i, (a, b) in enumerate(zip(value_self, value_other)):
          if a != b:
            s += ["attrib %r[%i] differ. self: %r, other: %r" % (attrib, i, a, b)]
    else:
      if value_self != value_other:
        s += ["attrib %r differ. self: %r, other: %r" % (attrib, value_self, value_other)]
  if s:
    return "\n".join(s)
  else:
    return "No diff."


def find_ranges(l):
  """
  :type l: list[int]
  :returns list of ranges (start,end) where end is exclusive
  such that the union of range(start,end) matches l.
  :rtype: list[(int,int)]
  We expect that the incoming list is sorted and strongly monotonic increasing.
  """
  if not l:
    return []
  ranges = [(l[0], l[0])]
  for k in l:
    assert k >= ranges[-1][1]  # strongly monotonic increasing
    if k == ranges[-1][1]:
      ranges[-1] = (ranges[-1][0], k + 1)
    else:
      ranges += [(k, k + 1)]
  return ranges


def initThreadJoinHack():
  import threading, thread
  mainThread = threading.currentThread()
  assert isinstance(mainThread, threading._MainThread)
  mainThreadId = thread.get_ident()

  # Patch Thread.join().
  join_orig = threading.Thread.join
  def join_hacked(threadObj, timeout=None):
    """
    :type threadObj: threading.Thread
    :type timeout: float|None
    """
    if timeout is None and thread.get_ident() == mainThreadId:
      # This is a HACK for Thread.join() if we are in the main thread.
      # In that case, a Thread.join(timeout=None) would hang and even not respond to signals
      # because signals will get delivered to other threads and Python would forward
      # them for delayed handling to the main thread which hangs.
      # See CPython signalmodule.c.
      # Currently the best solution I can think of:
      while threadObj.isAlive():
        join_orig(threadObj, timeout=0.1)
    else:
      # In all other cases, we can use the original.
      join_orig(threadObj, timeout=timeout)
  threading.Thread.join = join_hacked

  # Mostly the same for Condition.wait().
  cond_wait_orig = threading._Condition.wait
  def cond_wait_hacked(cond, timeout=None):
    if timeout is None and thread.get_ident() == mainThreadId:
      # Use a timeout anyway. This should not matter for the underlying code.
      cond_wait_orig(cond, timeout=0.1)
    else:
      cond_wait_orig(cond, timeout=timeout)
  threading._Condition.wait = cond_wait_hacked
