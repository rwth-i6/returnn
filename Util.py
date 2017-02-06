
import subprocess
from subprocess import CalledProcessError
import h5py
from collections import deque
import inspect
import os
import sys
import shlex
import numpy as np
import re
import time

PY3 = sys.version_info[0] >= 3

if PY3:
  import builtins
  unicode = str
  long = int
else:
  import __builtin__ as builtins
  unicode = builtins.unicode
  long = builtins.long


class NotSpecified(object):
  """
  This is just a placeholder, to be used as default argument to mark that it is not specified.
  """


def is_64bit_platform():
  """
  :return: True if we run on 64bit, False for 32bit
  :rtype: bool
  http://stackoverflow.com/questions/1405913/how-do-i-determine-if-my-python-shell-is-executing-in-32bit-or-64bit-mode-on-os
  """
  return sys.maxsize > 2**32


class BackendEngine:
  Theano = 0
  Default = Theano
  TensorFlow = 1
  selectedEngine = None

  @classmethod
  def select_engine(cls, engine=None, config=None):
    """
    :param int engine:
    :param Config.Config config:
    """
    assert cls.selectedEngine is None, "already set"
    if engine is None:
      if config is None:
        from Config import get_global_config
        config = get_global_config()
      engine = cls.Default
      if config.bool("use_theano", False):
        engine = cls.Theano
      if config.bool("use_tensorflow", False):
        engine = cls.TensorFlow
    cls.selectedEngine = engine

  @classmethod
  def get_selected_engine(cls):
    if cls.selectedEngine is not None:
      return cls.selectedEngine
    else:
      return cls.Default

  @classmethod
  def is_theano_selected(cls):
    return cls.get_selected_engine() == cls.Theano

  @classmethod
  def is_tensorflow_selected(cls):
    return cls.get_selected_engine() == cls.TensorFlow


def cmd(s):
  """
  :type s: str
  :rtype: list[str]
  :returns all stdout splitted by newline. Does not cover stderr.
  Raises CalledProcessError on error.
  """
  p = subprocess.Popen(s, stdout=subprocess.PIPE, shell=True, close_fds=True,
                       env=dict(os.environ, LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8"))
  result = [ tag.strip() for tag in p.communicate()[0].split('\n')[:-1]]
  p.stdout.close()
  if p.returncode != 0:
    raise CalledProcessError(p.returncode, s, "\n".join(result))
  return result


def sysexecOut(*args, **kwargs):
  from subprocess import Popen, PIPE
  kwargs.setdefault("shell", False)
  p = Popen(args, stdin=PIPE, stdout=PIPE, **kwargs)
  out, _ = p.communicate()
  if p.returncode != 0: raise CalledProcessError(p.returncode, args)
  out = out.decode("utf-8")
  return out

def sysexecRetCode(*args, **kwargs):
  import subprocess
  res = subprocess.call(args, shell=False, **kwargs)
  valid = kwargs.get("valid", (0,1))
  if valid is not None:
    if res not in valid: raise CalledProcessError(res, args)
  return res

def git_commitRev(commit="HEAD", gitdir="."):
  if commit is None: commit = "HEAD"
  return sysexecOut("git", "rev-parse", "--short", commit, cwd=gitdir).strip()

def git_isDirty(gitdir="."):
  r = sysexecRetCode("git", "diff", "--no-ext-diff", "--quiet", "--exit-code", cwd=gitdir)
  if r == 0: return False
  if r == 1: return True
  assert False, "bad return %i" % r

def git_commitDate(commit="HEAD", gitdir="."):
  return sysexecOut("git", "show", "-s", "--format=%ci", commit, cwd=gitdir).strip()[:-6].replace(":", "").replace("-", "").replace(" ", ".")

def git_describeHeadVersion(gitdir="."):
  cdate = git_commitDate(gitdir=gitdir)
  rev = git_commitRev(gitdir=gitdir)
  is_dirty = git_isDirty(gitdir=gitdir)
  return "%s--git-%s%s" % (cdate, rev, "-dirty" if is_dirty else "")

def describe_crnn_version():
  mydir = os.path.dirname(__file__)
  try:
    return git_describeHeadVersion(gitdir=mydir)
  except Exception as e:
    return "unknown(git exception: %r)" % e

def describe_theano_version():
  import theano
  try:
    tdir = os.path.dirname(theano.__file__)
  except Exception as e:
    tdir = "<unknown(exception: %r)>" % e
  try:
    version = theano.__version__
    if len(version) > 20:
      version = version[:20] + "..."
  except Exception as e:
    version = "<unknown(exception: %r)>" % e
  try:
    if tdir.startswith("<"):
      git_info = "<unknown-dir>"
    elif os.path.exists(tdir + "/../.git"):
      git_info = "git:" + git_describeHeadVersion(gitdir=tdir)
    elif "/site-packages/" in tdir:
      git_info = "<site-package>"
    else:
      git_info = "<not-under-git>"
  except Exception as e:
    git_info = "<unknown(git exception: %r)>" % e
  return "%s (%s in %s)" % (version, git_info, tdir)

def describe_tensorflow_version():
  import tensorflow as tf
  try:
    tdir = os.path.dirname(tf.__file__)
  except Exception as e:
    tdir = "<unknown(exception: %r)>" % e
  try:
    version = tf.__git_version__
  except Exception:
    try:
      version = tf.__version__
    except Exception as e:
      version = "<unknown(exception: %r)>" % e
  try:
    if tdir.startswith("<"):
      git_info = "<unknown-dir>"
    elif os.path.exists(tdir + "/../.git"):
      git_info = "git:" + git_describeHeadVersion(gitdir=tdir)
    elif "/site-packages/" in tdir:
      git_info = "<site-package>"
    else:
      git_info = "<not-under-git>"
  except Exception as e:
    git_info = "<unknown(git exception: %r)>" % e
  return "%s (%s in %s)" % (version, git_info, tdir)

def eval_shell_env(token):
  if token.startswith("$"):
    return os.environ.get(token[1:], "")
  return token

def eval_shell_str(s):
  """
  :type s: str | list[str] | ()->str | list[()->str] | ()->list[str] | ()->list[()->str]
  :rtype: list[str]

  Parses `s` as shell like arguments (via shlex.split) and evaluates shell environment variables (eval_shell_env).
  `s` or its elements can also be callable. In those cases, they will be called and the returned value is used.
  """
  tokens = []
  if callable(s):
    s = s()
  if isinstance(s, (list, tuple)):
    l = s
  else:
    assert isinstance(s, (str, unicode))
    l = shlex.split(s)
  for token in l:
    if callable(token):
      token = token()
    assert isinstance(token, (str, unicode))
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
  try:
    S=max([len(d) for d in data])
    dset = handle.create_dataset(name, (len(data),), dtype="S"+str(S))
    dset[...] = data
  except Exception:
    dt = h5py.special_dtype(vlen=unicode)
    del handle[name]
    dset = handle.create_dataset(name, (len(data),), dtype=dt)
    dset[...] = data

def model_epoch_from_filename(filename):
  if BackendEngine.is_theano_selected():
    return hdf5_dimension(filename, 'epoch')
  else:
    # We could check via:
    # tf.contrib.framework.python.framework.checkpoint_utils.load_variable()
    # once we save that in the model.
    # See TFNetwork.Network._create_saver().
    # For now, just parse it from filename.
    m = re.match(".*\.([0-9]+)", filename)
    assert m, "no match for %r" % filename
    return int(m.groups()[0])


def terminal_size(): # this will probably work on linux only
  import os, sys
  if not hasattr(sys.stdout, "fileno"):
    return -1, -1
  if not os.isatty(sys.stdout.fileno()):
    return -1, -1
  env = os.environ
  def ioctl_GWINSZ(fd):
    try:
      import fcntl, termios, struct, os
      cr = struct.unpack('hh', fcntl.ioctl(fd, termios.TIOCGWINSZ,'1234'))
    except Exception:
        return
    return cr
  cr = ioctl_GWINSZ(0) or ioctl_GWINSZ(1) or ioctl_GWINSZ(2)
  if not cr:
    try:
        fd = os.open(os.ctermid(), os.O_RDONLY)
        cr = ioctl_GWINSZ(fd)
        os.close(fd)
    except Exception:
        pass
  if not cr:
    cr = (env.get('LINES', 25), env.get('COLUMNS', 80))
  return int(cr[1]), int(cr[0])


def hms(s):
  m, s = divmod(s, 60)
  h, m = divmod(m, 60)
  return "%d:%02d:%02d" % (h, m, s)

def human_size(n, factor=1000, frac=0.8, prec=1):
  postfixs = ["", "K", "M", "G", "T"]
  i = 0
  while i < len(postfixs) - 1 and n > (factor ** (i + 1)) * frac:
    i += 1
  if i == 0:
    return str(n)
  return ("%." + str(prec) + "f") % (float(n) / (factor ** i)) + postfixs[i]


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


class _progress_bar_with_time_stats:
  start_time = None
  last_complete = None

def progress_bar_with_time(complete=1.0, prefix="", **kwargs):
  stats = _progress_bar_with_time_stats
  if stats.start_time is None:
    stats.start_time = time.time()
    stats.last_complete = complete
  if stats.last_complete > complete:
    stats.start_time = time.time()
  stats.last_complete = complete

  start_elapsed = time.time() - stats.start_time
  if complete > 0:
    total_time_estimated = start_elapsed / complete
    remaining_estimated = total_time_estimated - start_elapsed
    if prefix:
      prefix += ", " + hms(remaining_estimated)
    else:
      prefix = hms(remaining_estimated)
  progress_bar(complete, prefix=prefix, **kwargs)


def availablePhysicalMemoryInBytes():
  try:
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
  except Exception:
    mem_bytes = 1024 ** 4  # just some random number, 1TB
  return mem_bytes

def defaultCacheSizeInGBytes(factor=0.7):
  mem_gbytes = availablePhysicalMemoryInBytes() / (1024. ** 3)
  return int(mem_gbytes * factor)


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
    if not isinstance(item, (str, unicode)):
      raise KeyError(e)
    try:
      return getattr(self.__obj, item)
    except AttributeError as e:
      raise KeyError(e)

  def items(self):
    return vars(self.__obj).items()


class DictAsObj:
  def __init__(self, dikt):
    self.__dict__ = dikt


def dict_joined(*ds):
  res = {}
  for d in ds:
    res.update(d)
  return res


def obj_diff_str(self, other):
  if self is None and other is None:
    return "No diff."
  if self is None and other is not None:
    return "self is None and other is %r" % other
  if self is not None and other is None:
    return "other is None and self is %r" % self
  if self == other:
    return "No diff."
  s = []
  def _obj_attribs(obj):
    d = getattr(obj, "__dict__", None)
    if d is not None:
      return d.keys()
    return None
  self_attribs = _obj_attribs(self)
  other_attribs = _obj_attribs(other)
  if self_attribs is None or other_attribs is None:
    return "self: %r, other: %r" % (self, other)
  for attrib in sorted(set(self_attribs).union(other_attribs)):
    if attrib not in self_attribs or attrib not in other_attribs:
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
    elif isinstance(value_self, dict):
      if not isinstance(value_other, dict):
        s += ["attrib %r self is dict but other is %r" % (attrib, type(value_other))]
      elif value_self != value_other:
        s += ["attrib %r dict differs:" % attrib]
        s += ["  " + l for l in dict_diff_str(value_self, value_other).splitlines()]
    else:
      if value_self != value_other:
        s += ["attrib %r differ. self: %r, other: %r" % (attrib, value_self, value_other)]
  if s:
    return "\n".join(s)
  else:
    return "No diff."


def dict_diff_str(self, other):
  return obj_diff_str(DictAsObj(self), DictAsObj(other))


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
  def cond_wait_hacked(cond, timeout=None, *args):
    if timeout is None and thread.get_ident() == mainThreadId:
      # Use a timeout anyway. This should not matter for the underlying code.
      cond_wait_orig(cond, timeout=0.1)
    else:
      cond_wait_orig(cond, timeout=timeout)
  threading._Condition.wait = cond_wait_hacked


def start_daemon_thread(target, args=()):
  from threading import Thread
  t = Thread(target=target, args=args)
  t.daemon = True
  t.start()

def is_quitting():
  import rnn
  if rnn.quit:  # via rnn.finalize()
    return True
  if getattr(sys, "exited", False):  # set via Debug module when an unexpected SIGINT occurs, or here
    return True
  return False

def interrupt_main():
  import thread
  import threading
  is_main_thread = isinstance(threading.currentThread(), threading._MainThread)
  if is_quitting():  # ignore if we are already quitting
    if is_main_thread:  # strange to get again in main thread
      raise Exception("interrupt_main() from main thread while already quitting")
    # Not main thread. This will just exit the thread.
    sys.exit(1)
  sys.exited = True  # Don't do it twice.
  sys.exited_frame = sys._getframe()
  if is_main_thread:
    raise KeyboardInterrupt
  else:
    thread.interrupt_main()
    sys.exit(1)  # And exit the thread.


def try_run(func, args=(), catch_exc=Exception, default=None):
  try:
    return func(*args)
  except catch_exc:
    return default


def class_idx_seq_to_1_of_k(seq, num_classes):
  num_frames = len(seq)
  m = np.zeros((num_frames, num_classes))
  m[np.arange(num_frames), seq] = 1
  return m


def uniq(seq):
  """
  Like Unix tool uniq. Removes repeated entries.
  :param seq: numpy.array
  :return: seq
  """
  diffs = np.ones_like(seq)
  diffs[1:] = seq[1:] - seq[:-1]
  idx = diffs.nonzero()
  return seq[idx]


_have_inplace_increment = None
_native_inplace_increment = None

def inplace_increment(x, idx, y):
  """
  This basically does `x[idx] += y`.
  The difference to the Numpy version is that in case some index is there multiple
  times, it will only be incremented once (and it is not specified which one).
  See also theano.tensor.subtensor.AdvancedIncSubtensor documentation.
  """
  global _have_inplace_increment, inplace_increment, _native_inplace_increment
  if _have_inplace_increment is None:
    native_inpl_incr = None
    import theano
    if theano.config.cxx:
      import theano.gof.cutils  # needed to import cutils_ext
      try:
        from cutils_ext.cutils_ext import inplace_increment as native_inpl_incr
      except ImportError:
        pass
    if native_inpl_incr:
      _have_inplace_increment = True
      _native_inplace_increment = native_inpl_incr
      inplace_increment = native_inpl_incr  # replace myself
      return inplace_increment(x, idx, y)
    _have_inplace_increment = False
  if _have_inplace_increment is True:
    return _native_inplace_increment(x, idx, y)
  raise NotImplementedError("need Numpy 1.8 or later")



def parse_orthography_into_symbols(orthography, upper_case_special=True, word_based=False):
  """
  For Speech.
  Example:
    orthography = "hello [HESITATION] there "
    with word_based == False: returns list("hello ") + ["[HESITATION]"] + list(" there ").
    with word_based == True: returns ["hello", "[HESITATION]", "there"]
  No pre/post-processing such as:
  Spaces are kept as-is. No stripping at begin/end. (E.g. trailing spaces are not removed.)
  No tolower/toupper.
  Doesn't add [BEGIN]/[END] symbols or so.
  Any such operations should be done explicitly in an additional function.
  Anything in []-brackets are meant as special-symbols.
  Also see parse_orthography() which includes some preprocessing.

  :param str orthography: example: "hello [HESITATION] there "
  :param bool upper_case_special: whether the special symbols are always made upper case
  :param bool word_based: whether we split on space and return full words
  :rtype: list[str]
  """
  ret = []
  in_special = 0
  for c in orthography:
    if in_special:
      if c == "[":  # special-special
        in_special += 1
        ret[-1] += "["
      elif c == "]":
        in_special -= 1
        ret[-1] += "]"
      elif upper_case_special:
        ret[-1] += c.upper()
      else:
        ret[-1] += c
    else:  # not in_special
      if c == "[":
        in_special = 1
        ret += ["["]
      else:
        if word_based:
          if c.isspace():
            ret += [""]
          else:
            if not ret:
              ret += [""]
            ret[-1] += c
        else:  # not word_based
          ret += c
  return ret


def parse_orthography(orthography, prefix=(), postfix=("[END]",),
                      remove_chars="(){}", collapse_spaces=True, final_strip=True,
                      **kwargs):
  """
  For Speech. Full processing.
  Example:
    orthography = "hello [HESITATION] there "
    with word_based == False: returns list("hello ") + ["[HESITATION]"] + list(" there") + ["[END]"]
    with word_based == True: returns ["hello", "[HESITATION]", "there", "[END]"]
  Does some preprocessing on orthography and then passes it on to parse_orthography_into_symbols().

  :param str orthography: e.g. "hello [HESITATION] there "
  :param list[str] prefix: will add this prefix
  :param list[str] postfix: will add this postfix
  :param str remove_chars: those chars will just be removed at the beginning
  :param bool collapse_spaces: whether multiple spaces and tabs are collapsed into a single space
  :param bool final_strip: whether we strip left and right
  :param dict[str] kwargs: passed on to parse_orthography_into_symbols()
  :rtype: list[str]
  """
  for c in remove_chars:
    orthography = orthography.replace(c, "")
  if collapse_spaces:
    orthography = " ".join(orthography.split())
  if final_strip:
    orthography = orthography.strip()
  return list(prefix) + parse_orthography_into_symbols(orthography, **kwargs) + list(postfix)


def json_remove_comments(string, strip_space=True):
  """
  :type string: str
  :rtype: str

  via https://github.com/getify/JSON.minify/blob/master/minify_json.py,
  by Gerald Storer, Pradyun S. Gedam, modified by us.
  """
  tokenizer = re.compile('"|(/\*)|(\*/)|(//)|\n|\r')
  end_slashes_re = re.compile(r'(\\)*$')

  in_string = False
  in_multi = False
  in_single = False

  new_str = []
  index = 0

  for match in re.finditer(tokenizer, string):

    if not (in_multi or in_single):
      tmp = string[index:match.start()]
      if not in_string and strip_space:
        # replace white space as defined in standard
        tmp = re.sub('[ \t\n\r]+', '', tmp)
      new_str.append(tmp)

    index = match.end()
    val = match.group()

    if val == '"' and not (in_multi or in_single):
      escaped = end_slashes_re.search(string, 0, match.start())

      # start of string or unescaped quote character to end string
      if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):
        in_string = not in_string
      index -= 1 # include " character in next catch
    elif not (in_string or in_multi or in_single):
      if val == '/*':
        in_multi = True
      elif val == '//':
        in_single = True
    elif val == '*/' and in_multi and not (in_string or in_single):
      in_multi = False
    elif val in '\r\n' and not (in_multi or in_string) and in_single:
      in_single = False
    elif not ((in_multi or in_single) or (val in ' \r\n\t' and strip_space)):
      new_str.append(val)

  new_str.append(string[index:])
  return ''.join(new_str)


def load_json(filename=None, content=None):
  if content:
    assert not filename
  else:
    content = open(filename).read()
  import json
  content = json_remove_comments(content)
  try:
    json_content = json.loads(content)
  except ValueError as e:
    raise Exception("config looks like JSON but invalid json content, %r" % e)
  return json_content


class NumbersDict:
  """
  It's mostly like dict[str,float|int] & some optional broadcast default value.
  It implements the standard math bin ops in a straight-forward way.
  """

  def __init__(self, auto_convert=None, numbers_dict=None, broadcast_value=None):
    if auto_convert is not None:
      assert broadcast_value is None
      assert numbers_dict is None
      if isinstance(auto_convert, dict):
        numbers_dict = auto_convert
      elif isinstance(auto_convert, NumbersDict):
        numbers_dict = auto_convert.dict
        broadcast_value = auto_convert.value
      else:
        broadcast_value = auto_convert
    if numbers_dict is None:
      numbers_dict = {}
    else:
      numbers_dict = dict(numbers_dict)  # force copy

    self.dict = numbers_dict
    self.value = broadcast_value
    self.max = self.__max_error

  def copy(self):
    return NumbersDict(self)

  @property
  def keys_set(self):
    return set(self.dict.keys())

  def __getitem__(self, key):
    return self.dict.get(key, self.value)

  def __setitem__(self, key, value):
    self.dict[key] = value

  def __delitem__(self, key):
    del self.dict[key]

  def get(self, key, default=None):
    return self.dict.get(key, default)

  def pop(self, key, *args):
    return self.dict.pop(key, *args)

  def __iter__(self):
    # This can potentially cause confusion. So enforce explicitness.
    # For a dict, we would return the dict keys here.
    # Also, max(self) would result in a call to self.__iter__(),
    # which would only make sense for our values, not the dict keys.
    raise Exception("%s.__iter__ is undefined" % self.__class__.__name__)

  def keys(self):
    return self.dict.keys()

  def values(self):
    return list(self.dict.values()) + ([self.value] if self.value is not None else [])

  def has_values(self):
    return bool(self.dict) or self.value is not None

  @classmethod
  def bin_op_scalar_optional(cls, self, other, zero, op):
    if self is None and other is None:
      return None
    if self is None:
      self = zero
    if other is None:
      other = zero
    return op(self, other)

  @classmethod
  def bin_op(cls, self, other, op, zero, result=None):
    if not isinstance(self, NumbersDict):
      self = NumbersDict(self)
    if not isinstance(other, NumbersDict):
      other = NumbersDict(other)
    if result is None:
      result = NumbersDict()
    assert isinstance(result, NumbersDict)
    for k in self.keys_set | other.keys_set:
      result[k] = cls.bin_op_scalar_optional(self[k], other[k], zero=zero, op=op)
    result.value = cls.bin_op_scalar_optional(self.value, other.value, zero=zero, op=op)
    return result

  def __add__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a + b, zero=0)

  __radd__ = __add__

  def __iadd__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a + b, zero=0, result=self)

  def __sub__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a - b, zero=0)

  def __rsub__(self, other):
    return self.bin_op(self, other, op=lambda a, b: b - a, zero=0)

  def __isub__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a - b, zero=0, result=self)

  def __mul__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a * b, zero=1)

  __rmul__ = __mul__

  def __imul__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a * b, zero=1, result=self)

  def __div__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a / b, zero=1)

  def __rdiv__(self, other):
    return self.bin_op(self, other, op=lambda a, b: b / a, zero=1)

  def __idiv__(self, other):
    return self.bin_op(self, other, op=lambda a, b: a / b, zero=1, result=self)

  def elem_eq(self, other, result_with_default=False):
    """
    Element-wise equality check with other.
    Note about broadcast default value: Consider some key which is neither in self nor in other.
      This means that self[key] == self.default, other[key] == other.default.
      Thus, in case that self.default != other.default, we get res.default == False.
      Then, all(res.values()) == False, even when all other values are True.
      This is often not what we want.
      You can control the behavior via result_with_default.
    """
    res = self.bin_op(self, other, op=lambda a, b: a == b, zero=None)
    if not result_with_default:
      res.value = None
    return res

  def __eq__(self, other):
    return all(self.elem_eq(other).values())

  def __ne__(self, other):
    return not (self == other)

  def __cmp__(self, other):
    # There is no good straight-forward implementation
    # and it would just confuse.
    raise Exception("%s.__cmp__ is undefined" % self.__class__.__name__)

  @classmethod
  def max(cls, items):
    """
    Element-wise maximum for item in items.
    """
    if not items:
      return None
    if len(items) == 1:
      return items[0]
    if len(items) == 2:
      # max(x, None) == x, so this works.
      return cls.bin_op(items[0], items[1], op=max, zero=None)
    return cls.max([items[0], cls.max(items[1:])])

  @classmethod
  def min(cls, items):
    """
    Element-wise minimum for item in items.
    """
    if not items:
      return None
    if len(items) == 1:
      return items[0]
    if len(items) == 2:
      return cls.bin_op(items[0], items[1], op=min, zero=None)
    return cls.min([items[0], cls.min(items[1:])])

  @staticmethod
  def __max_error():
    # Will replace self.max for each instance. To be sure that we don't confuse it with self.max_value.
    raise Exception("Use max_value instead.")

  def max_value(self):
    """
    Maximum of our values.
    """
    return max(self.values())

  def __repr__(self):
    if self.value is None and not self.dict:
      return "%s()" % self.__class__.__name__
    if self.value is None and self.dict:
      return "%s(%r)" % (self.__class__.__name__, self.dict)
    if not self.dict and self.value is not None:
      return "%s(%r)" % (self.__class__.__name__, self.value)
    return "%s(numbers_dict=%r, broadcast_value=%r)" % (
           self.__class__.__name__, self.dict, self.value)


def collect_class_init_kwargs(cls):
  kwargs = set()
  for cls_ in inspect.getmro(cls):
    if not inspect.ismethod(cls_.__init__):  # Python function. could be builtin func or so
      continue
    arg_spec = inspect.getargspec(cls_.__init__)
    kwargs.update(arg_spec.args[1:])  # first arg is self, ignore
  return kwargs


def custom_exec(source, source_filename, user_ns, user_global_ns):
  if not source.endswith("\n"):
    source += "\n"
  co = compile(source, source_filename, "exec")
  user_global_ns["__package__"] = __package__  # important so that imports work when CRNN itself is loaded as a package
  eval(co, user_global_ns, user_ns)


class FrozenDict(dict):
  def __setitem__(self, key, value):
    raise ValueError("FrozenDict cannot be modified")

  def __hash__(self):
    return hash(tuple(sorted(self.items())))


def make_hashable(obj):
  """
  Theano needs hashable objects in some cases, e.g. the properties of Ops.
  This converts all objects as such, i.e. into immutable frozen types.
  """
  if isinstance(obj, dict):
    return FrozenDict([make_hashable(item) for item in obj.items()])
  elif isinstance(obj, (list, tuple)):
    return tuple([make_hashable(item) for item in obj])
  elif isinstance(obj, (str, unicode, float, int, long)):
    return obj
  elif obj is None:
    return obj
  else:
    assert False, "don't know how to make hashable: %r (%r)" % (obj, type(obj))


def make_dll_name(basename):
  if sys.platform == "darwin":
    return "lib%s.dylib" % basename
  elif sys.platform == "win32":
    return "%s.dll" % basename
  else:  # Linux, Unix
    return "lib%s.so" % basename


def escape_c_str(s):
  return '"%s"' % s.replace("\\\\", "\\").replace("\n", "\\n").replace("\"", "\\\"").replace("'", "\\'")


def attr_chain(base, attribs):
  if not isinstance(attribs, (list, tuple)):
    assert isinstance(attribs, str)
    attribs = [attribs]
  else:
    attribs = list(attribs)
  for i in range(len(attribs)):
    base = getattr(base, attribs[i])
  return base


def to_bool(v):
  try:
    return bool(int(v))
  except ValueError:
    pass
  if isinstance(v, (str, unicode)):
    v = v.lower()
    if v in ["true", "yes", "on", "1"]: return True
    if v in ["false", "no", "off", "0"]: return False
  raise ValueError("to_bool cannot handle %r" % v)


def as_str(s):
  if isinstance(s, str):
    return s
  if isinstance(s, bytes):
    return s.decode("utf8")
  assert False, "unknown type %s" % type(s)


def load_txt_vector(filename):
  """
  Expect line-based text encoding in file.
  We also support Sprint XML format, which has some additional xml header and footer,
  which we will just strip away.
  """
  return [float(l) for l in open(filename).read().splitlines() if l and not l.startswith("<")]


class CollectionReadCheckCovered:
  def __init__(self, collection):
    self.collection = collection
    self.got_items = set()

  def __getitem__(self, item):
    res = self.collection[item]
    self.got_items.add(item)
    return res

  def get(self, item, default=None):
    try:
      return self[item]
    except KeyError:
      return default

  def __len__(self):
    return len(self.collection)

  def __iter__(self):
    for k in self.collection:
      yield self[k]

  def assert_all_read(self):
    remaining = set(self.collection).difference(self.got_items)
    assert not remaining, "The keys %r were not read in the collection %r." % (remaining, self.collection)


def which(program):
  def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

  fpath, fname = os.path.split(program)
  if fpath:
    if is_exe(program):
      return program
  else:
    for path in os.environ["PATH"].split(os.pathsep):
      path = path.strip('"')
      exe_file = os.path.join(path, program)
      if is_exe(exe_file):
        return exe_file

  return None

_original_execv = None
_original_execve = None
_original_execvpe = None

def overwrite_os_exec(prefix_args):
  """
  :param list[str] prefix_args:
  """
  global _original_execv, _original_execve, _original_execvpe
  if not _original_execv:
    _original_execv = os.execv
  if not _original_execve:
    _original_execve = os.execve
  if not _original_execvpe:
    _original_execvpe = os._execvpe

  def wrapped_execvpe(file, args, env=None):
    new_args = prefix_args + [which(args[0])] + args[1:]
    sys.stderr.write("$ %s\n" % " ".join(new_args))
    sys.stderr.flush()
    _original_execvpe(file=prefix_args[0], args=new_args, env=env)

  def execv(path, args):
    if args[:len(prefix_args)] == prefix_args:
      _original_execv(path, args)
    else:
      wrapped_execvpe(path, args)

  def execve(path, args, env):
    if args[:len(prefix_args)] == prefix_args:
      _original_execve(path, args, env)
    else:
      wrapped_execvpe(path, args, env)

  def execl(file, *args):
    """execl(file, *args)

    Execute the executable file with argument list args, replacing the
    current process. """
    os.execv(file, args)

  def execle(file, *args):
    """execle(file, *args, env)

    Execute the executable file with argument list args and
    environment env, replacing the current process. """
    env = args[-1]
    os.execve(file, args[:-1], env)

  def execlp(file, *args):
    """execlp(file, *args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process. """
    os.execvp(file, args)

  def execlpe(file, *args):
    """execlpe(file, *args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env, replacing the current
    process. """
    env = args[-1]
    os.execvpe(file, args[:-1], env)

  def execvp(file, args):
    """execvp(file, args)

    Execute the executable file (which is searched for along $PATH)
    with argument list args, replacing the current process.
    args may be a list or tuple of strings. """
    wrapped_execvpe(file, args)

  def execvpe(file, args, env):
    """execvpe(file, args, env)

    Execute the executable file (which is searched for along $PATH)
    with argument list args and environment env , replacing the
    current process.
    args may be a list or tuple of strings. """
    wrapped_execvpe(file, args, env)

  os.execv = execv
  os.execve = execve
  os.execl = execl
  os.execle = execle
  os.execlp = execlp
  os.execlpe = execlpe
  os.execvp = execvp
  os.execvpe = execvpe
  os._execvpe = wrapped_execvpe


def get_lsb_release():
  d = {}
  for l in open("/etc/lsb-release").read().splitlines():
    k, v = l.split("=", 1)
    if v[0] == v[-1] == "\"":
      v = v[1:-1]
    d[k] = v
  return d


def get_ubuntu_major_version():
  """
  :rtype: int|None
  """
  d = get_lsb_release()
  if d["DISTRIB_ID"] != "Ubuntu":
    return None
  return int(float(d["DISTRIB_RELEASE"]))


def auto_prefix_os_exec_prefix_ubuntu(prefix_args, ubuntu_min_version=16):
  """
  :param list[str] prefix_args:
  :param int ubuntu_min_version:

  Example usage:
    auto_prefix_os_exec_prefix_ubuntu(["/u/zeyer/tools/glibc217/ld-linux-x86-64.so.2"])
  """
  ubuntu_version = get_ubuntu_major_version()
  if ubuntu_version is None:
    return
  if ubuntu_version >= ubuntu_min_version:
    return
  print("You are running Ubuntu %i, thus we prefix all os.exec with %s." % (ubuntu_version, prefix_args))
  assert os.path.exists(prefix_args[0])
  overwrite_os_exec(prefix_args=prefix_args)


def cleanup_env_var_path(env_var, path_prefix):
  """
  :param str env_var: e.g. "LD_LIBRARY_PATH"
  :param str path_prefix:

  Will remove all paths in os.environ[env_var] which are prefixed with path_prefix.
  """
  if env_var not in os.environ:
    return
  ps = os.environ[env_var].split(":")
  def f(p):
    if p == path_prefix or p.startswith(path_prefix + "/"):
      print("Removing %s from %s." % (p, env_var))
      return False
    return True
  ps = filter(f, ps)
  os.environ[env_var] = ":".join(ps)


def get_login_username():
  """
  :rtype: str
  :return: the username of the current user.
  Use this as a replacement for os.getlogin().
  """
  import pwd, os
  return pwd.getpwuid(os.getuid())[0]


def get_temp_dir():
  """
  :rtype: str
  :return: e.g. "/tmp/$USERNAME"
  """
  username = get_login_username()
  for envname in ['TMPDIR', 'TEMP', 'TMP']:
    dirname = os.getenv(envname)
    if dirname:
      return "%s/%s" % (dirname, username)
  return "/tmp/%s" % username
