
"""
Here are all subprocess, threading etc related utilities,
most of them quite low level.
"""

from threading import Lock, currentThread
import sys
import os
from StringIO import StringIO
from contextlib import contextmanager
import pickle
import types
import marshal
import importlib


def execInMainProc(func):
  global isMainProcess
  if isMainProcess:
    return func()
  else:
    assert _AsyncCallQueue.Self, "works only if called via asyncCall"
    return _AsyncCallQueue.Self.asyncExecClient(func)


def ExecInMainProcDecorator(func):
  def decoratedFunc(*args, **kwargs):
    return execInMainProc(lambda: func(*args, **kwargs))
  return decoratedFunc


class AsyncInterrupt(BaseException):
  pass


class ForwardedKeyboardInterrupt(Exception):
  pass


class _AsyncCallQueue:
  Self = None

  class Types:
    result = 0
    exception = 1
    asyncExec = 2

  def __init__(self, queue):
    assert not self.Self
    self.__class__.Self = self
    self.mutex = Lock()
    self.queue = queue

  def put(self, type, value):
    self.queue.put((type, value))

  def asyncExecClient(self, func):
    with self.mutex:
      self.put(self.Types.asyncExec, func)
      t, value = self.queue.get()
      if t == self.Types.result:
        return value
      elif t == self.Types.exception:
        raise value
      else:
        assert False, "bad behavior of asyncCall in asyncExec (%r)" % t

  @classmethod
  def asyncExecHost(clazz, task, func):
    q = task
    name = "<unknown>"
    try:
      name = repr(func)
      res = func()
    except Exception as exc:
      print "Exception in asyncExecHost", name, exc
      q.put((clazz.Types.exception, exc))
    else:
      try:
        q.put((clazz.Types.result, res))
      except IOError:
        # broken pipe or so. parent quit. treat like a SIGINT
        raise KeyboardInterrupt


def asyncCall(func, name=None, mustExec=False):
  """
  This executes func() in another process and waits/blocks until
  it is finished. The returned value is passed back to this process
  and returned. Exceptions are passed back as well and will be
  reraised here.

  If `mustExec` is set, the other process must `exec()` after the `fork()`.
  If it is not set, it might omit the `exec()`, depending on the platform.
  """

  def doCall(queue):
    q = _AsyncCallQueue(queue)
    try:
      try:
        res = func()
      except KeyboardInterrupt as exc:
        print "Exception in asyncCall", name, ": KeyboardInterrupt"
        q.put(q.Types.exception, ForwardedKeyboardInterrupt(exc))
      except BaseException as exc:
        print "Exception in asyncCall", name
        sys.excepthook(*sys.exc_info())
        q.put(q.Types.exception, exc)
      else:
        q.put(q.Types.result, res)
    except (KeyboardInterrupt, ForwardedKeyboardInterrupt):
      print "asyncCall: SIGINT in put, probably the parent died"
    # ignore

  task = AsyncTask(func=doCall, name=name, mustExec=mustExec)

  while True:
    # If there is an unhandled exception in doCall or the process got killed/segfaulted or so,
    # this will raise an EOFError here.
    # However, normally, we should catch all exceptions and just reraise them here.
    t,value = task.get()
    if t == _AsyncCallQueue.Types.result:
      return value
    elif t == _AsyncCallQueue.Types.exception:
      raise value
    elif t == _AsyncCallQueue.Types.asyncExec:
      _AsyncCallQueue.asyncExecHost(task, value)
    else:
      assert False, "unknown _AsyncCallQueue type %r" % t



def attrChain(base, *attribs, **kwargs):
  default = kwargs.get("default", None)
  obj = base
  for attr in attribs:
    if obj is None: return default
    obj = getattr(obj, attr, None)
  if obj is None: return default
  return obj


# This is needed in some cases to avoid pickling problems with bounded funcs.
def funcCall(attrChainArgs, args=()):
  f = attrChain(*attrChainArgs)
  return f(*args)


Unpickler = pickle.Unpickler
CellType = type((lambda x: lambda: x)(0).func_closure[0])

def makeFuncCell(value):
  return (lambda: value).func_closure[0]

def getModuleDict(modname):
  mod = importlib.import_module(modname)
  return mod.__dict__

def getModNameForModDict(obj):
  mods = {id(mod.__dict__): modname for (modname, mod) in sys.modules.items() if mod}
  modname = mods.get(id(obj), None)
  return modname

class Pickler(pickle.Pickler):
  """
  We extend the standard Pickler to be able to pickle some more types,
  such as lambdas and functions, code, func cells, buffer and more.
  """

  def __init__(self, *args, **kwargs):
    if not "protocol" in kwargs:
      kwargs["protocol"] = pickle.HIGHEST_PROTOCOL
    pickle.Pickler.__init__(self, *args, **kwargs)
  dispatch = pickle.Pickler.dispatch.copy()

  def save_func(self, obj):
    try:
      self.save_global(obj)
      return
    except pickle.PicklingError:
      pass
    assert type(obj) is types.FunctionType
    self.save(types.FunctionType)
    self.save((
      obj.func_code,
      obj.func_globals,
      obj.func_name,
      obj.func_defaults,
      obj.func_closure,
    ))
    self.write(pickle.REDUCE)
    self.memoize(obj)
  dispatch[types.FunctionType] = save_func

  def save_method(self, obj):
    try:
      self.save_global(obj)
      return
    except pickle.PicklingError:
      pass
    assert type(obj) is types.MethodType
    self.save(types.MethodType)
    self.save((obj.im_func, obj.im_self, obj.im_class))
    self.write(pickle.REDUCE)
    self.memoize(obj)
  dispatch[types.MethodType] = save_method

  def save_code(self, obj):
    assert type(obj) is types.CodeType
    self.save(marshal.loads)
    self.save((marshal.dumps(obj),))
    self.write(pickle.REDUCE)
    self.memoize(obj)
  dispatch[types.CodeType] = save_code

  def save_cell(self, obj):
    assert type(obj) is CellType
    self.save(makeFuncCell)
    self.save((obj.cell_contents,))
    self.write(pickle.REDUCE)
    self.memoize(obj)
  dispatch[CellType] = save_cell

  # We also search for module dicts and reference them.
  # This is for FunctionType.func_globals.
  def intellisave_dict(self, obj):
    modname = getModNameForModDict(obj)
    if modname:
      self.save(getModuleDict)
      self.save((modname,))
      self.write(pickle.REDUCE)
      self.memoize(obj)
      return
    self.save_dict(obj)
  dispatch[types.DictionaryType] = intellisave_dict

  def save_buffer(self, obj):
    self.save(buffer)
    self.save((str(obj),))
    self.write(pickle.REDUCE)
  dispatch[types.BufferType] = save_buffer

  # Some types in the types modules are not correctly referenced,
  # such as types.FunctionType. This is fixed here.
  def fixedsave_type(self, obj):
    try:
      self.save_global(obj)
      return
    except pickle.PicklingError:
      pass
    for modname in ["types"]:
      moddict = sys.modules[modname].__dict__
      for modobjname,modobj in moddict.iteritems():
        if modobj is obj:
          self.write(pickle.GLOBAL + modname + '\n' + modobjname + '\n')
          self.memoize(obj)
          return
    self.save_global(obj)
  dispatch[types.TypeType] = fixedsave_type

  # avoid pickling instances of ourself. this mostly doesn't make sense and leads to trouble.
  # however, also doesn't break. it mostly makes sense to just ignore.
  def __getstate__(self): return None
  def __setstate__(self, state): pass


class ExecingProcess:
  """
  This is a replacement for multiprocessing.Process which always
  uses fork+exec, not just fork.
  This ensures that you have a separate independent process.
  This can avoid many types of bugs, such as:
    http://stackoverflow.com/questions/24509650
    http://bugs.python.org/issue6721
    http://stackoverflow.com/questions/8110920
    http://stackoverflow.com/questions/23963997
    https://github.com/numpy/numpy/issues/654
    http://comments.gmane.org/gmane.comp.python.numeric.general/60204
  """

  def __init__(self, target, args, name):
    self.target = target
    self.args = args
    self.name = name
    self.daemon = True
    self.pid = None
    self.exit_status = None

  def start(self):
    assert self.pid is None
    assert self.exit_status is None
    def pipeOpen():
      readend, writeend = os.pipe()
      readend = os.fdopen(readend, "r")
      writeend = os.fdopen(writeend, "w")
      return readend, writeend
    self.pipe_c2p = pipeOpen()
    self.pipe_p2c = pipeOpen()
    self.parent_pid = os.getpid()
    pid = os.fork()
    if pid == 0: # child
      try:
        sys.stdin.close()  # Force no tty stdin.
        self.pipe_c2p[0].close()
        self.pipe_p2c[1].close()
        py_mod_file = os.path.splitext(__file__)[0] + ".py"
        assert os.path.exists(py_mod_file)
        binary = "python"
        full_binary_paths = [os.path.join(path, binary)
                             for path in os.environ["PATH"].split(os.pathsep)]
        full_binary_paths = filter(lambda path: os.access(path, os.X_OK), full_binary_paths)
        assert full_binary_paths, "%r not found in PATH %r" % (binary, os.environ["PATH"])
        args = [full_binary_paths[0],
                py_mod_file,
                "--forkExecProc",
                str(self.pipe_c2p[1].fileno()),
                str(self.pipe_p2c[0].fileno())]
        os.execv(args[0], args)  # Does not return if successful.
      except BaseException:
        print "ExecingProcess: Error at initialization."
        sys.excepthook(*sys.exc_info())
        sys.exit(1)
      finally:
        sys.exit()
    else: # parent
      self.pipe_c2p[1].close()
      self.pipe_p2c[0].close()
      self.pid = pid
      self.pickler = Pickler(self.pipe_p2c[1])
      self.pickler.dump(self.name)
      self.pickler.dump(self.target)
      self.pickler.dump(self.args)
      self.pipe_p2c[1].flush()

  def _wait(self, options=0):
    assert self.parent_pid == os.getpid()
    assert self.pid
    assert self.exit_status is None
    pid, exit_status = os.waitpid(self.pid, options)
    if pid != self.pid:
      assert pid == 0
      # It's still alive, otherwise we would have get the same pid.
      return
    self.exit_status = exit_status
    self.pid = None

  def is_alive(self):
    if self.pid is None:
      return False
    self._wait(os.WNOHANG)
    return self.pid is not None

  def join(self, timeout=None):
    if not self.is_alive():
      return
    if timeout:
      raise NotImplementedError
    self._wait()

  Verbose = False

  @staticmethod
  def checkExec():
    if "--forkExecProc" in sys.argv:
      import better_exchook
      better_exchook.install()
      argidx = sys.argv.index("--forkExecProc")
      writeFileNo = int(sys.argv[argidx + 1])
      readFileNo = int(sys.argv[argidx + 2])
      readend = os.fdopen(readFileNo, "r")
      writeend = os.fdopen(writeFileNo, "w")
      unpickler = Unpickler(readend)
      name = unpickler.load()
      if ExecingProcess.Verbose: print "ExecingProcess child %s (pid %i)" % (name, os.getpid())
      try:
        target = unpickler.load()
        args = unpickler.load()
      except EOFError:
        print "Error: unpickle incomplete"
        raise SystemExit
      ret = target(*args)
      # IOError is probably broken pipe. That probably means that the parent died.
      try: Pickler(writeend).dump(ret)
      except IOError: pass
      try: readend.close()
      except IOError: pass
      try: writeend.close()
      except IOError: pass
      if ExecingProcess.Verbose: print "ExecingProcess child %s (pid %i) finished" % (name, os.getpid())
      raise SystemExit


class ExecingProcess_ConnectionWrapper(object):
  """
  Wrapper around _multiprocessing.Connection.
  This is needed to use our own Pickler.
  """

  def __init__(self, fd=None):
    self.fd = fd
    if self.fd:
      from _multiprocessing import Connection
      self.conn = Connection(fd)

  def __getstate__(self): return self.fd
  def __setstate__(self, state): self.__init__(state)

  def __getattr__(self, attr): return getattr(self.conn, attr)

  def _check_closed(self): assert not self.conn.closed
  def _check_writable(self): assert self.conn.writable
  def _check_readable(self): assert self.conn.readable

  def send(self, value):
    self._check_closed()
    self._check_writable()
    buf = StringIO()
    Pickler(buf).dump(value)
    self.conn.send_bytes(buf.getvalue())

  def recv(self):
    self._check_closed()
    self._check_readable()
    buf = self.conn.recv_bytes()
    f = StringIO(buf)
    return Unpickler(f).load()


def ExecingProcess_Pipe():
  """
  This is like multiprocessing.Pipe(duplex=True).
  It uses our own ExecingProcess_ConnectionWrapper.
  """
  import socket
  s1, s2 = socket.socketpair()
  c1 = ExecingProcess_ConnectionWrapper(os.dup(s1.fileno()))
  c2 = ExecingProcess_ConnectionWrapper(os.dup(s2.fileno()))
  s1.close()
  s2.close()
  return c1, c2


isFork = False  # fork() without exec()
isMainProcess = True


class AsyncTask:
  """
  This uses multiprocessing.Process or ExecingProcess to execute some function.
  In addition, it provides a duplex pipe for communication. This is either
  multiprocessing.Pipe or ExecingProcess_Pipe.
  """

  def __init__(self, func, name=None, mustExec=False):
    """
    :param func: a function which gets a single parameter,
      which will be a reference to our instance in the fork,
      so that it can use our communication methods put/get.
    :type str name: name for the sub process
    :param bool mustExec: if True, we do fork+exec, not just fork
    """
    self.name = name or "unnamed"
    self.func = func
    self.mustExec = mustExec
    self.parent_pid = os.getpid()
    if mustExec and sys.platform != "win32":
      self.Process = ExecingProcess
      self.Pipe = ExecingProcess_Pipe
    else:
      from multiprocessing import Process, Pipe
      self.Process = Process
      self.Pipe = Pipe
    self.parent_conn, self.child_conn = self.Pipe()
    self.proc = self.Process(
      target = funcCall,
      args = ((AsyncTask, "_asyncCall"), (self,)),
      name = self.name + " worker process")
    self.proc.daemon = True
    self.proc.start()
    self.child_conn.close()
    self.child_pid = self.proc.pid
    assert self.child_pid
    self.conn = self.parent_conn

  @staticmethod
  def _asyncCall(self):
    assert self.isChild
    self.parent_conn.close()
    self.conn = self.child_conn # we are the child
    if not self.mustExec and sys.platform != "win32":
      global isFork
      isFork = True
    global isMainProcess
    isMainProcess = False
    try:
      self.func(self)
    except KeyboardInterrupt:
      print "Exception in AsyncTask", self.name, ": KeyboardInterrupt"
      sys.exit(1)
    except SystemExit:
      raise
    except BaseException:
      print "Exception in AsyncTask", self.name
      sys.excepthook(*sys.exc_info())
      sys.exit(1)
    finally:
      self.conn.close()

  def put(self, value):
    self.conn.send(value)

  def get(self):
    thread = currentThread()
    try:
      thread.waitQueue = self
      res = self.conn.recv()
    except EOFError: # this happens when the child died
      raise ForwardedKeyboardInterrupt()
    except Exception:
      raise
    finally:
      thread.waitQueue = None
    return res

  @property
  def isParent(self):
    return self.parent_pid == os.getpid()

  @property
  def isChild(self):
    if self.isParent: return False
    assert self.parent_pid == os.getppid()
    return True

  # This might be called from the module code.
  # See OnRequestQueue which implements the same interface.
  def setCancel(self):
    self.conn.close()
    if self.isParent and self.child_pid:
      import signal
      try:
        os.kill(self.child_pid, signal.SIGINT)
      except OSError:
        # Could be that the process already died or so. Just ignore and assume it is dead.
        pass
      self.child_pid = None

  terminate = setCancel  # alias

  def join(self, timeout=None):
    return self.proc.join(timeout=timeout)

  def is_alive(self):
    return self.proc.is_alive()


def WarnMustNotBeInForkDecorator(func):
  class Ctx:
    didWarn = False
  def decoratedFunc(*args, **kwargs):
    global isFork
    if isFork:
      if not Ctx.didWarn:
        print "Must not be in fork!"
        Ctx.didWarn = True
      return None
    return func(*args, **kwargs)
  return decoratedFunc


class ReadWriteLock(object):
  """Classic implementation of ReadWriteLock.
  Note that this partly supports recursive lock usage:
  - Inside a readlock, a writelock will always block!
  - Inside a readlock, another readlock is fine.
  - Inside a writelock, any other writelock or readlock is fine.
  """
  def __init__(self):
    import threading
    self.lock = threading.RLock()
    self.writeReadyCond = threading.Condition(self.lock)
    self.readerCount = 0
  @property
  @contextmanager
  def readlock(self):
    with self.lock:
      self.readerCount += 1
    try: yield
    finally:
      with self.lock:
        self.readerCount -= 1
        if self.readerCount == 0:
          self.writeReadyCond.notifyAll()
  @property
  @contextmanager
  def writelock(self):
    with self.lock:
      while self.readerCount > 0:
        self.writeReadyCond.wait()
      yield


if __name__ == "__main__":
  ExecingProcess.checkExec()  # Never returns if this proc is called via ExecingProcess.

  print "You are not expected to call this. This is for ExecingProcess."
  sys.exit(1)
