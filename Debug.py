
import os
import sys
import signal


signum_to_signame = {
  k: v for v, k in reversed(sorted(signal.__dict__.items()))
  if v.startswith('SIG') and not v.startswith('SIG_')}


global_exclude_thread_ids = set()

def auto_exclude_all_new_threads(func):
  def wrapped(*args, **kwargs):
    old_threads = set(sys._current_frames().keys())
    res = func(*args, **kwargs)
    new_threads = set(sys._current_frames().keys())
    new_threads -= old_threads
    global_exclude_thread_ids.update(new_threads)
    return res
  return wrapped


def dumpAllThreadTracebacks(exclude_thread_ids=set()):
  import better_exchook
  import threading

  if hasattr(sys, "_current_frames"):
    print("")
    threads = {t.ident: t for t in threading.enumerate()}
    for tid, stack in sys._current_frames().items():
      if tid in exclude_thread_ids: continue
      # This is a bug in earlier Python versions.
      # http://bugs.python.org/issue17094
      # Note that this leaves out all threads not created via the threading module.
      if tid not in threads: continue
      print("Thread %s:" % threads.get(tid, "unnamed with id %i" % tid))
      if tid in global_exclude_thread_ids:
        print("(Auto-ignored traceback.)")
      else:
        better_exchook.print_tb(stack)
      print("")
  else:
    print("Does not have sys._current_frames, cannot get thread tracebacks.")


def setupWarnWithTraceback():
  import warnings
  import better_exchook

  def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file, 'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    better_exchook.print_tb(sys._getframe(), file=log)

  warnings.showwarning = warn_with_traceback


def initBetterExchook():
  import thread
  import threading
  import better_exchook
  import pdb

  def excepthook(exc_type, exc_obj, exc_tb):
    try:
      is_main_thread = isinstance(threading.currentThread(), threading._MainThread)
    except Exception:  # Can happen at a very late state while quitting.
      if exc_type is KeyboardInterrupt:
        return
    else:
      if is_main_thread:
        if exc_type is KeyboardInterrupt and getattr(sys, "exited", False):
          # Got SIGINT twice. Can happen.
          return
        # An unhandled exception in the main thread. This means that we are going to quit now.
        sys.exited = True
    print("Unhandled exception %s in thread %s, proc %i." % (exc_type, threading.currentThread(), os.getpid()))
    if exc_type is KeyboardInterrupt:
      return

    if isinstance(threading.currentThread(), threading._MainThread):
      main_thread_id = thread.get_ident()
      if not isinstance(exc_type, Exception):
        # We are the main thread and we got an exit-exception. This is likely fatal.
        # This usually means an exit. (We ignore non-daemon threads and procs here.)
        # Print the stack of all other threads.
        dumpAllThreadTracebacks({main_thread_id})

    better_exchook.better_exchook(exc_type, exc_obj, exc_tb)

  sys.excepthook = excepthook

  from Util import to_bool
  if os.environ.get("DEBUG_WARN_WITH_TRACEBACK") and to_bool(os.environ.get("DEBUG_WARN_WITH_TRACEBACK")):
    setupWarnWithTraceback()


def format_signum(signum):
  """
  :param int signum:
  :return: string "signum (signame)"
  :rtype: str
  """
  return "%s (%s)" % (signum, signum_to_signame.get(signum, "unknown"))


def signal_handler(signum, frame):
  """
  Prints a message on stdout and dump all thread stacks.

  :param int signum: e.g. signal.SIGUSR1
  :param frame: ignored, will dump all threads
  """
  print("Signal handler: got signal %s" % format_signum(signum))
  dumpAllThreadTracebacks()


def install_signal_handler_if_default(signum, exceptions_are_fatal=False):
  """
  :param int signum: e.g. signal.SIGUSR1
  :param bool exceptions_are_fatal: if True, will reraise any exceptions. if False, will just print a message
  :return: True iff no exception, False otherwise. not necessarily that we registered our own handler
  :rtype: bool
  """
  try:
    if signal.getsignal(signum) == signal.SIG_DFL:
      signal.signal(signum, signal_handler)
    return True
  except Exception as exc:
    if exceptions_are_fatal:
      raise
    print("Cannot install signal handler for signal %s, exception %s" % (format_signum(signum), exc))
  return False


def installNativeSignalHandler():
  try:
    import ctypes
    # TODO: Move C code here, automatically compile it on-the-fly or so.
    # C code: https://github.com/albertz/playground/blob/master/signal_handler.c
    # Maybe not needed because on Linux there is libSegFault.so anyway (installLibSigSegfault()).
    lib = ctypes.CDLL("/u/zeyer/code/playground/signal_handler.so")
    lib.install_signal_handler.return_type = None
    lib.install_signal_handler()
    print("Installed signal_handler.so.")

  except Exception as exc:
    print("installNativeSignalHandler exception: %s" % exc)


def installLibSigSegfault():
  try:
    import ctypes
    import ctypes.util
    # libSegFault on Unix/Linux, not on MacOSX
    libfn = ctypes.util.find_library("SegFault")
    assert libfn
    # Nothing more needed than loading it, it will automatically register itself.
    ctypes.CDLL(libfn)
    print("Installed libSegFault.so.")

  except Exception as exc:
    print("installLibSigSegfault exception: %s" % exc)


def initFaulthandler(sigusr1_chain=False):
  """
  Maybe installs signal handlers, SIGUSR1 and SIGUSR2 and others.
  If no signals handlers are installed yet for SIGUSR1/2, we try to install our own Python handler.
  This also tries to install the handler from the fauldhandler module,
  esp for SIGSEGV and others.

  :param bool sigusr1_chain: whether the default SIGUSR1 handler should also be called.
  """
  # In case that sigusr1_chain, we expect that there is already some handler
  # for SIGUSR1, and then this will not overwrite this handler.
  if install_signal_handler_if_default(signal.SIGUSR1):
    # There is already some handler or we installed our own handler now,
    # so in any case, it's save that we chain then handler.
    sigusr1_chain = True
  # Why not also SIGUSR2... SGE can also send this signal.
  install_signal_handler_if_default(signal.SIGUSR2)
  try:
    import faulthandler
  except ImportError as e:
    print("faulthandler import error. %s" % e)
  else:
    # Only enable if not yet enabled -- otherwise, leave it in its current state.
    if not faulthandler.is_enabled():
      faulthandler.enable()
      if os.name != 'nt':
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=sigusr1_chain)
  from Util import to_bool
  if os.environ.get("DEBUG_SIGNAL_HANDLER") and to_bool(os.environ.get("DEBUG_SIGNAL_HANDLER")):
    installLibSigSegfault()
    installNativeSignalHandler()


@auto_exclude_all_new_threads
def initIPythonKernel():
  # You can remotely connect to this kernel. See the output on stdout.
  try:
    import IPython.kernel.zmq.ipkernel
    from IPython.kernel.zmq.ipkernel import Kernel
    from IPython.kernel.zmq.heartbeat import Heartbeat
    from IPython.kernel.zmq.session import Session
    from IPython.kernel import write_connection_file
    import zmq
    from zmq.eventloop import ioloop
    from zmq.eventloop.zmqstream import ZMQStream
    IPython.kernel.zmq.ipkernel.signal = lambda sig, f: None  # Overwrite.
  except ImportError as e:
    print("IPython import error, cannot start IPython kernel. %s" % e)
    return
  import atexit
  import socket
  import logging
  import threading

  # Do in mainthread to avoid history sqlite DB errors at exit.
  # https://github.com/ipython/ipython/issues/680
  assert isinstance(threading.currentThread(), threading._MainThread)
  try:
    ip = socket.gethostbyname(socket.gethostname())
    connection_file = "ipython-kernel-%s-%s.json" % (ip, os.getpid())
    def cleanup_connection_file():
      try:
        os.remove(connection_file)
      except (IOError, OSError):
        pass
    atexit.register(cleanup_connection_file)

    logger = logging.Logger("IPython")
    logger.addHandler(logging.NullHandler())
    session = Session(username=u'kernel')

    context = zmq.Context.instance()
    transport = "tcp"
    addr = "%s://%s" % (transport, ip)
    shell_socket = context.socket(zmq.ROUTER)
    shell_port = shell_socket.bind_to_random_port(addr)
    iopub_socket = context.socket(zmq.PUB)
    iopub_port = iopub_socket.bind_to_random_port(addr)
    control_socket = context.socket(zmq.ROUTER)
    control_port = control_socket.bind_to_random_port(addr)

    hb_ctx = zmq.Context()
    heartbeat = Heartbeat(hb_ctx, (transport, ip, 0))
    hb_port = heartbeat.port
    heartbeat.start()

    shell_stream = ZMQStream(shell_socket)
    control_stream = ZMQStream(control_socket)

    kernel = Kernel(session=session,
                    shell_streams=[shell_stream, control_stream],
                    iopub_socket=iopub_socket,
                    log=logger)

    write_connection_file(connection_file,
                          shell_port=shell_port, iopub_port=iopub_port, control_port=control_port, hb_port=hb_port,
                          ip=ip)

    #print "To connect another client to this IPython kernel, use:", \
    #      "ipython console --existing %s" % connection_file
  except Exception as e:
    print("Exception while initializing IPython ZMQ kernel. %s" % e)
    return

  def ipython_thread():
    kernel.start()
    try:
      ioloop.IOLoop.instance().start()
    except KeyboardInterrupt:
      pass

  thread = threading.Thread(target=ipython_thread, name="IPython kernel")
  thread.daemon = True
  thread.start()


def initCudaNotInMainProcCheck():
  import TaskSystem
  import theano.sandbox.cuda as cuda
  if cuda.use.device_number is not None:
    print("CUDA already initialized in proc %i" % os.getpid())
    return
  use_original = cuda.use
  def use_wrapped(device, **kwargs):
    print("CUDA.use %s in proc %i" % (device, os.getpid()))
    #assert not TaskSystem.isMainProcess, "multiprocessing is set to True in your config but the main proc tries to use CUDA"
    use_original(device=device, **kwargs)
  cuda.use = use_wrapped
  cuda.use.device_number = None


def debug_shell(user_ns=None, user_global_ns=None, exit_afterwards=True):
  print("Debug shell:")
  from Util import ObjAsDict
  import DebugHelpers
  user_global_ns_new = dict(ObjAsDict(DebugHelpers).items())
  if user_global_ns:
    user_global_ns_new.update(user_global_ns)  # may overwrite vars from DebugHelpers
  user_global_ns_new["debug"] = DebugHelpers  # make this available always
  print("Available debug functions/utils (via DebugHelpers):")
  for k, v in sorted(vars(DebugHelpers).items()):
    if k[:1] == "_": continue
    print("  %s (%s)" % (k, type(v)))
  print("Also DebugHelpers available as 'debug'.")
  if not user_ns:
    user_ns = {}
  if user_ns:
    print("Locals:")
    for k, v in sorted(user_ns.items()):
      print("  %s (%s)" % (k, type(v)))
  import better_exchook
  better_exchook.debug_shell(user_ns, user_global_ns_new)
  if exit_afterwards:
    print("Debug shell exit. Exit now.")
    sys.exit(1)

