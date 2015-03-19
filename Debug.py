
import os
import sys


def initBetterExchook():
  import thread
  import threading
  import better_exchook
  main_thread_id = thread.get_ident()

  def excepthook(exc_type, exc_obj, exc_tb):
    print "Unhandled exception %s in %s." % (exc_type, threading.currentThread())
    better_exchook.better_exchook(exc_type, exc_obj, exc_tb)

    if main_thread_id == thread.get_ident():
      if not isinstance(exc_type, Exception):
        # We are the main thread and we got an exit-exception. This is likely fatal.
        # This usually means an exit. (We ignore non-daemon threads and procs here.)
        # Print the stack of all other threads.
        if hasattr(sys, "_current_frames"):
          for tid, stack in sys._current_frames().items():
            if tid != main_thread_id:
              print "Thread %i:" % tid
              better_exchook.print_traceback(stack)

  sys.excepthook = excepthook


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
  except ImportError, e:
    print "IPython import error, cannot start IPython kernel. %s" % e
    return
  import atexit
  import socket
  import logging
  import threading

  # Do in mainthread to avoid history sqlite DB errors at exit.
  # https://github.com/ipython/ipython/issues/680
  assert isinstance(threading.currentThread(), threading._MainThread)
  try:
    connection_file = "kernel-%s.json" % os.getpid()
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
    ip = socket.gethostbyname(socket.gethostname())
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

    print "To connect another client to this IPython kernel, use:", \
          "ipython console --existing %s" % connection_file
  except Exception, e:
    print "Exception while initializing IPython ZMQ kernel. %s" % e
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
