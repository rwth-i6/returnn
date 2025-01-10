"""
Some generic debugging utilities.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Union, Any, Collection, Sequence, Callable, Tuple, List, Dict, TextIO
from types import FunctionType, CodeType, FrameType
import os
import sys
import signal

try:
    import thread
except ImportError:
    import _thread as thread
import threading


if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim
    import torch


signum_to_signame = {
    k: v for v, k in reversed(sorted(signal.__dict__.items())) if v.startswith("SIG") and not v.startswith("SIG_")
}


global_exclude_thread_ids = set()


def auto_exclude_all_new_threads(func):
    """
    :param T func:
    :return: func wrapped
    :rtype: T
    """

    def wrapped(*args, **kwargs):
        """
        :param args:
        :param kwargs:
        :return:
        """
        # noinspection PyProtectedMember,PyUnresolvedReferences
        old_threads = set(sys._current_frames().keys())
        res = func(*args, **kwargs)
        # noinspection PyProtectedMember,PyUnresolvedReferences
        new_threads = set(sys._current_frames().keys())
        new_threads -= old_threads
        global_exclude_thread_ids.update(new_threads)
        return res

    return wrapped


def dump_all_thread_tracebacks(
    *, exclude_thread_ids: Optional[Collection[int]] = None, exclude_self: bool = False, file: Optional[TextIO] = None
):
    """
    :param exclude_thread_ids:
    :param exclude_self:
    :param file:
    """
    if exclude_thread_ids is None:
        exclude_thread_ids = set()
    if file is None:
        file = sys.stdout

    from returnn.util.better_exchook import print_tb
    import threading

    if exclude_self:
        exclude_thread_ids = set(list(exclude_thread_ids) + [threading.current_thread().ident])

    if hasattr(sys, "_current_frames"):
        print("", file=file)
        threads = {t.ident: t for t in threading.enumerate()}
        # noinspection PyProtectedMember
        for tid, stack in sorted(sys._current_frames().items()):
            # This is a bug in earlier Python versions.
            # https://bugs.python.org/issue17094
            # Note that this leaves out all threads not created via the threading module.
            if tid not in threads:
                continue
            tags = []
            thread_ = threads.get(tid)
            if thread_:
                assert isinstance(thread_, threading.Thread)
                if thread_ is threading.current_thread():
                    tags += ["current"]
                # noinspection PyUnresolvedReferences,PyProtectedMember
                if isinstance(thread_, threading._MainThread):
                    tags += ["main"]
                tags += [str(thread_)]
            else:
                tags += ["unknown with id %i" % tid]
            print("Thread %s:" % ", ".join(tags), file=file)
            if tid in global_exclude_thread_ids:
                print("(Auto-ignored traceback.)", file=file)
            elif tid in exclude_thread_ids:
                print("(Excluded thread.)", file=file)
            else:
                print_tb(stack, file=file)
            print("", file=file)
        print("That were all threads.", file=file)
    else:
        print("Does not have sys._current_frames, cannot get thread tracebacks.", file=file)


def setup_warn_with_traceback():
    """
    Installs some hook for ``warnings.showwarning``.
    """
    import warnings
    from returnn.util.better_exchook import print_tb

    def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
        """
        :param message:
        :param category:
        :param filename:
        :param lineno:
        :param file:
        :param line:
        """
        log = file if hasattr(file, "write") else sys.stderr
        log.write(warnings.formatwarning(message, category, filename, lineno, line))
        # noinspection PyProtectedMember,PyUnresolvedReferences
        print_tb(sys._getframe(), file=log)

    warnings.showwarning = warn_with_traceback


def init_better_exchook():
    """
    Installs our own ``sys.excepthook``, which uses :mod:`better_exchook`,
    but adds some special handling for the main thread.
    """
    from returnn.util.better_exchook import better_exchook
    from returnn.log import log

    def excepthook(exc_type, exc_obj, exc_tb):
        """
        :param exc_type:
        :param exc_obj:
        :param exc_tb:
        """
        file = log.v1 or sys.stdout

        # noinspection PyBroadException
        try:
            # noinspection PyUnresolvedReferences,PyProtectedMember
            is_main_thread = isinstance(threading.current_thread(), threading._MainThread)
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
        print(
            "Unhandled exception %s in thread %s, proc %i." % (exc_type, threading.current_thread(), os.getpid()),
            file=file,
        )
        if exc_type is KeyboardInterrupt:
            return

        # noinspection PyUnresolvedReferences,PyProtectedMember
        if isinstance(threading.current_thread(), threading._MainThread):
            main_thread_id = thread.get_ident()
            if not isinstance(exc_type, Exception):
                # We are the main thread and we got an exit-exception. This is likely fatal.
                # This usually means an exit. (We ignore non-daemon threads and procs here.)
                # Print the stack of all other threads.
                dump_all_thread_tracebacks(exclude_thread_ids={main_thread_id}, file=file)

        better_exchook(exc_type, exc_obj, exc_tb, file=file)

    sys.excepthook = excepthook

    def threading_excepthook(args, /):
        """
        Thread-specific excepthook to ensure the main thread is killed on unhandled exceptions in sub threads.
        """
        log_out = log.v1 or sys.stdout
        print(
            f"Unhandled exception in thread {threading.current_thread()}, going to interrupt main thread:", file=log_out
        )
        better_exchook(args.exc_type, args.exc_value, args.exc_traceback, autodebugshell=False, file=log_out)
        thread.interrupt_main()

    threading.excepthook = threading_excepthook

    from returnn.util.basic import to_bool

    if os.environ.get("DEBUG_WARN_WITH_TRACEBACK") and to_bool(os.environ.get("DEBUG_WARN_WITH_TRACEBACK")):
        setup_warn_with_traceback()


def format_signum(signum):
    """
    :param int signum:
    :return: string "signum (signame)"
    :rtype: str
    """
    return "%s (%s)" % (signum, signum_to_signame.get(signum, "unknown"))


# noinspection PyUnusedLocal
def signal_handler(signum, frame):
    """
    Prints a message on stdout and dump all thread stacks.

    :param int signum: e.g. signal.SIGUSR1
    :param frame: ignored, will dump all threads
    """
    print("Signal handler: got signal %s" % format_signum(signum))
    dump_all_thread_tracebacks()


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


_native_signal_handler_lib_filename = None


def _get_native_signal_handler_lib_filename() -> str:
    """
    :return: path to our native_signal_handler lib. see :func:`install_native_signal_handler`
    """
    global _native_signal_handler_lib_filename
    if _native_signal_handler_lib_filename:
        return _native_signal_handler_lib_filename

    from returnn.util.basic import NativeCodeCompiler
    import textwrap

    native = NativeCodeCompiler(
        base_name="native_signal_handler",
        code_version=1,
        code=textwrap.dedent(
            # Derived from https://github.com/albertz/playground/blob/master/signal_handler.c.
            # language=C++
            """\
            #include <stdio.h>
            #include <execinfo.h>
            #include <signal.h>
            #include <stdlib.h>
            #include <unistd.h>


            // https://github.com/ruby/ruby/blob/bbfd735b887/vm_core.h#L118
            #if defined(NSIG_MAX)           /* POSIX issue 8 */
            # undef NSIG
            # define NSIG NSIG_MAX
            #elif defined(_SIG_MAXSIG)      /* FreeBSD */
            # undef NSIG
            # define NSIG _SIG_MAXSIG
            #elif defined(_SIGMAX)          /* QNX */
            # define NSIG (_SIGMAX + 1)
            #elif defined(NSIG)             /* 99% of everything else */
            # /* take it */
            #else                           /* Last resort */
            # define NSIG (sizeof(sigset_t) * CHAR_BIT + 1)
            #endif


            sig_t old_signal_handler[NSIG];


            void signal_handler(int sig) {
              void *array[16 * 1024];
              size_t size;

              // get void*'s for all entries on the stack
              size = backtrace(array, sizeof(array)/sizeof(array[0]));

              // print out all the frames to stderr
              fprintf(stderr, "Signal handler: signal %d:\\n", sig);
              backtrace_symbols_fd(array, size, STDERR_FILENO);

              // call previous handler
              signal(sig, old_signal_handler[sig]);
              raise(sig);
            }

            void install_signal_handler() {
              old_signal_handler[SIGSEGV] = signal(SIGSEGV, signal_handler);
              old_signal_handler[SIGBUS] = signal(SIGBUS, signal_handler);
              old_signal_handler[SIGILL] = signal(SIGILL, signal_handler);
              old_signal_handler[SIGABRT] = signal(SIGABRT, signal_handler);
              old_signal_handler[SIGFPE] = signal(SIGFPE, signal_handler);
            }
            """
        ),
        is_cpp=False,
    )
    _native_signal_handler_lib_filename = native.get_lib_filename()
    return _native_signal_handler_lib_filename


def install_native_signal_handler(*, reraise_exceptions: bool = False):
    """
    Installs some own custom C signal handler.
    """
    try:
        import ctypes

        # C code: https://github.com/albertz/playground/blob/master/signal_handler.c
        lib = ctypes.CDLL(_get_native_signal_handler_lib_filename())
        lib.install_signal_handler.return_type = None
        lib.install_signal_handler()
        print("Installed native_signal_handler.so.")

    except Exception as exc:
        print("installNativeSignalHandler exception: %s" % exc)
        if reraise_exceptions:
            raise


def install_lib_sig_segfault():
    """
    Installs libSegFault (common on Unix/Linux).
    """
    try:
        os.environ.setdefault("SEGFAULT_SIGNALS", "all")
        import ctypes
        import ctypes.util

        # libSegFault on Unix/Linux, not on MacOSX
        libfn = ctypes.util.find_library("SegFault")
        assert libfn, "libSegFault not found"
        # Nothing more needed than loading it, it will automatically register itself.
        ctypes.CDLL(libfn)
        print("Installed libSegFault.so.")

    except Exception as exc:
        print("installLibSigSegfault exception: %s" % exc)


def init_faulthandler(sigusr1_chain=False):
    """
    Maybe installs signal handlers, SIGUSR1 and SIGUSR2 and others.
    If no signals handlers are installed yet for SIGUSR1/2, we try to install our own Python handler.
    This also tries to install the handler from the fauldhandler module,
    esp for SIGSEGV and others.

    :param bool sigusr1_chain: whether the default SIGUSR1 handler should also be called.
    """
    # Enable libSigSegfault first, so that we can have both,
    # because faulthandler will also call the original sig handler.
    install_native_signal_handler()
    if sys.platform != "win32":
        # In case that sigusr1_chain, we expect that there is already some handler
        # for SIGUSR1, and then this will not overwrite this handler.
        if install_signal_handler_if_default(signal.SIGUSR1):
            # There is already some handler or we installed our own handler now,
            # so in any case, it's safe that we chain then handler.
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
            if sys.platform != "win32":
                faulthandler.register(signal.SIGUSR1, all_threads=True, chain=sigusr1_chain)


@auto_exclude_all_new_threads
def init_ipython_kernel():
    """
    Runs IPython in some background kernel, where you can connect to.
    """
    # You can remotely connect to this kernel. See the output on stdout.
    try:
        # noinspection PyPackageRequirements,PyUnresolvedReferences
        import IPython.kernel.zmq.ipkernel

        # noinspection PyPackageRequirements,PyUnresolvedReferences
        from IPython.kernel.zmq.ipkernel import Kernel

        # noinspection PyPackageRequirements,PyUnresolvedReferences
        from IPython.kernel.zmq.heartbeat import Heartbeat

        # noinspection PyPackageRequirements,PyUnresolvedReferences
        from IPython.kernel.zmq.session import Session

        # noinspection PyPackageRequirements,PyUnresolvedReferences
        from IPython.kernel import write_connection_file

        # noinspection PyPackageRequirements,PyUnresolvedReferences
        import zmq

        # noinspection PyPackageRequirements,PyUnresolvedReferences
        from zmq.eventloop import ioloop

        # noinspection PyPackageRequirements,PyUnresolvedReferences
        from zmq.eventloop.zmqstream import ZMQStream

        # noinspection PyPackageRequirements,PyUnresolvedReferences
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
    # noinspection PyUnresolvedReferences,PyProtectedMember
    assert isinstance(threading.current_thread(), threading._MainThread)
    try:
        ip = socket.gethostbyname(socket.gethostname())
        connection_file = "ipython-kernel-%s-%s.json" % (ip, os.getpid())

        def cleanup_connection_file():
            """
            Cleanup.
            """
            try:
                os.remove(connection_file)
            except (IOError, OSError):
                pass

        atexit.register(cleanup_connection_file)

        logger = logging.Logger("IPython")
        logger.addHandler(logging.NullHandler())
        session = Session(username="kernel")

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

        kernel = Kernel(
            session=session, shell_streams=[shell_stream, control_stream], iopub_socket=iopub_socket, log=logger
        )

        write_connection_file(
            connection_file,
            shell_port=shell_port,
            iopub_port=iopub_port,
            control_port=control_port,
            hb_port=hb_port,
            ip=ip,
        )

        # print "To connect another client to this IPython kernel, use:", \
        #      "ipython console --existing %s" % connection_file
    except Exception as e:
        print("Exception while initializing IPython ZMQ kernel. %s" % e)
        return

    def ipython_thread():
        """
        IPython thread.
        """
        kernel.start()
        try:
            ioloop.IOLoop.instance().start()
        except KeyboardInterrupt:
            pass

    thread_ = threading.Thread(target=ipython_thread, name="IPython kernel")
    thread_.daemon = True
    thread_.start()


def init_cuda_not_in_main_proc_check():
    """
    Installs some hook to Theano which checks that CUDA is only used in the main proc.
    """
    # noinspection PyUnresolvedReferences,PyPackageRequirements
    import theano.sandbox.cuda as cuda

    if cuda.use.device_number is not None:
        print("CUDA already initialized in proc %i" % os.getpid())
        return
    use_original = cuda.use

    def use_wrapped(device, **kwargs):
        """
        :param device:
        :param kwargs:
        """
        print("CUDA.use %s in proc %i" % (device, os.getpid()))
        use_original(device=device, **kwargs)

    cuda.use = use_wrapped
    cuda.use.device_number = None


def debug_shell(
    user_ns: Optional[Dict[str, Any]] = None,
    user_global_ns: Optional[Dict[str, Any]] = None,
    exit_afterwards: bool = True,
):
    """
    Provides some interactive Python shell.
    Uses IPython if possible.
    Wraps to ``better_exchook.debug_shell``.

    :param user_ns:
    :param user_global_ns:
    :param exit_afterwards: will do sys.exit(1) at the end
    """
    print("Debug shell:")
    from returnn.util.basic import ObjAsDict
    from . import debug_helpers

    user_global_ns_new = dict(ObjAsDict(debug_helpers).items())
    if user_global_ns:
        user_global_ns_new.update(user_global_ns)  # may overwrite vars from DebugHelpers
    user_global_ns_new["debug"] = debug_helpers  # make this available always
    print("Available debug functions/utils (via DebugHelpers):")
    for k, v in sorted(vars(debug_helpers).items()):
        if k[:1] == "_":
            continue
        print("  %s (%s)" % (k, type(v)))
    print("Also DebugHelpers available as 'debug'.")
    if not user_ns:
        user_ns = {}
    if user_ns:
        print("Locals:")
        for k, v in sorted(user_ns.items()):
            print("  %s (%s)" % (k, type(v)))
    from returnn.util.better_exchook import debug_shell

    debug_shell(user_ns, user_global_ns_new)
    if exit_afterwards:
        print("Debug shell exit. Exit now.")
        sys.exit(1)


class PyTracer:
    """
    Trace Python function execution to get intermediate outputs from the local variables.

    E.g. for PyTorch code, when comparing results, it can be useful to see the intermediate tensors.

    Example::

        with PyTracer([my_func], torch.Tensor) as trace_my_impl:
            ...

        with PyTracer([reference_func], torch.Tensor) as trace_ref_impl:
            ...

    Or another example::

        from returnn.tensor import Tensor

        with PyTracer([my_func], Tensor) as trace_my_impl:
            ...

        with PyTracer([reference_func], torch.Tensor) as trace_ref_impl:
            ...

        check_py_traces_rf_to_pt_equal(trace_my_impl.captured_locals, trace_ref_impl.captured_locals, [...])

    See also :func:`check_py_traces_rf_to_pt_equal` to compare the traces.

    This class uses the Python :func:`sys.settrace` mechanism to trace the locals.
    It accesses ``frame.f_locals`` to get the local variables.
    Note that this behavior is slightly buggy in versions of CPython <3.13,
    see for example:
    https://github.com/python/cpython/issues/113939
    https://github.com/python/cpython/issues/74929
    And thus the behavior might be different depending on the Python version.
    In Python >=3.13, you likely get a few more locals than before.
    """

    def __init__(
        self, funcs_to_trace_list: Sequence[Union[Callable, FunctionType]], capture_type: Union[type, Tuple[type, ...]]
    ):
        """
        :param funcs_to_trace_list: list of functions to trace the locals. only those functions will be traced.
        :param capture_type: only capture variables of this type, e.g. torch.Tensor.
        """

        def _get_func_code(func: FunctionType) -> CodeType:
            while getattr(func, "__wrapped__", None) is not None:
                func = func.__wrapped__
            return func.__code__

        self.funcs_to_trace_list = funcs_to_trace_list
        self._code_obj_to_func = {_get_func_code(func): func for func in self.funcs_to_trace_list}
        self.capture_type = capture_type

        self._prev_trace_func = None
        self.captured_locals = {}  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor

    def __enter__(self) -> PyTracer:
        self._prev_trace_func = sys.gettrace()
        # Note: We might get such a warning here when using PyDev/PyCharm:
        #   PYDEV DEBUGGER WARNING:
        #   sys.settrace() should not be used when the debugger is being used.
        #   This may cause the debugger to stop working correctly.
        # This is ok.
        # In fact, we even call this prev_trace_func in our own __call__ as well,
        # so debugging (breakpoints etc.) should still work (I tested it.).
        sys.settrace(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.settrace(self._prev_trace_func)
        self._prev_trace_func = None

    def __call__(self, frame: FrameType, event, arg) -> Optional[PyTracer]:
        """
        Trace func to get intermediate outputs.
        """
        prev_trace_func_res = None
        if self._prev_trace_func:
            prev_trace_func_res = self._prev_trace_func(frame, event, arg)
        func = self._code_obj_to_func.get(frame.f_code)
        if func:
            if event == "call":
                self.captured_locals.setdefault(func, []).append({})
            else:
                # Note that accessing frame.f_locals is not always totally accurate.
                # See the corresponding comment in our class docstring.
                for k, v in frame.f_locals.items():
                    if not isinstance(v, self.capture_type):
                        continue
                    prev = self.captured_locals[func][-1].get(k, None)
                    if prev is None or prev[-1] is not v:
                        var_ls = self.captured_locals[func][-1].setdefault(k, [])
                        print(
                            f"{func.__qualname__}[{len(self.captured_locals[func]) - 1}]"
                            f" ({type(v).__qualname__}) {k}[{len(var_ls)}] = {v}"
                        )
                        var_ls.append(v)
            return self
        return prev_trace_func_res


def check_py_traces_rf_to_pt_equal(
    trace_rf: Dict[Callable, List[Dict[str, List[Tensor]]]],
    trace_pt: Dict[Callable, List[Dict[str, List[torch.Tensor]]]],
    checks: List[
        Tuple[
            Tuple[Callable, int, str, int],
            Tuple[Callable, int, str, int],
            Union[Tuple[Union[Dim, str], ...], Callable[[torch.Tensor], Tensor]],
        ],
    ],
):
    """
    Compares traces from some RETURNN-frontend (RF) based implementation
    with some pure PyTorch (PT) based implementation.

    :param trace_rf: RETURNN-frontend trace, from :class:`PyTracer`
    :param trace_pt: pure PyTorch trace, from :class:`PyTracer`
    :param checks: list of checks to perform. each check is a tuple of:
        - RF trace entry, e.g. (func, i, name, j)
        - PT trace entry, e.g. (func, i, name, j)
        - PT dims, e.g. (batch_dim, other_dim, ...).
            Instead of Dim, you can also use a string, which will be resolved from the RF trace
            (then you also need ``Dim`` in ``capture_type`` of the :class:`PyTracer`).
            If callable, it gets the PyTorch tensor and should return the RETURNN tensor.
            Sometimes you might want to perform some reshaping, slicing, or similar,
            and then use rf.convert_to_tensor.
    """
    import random
    import torch
    from returnn.tensor import Tensor, Dim
    import returnn.frontend as rf

    # noinspection PyProtectedMember
    from torch.testing._comparison import make_tensor_mismatch_msg

    dummy_forward_compat_kwargs = {f"_random_forward_compat_arg_{random.randint(0, 1_000_000)}": "please"}

    def _get_entry(trace, func, i, name, j):
        return trace[func][i][name][j]

    def _resolve_dim(dim: Union[Dim, str]) -> Dim:
        if isinstance(dim, Dim):
            return dim
        elif isinstance(dim, str):
            dim = _get_entry(trace_rf, *check_rf[:2], dim, -1)
            assert isinstance(dim, Dim)
            return dim
        else:
            raise TypeError(f"invalid dim type: {dim!r}")

    def _format_check(check: Tuple[Union[FunctionType, Callable], int, str, int]) -> str:
        func, i, var_name, j = check
        return f"{func.__qualname__}[{i}] {var_name}[{j}]"

    non_matching = []
    for check_rf, check_pt, pt_dims in checks:
        print(f"checking {_format_check(check_rf)} vs {_format_check(check_pt)} ({pt_dims})...")
        tensor_rf: Tensor = _get_entry(trace_rf, *check_rf)
        tensor_pt: torch.Tensor = _get_entry(trace_pt, *check_pt)
        if callable(pt_dims):
            tensor_pt_ = pt_dims(tensor_pt, name=check_pt[2], resolve_dim=_resolve_dim, **dummy_forward_compat_kwargs)
        else:
            pt_dims = [_resolve_dim(dim) for dim in pt_dims]
            tensor_pt_ = rf.convert_to_tensor(tensor_pt, dims=pt_dims, name=check_pt[2])
        tensor_pt_ = tensor_pt_.copy_transpose(tensor_rf.dims)
        tensor_rf = tensor_rf.copy_masked(0.0)
        tensor_pt_ = tensor_pt_.copy_masked(0.0)
        rtol, atol = 1e-4, 1e-5  # defaults from torch.testing.assert_allclose
        matches = torch.isclose(tensor_rf.raw_tensor, tensor_pt_.raw_tensor, rtol=rtol, atol=atol)
        if matches.all():
            print("  matched!")
        else:
            msgs = make_tensor_mismatch_msg(
                tensor_rf.raw_tensor, tensor_pt_.raw_tensor, matches, rtol=rtol, atol=atol, identifier="Tensors"
            )
            msgs_prefix = [
                f"{check_rf} vs {check_pt}:",
                f"  RF: {tensor_rf} {tensor_rf.raw_tensor}",
                f"  PT: {tensor_pt_} {tensor_pt_.raw_tensor}",
            ]
            msgs = ["  " + line for line in msgs.splitlines() if line and "Tensors are not " not in line]
            indices = (~matches).nonzero().detach().cpu()
            for idx in indices[:5]:
                idx = tuple(idx.numpy())
                msgs.append(f"  non-matching at {idx}: RF={tensor_rf.raw_tensor[idx]} PT={tensor_pt_.raw_tensor[idx]}")
            if len(indices) > 5:
                msgs.append("  non-matching ...")
            non_matching.append("\n".join(msgs_prefix + msgs))
            print(f"  mismatch!")
            for msg in msgs:
                print(msg)

    if non_matching:
        raise AssertionError("\n\n".join(non_matching))
