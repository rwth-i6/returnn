"""
Async device assertion utility.
"""

from __future__ import annotations

import threading
from textwrap import dedent
from queue import Queue
import torch


def assert_(cond: torch.Tensor, message: str, *, stop: bool = True):
    """
    Does a device-side assertion.
    For CPU, this will directly check the condition and raise an error if false.
    For CUDA devices, this runs asynchronously on a separate thread (to avoid pin_memory in the current thread),
    and non-blocking (does not trigger a CUDA sync).

    :param cond: A boolean tensor indicating the condition to assert.
    :param message: The message to display if the assertion fails.
    :param stop: Whether to stop execution on assertion failure
    """
    if cond.device.type == "cpu":
        if not cond.item():
            if stop:
                raise AssertionError(message)
            else:
                print(f"[ASSERT FAILED WARNING]: {message}")
        return
    elif cond.device.type == "cuda":
        # This triggers the Lazy initialization on first call
        _CudaAsyncWorker().push(cond, message, stop=stop)
    else:
        raise NotImplementedError(f"assert_ not implemented for device type: {cond.device.type}")


def _get_ext():
    global _ext
    if _ext:
        return _ext

    from .native_op_code_compiler import OpCodeCompiler

    compiler = OpCodeCompiler(
        "async_assert_ext", use_cuda_if_available=True, code=_cpp_source + _cuda_source, is_python_module=True
    )
    _ext = compiler.load_module()
    return _ext


_ext = None

_cpp_source = dedent("""\
    #include <torch/extension.h>

    void async_assert_cuda(const at::Tensor& cond, const at::Tensor& msg_tensor, bool stop);

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("async_assert_cuda", torch::wrap_pybind_function(async_assert_cuda), "Asynchronous CUDA assert");
    }
    """)

_cuda_source = dedent("""\
    #include <torch/types.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDACachingAllocator.h>
    #include <assert.h>

    __global__ void assert_kernel(const bool* cond, const char* msg, bool stop) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (!(*cond)) {
                printf("[GPU %s]: %s\\n", stop ? "ASSERT FAILED" : "ASSERT FAILED WARNING", msg);
                if(stop) __trap();
            }
        }
    }

    void async_assert_cuda(const at::Tensor& cond, const at::Tensor& msg_tensor, bool stop) {
        auto stream = at::cuda::getCurrentCUDAStream();

        // Safety: Protect memory from GC while the kernel is in flight
        c10::cuda::CUDACachingAllocator::recordStream(cond.storage().data_ptr(), stream);
        c10::cuda::CUDACachingAllocator::recordStream(msg_tensor.storage().data_ptr(), stream);

        assert_kernel<<<1, 1, 0, stream>>>(
            cond.data_ptr<bool>(),
            (const char*)msg_tensor.data_ptr<uint8_t>(),
            stop
        );
    }
    """)


class _CudaAsyncWorker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(_CudaAsyncWorker, cls).__new__(cls)
                cls._instance._init_worker()
            return cls._instance

    def _init_worker(self):
        self.queue = Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            cond, message_str, stop, stream = self.queue.get()

            # Use the actual Stream object context
            with torch.cuda.stream(stream):
                # Convert string to pinned tensor (Avoiding read-only NP view)
                msg_bytes = list(message_str.encode("utf-8")) + [0]
                msg_cpu = torch.tensor(msg_bytes, dtype=torch.uint8, pin_memory=True)
                msg_gpu = msg_cpu.to("cuda", non_blocking=True)

                # Call JIT-compiled function
                _get_ext().async_assert_cuda(cond, msg_gpu, stop)

    def push(self, cond: torch.Tensor, message: str, stop: bool = True):
        """push to queue"""
        self.queue.put((cond, message, stop, torch.cuda.current_stream()))
