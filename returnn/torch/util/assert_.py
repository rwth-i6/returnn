"""
Async device assertion utility.
"""

from __future__ import annotations

import threading
from textwrap import dedent
from queue import Queue
import os
import torch


def _get_ext():
    global _ext
    if _ext:
        return _ext

    if not os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        from returnn.util.cuda_env import CudaEnv

        env = CudaEnv.get_instance()
        assert env.cuda_path, "CUDA_PATH, CUDA_HOME not set, and CUDA could not be found automatically."
        os.environ["CUDA_PATH"] = env.cuda_path

    from torch.utils.cpp_extension import load_inline

    _ext = load_inline(
        name="async_assert_ext",
        cpp_sources=_cpp_source,
        cuda_sources=_cuda_source,
        functions=["async_assert_cuda"],
        with_cuda=True,
        verbose=True,
    )
    return _ext


_ext = None

_cpp_source = "void async_assert_cuda(const at::Tensor& cond, const at::Tensor& msg_tensor);"

_cuda_source = dedent("""\
    #include <torch/extension.h>
    #include <ATen/cuda/CUDAContext.h>
    #include <c10/cuda/CUDACachingAllocator.h>
    #include <assert.h>

    __global__ void assert_kernel(const bool* cond, const char* msg) {
        if (blockIdx.x == 0 && threadIdx.x == 0) {
            if (!(*cond)) {
                printf("\\n[GPU ASSERT FAILED]: %s\\n", msg);
                assert(false);
            }
        }
    }

    void async_assert_cuda(const at::Tensor& cond, const at::Tensor& msg_tensor) {
        auto stream = at::cuda::getCurrentCUDAStream();

        // Safety: Protect memory from GC while the kernel is in flight
        c10::cuda::CUDACachingAllocator::recordStream(cond.storage().data_ptr(), stream);
        c10::cuda::CUDACachingAllocator::recordStream(msg_tensor.storage().data_ptr(), stream);

        assert_kernel<<<1, 1, 0, stream>>>(
            cond.data_ptr<bool>(),
            (const char*)msg_tensor.data_ptr<uint8_t>()
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
            cond, message_str, stream = self.queue.get()

            # Use the actual Stream object context
            with torch.cuda.stream(stream):
                # Convert string to pinned tensor (Avoiding read-only NP view)
                msg_bytes = list(message_str.encode("utf-8"))
                msg_cpu = torch.tensor(msg_bytes, dtype=torch.uint8, pin_memory=True)
                msg_gpu = msg_cpu.to("cuda", non_blocking=True)

                # Call JIT-compiled function
                _get_ext().async_assert_cuda(cond, msg_gpu)

    def push(self, cond: torch.Tensor, message: str):
        """push to queue"""
        self.queue.put((cond, message, torch.cuda.current_stream()))


def assert_(cond: torch.Tensor, message: str):
    """
    Does a device-side assertion.
    For CPU, this will directly check the condition and raise an error if false.
    For CUDA devices, this runs asynchronously on a separate thread (to avoid pin_memory in the current thread),
    and non-blocking (does not trigger a CUDA sync).
    """
    if cond.device.type == "cpu":
        if not cond.item():
            raise AssertionError(message)
        return
    elif cond.device.type == "cuda":
        # This triggers the Lazy initialization on first call
        _CudaAsyncWorker().push(cond, message)
    else:
        raise NotImplementedError(f"assert_ not implemented for device type: {cond.device.type}")
