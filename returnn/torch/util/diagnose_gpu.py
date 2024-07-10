"""
Diagnostic functions for GPU information, failings, memory usage, etc.
"""

from __future__ import annotations
from typing import Optional, Union, List, TextIO
import os
import sys
import gc
import subprocess
import torch
from returnn.util.better_exchook import better_exchook
from returnn.util.basic import human_bytes_size


def print_available_devices(*, file: Optional[TextIO] = None):
    """
    Print available devices, GPU (CUDA or other), etc.

    :param file: where to print to. stdout by default
    """
    if file is None:
        file = sys.stdout
    cuda_visible_devs = None
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES is set to %r." % os.environ["CUDA_VISIBLE_DEVICES"], file=file)
        cuda_visible_devs = dict(enumerate([int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d]))
    else:
        if torch.cuda.is_available():
            print("CUDA_VISIBLE_DEVICES is not set.", file=file)

    if torch.cuda.is_available():
        print("Available CUDA devices:")
        count = torch.cuda.device_count()
        if cuda_visible_devs is not None and len(cuda_visible_devs) != count:
            print(
                f"(Mismatch between CUDA device count {count}"
                f" and CUDA_VISIBLE_DEVICES {cuda_visible_devs} count {len(cuda_visible_devs)}?)",
                file=file,
            )
        for i in range(count):
            print(f"  {i + 1}/{count}: cuda:{i}", file=file)
            props = torch.cuda.get_device_properties(i)
            print(f"       name: {props.name}", file=file)
            print(f"       total_memory: {human_bytes_size(props.total_memory)}", file=file)
            print(f"       capability: {props.major}.{props.minor}", file=file)
            if cuda_visible_devs is not None:
                if len(cuda_visible_devs) == count:
                    dev_idx_s = cuda_visible_devs[i]
                else:
                    dev_idx_s = "?"
            else:
                dev_idx_s = i
            print(f"       device_index: {dev_idx_s}", file=file)
        if not count:
            print("  (None)")
    else:
        print("(CUDA not available)")


def print_using_cuda_device_report(dev: Union[str, torch.device], *, file: Optional[TextIO] = None):
    """
    Theano and TensorFlow print sth like: Using gpu device 2: GeForce GTX 980 (...)
    Print in a similar format so that some scripts which grep our stdout work just as before.
    """
    if file is None:
        file = sys.stdout
    if isinstance(dev, str):
        dev = torch.device(dev)
    assert dev.type == "cuda", f"expected CUDA device, got {dev}"
    if dev.index is not None:
        idx = dev.index
    else:
        idx = torch.cuda.current_device()
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        cuda_visible_devs = dict(enumerate([int(d) for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d]))
        idx_s = cuda_visible_devs.get(idx, torch.cuda.device_count() + idx)
    else:
        idx_s = idx
    print(f"Using gpu device {idx_s}:", torch.cuda.get_device_name(idx), file=file)
    free, total = torch.cuda.mem_get_info(idx)
    print(f"Total GPU {idx_s} memory {human_bytes_size(total)}, free {human_bytes_size(free)}", file=file)


def diagnose_no_gpu() -> List[str]:
    """
    Diagnose why we have no GPU.
    Print to stdout, but also prepare summary strings.

    :return: summary strings
    """
    # Currently we assume Nvidia CUDA here, but once we support other backends (e.g. ROCm),
    # first check which backend is most reasonable here.

    res = []
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))
    print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", None))

    try:
        torch.cuda.init()
    except Exception as exc:
        print("torch.cuda.init() failed:", exc)
        better_exchook(*sys.exc_info(), debugshell=False)
        res.append(f"torch.cuda.init() failed: {type(exc).__name__} {exc}")

    try:
        subprocess.check_call(["nvidia-smi"])
    except Exception as exc:
        print("nvidia-smi failed:", exc)
        better_exchook(*sys.exc_info(), debugshell=False)
        res.append(f"nvidia-smi failed")

    return res


def print_relevant_env_vars(*, file: Optional[TextIO] = None):
    """
    Print relevant environment variables which might affect the GPU usage,
    or PyTorch usage in general.
    For example:
    PYTORCH_CUDA_ALLOC_CONF, CUDA_LAUNCH_BLOCKING, etc.
    https://pytorch.org/docs/stable/torch_environment_variables.html

    :param file: where to print to. stdout by default
    """
    if file is None:
        file = sys.stdout
    _Prefixes = {"PYTORCH", "TORCH", "CUDA", "CUBLAS", "CUBLASLT", "CUDNN", "NVIDIA", "NCCL", "OMP", "MKL", "XLA"}
    for k, v in os.environ.items():
        if "_" not in k:
            continue
        prefix, _ = k.split("_", 1)
        if prefix in _Prefixes:
            print(f"{k}={v}", file=file)


def garbage_collect():
    """
    Perform garbage collection, including any special logic for GPU.

    Also see:
    https://github.com/pytorch/pytorch/issues/18853
    https://github.com/pytorch/pytorch/issues/27600
    https://pytorch.org/docs/stable/notes/faq.html#my-out-of-memory-exception-handler-can-t-allocate-memory
    https://github.com/Lightning-AI/pytorch-lightning/blob/7a4b0fc4331633cdf00b88776689e8a84ef96cb4/src/lightning/pytorch/utilities/memory.py#L83
    """
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stats = [
            f"alloc {human_bytes_size(torch.cuda.memory_allocated())}",
            f"reserved {human_bytes_size(torch.cuda.memory_reserved())}",
        ]
        print(f"CUDA memory usage after triggered GC:", " ".join(stats))
