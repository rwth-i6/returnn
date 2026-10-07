"""
CPU affinity of the process to the CPUs local to its GPU.

On a multi-socket node (or a multi-superchip node like GH200),
every GPU hangs off one CPU socket / NUMA node.
Nothing pins the training process by default,
so its host work (dataset workers, batch handoff, copies to the device)
runs on whatever cores the scheduler picks, often across the sockets.

Linux only: the kernel reports the local CPUs of a PCI device in sysfs.
"""

from __future__ import annotations
from typing import Optional, Set, TextIO
import os

import torch

__all__ = [
    "parse_cpulist",
    "get_gpu_pci_id",
    "find_pci_id_by_gpu_uuid",
    "read_pci_device_local_cpus",
    "set_gpu_local_cpu_affinity",
]


def parse_cpulist(cpulist: str) -> Set[int]:
    """
    :param cpulist: Linux cpulist format, e.g. "0-11,24-35"
    :return: the CPUs
    """
    cpus = set()
    for part in cpulist.strip().split(","):
        if not part:
            continue
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return cpus


def get_gpu_pci_id(device_index: int) -> str:
    """
    :param device_index: CUDA device index
    :return: the PCI id of the device as it appears in sysfs, e.g. "0000:1b:00.0".
        The CUDA device order is not the PCI order, so the id comes from the device properties
        (torch >= 2.8), or from the NVIDIA driver by the device UUID.
    """
    props = torch.cuda.get_device_properties(device_index)
    if hasattr(props, "pci_bus_id"):
        return f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    return find_pci_id_by_gpu_uuid(str(props.uuid))


def find_pci_id_by_gpu_uuid(uuid: str, *, proc_root: str = "/proc/driver/nvidia/gpus") -> str:
    """
    :param uuid: e.g. "8ff8d0c7-8d30-8e55-0980-ac69fc03a6b8", with or without the "GPU-" prefix
    :param proc_root: one dir per GPU, named by its PCI id, with an ``information`` file
    :return: the PCI id, e.g. "0000:1b:00.0"
    """
    uuid = _strip_gpu_prefix(uuid)
    for pci_id in sorted(os.listdir(proc_root)):
        with open(os.path.join(proc_root, pci_id, "information"), "rt") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key.strip() == "GPU UUID" and _strip_gpu_prefix(value) == uuid:
                    return pci_id
    raise RuntimeError(f"GPU with UUID {uuid} not found under {proc_root}")


def _strip_gpu_prefix(uuid: str) -> str:
    uuid = uuid.strip().lower()
    return uuid[len("gpu-") :] if uuid.startswith("gpu-") else uuid


def read_pci_device_local_cpus(pci_id: str, *, sysfs_root: str = "/sys/bus/pci/devices") -> Set[int]:
    """
    :param pci_id: e.g. "0000:1b:00.0"
    :param sysfs_root:
    :return: the CPUs local to the device
    """
    with open(os.path.join(sysfs_root, pci_id, "local_cpulist"), "rt") as f:
        return parse_cpulist(f.read())


def set_gpu_local_cpu_affinity(device_index: int, *, log_file: Optional[TextIO] = None) -> Set[int]:
    """
    Restrict all threads of this process to the CPUs local to the GPU,
    within the CPUs the process may use (e.g. the SLURM cpuset).
    Processes started afterwards (dataset workers) inherit it.

    :param device_index: CUDA device index
    :param log_file:
    :return: the CPUs
    """
    pci_id = get_gpu_pci_id(device_index)
    local_cpus = read_pci_device_local_cpus(pci_id)
    allowed = os.sched_getaffinity(0)
    cpus = local_cpus & allowed
    if not cpus:
        raise RuntimeError(
            f"CUDA device {device_index} ({pci_id}): no allowed CPU is local to it,"
            f" local {sorted(local_cpus)}, allowed {sorted(allowed)}"
        )
    for tid in os.listdir("/proc/self/task"):
        os.sched_setaffinity(int(tid), cpus)
    if log_file:
        print(
            f"CUDA device {device_index} ({pci_id}): CPU affinity set to the {len(cpus)} local CPUs"
            f" (of {len(allowed)} allowed): {sorted(cpus)}",
            file=log_file,
        )
    return cpus
