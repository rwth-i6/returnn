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
from typing import Callable, Optional, Set, TextIO
import os

import torch

__all__ = [
    "parse_cpulist",
    "get_gpu_pci_id",
    "find_pci_id_by_gpu_uuid",
    "read_pci_device_local_cpus",
    "select_gpu_local_cpus",
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


def read_cpu_socket(cpu: int, *, sysfs_root: str = "/sys/devices/system/cpu") -> int:
    """
    :param cpu:
    :param sysfs_root:
    :return: the socket (physical package) of the CPU
    """
    with open(os.path.join(sysfs_root, f"cpu{cpu}", "topology", "physical_package_id"), "rt") as f:
        return int(f.read())


def select_gpu_local_cpus(
    local_cpus: Set[int], allowed_cpus: Set[int], *, num_local_ranks: int, cpu_socket: Callable[[int], int]
) -> Optional[Set[int]]:
    """
    The CPUs a rank should pin itself to:
    the allowed CPUs local to its GPU (its NUMA node),
    widened to the GPU's socket when the NUMA node is a part of a socket (e.g. AMD NPS4)
    and alone too small for the rank's share of the allowed CPUs,
    or None when even the socket is too small (an unaligned cpuset),
    so pinning never leaves a rank with less than its share.

    :param local_cpus: local to the GPU
    :param allowed_cpus: the process may use, e.g. the SLURM cpuset
    :param num_local_ranks: ranks sharing those allowed CPUs
    :param cpu_socket: socket of a CPU
    :return: the CPUs to pin to, or None for no pinning
    """
    fair_share = len(allowed_cpus) // max(num_local_ranks, 1)
    cpus = local_cpus & allowed_cpus
    if len(cpus) >= fair_share:
        return cpus
    sockets = {cpu_socket(cpu) for cpu in local_cpus}
    cpus = {cpu for cpu in allowed_cpus if cpu_socket(cpu) in sockets}
    if len(cpus) >= fair_share:
        return cpus
    return None


def set_gpu_local_cpu_affinity(
    device_index: int, *, num_local_ranks: int = 1, log_file: Optional[TextIO] = None
) -> Optional[Set[int]]:
    """
    Restrict all threads of this process to the CPUs local to the GPU,
    within the CPUs the process may use (e.g. the SLURM cpuset), see :func:`select_gpu_local_cpus`.
    Processes started afterwards (dataset workers) inherit it.

    :param device_index: CUDA device index
    :param num_local_ranks: ranks sharing the allowed CPUs
    :param log_file:
    :return: the CPUs, or None when nothing was pinned
    """
    pci_id = get_gpu_pci_id(device_index)
    local_cpus = read_pci_device_local_cpus(pci_id)
    allowed = os.sched_getaffinity(0)
    cpus = select_gpu_local_cpus(local_cpus, allowed, num_local_ranks=num_local_ranks, cpu_socket=read_cpu_socket)
    if cpus is None:
        if log_file:
            print(
                f"CUDA device {device_index} ({pci_id}): CPU affinity not set,"
                f" the local CPUs {sorted(local_cpus)} and their socket cover less than the share"
                f" of one of {num_local_ranks} ranks of the {len(allowed)} allowed CPUs",
                file=log_file,
            )
        return None
    for tid in os.listdir("/proc/self/task"):
        os.sched_setaffinity(int(tid), cpus)
    if log_file:
        print(
            f"CUDA device {device_index} ({pci_id}): CPU affinity set to {len(cpus)} CPUs"
            f" (of {len(allowed)} allowed, {len(local_cpus)} local): {sorted(cpus)}",
            file=log_file,
        )
    return cpus
