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
from typing import Callable, List, Optional, Sequence, Set, Tuple
import os
import sys

import torch

from returnn.config import Config
from returnn.log import log

__all__ = [
    "set_gpu_local_cpu_affinity_from_config",
    "set_gpu_local_cpu_affinity",
    "select_gpu_local_cpus",
    "parse_cpulist",
    "get_gpu_pci_id",
    "find_pci_id_by_gpu_uuid",
    "read_pci_device_local_cpus",
    "read_cpu_socket",
]


def set_gpu_local_cpu_affinity_from_config(config: Config) -> Optional[Set[int]]:
    """
    Config option ``gpu_local_cpu_affinity`` (default True): pin this process to the CPUs local to its GPU,
    see :func:`set_gpu_local_cpu_affinity`.
    To be called once at startup, after the distributed context exists (the local rank is the device)
    and before the datasets are created, so their worker processes inherit the affinity.
    In distributed training, the ranks on a host share their allowed CPUs
    (one cpuset, e.g. ``torchrun`` in a SLURM job).
    When the launcher already bound them to different CPUs (``srun`` per-task binding, ``mpirun``),
    that binding is kept, nothing is pinned.

    :param config:
    :return: the CPUs, or None when nothing was pinned (option off, no CUDA device, unknown topology, already bound)
    """
    if not config.bool("gpu_local_cpu_affinity", True) or sys.platform != "linux":
        return None
    from returnn.torch.engine import get_device_from_config

    device = torch.device(get_device_from_config(config).result)
    if device.type != "cuda":
        return None
    num_local_ranks = 1
    if config.typed_value("torch_distributed") is not None:
        from returnn.torch.distributed import get_ctx

        get_ctx(config=config)
        allowed_per_local_rank = _gather_allowed_cpus_of_local_ranks()
        if any(allowed != allowed_per_local_rank[0] for allowed in allowed_per_local_rank):
            print(
                "CPU affinity not set, the launcher already bound the ranks on this host to different CPUs:"
                f" {[sorted(allowed) for allowed in allowed_per_local_rank]}",
                file=log.v3,
            )
            return None
        num_local_ranks = len(allowed_per_local_rank)
    # a bare "cuda" means the current (default) device, as in the engine
    index = device.index if device.index is not None else torch.cuda.current_device()
    return set_gpu_local_cpu_affinity(index, num_local_ranks=num_local_ranks)


def _gather_allowed_cpus_of_local_ranks() -> List[Set[int]]:
    """
    :return: the allowed CPUs (:func:`os.sched_getaffinity`) of every rank on this host, in rank order,
        via a gloo group, so no CUDA device is touched
    """
    import socket
    import torch.distributed as dist

    group = dist.new_group(backend="gloo")
    try:
        hostname = socket.gethostname()
        gathered: List[Optional[Tuple[str, List[int]]]] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, (hostname, sorted(os.sched_getaffinity(0))), group=group)
    finally:
        dist.destroy_process_group(group)
    return [set(allowed) for host, allowed in gathered if host == hostname]


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
    # not in the torch type stubs (also missing in torch < 2.8), so by name
    domain, bus, device = (getattr(props, name, None) for name in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
    if bus is not None:
        return f"{domain:04x}:{bus:02x}:{device:02x}.0"
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
    :param sysfs_root: the PCI devices dir of sysfs
    :return: the CPUs local to the device
    """
    with open(os.path.join(sysfs_root, pci_id, "local_cpulist"), "rt") as f:
        return parse_cpulist(f.read())


def read_cpu_socket(cpu: int, *, sysfs_root: str = "/sys/devices/system/cpu") -> int:
    """
    :param cpu: logical CPU index
    :param sysfs_root: the CPU dir of sysfs
    :return: the socket (physical package) of the CPU
    """
    with open(os.path.join(sysfs_root, f"cpu{cpu}", "topology", "physical_package_id"), "rt") as f:
        return int(f.read())


def select_gpu_local_cpus(
    local_cpus_per_rank: Sequence[Set[int]],
    local_rank: int,
    allowed_cpus: Set[int],
    *,
    cpu_socket: Callable[[int], int],
) -> Optional[Set[int]]:
    """
    The CPUs a rank should pin itself to.

    The candidate set is the allowed CPUs local to its GPU (its NUMA node),
    widened to the GPU's socket when the NUMA node is a part of a socket (e.g. AMD NPS4)
    and alone too small for the rank's share of the allowed CPUs (allowed / ranks).
    Ranks whose GPUs share that set get disjoint contiguous slices of it when each slice still covers the share
    (no two ranks compete for the same cores), else they share the whole set.
    None when even the socket is too small (an unaligned cpuset),
    so pinning never leaves a rank with less than its share.

    :param local_cpus_per_rank: per local rank, the CPUs local to its GPU
    :param local_rank: the rank to select for
    :param allowed_cpus: the process may use, e.g. the SLURM cpuset
    :param cpu_socket: socket of a CPU
    :return: the CPUs to pin to, or None for no pinning
    """
    assert local_cpus_per_rank, "no ranks"
    fair_share = max(1, len(allowed_cpus) // len(local_cpus_per_rank))

    def _socket_cpus(local_cpus: Set[int]) -> Set[int]:
        sockets = {cpu_socket(cpu) for cpu in local_cpus}
        return {cpu for cpu in allowed_cpus if cpu_socket(cpu) in sockets}

    for widen in (lambda local_cpus: local_cpus & allowed_cpus, _socket_cpus):
        candidates: List[Set[int]] = [widen(local_cpus) for local_cpus in local_cpus_per_rank]
        cpus = candidates[local_rank]
        if len(cpus) < fair_share:
            continue
        peers = [rank for rank, other in enumerate(candidates) if other == cpus]
        if len(cpus) >= fair_share * len(peers):
            cpus_sorted = sorted(cpus)
            slice_len = len(cpus_sorted) // len(peers)
            idx = peers.index(local_rank)
            return set(cpus_sorted[idx * slice_len : (idx + 1) * slice_len if idx + 1 < len(peers) else None])
        return cpus
    return None


def set_gpu_local_cpu_affinity(local_rank: int, *, num_local_ranks: int = 1) -> Optional[Set[int]]:
    """
    Restrict all threads of this process to the CPUs local to the GPU,
    within the CPUs the process may use (e.g. the SLURM cpuset), see :func:`select_gpu_local_cpus`.
    Processes started afterwards (dataset workers) inherit it.
    Local rank ``i`` is assumed to use CUDA device ``i``, as the engine does.
    When the topology cannot be read (no sysfs or NVIDIA driver files, fewer visible devices than ranks),
    nothing is pinned, logged.

    :param local_rank: and CUDA device index
    :param num_local_ranks: ranks on this node, sharing the allowed CPUs
    :return: the CPUs, or None when nothing was pinned
    """
    if num_local_ranks > torch.cuda.device_count():
        print(
            f"CPU affinity not set: {num_local_ranks} local ranks but {torch.cuda.device_count()} visible CUDA devices",
            file=log.v3,
        )
        return None
    try:
        pci_ids = [get_gpu_pci_id(i) for i in range(num_local_ranks)]
        local_cpus_per_rank = [read_pci_device_local_cpus(pci_id) for pci_id in pci_ids]
        allowed = os.sched_getaffinity(0)
        cpus = select_gpu_local_cpus(local_cpus_per_rank, local_rank, allowed, cpu_socket=read_cpu_socket)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"CPU affinity not set, GPU topology unknown: {type(exc).__name__}: {exc}", file=log.v3)
        return None
    pci_id, local_cpus = pci_ids[local_rank], local_cpus_per_rank[local_rank]
    if cpus is None:
        print(
            f"CUDA device {local_rank} ({pci_id}): CPU affinity not set,"
            f" the local CPUs {sorted(local_cpus)} and their socket cover less than the share"
            f" of one of {num_local_ranks} ranks of the {len(allowed)} allowed CPUs",
            file=log.v3,
        )
        return None
    for tid in os.listdir("/proc/self/task"):
        os.sched_setaffinity(int(tid), cpus)
    print(
        f"CUDA device {local_rank} ({pci_id}): CPU affinity set to {len(cpus)} CPUs"
        f" (of {len(allowed)} allowed, {len(local_cpus)} local): {sorted(cpus)}",
        file=log.v3,
    )
    return cpus
