"""
CUDA environment detection and information.
"""

from __future__ import annotations
from typing import Dict, Tuple, List
import os
import re


class CudaEnv:
    """
    Information about the Nvidia CUDA environment, and library.
    Also path to ``nvcc``, the CUDA compiler.
    """

    _instance_per_cls: Dict[type, CudaEnv] = {}
    verbose_find_cuda = False

    def __init__(self):
        from returnn.util.basic import to_bool

        if to_bool(os.environ.get("DISABLE_CUDA", "0")):
            self.cuda_path = None
            if self.verbose_find_cuda:
                print("CUDA disabled via env DISABLE_CUDA.")
        elif os.environ.get("CUDA_VISIBLE_DEVICES", None) in ["", "-1"]:
            self.cuda_path = None
            if self.verbose_find_cuda:
                print(f"CUDA disabled via env CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}.")
        else:
            self.cuda_path = self._find_cuda_path()
            if self.verbose_find_cuda:
                print("CUDA path:", self.cuda_path)

        self._max_compute_capability = None
        self._cuda_version = None

    @classmethod
    def _find_nvcc_in_path(cls):
        """
        :return: yields full path to nvcc
        :rtype: list[str]
        """
        for p in os.environ["PATH"].split(":"):
            pp = "%s/nvcc" % p
            if os.path.exists(pp):
                yield pp

    @classmethod
    def _find_lib_in_ld_path(cls):
        """
        :return: yields full path to libcudart.so
        :rtype: list[str]
        """
        from returnn.util.basic import get_ld_paths

        for p in get_ld_paths():
            pp = "%s/libcudart.so" % p
            if os.path.exists(pp):
                yield pp

    @classmethod
    def _get_lib_dir_name(cls, base_path):
        """
        :return: dir name in base path
        :rtype: str
        """
        from returnn.util.basic import is_64bit_platform, get_ld_paths

        for ld_path in get_ld_paths():
            # We also want to allow "lib/x86_64-linux-gnu" for "/usr".
            # However, this logic should not be triggered for incorrect cases.
            # E.g. base_path="/usr" would be the prefix for most LD paths.
            if ld_path.startswith(base_path + "/lib") and os.path.exists("%s/libcudart.so" % ld_path):
                return ld_path[len(base_path) + 1 :]
        if is_64bit_platform():
            return "lib64"
        return "lib"

    _runtime_libcudart_path_must_be_valid: bool = False

    @classmethod
    def _cuda_path_candidate_via_proc_map_libcudart(cls):
        from returnn.util.basic import find_libcudart_from_runtime

        fn = find_libcudart_from_runtime()
        if cls.verbose_find_cuda:
            print("libcudart.so found from /proc/maps:", fn)
        if not fn:
            return None
        # fn is e.g. '/usr/local/cuda-8.0/targets/x86_64-linux/lib/libcudart.so.8.0.61',
        # or maybe '/usr/local/cuda-8.0/lib64/libcudart.so'
        # or maybe ".../site-packages/nvidia/cuda_runtime/lib/libcudart.so.12"
        # or ".../site-packages/nvidia/cu13/lib/libcudart.so.13"
        p = os.path.dirname(os.path.dirname(fn))
        while not cls._check_valid_cuda_path(p):
            p = os.path.dirname(p)
            if p in ["", "/"]:
                if cls.verbose_find_cuda:
                    print(f"Loaded lib {fn} does not seem to be in valid CUDA path.")
                assert not cls._runtime_libcudart_path_must_be_valid
                return None
        assert cls._check_valid_cuda_path(p)
        return p

    @classmethod
    def _cuda_path_candidates(cls):
        p = cls._cuda_path_candidate_via_proc_map_libcudart()
        if p:
            yield p
        if os.environ.get("CUDA_HOME"):
            yield os.environ.get("CUDA_HOME")
        if os.environ.get("CUDA_PATH"):
            yield os.environ.get("CUDA_PATH")
        for p in cls._find_nvcc_in_path():
            # Expect p == "/usr/local/cuda-8.0/bin/nvcc" or so.
            postfix = "/bin/nvcc"
            if cls.verbose_find_cuda:
                print("found cuda nvcc (wanted postfix: %r): %s" % (postfix, p))
            if not p.endswith(postfix):
                continue
            yield p[: -len(postfix)] or "/"
        for p in cls._find_lib_in_ld_path():
            # Expect p == "/usr/local/cuda-8.0/lib64/libcudart.so" or so.
            d = "/".join(p.split("/")[:-2]) or "/"  # Get "/usr/local/cuda-8.0".
            if cls.verbose_find_cuda:
                print("found cuda lib: %s (path %s)" % (p, d))
            yield d
        # Check common installation location.
        for path in get_cuda_path_candidates_from_common_install_locations():
            yield path

    @classmethod
    def _check_valid_cuda_path(cls, p):
        """
        :param str p: path to CUDA, e.g. "/usr/local/cuda-8.0"
        :return: whether this is a valid CUDA path, i.e. we find all what we need
        :rtype: bool
        """
        if cls.verbose_find_cuda:
            print("check valid CUDA path: %s" % p)
        if not os.path.exists("%s/bin/nvcc" % p):
            return False
        if not os.path.exists("%s/include/cuda.h" % p):
            return False
        if not os.path.exists("%s/%s/libcudart.so" % (p, cls._get_lib_dir_name(p))):
            return False
        return True

    @classmethod
    def _find_cuda_path(cls):
        """
        :return: base CUDA path if we find one, otherwise None
        :rtype: str|None
        """
        for p in cls._cuda_path_candidates():
            if cls._check_valid_cuda_path(p):
                return p
        return None

    def is_available(self):
        """
        :rtype: bool
        """
        return bool(self.cuda_path)

    def get_cuda_version(self) -> Tuple[int, int]:
        """
        Get CUDA version as (major, minor).
        """
        if self._cuda_version:
            return self._cuda_version
        assert self.cuda_path
        # Parse CUDA_VERSION from cuda.h.
        cuda_h_path = f"{self.cuda_path}/include/cuda.h"
        self._cuda_version = _parse_cuda_version_from_cuda_h(cuda_h_path)
        return self._cuda_version

    def get_max_compute_capability(self):
        """
        :return: the highest compute capability supported by nvcc, or float("inf") if not known
        :rtype: float
        """
        if self._max_compute_capability is None:
            cuda_occupancy_path = "%s/include/cuda_occupancy.h" % self.cuda_path
            if os.path.exists(cuda_occupancy_path):
                major, minor = None, 0
                for line in open(cuda_occupancy_path).read().splitlines():
                    m = re.match("^#define\\s+__CUDA_OCC_(MAJOR|MINOR)__\\s+([0-9]+)$", line)
                    if m:
                        s, v = m.groups()
                        v = int(v)
                        if s == "MAJOR":
                            major = v
                        else:
                            minor = v
                if major:
                    self._max_compute_capability = float(major) + float(minor) * 0.1
        if self._max_compute_capability is None:
            self._max_compute_capability = float("inf")
        return self._max_compute_capability

    @staticmethod
    def get_cc_bin() -> str:
        """
        :return: path
        """
        from .native_code_compiler import get_cc_bin

        return get_cc_bin()

    def get_compiler_opts(self):
        """
        :rtype: list[str]
        """
        return [
            "-ccbin",
            self.get_cc_bin(),
            "-I",
            "%s/targets/x86_64-linux/include" % self.cuda_path,
            "-I",
            "%s/include" % self.cuda_path,
            "-L",
            self.get_lib_dir_path(),
            "-x",
            "cu",
            "-v",
        ]

    def get_lib_dir_path(self) -> str:
        """library path"""
        return "%s/%s" % (self.cuda_path, self._get_lib_dir_name(self.cuda_path))

    def get_compiler_bin(self):
        """
        :return: path
        :rtype: str
        """
        assert self.cuda_path
        return "%s/bin/nvcc" % self.cuda_path

    @classmethod
    def get_instance(cls) -> CudaEnv:
        """
        :return: instance for this class
        """
        if cls._instance_per_cls.get(cls) is not None:
            return cls._instance_per_cls[cls]
        cls._instance_per_cls[cls] = cls()
        return cls._instance_per_cls[cls]


def get_cuda_path_candidates_from_common_install_locations() -> List[str]:
    """
    :return: list of possible CUDA installation paths from common locations
    """
    cuda_paths = []

    if os.path.exists("/usr/local"):
        for name in sorted(os.listdir("/usr/local")):
            if name.startswith("cuda-") or name == "cuda":
                p = f"/usr/local/{name}"
                if _check_valid_cuda_path_with_nvcc(p):
                    version = _parse_cuda_version_from_cuda_h(f"{p}/include/cuda.h")
                    cuda_paths.append((version, p))

    # (stable) sort by version, highest version first
    cuda_paths.sort(key=lambda x: x[0], reverse=True)
    return [p for (_, p) in cuda_paths]


def get_best_nvcc_path_for_cuda_version(cuda_version: Tuple[int, int]) -> str:
    """
    :return: path to nvcc
    :rtype: str
    """
    cuda_paths = []

    # noinspection PyProtectedMember
    for p in CudaEnv._cuda_path_candidates():
        if _check_valid_cuda_path_with_nvcc(p):
            version = _parse_cuda_version_from_cuda_h(f"{p}/include/cuda.h")
            if version == cuda_version:
                # if we found a matching one, directly return it
                return f"{p}/bin/nvcc"
            cuda_paths.append((version, p))

    if not cuda_paths:
        raise RuntimeError(f"No valid CUDA installation found for version {cuda_version}.")

    only_higher_versions = [(version, p) for (version, p) in cuda_paths if version >= cuda_version]
    if only_higher_versions:
        only_higher_versions.sort(key=lambda x: x[0])
        # return the lowest higher version
        if only_higher_versions[0][0] != cuda_version[0]:  # major version differs
            print(
                f"Warning: No exact match for CUDA version {cuda_version}, "
                f"using version {only_higher_versions[0]} instead."
            )
        return f"{only_higher_versions[0][1]}/bin/nvcc"

    cuda_paths.sort(key=lambda x: x[0])
    # return the highest lower version
    print(f"Warning: No exact match for CUDA version {cuda_version}, using lower version {cuda_paths[-1][0]} instead.")
    return f"{cuda_paths[-1][1]}/bin/nvcc"


def _check_valid_cuda_path_with_nvcc(p: str) -> bool:
    """
    :param str p: path to CUDA, e.g. "/usr/local/cuda-8.0"
    :return: whether this is a valid CUDA path, i.e. we find all what we need
    :rtype: bool
    """
    if not os.path.exists("%s/bin/nvcc" % p):
        return False
    if not os.path.exists("%s/include/cuda.h" % p):
        return False
    return True


def _parse_cuda_version_from_cuda_h(cuda_h_path: str) -> Tuple[int, int]:
    assert os.path.exists(cuda_h_path)
    for line in open(cuda_h_path).read().splitlines():
        # Like: #define CUDA_VERSION 12080
        m = re.match(r"^#define\s+CUDA_VERSION\s+([0-9]+)$", line)
        if m:
            version_num = int(m.group(1))
            major = version_num // 1000
            minor = (version_num % 1000) // 10
            return major, minor
    raise RuntimeError(f"Could not determine CUDA version from {cuda_h_path}.")
