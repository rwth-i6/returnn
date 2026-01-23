"""
Helper to compile Torch ops on-the-fly, similar to Theano / :class:`returnn.tf.util.basic.OpCodeCompiler`,
similar to :mod:`torch.utils.cpp_extension`.

See :class:`OpCodeCompiler`.
"""

from __future__ import annotations
from typing import Union, Optional, Sequence, Dict, List
import os
import sysconfig

import torch
from torch.utils import cpp_extension

from returnn.util.basic import NativeCodeCompiler
from returnn.util.cuda_env import CudaEnv as _CudaEnvBase, get_best_nvcc_path_for_cuda_version


class OpCodeCompiler(NativeCodeCompiler):
    """
    Helper class to compile Torch ops on-the-fly, similar to Theano,
    and similar to :class:`returnn.tf.util.basic.OpCodeCompiler`.

    Note that PyTorch already has its own code for this,
    see :mod:`torch.utils.cpp_extension`, :func:`torch.utils.cpp_extension.load_inline`, etc.
    However, there are some shortcomings there that we try to do better:

    * The way we find CUDA/nvcc is more robust.
    * The way we find the C/C++ compiler is more robust.
    * The automatic selection of options for nvcc is more robust.
      E.g. the compute version is not higher than what the selected CUDA supports.

    https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
    """

    CacheDirName = "returnn_torch_cache/ops"

    def __init__(
        self,
        base_name: str,
        *,
        code: str,
        use_cuda_if_available: bool = True,
        cuda_auto_min_compute_capability: bool = True,
        include_paths: Sequence[str] = (),
        ld_flags: Sequence[str] = (),
        c_macro_defines: Optional[Dict[str, Union[str, int, None]]] = None,
        is_python_module: bool = False,
        **kwargs,
    ):
        self._cuda_env = None
        if use_cuda_if_available and torch.cuda.is_available():
            self._cuda_env = CudaEnv.get_instance()
            # Currently we assume that if we provide CUDA code (thus set use_cuda_if_available=True),
            # that if there is a GPU available (as TF reports it),
            # we also expect that we find CUDA.
            # Otherwise you would end up with ops compiled for CPU only although they support CUDA
            # and the user expects them to run on GPU.
            assert self._with_cuda(), "OpCodeCompiler: use_cuda_if_available=True but no CUDA found"

        self._nvcc_opts = []
        if self._with_cuda() and cuda_auto_min_compute_capability:
            # Get CUDA compute capability of the current GPU device.
            min_compute_capability = _get_available_gpu_cuda_min_compute_capability()
            if min_compute_capability:
                min_compute_capability = min(min_compute_capability, self._cuda_env.get_max_compute_capability())
                self._nvcc_opts += ["-arch", "compute_%i" % int(min_compute_capability * 10)]

        if self._with_cuda():
            self._nvcc_opts += cpp_extension.COMMON_NVCC_FLAGS

        # Example call from torch.utils.cpp_extension:
        # /usr/local/cuda-11.0/bin/nvcc
        # --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d
        # -DTORCH_EXTENSION_NAME=async_assert_ext
        # -DTORCH_API_INCLUDE_EXTENSION_H
        # -isystem /home/az/py-venv/py3.12-torch2.9/lib/python3.12/site-packages/torch/include
        # -isystem /home/az/py-venv/py3.12-torch2.9/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
        # -isystem /usr/local/cuda-11.0/include -isystem /usr/include/python3.12
        # -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__
        # -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__
        # --expt-relaxed-constexpr
        # -gencode=arch=compute_86,code=compute_86 -gencode=arch=compute_86,code=sm_86
        # --compiler-options '-fPIC' -std=c++17
        # -c /home/az/.cache/torch_extensions/py312_cu128/async_assert_ext/cuda.cu -o cuda.cuda.o

        torch_path = os.path.dirname(torch.__file__)
        torch_include = torch_path + "/include"
        assert os.path.isdir(torch_include)

        python_include = sysconfig.get_path("include", scheme="posix_prefix")

        include_paths = list(include_paths) + [torch_include, torch_include + "/torch/csrc/api/include", python_include]

        c_macro_defines = {} if c_macro_defines is None else c_macro_defines.copy()
        c_macro_defines.setdefault("TORCH_EXTENSION_NAME", base_name)
        c_macro_defines.setdefault("TORCH_API_INCLUDE_EXTENSION_H", "")
        # We have some assert in our kernels that we want to disable.
        c_macro_defines.setdefault("NDEBUG", 1)

        ld_flags = list(ld_flags)
        ld_flags.append("--no-as-needed")
        ld_flags.append(f"-L{cpp_extension.TORCH_LIB_PATH}")
        ld_flags.append("-lc10")
        if self._with_cuda():
            ld_flags.append("-lc10_cuda")
        ld_flags.append("-ltorch_cpu")
        if self._with_cuda():
            ld_flags.append("-ltorch_cuda")
        ld_flags.append("-ltorch")
        ld_flags.append("-ltorch_python")

        if self._with_cuda():
            ld_flags.append(self._cuda_env.get_ld_flag_for_linking_cudart())
            # maybe add CUDNN?

        # noinspection PyUnresolvedReferences,PyProtectedMember
        use_cxx11_abi = torch._C._GLIBCXX_USE_CXX11_ABI

        super().__init__(
            base_name=base_name,
            code=code,
            include_paths=include_paths,
            c_macro_defines=c_macro_defines,
            ld_flags=ld_flags,
            use_cxx11_abi=use_cxx11_abi,
            **kwargs,
        )
        self.is_python_module = is_python_module
        self._mod = None

    def __repr__(self):
        return "<%s %r CUDA %s in %r>" % (self.__class__.__name__, self.base_name, self._with_cuda(), self._mod_path)

    _relevant_info_keys = NativeCodeCompiler._relevant_info_keys + (
        "torch_version",
        "with_cuda",
        "cuda_path",
        "nvcc_opts",
    )

    def _make_info_dict(self):
        from returnn.util.basic import describe_torch_version

        d = super()._make_info_dict()
        d.update(
            {
                "torch_version": describe_torch_version(),
                "with_cuda": self._with_cuda(),
                "cuda_path": self._cuda_env.cuda_path if self._with_cuda() else None,
                "nvcc_opts": (
                    (tuple(self._cuda_env.get_compiler_opts()) + tuple(self._nvcc_opts)) if self._with_cuda() else None
                ),
            }
        )
        return d

    @classmethod
    def cuda_available(cls):
        """
        :return: whether CUDA is available. if True, and you initiate with use_cuda_if_available=True,
          then _with_cuda() should also be True.
        :rtype: bool
        """
        if not torch.cuda.is_available():
            return False
        cuda_env = CudaEnv.get_instance()
        return cuda_env.is_available()

    def _with_cuda(self):
        return bool(self._cuda_env and self._cuda_env.is_available())

    cpp_version = 17

    def _get_compiler_bin(self):
        if self._with_cuda():
            return self._cuda_env.get_compiler_bin()
        return super()._get_compiler_bin()

    def _transform_compiler_opts(self, opts: List[str]) -> List[str]:
        if self._with_cuda():
            nvcc_opts = self._cuda_env.get_compiler_opts()
            for opt in opts:
                nvcc_opts += ["-Xcompiler", opt]
            nvcc_opts += self._nvcc_opts
            return nvcc_opts
        return super()._transform_compiler_opts(opts)

    def _transform_ld_flags(self, opts: Sequence[str]) -> Sequence[str]:
        if self._with_cuda():
            res = []
            for opt in opts:
                if opt.startswith("-L") or opt.startswith("-l"):
                    res.append(opt)
                else:
                    res += ["-Xlinker", opt]
            return res
        return super()._transform_ld_flags(opts)

    def load_module(self):
        """
        :return: module
        """
        if self._mod:
            return self._mod
        self._maybe_compile()

        if self.is_python_module:
            # Load as a Python module.
            # E.g. PYBIND11_MODULE or so was used in the code.
            import importlib.util

            # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
            spec = importlib.util.spec_from_file_location(self.base_name, self._so_filename)
            assert spec is not None
            module = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(module)

        else:
            # Load as a Torch extension.
            # TORCH_LIBRARY / TORCH_LIBRARY_IMPL was used in the code.
            torch.ops.load_library(self._so_filename)
            module = getattr(torch.ops, self.base_name)

        self._mod = module
        return module


class CudaEnv(_CudaEnvBase):
    """specialized CudaEnv for PyTorch"""

    # If cudart is loaded (e.g. via Torch), we really want to use that one.
    _runtime_libcudart_path_must_be_valid = True

    def __init__(self):
        super().__init__()

        from returnn.util.basic import find_libcudart_from_runtime

        self._runtime_libcudart = find_libcudart_from_runtime()
        self._compiler_bin = None
        if self.cuda_path:
            if os.path.exists(f"{self.cuda_path}/bin/nvcc"):
                self._compiler_bin = f"{self.cuda_path}/bin/nvcc"
            else:
                self._compiler_bin = get_best_nvcc_path_for_cuda_version(self.get_cuda_version())

    @classmethod
    def _check_valid_cuda_path(cls, p: str) -> bool:
        """
        :param p: path to CUDA, e.g. "/usr/local/cuda-8.0"
        :return: whether this is a valid CUDA path, i.e. we find all what we need
        """
        if cls.verbose_find_cuda:
            print("check valid CUDA path: %s" % p)
        # Don't check nvcc here yet.
        # The pip package might not have it, but otherwise provides lib + headers
        # that we want to use, as this is likely the same that PyTorch uses.
        if not os.path.exists("%s/include/cuda.h" % p):
            return False
        if p.endswith("/site-packages/nvidia/cuda_runtime"):
            # special case for the nvidia CUDA pip package
            if not any(name.startswith("libcudart.") for name in os.listdir(p + "/lib")):
                return False
        else:
            if not os.path.exists("%s/%s/libcudart.so" % (p, cls._get_lib_dir_name(p))):
                return False
        return True

    def get_lib_dir_path(self) -> str:
        """
        :return: path
        """
        if self._runtime_libcudart:
            return os.path.dirname(self._runtime_libcudart)
        return super().get_lib_dir_path()

    def get_ld_flag_for_linking_cudart(self) -> str:
        """ld flag"""
        if self._runtime_libcudart:
            return f"-l:{os.path.basename(self._runtime_libcudart)}"
        return "-lcudart"

    def get_compiler_bin(self) -> str:
        """
        :return: path
        """
        return self._compiler_bin


def _get_available_gpu_cuda_min_compute_capability() -> Optional[float]:
    """
    Uses :func:`get_available_gpu_devices`.

    :return: e.g. 3.0, or 5.0, etc, or None
    """
    count = torch.cuda.device_count()
    cap = None
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        dev_cap = float(f"{props.major}.{props.minor}")
        if cap is None:
            cap = dev_cap
        else:
            cap = min(cap, dev_cap)
    return cap
