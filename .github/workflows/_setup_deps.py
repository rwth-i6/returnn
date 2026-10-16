#!/usr/bin/env python3

"""
Install necessary dependencies for CI.
Some are potentially already installed (e.g. via cache).
"""

from __future__ import annotations
from typing import Optional, List, Dict, Set
import os
import sys
import time
import argparse
import subprocess


__my_dir__ = os.path.dirname(os.path.abspath(__file__))


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--python", help="Verify Python version (optional)")
    arg_parser.add_argument("--torch", help="PyTorch version to install (optional)")
    arg_parser.add_argument("--tf", help="TensorFlow version to install (optional)")
    arg_parser.add_argument("--jax", help="JAX version to install (optional)")
    arg_parser.add_argument("--espnet", help="Whether to install ESPnet (optional)")
    arg_parser.add_argument(
        "--espnet-no-deps", help="install ESPnet without its dependencies, for the code inspection (optional)"
    )
    arg_parser.add_argument("--hf-datasets", help="Whether to install HF datasets (optional)")
    arg_parser.add_argument("--nemo-speech", help="Whether to install NeMo Speech, for NemoSpeechDataset (optional)")
    args = arg_parser.parse_args()

    print("Python:", sys.version, sys.executable)

    if args.python:
        print("Verifying Python version:", args.python)
        version = tuple(int(n) for n in args.python.split("."))
        assert sys.version_info[: len(version)] == version, f"Expected Python {args.python}"

    with StdoutTextFold("free disk space"):
        _run(f"{__my_dir__}/_free_disk_space.sh")

    py = sys.executable
    pip = [py, "-m", "pip"]
    installed_reqs: List[str] = []
    installed_reqs_no_deps: List[str] = []

    def pip_install(*install_args: str):
        _run(*pip, "install", "--user", "--progress-bar=off", *install_args)
        reqs = [arg for arg in install_args if not arg.startswith("-")]
        (installed_reqs_no_deps if "--no-deps" in install_args else installed_reqs).extend(reqs)

    with StdoutTextFold("install dependencies"):
        # Any previous cache should not really be needed, in case we install something new below.
        _run(*pip, "cache", "purge")

        pip_install("--upgrade", "pip", "setuptools", "wheel")
        pip_install("--upgrade", "pytest")
        pip_install("--upgrade", "psutil")  # for some tests
        _run("sudo", "apt-get", "install", "-y", "libsndfile1")  # soundfile, librosa, ESPnet
        pip_install("--upgrade", "dm-tree", "h5py")
        if args.espnet and not args.espnet_no_deps:
            numpy_req = "numpy==1.23.5"  # for ESPnet, ctc-segmentation, etc
        elif args.jax:
            numpy_req = "numpy>=2.1"  # JAX requires it
        else:
            numpy_req = "numpy<2"
        pip_install(numpy_req)
        pip_install("--upgrade", "scipy")  # for some tests
        pip_install("--upgrade", "sentencepiece")  # for some tests

        if sys.version_info[:2] == (3, 10):
            # https://github.com/rwth-i6/returnn/issues/1803, https://github.com/AnswerDotAI/fastcore/issues/751
            pip_install("fastcore==1.12.1")

        if args.tf:
            print("Installing TensorFlow version:", args.tf)
            tf_version = tuple(int(n) for n in args.tf.split(".")) if args.tf else ()

            if tf_version[:2] == (2, 10):
                # TF 2.10 requires gast<=0.4.0,>=0.2.1. But for example, with gast 0.2.2, we get some AutoGraph error:
                # Cause: module 'gast' has no attribute 'Constant'
                # Similar like: https://github.com/tensorflow/tensorflow/issues/47802
                pip_install("gast<=0.4.0")

            # Retry several times in case download breaks. https://github.com/pypa/pip/issues/4796
            for try_nr in range(3):
                try:
                    pip_install(f"tensorflow=={args.tf}")
                    break
                except subprocess.CalledProcessError:
                    if try_nr >= 2:
                        raise
        else:
            tf_version = ()

        if args.torch:
            print("Installing PyTorch version:", args.torch)

            ex_torch_version = _get_torch_version()
            if ex_torch_version:
                if ex_torch_version != args.torch:
                    # Free disk space first. Can run out of disk space otherwise.
                    # Also, remove any nvidia packages to avoid conflicts (https://github.com/rwth-i6/returnn/issues/1802).
                    for pkg in subprocess.check_output([*pip, "freeze"]).splitlines():
                        if pkg.startswith(b"nvidia-"):
                            pkg_s = pkg.decode("utf-8").strip()
                            _run(*pip, "uninstall", "-y", pkg_s)
                    _run(*pip, "uninstall", "-y", "torch")
                else:
                    # Already installed, matching Torch version.
                    # nvidia packages of another Torch version are removed with the other unneeded packages below.
                    print(f"PyTorch {ex_torch_version} already installed.")

            pip_install(f"torch=={args.torch}")
            # numpy pin in the SAME resolve: recent onnxruntime requires numpy>=2, and a bare
            # install silently upgraded numpy, breaking the ABI of torch built against 1.x
            # (CI 2026-08-17: "_ARRAY_API not found" on the torch 1.13 jobs).
            pip_install("onnx", "onnxruntime", numpy_req)
            # lovely-numpy==0.2.20, lovely-tensors==0.1.22 cause some issues:
            # https://github.com/rwth-i6/returnn/issues/1819
            # Maybe specifically with numpy==1.23.5?
            pip_install("lovely-numpy==0.2.18", "lovely-tensors==0.1.21")

            # Needed for some tests.
            # transformers 4.50 requires PyTorch >2.0, so stick to transformers 4.49 for now.
            # (https://github.com/rwth-i6/returnn/issues/1706)
            # safetensors >=0.6 requires PyTorch >2.0, so stick to safetensors 0.5.3 for now.
            # (https://github.com/rwth-i6/returnn/issues/1747)
            if sys.version_info[:2] <= (3, 8):
                # Need older version for Python 3.8. Install whatever is available.
                pip_install("transformers")
            else:
                pip_install("safetensors==0.5.3", "transformers==4.49.0")

        if args.jax:
            print("Installing JAX version:", args.jax)
            # The plain package is CPU-only (unlike torch, whose PyPI wheel is the CUDA build).
            # The GPU support would be the cuda12 extra, ~225 MB of extra wheels
            # for a plugin that a CI runner without a GPU never loads.
            # jaxlib is pinned to the same version by jax itself.
            # optax and orbax-checkpoint are separate packages (not part of jax/jaxlib) but hard
            # dependencies of the RF JAX backend: JAX ships neither losses/optimizers nor checkpoints.
            pip_install(f"jax=={args.jax}", "optax", "orbax-checkpoint")

        if args.hf_datasets:
            assert args.torch, "Need to specify --torch when specifying --hf-datasets"
            assert args.hf_datasets.lower() in ["yes", "1", "true"], (
                f"Invalid value for --hf-datasets: {args.hf_datasets}"
            )
            _run("sudo", "apt-get", "update")
            _run("sudo", "apt-get", "install", "-y", "ffmpeg")  # for torchcodec
            pip_install("torchcodec==0.7")  # for HF datasets
            pip_install("datasets")

        if args.nemo_speech:
            assert args.torch, "Need to specify --torch when specifying --nemo-speech"
            assert args.nemo_speech.lower() in ["yes", "1", "true"], (
                f"Invalid value for --nemo-speech: {args.nemo_speech}"
            )
            # Git main: needs Lhotse indexed data access (index packs etc), not in a NeMo release yet.
            pip_install("nemo-toolkit[asr] @ git+https://github.com/NVIDIA-NeMo/Speech.git", f"torch=={args.torch}")

        if args.espnet and not args.espnet_no_deps and sys.version_info[:2] <= (3, 8):
            # https://github.com/rwth-i6/returnn/issues/1729
            pip_install("ctc-segmentation==1.6.6", "pyworld==0.3.4")

        if args.espnet:
            if args.espnet.lower() in ["yes", "1", "true"]:
                espnet = "202506"  # last version w/ numpy<2
            else:
                raise ValueError(f"Invalid value for --espnet: {args.espnet}")

            if args.espnet_no_deps:
                # The code inspection only needs to resolve the espnet imports, it never imports them,
                # and nothing else in that job does either.
                # With its dependencies, ESPnet would pin numpy<1.24,
                # which modern TF (numpy>=1.26) and modern JAX cannot live with.
                pip_install("--no-deps", f"espnet=={espnet}")
            else:
                assert args.torch, "Need to specify --torch when specifying --espnet"
                # Install ESPnet.
                # Fix Torch version (https://github.com/rwth-i6/returnn/issues/1770#issuecomment-3536630432).
                pip_install(f"espnet=={espnet}", f"torch=={args.torch}")

                # TorchAudio needed by ESPnet.
                # https://pytorch.org/audio/stable/installation.html#compatibility-matrix
                if args.torch == "2.0.0":
                    pip_install("torchaudio==2.0.1")
                elif args.torch == "1.13.1":
                    pip_install("torchaudio==0.13.1")
                else:
                    pip_install(f"torchaudio=={args.torch}")

        if args.tf and tf_version[:2] <= (2, 3):
            # Do this after installing other packages, as those other packages might install newer numpy.
            # Older TF needs older NumPy version.
            # https://github.com/rwth-i6/returnn/pull/1160#issuecomment-1284537803
            pip_install("numpy==1.19.5")

        if args.tf and tf_version[:2] <= (2, 10):
            # Do this after installing other packages, as those other packages might install newer protobuf.
            # Older TF needs also older protobuf version.
            # https://github.com/rwth-i6/returnn/issues/1209
            pip_install("protobuf<=3.20.1")

        # The cache can come from another job (restore-keys): remove the packages this job does not install.
        _uninstall_user_packages_not_required(installed_reqs, no_deps_reqs=installed_reqs_no_deps)

        # Any cached files should not be needed anymore.
        _run(*pip, "cache", "purge")

    with StdoutTextFold("post-install space usage"):
        print("Space usage info after installing dependencies:")
        _run("df", "-h")
        print("Local home dir usage:")
        _run("du", "-h", "-d1", os.path.expanduser("~"))
        _run("du", "-h", "-d1", os.path.expanduser("~/.local"))
        print("Cache dir sizes:")
        _run("du", "-h", "-d1", os.path.expanduser("~/.cache"))

    with StdoutTextFold("installed packages"):
        # List all what we have installed.
        _run(*pip, "freeze")

    print("Dependency setup completed.")
    print("Testing versions:")

    _run(py, "-c", "import numpy; print('NumPy:', numpy.version.full_version)")
    if args.torch:
        try:
            _run(py, "-c", "import torch")
        except subprocess.CalledProcessError:
            _run(py, f"{__my_dir__}/_pytorch_collect_env_ext.py")
        _run(py, "-c", "import torch; print('PyTorch:', torch.__version__, torch.__file__)")
        # torch.__version__ is for example "1.13.1+cu117"
        _run(
            py,
            "-c",
            f"import torch; assert (torch.__version__ + '+').startswith('{args.torch}+')",
        )
    if args.tf:
        _run(py, "-c", "import tensorflow as tf; print('TensorFlow:', tf.__git_version__, tf.__version__, tf.__file__)")
        _run(py, "-c", f"import tensorflow as tf; assert tf.__version__ == '{args.tf}'")
    if args.jax:
        _run(py, "-c", "import jax; print('JAX:', jax.__version__, jax.__file__, jax.devices())")
        _run(py, "-c", f"import jax; assert jax.__version__ == '{args.jax}'")

    print("Pytest env:")
    _run(py, "-m", "pytest", "--version")

    print("All done.")


def _run(*args):
    print("$", " ".join(args))
    sys.stdout.flush()
    subprocess.run(args, check=True)


def _get_torch_version() -> Optional[str]:
    """
    pip show torch | grep "^Version:" | cut -d' ' -f2
    """
    py = sys.executable
    pip = [py, "-m", "pip"]
    try:
        out = subprocess.check_output(pip + ["show", "torch"])
    except subprocess.CalledProcessError:
        return None
    for line in out.splitlines():
        if line.startswith(b"Version:"):
            line_s = line.decode("utf-8")
            return line_s.split(" ")[1].strip()
    raise RuntimeError("Cannot determine torch version: output:\n%s" % out.decode("utf-8"))


def _uninstall_user_packages_not_required(reqs: List[str], *, no_deps_reqs: List[str]):
    """
    Uninstall all packages in the user site which are not in the dependency closure of the given requirements.

    :param reqs: requirements (as for pip install), together with their dependencies (recursively, with extras)
    :param no_deps_reqs: requirements to keep, without their dependencies
    """
    import site
    import importlib.metadata

    user_site = site.getusersitepackages()
    site.addsitedir(user_site)  # it might not have existed at startup
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    dists: Dict[str, List[importlib.metadata.Distribution]] = {}
    for dist in importlib.metadata.distributions():
        dists.setdefault(canonicalize_name(dist.metadata["Name"]), []).append(dist)

    visited_extras: Dict[str, Set[str]] = {}  # name -> extras
    queue = [(Requirement(req), True) for req in reqs] + [(Requirement(req), False) for req in no_deps_reqs]
    while queue:
        req, with_deps = queue.pop()
        name = canonicalize_name(req.name)
        extras = {""} | req.extras
        if extras <= visited_extras.get(name, set()):
            continue
        visited_extras.setdefault(name, set()).update(extras)
        if not with_deps:
            continue
        for dist in dists.get(name, []):
            for dep_s in dist.requires or []:
                dep = Requirement(dep_s)
                if dep.marker is None or any(dep.marker.evaluate({"extra": extra}) for extra in extras):
                    queue.append((dep, True))

    unneeded = sorted(
        dist.metadata["Name"]
        for name, dists_ in dists.items()
        if name not in visited_extras
        for dist in dists_
        if os.path.realpath(dist.locate_file("")) == os.path.realpath(user_site)
    )
    print("Packages in the user site not required by this job:", unneeded)
    if unneeded:
        _run(sys.executable, "-m", "pip", "uninstall", "-y", *unneeded)


IsTravisEnv = os.environ.get("TRAVIS") == "true"
IsGithubActionsEnv = os.environ.get("GITHUB_ACTIONS") == "true"


class StdoutTextFold:
    """
    Context manager to create a fold in stdout (for CI systems that support it).
    """

    def __init__(self, name: str):
        self.name = name
        self.start_time: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()

        if IsGithubActionsEnv:
            # https://github.community/t/has-github-action-somthing-like-travis-fold/16841
            print("::group::%s" % self.name)

        if IsTravisEnv:
            # travis_fold: https://github.com/travis-ci/travis-ci/issues/1065
            print("travis_fold:start:%s" % self.name)

        sys.stdout.flush()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def finish(self):
        """
        End fold.
        """
        elapsed_time = time.time() - self.start_time
        print("%s: Elapsed time: %s" % (self.name, hms(elapsed_time)))

        if IsTravisEnv:
            print("travis_fold:end:%s" % self.name)

        if IsGithubActionsEnv:
            print("::endgroup::")

        sys.stdout.flush()


def hms(s):
    """
    :param float|int s: seconds
    :return: e.g. "1:23:45" (hs:ms:secs). see hms_fraction if you want to get fractional seconds
    :rtype: str
    """
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Error: Command failed: {exc.cmd}")
        sys.exit(1)
