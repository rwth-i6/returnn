#!/usr/bin/env python3

"""
Install necessary dependencies for CI.
Some are potentially already installed (e.g. via cache).
"""

import os
import sys
import re
import argparse
import subprocess


__my_dir__ = os.path.dirname(os.path.abspath(__file__))


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--python", help="Verify Python version (optional)")
    arg_parser.add_argument("--torch", help="PyTorch version to install (optional)")
    arg_parser.add_argument("--tf", help="TensorFlow version to install (optional)")
    arg_parser.add_argument("--espnet", help="Whether to install ESPnet (optional)")
    arg_parser.add_argument("--hf-datasets", help="Whether to install HF datasets (optional)")
    args = arg_parser.parse_args()

    print("Python:", sys.version, sys.executable)

    if args.python:
        print("Verifying Python version:", args.python)
        version = tuple(int(n) for n in args.python.split("."))
        assert sys.version_info[: len(version)] == version, f"Expected Python {args.python}"

    _run(f"{__my_dir__}/_free_disk_space.sh")

    py = sys.executable
    pip = [py, "-m", "pip"]
    pip_install = [*pip, "install", "--user", "--progress-bar=off", "--upgrade"]

    # Any previous cache should not really be needed, in case we install something new below.
    _run(*pip, "cache", "purge")

    _run(*pip_install, "pip", "setuptools", "wheel")
    _run(*pip_install, "pytest")
    _run("sudo", "apt-get", "install", "-y", "libsndfile1")  # soundfile, librosa, ESPnet
    _run(*pip_install, "dm-tree", "h5py")
    if args.espnet:
        _run(*pip_install, "numpy==1.23.5")  # for ESPnet, ctc-segmentation, etc
    else:
        _run(*pip_install, "numpy<2")
    _run(*pip_install, "scipy")  # for some tests

    if sys.version_info[:2] == (3, 10):
        # https://github.com/rwth-i6/returnn/issues/1803, https://github.com/AnswerDotAI/fastcore/issues/751
        _run(*pip_install, "fastcore==1.12.1")

    if args.tf:
        print("Installing TensorFlow version:", args.tf)

        if re.match(r"^2\.[0123]\..*$", args.tf):
            # Older TF needs older NumPy version.
            # https://github.com/rwth-i6/returnn/pull/1160#issuecomment-1284537803
            _run(*pip_install, "numpy==1.19.5")
            # Older TF needs also older protobuf version.
            # https://github.com/rwth-i6/returnn/issues/1209
            _run(*pip_install, "protobuf<=3.20.1")

        if args.tf.startswith("2.10."):
            # TF 2.10 requires gast<=0.4.0,>=0.2.1. But for example, with gast 0.2.2, we get some AutoGraph error:
            # Cause: module 'gast' has no attribute 'Constant'
            # Similar like: https://github.com/tensorflow/tensorflow/issues/47802
            _run(*pip_install, "gast<=0.4.0")

        # Retry several times in case download breaks. https://github.com/pypa/pip/issues/4796
        for try_nr in range(3):
            try:
                _run(*pip_install, f"tensorflow=={args.tf}")
                break
            except subprocess.CalledProcessError:
                if try_nr >= 2:
                    raise

    if args.torch:
        print("Installing PyTorch version:", args.torch)

        if _get_torch_version() != args.torch:
            # Free disk space first. Can run out of disk space otherwise.
            # Also, remove any nvidia packages to avoid conflicts (https://github.com/rwth-i6/returnn/issues/1802).
            for pkg in subprocess.check_output([*pip, "freeze"]).splitlines():
                if pkg.startswith(b"nvidia-"):
                    pkg_s = pkg.decode("utf-8").strip()
                    _run(*pip, "uninstall", "-y", pkg_s)
            _run(*pip, "uninstall", "-y", "torch")

        _run(*pip_install, f"torch=={args.torch}")
        _run(*pip_install, "onnx", "onnxruntime")
        _run(*pip_install, "lovely_tensors")

        # Needed for some tests.
        # transformers 4.50 requires PyTorch >2.0, so stick to transformers 4.49 for now.
        # (https://github.com/rwth-i6/returnn/issues/1706)
        # safetensors >=0.6 requires PyTorch >2.0, so stick to safetensors 0.5.3 for now.
        # (https://github.com/rwth-i6/returnn/issues/1747)
        if sys.version_info[:2] <= (3, 8):
            # Need older version for Python 3.8. Install whatever is available.
            _run(*pip_install, "transformers")
        else:
            _run(*pip_install, "safetensors==0.5.3", "transformers==4.49.0")

    if args.hf_datasets:
        assert args.torch, "Need to specify --torch when specifying --hf-datasets"
        assert args.hf_datasets.lower() in ["yes", "1", "true"], f"Invalid value for --hf-datasets: {args.hf_datasets}"
        _run("sudo", "apt-get", "update")
        _run("sudo", "apt-get", "install", "-y", "ffmpeg")  # for torchcodec
        _run(*pip_install, "torchcodec==0.7")  # for HF datasets
        _run(*pip_install, "datasets")

    if args.espnet and sys.version_info[:2] <= (3, 8):
        # https://github.com/rwth-i6/returnn/issues/1729
        _run(*pip_install, "ctc-segmentation==1.6.6", "pyworld==0.3.4")

    if args.espnet:
        assert args.torch, "Need to specify --torch when specifying --espnet"
        if args.espnet.lower() in ["yes", "1", "true"]:
            espnet = "202506"  # last version w/ numpy<2
        else:
            raise ValueError(f"Invalid value for --espnet: {args.espnet}")
        # Install ESPnet. Fix Torch version (https://github.com/rwth-i6/returnn/issues/1770#issuecomment-3536630432).
        _run(*pip_install, f"espnet=={espnet}", f"torch=={args.torch}")

        # TorchAudio needed by ESPnet.
        # https://pytorch.org/audio/stable/installation.html#compatibility-matrix
        if args.torch == "2.0.0":
            _run(*pip_install, "torchaudio==2.0.1")
        elif args.torch == "1.13.1":
            _run(*pip_install, "torchaudio==0.13.1")
        else:
            _run(*pip_install, f"torchaudio=={args.torch}")

    # Any cached files should not be needed anymore.
    _run(*pip, "cache", "purge")

    print("Space usage info after installing dependencies:")
    _run("df", "-h")
    print("Local home dir usage:")
    _run("du", "-h", "-d1", os.path.expanduser("~"))
    _run("du", "-h", "-d1", os.path.expanduser("~/.local"))
    print("Cache dir sizes:")
    _run("du", "-h", "-d1", os.path.expanduser("~/.cache"))

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

    print("Pytest env:")
    _run(py, "-m", "pytest", "--version")

    print("All done.")


def _run(*args):
    print("$", " ".join(args))
    sys.stdout.flush()
    subprocess.run(args, check=True)


def _get_torch_version() -> str:
    """
    pip show torch | grep "^Version:" | cut -d' ' -f2
    """
    py = sys.executable
    pip = [py, "-m", "pip"]
    out = subprocess.check_output(pip + ["show", "torch"])
    for line in out.splitlines():
        if line.startswith(b"Version:"):
            line_s = line.decode("utf-8")
            return line_s.split(" ")[1].strip()
    raise RuntimeError("Cannot determine torch version: output:\n%s" % out.decode("utf-8"))


if __name__ == "__main__":
    main()
