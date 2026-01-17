#!/usr/bin/env python

"""
Extends the PyTorch collect_env.py script.
"""

from typing import Optional
import os
import subprocess
import importlib.util


__my_dir__ = os.path.dirname(os.path.abspath(__file__))


def main():
    _run("python", __my_dir__ + "/_pytorch_collect_env.py")

    print("\nEnvironment Variables:")
    for k, v in os.environ.items():
        print(f"{k}={v}")

    torch_path = _get_torch_path()
    print("\nPyTorch installation path:", torch_path)

    if torch_path:
        # Iterate through *.so files in the torch directory and run ldd on them
        print("\nShared library dependencies for PyTorch .so files:")
        for root, _, files in os.walk(torch_path):
            for file in files:
                if file.endswith(".so"):
                    so_path = os.path.join(root, file)
                    print(f"\nldd output for {so_path}:")
                    _run("ldd", so_path)


def _run(*args):
    print("$", " ".join(args))
    subprocess.run(args, check=True)


def _get_torch_path() -> Optional[str]:
    spec = importlib.util.find_spec("torch")
    if spec is None or spec.origin is None:
        return None
    # spec.origin is e.g. ".../site-packages/torch/__init__.py"
    return os.path.dirname(spec.origin)


if __name__ == "__main__":
    main()
