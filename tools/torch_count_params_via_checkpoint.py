#!/usr/bin/env python3

"""
Inspect checkpoint
"""

from __future__ import annotations
from typing import Tuple
import argparse
import numpy
import torch


def main():
    """main"""
    # First set PyTorch default print options. This might get overwritten by the args below.
    numpy.set_printoptions(precision=4, linewidth=80)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("checkpoint")
    arg_parser.add_argument("--ignore-prefix", nargs="*", default=[])
    args = arg_parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu", mmap=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    assert isinstance(state, dict)

    counts = {}

    for k, v in state.items():
        if any(k.startswith(prefix) for prefix in args.ignore_prefix):
            continue
        if isinstance(v, numpy.ndarray):
            v = torch.tensor(v)
        assert isinstance(v, torch.Tensor)
        print(f"{k}: {v.dtype} {_format_shape(v.shape)} {v.numel()}")
        k_ = tuple(k.split("."))
        for i in reversed(range(len(k_) + 1)):
            k__ = k_[:i]
            prev = counts.pop(k__, 0)  # pop to insert the new value at the end of the dict
            counts[k__] = prev + v.numel()

    print("Counts:")
    for k, v in counts.items():
        print(f"{'.'.join(k) or '.'}: {v} {human_size(v)}")


def _format_shape(shape: Tuple[int, ...]) -> str:
    return "[%s]" % ",".join(map(str, shape))


def human_size(n, factor=1000, frac=0.8, prec=1):
    """
    :param int|float n:
    :param int factor: for each of the units K, M, G, T
    :param float frac: when to go over to the next bigger unit
    :param int prec: how much decimals after the dot
    :return: human readable size, using K, M, G, T
    :rtype: str
    """
    postfixes = ["", "K", "M", "G", "T"]
    i = 0
    while i < len(postfixes) - 1 and n > (factor ** (i + 1)) * frac:
        i += 1
    if i == 0:
        return str(n)
    return ("%." + str(prec) + "f") % (float(n) / (factor**i)) + postfixes[i]


if __name__ == "__main__":
    main()
