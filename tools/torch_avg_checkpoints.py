#!/usr/bin/env python3

"""
Average model checkpoints.

References:
    Our :file:`tf_avg_checkpoins.py`.
    https://github.com/espnet/espnet/blob/master/utils/average_checkpoints.py
    https://github.com/facebookresearch/fairseq/blob/main/scripts/average_checkpoints.py
"""

from __future__ import annotations

import os
from typing import Optional, Any, Sequence, Dict
from collections import defaultdict
import argparse
import torch


def main():
    """main entry"""
    print(f"RETURNN {os.path.basename(__file__)} -- average PyTorch model checkpoints")
    print("PyTorch version:", torch.__version__)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--checkpoints", nargs="+", required=True, help="comma-separated (or multiple provided) input checkpoints"
    )
    arg_parser.add_argument("--separator", default=",", help="custom separator for --checkpoints, can also disable it")
    arg_parser.add_argument("--prefix", default="", help="add this as a prefix to the input checkpoints")
    arg_parser.add_argument("--postfix", default="", help="add this as a postfix to the input checkpoints (e.g. '.pt')")
    arg_parser.add_argument("--output_path", required=True, help="output checkpoint")
    args = arg_parser.parse_args()

    in_ckpts = []

    def _add_in_ckpt(name: str):
        in_ckpt__ = args.prefix + name + args.postfix
        if not os.path.exists(in_ckpt__):
            raise Exception(
                f"input checkpoint not found: {in_ckpt__!r}, "
                f"prefix {args.prefix!r}, postfix {args.postfix!r}, name {name!r}"
            )
        print("in ckpt:", in_ckpt__)
        in_ckpts.append(in_ckpt__)

    for in_ckpt in args.checkpoints:
        in_ckpt: str
        if args.separator:
            for in_ckpt_ in in_ckpt.split(args.separator):
                _add_in_ckpt(in_ckpt_)
        else:
            _add_in_ckpt(in_ckpt)

    print("out ckpt:", args.output_path)
    merge_checkpoints(in_ckpts=in_ckpts, out_ckpt=args.output_path)
    print("Done.")


def merge_checkpoints(in_ckpts: Sequence[str], out_ckpt: str, extra_state: Optional[Dict[str, Any]] = None):
    """
    Merge checkpoints
    """
    out_model_state: Dict[str, torch.Tensor] = {}
    out_model_state_num: Dict[str, int] = defaultdict(int)
    out_state: Dict[str, Any] = {"model": out_model_state, "merged_epochs": [], "merged_steps": []}
    for in_ckpt in in_ckpts:
        print("read ckpt:", in_ckpt)
        torch_version = tuple(int(s) for s in str(torch.__version__).split(".")[:2])
        load_kwargs = dict(map_location=torch.device("cpu"), mmap=True)
        if torch_version < (2, 1):
            load_kwargs.pop("mmap")  # mmap flag only introduced from 2.1.0 onwards
        in_state = torch.load(in_ckpt, **load_kwargs)
        assert isinstance(in_state, dict)

        assert "model" in in_state and isinstance(in_state["model"], dict)
        covered_keys = {"model"}
        for k, v in in_state["model"].items():
            k: str
            v: torch.Tensor
            if k in out_model_state:
                out_model_state[k] += v
            else:
                out_model_state[k] = v
            out_model_state_num[k] += 1

        # Take max value of the following
        for k in ["epoch", "step"]:
            covered_keys.add(k)
            if k in in_state:
                if k in out_state:
                    out_state[k] = max(out_state[k], in_state[k])
                else:
                    out_state[k] = in_state[k]

        # Extend list of following
        for k in ["merged_epochs", "merged_steps"]:
            covered_keys.add(k)
            if k in in_state:
                out_state[k].extend(in_state[k])
        out_state["merged_epochs"].append(in_state.get("epoch"))
        out_state["merged_steps"].append(in_state.get("step"))

        # Take first value of all remaining keys
        for k, v in in_state.items():
            if k in covered_keys:
                continue
            if k not in out_state:
                out_state[k] = v

    # Average
    for k in out_model_state:
        out_model_state[k] = out_model_state[k] / out_model_state_num[k]

    if extra_state:
        out_state.update(extra_state)

    print("save ckpt:", out_ckpt)
    torch.save(out_state, out_ckpt)


if __name__ == "__main__":
    main()
