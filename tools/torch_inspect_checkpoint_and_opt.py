#!/usr/bin/env python3

"""
Inspect checkpoint
"""

from __future__ import annotations

import os
import argparse
import numpy

import _setup_returnn_env  # noqa
from torch_inspect_checkpoint import parse_numpy_printoption, print_object, _print_key_value  # noqa
from returnn.config import Config, set_global_config
from returnn.log import log
from returnn.torch.engine import Engine


def main():
    """main"""
    print(f"{os.path.basename(__file__)}: {__doc__.strip()}")

    # First set PyTorch default print options. This might get overwritten by the args below.
    numpy.set_printoptions(precision=4, linewidth=80)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("returnn_config")
    arg_parser.add_argument("--cwd")
    arg_parser.add_argument("--key", type=str, default="", help="Name of the tensor or object to inspect")
    arg_parser.add_argument("--all_tensors", action="store_true", help="If True, print the values of all the tensors.")
    arg_parser.add_argument("--stats_only", action="store_true")
    arg_parser.add_argument(
        "--printoptions",
        nargs="*",
        type=parse_numpy_printoption,
        help="Argument for numpy.set_printoptions(), in the form 'k=v'.",
    )
    arg_parser.add_argument("--device", default="cpu")
    args = arg_parser.parse_args()

    if args.cwd:
        print("* Change working dir:", args.cwd)
        os.chdir(args.cwd)

    log.initialize(verbosity=[5])
    config = Config()
    print("* Load config:", args.returnn_config)
    config.load_file(args.returnn_config)
    set_global_config(config)
    for k in [
        "train",
        "dev",
        "eval",
        "eval_datasets",
        "torch_amp",
        "grad_scaler",
        "torch_distributed",
        "learning_rate_control",
        "learning_rate_file",
    ]:
        config.typed_dict.pop(k, None)
    config.set("device", args.device)

    print("* Setup RETURNN engine")
    engine = Engine(config=config)
    print("* Load model and optimizer state")
    engine.init_train_from_config()
    model = engine.get_pt_model()
    assert model is not None, "No model loaded?"
    opt = engine.get_pt_optimizer()
    assert opt is not None, "No optimizer loaded?"
    print("* Loaded.")

    if args.key:
        obj = model.get_parameter(args.key)
        print(f"{args.key}:")
        print_object(obj, stats_only=args.stats_only)
        print("Optimizer state:")
        if obj in opt.state:
            print_object(opt.state[obj], stats_only=args.stats_only)
        else:
            print("(None)")
    else:
        for name, param in model.named_parameters():
            _print_key_value(name, param, print_all_tensors=args.all_tensors, stats_only=args.stats_only)
            if param in opt.state:
                print("  Optimizer state:")
                print_object(
                    opt.state[param], prefix="  ", print_all_tensors=args.all_tensors, stats_only=args.stats_only
                )
            else:
                print("  Optimizer state: (None)")


if __name__ == "__main__":
    main()
