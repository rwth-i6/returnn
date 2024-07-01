#!/usr/bin/env python3

"""
Inspect checkpoint
"""

from __future__ import annotations
from typing import Optional, Union, Any, Tuple, List, Dict
import argparse
import numpy
import torch


def main():
    """main"""
    # First set PyTorch default print options. This might get overwritten by the args below.
    numpy.set_printoptions(precision=4, linewidth=80)

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("checkpoint")
    arg_parser.add_argument(
        "--key",
        type=str,
        default="",
        help="Name of the tensor or object to inspect."
        " If not given, list them all (but without values, unless --all_tensors).",
    )
    arg_parser.add_argument("--all_tensors", action="store_true", help="If True, print the values of all the tensors.")
    arg_parser.add_argument(
        "--stats_only",
        action="store_true",
        help="with --all_tensors or --key, print only stats of tensor, not all values",
    )
    arg_parser.add_argument(
        "--stats",
        action="store_true",
        help="with --key, just like --stats_only, otherwise like --all_tensors --stats_only",
    )
    arg_parser.add_argument(
        "--interesting_exclude", help="list of tensors to exclude, separated by comma, matching part of name"
    )
    arg_parser.add_argument(
        "--printoptions",
        nargs="*",
        type=parse_numpy_printoption,
        help="Argument for numpy.set_printoptions(), in the form 'k=v'.",
    )
    arg_parser.add_argument("--device", default="cpu")
    arg_parser.add_argument("--mmap", action="store_true")
    args = arg_parser.parse_args()

    kwargs = {}
    if args.mmap:
        kwargs["mmap"] = True
    state = torch.load(args.checkpoint, map_location=args.device, **kwargs)
    if args.key:
        assert isinstance(state, dict)
        if args.key not in state and "model" in state:
            state = state["model"]
        obj = state[args.key]
        print(f"{args.key}:")
        print_object(obj, stats_only=args.stats_only or args.stats)
    else:
        ctx = PrintCtx(exclude=args.interesting_exclude.split(",") if args.interesting_exclude else [])
        print_object(
            state, print_all_tensors=args.all_tensors or args.stats, stats_only=args.stats_only or args.stats, ctx=ctx
        )
        ctx.report()


def print_object(
    obj: Any,
    *,
    print_all_tensors: bool = False,
    stats_only: bool = False,
    prefix: str = "",
    ctx: Optional[PrintCtx] = None,
    ctx_name: Optional[str] = None,
):
    """print object"""
    if isinstance(obj, (dict, list, tuple)):
        for k, v in obj.items() if isinstance(obj, dict) else enumerate(obj):
            _print_key_value(
                k,
                v,
                print_all_tensors=print_all_tensors,
                stats_only=stats_only,
                prefix=prefix,
                ctx=ctx,
                ctx_name=f"{ctx_name}.{k}" if ctx_name else f"{k}",
            )
    elif isinstance(obj, (numpy.ndarray, torch.Tensor)):
        print_tensor(obj, stats_only=stats_only, prefix=prefix, ctx=ctx, ctx_name=ctx_name)
    else:
        print(f"{prefix}({type(obj)}) {obj}")


def _print_key_value(
    k: Any,
    v: Union[numpy.ndarray, torch.Tensor],
    *,
    print_all_tensors: bool = False,
    stats_only: bool = False,
    prefix: str = "",
    ctx: PrintCtx,
    ctx_name: str,
):
    if isinstance(v, numpy.ndarray):
        v = torch.tensor(v)
    if isinstance(v, torch.Tensor):
        if v.numel() <= 1 and v.device.type != "meta":
            print(f"{prefix}{k}: {v.dtype} {_format_shape(v.shape)} {_r(v)}")
        else:
            print(f"{prefix}{k}: {v.dtype} {_format_shape(v.shape)}")
            if print_all_tensors:
                print_tensor(
                    v,
                    with_type_and_shape=False,
                    stats_only=stats_only,
                    prefix=prefix + "  ",
                    ctx=ctx,
                    ctx_name=ctx_name,
                )
    elif isinstance(v, (dict, list, tuple)):
        print(f"{prefix}{k}: ({type(v).__name__})")
        print_object(
            v,
            print_all_tensors=print_all_tensors,
            stats_only=stats_only,
            prefix=prefix + "  ",
            ctx=ctx,
            ctx_name=ctx_name,
        )
    else:
        print(f"{prefix}{k}: ({type(v).__name__}) {v}")


def print_tensor(
    v: Union[numpy.ndarray, torch.Tensor],
    *,
    prefix: str = "",
    with_type_and_shape: bool = True,
    stats_only: bool = False,
    ctx: Optional[PrintCtx] = None,
    ctx_name: Optional[str] = None,
):
    """print tensor"""
    if isinstance(v, numpy.ndarray):
        v = torch.tensor(v)
    assert isinstance(v, torch.Tensor)
    if with_type_and_shape:
        print(f"{prefix}{v.dtype}, {_format_shape(v.shape)}")
    if not stats_only:
        print(v.detach().cpu().numpy())
    n = v.numel()
    if n > 1:
        v_, _ = v.flatten().sort()
        if v.is_floating_point():
            # See :func:`variable_scalar_summaries_dict`.
            mean = torch.mean(v)
            max_abs = max(torch.abs(v_[[0, -1]]))
            print(
                f"{prefix}mean, stddev, max abs:"
                f" {_r(mean)}, {_r(torch.sqrt(torch.mean(torch.square(v - mean))))}, {_r(max_abs)}"
            )
            if ctx and ctx_name:
                ctx.visit_tensor(name=ctx_name, tensor=v, max_abs=max_abs)
        print(
            f"{prefix}min, p05, p50, p95, max:"
            f" {_r(v_[0])},"
            f" {', '.join(_r(v_[int(n * q)]) for q in (0.05, 0.5, 0.95))},"
            f" {_r(v_[-1])}"
        )


def _format_shape(shape: Tuple[int, ...]) -> str:
    return "[%s]" % ",".join(map(str, shape))


def _r(num: Union[torch.Tensor, float]) -> str:
    return numpy.array2string(num.detach().cpu().numpy() if isinstance(num, torch.Tensor) else num)


class PrintCtx:
    """print ctx, maybe collect interesting global info"""

    def __init__(self, *, exclude: List[str]):
        self.interesting: Dict[str, Tuple[float, str, torch.Tensor]] = {}
        self.exclude = exclude

    def visit_tensor(self, *, name: str, tensor: torch.Tensor, max_abs: float):
        """visit"""
        for exclude_name in self.exclude:
            if exclude_name in name:
                return
        if tensor.ndim > 0:  # ignore scalars for max_abs
            key = "largest max abs"
            if key not in self.interesting or self.interesting[key][0] < max_abs:
                self.interesting[key] = (max_abs, name, tensor)
            key = "smallest max abs"
            if key not in self.interesting or self.interesting[key][0] > max_abs:
                self.interesting[key] = (max_abs, name, tensor)

    def report(self):
        """report"""
        if not self.interesting:
            return
        print("Collected interesting tensors:")
        for k, (v, name, tensor) in self.interesting.items():
            print(f"{k} {_r(v)}: {name}: {tensor.dtype} {_format_shape(tensor.shape)}")
            print_tensor(tensor, with_type_and_shape=False, stats_only=True, prefix="  ")


def parse_numpy_printoption(kv_str):
    """Sets a single numpy printoption from a string of the form 'x=y'.

    See documentation on numpy.set_printoptions() for details about what values
    x and y can take. x can be any option listed there other than 'formatter'.

    Args:
      kv_str: A string of the form 'x=y', such as 'threshold=100000'

    Raises:
      argparse.ArgumentTypeError: If the string couldn't be used to set any
          nump printoption.
    """
    k_v_str = kv_str.split("=", 1)
    if len(k_v_str) != 2 or not k_v_str[0]:
        raise argparse.ArgumentTypeError("'%s' is not in the form k=v." % kv_str)
    k, v_str = k_v_str
    printoptions: Dict[str, Any] = numpy.get_printoptions()
    if k not in printoptions:
        raise argparse.ArgumentTypeError("'%s' is not a valid printoption." % k)
    v_type = type(printoptions[k])
    if printoptions[k] is None:
        raise argparse.ArgumentTypeError("Setting '%s' from the command line is not supported." % k)
    try:
        v = v_type(v_str) if v_type is not bool else _to_bool(v_str)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    numpy.set_printoptions(**{k: v})


def _to_bool(s: str) -> bool:
    """
    :param s: str to be converted to bool, e.g. "1", "0", "true", "false"
    :return: boolean value, or fallback
    """
    s = s.lower()
    if s in ["1", "true", "yes", "y"]:
        return True
    if s in ["0", "false", "no", "n"]:
        return False
    raise ValueError(f"invalid bool: {s!r}")


if __name__ == "__main__":
    main()
