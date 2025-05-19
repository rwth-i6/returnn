#!/usr/bin/env python3

"""
Scale tuning
"""

from __future__ import annotations
import argparse
import os
import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, Tuple, Set

import _setup_returnn_env  # noqa

from returnn.config import set_global_config, Config
from returnn.log import log
from returnn.util.basic import describe_returnn_version, BehaviorVersion
from returnn.util import better_exchook
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
import torch
import json


def main():
    """main"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--names", required=True, nargs="+", help="names for scores")
    arg_parser.add_argument("--scores", required=True, nargs="+", help="score-paths")
    arg_parser.add_argument("--evaluation", required=True, help="'edit_distance'")
    arg_parser.add_argument("--ref", help="ref-path")
    arg_parser.add_argument("--fixed-scales", nargs="*", help="(name, scale) pairs, fixed scales")
    arg_parser.add_argument("--negative-scales", nargs="*", help="list of names, negative scales")
    arg_parser.add_argument("--scale-relative-to", nargs="*", help="(name, other) pairs, scale relative to")
    arg_parser.add_argument("--max-scale", type=float, default=2.0)
    arg_parser.add_argument("--min-scales", nargs="*", help="(name, scale) pairs, min scales")
    arg_parser.add_argument("--max-scales", nargs="*", help="(name, scale) pairs, max scales")
    arg_parser.add_argument("--num-iterations", type=int, default=10)
    arg_parser.add_argument("--num-steps", type=int, default=21)
    arg_parser.add_argument("--search-mode", default="auto", help="'auto', 'grid', 'single'")
    arg_parser.add_argument("--device", default="cpu", help="auto, cpu, cuda, ...")
    arg_parser.add_argument("--batch-size", type=int, default=1024)
    arg_parser.add_argument("--output-scales", help="file to write (relative) scales into")
    arg_parser.add_argument("--output-real-scales", help="file to write real scales into")
    arg_parser.add_argument("--output-grid-plot", help="file to write grid plot into")
    args = arg_parser.parse_args()

    print(f"{os.path.basename(__file__)}, RETURNN {describe_returnn_version()}")

    config = Config()
    set_global_config(config)

    log.initialize(verbosity=[5])
    BehaviorVersion.set_min_behavior_version(22)
    better_exchook.install()
    rf.select_backend_torch()

    try:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ImportError:
        pass  # ignore

    names = list(args.names)
    names_dim = Dim(len(names), name="names")

    # Resort scale names, such that we first have the fixed scales, then the scale_relative_to.
    fixed_scales_names = set()
    if args.fixed_scales:
        assert len(args.fixed_scales) % 2 == 0
        for name, scale in zip(args.fixed_scales[::2], args.fixed_scales[1::2]):
            fixed_scales_names.add(name)
    scale_relative_to_target_names = set()
    if args.scale_relative_to:
        assert len(args.scale_relative_to) % 2 == 0
        for name, other in zip(args.scale_relative_to[::2], args.scale_relative_to[1::2]):
            scale_relative_to_target_names.add(other)
    names.sort(key=lambda key: (key not in fixed_scales_names, key not in scale_relative_to_target_names))
    print("Names:", names)

    # Prepare scale relation info, based on scale/name idx.
    fixed_scales: Dict[int, float] = {}
    if args.fixed_scales:
        assert len(args.fixed_scales) % 2 == 0
        for name, scale in zip(args.fixed_scales[::2], args.fixed_scales[1::2]):
            name_idx = names.index(name)
            assert name_idx not in fixed_scales
            fixed_scales[name_idx] = float(scale)
    scale_relative_to: Dict[int, int] = {}  # name idx -> other name idx (before)
    if args.scale_relative_to:
        assert len(args.scale_relative_to) % 2 == 0
        for name, other in zip(args.scale_relative_to[::2], args.scale_relative_to[1::2]):
            name_idx = names.index(name)
            other_idx = names.index(other)
            assert name_idx not in scale_relative_to
            assert name_idx not in fixed_scales
            assert other_idx not in fixed_scales
            assert other_idx < name_idx, "make sure, when --scale-relative-to, that the other name is before"
            scale_relative_to[name_idx] = other_idx
    neg_scales: Set[int] = set()
    if args.negative_scales:
        for name in args.negative_scales:
            name_idx = names.index(name)
            assert name_idx not in fixed_scales
            neg_scales.add(name_idx)

    print("Num scales to search over:", len(names) - len(fixed_scales))
    search_mode = args.search_mode
    if search_mode == "auto":
        if len(names) - len(fixed_scales) == 2:
            search_mode = "grid"
        else:
            search_mode = "single"
    print("Search mode:", search_mode)

    # Load data
    vocab: Dict[str, int] = {}
    hyps: Dict[str, List[List[int]]] = {}  # seq_tag -> beam of seqs
    scores: Dict[str, Dict[str, List[float]]] = {}  # name -> seq_tag -> beam of score
    assert args.scores and len(args.scores) == len(names)
    for name, path in zip(args.names, args.scores):
        if not os.path.exists(path):
            raise FileNotFoundError(f"score file not found: {path}, for name: {name}")
        scores[name], hyps = _load_text_dict_hyps_file(
            path, name=name, vocab=vocab, expect_same_hyps=hyps if hyps else None
        )
        assert scores[name] and hyps
    print("num seqs:", len(hyps))

    assert scores and hyps
    print("len vocab after reading hyps:", len(vocab))
    assert vocab, "no labels found?"
    ref: Optional[Dict[str, List[int]]] = (
        _load_text_dict_file(args.ref, name="ref", vocab=vocab) if args.ref is not None else None
    )
    print("len vocab after reading ref:", len(vocab))
    if len(vocab) < 2**15:
        dtype = torch.int16
    elif len(vocab) < 2**31:
        dtype = torch.int32
    elif len(vocab) < 2**63:
        dtype = torch.int64
    else:
        raise ValueError(f"vocab too large: {len(vocab)}")
    print("dtype:", dtype)

    if ref:
        total_ref_seq_len = sum(len(seq) for seq in ref.values())
        print("total ref seq len:", total_ref_seq_len)
        avg_ref_seq_len = total_ref_seq_len / len(ref)
        print("avg ref seq len:", avg_ref_seq_len)
    else:
        avg_ref_seq_len = None

    # Sort by lengths (reversed) to be able to prepare batches without too much padding.
    if ref:
        seq_list_ordered_by_len = sorted(ref.keys(), key=lambda tag: len(ref[tag]), reverse=True)
    else:
        seq_list_ordered_by_len = sorted(hyps.keys(), key=lambda tag: len(hyps[tag][0]), reverse=True)

    if args.device == "auto":
        dev_s = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        dev_s = args.device
    dev = torch.device(dev_s)
    print("Device:", dev)
    _report_dev_memory(dev)

    # Prepare batches, load all into device memory
    # Note: If we get GPU OOM here, we could instead keep this on CPU... Currently, not yet implemented...
    print("Preparing batches, calculating evaluation on all hyps...")
    batches = []
    for i in range(0, len(seq_list_ordered_by_len), args.batch_size):
        print(
            f"Batch {len(batches)}, seqs {i} - {min(i + args.batch_size, len(seq_list_ordered_by_len))}"
            f" / {len(seq_list_ordered_by_len)}, {i / len(seq_list_ordered_by_len) * 100:.1f}%"
        )
        batch_seq_tag_list: List[str] = seq_list_ordered_by_len[i : i + args.batch_size]

        beam_sizes_ = [len(hyps[tag]) for tag in batch_seq_tag_list]
        beam_sizes_t = torch.tensor(beam_sizes_, dtype=torch.int32)  # [Batch], int32
        hyps_seq_lens_: List[List[int]] = [
            [len(hyps[tag][beam]) for beam in range(len(hyps[tag]))] for tag in batch_seq_tag_list
        ]
        hyps_seq_lens_t = _make_padded_tensor_2d(hyps_seq_lens_, dtype=torch.int32)  # [Batch,Beam], int32
        hyps__: List[List[List[int]]] = [hyps[tag] for tag in batch_seq_tag_list]
        hyps_t = _make_padded_tensor_3d(hyps__, dtype=dtype, device=dev)  # [Batch,Beam,HypSeq], int16|int32|int64

        batch_scores: List[List[List[float]]] = [[scores[name][tag] for tag in batch_seq_tag_list] for name in names]
        batch_scores_t = _make_padded_tensor_3d(
            batch_scores, dtype=torch.float32, device=dev
        )  # [Names,Batch,Beam], float32

        if ref is not None:
            ref_: List[List[int]] = [ref[tag] for tag in batch_seq_tag_list]
            ref_t = _make_padded_tensor_2d(ref_, dtype=dtype, device=dev)  # [Batch,RefSeq], int16|int32|int64
            ref_seq_lens = [len(ref[tag]) for tag in batch_seq_tag_list]
            ref_seq_lens_t = torch.tensor(ref_seq_lens, dtype=torch.int32)  # [Batch], int32
        else:
            ref_t = None
            ref_seq_lens_t = None

        batch_dim = Dim(len(batch_seq_tag_list), name="batch")
        beam_sizes_rf = rf.convert_to_tensor(beam_sizes_t, dims=[batch_dim])
        beam_dim = Dim(beam_sizes_rf, name="beam")
        hyps_seq_lens_rf = rf.convert_to_tensor(hyps_seq_lens_t, dims=[batch_dim, beam_dim])
        hyps_seq_dim = Dim(hyps_seq_lens_rf, name="hyps_seq")
        hyps_rf = rf.convert_to_tensor(hyps_t, dims=[batch_dim, beam_dim, hyps_seq_dim])
        if ref_t is not None:
            ref_seq_lens_rf = rf.convert_to_tensor(ref_seq_lens_t, dims=[batch_dim])
            ref_seq_dim = Dim(ref_seq_lens_rf, name="ref_seq")
            ref_rf = rf.convert_to_tensor(ref_t, dims=[batch_dim, ref_seq_dim])
        else:
            ref_seq_dim = None
            ref_rf = None

        if args.evaluation == "edit_distance":
            assert ref_rf is not None, "need --ref for edit_distance"
            res = rf.edit_distance(ref_rf, ref_seq_dim, hyps_rf, hyps_seq_dim)
            assert res.dims_set == {batch_dim, beam_dim}
            res = res.copy_transpose((batch_dim, beam_dim))
            res = rf.cast(res, "float64")
            # Using avg_ref_seq_len makes this numerically more stable, as the numbers are not so small,
            # and we can use reduce_mean below, which is also more stable.
            res /= avg_ref_seq_len  # WER / avg_len = (WER / total_len) * num_seqs -> can use reduce_mean
        else:
            raise ValueError(f"unknown evaluation {args.evaluation!r}")

        batch_scores_rf = rf.convert_to_tensor(batch_scores_t, dims=[names_dim, batch_dim, beam_dim])

        batches.append(
            Batch(
                seq_tags=batch_seq_tag_list,
                batch_dim=batch_dim,
                beam_dim=beam_dim,
                hyps_eval=res,
                names_dim=names_dim,
                scores=batch_scores_rf,
            )
        )
    print("num batches:", len(batches))
    _report_dev_memory(dev)

    def _eval_max_min(*, use_max: bool) -> float:
        hyps_eval_ts = []
        for batch in batches:
            hyps_eval = (rf.reduce_max if use_max else rf.reduce_min)(
                batch.hyps_eval, axis=batch.beam_dim
            )  # [Batch], float64
            assert hyps_eval.dims_set == {batch.batch_dim}
            hyps_eval_ts.append(hyps_eval.raw_tensor)
        hyps_eval_t = torch.concatenate(hyps_eval_ts)  # [NumSeqs], float64
        hyps_eval_t = torch.mean(hyps_eval_t)  # scalar, float64
        return hyps_eval_t.cpu().item()

    best_eval, worst_eval = _eval_max_min(use_max=False), _eval_max_min(use_max=True)
    print(f"Best {args.evaluation}: {best_eval}, worst {args.evaluation}: {worst_eval}")

    scales_min = [0.0] * len(names)
    scales_max = [args.max_scale] * len(names)
    if args.min_scales:
        assert len(args.min_scales) % 2 == 0
        for name, scale in zip(args.min_scales[::2], args.min_scales[1::2]):
            scales_min[names.index(name)] = float(scale)
    if args.max_scales:
        assert len(args.max_scales) % 2 == 0
        for name, scale in zip(args.max_scales[::2], args.max_scales[1::2]):
            scales_max[names.index(name)] = float(scale)

    scales = [(scales_min[i] + scales_max[i]) / 2 for i in range(len(names))]
    for name_idx, scale in fixed_scales.items():
        scales[name_idx] = scale

    def _real_scales(scales_: List[float]):
        scales_ = list(scales_)
        for i_ in range(len(scales_)):
            if i_ in scale_relative_to:
                scales_[i_] *= scales_[scale_relative_to[i_]]
            if i_ in neg_scales:
                scales_[i_] *= -1
        return scales_

    def _eval_for_scales(scales_: List[float]) -> float:
        real_scales_t = torch.tensor(_real_scales(scales_), dtype=torch.float32).to(dev)  # [Names], float32
        real_scales = rf.convert_to_tensor(real_scales_t, dims=[names_dim])
        hyps_eval_ts = []
        for batch in batches:
            scores_scaled = rf.matmul(batch.scores, real_scales, reduce=names_dim)  # [Batch,Beam], float32
            beam_idx = rf.reduce_argmax(scores_scaled, axis=batch.beam_dim)  # [Batch] -> Beam, int32
            hyps_eval = rf.gather(batch.hyps_eval, indices=beam_idx)  # [Batch], float64
            assert hyps_eval.dims_set == {batch.batch_dim}
            hyps_eval_ts.append(hyps_eval.raw_tensor)
        hyps_eval_t = torch.concatenate(hyps_eval_ts)  # [NumSeqs], float64
        hyps_eval_t = torch.mean(hyps_eval_t)  # scalar, float64
        return hyps_eval_t.cpu().item()

    if search_mode == "single":
        print("Search mode single, each dimension/scale optimized separately")
        for iter_idx in range(args.num_iterations):
            print("*** Iter", iter_idx)
            has_change = False
            for scale_idx in range(len(scales) - 1, -1, -1):
                if scale_idx in fixed_scales:
                    continue
                print(f"** Scale {names[scale_idx]} in range {scales_min[scale_idx]} - {scales_max[scale_idx]}")
                print(
                    "  Other scales:",
                    ", ".join(f"{names[i]}: {scales[i]}" for i in range(len(scales)) if i != scale_idx),
                )
                evals: List[Tuple[float, float]] = []  # (eval, scale)
                for scale in np.linspace(scales_min[scale_idx], scales_max[scale_idx], num=args.num_steps):
                    scales[scale_idx] = scale
                    eval_ = _eval_for_scales(scales)
                    print(f"Scale {names[scale_idx]}: {scale}, {args.evaluation}: {eval_}")
                    evals.append((eval_, scale))
                eval_p = np.percentile([eval_ for eval_, _ in evals], 5)
                print("Eval p5:", eval_p, "best:", min(eval_ for eval_, _ in evals))
                prev_min, prev_max = scales_min[scale_idx], scales_max[scale_idx]
                scales_min[scale_idx] = min(scale for eval_, scale in evals if eval_ <= eval_p * 1.0001)
                scales_max[scale_idx] = max(scale for eval_, scale in evals if eval_ <= eval_p * 1.0001)
                print(f"New {names[scale_idx]} scales min/max:", scales_min[scale_idx], scales_max[scale_idx])
                scales[scale_idx] = (scales_min[scale_idx] + scales_max[scale_idx]) / 2
                if prev_min != scales_min[scale_idx] or prev_max != scales_max[scale_idx]:
                    has_change = True
                else:
                    print(f"No change for scale {names[scale_idx]}")
            if not has_change:
                print(f"No change in this iteration {iter_idx}, stop")
                break
    elif search_mode == "grid":
        print("Search mode grid, all dimensions/scales optimized together")
        for iter_idx in range(args.num_iterations):
            print("*** Iter", iter_idx)
            has_change = False
            scale_indices = []
            spaces = []
            for scale_idx in range(len(names)):
                if scale_idx in fixed_scales:
                    continue
                scale_indices.append(scale_idx)
                spaces.append(np.linspace(scales_min[scale_idx], scales_max[scale_idx], num=args.num_steps))
            evals: List[Tuple[float, List[float]]] = []  # (eval, scales)
            best_eval_so_far = np.inf
            spaces = np.meshgrid(*spaces)
            it = np.nditer(spaces)
            with it:
                for scale_values in it:
                    assert len(scale_values) == len(scale_indices)
                    scales = [0.0] * len(names)
                    for scale_idx, scale in fixed_scales.items():
                        scales[scale_idx] = scale
                    for scale_idx, scale in zip(scale_indices, scale_values):
                        scales[scale_idx] = float(scale)
                    eval_ = _eval_for_scales(scales)
                    if eval_ < best_eval_so_far:
                        best_eval_so_far = eval_
                        print(f"New best {args.evaluation}: {eval_}, scales: {scales}")
                    evals.append((eval_, scales))
            if args.output_grid_plot:
                assert len(scale_indices) == 2, "only implemented for 2 scales"
                _plot_grid(
                    evals,
                    scale_indices=scale_indices,
                    title="",
                    cbar_label=args.evaluation,
                    x_axis_name=names[scale_indices[0]],
                    y_axis_name=names[scale_indices[1]],
                    out_plot_filename=f"{args.output_grid_plot}.{iter_idx}.pdf",
                )
            eval_p = np.percentile([eval_ for eval_, _ in evals], 5)
            print(
                f"Evaluated grid size {len(evals)}, eval p5: {eval_p},",
                f"best: {min(eval_ for eval_, _ in evals)},",
                f"worst: {max(eval_ for eval_, _ in evals)}",
            )
            for scale_idx in scale_indices:
                prev_min, prev_max = scales_min[scale_idx], scales_max[scale_idx]
                scales_min[scale_idx] = min(scale[scale_idx] for eval_, scale in evals if eval_ <= eval_p * 1.0001)
                scales_max[scale_idx] = max(scale[scale_idx] for eval_, scale in evals if eval_ <= eval_p * 1.0001)
                print(f"New {names[scale_idx]} scales min/max:", scales_min[scale_idx], scales_max[scale_idx])
                if prev_min != scales_min[scale_idx] or prev_max != scales_max[scale_idx]:
                    has_change = True
                else:
                    print(f"No change for scale {names[scale_idx]}")
            # Select current best.
            scales = min(evals)[1]
            if not has_change:
                print(f"No change in this iteration {iter_idx}, stop")
                break
    else:
        raise ValueError(f"unknown search mode {search_mode!r}")

    print("Final scales:")
    for name, scale, real_scale in zip(names, scales, _real_scales(scales)):
        print(f"{name}: {scale} (real: {real_scale})")
    eval_ = _eval_for_scales(scales)
    print(f"Final {args.evaluation}: {eval_}")

    if args.output_scales:
        print("Writing scales to", args.output_scales)
        with open(args.output_scales, "w") as f:
            f.write(json.dumps(dict(zip(names, scales))) + "\n")
    if args.output_real_scales:
        print("Writing real scales to", args.output_real_scales)
        with open(args.output_real_scales, "w") as f:
            f.write(json.dumps(dict(zip(names, _real_scales(scales)))) + "\n")


@dataclass
class Batch:
    """batch"""

    seq_tags: List[str]
    batch_dim: Dim  # scalar, int32
    beam_dim: Dim  # [Batch], int32
    hyps_eval: Tensor  # [Batch,Beam], float64
    names_dim: Dim  # scalar, int32
    scores: Tensor  # [Names,Batch,Beam], float32


def _make_padded_tensor_2d(
    lst: List[List[Union[int, float]]], *, dtype: torch.dtype, device: Optional[torch.device] = None
) -> torch.Tensor:
    max_len = max(len(l_) for l_ in lst)
    res = torch.zeros((len(lst), max_len), dtype=dtype)
    for i, l in enumerate(lst):
        for j, v in enumerate(l):
            res[i, j] = v
    return res.to(device)


def _make_padded_tensor_3d(
    lst: List[List[List[Union[int, float]]]], *, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    max_len = max(len(l_) for l_ in lst)
    max_len2 = max(len(l__) for l_ in lst for l__ in l_)
    res = torch.zeros((len(lst), max_len, max_len2), dtype=dtype)
    for i, l in enumerate(lst):
        for j, l2 in enumerate(l):
            for k, v in enumerate(l2):
                res[i, j, k] = v
    return res.to(device)


def _load_text_dict_hyps_file(
    filename: str,
    *,
    name: str,
    vocab: Dict[str, int],
    expect_same_hyps: Optional[Dict[str, List[List[int]]]] = None,
) -> Tuple[Dict[str, List[float]], Dict[str, List[List[int]]]]:
    # See also code in RETURNN TextDictDataset.
    print(f"Loading text dict file {name} from {filename} ...")

    if filename.endswith(".gz"):
        import gzip

        txt = gzip.GzipFile(filename, "rb").read()
    else:
        txt = open(filename, "rb").read()
    data: Dict[str, List[Tuple[float, str]]] = eval(txt)
    assert isinstance(data, dict)
    res_scores = {}
    res_hyps = {}
    if expect_same_hyps is not None:
        assert set(data.keys()) == set(expect_same_hyps.keys())
    for tag, hyps in data.items():
        res_scores[tag], res_hyps[tag] = _hyps_to_indices(
            hyps,
            vocab=vocab,
            expect_same_hyps=expect_same_hyps[tag] if expect_same_hyps else None,
        )
    return res_scores, res_hyps


def _load_text_dict_file(filename: str, *, name: str, vocab: Dict[str, int]) -> Dict[str, List[int]]:
    # See also code in RETURNN TextDictDataset.
    print(f"Loading text dict file {name} from {filename} ...")

    if filename.endswith(".gz"):
        import gzip

        txt = gzip.GzipFile(filename, "rb").read()
    else:
        txt = open(filename, "rb").read()
    data: Dict[str, str] = eval(txt)
    assert isinstance(data, dict)
    res = {}
    for tag, hyps in data.items():
        res[tag] = _hyp_to_indices(hyps, vocab=vocab)
    return res


def _hyps_to_indices(
    hyp: List[Tuple[float, str]],
    *,
    vocab: Dict[str, int],
    expect_same_hyps: Optional[List[List[int]]] = None,
) -> Tuple[List[float], List[List[int]]]:
    assert isinstance(hyp, list)
    if expect_same_hyps is not None:
        assert len(expect_same_hyps) == len(hyp)
    res_scores = []
    res_hyps = []
    for i, (score, hyp_) in enumerate(hyp):
        assert isinstance(score, float) and isinstance(hyp_, str)
        res_scores.append(score)
        seq = _hyp_to_indices(hyp_, vocab=vocab, assert_in_vocab=expect_same_hyps is not None)
        if expect_same_hyps is not None:
            assert seq == expect_same_hyps[i]
        res_hyps.append(seq)
    return res_scores, res_hyps


def _hyp_to_indices(hyp: str, *, vocab: Dict[str, int], assert_in_vocab: bool = False) -> List[int]:
    assert isinstance(hyp, str)
    res = []
    for label in hyp.split():
        if label not in vocab:
            assert not assert_in_vocab, f"unknown label {label}"
            vocab[label] = len(vocab)
        res.append(vocab[label])
    return res


def _report_dev_memory(dev: torch.device):
    import torch
    from returnn.util import basic as rutil

    if dev.type == "cuda":
        stats = [
            f"alloc cur {rutil.human_bytes_size(torch.cuda.memory_allocated(dev))}",
            f"alloc peak {rutil.human_bytes_size(torch.cuda.max_memory_allocated(dev))}",
            f"reserved cur {rutil.human_bytes_size(torch.cuda.memory_reserved(dev))}",
            f"reserved peak {rutil.human_bytes_size(torch.cuda.max_memory_reserved(dev))}",
        ]
        print(f"Memory usage ({dev}):", " ".join(stats))


def _plot_grid(
    evals: List[Tuple[float, List[float]]],
    *,
    scale_indices: List[int],
    title: str,
    cbar_label: str,
    y_axis_name: str,
    x_axis_name: str,
    out_plot_filename: str,
):
    # noinspection PyPackageRequirements
    import matplotlib.pyplot as plt

    # noinspection PyPackageRequirements
    import matplotlib.ticker as ticker

    results = {}  # (x,y) -> z
    for eval_, scales in evals:
        results[tuple([scales[i] for i in scale_indices])] = eval_
    xs = sorted(set(scales[scale_indices[0]] for _, scales in evals))
    ys = sorted(set(scales[scale_indices[1]] for _, scales in evals))

    plt.figure(figsize=(8, 8))

    zs = np.zeros((len(ys), len(xs)))
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            zs[y_idx, x_idx] = results[(x, y)]

    best = np.min(zs.flatten())
    worst_limit = best * 1.3

    ax = plt.subplot(1, 1, 1)
    plt.contourf(xs, ys, zs, levels=np.geomspace(best, worst_limit, 30))

    ax.set_title(title)
    ax.set_ylabel(y_axis_name)
    ax.set_xlabel(x_axis_name)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    cbar = plt.colorbar()
    cbar.set_label(cbar_label)

    print("Saving plot to", out_plot_filename)
    plt.savefig(out_plot_filename)


if __name__ == "__main__":
    main()
