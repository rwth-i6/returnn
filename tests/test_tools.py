"""
Test cases for different tools
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import tempfile
import _setup_test_env  # noqa
from subprocess import Popen, PIPE, STDOUT, CalledProcessError, check_call
import json
import os
import sys
import unittest


my_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(my_dir)
py = sys.executable
print("Python:", py)
_run_count = 0


def run(*args):
    args = list(args)
    print("run:", args)
    global _run_count
    if _run_count == 0:
        _run_count += 1
        # For the first run, as a special case, directly run the script in the current env.
        # This is easier for debugging.
        from returnn.util.basic import generic_import_module

        mod = generic_import_module(os.path.join(base_dir, args[0]))
        # noinspection PyUnresolvedReferences
        mod.main(args)
        return
    _run_count += 1
    # RETURNN by default outputs on stderr, so just merge both together
    p = Popen(args, stdout=PIPE, stderr=STDOUT, cwd=base_dir)
    out, _ = p.communicate()
    if p.returncode != 0:
        print("Return code is %i" % p.returncode)
        print("std out/err:\n---\n%s\n---\n" % out.decode("utf8"))
        raise CalledProcessError(cmd=args, returncode=p.returncode, output=out)
    return out.decode("utf8")


###############################
# Tests for compile_tf_graph.py
###############################

rec_encoder_decoder_simple_config = """
#!rnn.py
network = {
    "enc0": {"class": "linear", "from": "data", "activation": "sigmoid", "n_out": 3},
    "enc1": {"class": "reduce", "mode": "max", "axis": "t", "from": "enc0"},
    "output": {
      "class": "rec", "from": [], "target": "classes",
      "unit": {
        "embed": {"class": "linear", "from": "prev:output", "activation": "sigmoid", "n_out": 3},
        "s": {"class": "rec", "unit": "lstm", "from": ["embed", "base:enc1"], "n_out": 3},
        "prob": {"class": "softmax", "from": "s", "loss": "ce", "target": "classes"},
        "output": {"class": "choice", "beam_size": 4, "from": "prob", "target": "classes", "initial_output": 0},
        "end": {"class": "compare", "from": "output", "value": 0}
      }
    },
    "decision": {"class": "decide", "from": "output", "loss": "edit_distance"}
}
num_inputs = 5
num_outputs = 3
use_tensorflow = True
"""


rec_encoder_decoder_att_config = """
#!rnn.py
network = {
    "encoder": {"class": "linear", "from": "data", "activation": "sigmoid", "n_out": 3},
    "enc_ctx": {"class": "linear", "from": "encoder", "n_out": 10},
    "inv_fertility": {"class": "linear", "activation": "sigmoid", "with_bias": False, "from": "encoder", "n_out": 1},
    "output": {
      "class": "rec", "from": [], "target": "classes",
      "unit": {
        "weight_feedback": {"class": "linear", "activation": None, "with_bias": False, "from": "prev:accum_att_weights", "n_out": 10},
        "prev_s_transformed": {"class": "linear", "activation": None, "with_bias": False, "from": "prev:s", "n_out": 10},
        "energy_in": {"class": "combine", "kind": "add", "from": ["base:enc_ctx", "weight_feedback", "prev_s_transformed"], "n_out": 10},
        "energy_tanh": {"class": "activation", "activation": "tanh", "from": "energy_in"},
        "energy": {"class": "linear", "activation": None, "with_bias": False, "from": "energy_tanh", "n_out": 1},
        "att_weights": {"class": "softmax_over_spatial", "from": "energy"},
        "accum_att_weights": {
          "class": "eval", "from": ["prev:accum_att_weights", "att_weights", "base:inv_fertility"],
          "eval": "source(0) + source(1) * source(2) * 0.5",
          "out_type": {"dim": 1, "shape": (None, 1)}},
        "att": {"class": "generic_attention", "weights": "att_weights", "base": "base:encoder", "auto_squeeze": True},

        "s": {"class": "rec", "unit": "lstm", "from": ["att", "prev:embed"], "n_out": 3},
        "prob": {"class": "softmax", "from": "s", "loss": "ce", "target": "classes"},
        "output": {"class": "choice", "beam_size": 4, "from": "prob", "target": "classes", "initial_output": 0},
        "embed": {"class": "linear", "from": "prev:output", "activation": "sigmoid", "n_out": 3},
        "end": {"class": "compare", "from": "output", "value": 0}
      }
    },
    "decision": {"class": "decide", "from": "output", "loss": "edit_distance"}
}
num_inputs = 5
num_outputs = 3
use_tensorflow = True
"""


rec_transducer_time_sync_config = """
#!rnn.py
network = {
    "encoder": {"class": "linear", "from": "data", "activation": "sigmoid", "n_out": 3},
    "output": {
      "class": "rec", "from": "encoder", "target": "classes",
      "unit": {
        "embed": {"class": "linear", "from": "prev:output", "activation": "sigmoid", "n_out": 3},
        "s": {"class": "rec", "unit": "lstm", "from": ["embed", "data:source"], "n_out": 3},
        "prob": {"class": "softmax", "from": "s", "loss": "ce", "target": "classes"},
        "output": {"class": "choice", "beam_size": 4, "from": "prob", "target": "classes", "initial_output": 0},
      }
    },
}
num_inputs = 5
num_outputs = 3
use_tensorflow = True
"""


rec_transducer_time_sync_delayed_config = """
#!rnn.py
network = {
    "encoder": {"class": "linear", "from": "data", "activation": "sigmoid", "n_out": 3},
    "output": {
      "class": "rec", "from": "encoder", "target": "classes",
      "unit": {
        "s": {"class": "rec", "unit": "lstm", "from": ["prev:embed", "prev:s2", "data:source"], "n_out": 3},
        "prob": {"class": "softmax", "from": "s", "loss": "ce", "target": "classes"},
        "output": {"class": "choice", "beam_size": 4, "from": "prob", "target": "classes", "initial_output": 0},
        "embed": {"class": "linear", "from": "output", "activation": "sigmoid", "n_out": 3},
        "s2": {"class": "rec", "unit": "lstm", "from": ["embed", "prev:s"], "n_out": 3},
      }
    },
}
num_inputs = 5
num_outputs = 3
use_tensorflow = True
"""


def test_compile_tf_graph_basic():
    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "returnn.config"), "wt") as config:
        config.write(rec_encoder_decoder_simple_config)
    args = [
        "tools/compile_tf_graph.py",
        "--output_file",
        os.path.join(tmp_dir, "graph.metatxt"),
        os.path.join(tmp_dir, "returnn.config"),
    ]
    run(*args)


def test_compile_tf_graph_basic_second_run():
    # Just to make sure that the second run works as well,
    # which behaves different due to the debug case of the first run.
    # See :func:`run` above.
    test_compile_tf_graph_basic()


def test_compile_tf_graph_enc_dec_simple_recurrent_step():
    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "returnn.config"), "wt") as config:
        config.write(rec_encoder_decoder_simple_config)
    args = [
        "tools/compile_tf_graph.py",
        "--output_file",
        os.path.join(tmp_dir, "graph.metatxt"),
        "--rec_step_by_step",
        "output",
        os.path.join(tmp_dir, "returnn.config"),
    ]
    run(*args)


def test_compile_tf_graph_enc_dec_att_recurrent_step():
    # https://github.com/rwth-i6/returnn/issues/1016
    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "returnn.config"), "wt") as config:
        config.write(rec_encoder_decoder_att_config)
    args = [
        "tools/compile_tf_graph.py",
        "--output_file",
        os.path.join(tmp_dir, "graph.metatxt"),
        "--rec_step_by_step",
        "output",
        os.path.join(tmp_dir, "returnn.config"),
    ]
    run(*args)


def test_compile_tf_graph_transducer_time_sync_recurrent_step():
    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "returnn.config"), "wt") as config:
        config.write(rec_transducer_time_sync_config)
    args = [
        "tools/compile_tf_graph.py",
        "--output_file",
        os.path.join(tmp_dir, "graph.metatxt"),
        "--rec_step_by_step",
        "output",
        os.path.join(tmp_dir, "returnn.config"),
    ]
    run(*args)


def test_compile_tf_graph_transducer_time_sync_delayed_recurrent_step():
    tmp_dir = tempfile.mkdtemp()
    with open(os.path.join(tmp_dir, "returnn.config"), "wt") as config:
        config.write(rec_transducer_time_sync_delayed_config)
    args = [
        "tools/compile_tf_graph.py",
        "--output_file",
        os.path.join(tmp_dir, "graph.metatxt"),
        "--rec_step_by_step",
        "output",
        os.path.join(tmp_dir, "returnn.config"),
    ]
    run(*args)


#################################
# Tests for torch_scale_tuning.py
#################################


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch not available")


def _torch_scale_tuning_grid_search(
    eval_for_scales: Callable[[List[float]], float], *, keep_best: bool
) -> Tuple[List[float], List[float]]:
    _require_torch()
    from returnn.util.basic import generic_import_module

    mod = generic_import_module(os.path.join(base_dir, "tools/torch_scale_tuning.py"))
    evals = []

    def _eval(scales: List[float]) -> float:
        evals.append(eval_for_scales(scales))
        return evals[-1]

    # noinspection PyProtectedMember
    scales = mod._grid_search(
        _eval,
        scales=[1.0, 1.0, 1.0],
        names=["fixed", "a", "b"],
        fixed_scales={0: 1.0},
        scales_min=[0.0, 0.0, 0.0],
        scales_max=[2.0, 2.0, 2.0],
        num_iterations=10,
        num_steps=5,
        evaluation="eval",
        output_grid_plot=None,
        keep_best=keep_best,
    )
    return scales, evals


def test_torch_scale_tuning_grid_search_keep_best():
    def _eval(scales: List[float]) -> float:
        return {(1.0, 1.0): 0.5, (0.5, 0.5): 1.0, (2.0, 2.0): 1.0}.get((scales[1], scales[2]), 1.1)

    # (1, 1) is only on the first grid. The second grid over [0.5, 2] misses it, and its range stays, so it stops.
    scales, evals = _torch_scale_tuning_grid_search(_eval, keep_best=False)
    assert len(evals) == 2 * 5 * 5
    assert scales == [1.0, 0.5, 0.5]  # best of the last grid, tie with (2, 2)
    scales, evals = _torch_scale_tuning_grid_search(_eval, keep_best=True)
    assert len(evals) == 2 * 5 * 5
    assert scales == [1.0, 1.0, 1.0]
    assert _eval(scales) == min(evals) == 0.5


def test_torch_scale_tuning_grid_search_last_grid_improves():
    def _eval(scales: List[float]) -> float:
        return round(abs(scales[1] - 1.3) + abs(scales[2] - 0.7), 2)

    for keep_best in [False, True]:
        scales, evals = _torch_scale_tuning_grid_search(_eval, keep_best=keep_best)
        assert _eval(scales) == min(evals) < min(evals[: 5 * 5])


def test_torch_scale_tuning_cli_grid_keep_best():
    _require_torch()
    names = ["am", "lm", "prior"]
    hyps = {
        "seq0": ["x x x", "x x c", "a b c"],
        "seq1": ["x b c", "a b c", "x x c"],
        "seq2": ["x x c", "x b c", "x x x"],
    }
    scores = {  # per hyp: am, lm, prior
        "seq0": [(-1.9, -2.7, -3.8), (-5.4, -4.5, -4.4), (-3.5, -0.0, -3.8)],
        "seq1": [(-0.8, -4.7, -5.4), (-3.4, -2.4, -3.3), (-5.0, -0.8, -1.3)],
        "seq2": [(-3.1, -3.8, -2.7), (-2.2, -2.1, -2.3), (-3.3, -3.2, -4.6)],
    }
    tmp_dir = tempfile.mkdtemp()
    score_filenames = []
    for i, name in enumerate(names):
        score_filenames.append(os.path.join(tmp_dir, f"{name}.py"))
        with open(score_filenames[-1], "wt") as f:
            f.write(repr({tag: [(s[i], h) for s, h in zip(scores[tag], hyps[tag])] for tag in hyps}))
    ref_filename = os.path.join(tmp_dir, "ref.py")
    with open(ref_filename, "wt") as f:
        f.write(repr({tag: "a b c" for tag in hyps}))

    def _run(*extra_args: str) -> Tuple[Dict[str, float], Dict[str, float]]:
        out_scales = os.path.join(tmp_dir, "scales.txt")
        out_real_scales = os.path.join(tmp_dir, "real_scales.txt")
        check_call(
            [py, "tools/torch_scale_tuning.py", "--names", *names, "--scores", *score_filenames]
            + ["--evaluation", "edit_distance", "--ref", ref_filename, "--fixed-scales", "am", "1"]
            + ["--negative-scales", "prior", "--scale-relative-to", "prior", "lm", "--num-steps", "7"]
            + ["--output-scales", out_scales, "--output-real-scales", out_real_scales, *extra_args],
            cwd=base_dir,
        )
        with open(out_scales) as f_scales, open(out_real_scales) as f_real_scales:
            return json.load(f_scales), json.load(f_real_scales)

    last_grid = {"am": 1.0, "lm": 2 / 3, "prior": 0.0}  # edit distance 2/9
    best = {"am": 1.0, "lm": 5 / 3, "prior": 1 / 3}  # edit distance 1/9, evaluated in an earlier grid
    for extra_args, expected in [
        ((), last_grid),
        (("--behavior-version", "36"), best),
        (("--behavior-version", "36", "--no-grid-keep-best"), last_grid),
        (("--grid-keep-best",), best),
    ]:
        scales, real_scales = _run(*extra_args)
        assert scales.keys() == real_scales.keys() == expected.keys()
        for name in names:
            assert abs(scales[name] - expected[name]) < 1e-6, (extra_args, scales)
        assert real_scales["am"] == scales["am"] and real_scales["lm"] == scales["lm"]
        assert abs(real_scales["prior"] + scales["lm"] * scales["prior"]) < 1e-6, (extra_args, real_scales)
