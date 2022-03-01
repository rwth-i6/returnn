"""
Test cases for different tools
"""

from __future__ import print_function

import tempfile
import _setup_test_env  # noqa
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import os
import sys


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
    # For the first run, as a special case, directly run the script in the current env.
    # This is easier for debugging.
    from returnn.util.basic import generic_import_module
    mod = generic_import_module(os.path.join(base_dir, args[0]))
    # noinspection PyUnresolvedReferences
    mod.main(args)
    _run_count += 1
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

rec_encoder_decoder_config = """
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
    config.write(rec_encoder_decoder_config)
  args = [
    "tools/compile_tf_graph.py",
    "--output_file",
    os.path.join(tmp_dir, "graph.metatxt"),
    os.path.join(tmp_dir, "returnn.config")
  ]
  run(*args)


def test_compile_tf_graph_basic_second_run():
  # Just to make sure that the second run works as well,
  # which behaves different due to the debug case of the first run.
  # See :func:`run` above.
  test_compile_tf_graph_basic()


def test_compile_tf_graph_enc_dec_recurrent_step():
  tmp_dir = tempfile.mkdtemp()
  with open(os.path.join(tmp_dir, "returnn.config"), "wt") as config:
    config.write(rec_encoder_decoder_config)
  args = [
    "tools/compile_tf_graph.py",
    "--output_file",
    os.path.join(tmp_dir, "graph.metatxt"),
    "--rec_step_by_step",
    "output",
    os.path.join(tmp_dir, "returnn.config")
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
    os.path.join(tmp_dir, "returnn.config")
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
    os.path.join(tmp_dir, "returnn.config")
  ]
  run(*args)
