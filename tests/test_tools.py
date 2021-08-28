"""
Test cases for different tools
"""

from __future__ import print_function

import tempfile
import _setup_test_env  # noqa
from subprocess import Popen, PIPE, STDOUT, CalledProcessError
import os
import sys


py = sys.executable
print("Python:", py)


def run(*args):
  args = list(args)
  print("run:", args)
  # RETURNN by default outputs on stderr, so just merge both together
  p = Popen(args, stdout=PIPE, stderr=STDOUT)
  out, _ = p.communicate()
  if p.returncode != 0:
    print("Return code is %i" % p.returncode)
    print("std out/err:\n---\n%s\n---\n" % out.decode("utf8"))
    raise CalledProcessError(cmd=args, returncode=p.returncode, output=out)
  return out.decode("utf8")


###############################
# Tests for compile_tf_graph.py
###############################

rec_decoder_config = """
#!rnn.py
network = {
    "enc0": {"class": "linear", "from": "data", "activation": "sigmoid", "n_out": 3},
    "enc1": {"class": "reduce", "mode": "max", "axis": "t", "from": "enc0"},
    "output": {
      "class": "rec", "from": [], "target": "classes",
      "unit": {
        "embed": {"class": "linear", "from": "prev:output", "activation": "sigmoid", "n_out": 3},
        "prob": {"class": "softmax", "from": ["embed", "base:enc1"], "loss": "ce", "target": "classes"},
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


def test_compile_tf_graph_basic():
  tmp_dir = tempfile.mkdtemp()
  with open(os.path.join(tmp_dir, "returnn.config"), "wt") as config:
    config.write(rec_decoder_config)
  args = [
    "tools/compile_tf_graph.py",
    "--output_file",
    os.path.join(tmp_dir, "graph.metatxt"),
    os.path.join(tmp_dir, "returnn.config")
  ]
  run(*args)


def test_compile_tf_graph_recurrent_step():
  tmp_dir = tempfile.mkdtemp()
  with open(os.path.join(tmp_dir, "returnn.config"), "wt") as config:
    config.write(rec_decoder_config)
  args = [
    "tools/compile_tf_graph.py",
    "--output_file",
    os.path.join(tmp_dir, "graph.metatxt"),
    "--rec_step_by_step",
    "output",
    os.path.join(tmp_dir, "returnn.config")
  ]
  run(*args)
