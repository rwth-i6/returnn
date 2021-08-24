
from __future__ import print_function

import sys
import os

import _setup_test_env  # noqa
import sys
import numpy
import theano
import theano.tensor as T
import theano.scan_module.scan_op
import unittest
from nose.tools import assert_equal, assert_is, assert_is_instance
from numpy.testing.utils import assert_almost_equal, assert_allclose
import theano.printing
from pprint import pprint
from returnn.datasets.generating import Task12AXDataset
from returnn.theano.updater import Updater
from returnn.util.basic import have_gpu
from returnn.util.basic import NumbersDict
from returnn.config import Config
from returnn.theano.layers.hidden import DumpLayer
import returnn.__main__ as rnn
import returnn.theano.engine_util as engine_util
import returnn.theano.util as theano_util
import returnn.theano.network as network
from returnn.util import better_exchook

theano_util.monkey_patches()

# Some code uses get_global_config().
# Not sure about the most clean solution.
rnn.config = Config()


class DummyDevice:
  """
  Behave like Device.
  Only needed for assign_dev_data.
  """
  blocking = True
  used_data_keys = ("data", "classes")
  targets = None
  output_index = None

  def __init__(self):
    self.num_frames = NumbersDict(0)
    self.y = {}
    self.j = {}

  def alloc_data(self, shapes, max_ctc_length=0):
    """
    :param dict[str,list[int]] shapes: by data-key. format usually (time,batch,features)
    :type max_ctc_length: int
    """
    assert all([s > 0 for s in shapes["data"]])
    # For output_shape, we allow zeros, because e.g. in forwarding, we don't know them and will not use it.
    self.targets = {k: numpy.full(shapes[k], -1, dtype=theano.config.floatX) for k in self.used_data_keys}
    self.ctc_targets = numpy.zeros((shapes.get('classes', [0, 0])[1], max_ctc_length), dtype=theano.config.floatX)
    self.output_index = {k: numpy.zeros(shapes[k][0:2], dtype='int8') for k in self.used_data_keys}
    self.tags = [None] * shapes["data"][1]  # seq-name for each batch slice

  def initialize(self, net):
    self.y = {k: theano.shared(numpy.zeros((1,) * net.y[k].ndim, dtype=net.y[k].dtype),
                               borrow=True, name='y_%s' % k)
              for k in self.used_data_keys}
    self.j = {k: theano.shared(numpy.zeros((1, 1), dtype='int8'), borrow=True, name='j_%s' % k)
              for k in self.used_data_keys}

  def update_data(self):
    for target in self.used_data_keys:
      self.y[target].set_value(self.targets[target].astype(self.y[target].dtype), borrow = True)
    for k in self.used_data_keys:
      self.j[k].set_value(self.output_index[k], borrow = True)


def test_DummyDevice():
  dataset = Task12AXDataset(num_seqs=1000, seq_ordering="random", chunking="200:200")
  dataset.init_seq_order(epoch=1)
  batch_gen = dataset.generate_batches(recurrent_net=True, batch_size=1000, max_seqs=3)
  batches = batch_gen.peek_next_n(1)
  dev = DummyDevice()
  assign_success, _ = engine_util.assign_dev_data(device=dev, dataset=dataset, batches=batches)
  assert assign_success


def load(lstm_opts=None):
  if not lstm_opts: lstm_opts = {"class": "lstm2"}
  lstm_opts = lstm_opts.copy()
  lstm_opts.update({"n_out": 10, "from": "in"})
  num_inputs = 9
  num_outputs = 2
  net_topo = {
    "in": {"class": "dump", "filename": "in"},
    "lstm": lstm_opts,
    "lstm_dump": {"class": "dump", "from": "lstm", "filename": "lstm"},
    "output": {"class": "softmax", "loss": "ce", "from": "lstm_dump"}
  }

  collected_data = {}
  DumpLayer.global_debug_container = collected_data

  net = network.LayerNetwork.from_json(
    json_content=net_topo,
    n_in=num_inputs,
    n_out={"classes": (num_outputs, 1)},
    train_flag=True
  )
  net.declare_train_params()

  # Init dataset and prepare one minibatch.
  epoch = 1
  dataset = Task12AXDataset(num_seqs=1000, seq_ordering="random", chunking="200:200")
  dataset.init_seq_order(epoch=epoch)
  batch_gen = dataset.generate_batches(
    recurrent_net=net.recurrent,
    batch_size=5000,
    max_seqs=10)
  batches = batch_gen.peek_next_n(1)
  # We need the DummyDevice for assign_dev_data.
  dev = DummyDevice()
  assign_success, _ = engine_util.assign_dev_data(device=dev, dataset=dataset, batches=batches)
  assert assign_success
  dev.initialize(net)
  dev.update_data()
  givens = [(net.y[k], dev.y[k]) for k in dev.used_data_keys]
  givens += [(net.j[k], dev.j[k]) for k in dev.used_data_keys]

  # Now gradients, updates and compile everything.
  gradients = {p: T.grad(net.get_objective(), p, known_grads=net.known_grads)
               for p in net.train_params_vars}
  updater = Updater(adam=True)
  updater.initVars(net, gradients)
  updater.setLearningRate(learning_rate=0.01)
  trainer = theano.function(
    inputs=[],
    outputs=[net.total_cost],
    givens=givens,
    updates=updater.getUpdateList(),
    on_unused_input='warn',
    name="train_and_updater")

  for p in net.train_params_vars:
    collected_data["param:%s" % p.name] = p.get_value()

  # And finally, run it.
  cost = trainer()
  collected_data["cost"] = cost
  return collected_data


def test_load():
  load()


atol = 1e-7


def compare_lstm(lstm_opts=None):
  res1 = load()
  res2 = load(lstm_opts=lstm_opts)
  fail = False
  print("keys in res1:", sorted(res1.keys()))
  print("keys in res2:", sorted(res2.keys()))
  for key in sorted(res1.keys()):
    if key not in res2:
      print("ERROR: %r not in res2" % key)
      fail = True
    v1 = res1[key]
    v2 = res2[key]
    v1 = numpy.asarray(v1)
    v2 = numpy.asarray(v2)
    if v1.shape != v2.shape:
      print("shape does not match for %r" % key)
      print("v1 shape:", v1.shape)
      print("v2 shape:", v2.shape)
      fail = True
    elif not numpy.allclose(v1, v2, atol=atol):
      print("not equal: %r" % key)
      c = 0
      for idx in zip(*numpy.unravel_index(range(numpy.prod(v1.shape)), v1.shape)):
        e1 = v1[idx]
        e2 = v2[idx]
        if not numpy.isclose(e1, e2, atol=atol):
          print("idx %r differs: %r vs %r" % (idx, e1, e2))
          c += 1
          if c >= 10: break
      fail = True
  for key in sorted(res2.keys()):
    if key not in res1:
      print("ERROR: %r not in res1" % key)
      fail = True
  assert not fail


def test_native_lstm():
  compare_lstm({"class": "native_lstm"})


@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_fast_bw():
  print("Make op...")
  from returnn.native_op import FastBaumWelchOp
  op = FastBaumWelchOp().make_theano_op()  # (am_scores, edges, weights, start_end_states, float_idx, state_buffer)
  print("Op:", op)
  n_batch = 3
  seq_len = 5
  n_classes = 5
  from returnn.util.fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  fsa.add_inf_loop(state_idx=0, num_emission_labels=n_classes)
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = fast_bw_fsa.edges.view("float32")
  edges_placeholder = T.fmatrix(name="edges")
  weights = fast_bw_fsa.weights
  weights_placeholder = T.fvector(name="weights")
  start_end_states = fast_bw_fsa.start_end_states.view("float32")
  start_end_states_placeholder = T.fmatrix(name="start_end_states")
  am_scores = numpy.random.normal(size=(seq_len, n_batch, n_classes)).astype("float32")  # in -log space
  am_scores_placeholder = T.ftensor3(name="am_scores")
  float_idx = numpy.ones((seq_len, n_batch), dtype="float32")
  float_idx_placeholder = T.fmatrix(name="float_idx")
  last_state_idx = numpy.max(fast_bw_fsa.start_end_states[1])  # see get_automata_for_batch
  state_buffer = numpy.zeros((2, last_state_idx + 1), dtype="float32")
  state_buffer_placeholder = T.fmatrix(name="state_buffer")
  print("Construct call...")
  fwdbwd, obs_scores = op(
    am_scores_placeholder, edges_placeholder, weights_placeholder, start_end_states_placeholder, float_idx_placeholder, state_buffer_placeholder)
  f = theano.function(inputs=[am_scores_placeholder, edges_placeholder, weights_placeholder, start_end_states_placeholder, float_idx_placeholder, state_buffer_placeholder], outputs=[fwdbwd, obs_scores])
  print("Done.")
  print("Eval:")
  _, score = f(am_scores, edges, weights, start_end_states, float_idx, state_buffer)
  print("score:", score)


@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_fast_bw_uniform():
  print("Make op...")
  from returnn.native_op import FastBaumWelchOp
  op = FastBaumWelchOp().make_theano_op()  # (am_scores, edges, weights, start_end_states, float_idx, state_buffer)
  print("Op:", op)
  n_batch = 3
  seq_len = 7
  n_classes = 5
  from returnn.util.fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  for i in range(n_classes):
    fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
    fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
  assert n_classes <= seq_len
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  print("edges:")
  print(fast_bw_fsa.edges)
  edges = fast_bw_fsa.edges.view("float32")
  edges_placeholder = T.fmatrix(name="edges")
  weights = fast_bw_fsa.weights
  weights_placeholder = T.fvector(name="weights")
  print("start_end_states:")
  print(fast_bw_fsa.start_end_states)
  start_end_states = fast_bw_fsa.start_end_states.view("float32")
  start_end_states_placeholder = T.fmatrix(name="start_end_states")
  am_scores = numpy.ones((seq_len, n_batch, n_classes), dtype="float32") * numpy.float32(1.0 / n_classes)
  am_scores = -numpy.log(am_scores)  # in -log space
  am_scores_placeholder = T.ftensor3(name="am_scores")
  float_idx = numpy.ones((seq_len, n_batch), dtype="float32")
  float_idx_placeholder = T.fmatrix(name="float_idx")
  last_state_idx = numpy.max(fast_bw_fsa.start_end_states[1])  # see get_automata_for_batch
  state_buffer = numpy.zeros((2, last_state_idx + 1), dtype="float32")
  state_buffer_placeholder = T.fmatrix(name="state_buffer")
  print("Construct call...")
  fwdbwd, obs_scores = op(
    am_scores_placeholder, edges_placeholder, weights_placeholder, start_end_states_placeholder, float_idx_placeholder, state_buffer_placeholder)
  f = theano.function(inputs=[am_scores_placeholder, edges_placeholder, weights_placeholder, start_end_states_placeholder, float_idx_placeholder, state_buffer_placeholder], outputs=[fwdbwd, obs_scores])
  print("Done.")
  print("Eval:")
  fwdbwd, score = f(am_scores, edges, weights, start_end_states, float_idx, state_buffer)
  print("score:")
  print(repr(score))
  assert_equal(score.shape, (seq_len, n_batch))
  bw = numpy.exp(-fwdbwd)
  print("Baum-Welch soft alignment:")
  print(repr(bw))
  assert_equal(bw.shape, (seq_len, n_batch, n_classes))
  from numpy import array, float32
  if seq_len == n_classes:
    print("Extra check identity...")
    for i in range(n_batch):
      assert_almost_equal(numpy.identity(n_classes), bw[:, i])
  if seq_len == 7 and n_classes == 5:
    print("Extra check ref_align (7,5)...")
    assert_allclose(score, 8.55801582, rtol=1e-5)  # should be the same everywhere
    ref_align = \
      array([[[1., 0., 0., 0., 0.]],
             [[0.33333316, 0.66666663, 0., 0., 0.]],
             [[0.06666669, 0.53333354, 0.40000018, 0., 0.]],
             [[0., 0.20000014, 0.60000014, 0.19999999, 0.]],
             [[0., 0., 0.39999962, 0.53333312, 0.06666663]],
             [[0., 0., 0., 0.66666633, 0.33333316]],
             [[0., 0., 0., 0., 0.99999982]]], dtype=float32)
    assert_equal(ref_align.shape, (seq_len, 1, n_classes))
    ref_align = numpy.tile(ref_align, (1, n_batch, 1))
    assert_equal(ref_align.shape, bw.shape)
    # print("Reference alignment:")
    # print(repr(ref_align))
    print("mean square diff:", numpy.mean(numpy.square(ref_align - bw)))
    print("max square diff:", numpy.max(numpy.square(ref_align - bw)))
    assert_allclose(ref_align, bw, rtol=1e-5)
  print("Done.")


if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          try:
            v()
          except unittest.SkipTest as exc:
            print("SkipTest:", exc)
          print("-" * 40)
      print("Finished all tests.")
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    import threading
    #if len(list(threading.enumerate())) > 1:
    #  print("Warning, more than one thread at exit:")
    #  better_exchook.dump_all_thread_tracebacks()

