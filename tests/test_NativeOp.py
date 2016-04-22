

import sys
import numpy
import theano
import theano.tensor as T
import theano.scan_module.scan_op
from nose.tools import assert_equal, assert_is, assert_is_instance
import theano.printing
from pprint import pprint
from GeneratingDataset import Task12AXDataset
from Updater import Updater
from Device import Device
from Util import NumbersDict
from Config import Config
from NetworkHiddenLayer import DumpLayer
import rnn
import EngineUtil
import Network
import better_exchook
from Log import log

better_exchook.replace_traceback_format_tb()
log.initialize()  # some code needs it

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
  assign_success, _ = EngineUtil.assign_dev_data(device=dev, dataset=dataset, batches=batches)
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

  net = Network.LayerNetwork.from_json(
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
  assign_success, _ = EngineUtil.assign_dev_data(device=dev, dataset=dataset, batches=batches)
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
  print "keys in res1:", sorted(res1.keys())
  print "keys in res2:", sorted(res2.keys())
  for key in sorted(res1.keys()):
    if key not in res2:
      print "ERROR: %r not in res2" % key
      fail = True
    v1 = res1[key]
    v2 = res2[key]
    v1 = numpy.asarray(v1)
    v2 = numpy.asarray(v2)
    if v1.shape != v2.shape:
      print "shape does not match for %r" % key
      print "v1 shape:", v1.shape
      print "v2 shape:", v2.shape
      fail = True
    elif not numpy.allclose(v1, v2, atol=atol):
      print "not equal: %r" % key
      c = 0
      for idx in zip(*numpy.unravel_index(range(numpy.prod(v1.shape)), v1.shape)):
        e1 = v1[idx]
        e2 = v2[idx]
        if not numpy.isclose(e1, e2, atol=atol):
          print "idx %r differs: %r vs %r" % (idx, e1, e2)
          c += 1
          if c >= 10: break
      fail = True
  for key in sorted(res2.keys()):
    if key not in res1:
      print "ERROR: %r not in res1" % key
      fail = True
  assert not fail


def test_native_lstm():
  compare_lstm({"class": "native_lstm"})


