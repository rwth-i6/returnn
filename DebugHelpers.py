
"""
This file is going to be imported by Debug.debug_shell() and available as interactive commands.
"""

import sys
import os
import theano
import theano.tensor as tt
import theano.sandbox.cuda as cuda
from TheanoUtil import make_var_tuple
import numpy
import h5py
from Network import LayerNetwork


def find_obj_in_stack(cls, stack=None, all_threads=True):
  if all_threads:
    assert stack is None
    for tid, stack in sys._current_frames().items():
      obj = find_obj_in_stack(cls=cls, stack=stack, all_threads=False)
      if obj is not None:
        return obj
    return None

  assert not all_threads
  if stack is None:
    stack = sys._getframe()
    assert stack, "could not get stack"

  import inspect
  isframe = inspect.isframe

  _tb = stack
  while _tb is not None:
    if isframe(_tb): f = _tb
    else: f = _tb.tb_frame

    for obj in f.f_locals.values():
      if isinstance(obj, cls):
        return obj

    if isframe(_tb): _tb = _tb.f_back
    else: _tb = _tb.tb_next

  return None


_device = None

def get_device():
  """
  :rtype: Device.Device
  """
  global _device
  if _device:
    return _device
  from Device import Device
  _device = find_obj_in_stack(Device)
  return _device


def compute(var, trainnet=True):
  """
  :param theano.Variable var: variable which we should compute the value of
  :param bool trainnet: whether to make givens based on dev.trainnet or dev.testnet
  :return: the computed value
  :rtype: numpy.ndarray
  This expects to calculate some value of the trainnet or testnet of the current Device.
  """
  dev = get_device()
  assert dev, "no Device instance found"
  if trainnet:
    network = dev.trainnet
  else:
    network = dev.testnet
  givens = dev.make_givens(network)
  if isinstance(var, list):
    outputs = var
  else:
    outputs = [var]
  func = theano.function(inputs=[dev.block_start, dev.block_end],
                         outputs=outputs,
                         givens=givens,
                         on_unused_input='warn',
                         name="debug compute")
  batch_dim = dev.y["data"].get_value(borrow=True, return_internal_type=True).shape[1]
  batch_start = 0
  batch_end = batch_dim
  result = func(batch_start, batch_end)
  if not isinstance(var, list):
    result = result[0]
  return result


class DebugNn:
  def __init__(self, filename):
    self.network = LayerNetwork.from_hdf(filename, mask="unity", train_flag=False, eval_flag=True)
    self.f_forwarder = None

  def compile_forwarder(self):
    network = self.network
    data_keys = list(sorted(network.j.keys()))
    # All input seqs expected to have same length.
    givens = [(network.j[k], tt.ones(network.y["data"].shape[:2], dtype="int8")) for k in data_keys]
    self.f_forwarder = theano.function(
      inputs=[network.y["data"]],
      outputs=[network.output["output"].output] + [layer.output for name, layer in sorted(network.output.items()) if name != "output"],
      givens=givens,
      on_unused_input='warn',
      name="forwarder")

  def forward(self, data, output_index=0):
    assert data.ndim == 2
    data = data[:, None, :]  # add batch-dim
    assert self.f_forwarder
    res = self.f_forwarder(data)
    res = make_var_tuple(res)[output_index]
    assert res.ndim == 3
    assert res.shape[1] == 1
    res = res[:, 0]
    return res


class SimpleHdf:
  def __init__(self, filename):
    self.hdf = h5py.File(filename)
    self.seq_tag_to_idx = {name: i for (i, name) in enumerate(self.hdf["seqTags"])}
    self.num_seqs = len(self.hdf["seqTags"])
    assert self.num_seqs == len(self.seq_tag_to_idx), "not unique seq tags"
    seq_lens = self.hdf["seqLengths"]
    if seq_lens.ndim == 2: seq_lens = seq_lens[:, 0]
    assert self.num_seqs == len(seq_lens)
    self.seq_starts = [0] + list(numpy.cumsum(seq_lens))
    total_len = self.seq_starts[-1]
    inputs_len = self.hdf["inputs"].shape[0]
    assert total_len == inputs_len, "time-dim does not match: %i vs %i" % (total_len, inputs_len)
    assert self.seq_starts[-1] == self.hdf["targets/data/classes"].shape[0]

  def get_seq_tags(self):
    return self.hdf["seqTags"]

  def get_data(self, seq_idx):
    seq_t0, seq_t1 = self.seq_starts[seq_idx:seq_idx + 2]
    return self.hdf["inputs"][seq_t0:seq_t1]

  def get_targets(self, seq_idx):
    seq_t0, seq_t1 = self.seq_starts[seq_idx:seq_idx + 2]
    return self.hdf["targets/data/classes"][seq_t0:seq_t1]

  def get_data_dict(self, seq_idx):
    return {"data": self.get_data(seq_idx), "classes": self.get_targets(seq_idx)}
