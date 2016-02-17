
"""
This file is going to be imported by Debug.debug_shell() and available as interactive commands.
"""

import sys
import os
import theano
import theano.tensor as tt
import theano.sandbox.cuda as cuda
import numpy


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
