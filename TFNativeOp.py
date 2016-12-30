
from __future__ import print_function

import tensorflow as tf
import NativeOp
import TFUtil


class OpMaker(object):
  def __init__(self, gen_base, name=None):
    """
    :param NativeOp.NativeOpGenBase gen_base:
    :param str|None name: e.g. "LstmGenericBase", or automatically via gen_base.__name__
    """
    if not name:
      name = gen_base.__name__
    self.name = name
    self.gen_base = gen_base

  def _make_code(self):
    pass  # TODO...

  def _make_mod(self):
    comp = TFUtil.OpCodeCompiler(
      base_name=self.name, code_version=self.gen_base.code_version,
      code=self._make_code())
    mod = comp.load_module()
    return mod

  def make_op(self):
    mod = self._make_mod()
    return mod.op


def make_lstm_op():
  """
  Demo.
  :return: op
  :rtype: (tf.Tensor) -> tf.Tensor
  """
  maker = OpMaker(NativeOp.LstmGenericBase)
  return maker.make_op()
