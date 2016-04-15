
"""
Generic interface which automatically creates:
* CPU and GPU op
* inplace and not inplace
* grad variants
"""

import numpy
import theano
import theano.sandbox.cuda
import theano.tensor as T
from theano.gof.opt import OpSub
from theano.compile import optdb
from theano import gof


class GenericOp(theano.Op):
  inplace = False


class GpuGenericOp(GenericOp, theano.sandbox.cuda.GpuOp):
  pass


class LstmG:
  def code(self):
    pass

