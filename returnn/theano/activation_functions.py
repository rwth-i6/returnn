
import theano.tensor as T
from returnn.theano.util import complex_bound
import numpy


def relu(z):
  # Use fastest implementation.
  # https://github.com/Theano/Theano/issues/2698
  # https://github.com/Lasagne/Lasagne/pull/163#issuecomment-81806482
  return (z + abs(z)) / 2.0

def clipped01lu(z):
  """
  0 for x <= 0
  x for 0 <= x <= 1
  1 for 1 <= x
  """
  # Not sure about the fastest implementation...
  return relu(z) - relu(z - numpy.float32(1))

def clippedlu(z):
  """
  -1 for  x <= -1
   x for -1 <=  x <= 1
   1 for  1 <=  x
  """
  # Not sure about the fastest implementation...
  return relu(z + numpy.float32(1)) - relu(z - numpy.float32(1)) - numpy.float32(1)

def elu(z): # https://arxiv.org/pdf/1511.07289v1.pdf
  return T.switch(T.ge(z,0), z, T.exp(z) - 1)

def identity(z):
  return z

def softsign(z):
  return z / (1.0 + abs(z))

def softsquare(z):
  return 1 / (1.0 + z * z)

def maxout(z):
  return T.max(z, axis=0)

def softmax(z):
  assert z.ndim >= 1
  if z.ndim <= 2:
    return T.nnet.softmax(z)
  else:
    from returnn.theano.util import time_batch_make_flat
    z_flat = time_batch_make_flat(z)
    assert z_flat.ndim == 2
    return T.reshape(T.nnet.softmax(z_flat), z.shape)

def log_softmax(z):
  assert z.ndim >= 1
  if z.ndim <= 2:
    return T.nnet.logsoftmax(z)
  else:
    from returnn.theano.util import time_batch_make_flat
    z_flat = time_batch_make_flat(z)
    assert z_flat.ndim == 2
    return T.reshape(T.nnet.logsoftmax(z_flat), z.shape)

def gauss(z):
  return T.exp(-T.sqr(z))

def cdf(z):
  """Cumulative distribution function via erf (Error function)"""
  return (numpy.float32(1) + T.erf(z)) / numpy.float32(2)

def constant_one():
  return 1

def constant_zero():
  return 0


# from https://github.com/MatthieuCourbariaux/BinaryNet/blob/master/Train-time/binary_net.py
from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise
class Round3(UnaryScalarOp):
  def c_code(self, node, name, x, z, sub):
    x, = x
    z, = z
    return "%(z)s = round(%(x)s);" % locals()

  def grad(self, inputs, gout):
    (gz,) = gout
    return gz,


round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)


def hard_sigmoid(x):
  return T.clip((x + 1.) / 2., 0, 1)


# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1
# during back propagation
def binary_tanh(x):
  return 2. * round3(hard_sigmoid(x)) - 1.


def binary_sigmoid(x):
  return round3(hard_sigmoid(x))


ActivationFunctions = {
  'logistic': T.nnet.sigmoid,
  'sigmoid': T.nnet.sigmoid,  # alias
  'tanh': T.tanh,
  'relu': relu,
  'clipped01lu': clipped01lu,
  'clippedlu': clippedlu,
  'elu': elu,
  'identity': identity,
  'one': constant_one,
  'zero': constant_zero,
  'softsign': softsign,
  'softsquare': softsquare,
  'maxout': maxout,
  'sin': T.sin,
  'cos': T.cos,
  'complex_bound': complex_bound,
  'softmax': softmax,
  'log_softmax': log_softmax,
  'gauss': gauss,
  "erf": T.erf,
  "exp": T.exp,
  "abs": T.abs_,
  "sqr": T.sqr,
  "sqrt": T.sqrt,
  "binary_sigmoid" : binary_sigmoid,
  "binary_tanh" : binary_tanh,
  "cdf": cdf
}


def strtoact(act):
  """
  :type act: str | list[str]
  :param act: activation function name, or multiple such as a list or separated by ":"
  :rtype: theano.Op | list[theano.Op]
  """
  if isinstance(act, (list, tuple)):
    return [strtoact(a) for a in act]
  if ":" in act:
    return [strtoact(a) for a in act.split(":")]
  assert act in ActivationFunctions, "invalid activation function: %s" % act
  return ActivationFunctions[act]

def strtoact_single_joined(act):
  """
  :type act: str | None
  :param act: activation function name, or multiple such as a list or separated by ":"
  :rtype: theano.Op
  """
  if not act:
    return identity
  if ":" in act:
    joined = identity
    for f in [strtoact_single_joined(a) for a in act.split(":")]:
      joined = lambda x: f(joined(x))
    return joined
  assert act in ActivationFunctions, "invalid activation function: %s" % act
  return ActivationFunctions[act]
