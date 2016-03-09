
import theano.tensor as T
from TheanoUtil import complex_bound


def relu(z):
  # Use fastest implementation.
  # https://github.com/Theano/Theano/issues/2698
  # https://github.com/Lasagne/Lasagne/pull/163#issuecomment-81806482
  return (z + abs(z)) / 2.0


def elu(z): # http://arxiv.org/pdf/1511.07289v1.pdf
  return T.switch(T.ge(z,0), z, T.exp(z) - 1)

def identity(z):
  return z

def softsign(z):
  return z / (1.0 + abs(z))

def softsquare(z):
  return 1 / (1.0 + z * z)

def maxout(z):
  return T.max(z, axis=0)

def constant_one():
  return 1

def constant_zero():
  return 0


ActivationFunctions = {
  'logistic': T.nnet.sigmoid,
  'sigmoid': T.nnet.sigmoid,  # alias
  'tanh': T.tanh,
  'relu': relu,
  'elu': elu,
  'identity': identity,
  'one': constant_one,
  'zero': constant_zero,
  'softsign': softsign,
  'softsquare': softsquare,
  'maxout': maxout,
  'sin': T.sin,
  'cos': T.cos,
  'complex_bound': complex_bound
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
  assert ActivationFunctions.has_key(act), "invalid activation function: %s" % act
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
  assert ActivationFunctions.has_key(act), "invalid activation function: %s" % act
  return ActivationFunctions[act]
