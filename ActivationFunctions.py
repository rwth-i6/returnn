
import theano.tensor as T


def relu(z):
  return (T.sgn(z) + 1) * z * 0.5

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
  'identity': identity,
  'one': constant_one,
  'zero': constant_zero,
  'softsign': softsign,
  'softsquare': softsquare,
  'maxout': maxout,
  'sin': T.sin,
  'cos': T.cos
}


def strtoact(act):
  """
  :param str act: activation function name
  :rtype: theano.Op
  """
  assert ActivationFunctions.has_key(act), "invalid activation function: " + act
  return ActivationFunctions[act]
