
import theano.tensor as T


def relu(z):
  # Use fastest implementation.
  # https://github.com/Theano/Theano/issues/2698
  # https://github.com/Lasagne/Lasagne/pull/163#issuecomment-81806482
  return (z + abs(z)) / 2.0

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
