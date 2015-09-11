import unittest
import theano
import theano.tensor as T
import numpy
from Device import have_gpu
from OpLSTMCustom import LSTMCustomOpInstance

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_does_not_crash():
  Z = T.ftensor3('Z')
  X = T.ftensor3('X') #base
  W_re = T.fmatrix('W_re')
  c = T.fmatrix('c') #initial state
  i = T.matrix('i',dtype='int8')
  Y, H, d = LSTMCustomOpInstance(Z, X, c, i, W_re)

  f = theano.function(inputs=[Z, X, c, i, W_re], outputs=[Y])
  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_inp_dim)).astype('float32')
  X_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  Y_val = f(Z_val, X_val, c_val, i_val, W_re_val)
  print Y_val
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_compatible_with_OpLSTM():
  #TODO
  pass

if __name__ == '__main__':
  test_does_not_crash()
  test_compatible_with_OpLSTM()
