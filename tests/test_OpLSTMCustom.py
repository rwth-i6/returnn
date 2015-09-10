import unittest
import theano
import theano.tensor as T
import numpy
from Device import have_gpu
from OpLSTMCustom import LSTMCustomOpInstance
from OpLSTM import LSTMOpInstance

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_does_not_crash():
  Z = T.ftensor3('Z')
  B = T.ftensor3('B') #base
  W_re = T.fmatrix('W_re')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')
  Y, H, d = LSTMCustomOpInstance(Z, B, c, y0, i, W_re)

  f = theano.function(inputs=[Z, B, c, y0, i, W_re], outputs=Y)

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  Y_val = numpy.asarray(f(Z_val, B_val, c_val, y0_val, i_val, W_re_val))
  #print Y_val
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_fwd_pass_compatible_with_OpLSTM():
  Z = T.ftensor3('Z')
  B = T.ftensor3('B') #base
  W_re = T.fmatrix('W_re')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')

  Y, H, d = LSTMCustomOpInstance(Z, B, c, y0, i, W_re)
  Z_modified = T.inc_subtensor(Z[0], T.dot(y0,W_re))
  Y2, H2, d2 = LSTMOpInstance(Z_modified, W_re, c, i)

  f = theano.function(inputs=[Z, B, c, y0, i, W_re], outputs=Y)
  g = theano.function(inputs=[Z, W_re, c, y0, i], outputs=Y2)

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  Y_val = numpy.asarray(f(Z_val, B_val, c_val, y0_val, i_val, W_re_val))
  Y2_val = numpy.asarray(g(Z_val, W_re_val, c_val, y0_val, i_val))
  assert numpy.allclose(Y_val, Y2_val), (Y_val, Y2_val)
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_bwd_pass_compatible_with_OpLSTM():
  Z = T.ftensor3('Z')
  B = T.ftensor3('B') #base
  W_re = T.fmatrix('W_re')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')
  Y, H, d = LSTMCustomOpInstance(Z, B, c, y0, i, W_re)
  Z_modified = T.inc_subtensor(Z[0], T.dot(y0,W_re))
  Y2, H2, d2 = LSTMOpInstance(Z_modified, W_re, c, i)

  cost = Y.sum()
  DZ = T.grad(cost, Z)
  DW_re = T.grad(cost, W_re)
  cost2 = Y2.sum()
  DZ2 = T.grad(cost2, Z)
  DW_re2 = T.grad(cost2, W_re)

  f = theano.function(inputs=[Z, B, c, y0, i, W_re], outputs=[DZ, DW_re])
  g = theano.function(inputs=[Z, W_re, c, y0, i], outputs=[DZ2, DW_re2])

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  vals = f(Z_val, B_val, c_val, y0_val, i_val, W_re_val)
  DZ_val, DW_re_val = (numpy.asarray(vals[0]), numpy.asarray(vals[1]))
  vals2 = g(Z_val, W_re_val, c_val, y0_val, i_val)
  DZ2_val, DW_re2_val = (numpy.asarray(vals2[0]), numpy.asarray(vals2[1]))
  assert numpy.allclose(DZ_val, DZ2_val), (DZ_val, DZ2_val)
  assert numpy.allclose(DW_re_val, DW_re2_val), (DW_re_val, DW_re2_val)
  print "success"

if __name__ == '__main__':
  print "calling test_does_not_crash()"
  test_does_not_crash()
  print "calling test_fwd_pass_compatible_with_OpLSTM()"
  test_fwd_pass_compatible_with_OpLSTM()
  print "calling test_bwd_pass_compatible_with_OpLSTM()"
  test_bwd_pass_compatible_with_OpLSTM()
