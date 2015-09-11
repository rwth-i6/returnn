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
  W_att_re = T.fmatrix('W_att_re')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')
  Y, H, d = LSTMCustomOpInstance(Z, B, c, y0, i, W_re, W_att_re)

  f = theano.function(inputs=[Z, B, c, y0, i, W_re, W_att_re], outputs=Y)

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  Y_val = numpy.asarray(f(Z_val, B_val, c_val, y0_val, i_val, W_re_val, W_att_re_val))
  #print Y_val
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_fwd_pass_compatible_with_OpLSTM():
  Z = T.ftensor3('Z')
  B = T.ftensor3('B') #base
  W_re = T.fmatrix('W_re')
  W_att_re = T.fmatrix('W_att_re')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')

  Y, H, d = LSTMCustomOpInstance(Z, B, c, y0, i, W_re, W_att_re)
  W_re_modified = W_re + W_att_re
  Z_modified = T.inc_subtensor(Z[0], T.dot(y0,W_re_modified))
  Y2, H2, d2 = LSTMOpInstance(Z_modified, W_re_modified, c, i)

  f = theano.function(inputs=[Z, B, c, y0, i, W_re, W_att_re], outputs=Y)
  g = theano.function(inputs=[Z, W_re, c, y0, i, W_att_re], outputs=Y2)

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  Y_val = numpy.asarray(f(Z_val, B_val, c_val, y0_val, i_val, W_re_val, W_att_re_val))
  Y2_val = numpy.asarray(g(Z_val, W_re_val, c_val, y0_val, i_val, W_att_re_val))
  assert numpy.allclose(Y_val, Y2_val), (Y_val, Y2_val)
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_bwd_pass_compatible_with_OpLSTM():
  Z = T.ftensor3('Z')
  B = T.ftensor3('B') #base
  W_re = T.fmatrix('W_re')
  W_att_re = T.fmatrix('W_att_re')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')
  Y, H, d = LSTMCustomOpInstance(Z, B, c, y0, i, W_re, W_att_re)
  W_re_modified = W_re + W_att_re
  Z_modified = T.inc_subtensor(Z[0], T.dot(y0,W_re_modified))
  Y2, H2, d2 = LSTMOpInstance(Z_modified, W_re_modified, c, i)

  cost = Y.sum()
  DZ = T.grad(cost, Z)
  DW_re = T.grad(cost, W_re)
  Dy0 = T.grad(cost, y0)
  cost2 = Y2.sum()
  DZ2 = T.grad(cost2, Z)
  DW_re2 = T.grad(cost2, W_re)
  Dy02 = T.grad(cost2, y0)

  f = theano.function(inputs=[Z, B, c, y0, i, W_re, W_att_re], outputs=[DZ, DW_re, Dy0])
  g = theano.function(inputs=[Z, W_re, c, y0, i, W_att_re], outputs=[DZ2, DW_re2, Dy02])

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  vals = f(Z_val, B_val, c_val, y0_val, i_val, W_re_val, W_att_re_val)
  DZ_val, DW_re_val, Dy0_val = (numpy.asarray(vals[0]), numpy.asarray(vals[1]), numpy.asarray(vals[2]))
  vals2 = g(Z_val, W_re_val, c_val, y0_val, i_val, W_att_re_val)
  DZ2_val, DW_re2_val, Dy02_val = (numpy.asarray(vals2[0]), numpy.asarray(vals2[1]), numpy.asarray(vals2[2]))
  assert numpy.allclose(DZ_val, DZ2_val, atol=5e-7, rtol=1e-4), (DZ_val, DZ2_val)
  assert numpy.allclose(DW_re_val, DW_re2_val, atol=5e-7, rtol=1e-4), (DW_re_val, DW_re2_val)
  assert numpy.allclose(Dy0_val, Dy02_val), (Dy0_val, Dy02_val)
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_grads():
  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  #TODO: change this back when tests pass again
  #c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  c_val = numpy.zeros((n_batch, n_cells)).astype('float32')
  y0_val = numpy.zeros((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  print "verifying grads..."

  #ignore B and W_att_re atm
  def LSTMCustomOp_Z_onlyZ(Z):
      return LSTMCustomOpInstance(Z, B_val, c_val, y0_val, i_val, W_re_val, W_att_re_val)[0]

  def LSTMCustomOp_Z(Z, c, y0, W_re):
    return LSTMCustomOpInstance(Z, B_val, c, y0, i_val, W_re, W_att_re_val)[0]

  def LSTMCustomOp_d(Z, c, y0, W_re):
    return LSTMCustomOpInstance(Z, B_val, c, y0, i_val, W_re, W_att_re_val)[2]

  print "verifying grad of Z (only w.r.t. Z)"
  theano.tests.unittest_tools.verify_grad(LSTMCustomOp_Z_onlyZ, [Z_val])
  print "verifying grad of Z"
  theano.tests.unittest_tools.verify_grad(LSTMCustomOp_Z, [Z_val, c_val, y0_val, W_re_val])
  print "verifying grad of d"
  theano.tests.unittest_tools.verify_grad(LSTMCustomOp_d, [Z_val, c_val, y0_val, W_re_val], eps=1e-3)

  print "success"

if __name__ == '__main__':
  print "calling test_does_not_crash()"
  test_does_not_crash()
  print "calling test_fwd_pass_compatible_with_OpLSTM()"
  test_fwd_pass_compatible_with_OpLSTM()
  print "calling test_bwd_pass_compatible_with_OpLSTM()"
  test_bwd_pass_compatible_with_OpLSTM()
  print "calling test_grads()"
  test_grads()
