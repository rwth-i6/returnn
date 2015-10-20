
import unittest
import theano
import theano.tensor as T
import numpy
from Device import have_gpu
import RecurrentTransform
import theano.sandbox.cuda as cuda
from Log import log
import better_exchook
import CustomLSTMFunctions

log.initialize(verbosity=[5])
better_exchook.replace_traceback_format_tb()
CustomLSTMFunctions.debug_function_hook = True


def get_attention(att_class, **kwargs):
  import OpLSTMCustom
  recurrent_transform = RecurrentTransform.get_dummy_recurrent_transform(att_class.name, **kwargs)
  assert isinstance(recurrent_transform, att_class)
  f = OpLSTMCustom.register_func(recurrent_transform)
  return f

LSTMCustomTestOpNoInplaceInstance = get_attention(RecurrentTransform.AttentionTest)
LSTMCustomDotAttentionOpNoInplaceInstance = get_attention(RecurrentTransform.AttentionDot)
from OpLSTM import LSTMOpInstance


@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_does_not_crash():
  Z = T.ftensor3('Z')
  W_re = T.fmatrix('W_re')
  W_att_in = T.fmatrix('W_att_in')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')
  Y, H, d = LSTMCustomTestOpNoInplaceInstance(Z, c, y0, i, W_re, W_att_in)

  f = theano.function(inputs=[Z, c, y0, i, W_re, W_att_in], outputs=Y)

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  numpy.random.seed(1234)
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_in_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #i_val = numpy.ones((n_T, n_batch), dtype='int8')
  i_val = numpy.array([[1,1,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,0,0]], dtype='int8').T

  Y_val = numpy.asarray(f(Z_val, c_val, y0_val, i_val, W_re_val, W_att_in_val))
  #print Y_val
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_fwd_pass_compatible_with_OpLSTM():
  Z = T.ftensor3('Z')
  W_re = T.fmatrix('W_re')
  W_att_in = T.fmatrix('W_att_in')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')

  Y, H, d = LSTMCustomTestOpNoInplaceInstance(Z, c, y0, i, W_re, W_att_in)
  W_re_modified = W_re + W_att_in
  Z_modified = T.inc_subtensor(Z[0], T.dot(y0,W_re_modified))
  Y2, H2, d2 = LSTMOpInstance(Z_modified, W_re_modified, c, i)

  f = theano.function(inputs=[Z, c, y0, i, W_re, W_att_in], outputs=Y)
  g = theano.function(inputs=[Z, W_re, c, y0, i, W_att_in], outputs=Y2)

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  numpy.random.seed(1234)
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_in_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #i_val = numpy.ones((n_T, n_batch), dtype='int8')
  i_val = numpy.array([[1,1,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,0,0]], dtype='int8').T

  Y_val = numpy.asarray(f(Z_val, c_val, y0_val, i_val, W_re_val, W_att_in_val))
  Y2_val = numpy.asarray(g(Z_val, W_re_val, c_val, y0_val, i_val, W_att_in_val))
  assert numpy.allclose(Y_val, Y2_val)
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_bwd_pass_compatible_with_OpLSTM():
  Z = T.ftensor3('Z')
  W_re = T.fmatrix('W_re')
  W_att_in = T.fmatrix('W_att_in')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')
  Y, H, d = LSTMCustomTestOpNoInplaceInstance(Z, c, y0, i, W_re, W_att_in)
  W_re_modified = W_re + W_att_in
  Z_modified = T.inc_subtensor(Z[0], T.dot(y0,W_re_modified))
  Y2, H2, d2 = LSTMOpInstance(Z_modified, W_re_modified, c, i)

  cost = Y.sum()
  DZ = T.grad(cost, Z)
  DW_re = T.grad(cost, W_re)
  DW_att_in = T.grad(cost, W_att_in)
  Dc = T.grad(cost, c)
  Dy0 = T.grad(cost, y0)
  cost2 = Y2.sum()
  DZ2 = T.grad(cost2, Z)
  DW_re2 = T.grad(cost2, W_re)
  DW_att_in2 = T.grad(cost2, W_att_in)
  Dc2 = T.grad(cost2, c)
  Dy02 = T.grad(cost2, y0)

  f = theano.function(inputs=[Z, c, y0, i, W_re, W_att_in], outputs=[DZ, DW_re, Dc, Dy0, DW_att_in])
  g = theano.function(inputs=[Z, W_re, c, y0, i, W_att_in], outputs=[DZ2, DW_re2, Dc2, Dy02, DW_att_in2])

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  numpy.random.seed(1234)
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_in_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  #W_att_in_val = numpy.zeros((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #i_val = numpy.ones((n_T, n_batch), dtype='int8')
  i_val = numpy.array([[1,1,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,0,0]], dtype='int8').T

  vals = f(Z_val, c_val, y0_val, i_val, W_re_val, W_att_in_val)
  DZ_val, DW_re_val, Dc_val, Dy0_val, DW_att_in_val = [numpy.asarray(x) for x in vals]
  vals2 = g(Z_val, W_re_val, c_val, y0_val, i_val, W_att_in_val)
  DZ2_val, DW_re2_val, Dc2_val, Dy02_val, DW_att_in2_val = [numpy.asarray(x) for x in vals2]
  assert numpy.allclose(DZ_val, DZ2_val, atol=5e-7, rtol=1e-4)
  assert numpy.allclose(DW_re_val, DW_re2_val, atol=5e-7, rtol=1e-4)
  assert numpy.allclose(Dc_val, Dc2_val)
  assert numpy.allclose(Dy0_val, Dy02_val)
  assert numpy.allclose(DW_att_in_val, DW_att_in2_val, atol=5e-7, rtol=1e-4)
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_grads():
  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  numpy.random.seed(1234)
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_in_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #i_val = numpy.ones((n_T, n_batch), dtype='int8')
  i_val = numpy.array([[1,1,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,0,0]], dtype='int8').T

  print "verifying grads..."

  #ignore W_att_in atm
  def LSTMCustomOp_Z_onlyZ(Z):
      return LSTMCustomTestOpNoInplaceInstance(Z, c_val, y0_val, i_val, W_re_val, W_att_in_val)[0]

  def LSTMCustomOp_Z(Z, c, y0, W_re, W_att_in):
    return LSTMCustomTestOpNoInplaceInstance(Z, c, y0, i_val, W_re, W_att_in)[0]

  def LSTMCustomOp_d(Z, c, y0, W_re, W_att_in):
    return LSTMCustomTestOpNoInplaceInstance(Z, c, y0, i_val, W_re, W_att_in)[2]

  print "verifying grad of Z (only w.r.t. Z)"
  theano.tests.unittest_tools.verify_grad(LSTMCustomOp_Z_onlyZ, [Z_val])
  print "verifying grad of Z"
  theano.tests.unittest_tools.verify_grad(LSTMCustomOp_Z, [Z_val, c_val, y0_val, W_re_val, W_att_in_val])
  print "verifying grad of d"
  theano.tests.unittest_tools.verify_grad(LSTMCustomOp_d, [Z_val, c_val, y0_val, W_re_val, W_att_in_val], eps=1e-3)

  print "success"

#---- attention functions

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_attention_dot_does_not_crash():
  Z = T.ftensor3('Z')
  B = T.ftensor3('B') #base
  W_re = T.fmatrix('W_re')
  W_att_quadr = T.fmatrix("W_att_quadr")
  W_att_in = T.fmatrix('W_att_in')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')
  Y, H, d = LSTMCustomDotAttentionOpNoInplaceInstance(Z, c, y0, i, W_re, B, W_att_in, W_att_quadr)

  f = theano.function(inputs=[Z, B, c, y0, i, W_re, W_att_in, W_att_quadr], outputs=Y)

  n_B = 8
  n_T = 5
  n_batch = 4
  n_cells = 8
  numpy.random.seed(1234)
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_B,n_batch,n_cells)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_quadr_val = numpy.eye(n_B).astype('float32')
  W_att_in_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #i_val = numpy.ones((n_T, n_batch), dtype='int8')
  i_val = numpy.array([[1,1,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,0,0]], dtype='int8').T

  Y_val = numpy.asarray(f(Z_val, B_val, c_val, y0_val, i_val, W_re_val, W_att_in_val, W_att_quadr_val))
  #print Y_val
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_attention_dot_grads():
  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  n_B = 8
  numpy.random.seed(1234)
  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_quadr_val = numpy.eye(n_B).astype('float32')
  W_att_in_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_B,n_batch,n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #i_val = numpy.ones((n_T, n_batch), dtype='int8')
  i_val = numpy.array([[1,1,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,0,0]], dtype='int8').T

  print "verifying grads..."

  def LSTMCustomOp_Z(Z, c, y0, W_re, B, W_att_in, W_att_quadr):
    return LSTMCustomDotAttentionOpNoInplaceInstance(Z, c, y0, i_val, W_re, B, W_att_in, W_att_quadr)[0]

  def LSTMCustomOp_d(Z, c, y0, W_re, B, W_att_in, W_att_quadr):
    return LSTMCustomDotAttentionOpNoInplaceInstance(Z, c, y0, i_val, W_re, B, W_att_in, W_att_quadr)[2]

  print "verifying grad of Z"
  theano.tests.unittest_tools.verify_grad(LSTMCustomOp_Z, [Z_val, c_val, y0_val, W_re_val, B_val, W_att_in_val, W_att_quadr_val])
  print "verifying grad of d"
  theano.tests.unittest_tools.verify_grad(LSTMCustomOp_d, [Z_val, c_val, y0_val, W_re_val, B_val, W_att_in_val, W_att_quadr_val], eps=1e-3)

  print "success"


@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_attention_time_gauss():
  n_T = 4
  n_batch = 2
  n_inp_dim = 3
  n_cells = 5
  n_B = 5

  custom_op = get_attention(RecurrentTransform.AttentionTimeGauss,
                            n_out=n_cells, n_batches=n_batch, n_input_t=n_B, n_input_dim=n_inp_dim)
  att = custom_op.recurrent_transform

  Z_val = numpy.random.ranf((n_T,n_batch,4*n_cells)).astype('float32')
  W_re_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  W_att_quadr_val = numpy.eye(n_B).astype('float32')
  W_att_in_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  B_val = numpy.random.ranf((n_B,n_batch,n_cells)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  Z = T.ftensor3('Z')
  B = T.ftensor3('B') #base
  W_re = T.fmatrix('W_re')
  W_att_quadr = T.fmatrix("W_att_quadr")
  W_att_in = T.fmatrix('W_att_in')
  c = T.fmatrix('c') #initial state
  y0 = T.fmatrix('y0') #initial activation
  i = T.matrix('i',dtype='int8')
  t0 = T.fvector('t0')
  custom_vars = att.get_sorted_custom_vars()
  initial_state_vars = att.get_sorted_state_vars_initial()
  custom_op_inputs = [Z, c, y0, i, W_re] + custom_vars + initial_state_vars
  print "input args num:", len(custom_op_inputs)
  print "input args:", custom_op_inputs
  custom_op_outputs = custom_op(*custom_op_inputs)
  print "output args num:", len(custom_op_outputs)
  custom_op_outputs = [cuda.host_from_gpu(v) for v in custom_op_outputs]
  f = theano.function(inputs=[Z, c, y0, i, W_re], outputs=custom_op_outputs)

  res = f(Z_val, c_val, y0_val, i_val, W_re_val)

  #print res
  # res: (output) Y, (gates and cell state) H, (final cell state) d, state vars sequences
  (Y, H, d), state_var_seqs = res[:3], res[3:]

  # print "running custom dumped data"
  # custom_op_inputs = [theano.shared(numpy.load("../op.i.%i" % i)) for i in range(12)]
  # custom_op_outputs = custom_op(*custom_op_inputs)
  # custom_op_outputs = [cuda.host_from_gpu(v) for v in custom_op_outputs]
  # f = theano.function(inputs=[], outputs=custom_op_outputs)
  # res = f()

  print res

  assert False


if __name__ == '__main__':
  #tests with attention
  print "calling test_attention_dot_does_not_crash()"
  test_attention_dot_does_not_crash()
  print "calling test_attention_dot_grads"
  test_attention_dot_grads()

  #tests without attention
  print "calling test_does_not_crash()"
  test_does_not_crash()
  print "calling test_fwd_pass_compatible_with_OpLSTM()"
  test_fwd_pass_compatible_with_OpLSTM()
  print "calling test_bwd_pass_compatible_with_OpLSTM()"
  test_bwd_pass_compatible_with_OpLSTM()
  print "calling test_grads()"
  test_grads()
