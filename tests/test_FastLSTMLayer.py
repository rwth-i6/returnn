import numpy
import theano
import theano.tensor as T
from FastLSTM import LSTMOp2Instance
from OpLSTM import LSTMOpInstance
import unittest
from Device import have_gpu

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_grad():
  X = T.ftensor3('X')
  W = T.fmatrix('W')
  V_h = T.fmatrix('V_h')
  b = T.fvector('b')
  c = T.fmatrix('c') #initial state
  i = T.matrix('i',dtype='int8')
  Z, H, d = LSTMOp2Instance(V_h, c, b, i, X, W)
  objective = Z.sum() + d.sum()
  DX = T.grad(objective, X)
  DW = T.grad(objective, W)
  DV_h = T.grad(objective, V_h)
  Db = T.grad(objective, b)
  Dc = T.grad(objective, c)
  f = theano.function(inputs=[V_h, c, b, i, X, W], outputs=[Z, d, DX, DW, DV_h, Dc, Db])
  #g = theano.function(inputs=[X, W, V_h, b], outputs=[Z,H])

  X_val_mat0 = 0.1 * numpy.array([[1,2,3], [4,5,6]], dtype='float32')
  X_val_mat1 = 0.1 * numpy.array([[5,1,8], [7,0,1]], dtype='float32')
  X_val_mat2 = 0.1 * numpy.array([[2,1,1], [-7,0,-1]], dtype='float32')
  X_val = numpy.zeros((3,2,3), dtype='float32')
  X_val[0, :, :] = X_val_mat0
  X_val[1, :, :] = X_val_mat1
  X_val[2, :, :] = X_val_mat2
  #should be divisable by 4 for lstm, attention: note the .T
  W_val = 0.1 * numpy.array([[3,1,2], [4,8,0], [7,7,1], [4,2,-5],
                             [6,-1,-2], [-4,8,0], [-7,2,1], [4,-2,-5],
                             [6,5,-2], [-4,8,-6], [-7,3,-1], [4,2,-5]], dtype='float32').T
  #(for lstm) size 1/4th
  V_h_val = 0.1 * numpy.array([[1,3,5], [2,-1,-1], [4, 8,-5], [0,-2,3],
                               [7,7,7], [1,2,3], [5,2,1], [-4,8,-4],
                               [-3,7,-7], [2,-2,-3], [-5,2,1], [-4,-5,-4]],
                              dtype='float32').T
  b_val = 0.1 * numpy.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype='float32')
  #c_val = numpy.zeros((2,3), dtype='float32')
  c_val = 0.1 * numpy.array([[1,2,-3],[6,-5,4]], dtype='float32')
  i_val = numpy.ones((3,2), dtype='int8')

  #print "calling g"
  #Z_val, H_val = g(X_val, W_val, V_h_val, b_val)
  #print numpy.asarray(Z_val), '\n', numpy.asarray(H_val)
  #print "done calling g"

  print "calling f"
  Z_val, d_val, DX_val, DW_val, DV_h_val, Dc_val, Db_val = f(V_h_val, c_val, b_val, i_val, X_val, W_val)
  #print numpy.asarray(Z_val), '\n', numpy.asarray(d_val), '\n', numpy.asarray(DX_val), '\n', \
    #numpy.asarray(DW_val), '\n', numpy.asarray(DV_h_val), '\n', numpy.asarray(Dc_val), '\n', numpy.asarray(Db_val)
  #print "----------"
  #print numpy.asarray(DX_val)
  #print "----------"
  print "done calling f"

  print "verifying grad..."

  #def testOp_only_b(b):
  #  return TestOp()(X_val, W_val, V_h_val, b)[0]
  #theano.tests.unittest_tools.verify_grad(testOp_only_b, [b_val])

  def LSTMOp_Z(V_h, c, b, X, W):
    return LSTMOp2Instance(V_h, c, b, i_val, X, W)[0]

  def LSTMOp_d(V_h, c, b, X, W):
    return LSTMOp2Instance(V_h, c, b, i_val, X, W)[2]

  print "verifying grad of Z"
  theano.tests.unittest_tools.verify_grad(LSTMOp_Z, [V_h_val, c_val, b_val, X_val, W_val])
  print "verifying grad of d"
  theano.tests.unittest_tools.verify_grad(LSTMOp_d, [V_h_val, c_val, b_val, X_val, W_val])

  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_grad_large():
  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  X_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_val = numpy.random.ranf((n_inp_dim, 4 * n_cells)).astype('float32')
  V_h_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  b_val = numpy.random.ranf((4 * n_cells,)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #c_val = numpy.zeros((n_batch, n_cells), dtype='float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  print "verifying grad..."

  def LSTMOp_Z(V_h, c, b, X, W):
    return LSTMOp2Instance(V_h, c, b, i_val, X, W)[0]

  def LSTMOp_d(V_h, c, b, X, W):
    return LSTMOp2Instance(V_h, c, b, i_val, X, W)[2]

  print "verifying grad of Z"
  theano.tests.unittest_tools.verify_grad(LSTMOp_Z, [V_h_val, c_val, b_val, X_val, W_val])
  print "verifying grad of d"
  theano.tests.unittest_tools.verify_grad(LSTMOp_d, [V_h_val, c_val, b_val, X_val, W_val], eps=1e-3)

  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_grad_large_with_index():
  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  X_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_val = numpy.random.ranf((n_inp_dim, 4 * n_cells)).astype('float32')
  V_h_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  b_val = numpy.random.ranf((4 * n_cells,)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  #c_val = numpy.zeros((n_batch, n_cells), dtype='float32')
  i_vals = [numpy.ones((n_T, n_batch), dtype='int8'),
            numpy.array([[1,1,1,1,1], [0,0,1,1,1], [0,0,1,1,1], [0,0,1,0,0]], dtype='int8').T]

  print "verifying grad..."

  for i_val in i_vals:
    def LSTMOp_Z(V_h, c, b, X, W):
      return LSTMOp2Instance(V_h, c, b, i_val, X, W)[0]

    def LSTMOp_d(V_h, c, b, X, W):
      return LSTMOp2Instance(V_h, c, b, i_val, X, W)[2]

    print "verifying grad of Z"
    theano.tests.unittest_tools.verify_grad(LSTMOp_Z, [V_h_val, c_val, b_val, X_val, W_val])
    print "verifying grad of d"
    theano.tests.unittest_tools.verify_grad(LSTMOp_d, [V_h_val, c_val, b_val, X_val, W_val], eps=1e-3)

  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_compatible_with_other_implementation():
  X = T.ftensor3('X')
  W = T.fmatrix('W')
  V_h = T.fmatrix('V_h')
  b = T.fvector('b')
  c = T.fmatrix('c') #initial state
  i = T.matrix('i',dtype='int8')
  #Y, _, _ = LSTMOp2Instance(V_h, c, b, i, X, W)
  Y, _, _ = LSTMOpInstance(T.dot(X,W) + b, V_h, c, i)
  DX = T.grad(Y.sum(), X)
  DW = T.grad(Y.sum(), W)
  DV_h = T.grad(Y.sum(), V_h)
  Db = T.grad(Y.sum(), b)

  n_T = 5
  n_batch = 4
  n_inp_dim = 3
  n_cells = 8
  X_val = numpy.random.ranf((n_T,n_batch,n_inp_dim)).astype('float32')
  W_val = numpy.random.ranf((n_inp_dim, 4 * n_cells)).astype('float32')
  V_h_val = numpy.random.ranf((n_cells, 4 * n_cells)).astype('float32')
  b_val = numpy.random.ranf((4 * n_cells,)).astype('float32')
  c_val = numpy.random.ranf((n_batch, n_cells)).astype('float32')
  y0_val = numpy.zeros((n_batch, n_cells), dtype='float32')
  i_val = numpy.ones((n_T, n_batch), dtype='int8')

  def _step(x_t, c_tm1, y_tm1):
    z_t = T.dot(x_t, W) + T.dot(y_tm1, V_h) + b
    partition = z_t.shape[1] / 4
    ingate = T.nnet.sigmoid(z_t[:,:partition])
    forgetgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
    outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
    input = T.tanh(z_t[:,3*partition:4*partition])
    c_t = forgetgate * c_tm1 + ingate * input
    y_t = outgate * T.tanh(c_t)
    return c_t, y_t

  [state, Y2], _ = theano.scan(_step, sequences=[X],
                          outputs_info=[c, y0_val])

  DX2 = T.grad(Y2.sum(), X)
  DW2 = T.grad(Y2.sum(), W)
  DV_h2 = T.grad(Y2.sum(), V_h)
  Db2 = T.grad(Y2.sum(), b)

  f = theano.function(inputs=[X, W, V_h, c, b, i], outputs=[Y, Y2, DX, DX2, DW, DW2, DV_h, DV_h2, Db, Db2])
  Y_val, Y2_val, DX_val, DX2_val, DW_val, DW2_val, DV_h_val, DV_h2_val, Db_val, Db2_val = f(X_val, W_val, V_h_val, c_val, b_val, i_val)
  vals_fast = [Y_val, DX_val, DW_val, DV_h_val, Db_val]
  vals_fast = [numpy.asarray(A, dtype='float32') for A in vals_fast]
  vals_simple = [Y2_val, DX2_val, DW2_val, DV_h2_val, Db2_val]

  names = ["Y_val", "DX_val", "DW_val", "DV_h_val", "Db_val"]
  for f, s, n in zip(vals_fast, vals_simple, names):
    assert numpy.allclose(f, s, rtol=3e-5), (n, f, s)

  print "sucess"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_compatible_with_other_implementation_and_index_vector():
  X = T.ftensor3('X')
  W = T.fmatrix('W')
  V_h = T.fmatrix('V_h')
  b = T.fvector('b')
  c = T.fmatrix('c') #initial state
  i = T.matrix('i', dtype='int8')
  #Z, _, h = LSTMOp2Instance(V_h, c, b, i, X, W)
  Z, _, h = LSTMOpInstance(T.dot(X,W) + b, V_h, c, i)
  obj = Z.sum() + h.sum()
  DX = T.grad(obj, X)
  DW = T.grad(obj, W)
  DV_h = T.grad(obj, V_h)
  Db = T.grad(obj, b)
  X_val_mat0 = 0.1 * numpy.array([[1,2,3], [4,5,6]], dtype='float32')
  X_val_mat1 = 0.1 * numpy.array([[5,1,8], [7,0,1]], dtype='float32')
  X_val_mat2 = 0.1 * numpy.array([[2,1,1], [-7,0,-1]], dtype='float32')
  X_val = numpy.zeros((3,2,3), dtype='float32')
  X_val[0, :, :] = X_val_mat0
  X_val[1, :, :] = X_val_mat1
  X_val[2, :, :] = X_val_mat2
  #should be divisable by 4 for lstm, attention: note the .T
  W_val = 0.1 * numpy.array([[3,1,2], [4,8,0], [7,7,1], [4,2,-5],
                             [6,-1,-2], [-4,8,0], [-7,2,1], [4,-2,-5],
                             [6,5,-2], [-4,8,-6], [-7,3,-1], [4,2,-5]], dtype='float32').T
  #(for lstm) size 1/4th
  V_h_val = 0.1 * numpy.array([[1,3,5], [2,-1,-1], [4, 8,-5], [0,-2,3],
                               [7,7,7], [1,2,3], [5,2,1], [-4,8,-4],
                               [-3,7,-7], [2,-2,-3], [-5,2,1], [-4,-5,-4]],
                              dtype='float32').T
  b_val = 0.1 * numpy.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype='float32')
  c_val = numpy.zeros((2,3), dtype='float32')
  i_vals = [numpy.array([[0,1], [1,1], [1,0]], dtype='int8'),
            numpy.array([[0,1], [0,1], [0,1]], dtype='int8'),
            numpy.ones((3,2), dtype='int8')] #layout of index vector: time x batch

  o_output = T.as_tensor(numpy.ones((3,), dtype='float32'))
  o_h = T.as_tensor(numpy.ones((3,), dtype='float32'))
  def _step(x_t, i_t, c_tm1, y_tm1):
    #z_t = T.dot(x_t, W) + T.dot(y_tm1, V_h) + b
    z_t = x_t + T.dot(y_tm1, V_h)
    partition = z_t.shape[1] / 4
    ingate = T.nnet.sigmoid(z_t[:,:partition])
    forgetgate = T.nnet.sigmoid(z_t[:,partition:2*partition])
    outgate = T.nnet.sigmoid(z_t[:,2*partition:3*partition])
    input = T.tanh(z_t[:,3*partition:4*partition])
    c_t = forgetgate * c_tm1 + ingate * input
    y_t = outgate * T.tanh(c_t)
    i_output = T.outer(i_t, o_output)
    i_h = T.outer(i_t, o_h)
    return c_t * i_h + c_tm1 * (1 - i_h), y_t * i_output

  #[state, Z2], _ = theano.scan(_step, sequences=[X, i],
  #                        outputs_info=[c, c])
  [state, Z2], _ = theano.scan(_step, sequences=[T.dot(X,W)+b, i],
                          outputs_info=[c, c])

  h2 = state[-1]
  obj2 = Z2.sum() + h2.sum()
  DX2 = T.grad(obj2, X)
  DW2 = T.grad(obj2, W)
  DV_h2 = T.grad(obj2, V_h)
  Db2 = T.grad(obj2, b)

  f = theano.function(inputs=[X, W, V_h, c, b, i], outputs=[Z, Z2, DX, DX2, DW, DW2, DV_h, DV_h2, Db, Db2, h2, h])
  for i_val in i_vals:
    Z_val, Z2_val, DX_val, DX2_val, DW_val, DW2_val, DV_h_val, DV_h2_val, \
      Db_val, Db2_val, h2_val, h_val = f(X_val, W_val, V_h_val, c_val, b_val, i_val)
    vals_fast_fwd = [Z_val, h_val]
    vals_fast_fwd = [numpy.asarray(A, dtype='float32') for A in vals_fast_fwd]
    vals_fast_grad = [DX_val, DW_val, DV_h_val, Db_val]
    vals_fast_grad = [numpy.asarray(A, dtype='float32') for A in vals_fast_grad]
    vals_simple_fwd = [Z2_val, h2_val]
    vals_simple_grad = [DX2_val, DW2_val, DV_h2_val, Db2_val]
    #print vals_fast_fwd
    #print vals_simple_fwd
    for fa, sl in zip(vals_fast_fwd, vals_simple_fwd):
      assert numpy.allclose(fa, sl)
    for fa, sl in zip(vals_fast_grad, vals_simple_grad):
      assert numpy.allclose(fa, sl)
  #print numpy.asarray(Z_val, 'float32')
  #print Z2_val
  print "success"

@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_multiple_inputs():
  X = T.ftensor3('X')
  X2 = T.ftensor3('X')
  W = T.fmatrix('W')
  V_h = T.fmatrix('V_h')
  b = T.fvector('b')
  c = T.fmatrix('c') #initial state
  i = T.matrix('i',dtype='int8')
  X_val_mat0 = 0.1 * numpy.array([[1,2,3], [4,5,6]], dtype='float32')
  X_val_mat1 = 0.1 * numpy.array([[5,1,8], [7,0,1]], dtype='float32')
  X_val_mat2 = 0.1 * numpy.array([[2,1,1], [-7,0,-1]], dtype='float32')
  X_val = numpy.zeros((3,2,3), dtype='float32')
  X_val[0, :, :] = X_val_mat0
  X_val[1, :, :] = X_val_mat1
  X_val[2, :, :] = X_val_mat2
  X_val2 = numpy.zeros_like(X_val)
  #should be divisable by 4 for lstm, attention: note the .T
  W_val = 0.1 * numpy.array([[3,1,2], [4,8,0], [7,7,1], [4,2,-5],
                             [6,-1,-2], [-4,8,0], [-7,2,1], [4,-2,-5],
                             [6,5,-2], [-4,8,-6], [-7,3,-1], [4,2,-5]], dtype='float32').T
  #(for lstm) size 1/4th
  V_h_val = 0.1 * numpy.array([[1,3,5], [2,-1,-1], [4, 8,-5], [0,-2,3],
                               [7,7,7], [1,2,3], [5,2,1], [-4,8,-4],
                               [-3,7,-7], [2,-2,-3], [-5,2,1], [-4,-5,-4]],
                              dtype='float32').T
  b_val = 0.1 * numpy.array([1,2,3,4,5,6,7,8,9,10,11,12], dtype='float32')
  c_val = numpy.zeros((2,3), dtype='float32')
  i_val = numpy.ones((3,2),dtype='int8')

  Z1, H1, d1 = LSTMOp2Instance(V_h, c, b, i, X, W)
  Z2, H2, d2 = LSTMOp2Instance(V_h, c, b, i, X, X2, W, W)
  Z3, H3, d3 = LSTMOp2Instance(V_h, c, b, i) # no inputs!
  DX1 = T.grad(Z1.sum(), X)
  DW1 = T.grad(Z1.sum(), W)
  DV_h1 = T.grad(Z1.sum(), V_h)
  Db1 = T.grad(Z1.sum(), b)
  Dc1 = T.grad(Z1.sum(), c)

  DX2 = T.grad(Z2.sum(), X)
  DW2 = T.grad(Z2.sum(), W)
  DV_h2 = T.grad(Z2.sum(), V_h)
  Db2 = T.grad(Z2.sum(), b)
  Dc2 = T.grad(Z2.sum(), c)

  DV_h3 = T.grad(Z3.sum(), V_h)

  f = theano.function(inputs=[X, W, V_h, c, b, i], outputs=[Z1, DX1, DW1])
  g = theano.function(inputs=[X, X2, W, V_h, c, b, i], outputs=[Z2, DX2, DW2])
  h = theano.function(inputs=[V_h, c, b, i], outputs=[Z3, DV_h3])
  h_res = [numpy.asarray(A, dtype='float32') for A in h(V_h_val, c_val, b_val, i_val)]
  #print h_res[0], h_res[1]
  f_res = [numpy.asarray(A, dtype='float32') for A in f(X_val, W_val, V_h_val, c_val, b_val, i_val)]
  g_res = [numpy.asarray(A, dtype='float32') for A in g(X_val, X_val2, W_val, V_h_val, c_val, b_val, i_val)]
  for A1, A2 in zip(f_res, g_res):
    assert numpy.allclose(A1, A2)
  #print f_res[0], g_res[0]

  print "success"

if __name__ == '__main__':
  print "calling test_compatible_with_other_implementation()"
  test_compatible_with_other_implementation()
  print "test_compatible_with_other_implementation(): success"
  print "------------------"

  print "calling test_compatible_with_other_implementation_and_index_vector()"
  test_compatible_with_other_implementation_and_index_vector()
  print "test_compatible_with_other_implementation_and_index_vector(): success"
  print "------------------"

  print "calling test_multiple_inputs()"
  test_multiple_inputs()
  print "test_multiple_inputs(): success"
  print "------------------"

  print "calling test_grad()"
  test_grad()
  print "test_grad(): success"
  print "------------------"

  print "calling test_grad_large()"
  test_grad_large()
  print "test_grad_large(): success"
  print "------------------"

  print "calling test_grad_large_with_index()"
  test_grad_large_with_index()
  print "test_grad_large_with_index(): success"
  print "------------------"
