#TODO change to work with nose
import numpy
import theano
import theano.tensor as T
from FastLSTM import LSTMOp2Instance

def test_grad():
  X = T.ftensor3('X')
  W = T.fmatrix('W')
  V_h = T.fmatrix('V_h')
  b = T.fvector('b')
  c = T.fmatrix('c') #initial state
  i = T.matrix('i',dtype='int8')
  Z, H = LSTMOp2Instance(X, W, V_h, c, b, i)
  DX = T.grad(Z.sum(), X)
  DW = T.grad(Z.sum(), W)
  DV_h = T.grad(Z.sum(), V_h)
  Db = T.grad(Z.sum(), b)
  Dc = T.grad(Z.sum(), c)
  f = theano.function(inputs=[X, W, V_h, c, b], outputs=[Z, DX, DW, DV_h, Dc, Db])
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
  c_val = numpy.zeros((2,3), dtype='float32')

  #print "calling g"
  #Z_val, H_val = g(X_val, W_val, V_h_val, b_val)
  #print numpy.asarray(Z_val), '\n', numpy.asarray(H_val)
  #print "done calling g"

  print "calling f"
  Z_val, DX_val, DW_val, DV_h_val, Dc_val, Db_val = f(X_val, W_val, V_h_val, c_val, b_val)
  print numpy.asarray(Z_val), '\n', numpy.asarray(DX_val), '\n', \
    numpy.asarray(DW_val), '\n', numpy.asarray(DV_h_val), '\n', numpy.asarray(Dc_val), '\n', numpy.asarray(Db_val)
  print "done calling f"

  print "verifying grad..."

  #def testOp_only_b(b):
  #  return TestOp()(X_val, W_val, V_h_val, b)[0]
  #theano.tests.unittest_tools.verify_grad(testOp_only_b, [b_val])

  def LSTMOp_Z(X, W, V_h, c, b):
    return LSTMOp2Instance(X, W, V_h, c, b)[0]

  theano.tests.unittest_tools.verify_grad(LSTMOp_Z, [X_val, W_val, V_h_val, c_val, b_val])

  print "success"

def test_compatible_with_other_implementation():
  X = T.ftensor3('X')
  W = T.fmatrix('W')
  V_h = T.fmatrix('V_h')
  b = T.fvector('b')
  c = T.fmatrix('c') #initial state
  i = T.matrix('i',dtype='int8')
  Z, H = LSTMOp2Instance(X, W, V_h, c, b, i)
  DX = T.grad(Z.sum(), X)
  DW = T.grad(Z.sum(), W)
  DV_h = T.grad(Z.sum(), V_h)
  Db = T.grad(Z.sum(), b)
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
  i_val = numpy.ones((3,2),dtype='int8')

  n_cells = 3

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

  [state, Z2], _ = theano.scan(_step, sequences=[X],
                          outputs_info=[c, c])

  DX2 = T.grad(Z2.sum(), X)
  DW2 = T.grad(Z2.sum(), W)
  DV_h2 = T.grad(Z2.sum(), V_h)
  Db2 = T.grad(Z2.sum(), b)

  f = theano.function(inputs=[X, W, V_h, c, b, i], outputs=[Z, Z2, DX, DX2, DW, DW2, DV_h, DV_h2, Db, Db2])
  Z_val, Z2_val, DX_val, DX2_val, DW_val, DW2_val, DV_h_val, DV_h2_val, Db_val, Db2_val = f(X_val, W_val, V_h_val, c_val, b_val, i_val)
  vals_fast = [Z_val, DX_val, DW_val, DV_h_val, Db_val]
  vals_fast = [numpy.asarray(A, dtype='float32') for A in vals_fast]
  vals_simple = [Z2_val, DX2_val, DW2_val, DV_h2_val, Db2_val]

  for f, s in zip(vals_fast, vals_simple):
    assert numpy.allclose(f, s)
  print numpy.asarray(Z_val, 'float32')
  print Z2_val
  print "sucess"

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

  Z1, H1 = LSTMOp2Instance(V_h, c, b, i, X, W)
  Z2, H2 = LSTMOp2Instance(V_h, c, b, i, X, X2, W, W)
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

  f = theano.function(inputs=[X, W, V_h, c, b, i], outputs=[Z1, DX1, DW1])
  g = theano.function(inputs=[X, X2, W, V_h, c, b, i], outputs=[Z2, DX2, DW2])
  f_res = [numpy.asarray(A, dtype='float32') for A in f(X_val, W_val, V_h_val, c_val, b_val, i_val)]
  g_res = [numpy.asarray(A, dtype='float32') for A in g(X_val, X_val2, W_val, V_h_val, c_val, b_val, i_val)]
  for A1, A2 in zip(f_res, g_res):
    assert numpy.allclose(A1, A2)
  print f_res[0], g_res[0]

if __name__ == '__main__':
  #test_compatible_with_other_implementation()
  test_multiple_inputs()
