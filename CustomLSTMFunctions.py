import theano
import theano.tensor as T
import numpy


#TODO: pass inputs as shared variables to avoid alot of copying
def make_test_fun_fwd():
  #B is not used in this test
  y_p = T.fmatrix("y_p")
  B = T.ftensor3("B")
  W_att_in = T.fmatrix("W_att_in")
  z_re = T.dot(y_p, W_att_in) #TODO: use context here
  out = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_output_shared")
  updates = [(out, z_re)]
  return theano.function(inputs=[y_p, B, W_att_in], outputs=[], updates=updates, on_unused_input="warn"), out


def make_test_fun_bwd():
  #B is not used in this test
  y_p = T.fmatrix("y_p")
  B = T.ftensor3("B")
  W_att_in = T.fmatrix("W_att_in")
  z_re = T.dot(y_p, W_att_in)
  Dz_re = T.fmatrix("Dz_re")
  known_grads = {z_re: Dz_re}
  Dy_p = T.grad(None, y_p, known_grads=known_grads)

  out_Dy_p = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="out_Dy_p")
  updates = [(out_Dy_p,Dy_p)]
  return theano.function(inputs=[y_p, B, W_att_in, Dz_re], outputs=[], updates=updates, on_unused_input="warn"), out_Dy_p

test_fun_fwd, test_fun_fwd_res0 = make_test_fun_fwd()
test_fun_bwd, test_fun_bwd_res0 = make_test_fun_bwd()


def attention_dot(y_p, B, W_att_in):
  #TODO: atm we only use B[0]
  e = T.batched_dot(B[0],y_p)
  linear_combination = e.dimshuffle(0,'x') * B[0]
  z_re = T.dot(linear_combination, W_att_in)

  #TODO: softmax and stuff
  #f_z = T.sum(att_x * T.tanh(T.dot(h_p, self.W_att_re)).dimshuffle('x',0,1).repeat(att_x.shape[0],axis=0), axis=2, keepdims=True)
  #f_e = T.exp(f_z)
  #w_t = f_e / T.sum(f_e, axis=0, keepdims=True)
  #z_t += T.dot(T.sum(att_x * w_t, axis=0, keepdims=False), self.W_att_in)

  #alpha = T.nnet.softmax(e)
  #linear_combination = T.dot(B.T, alpha)
  #z_re = T.dot(linear_combination, W_att_in)

  return z_re


def make_attention_dot_fun_fwd():
  #here we assume, that y_p and B have the same last dimension (so we can do a dotproduct without a matrix)
  y_p = T.fmatrix("y_p")
  B = T.ftensor3("B")
  W_att_in = T.fmatrix("W_att_in")
  z_re = attention_dot(y_p, B, W_att_in)

  out = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_output_shared")
  updates = [(out, z_re)]
  return theano.function(inputs=[y_p, B, W_att_in], outputs=[], updates=updates, on_unused_input="warn"), out


def make_attention_dot_fun_bwd():
  #here we assume, that y_p and B have the same last dimension (so we can do a dotproduct without a matrix)
  y_p = T.fmatrix("y_p")
  B = T.ftensor3("B")
  W_att_in = T.fmatrix("W_att_in")
  z_re = attention_dot(y_p, B, W_att_in)

  Dz_re = T.fmatrix("Dz_re")
  known_grads = {z_re: Dz_re}
  Dy_p = T.grad(None, y_p, known_grads=known_grads)

  out_Dy_p = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="out_Dy_p")
  updates = [(out_Dy_p,Dy_p)]
  return theano.function(inputs=[y_p, B, W_att_in, Dz_re], outputs=[], updates=updates, on_unused_input="warn"), out_Dy_p

attention_dot_fun_fwd, attention_dot_fun_fwd_res0 = make_attention_dot_fun_fwd()
attention_dot_fun_bwd, attention_dot_fun_bwd_res0 = make_attention_dot_fun_bwd()
