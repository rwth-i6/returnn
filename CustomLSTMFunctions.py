import theano
import theano.tensor as T
import numpy

#def make_fwd_fun():
#  x = T.fscalar()
#  y = theano.shared(value=numpy.zeros((),dtype="float32"), name="fwd_fun_output_shared")
#  updates = [(y,x)]
#  return theano.function(inputs=[x], outputs=[], updates=updates), y

#TODO: pass inputs as shared variables to avoid alot of copying
def make_fwd_fun():
  #TODO later also use context as input
  Y = T.ftensor3("Y")
  W_re = T.fmatrix("W_re")
  idx_f = T.fscalar("idx")
  idx = T.cast(idx_f, "int32")
  y_p = Y[idx - 1]
  z_re = T.dot(y_p, W_re)
  out = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_output_shared")
  updates = [(out,z_re)]
  return theano.function(inputs=[Y, W_re, idx_f], outputs=[], updates=updates), out

def make_bwd_fun():
  Y = T.ftensor3("Y")
  W_re = T.fmatrix("W_re")
  idx_f = T.fscalar("idx")
  idx = T.cast(idx_f, "int32")
  y_p = Y[idx - 1]
  z_re = T.dot(y_p, W_re)

  Dz_re = T.fmatrix("Dz_re")
  known_grads = {z_re: Dz_re}
  Dy_p = T.grad(None, y_p, known_grads=known_grads)
  DW_re = T.grad(None, W_re, known_grads=known_grads)

  #TODO
  out_Dy_p = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="bwd_fun_output_shared")
  out_DW_re = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="bwd_fun_output_shared")
  updates = [(out_Dy_p,Dy_p), (out_DW_re,DW_re)]
  return theano.function(inputs=[Y, W_re, idx_f, Dz_re], outputs=[], updates=updates), out_Dy_p, out_DW_re

fwd_fun, fwd_fun_res = make_fwd_fun()

bwd_fun, bwd_fun_res0, bwd_fun_res1 = make_bwd_fun()
bwd_fun_res = (bwd_fun_res0, bwd_fun_res1)
