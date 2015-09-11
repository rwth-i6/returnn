import theano
import theano.tensor as T
import numpy

#TODO: pass inputs as shared variables to avoid alot of copying
def make_fwd_fun():
  #TODO later also use context as input
  y_p = T.fmatrix("y_p")
  W_att_re = T.fmatrix("W_att_re")
  z_re = T.dot(y_p, W_att_re) #TODO: use context here
  out = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_output_shared")
  updates = [(out, z_re)]
  return theano.function(inputs=[y_p, W_att_re], outputs=[], updates=updates, on_unused_input="warn"), out

def make_bwd_fun():
  y_p = T.fmatrix("y_p")
  W_att_re = T.fmatrix("W_att_re")
  z_re = T.dot(y_p, W_att_re)
  Dz_re = T.fmatrix("Dz_re")
  known_grads = {z_re: Dz_re}
  Dy_p = T.grad(None, y_p, known_grads=known_grads)

  #TODO
  out_Dy_p = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="out_Dy_p")
  updates = [(out_Dy_p,Dy_p)]
  return theano.function(inputs=[y_p, W_att_re, Dz_re], outputs=[], updates=updates, on_unused_input="warn"), out_Dy_p

fwd_fun, fwd_fun_res0 = make_fwd_fun()

bwd_fun, bwd_fun_res0 = make_bwd_fun()
