import theano
import theano.tensor as T

def make_fwd_fun():
  s = T.fscalar()
  return theano.function(inputs=[s], outputs=100*s)

#def make_fwd_fun():
#  #s: state
#  #later also use context as input
#  s = T.ftensor3("s")
#  W_re = T.fmatrix("W_re")
#  #TODO check if we need to cast
#  idx_f = T.fscalar("idx")
#  idx = T.cast(idx_f, "int32")
#  z_re = T.dot(s[idx - 1], W_re)
#  return theano.function(inputs=[s, W_re, idx], outputs=[z_re])

def make_bwd_fun():
  s = T.ftensor3("s")
  W_re = T.fmatrix("W_re")
  idx_f = T.fscalar("idx")
  idx = T.cast(idx_f, "int32")
  Dz_re = T.fmatrix("Dz_re")

  z_re = T.dot(s[idx - 1], W_re)
  known_grads = {z_re: Dz_re}
  Ds = T.grad(None, s, known_grads=known_grads)
  DW_re = T.grad(None, W_re, known_grads=known_grads)
  return theano.function(inputs=[s, W_re, idx, Dz_re], outputs=[Ds, DW_re])

fwd_fun = make_fwd_fun()
bwd_fun = make_bwd_fun()
