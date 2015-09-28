import theano
import theano.tensor as T
import numpy
import theano.sandbox.cuda as cuda

def make_fwd_fun(custom_fun_maker):
  y_p, z_re, custom_vars = custom_fun_maker()

  z_re_shared = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_z_re_shared")
  updates = [(z_re_shared, z_re)]
  fwd_fun = theano.function(inputs=[y_p] + custom_vars, outputs=[], updates=updates, on_unused_input="warn")
  return fwd_fun, z_re_shared, []


def make_bwd_fun(custom_fun_maker):
  y_p, z_re, custom_vars = custom_fun_maker()

  Dz_re = cuda.fmatrix("Dz_re")
  known_grads = {z_re: Dz_re}
  Dy_p = T.grad(None, y_p, known_grads=known_grads)

  out_Dy_p = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="out_Dy_p")

  custom_grads = [T.grad(None, var, known_grads=known_grads) for var in custom_vars]
  custom_out = [theano.shared(value=numpy.zeros([1] * var.ndim, dtype="float32"), name=var.name) for var in custom_vars]
  custom_updates = [(out, out + grad) for out, grad in zip(custom_out, custom_grads)]
  custom_reset_updates = [(out, T.zeros_like(var)) for out, var in zip(custom_out, custom_vars)]
  custom_reset_fn = theano.function(inputs=custom_vars, outputs=None, updates=custom_reset_updates)

  updates = [(out_Dy_p, Dy_p)] + custom_updates
  bwd_fun = theano.function(inputs=[y_p, Dz_re] + custom_vars, outputs=[], updates=updates, on_unused_input="warn")
  return bwd_fun, custom_reset_fn, out_Dy_p, custom_out


def test_fun():
  y_p = cuda.fmatrix("y_p")
  W_att_in = cuda.fmatrix("W_att_in")
  z_re = T.dot(y_p, W_att_in)
  custom_vars = [W_att_in]
  return y_p, z_re, custom_vars


def attention_dot():
  y_p = cuda.fmatrix("y_p")
  B = cuda.ftensor3("B")
  W_att_in = cuda.fmatrix("W_att_in")
  W_att_quadr = cuda.fmatrix("W_att_quadr")
  custom_vars = [B,W_att_in,W_att_quadr]

  f_z = T.sum(B * T.tanh(T.dot(y_p, W_att_quadr)).dimshuffle('x',0,1).repeat(B.shape[0],axis=0), axis=2, keepdims=True)
  f_e = T.exp(f_z)
  w_t = f_e / T.sum(f_e, axis=0, keepdims=True)

  #import theano.printing
  #w_t = theano.printing.Print("w_t")(T.sum(T.log(w_t)*w_t))
  #theano.printing.Print("w_t")(T.argmax(w_t,axis=0))

  z_re = T.dot(T.sum(B * w_t, axis=0, keepdims=False), W_att_in)

  return y_p, z_re, custom_vars


def attention_time_gauss():
  y_p = cuda.fmatrix("y_p")  # s_t-1
  B = cuda.ftensor3("B")  # h

  W_att_in = cuda.fmatrix("W_att_in")
  W_att_quadr = cuda.fmatrix("W_att_quadr")
  custom_vars = [B,W_att_in,W_att_quadr]

  # TODO...
  f_z = T.sum(B * T.tanh(T.dot(y_p, W_att_quadr)).dimshuffle('x',0,1).repeat(B.shape[0],axis=0), axis=2, keepdims=True)
  f_e = T.exp(f_z)
  w_t = f_e / T.sum(f_e, axis=0, keepdims=True)

  z_re = T.dot(T.sum(B * w_t, axis=0, keepdims=False), W_att_in)


  return y_p, z_re, custom_vars



test_fun_fwd, test_fun_fwd_res0, test_fun_fwd_res1 = make_fwd_fun(test_fun)
test_fun_bwd, test_fun_reset, test_fun_bwd_res0, test_fun_bwd_res1 = make_bwd_fun(test_fun)



def setup_functions():
  fwd_names = ["_fun_fwd", "_fun_fwd_res0", "_fun_fwd_res1"]
  bwd_names = ["_fun_bwd", "_fun_reset", "_fun_bwd_res0", "_fun_bwd_res1"]
  for fn, f in list(globals().items()):
    if not fn.startswith("attention_"): continue
    vs_fwd = make_fwd_fun(f)
    vs_bwd = make_bwd_fun(f)
    assert len(vs_fwd) == len(fwd_names)
    assert len(vs_bwd) == len(bwd_names)
    for v, postfix in zip(vs_fwd + vs_bwd, fwd_names + bwd_names):
      globals()[fn + postfix] = v

setup_functions()
