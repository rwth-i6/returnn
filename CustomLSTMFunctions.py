import theano
import theano.tensor as T
import numpy
from Device import have_gpu
from Log import log

if have_gpu():
  import theano.sandbox.cuda as theano_cuda
  tt = theano_cuda
else:
  tt = T


def make_fwd_fun(custom_fun_maker):
  y_p, z_re, custom_vars = custom_fun_maker()

  z_re_shared = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_z_re_shared")
  updates = [(z_re_shared, z_re)]
  fwd_fun = theano.function(inputs=[y_p] + custom_vars, outputs=[], updates=updates, on_unused_input="warn")
  return fwd_fun, z_re_shared, []


def make_bwd_fun(custom_fun_maker):
  y_p, z_re, custom_vars = custom_fun_maker()

  Dz_re = tt.fmatrix("Dz_re")
  known_grads = {z_re: Dz_re}
  Dy_p = T.grad(None, y_p, known_grads=known_grads, disconnected_inputs="ignore")

  out_Dy_p = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="out_Dy_p")

  custom_grads = [T.grad(None, var, known_grads=known_grads, disconnected_inputs="ignore") for var in custom_vars]
  custom_out = [theano.shared(value=numpy.zeros([1] * var.ndim, dtype="float32"), name=var.name) for var in custom_vars]
  custom_updates = [(out, out + grad) for out, grad in zip(custom_out, custom_grads)]
  custom_reset_updates = [(out, T.zeros_like(var)) for out, var in zip(custom_out, custom_vars)]
  custom_reset_fn = theano.function(inputs=custom_vars, outputs=None, updates=custom_reset_updates)

  updates = [(out_Dy_p, Dy_p)] + custom_updates
  bwd_fun = theano.function(inputs=[y_p, Dz_re] + custom_vars, outputs=[], updates=updates, on_unused_input="warn")
  return bwd_fun, custom_reset_fn, out_Dy_p, custom_out


def print_wt(op,x):
  print x.argmax(axis=0)

def setup_parent_functions(fn):
  if fn in globals(): return
  import RecurrentTransform
  fn = "_".join(fn.split('_')[:-2])
  print >> log.v4, "loading function",fn
  for att_clazz in RecurrentTransform.transforms.values():
    if att_clazz.name == fn:
      att = att_clazz(force_gpu=True)
      assert isinstance(att, RecurrentTransform.RecurrentTransformBase)
      fn = att.name
      globals()[fn] = att.function_for_custom_op
      _setup_func(fn)
      break

def _setup_func(fn):
  f = globals()[fn]
  fwd_names = ["_fun_fwd", "_fun_fwd_res0", "_fun_fwd_res1"]
  bwd_names = ["_fun_bwd", "_fun_reset", "_fun_bwd_res0", "_fun_bwd_res1"]
  vs_fwd = make_fwd_fun(f)
  vs_bwd = make_bwd_fun(f)
  assert len(vs_fwd) == len(fwd_names)
  assert len(vs_bwd) == len(bwd_names)
  for v, postfix in zip(vs_fwd + vs_bwd, fwd_names + bwd_names):
    globals()[fn + postfix] = v

def _setup_functions():
  import RecurrentTransform
  for att_clazz in RecurrentTransform.transforms.values():
    att = att_clazz(force_gpu=True)
    assert isinstance(att, RecurrentTransform.RecurrentTransformBase)
    fn = att.name
    globals()[fn] = att.function_for_custom_op
    _setup_func(fn)

#_setup_functions()
