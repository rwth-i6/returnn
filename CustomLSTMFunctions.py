
import os
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


def make_fwd_fun(recurrent_transform):
  y_p = recurrent_transform.y_p
  z_re, state_updates = recurrent_transform.step(y_p)
  custom_vars = recurrent_transform.get_sorted_custom_vars()

  z_re_shared = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_z_re_shared")
  updates = [(z_re_shared, z_re)]
  custom_out = []
  for k, v in state_updates.items():
    updates += [(k, v)]
  fwd_fun = theano.function(inputs=[y_p] + custom_vars, outputs=[], updates=updates, on_unused_input="warn")
  return fwd_fun, z_re_shared, custom_out


def make_bwd_fun(recurrent_transform):
  y_p = recurrent_transform.y_p
  z_re, state_updates = recurrent_transform.step(y_p)
  custom_vars = recurrent_transform.get_sorted_custom_vars()

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
  bwd_fun = theano.function(inputs=[y_p] + custom_vars + [Dz_re], outputs=[], updates=updates, on_unused_input="warn")
  return bwd_fun, custom_reset_fn, out_Dy_p, custom_out


def print_wt(op,x):
  print x.argmax(axis=0)

functions = {}

def setup_parent_functions(fn, recurrent_transform_id):
  import RecurrentTransform
  fn = "_".join(fn.split('_')[:-2])
  if fn in functions: return
  print >> log.v4, "loading function", fn, "(pid %i)" % os.getpid()
  transform = RecurrentTransform.transforms_by_id[recurrent_transform_id]
  # New instance for the custom op.
  transform = transform.__class__(force_gpu=True, for_custom=True, layer=transform.layer)
  assert isinstance(transform, RecurrentTransform.RecurrentTransformBase)
  _setup_func(fn, transform)
  functions[fn] = transform

def _setup_func(fn, recurrent_transform):
  fwd_names = ["_fun_fwd", "_fun_fwd_res0", "_fun_fwd_res1"]
  bwd_names = ["_fun_bwd", "_fun_reset", "_fun_bwd_res0", "_fun_bwd_res1"]
  recurrent_transform.create_vars_for_custom()
  vs_fwd = make_fwd_fun(recurrent_transform)
  vs_bwd = make_bwd_fun(recurrent_transform)
  assert len(vs_fwd) == len(fwd_names)
  assert len(vs_bwd) == len(bwd_names)
  for v, postfix in zip(vs_fwd + vs_bwd, fwd_names + bwd_names):
    globals()[fn + postfix] = v
