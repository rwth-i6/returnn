
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
  state_vars = recurrent_transform.get_sorted_state_vars()

  z_re_shared = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_z_re_shared")
  updates = [(z_re_shared, z_re)]
  custom_out = []
  state_shared_vars = {v: theano.shared(value=numpy.zeros((1,) * v.ndim, dtype="float32"), name=v.name) for v in state_vars}
  for v in state_vars:
    v_upd = state_updates[v]
    updates += [(state_shared_vars[v], v_upd)]
    custom_out += [state_shared_vars[v]]
  fwd_fun = theano.function(inputs=[y_p] + custom_vars + state_vars, outputs=[],
                            updates=updates, on_unused_input="warn")
  return fwd_fun, z_re_shared, custom_out


def make_bwd_fun(recurrent_transform):
  y_p = recurrent_transform.y_p
  z_re, state_updates = recurrent_transform.step(y_p)
  custom_vars = recurrent_transform.get_sorted_custom_vars()
  state_vars_prev = recurrent_transform.get_sorted_state_vars()

  Dz_re = tt.fmatrix("Dz_re")
  state_var_new_grads = {state_updates[k]: v.type("D_" + v.name) for (k, v) in state_vars_prev}
  known_grads = {z_re: Dz_re}
  known_grads.update(state_var_new_grads)

  Dy_p = T.grad(None, y_p, known_grads=known_grads, disconnected_inputs="ignore")
  custom_grads = [T.grad(None, var, known_grads=known_grads, disconnected_inputs="ignore") for var in custom_vars]
  state_var_prev_grads = [T.grad(None, var, known_grads=known_grads, disconnected_inputs="ignore") for var in state_vars_prev]

  out_Dy_p = theano.shared(value=numpy.zeros((1,1),dtype="float32"), name="out_Dy_p")
  out_custom_grads = [theano.shared(value=numpy.zeros([1] * var.ndim, dtype="float32"), name="out_D_" + var.name) for var in custom_vars]
  out_state_var_prev_grads = [theano.shared(value=numpy.zeros([1] * var.ndim, dtype="float32"), name="out_D_" + var.name) for var in state_vars_prev]

  updates = [(out_Dy_p, Dy_p)]
  updates += [(out, out + grad) for out, grad in zip(out_custom_grads, custom_grads)]  # we accumulate the custom input grads
  updates += [(out, grad) for out, grad in zip(out_state_var_prev_grads, state_var_prev_grads)]
  bwd_fun = theano.function(inputs=[y_p] + custom_vars + [Dz_re] + state_vars_prev,
                            outputs=[],
                            updates=updates,
                            on_unused_input="warn")

  # Before we can accumulate the custom input grads, we need to initialize them with 0.
  custom_reset_updates = [(out, T.zeros_like(var)) for out, var in zip(out_custom_grads, custom_vars)]
  custom_reset_fn = theano.function(inputs=custom_vars, outputs=None, updates=custom_reset_updates)

  return bwd_fun, custom_reset_fn, out_Dy_p, out_custom_grads + out_state_var_prev_grads


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
