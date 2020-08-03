
from __future__ import print_function

import os
import theano
import theano.tensor as T
import numpy
from returnn.log import log
import theano.sandbox.cuda as theano_cuda
from collections import OrderedDict


debug_function_hook = None

def make_fwd_fun(recurrent_transform):
  y_p = recurrent_transform.y_p
  z_re, state_updates = recurrent_transform.step(y_p)
  custom_vars = recurrent_transform.get_sorted_custom_vars()
  state_vars = recurrent_transform.get_sorted_state_vars()

  z_re_shared = recurrent_transform.layer.shared(value=numpy.zeros((1,1),dtype="float32"), name="fwd_fun_z_re_shared")
  updates = [(z_re_shared, z_re)]
  custom_out = []
  state_shared_vars = {v: recurrent_transform.layer.shared(value=numpy.zeros((1,) * v.ndim, dtype="float32"), name=v.name) for v in state_vars}
  for v in state_vars:
    v_upd = state_updates[v]
    updates += [(state_shared_vars[v], v_upd)]
    custom_out += [state_shared_vars[v]]
  fwd_fun = theano.function(inputs=[y_p] + custom_vars + state_vars, outputs=[],
                            updates=updates, on_unused_input="ignore")
  if debug_function_hook:
    fwd_fun = debug_make_theano_function_wrapper(fwd_fun, "att_%i_fwd" % id(recurrent_transform), debug_function_hook, state_shared_vars.values())
  return fwd_fun, z_re_shared, custom_out


def make_bwd_fun(recurrent_transform):
  y_p = recurrent_transform.y_p
  z_re, state_updates = recurrent_transform.step(y_p)
  custom_vars = recurrent_transform.get_sorted_custom_vars()
  state_vars_prev = recurrent_transform.get_sorted_state_vars()

  Dz_re = recurrent_transform.tt.fmatrix("Dz_re")
  state_var_new_grads = {state_updates[v]: v.type("D_" + v.name) for v in state_vars_prev}
  state_var_new_grads_list = [state_var_new_grads[state_updates[k]] for k in state_vars_prev]
  known_grads = {z_re: Dz_re}
  known_grads.update(state_var_new_grads)
  if recurrent_transform.force_gpu:
    # We need the symbolic host representation.
    # See HostFromGpu.grad(). It expects that the output_grads are on the host, i.e. from type T.TensorType.
    # When this is taken out of known_grads, it will fail because they are all CudaNdarrayType.
    # This should anyway be optimized all away and fully taken to the GPU in the final function.
    for k, v in known_grads.items():
      known_grads[k] = theano_cuda.host_from_gpu(v)

  all_wrt = [y_p] + custom_vars + state_vars_prev
  all_grads = T.grad(None, all_wrt, known_grads=OrderedDict(known_grads), disconnected_inputs="ignore")
  assert len(all_grads) == 1 + len(custom_vars) + len(state_vars_prev)
  Dy_p = all_grads[0]
  custom_grads = all_grads[1:len(custom_vars)+1]
  state_var_prev_grads = all_grads[len(custom_vars)+1:]

  out_Dy_p = recurrent_transform.layer.shared(value=numpy.zeros((1,1),dtype="float32"), name="out_Dy_p")
  out_custom_grads = [recurrent_transform.layer.shared(value=numpy.zeros([1] * var.ndim, dtype="float32"), name="out_D_" + var.name) for var in custom_vars]
  out_state_var_prev_grads = [recurrent_transform.layer.shared(value=numpy.zeros([1] * var.ndim, dtype="float32"), name="out_D_" + var.name) for var in state_vars_prev]

  updates = [(out_Dy_p, Dy_p)]
  updates += [(out, out + grad) for out, grad in zip(out_custom_grads, custom_grads)]  # we accumulate the custom input grads
  updates += [(out, grad) for out, grad in zip(out_state_var_prev_grads, state_var_prev_grads)]
  bwd_fun = theano.function(inputs=[y_p] + custom_vars + state_vars_prev + [Dz_re] + state_var_new_grads_list,
                            outputs=[],
                            updates=updates,
                            on_unused_input="ignore")

  # Before we can accumulate the custom input grads, we need to initialize them with 0.
  custom_reset_updates = [(out, T.zeros_like(var)) for out, var in zip(out_custom_grads, custom_vars)]
  custom_reset_fn = theano.function(inputs=custom_vars, outputs=None, updates=custom_reset_updates)

  if debug_function_hook:
    bwd_fun = debug_make_theano_function_wrapper(bwd_fun, "att_%i_bwd" % id(recurrent_transform), debug_function_hook, [])
  return bwd_fun, custom_reset_fn, out_Dy_p, out_custom_grads + out_state_var_prev_grads


def print_wt(op,x):
  print(x.argmax(axis=0))


functions = {}

def setup_parent_functions(fn, recurrent_transform_id):
  import RecurrentTransform
  fn = "_".join(fn.split('_')[:-2])
  if fn in functions: return
  print("loading function", fn, "(pid %i)" % os.getpid(), file=log.v4)
  transform = RecurrentTransform.transforms_by_id[recurrent_transform_id]
  # New instance for the custom op.
  transform = transform.copy_for_custom()
  assert isinstance(transform, RecurrentTransform.RecurrentTransformBase)
  _setup_func(fn, transform)
  functions[fn] = transform

def _setup_func(fn, recurrent_transform):
  fwd_names = ["_fun_fwd", "_fun_fwd_res0", "_fun_fwd_res1"]
  bwd_names = ["_fun_bwd", "_fun_reset", "_fun_bwd_res0", "_fun_bwd_res1"]
  vs_fwd = make_fwd_fun(recurrent_transform)
  vs_bwd = make_bwd_fun(recurrent_transform)
  assert len(vs_fwd) == len(fwd_names)
  assert len(vs_bwd) == len(bwd_names)
  for v, postfix in zip(vs_fwd + vs_bwd, fwd_names + bwd_names):
    globals()[fn + postfix] = v

def debug_make_theano_function_wrapper(f, name, hook, other_values):
  def to_str(v):
    if isinstance(v, (list, tuple)):
      return "[%s]" % ", ".join([to_str(v0) for v0 in v])
    if isinstance(v, theano.compile.SharedVariable):
      return "%s = %s" % (str(v), to_str(v.get_value(borrow=True, return_internal_type=True)))
    v = numpy.asarray(v)
    if len(v.shape) >= 2:
      return "\n" + str(v)
    return str(v)
  def theano_func_wrapped(*args):
    res = f(*args)
    print("called", name, "args:", to_str(args), "res:", to_str(res), "other:", to_str(other_values))
    if hook and hook is not True:
      hook(f=f, name=name, args=args, res=res)
    return res
  return theano_func_wrapped
