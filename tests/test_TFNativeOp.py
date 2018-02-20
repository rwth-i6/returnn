
# start test like this:  nosetests-2.7  tests/test_TFNativeOp.py  --nologcapture

from __future__ import print_function

import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
import os
import sys
print("__file__:", __file__)
base_path = os.path.realpath(os.path.dirname(os.path.abspath(__file__)) + "/..")
print("base path:", base_path)
sys.path.insert(0, base_path)
from TFNativeOp import *
from TFUtil import is_gpu_available, get_available_gpu_min_compute_capability, CudaEnv
import Util
import unittest
from nose.tools import assert_equal, assert_is_instance
import numpy
import numpy.testing
from numpy.testing.utils import assert_almost_equal, assert_allclose
import os
import better_exchook
better_exchook.replace_traceback_format_tb()


print("TF version:", tf.__version__)

CudaEnv.verbose_find_cuda = True
session = tf.InteractiveSession()


def sys_exec(*args, **kwargs):
  print("$ %s" % " ".join(args))
  out = Util.sysexecOut(*args, **kwargs)
  print(out)


def debug_lib_so(f, syms=()):
  ldd = "ldd"
  if sys.platform == "darwin":
    ldd = "otool -L"
  objdump = "objdump -T"
  if sys.platform == "darwin":
    objdump = "otool -IHGv"
  cmd = "%s %s" % (ldd, f)
  sys_exec(cmd, shell=True)
  for sym in syms:
    cmd = "%s %s | { grep %s || true; }" % (objdump, f, sym)
    sys_exec(cmd, shell=True)


def dump_info():
  # Some generic stuff.
  sys_exec("g++", "--version")
  print("TF include:", tf.sysconfig.get_include())
  print("TF lib:", tf.sysconfig.get_lib())
  tf_lib_so = tf.sysconfig.get_lib() + "/libtensorflow_framework.so"
  tf_pywrap_so = tf.sysconfig.get_lib() + "/python/_pywrap_tensorflow_internal.so"
  sys_exec("ls", "-la", tf.sysconfig.get_lib())
  if os.path.exists(tf_lib_so):
    print("TF lib so exists:", tf_lib_so)
    debug_lib_so(tf_lib_so, ["_ZTIN10tensorflow8OpKernelE"])
  else:
    print("TF lib so does not(!) exist:", tf_lib_so)
  if os.path.exists(tf_pywrap_so):
    print("TF pywrap so exists:", tf_pywrap_so)
    debug_lib_so(tf_pywrap_so, ["_ZTIN10tensorflow8OpKernelE"])
  else:
    print("TF pywrap so does not(!) exist:", tf_pywrap_so)
  # See OpCodeCompiler. Is already not used anymore but still maybe relevant.
  if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
    print("have (set|get)dlopenflags")
    import ctypes
    print("Cur flags: %r, RTLD_GLOBAL is set: %r" % (sys.getdlopenflags(), sys.getdlopenflags() & ctypes.RTLD_GLOBAL))
  if os.path.exists("/proc"):
    print("Have /proc")
    sys_exec("cat", "/proc/%i/maps" % os.getpid())
  # Numpy stuff, debugging if sgemm was not found:
  numpy_path = os.path.dirname(numpy.__file__)
  print("Numpy path: %r" % numpy_path)
  so_files = Util.sysexecOut("find %s | grep \"\.so\"" % numpy_path, shell=True)
  print("Numpy so files:\n---\n%s\n---\n" % so_files)
  so_files = [f for f in so_files.splitlines() if f]
  for f in so_files:
    debug_lib_so(f, ["sgemm"])


def test_dummy():
  dump_info()
  #assert False


def test_make_lstm_op_auto_cuda():
  try:
    make_lstm_op(compiler_opts={"verbose": True})
  except tf.errors.NotFoundError:
    dump_info()
    raise


def test_make_lstm_op_no_cuda():
  try:
    OpMaker.with_cuda = False
    make_lstm_op(compiler_opts={"verbose": True})
  except tf.errors.NotFoundError:
    dump_info()
    raise
  finally:
    OpMaker.with_cuda = None


def test_NativeLstmCell():
  n_time = 2
  n_batch = 1
  n_hidden = 3
  cell = NativeLstmCell(n_hidden=n_hidden)
  inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
  index = tf.ones([n_time, n_batch])
  outputs, final_state = cell(inputs, index)


def test_NativeLstmCell_run():
  from pprint import pprint
  from Util import describe_tensorflow_version
  print("TensorFlow:", describe_tensorflow_version())
  n_time = 2
  n_batch = 1
  n_hidden = 3
  with tf.Session() as session:
    with tf.variable_scope("test_NativeLstmCell_run"):
      cell = NativeLstmCell(n_hidden=n_hidden)
      inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
      index = tf.ones([n_time, n_batch])
      outputs, final_state = cell(inputs, index)
      session.run(tf.global_variables_initializer())
      res = session.run(outputs)
      pprint(res)


def test_NativeLstmLowMemCell():
  n_time = 2
  n_batch = 1
  n_in = 5
  n_hidden = 3
  cell = NativeLstmLowMemCell(n_hidden=n_hidden, n_input_dim=n_in)
  inputs = tf.zeros([n_time, n_batch, n_in])
  index = tf.ones([n_time, n_batch])
  outputs, final_state = cell(inputs, index)


def test_LstmLowMem_fwd_simple_1():
  lstm = make_op(NativeOp.LstmLowMem)
  n_time = 1
  n_batch = 1
  n_in = 1
  n_cells = 1
  vX = 0.2
  vb = 0.3
  vy0 = 0.1
  vc0 = 0.1
  X = tf.ones((n_time, n_batch, n_in)) * vX
  W = tf.ones((n_in + n_cells, n_cells * 4))
  b = tf.ones((n_cells * 4,)) * vb
  y0 = tf.ones((n_batch, n_cells)) * vy0
  c0 = tf.ones((n_batch, n_cells)) * vc0
  i = tf.ones((n_time, n_batch))
  start = tf.constant(0)
  step = tf.constant(1)
  Y, C, d = lstm(X, W, b, y0, c0, i, start, step)
  vY, vC, vd = session.run((Y, C, d))
  print("vY:", vY)
  print("vC:", vC)
  print("vd:", vd)
  assert_equal(vY.shape, (n_time, n_batch, n_cells))
  assert_equal(vC.shape, (n_time, n_batch, n_cells))
  assert_equal(d.shape, (n_batch, n_cells))
  vintern = vX + vy0 + vb
  vcellIn = numpy.tanh(vintern)
  vgates = 1. / (1. + numpy.exp(-vintern))  # sigmoid
  vc = vgates * vc0 + vgates * vcellIn
  vh = vgates * numpy.tanh(vc)
  assert_allclose(vY[0, 0, 0], vh)
  assert_allclose(vC[0, 0, 0], vc)
  assert_allclose(vd[0, 0], vc)


def test_LstmLowMem_bwd_simple_1():
  lstm = make_op(NativeOp.LstmLowMem)
  lstm_grad = lstm.grad_op
  n_time = 1
  n_batch = 1
  n_in = 1
  n_cells = 1
  vX = 0.2
  vb = 0.3
  vy0 = 0.1
  vc0 = 0.1
  vx_h = numpy.array([vX, vy0])
  print("vx_h:", vx_h)
  vintern = numpy.sum(vx_h) + vb
  print("vintern:", vintern)
  vcellIn = numpy.tanh(vintern)
  vgates = 1. / (1. + numpy.exp(-vintern))  # sigmoid
  print("vgates:", vgates)
  vc = vgates * vc0 + vgates * vcellIn
  vh = vgates * numpy.tanh(vc)
  X = tf.ones((n_time, n_batch, n_in)) * vX
  W = tf.ones((n_in + n_cells, n_cells * 4))
  b = tf.ones((n_cells * 4,)) * vb
  y0 = tf.ones((n_batch, n_cells)) * vy0
  c0 = tf.ones((n_batch, n_cells)) * vc0
  i = tf.ones((n_time, n_batch))
  start = tf.constant(0)
  step = tf.constant(1)
  Y = tf.ones((n_time, n_batch, n_cells)) * vh
  C = tf.ones((n_time, n_batch, n_cells)) * vc
  vDY = 1.5
  vDd = 1.2
  DY = tf.ones((n_time, n_batch, n_cells)) * vDY
  Dd = tf.ones((n_batch, n_cells)) * vDd
  DX, DW, Db, Dh, Dc = lstm_grad(X, W, b, y0, c0, i, start, step,   Y, C,   DY, Dd)
  vDX, vDW, vDb, vDh, vDc = session.run((DX, DW, Db, Dh, Dc))
  print("op vDX:", vDX)
  print("op vDW:", vDW)
  print("op vDb:", vDb)
  print("op vDh:", vDh)
  print("op vDc:", vDc)
  assert_equal(vDX.shape, (n_time, n_batch, n_in))
  assert_equal(vDW.shape, (n_in + n_cells, n_cells * 4))
  assert_equal(vDb.shape, (n_cells * 4,))
  assert_equal(vDh.shape, (n_batch, n_cells))
  assert_equal(vDc.shape, (n_batch, n_cells))
  vDh1 = vDY
  vgc = numpy.tanh(vc)
  vDoutGate_in = (1. - vgates) * vgates * vgc * vDh1
  vDc2 = (1. - vgc * vgc) * vgates * vDh1 + vDd
  vDcellIn_in = (1. - vcellIn * vcellIn) * vgates * vDc2
  vDinpGate_in = (1. - vgates) * vgates * vcellIn * vDc2
  vDfgtGate_in = (1. - vgates) * vgates * vc0 * vDc2
  vDintern = numpy.array([vDcellIn_in, vDinpGate_in, vDfgtGate_in, vDoutGate_in])
  vDb_ = vDintern
  assert_equal(vDb_.shape, vDb.shape)
  assert_allclose(vDb, vDb_)
  vDW_ = numpy.array([vX * vDintern, vy0 * vDintern])
  assert_equal(vDW_.shape, vDW.shape)
  assert_allclose(vDW, vDW_)
  vDx1 = numpy.sum(vDintern)
  assert_allclose(vDX, vDx1)
  vDh0 = numpy.sum(vDintern)
  assert_allclose(vDh, vDh0)
  vDc0 = vgates * vDc2
  assert_allclose(vDc, vDc0)


def lstm_step_op(x_t, h_tm1, c_tm1, mask_t, W_f, W_r, b, n_batch, n_in_dim, n_cells):
  """
  :param tf.Tensor x_t: shape (n_batch, n_in_dim)
  :param tf.Tensor h_tm1: shape (n_batch, n_cells)
  :param tf.Tensor c_tm1: shape (n_batch, n_cells)
  :param tf.Tensor mask_t: shape (n_batch,)
  :param tf.Tensor W_f: shape (n_in_dim, n_cells * 4)
  :param tf.Tensor W_r: shape (n_cells, n_cells * 4)
  :param tf.Tensor b: shape (n_cells * 4,)
  :param int n_batch:
  :param int n_in_dim:
  :param int n_cells:
  :return: h_t, c_t
  :rtype: (tf.Tensor, tf.Tensor)
  """
  x_t = tf.convert_to_tensor(x_t)
  h_tm1 = tf.convert_to_tensor(h_tm1)
  c_tm1 = tf.convert_to_tensor(c_tm1)
  mask_t = tf.convert_to_tensor(mask_t)
  W_f = tf.convert_to_tensor(W_f)
  W_r = tf.convert_to_tensor(W_r)
  b = tf.convert_to_tensor(b)
  x_t.set_shape(tf.TensorShape((n_batch, n_in_dim)))
  h_tm1.set_shape(tf.TensorShape((n_batch, n_cells)))
  c_tm1.set_shape(tf.TensorShape((n_batch, n_cells)))
  mask_t.set_shape(tf.TensorShape((n_batch,)))
  W_f.set_shape(tf.TensorShape((n_in_dim, n_cells * 4)))
  W_r.set_shape(tf.TensorShape((n_cells, n_cells * 4)))
  b.set_shape(tf.TensorShape((n_cells * 4,)))
  intern_x = tf.matmul(x_t, W_f)
  intern_x.set_shape(tf.TensorShape((n_batch, n_cells * 4)))
  intern_h = tf.matmul(h_tm1, W_r)
  intern_h.set_shape(tf.TensorShape((n_batch, n_cells * 4)))
  b_c = tf.expand_dims(b, 0)
  b_c.set_shape(tf.TensorShape((1, n_cells * 4)))
  intern = intern_x + intern_h + b_c
  intern.set_shape(tf.TensorShape((n_batch, n_cells * 4)))
  cell_in = tf.tanh(intern[:, :n_cells])
  inp_gate = tf.sigmoid(intern[:, n_cells:2 * n_cells])
  fgt_gate = tf.sigmoid(intern[:, 2 * n_cells:3 * n_cells])
  out_gate = tf.sigmoid(intern[:, 3 * n_cells:])
  cell_in.set_shape(tf.TensorShape((n_batch, n_cells)))
  inp_gate.set_shape(tf.TensorShape((n_batch, n_cells)))
  fgt_gate.set_shape(tf.TensorShape((n_batch, n_cells)))
  out_gate.set_shape(tf.TensorShape((n_batch, n_cells)))
  c_t = c_tm1 * fgt_gate + cell_in * inp_gate
  h_t = tf.tanh(c_t) * out_gate
  h_t.set_shape(tf.TensorShape((n_batch, n_cells)))
  c_t.set_shape(tf.TensorShape((n_batch, n_cells)))
  mask_t_c = tf.expand_dims(mask_t, 1)
  mask_t_c.set_shape(tf.TensorShape((n_batch, 1)))
  h_t = h_t * mask_t_c + h_tm1 * (1. - mask_t_c)
  c_t = c_t * mask_t_c + c_tm1 * (1. - mask_t_c)
  h_t.set_shape(tf.TensorShape((n_batch, n_cells)))
  c_t.set_shape(tf.TensorShape((n_batch, n_cells)))
  return h_t, c_t


def lstm(x, h_0, c_0, mask, W_f, W_r, b, n_time, n_batch, n_in_dim, n_cells, start=0, step=1, name="ref_lstm"):
  """
  :param tf.Tensor x: (n_time, n_batch, n_in_dim)
  :param tf.Tensor h_0: (n_batch, n_cells)
  :param tf.Tensor c_0: (n_batch, n_cells)
  :param tf.Tensor mask: (n_time, n_batch)
  :param tf.Tensor W_f: (n_in_dim, n_cells * 4)
  :param tf.Tensor W_r: (n_in_dim, n_cells * 4)
  :param tf.Tensor b: (n_cells * 4,)
  :param int n_time:
  :param int n_batch:
  :param int n_in_dim:
  :param int n_cells:
  :param int start:
  :param int step:
  :param str name:
  :return: h, c, c_T
  :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
  """
  assert n_time > 0
  assert 0 <= start < n_time
  assert step != 0
  x = tf.convert_to_tensor(x)
  h_0 = tf.convert_to_tensor(h_0)
  c_0 = tf.convert_to_tensor(c_0)
  mask = tf.convert_to_tensor(mask)
  W_f = tf.convert_to_tensor(W_f)
  W_r = tf.convert_to_tensor(W_r)
  b = tf.convert_to_tensor(b)
  x.set_shape(tf.TensorShape((n_time, n_batch, n_in_dim)))
  h_0.set_shape(tf.TensorShape((n_batch, n_cells)))
  c_0.set_shape(tf.TensorShape((n_batch, n_cells)))
  mask.set_shape(tf.TensorShape((n_time, n_batch)))
  W_f.set_shape(tf.TensorShape((n_in_dim, n_cells * 4)))
  W_r.set_shape(tf.TensorShape((n_cells, n_cells * 4)))
  b.set_shape(tf.TensorShape((n_cells * 4,)))
  h = tf.TensorArray(h_0.dtype, size=n_time, element_shape=tf.TensorShape((n_batch, n_cells)))
  c = tf.TensorArray(c_0.dtype, size=n_time, element_shape=tf.TensorShape((n_batch, n_cells)))
  h_t = h_0
  c_t = c_0
  ts = list(range(n_time))
  if step >= 0:
    end = n_time
  else:
    ts.reverse()
    start = n_time - start - 1
    end = -1
  ts_steps = list(range(start, end, step))
  print(name, "ts:", ts, "ts_steps:", ts_steps, "start:", start, "end:", end, "step:", step, "n_time", n_time)
  assert ts
  assert ts_steps
  for t in ts:
    #mask = tf.Print(mask, [name, "t:", t, "mask[t]:", mask[t]])
    #x = tf.Print(x, [name, "t:", t, "x[t]:", x[t]])
    if t in ts_steps:
      h_t, c_t = lstm_step_op(
        x_t=x[t], h_tm1=h_t, c_tm1=c_t, mask_t=mask[t],
        W_f=W_f, W_r=W_r, b=b,
        n_batch=n_batch, n_in_dim=n_in_dim, n_cells=n_cells)
      #h_t = tf.Print(h_t, [name, "t:", t, "h_t", h_t])
      #c_t = tf.Print(c_t, [name, "t:", t, "c_t", c_t])
      h = h.write(index=t, value=h_t)
      c = c.write(index=t, value=c_t)
    else:
      h = h.write(index=t, value=tf.zeros_like(h_0))
      c = c.write(index=t, value=tf.zeros_like(c_0))
  h = h.stack()
  c = c.stack()
  h.set_shape(tf.TensorShape((n_time, n_batch, n_cells)))
  c.set_shape(tf.TensorShape((n_time, n_batch, n_cells)))
  # TF 1.3.0 bug: https://github.com/tensorflow/tensorflow/issues/13355
  h = h[::1]
  c = c[::1]
  return h, c, c_t


def test_strided_slice_grad_grad():
  shape = (3, 2, 2)
  from tensorflow.python.ops.gen_array_ops import strided_slice_grad
  x = tf.reshape(tf.range(0, numpy.prod(shape), dtype=tf.float32), shape)
  y = strided_slice_grad(
    shape=tf.convert_to_tensor(shape),
    begin=tf.constant([0]),
    end=tf.constant([0], dtype=tf.int32),
    end_mask=1,
    strides=tf.constant([1]),
    dy=x)
  y.set_shape(x.get_shape())
  vx, vy = session.run([x, y])
  print("vx, vy:", vx, vy)
  assert_almost_equal(vx, vy)
  dy = tf.reshape(tf.range(0, numpy.prod(shape), dtype=tf.float32) * 0.1 - 5.135, shape)
  dx, = tf.gradients(ys=[y], xs=[x], grad_ys=[dy])
  vdx, vdy = session.run([dx, dy])
  print("vdx, vdy:", vdx, vdy)
  assert_almost_equal(vdx, vdy)


def test_strided_slice_grad_and_back_grad():
  shape = (3, 2, 2)
  from tensorflow.python.ops.gen_array_ops import strided_slice_grad
  x = tf.reshape(tf.range(0, numpy.prod(shape), dtype=tf.float32), shape)
  x_ = x[0::1]
  y = strided_slice_grad(
    shape=tf.convert_to_tensor(shape),
    begin=tf.constant([0]),
    end=tf.constant([0], dtype=tf.int32),
    end_mask=1,
    strides=tf.constant([1]),
    dy=x_)
  y.set_shape(x.get_shape())
  vx, vx_, vy = session.run([x, x_, y])
  print("vx, vx_, vy:", vx, vx_, vy)
  assert_almost_equal(vx, vx_)
  assert_almost_equal(vx, vy)
  dy = tf.reshape(tf.range(0, numpy.prod(shape), dtype=tf.float32) * 0.1 - 5.135, shape)
  dx, = tf.gradients(ys=[y], xs=[x], grad_ys=[dy])
  vdx, vdy = session.run([dx, dy])
  print("vdx, vdy:", vdx, vdy)
  assert_almost_equal(vdx, vdy)


def test_tensor_array_strided_slice_grad_grad():
  shape = (3, 2, 2)
  from tensorflow.python.ops.gen_array_ops import strided_slice_grad
  x = tf.reshape(tf.range(0, numpy.prod(shape), dtype=tf.float32), shape)
  x_ = tf.TensorArray(dtype=tf.float32, size=shape[0], element_shape=shape[1:])
  for t in range(shape[0]):
    x_ = x_.write(index=t, value=x[t])
  x__ = x_.stack()
  y = strided_slice_grad(
    shape=tf.convert_to_tensor(shape),
    begin=tf.constant([0]),
    end=tf.constant([0], dtype=tf.int32),
    end_mask=1,
    strides=tf.constant([1]),
    dy=x__)
  y.set_shape(x.get_shape())
  vx, vx__, vy = session.run([x, x__, y])
  print("vx, vx__, vy:", vx, vx__, vy)
  assert_almost_equal(vx, vx__)
  assert_almost_equal(vx, vy)
  dy = tf.reshape(tf.range(0, numpy.prod(shape), dtype=tf.float32) * 0.1 - 5.135, shape)
  dx, = tf.gradients(ys=[y], xs=[x], grad_ys=[dy])
  vdx, vdy = session.run([dx, dy])
  print("vdx, vdy:", vdx, vdy)
  assert_almost_equal(vdx, vdy)


def wrap_lstm_slice_start_step(op, x, h_0, c_0, mask, W_f, W_r, b, n_time, n_batch, n_in_dim, n_cells, start, step, name):
  x = tf.convert_to_tensor(x)
  mask = tf.convert_to_tensor(mask)
  if step >= 0:
    start_ = start
  else:
    start_ = n_time - start - 1
  x_ = x[start_::step]
  mask_ = mask[start_::step]
  abs_step = abs(step)
  num_steps = (n_time - start + abs_step - 1) // abs_step
  print(name, "start", start, "start_", start_, "step", step, "num_steps", num_steps)
  x_.set_shape(tf.TensorShape((num_steps, n_batch, n_in_dim)))
  mask_.set_shape(tf.TensorShape((num_steps, n_batch)))
  h, c, d = op(
    x=x_, h_0=h_0, c_0=c_0, mask=mask_,
    W_f=W_f, W_r=W_r, b=b,
    n_time=num_steps, n_batch=n_batch, n_in_dim=n_in_dim, n_cells=n_cells,
    start=0, step=1,
    name=name)
  h.set_shape(tf.TensorShape((num_steps, n_batch, n_cells)))
  c.set_shape(tf.TensorShape((num_steps, n_batch, n_cells)))
  d.set_shape(tf.TensorShape((n_batch, n_cells)))
  def reverse_strided_slice(shape, out):
    #if start == 0 and step == 1:
    #  return out
    from tensorflow.python.ops.gen_array_ops import strided_slice_grad
    out = strided_slice_grad(
      shape=tf.convert_to_tensor(shape),
      begin=tf.constant([start_]),
      end=tf.constant([0], dtype=tf.int32),
      end_mask=1,
      strides=tf.constant([step]),
      dy=out)
    assert isinstance(out, tf.Tensor)
    out.set_shape(tf.TensorShape(shape))
    return out
  h = reverse_strided_slice(shape=(n_time, n_batch, n_cells), out=h)
  c = reverse_strided_slice(shape=(n_time, n_batch, n_cells), out=c)
  return h, c, d


def lstm_slice(name, **kwargs):
  return wrap_lstm_slice_start_step(op=lstm, name="%s_slice" % name, **kwargs)


def native_lstm2(x, h_0, c_0, mask, W_f, W_r, b, n_time, n_batch, n_in_dim, n_cells, start=0, step=1, name="native_lstm2"):
  """
  :param tf.Tensor x: (n_time, n_batch, n_in_dim)
  :param tf.Tensor h_0: (n_batch, n_cells)
  :param tf.Tensor c_0: (n_batch, n_cells)
  :param tf.Tensor mask: (n_time, n_batch)
  :param tf.Tensor W_f: (n_in_dim, n_cells * 4)
  :param tf.Tensor W_r: (n_in_dim, n_cells * 4)
  :param tf.Tensor b: (n_cells * 4,)
  :param int n_time:
  :param int n_batch:
  :param int n_in_dim:
  :param int n_cells:
  :param int start:
  :param int step:
  :param str name:
  :return: h, c
  :rtype: (tf.Tensor, tf.Tensor)
  """
  assert n_time > 0
  assert 0 <= start < n_time
  assert step != 0
  x = tf.convert_to_tensor(x)
  h_0 = tf.convert_to_tensor(h_0)
  c_0 = tf.convert_to_tensor(c_0)
  mask = tf.convert_to_tensor(mask)
  W_f = tf.convert_to_tensor(W_f)
  W_r = tf.convert_to_tensor(W_r)
  b = tf.convert_to_tensor(b)
  x.set_shape(tf.TensorShape((n_time, n_batch, n_in_dim)))
  h_0.set_shape(tf.TensorShape((n_batch, n_cells)))
  c_0.set_shape(tf.TensorShape((n_batch, n_cells)))
  mask.set_shape(tf.TensorShape((n_time, n_batch)))
  W_f.set_shape(tf.TensorShape((n_in_dim, n_cells * 4)))
  W_r.set_shape(tf.TensorShape((n_cells, n_cells * 4)))
  b.set_shape(tf.TensorShape((n_cells * 4,)))
  op = make_op(NativeOp.NativeLstm2)
  from TFUtil import dot, expand_multiple_dims
  intern = dot(x, W_f)
  intern.set_shape(tf.TensorShape((n_time, n_batch, n_cells * 4)))
  intern += expand_multiple_dims(b, (0, 1))
  intern.set_shape(tf.TensorShape((n_time, n_batch, n_cells * 4)))
  y, c, _, d = op(intern, W_r, h_0, c_0, mask, start, step)
  y.set_shape(tf.TensorShape((n_time, n_batch, n_cells)))
  c.set_shape(tf.TensorShape((n_time, n_batch, n_cells)))
  d.set_shape(tf.TensorShape((n_batch, n_cells)))
  return y, c, d


def native_lstm2_slice(name, **kwargs):
  return wrap_lstm_slice_start_step(op=native_lstm2, name="%s_slice" % name, **kwargs)


def check_lstm_ops(op1, op2, name1, name2, rtol=1e-7, **kwargs):
  mask_bc = tf.expand_dims(kwargs["mask"], axis=2)
  mask_bc.set_shape(tf.TensorShape((kwargs["n_time"], kwargs["n_batch"], 1)))
  for start, step in [(0, 1), (0, -1), (1, 1), (1, -1), (0, 2), (0, -2)]:
    print("start, step:", start, step)
    if step >= 0:
      start_ = start
    else:
      start_ = kwargs["n_time"] - start - 1
    h1, c1, d1 = op1(start=start, step=step, name="%s_%i_%i" % (name1, start, step), **kwargs)
    h2, c2, d2 = op2(start=start, step=step, name="%s_%i_%i" % (name2, start, step), **kwargs)
    vh1, vh2, vc1, vc2, vd1, vd2, vmask = session.run((h1, h2, c1, c2, d1, d2, mask_bc))
    print("vh1:")
    print(vh1)
    print("vh2:")
    print(vh2)
    print("vc1:")
    print(vc1)
    print("vc2:")
    print(vc2)
    print("vd1:")
    print(vd1)
    print("vd2:")
    print(vd2)
    assert_allclose(vh1[start_::step] * vmask[start_::step], vh2[start_::step] * vmask[start_::step], rtol=rtol)
    assert_allclose(vc1[start_::step] * vmask[start_::step], vc2[start_::step] * vmask[start_::step], rtol=rtol)
    assert_allclose(vd1, vd2, rtol=rtol)


def check_lstm_op_start_step(op, name, **kwargs):
  name2 = "%s_slice" % name
  def wrapped_lstm_op(**kwargs):
    return wrap_lstm_slice_start_step(op=op, **kwargs)
  check_lstm_ops(op1=op, op2=wrapped_lstm_op, name1=name, name2=name2, **kwargs)


def lstm_kwargs():
  kwargs = {
    "n_time": 3,
    "n_batch": 2,
    "n_in_dim": 1,
    "n_cells": 2}
  kwargs["mask"] = numpy.array([[1, 1, 0], [1, 0, 0]]).astype("float32").transpose()
  assert kwargs["mask"].shape == (kwargs["n_time"], kwargs["n_batch"])
  def gen(shape, offset):
    return numpy.sin(numpy.arange(numpy.prod(shape)) + offset).reshape(shape).astype("float32")
  kwargs["x"] = gen((kwargs["n_time"], kwargs["n_batch"], kwargs["n_in_dim"]), offset=0.1)
  kwargs["h_0"] = gen((kwargs["n_batch"], kwargs["n_cells"]), offset=0.2)
  kwargs["c_0"] = gen((kwargs["n_batch"], kwargs["n_cells"]), offset=0.3)
  kwargs["W_f"] = numpy.ones((kwargs["n_in_dim"], kwargs["n_cells"] * 4), dtype="float32")
  kwargs["W_r"] = numpy.ones((kwargs["n_cells"], kwargs["n_cells"] * 4), dtype="float32")
  kwargs["b"] = numpy.zeros((kwargs["n_cells"] * 4,), dtype="float32")
  return kwargs


def test_ref_lstm_start_step_impl():
  kwargs = lstm_kwargs()
  check_lstm_op_start_step(op=lstm, name="ref_lstm", **kwargs)


def test_native_lstm2_start_step_impl():
  kwargs = lstm_kwargs()
  check_lstm_op_start_step(op=native_lstm2, name="native_lstm2", **kwargs)


def test_native_lstm2_impl():
  kwargs = lstm_kwargs()
  check_lstm_ops(
    op1=lstm, op2=native_lstm2, name1="ref_lstm", name2="native_lstm2",
    rtol=1e-6,
    **kwargs)


def lstm_grad_kwargs():
  def gen(shape, offset, factor=3.):
    return numpy.sin(numpy.arange(numpy.prod(shape)) * factor + offset).reshape(shape).astype("float32")
  kwargs = lstm_kwargs()
  mask_bc = kwargs["mask"][:, :, None]
  assert mask_bc.shape == (kwargs["n_time"], kwargs["n_batch"], 1)
  kwargs["dy"] = gen((kwargs["n_time"], kwargs["n_batch"], kwargs["n_cells"]), offset=0.15) * mask_bc
  kwargs["dd"] = gen((kwargs["n_batch"], kwargs["n_cells"]), offset=17)
  return kwargs


def wrap_lstm_grad(op, x, h_0, c_0, dy, dd, mask, W_f, W_r, b, n_time, n_batch, n_in_dim, n_cells, start, step, name):
  assert n_time > 0
  assert 0 <= start < n_time
  assert step != 0
  x = tf.convert_to_tensor(x)
  h_0 = tf.convert_to_tensor(h_0)
  c_0 = tf.convert_to_tensor(c_0)
  with tf.name_scope("gradients"):
    # Note that tensor_array_grad._GetGradSource() has this ugly hack
    # which requires that we have the "gradients" prefix.
    dy = tf.identity(tf.convert_to_tensor(dy), name="dy")
    dd = tf.identity(tf.convert_to_tensor(dd), name="dd")
  mask = tf.convert_to_tensor(mask)
  W_f = tf.convert_to_tensor(W_f)
  W_r = tf.convert_to_tensor(W_r)
  b = tf.convert_to_tensor(b)
  x.set_shape(tf.TensorShape((n_time, n_batch, n_in_dim)))
  h_0.set_shape(tf.TensorShape((n_batch, n_cells)))
  c_0.set_shape(tf.TensorShape((n_batch, n_cells)))
  dy.set_shape(tf.TensorShape((n_time, n_batch, n_cells)))
  dd.set_shape(tf.TensorShape((n_batch, n_cells)))
  mask.set_shape(tf.TensorShape((n_time, n_batch)))
  W_f.set_shape(tf.TensorShape((n_in_dim, n_cells * 4)))
  W_r.set_shape(tf.TensorShape((n_cells, n_cells * 4)))
  b.set_shape(tf.TensorShape((n_cells * 4,)))
  y, _, d = op(
    x=x, h_0=h_0, c_0=c_0, W_f=W_f, W_r=W_r, b=b, mask=mask,
    n_time=n_time, n_batch=n_batch, n_in_dim=n_in_dim, n_cells=n_cells,
    start=start, step=step, name=name)
  y.set_shape(tf.TensorShape((n_time, n_batch, n_cells)))
  d.set_shape(tf.TensorShape((n_batch, n_cells)))
  dx, dh0, dc0, dWf, dWr, db = tf.gradients(
    ys=[y, d], grad_ys=[dy, dd], xs=[x, h_0, c_0, W_f, W_r, b])
  print("grad", name, ":", dx, dh0, dc0, dWf, dWr, db)
  assert dx is not None
  assert dh0 is not None
  assert dc0 is not None
  assert dWf is not None
  assert dWr is not None
  assert db is not None
  dx.set_shape(x.get_shape())
  dh0.set_shape(h_0.get_shape())
  dc0.set_shape(c_0.get_shape())
  dWf.set_shape(W_f.get_shape())
  dWr.set_shape(W_r.get_shape())
  db.set_shape(b.get_shape())
  return y, d, dx, dh0, dc0, dWf, dWr, db


def check_lstm_grad_ops_single(op1, op2, name1, name2, dy, dd, rtol=1e-7, exclude=(), **kwargs):
  mask_bc = tf.expand_dims(kwargs["mask"], axis=2)
  mask_bc.set_shape(tf.TensorShape((kwargs["n_time"], kwargs["n_batch"], 1)))
  y1, d1, dx1, dh01, dc01, dWf1, dWr1, db1 = wrap_lstm_grad(op=op1, dy=dy, dd=dd, name=name1, **kwargs)
  y2, d2, dx2, dh02, dc02, dWf2, dWr2, db2 = wrap_lstm_grad(op=op2, dy=dy, dd=dd, name=name2, **kwargs)
  vy1, vd1, vy2, vd2, vdx1, vdh01, vdc01, vdWf1, vdWr1, vdb1, vdx2, vdh02, vdc02, vdWf2, vdWr2, vdb2, vmask = session.run([
    y1, d1, y2, d2, dx1, dh01, dc01, dWf1, dWr1, db1, dx2, dh02, dc02, dWf2, dWr2, db2, mask_bc])
  step = kwargs["step"]
  if step >= 0:
    start_ = kwargs["start"]
  else:
    start_ = kwargs["n_time"] - kwargs["start"] - 1
  print("check_lstm_grad_ops", name1, name2, start_, step)
  for k in ["y", "d", "dx", "dh0", "dc0", "dWf", "dWr", "db"]:
    for i in (1, 2):
      print("%s%i:" % (k, i))
      print(locals()["v%s%i" % (k, i)])
  for k in ["y", "d", "dx", "dh0", "dc0", "dWf", "dWr", "db"]:
    if k in exclude:
      continue
    v1 = locals()["v%s1" % k]
    v2 = locals()["v%s2" % k]
    if k in ["y"]:  # not for dx because the grad should be zero for masked frames
      v1 = (v1 * vmask)[start_::step]
      v2 = (v2 * vmask)[start_::step]
    print("check", k)
    assert_allclose(v1, v2, rtol=rtol, err_msg="no match for %s" % k)


def check_lstm_grad_ops(name1, name2, **kwargs):
  for start, step in [(0, 1), (0, -1), (1, 1), (1, -1), (0, 2), (0, -2)]:
    print(">>> check_lstm_grad_ops", start, step)
    check_lstm_grad_ops_single(
      name1="%s_%i_%i" % (name1, start, step),
      name2="%s_%i_%i" % (name2, start, step),
      start=start, step=step,
      **kwargs)


def check_lstm_grad_start_step(op, name, **kwargs):
  name2 = "%s_slice" % name
  def wrapped_lstm_op(**kwargs):
    return wrap_lstm_slice_start_step(op=op, **kwargs)
  check_lstm_grad_ops(
    op1=op, name1=name, op2=wrapped_lstm_op, name2=name2, **kwargs)


def test_ref_lstm_grad_start_step():
  kwargs = lstm_grad_kwargs()
  check_lstm_grad_start_step(op=lstm, name="ref_lstm", **kwargs)


def test_native_lstm2_grad_start_step():
  kwargs = lstm_grad_kwargs()
  check_lstm_grad_start_step(
    op=native_lstm2, name="native_lstm2", rtol=1e-5, **kwargs)


def test_native_lstm2_grad():
  kwargs = lstm_grad_kwargs()
  check_lstm_grad_ops(
    op1=lstm, name1="ref_lstm", op2=native_lstm2, name2="native_lstm2",
    rtol=1e-5, **kwargs)


def dummy_lstm_op(x, h_0, c_0, mask, W_f, W_r, b, n_time, n_batch, n_in_dim, n_cells, start, step, name):
  x = tf.convert_to_tensor(x)
  x.set_shape(tf.TensorShape((n_time, n_batch, n_in_dim)))
  # Have gradients for all.
  y = tf.reduce_mean(x, axis=2, keep_dims=True)
  y += tf.zeros((n_time, n_batch, n_cells))
  y *= tf.reduce_mean(W_f)
  y *= tf.reduce_mean(W_r)
  y += tf.reduce_mean(b)
  return y, y, h_0 + c_0


@unittest.skipIf(
  not TFUtil.have_min_tf_version((1, 5)), "TF 1.3.0 bug: https://github.com/tensorflow/tensorflow/issues/13355")
def test_tensorarray_grad():
  def gen(shape, offset):
    return (numpy.arange(numpy.prod(shape)) + offset).reshape(shape).astype("float32")

  n_time = 3
  n_batch = 2
  n_dim = 2
  x = gen((n_time, n_batch, n_dim), offset=-4)
  dy = gen((n_time, n_batch, n_dim), offset=-7)
  start = 0
  step = 1

  x = tf.convert_to_tensor(x)
  x.set_shape(tf.TensorShape((n_time, n_batch, n_dim)))
  with tf.name_scope("gradients"):
    # Note that tensor_array_grad._GetGradSource() has this ugly hack
    # which requires that we have the "gradients" prefix.
    dy = tf.identity(tf.convert_to_tensor(dy), name="dy")
  dy.set_shape(tf.TensorShape((n_time, n_batch, n_dim)))

  def ta_identity(x):
    h = tf.TensorArray(tf.float32, size=n_time, element_shape=tf.TensorShape((n_batch, n_dim)))
    for t in range(n_time):
      h = h.write(index=t, value=x[t])
    h = h.stack()
    h.set_shape(tf.TensorShape((n_time, n_batch, n_dim)))
    return h

  def wrapped(x):
    x_ = x[start::step]
    x_.set_shape(tf.TensorShape((n_time, n_batch, n_dim)))
    h = ta_identity(x=x_)
    h.set_shape(tf.TensorShape((n_time, n_batch, n_dim)))

    def reverse_strided_slice(shape, out):
      #return out
      # if start == 0 and step == 1:
      #  return out
      from tensorflow.python.ops.gen_array_ops import strided_slice_grad
      out = strided_slice_grad(
        shape=tf.convert_to_tensor(shape),
        begin=tf.constant([start]),
        end=tf.constant([0], dtype=tf.int32),
        end_mask=1,
        strides=tf.constant([step]),
        dy=out)
      assert isinstance(out, tf.Tensor)
      out.set_shape(tf.TensorShape(shape))
      return out

    h = reverse_strided_slice(shape=(n_time, n_batch, n_dim), out=h)
    return h

  y1 = ta_identity(x)
  y2 = wrapped(x)
  #y1 = tf.identity(y1)
  #y1 = y1[start::step]
  #y2 = tf.constant(0)
  dx1, = tf.gradients(ys=[y1], grad_ys=[dy], xs=[x])
  dx2, = tf.gradients(ys=[y2], grad_ys=[dy], xs=[x])
  #dx2 = tf.constant(-1)
  if dx1 is None:
    dx1 = tf.constant(-1)
  if dx2 is None:
    dx2 = tf.constant(-1)
  vx, vdy, vy1, vy2, vdx1, vdx2 = session.run([x, dy, y1, y2, dx1, dx2])
  print("x:")
  print(vx)
  print("y1:")
  print(vy1)
  print("y2:")
  print(vy2)
  print("dy:")
  print(vdy)
  print("dx1:")
  print(vdx1)
  print("dx2:")
  print(vdx2)
  assert_allclose(vx, vy1)
  assert_allclose(vy1[start::step], vy2[start::step])
  assert_allclose(vdy, vdx1)
  assert_allclose(vdx1[start::step], vdx2[start::step])


@unittest.skipIf(
  not TFUtil.have_min_tf_version((1, 5)), "TF 1.3.0 bug: https://github.com/tensorflow/tensorflow/issues/13355")
def test_tensorarray_grad_simple():
  n_time = 1
  n_dim = 1
  x = [[1.42]]
  dy = [[2.42]]

  x = tf.convert_to_tensor(x)
  x.set_shape(tf.TensorShape((n_time, n_dim)))
  with tf.name_scope("gradients"):
    # Note that tensor_array_grad._GetGradSource() has this ugly hack
    # which requires that we have the "gradients" prefix.
    dy = tf.identity(tf.convert_to_tensor(dy), name="dy")
  dy.set_shape(tf.TensorShape((n_time, n_dim)))

  ta = tf.TensorArray(tf.float32, size=n_time, element_shape=tf.TensorShape((n_dim,)))
  for t in range(n_time):
    ta = ta.write(index=t, value=x[t])
  y = ta.stack()
  y.set_shape(tf.TensorShape((n_time, n_dim)))
  # y = y[::1]  -- if you add this, the test passes
  dx, = tf.gradients(ys=[y], grad_ys=[dy], xs=[x])
  vx, vdy, vy, vdx = session.run([x, dy, y, dx])
  print("x:", vx)
  print("y:", vy)
  print("dy:", vdy)
  print("dx:", vdx)
  assert_allclose(vx, vy)
  assert_allclose(vdy, vdx)


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_FastBaumWelch():
  print("Make op...")
  op = make_fast_baum_welch_op(compiler_opts=dict(verbose=True))  # will be cached, used inside :func:`fast_baum_welch`
  print("Op:", op)
  n_batch = 3
  seq_len = 5
  n_classes = 10
  from Fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  fsa.add_inf_loop(state_idx=0, num_emission_labels=n_classes)
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = tf.constant(fast_bw_fsa.edges, dtype=tf.int32)
  weights = tf.constant(fast_bw_fsa.weights, dtype=tf.float32)
  start_end_states = tf.constant(fast_bw_fsa.start_end_states, dtype=tf.int32)
  am_scores = tf.constant(numpy.random.normal(size=(seq_len, n_batch, n_classes)), dtype=tf.float32)  # in -log space
  float_idx = tf.ones((seq_len, n_batch), dtype=tf.float32)
  print("Construct call...")
  fwdbwd, obs_scores = fast_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("Done.")
  print("Eval:")
  _, score = session.run([fwdbwd, obs_scores])
  print("score:", score)


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_fast_bw_uniform():
  print("Make op...")
  op = make_fast_baum_welch_op(compiler_opts=dict(verbose=True))  # will be cached, used inside :func:`fast_baum_welch`
  # args: (am_scores, edges, weights, start_end_states, float_idx, state_buffer)
  print("Op:", op)
  n_batch = 3
  seq_len = 7
  n_classes = 5
  from Fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  for i in range(n_classes):
    fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
    fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
  assert n_classes <= seq_len
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = tf.constant(fast_bw_fsa.edges, dtype=tf.int32)
  weights = tf.constant(fast_bw_fsa.weights, dtype=tf.float32)
  start_end_states = tf.constant(fast_bw_fsa.start_end_states, dtype=tf.int32)
  am_scores = numpy.ones((seq_len, n_batch, n_classes), dtype="float32") * numpy.float32(1.0 / n_classes)
  am_scores = -numpy.log(am_scores)  # in -log space
  am_scores = tf.constant(am_scores, dtype=tf.float32)
  float_idx = tf.ones((seq_len, n_batch), dtype=tf.float32)
  # from TFUtil import sequence_mask_time_major
  # float_idx = tf.cast(sequence_mask_time_major(tf.convert_to_tensor(list(range(seq_len - n_batch + 1, seq_len + 1)))), dtype=tf.float32)
  print("Construct call...")
  fwdbwd, obs_scores = fast_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("Done.")
  print("Eval:")
  fwdbwd, score = session.run([fwdbwd, obs_scores])
  print("score:")
  print(repr(score))
  assert_equal(score.shape, (seq_len, n_batch))
  bw = numpy.exp(-fwdbwd)
  print("Baum-Welch soft alignment:")
  print(repr(bw))
  assert_equal(bw.shape, (seq_len, n_batch, n_classes))
  from numpy import array, float32
  if seq_len == n_classes:
    print("Extra check identity...")
    for i in range(n_batch):
      assert_almost_equal(numpy.identity(n_classes), bw[:, i])
  if seq_len == 7 and n_classes == 5:
    print("Extra check ref_align (7,5)...")
    assert_allclose(score, 8.55801582, rtol=1e-5)  # should be the same everywhere
    ref_align = \
      array([[[1., 0., 0., 0., 0.]],
             [[0.33333316, 0.66666663, 0., 0., 0.]],
             [[0.06666669, 0.53333354, 0.40000018, 0., 0.]],
             [[0., 0.20000014, 0.60000014, 0.19999999, 0.]],
             [[0., 0., 0.39999962, 0.53333312, 0.06666663]],
             [[0., 0., 0., 0.66666633, 0.33333316]],
             [[0., 0., 0., 0., 0.99999982]]], dtype=float32)
    assert_equal(ref_align.shape, (seq_len, 1, n_classes))
    ref_align = numpy.tile(ref_align, (1, n_batch, 1))
    assert_equal(ref_align.shape, bw.shape)
    # print("Reference alignment:")
    # print(repr(ref_align))
    print("mean square diff:", numpy.mean(numpy.square(ref_align - bw)))
    print("max square diff:", numpy.max(numpy.square(ref_align - bw)))
    assert_allclose(ref_align, bw, rtol=1e-5)
  print("Done.")


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
@unittest.skipIf(is_gpu_available() and get_available_gpu_min_compute_capability() < 3.5, "too low compute capability")
def test_init_blocksparse():
  assert have_blocksparse_requirements()
  init_blocksparse()


@unittest.skipIf(not have_blocksparse_requirements(), "do not have Blocksparse requirements")
def test_blocksparse_simple():
  init_blocksparse()

  from blocksparse.matmul import BlocksparseMatMul
  import tensorflow as tf
  import numpy as np

  hidden_size = 4096
  block_size = 32
  minibatch_size = 64

  # Create a (random) sparsity pattern
  sparsity = np.random.randint(2, size=(hidden_size // block_size, hidden_size // block_size))

  # Initialize the sparse matrix multiplication object
  bsmm = BlocksparseMatMul(sparsity, block_size=block_size, feature_axis=0)

  # Input to graph
  x = tf.placeholder(tf.float32, shape=[None, hidden_size])

  # Initialize block-sparse weights
  w = tf.get_variable("w", bsmm.w_shape, dtype=tf.float32)

  # Block-sparse matrix multiplication
  y = bsmm(x, w)

  # Run
  session.run(tf.global_variables_initializer())
  result = session.run([y], feed_dict={x: np.ones((minibatch_size, hidden_size), dtype='float32')})
  print(result)


if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          try:
            v()
          except unittest.SkipTest as exc:
            print("SkipTest: %s" % exc)
          print("-" * 40)
      print("All passed.")
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    session.close()
    del session
    tf.reset_default_graph()
    import threading
    if len(list(threading.enumerate())) > 1:
      print("Warning, more than one thread at exit:")
      better_exchook.dump_all_thread_tracebacks()
