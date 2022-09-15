
# start test like this:  nosetests-2.7  tests/test_TFNativeOp.py  --nologcapture

from __future__ import print_function

import _setup_test_env  # noqa
import _set_num_threads1  # noqa
import os
import sys
import typing
import logging
import tensorflow as tf
from returnn.tf.native_op import *
import returnn.tf.compat as tf_compat
import returnn.tf.util.basic as tf_util
from returnn.tf.util.basic import is_gpu_available, is_tf_cuda_build, CudaEnv
import returnn.util.basic as util
import returnn.native_op as native_op
from returnn.util.basic import unicode
import unittest
from nose.tools import assert_equal, assert_is_instance
import numpy
import numpy.testing
from numpy.testing.utils import assert_almost_equal, assert_allclose
import os
from pprint import pprint
from returnn.util import better_exchook
import returnn.util.debug as debug


print("__file__:", __file__)
base_path = os.path.realpath(os.path.dirname(os.path.abspath(__file__)) + "/..")
print("base path:", base_path)
print("TF version:", tf.__version__)

CudaEnv.verbose_find_cuda = True
session = tf_compat.v1.InteractiveSession()
tf_compat.v1.set_random_seed(42)


def sys_exec(*args, **kwargs):
  print("$ %s" % " ".join(args))
  out = util.sys_exec_out(*args, **kwargs)
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


def test_numpy_gemm():
  a = numpy.random.randn(100, 50).astype(numpy.float32)
  b = numpy.random.randn(50, 13).astype(numpy.float32)
  c = numpy.dot(a, b)
  assert numpy.isfinite(c).all()


def dump_info():
  # Some generic stuff.
  print("Number available CPUs:", util.get_number_available_cpus())
  sys_exec("g++", "--version")
  print("TF __file__:", tf.__file__)
  print("TF version:", tf.__version__)
  print("TF describe version:", util.describe_tensorflow_version())
  print("TF include:", tf.sysconfig.get_include())
  print("TF lib:", tf.sysconfig.get_lib())
  if tf_util.have_min_tf_version((1, 14)):
    print("TF link flags:", tf.sysconfig.get_link_flags())
    print("TF compile flags:", tf.sysconfig.get_compile_flags())
  if hasattr(tf, "sysconfig") and hasattr(tf.sysconfig, "CXX11_ABI_FLAG"):
    cxx11_abi_flag = tf.sysconfig.CXX11_ABI_FLAG
  else:
    cxx11_abi_flag = getattr(tf, 'CXX11_ABI_FLAG', "<undefined>")
  print("TF cxx11 abi flag:", cxx11_abi_flag)
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
  print("TF compiler version:", getattr(tf, "__compiler_version__", None))
  print("GCC for TF:", tf_util.get_tf_gcc_path())
  print("Available GCC versions:")
  for p in os.environ["PATH"].split(":"):
    if os.path.isdir(p):
      for name in os.listdir(p):
        if name.startswith("gcc"):
          print("  %s/%s" % (p, name))
  # See OpCodeCompiler. Is already not used anymore but still maybe relevant.
  if hasattr(sys, "getdlopenflags") and hasattr(sys, "setdlopenflags"):
    print("have (set|get)dlopenflags")
    import ctypes
    print("Cur flags: %r, RTLD_GLOBAL is set: %r" % (sys.getdlopenflags(), sys.getdlopenflags() & ctypes.RTLD_GLOBAL))
  if os.path.exists("/proc"):
    print("Have /proc")
    # sys_exec("cat", "/proc/%i/maps" % os.getpid())
    print("Mapped executables/libs:")
    fns = util.collect_proc_maps_exec_files()
    pprint(fns)
    fns_with_sgemm = []
    for fn in fns:
      out = util.find_sym_in_exec(fn, "sgemm_")
      if out:
        print(out)
        fns_with_sgemm.append(fn)
    print("Found libs with sgemm:")
    pprint(fns_with_sgemm)
  else:
    print("Does not have /proc.")
  # Numpy stuff, debugging if sgemm was not found:
  numpy_path = os.path.dirname(numpy.__file__)
  print("Numpy path: %r" % numpy_path)
  print("Numpy config:")
  numpy.show_config()
  so_files = util.sys_exec_out("find %s | grep \"\\.so\"" % numpy_path, shell=True)
  print("Numpy so files:\n---\n%s\n---\n" % so_files)
  so_files = [f for f in so_files.splitlines() if f]
  for f in so_files:
    debug_lib_so(f, ["sgemm"])
  print("find_libcudart_from_runtime:", util.find_libcudart_from_runtime())
  print("_cuda_path_candidate_via_proc_map_libcudart:", tf_util.CudaEnv._cuda_path_candidate_via_proc_map_libcudart())
  for p in tf_util.CudaEnv._cuda_path_candidates():
    print("CUDA path candidate %s, lib dir %s, valid %s" % (
      p, tf_util.CudaEnv._get_lib_dir_name(p), tf_util.CudaEnv._check_valid_cuda_path(p)))


# Do this here such that we always see this log in Travis.
orig_stdout = sys.stdout
try:
  sys.stdout = sys.__stdout__  # Nosetests has overwritten sys.stdout
  print("travis_fold:start:script.dump_info")  # https://github.com/travis-ci/travis-ci/issues/1065
  dump_info()
except Exception as exc:
  print("dump_info exception:", exc)
  print("(See more info in test_dummy output.)")

  # Define this test only in case we failed.
  def test_dummy():
    dump_info()

finally:
  print("travis_fold:end:script.dump_info")
  sys.stdout = orig_stdout


def test_NativeLstm2_compile():
  op = make_op(native_op.NativeLstm2, compiler_opts={"verbose": True})
  print("op:", op)
  maker = op._op_maker
  print("op maker:", maker)
  mod = op._op_module
  print("op mod:", mod)
  comp = mod._op_compiler
  print("op compiler:", comp)
  assert isinstance(comp, tf_util.OpCodeCompiler)
  print("info dict:")
  pprint(comp._info_dict)


# Do this here such that we always see this log in Travis.
try:
  sys.stdout = sys.__stdout__
  print("travis_fold:start:script.nativelstm2compile")
  test_NativeLstm2_compile()
except Exception as exc:
  print("NativeLstm2 compile exception:", exc)
finally:
  print("travis_fold:end:script.nativelstm2compile")
  sys.stdout = orig_stdout


# Do this now such that we ensure that some Numpy gemm function was called early,
# which might trigger OpenBlas init or so.
test_numpy_gemm()


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
  from returnn.util.basic import describe_tensorflow_version
  print("TensorFlow:", describe_tensorflow_version())
  n_time = 2
  n_batch = 1
  n_hidden = 3
  with tf_compat.v1.Session() as session:
    with tf_compat.v1.variable_scope("test_NativeLstmCell_run"):
      cell = NativeLstmCell(n_hidden=n_hidden)
      inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
      index = tf.ones([n_time, n_batch])
      outputs, final_state = cell(inputs, index)
      session.run(tf_compat.v1.global_variables_initializer())
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
  lstm = make_op(native_op.LstmLowMem)
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
  lstm = make_op(native_op.LstmLowMem)
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


def test_NativeLstm2_run():
  from pprint import pprint
  from returnn.util.basic import describe_tensorflow_version
  print("TensorFlow:", describe_tensorflow_version())
  n_time = 2
  n_batch = 1
  n_hidden = 3
  with tf_compat.v1.Session() as session:
    with tf_compat.v1.variable_scope("test_NativeLstm2_run"):
      cell = NativeLstm2(n_hidden=n_hidden)
      inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
      index = tf.ones([n_time, n_batch])
      outputs, final_state = cell(inputs, index)
      session.run(tf_compat.v1.global_variables_initializer())
      res = session.run(outputs)
      pprint(res)


def test_NativeLstm2_shape_inference_normal():
  op = make_op(native_op.NativeLstm2, compiler_opts={"verbose": True})
  n_time = 2
  n_batch = 1
  n_hidden = 3
  with tf_compat.v1.variable_scope("test_NativeLstm2_shape_inference_normal"):
    weights = tf_compat.v1.get_variable(name="W_re", shape=(n_hidden, n_hidden * 4))
    inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
    index = tf.ones([n_time, n_batch])
    n_batch_ = tf.shape(inputs)[1]
    c0 = tf.zeros((n_batch_, n_hidden), dtype=tf.float32, name="initial_c")
    y0 = tf.zeros((n_batch_, n_hidden), dtype=tf.float32, name="initial_h")
    start = tf.constant(0, name="start")
    step = tf.constant(1, name="step")
    print("inputs:", inputs, "shape:", inputs.shape)
    out, _, _, final_cell_state = op(inputs, weights, y0, c0, index, start, step)
    print("out:", out)
    assert isinstance(out, tf.Tensor)
    assert_equal(out.shape.as_list(), [n_time, n_batch, n_hidden])


def test_NativeLstm2_shape_inference_unknown_batchnlen():
  op = make_op(native_op.NativeLstm2, compiler_opts={"verbose": True})
  n_time = None
  n_batch = None
  n_hidden = 3
  with tf_compat.v1.variable_scope("test_NativeLstm2_shape_inference_unknown_batchnlen"):
    weights = tf_compat.v1.get_variable(name="W_re", shape=(n_hidden, n_hidden * 4))
    inputs = tf_compat.v1.placeholder(tf.float32, [n_time, n_batch, n_hidden * 4], name="inputs")
    index = tf_compat.v1.placeholder(tf.float32, [n_time, n_batch], name="index")
    n_batch = tf.shape(inputs)[1]
    c0 = tf.zeros((n_batch, n_hidden), dtype=tf.float32, name="initial_c")
    y0 = tf.zeros((n_batch, n_hidden), dtype=tf.float32, name="initial_h")
    start = tf.constant(0, name="start")
    step = tf.constant(1, name="step")
    print("inputs:", inputs, "shape:", inputs.shape)
    out, _, _, final_cell_state = op(inputs, weights, y0, c0, index, start, step)
    print("out:", out)
    assert isinstance(out, tf.Tensor)
    assert_equal(out.shape.as_list(), [None, None, n_hidden])


def test_NativeLstm2_shape_inference_unknown_rank():
  op = make_op(native_op.NativeLstm2, compiler_opts={"verbose": True})
  n_hidden = 3
  with tf_compat.v1.variable_scope("test_NativeLstm2_shape_inference_unknown_rank"):
    weights = tf_compat.v1.get_variable(name="W_re", shape=(n_hidden, n_hidden * 4))
    inputs = tf_compat.v1.placeholder(tf.float32, name="inputs")
    index = tf.reduce_sum(inputs, axis=2)
    n_batch = tf.shape(inputs)[1]
    c0 = tf.zeros((n_batch, n_hidden), dtype=tf.float32, name="initial_c")
    y0 = tf.zeros((n_batch, n_hidden), dtype=tf.float32, name="initial_h")
    start = tf.constant(0, name="start")
    step = tf.constant(1, name="step")
    print("inputs:", inputs, "shape:", inputs.shape)
    out, _, _, final_cell_state = op(inputs, weights, y0, c0, index, start, step)
    print("out:", out)
    assert isinstance(out, tf.Tensor)
    assert_equal(out.shape.as_list(), [None, None, n_hidden])


def test_NativeLstm2_0len_run():
  from pprint import pprint
  from returnn.util.basic import describe_tensorflow_version
  print("TensorFlow:", describe_tensorflow_version())
  n_time = 0
  n_batch = 1
  n_hidden = 3
  with tf_compat.v1.Session() as session:
    with tf_compat.v1.variable_scope("test_NativeLstm2_0len_run"):
      cell = NativeLstm2(n_hidden=n_hidden)
      inputs = tf.zeros([n_time, n_batch, n_hidden * 4])
      index = tf.ones([n_time, n_batch])
      outputs, final_state = cell(inputs, index)
      session.run(tf_compat.v1.global_variables_initializer())
      res = session.run(outputs)
      pprint(res)


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


def pure_tf_unrolled_lstm(x, h_0, c_0, mask, W_f, W_r, b, n_time, n_batch, n_in_dim, n_cells, start=0, step=1, name="ref_lstm"):
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
  """
  Will call the op (e.g. native_lstm2) with default start=0, step=1,
  and we do the striding/slicing via standard TF functions (e.g. x[start_::step]),
  as well as reverse the strided slice of the output.

  :param op:
  """
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
  return wrap_lstm_slice_start_step(op=pure_tf_unrolled_lstm, name="%s_slice" % name, **kwargs)


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
  x = tf.convert_to_tensor(x, name="x")
  h_0 = tf.convert_to_tensor(h_0, name="h_0")
  c_0 = tf.convert_to_tensor(c_0, name="c_0")
  mask = tf.convert_to_tensor(mask, name="mask")
  W_f = tf.convert_to_tensor(W_f, name="W_f")
  W_r = tf.convert_to_tensor(W_r, name="W_r")
  b = tf.convert_to_tensor(b, name="b")
  x.set_shape(tf.TensorShape((n_time, n_batch, n_in_dim)))
  h_0.set_shape(tf.TensorShape((n_batch, n_cells)))
  c_0.set_shape(tf.TensorShape((n_batch, n_cells)))
  mask.set_shape(tf.TensorShape((n_time, n_batch)))
  W_f.set_shape(tf.TensorShape((n_in_dim, n_cells * 4)))
  W_r.set_shape(tf.TensorShape((n_cells, n_cells * 4)))
  b.set_shape(tf.TensorShape((n_cells * 4,)))
  op = make_op(native_op.NativeLstm2)
  from returnn.tf.util.basic import dot, expand_multiple_dims
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


def check_lstm_ops(op1, op2, name1, name2, rtol=1e-6, **kwargs):
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
  """
  :return: kwargs for check_lstm_ops. some dummy input
  :rtype: dict[str]
  """
  kwargs = {"n_time": 3, "n_batch": 2, "n_in_dim": 1, "n_cells": 2,
            "mask": numpy.array([[1, 1, 0], [1, 0, 0]]).astype("float32").transpose()}
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
  check_lstm_op_start_step(op=pure_tf_unrolled_lstm, name="ref_lstm", **kwargs)


def test_native_lstm2_start_step_impl():
  kwargs = lstm_kwargs()
  check_lstm_op_start_step(op=native_lstm2, name="native_lstm2", **kwargs)


def test_native_lstm2_impl():
  kwargs = lstm_kwargs()
  check_lstm_ops(
    op1=pure_tf_unrolled_lstm, op2=native_lstm2, name1="ref_lstm", name2="native_lstm2",
    rtol=1e-6,
    **kwargs)


def lstm_grad_kwargs():
  """
  :return: kwargs for check_lstm_grad_ops, some dummy input
  :rtype: dict[str]
  """
  def gen(shape, offset, factor=3.):
    return numpy.sin(numpy.arange(numpy.prod(shape)) * factor + offset).reshape(shape).astype("float32")
  kwargs = lstm_kwargs()
  mask_bc = kwargs["mask"][:, :, None]
  assert mask_bc.shape == (kwargs["n_time"], kwargs["n_batch"], 1)
  kwargs["dy"] = gen((kwargs["n_time"], kwargs["n_batch"], kwargs["n_cells"]), offset=0.15) * mask_bc
  kwargs["dd"] = gen((kwargs["n_batch"], kwargs["n_cells"]), offset=17)
  return kwargs


def wrap_lstm_grad(op, x, h_0, c_0, dy, dd, mask, W_f, W_r, b, n_time, n_batch, n_in_dim, n_cells, start, step, name):
  """
  Uses the LSTM op (for the forward path), and tf.gradients to get the gradients.

  :param op: lstm, e.g. pure_tf_unrolled_lstm or native_lstm2
  """
  assert n_time > 0
  assert 0 <= start < n_time
  assert step != 0
  x = tf.convert_to_tensor(x, name="x")
  h_0 = tf.convert_to_tensor(h_0, name="h_0")
  c_0 = tf.convert_to_tensor(c_0, name="c_0")
  with tf.name_scope("gradients"):
    # Note that tensor_array_grad._GetGradSource() has this ugly hack
    # which requires that we have the "gradients" prefix.
    dy = tf.identity(tf.convert_to_tensor(dy, name="dy"), name="dy")
    dd = tf.identity(tf.convert_to_tensor(dd, name="dd"), name="dd")
  mask = tf.convert_to_tensor(mask, name="mask")
  W_f = tf.convert_to_tensor(W_f, name="W_f")
  W_r = tf.convert_to_tensor(W_r, name="W_r")
  b = tf.convert_to_tensor(b, name="b")
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


def check_lstm_grad_ops_single(op1, op2, name1, name2, dy, dd, rtol=5e-5, exclude=(), **kwargs):
  dy = tf.convert_to_tensor(dy, name="dy")
  dd = tf.convert_to_tensor(dd, name="dd")
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
  not_all_close = []
  nan_tensors = []
  for k in ["y", "d", "dx", "dh0", "dc0", "dWf", "dWr", "db"]:
    if k in exclude:
      continue
    v1 = locals()["v%s1" % k]
    v2 = locals()["v%s2" % k]
    if k in ["y"]:  # not for dx because the grad should be zero for masked frames
      v1 = (v1 * vmask)[start_::step]
      v2 = (v2 * vmask)[start_::step]
    print("check", k)
    if not numpy.allclose(v1, v2, rtol=rtol, equal_nan=True):
      print("ERROR, not all close %s1 vs %s2" % (k, k))
      print("%s1:" % k)
      print(v1)
      print("%s2:" % k)
      print(v2)
      not_all_close.append(k)
      if numpy.isnan(v1).any():
        nan_tensors.append(locals()[k + "1"])
      if numpy.isnan(v2).any():
        nan_tensors.append(locals()[k + "2"])
  assert isinstance(dWr1, tf.Tensor) and isinstance(dWr1, tf.Tensor)
  print("dWr1 op:", dWr1.op.name, dWr1.op.type)
  print("dWr2 op:", dWr2.op.name, dWr2.op.type)
  if dWr1 in nan_tensors:
    example_dWr = dWr1
  elif dWr2 in nan_tensors:
    example_dWr = dWr2
  else:
    example_dWr = dWr1
  print("example dWr op:", example_dWr.op.name, example_dWr.op.type)
  if not_all_close:
    print("not all close (%r). print some debug info." % (not_all_close,))
    if nan_tensors:
      print("Have nan tensors:", nan_tensors)
      from returnn.tf.util.basic import add_check_numerics_ops
      check_op = add_check_numerics_ops(nan_tensors)
      try:
        session.run(nan_tensors + [check_op])
      except tf.errors.OpError as exc:
        print("As expected, got TF exception with add_check_numerics_ops:")
        print(exc)
    print("graph of %s:" % example_dWr.name)
    from returnn.tf.util.basic import print_graph_output
    print_graph_output(example_dWr)
    from returnn.extern import graph_editor
    all_ops = graph_editor.get_backward_walk_ops(
      [y1, dWr1, y2, dWr2, example_dWr], inclusive=True, stop_at_ts=[dy, dd])
    print("all relevant ops:")
    pprint(all_ops)
  v_op_ins, v_op_outs, vdWr1_, vdWr2_ = session.run(
    [list(example_dWr.op.inputs), list(example_dWr.op.outputs), dWr1, dWr2])
  if not numpy.allclose(vdWr1_, vdWr2_, rtol=rtol, equal_nan=True):
    not_all_close.append("dWr (extra run)")
  if not_all_close:
    print("inputs:")
    for x, v in zip(example_dWr.op.inputs, v_op_ins):
      print("%s:" % x)
      print(v)
    print("outputs:")
    for x, v in zip(example_dWr.op.outputs, v_op_outs):
      print("%s:" % x)
      print(v)
    print("dWr1:")
    print(vdWr1_)
    print("dWr2:")
    print(vdWr2_)
  for i in range(5):  # run multiple times. maybe this triggers an exception
    v_op_outs_direct, vdWr1_, vdWr2_ = session.run(
      [list(example_dWr.op.outputs), dWr1, dWr2], {x: v for (x, v) in zip(example_dWr.op.inputs, v_op_ins)})
    for x, v, v_ in zip(example_dWr.op.outputs, v_op_outs_direct, v_op_outs):
      if not numpy.allclose(v, v_, rtol=rtol):
        print("! mismatch for %s direct-vs-def, run %i" % (x, i))
        for x_, v__ in zip(example_dWr.op.inputs, v_op_ins):
          print("direct input", x_, "=")
          print(v__)
        not_all_close.append("%s direct-vs-def, run %i" % (x, i))
      if not_all_close:
        print("output direct %s:" % x)
        print(v)
        print("output def:")
        print(v_)
    assert_allclose(vdWr1_, vdWr2_, rtol=rtol, err_msg="mismatch for dWr (extra run %i)" % i)
  if not_all_close:
    print("raise exception now: not all close: %r" % (not_all_close,))
    raise Exception("not all close: %r" % (not_all_close,))


def check_lstm_grad_ops(name1, name2, **kwargs):
  for start, step in [(0, 1), (0, -1), (1, 1), (1, -1), (0, 2), (0, -2)]:
    print(">>> check_lstm_grad_ops", start, step)
    check_lstm_grad_ops_single(
      name1="%s_%i_%i" % (name1, start, step),
      name2="%s_%i_%i" % (name2, start, step),
      start=start, step=step,
      **kwargs)


def check_lstm_grad_start_step(op, name, **kwargs):
  """
  :param op: e.g. pure_tf_unrolled_lstm or native_lstm2
  :param str name: name for the op
  """
  name2 = "%s_slice" % name
  def wrapped_lstm_op(**kwargs):
    return wrap_lstm_slice_start_step(op=op, **kwargs)
  check_lstm_grad_ops(
    op1=op, name1=name, op2=wrapped_lstm_op, name2=name2, **kwargs)


def test_ref_lstm_grad_start_step():
  kwargs = lstm_grad_kwargs()
  check_lstm_grad_start_step(op=pure_tf_unrolled_lstm, name="ref_lstm", **kwargs)


def test_native_lstm2_grad_start_step():
  kwargs = lstm_grad_kwargs()
  check_lstm_grad_start_step(
    op=native_lstm2, name="native_lstm2", rtol=1e-5, **kwargs)


def test_native_lstm2_grad():
  kwargs = lstm_grad_kwargs()
  check_lstm_grad_ops(
    op1=pure_tf_unrolled_lstm, name1="ref_lstm", op2=native_lstm2, name2="native_lstm2",
    rtol=1e-5, **kwargs)


def dummy_lstm_op(x, h_0, c_0, mask, W_f, W_r, b, n_time, n_batch, n_in_dim, n_cells, start, step, name):
  x = tf.convert_to_tensor(x)
  x.set_shape(tf.TensorShape((n_time, n_batch, n_in_dim)))
  # Have gradients for all.
  y = tf.reduce_mean(x, axis=2, keepdims=True)
  y += tf.zeros((n_time, n_batch, n_cells))
  y *= tf.reduce_mean(W_f)
  y *= tf.reduce_mean(W_r)
  y += tf.reduce_mean(b)
  return y, y, h_0 + c_0


@unittest.skipIf(
  not tf_util.have_min_tf_version((1, 5)), "TF 1.3.0 bug: https://github.com/tensorflow/tensorflow/issues/13355")
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
  not tf_util.have_min_tf_version((1, 5)), "TF 1.3.0 bug: https://github.com/tensorflow/tensorflow/issues/13355")
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


def pure_tf_chunk(x, index, chunk_size, chunk_step):
  """
  :param tf.Tensor x: (time,batch,dim)
  :param tf.Tensor index:
  :param int|tf.Tensor chunk_size:
  :param int|tf.Tensor chunk_step:
  :return: out, oindex.
    out is of shape (chunk_size, n_batch * n_chunks, n_dim), oindex of shape (chunk_size, n_batch * n_chunks).
  :rtype: (tf.Tensor,tf.Tensor)
  """
  from returnn.tf.util.basic import windowed_nd, get_shape_dim
  assert x.get_shape().ndims == 3
  n_time = get_shape_dim(x, 0)
  n_batch = get_shape_dim(x, 1)
  n_dim = get_shape_dim(x, 2)
  if all([isinstance(obj, int) for obj in [n_time, chunk_size, chunk_step]]):
    max_func = max
  else:
    max_func = tf.maximum
  n_chunks = max_func(n_time - chunk_size + chunk_step - 1, 0) // chunk_step + 1

  x_win = windowed_nd(
    x, window_size=chunk_size, window_left=0, time_axis=0, new_window_axis=0)  # (chunk_size,time,batch,dim)
  out = x_win[:, ::chunk_step]  # (chunk_size,n_chunks+...,batch,dim)
  out = out[:, :n_chunks]  # (chunk_size,n_chunks,batch,dim)
  out = tf.transpose(out, [0, 2, 1, 3])
  out = tf.reshape(out, [chunk_size, n_chunks * n_batch, n_dim])

  i_win = windowed_nd(
    index, window_size=chunk_size, window_left=0, time_axis=0, new_window_axis=0)  # (chunk_size,time,batch)
  oindex = i_win[:, ::chunk_step]  # (chunk_size,n_chunks+...,batch)
  oindex = oindex[:, :n_chunks]  # (chunk_size,n_chunks,batch)
  oindex = tf.transpose(oindex, [0, 2, 1])
  oindex = tf.reshape(oindex, [chunk_size, n_chunks * n_batch])

  return out, oindex


def check_chunk(x, index, chunk_size, chunk_step):
  """
  :param tf.Tensor x: (time,batch,dim)
  :param tf.Tensor index:
  :param int|tf.Tensor chunk_size:
  :param int|tf.Tensor chunk_step:

  Expected return from op:
    out, oindex.
    out is of shape (chunk_size, n_batch * n_chunks, n_dim), oindex of shape (chunk_size, n_batch * n_chunks).
  """
  x = tf.convert_to_tensor(x)
  out1, oindex1 = pure_tf_chunk(x, index=index, chunk_size=chunk_size, chunk_step=chunk_step)
  dout = tf_compat.v1.random_normal(tf.shape(out1))
  dx1, = tf.gradients(ys=[out1], xs=[x], grad_ys=[dout])
  out2, oindex2 = chunk(x, index=index, chunk_size=chunk_size, chunk_step=chunk_step)
  dx2, = tf.gradients(ys=[out2], xs=[x], grad_ys=[dout])
  index, out1, oindex1, dx1, out2, oindex2, dx2 = session.run((index, out1, oindex1, dx1, out2, oindex2, dx2))
  time_, batch_, dim = out1.shape
  assert out1.shape == out2.shape == (time_, batch_, dim)
  assert oindex1.shape == oindex2.shape == (time_, batch_)
  for t in range(time_):
    for b in range(batch_):
      assert_almost_equal(oindex1[t, b], oindex2[t, b])
      if oindex1[t, b] < 0.5:
        continue
      assert_almost_equal(out1[t, b], out2[t, b], decimal=5)
  time, batch, _ = dx1.shape
  assert dx1.shape == dx2.shape == (time, batch, dim)
  assert index.shape == (time, batch)
  for t in range(time):
    for b in range(batch):
      if index[t, b] < 0.5:
        continue
      assert_almost_equal(dx1[t, b], dx2[t, b], decimal=5)


def test_chunk_simple():
  n_batch = 1
  n_time = 17
  n_dim = 3
  x = tf_compat.v1.random_normal((n_time, n_batch, n_dim))
  index = [[1.] * n_time] * n_batch
  index = tf.convert_to_tensor(index)
  index = tf.transpose(index)
  check_chunk(x=x, index=index, chunk_size=2, chunk_step=1)
  check_chunk(x=x, index=index, chunk_size=2, chunk_step=2)
  check_chunk(x=x, index=index, chunk_size=4, chunk_step=2)
  check_chunk(x=x, index=index, chunk_size=4, chunk_step=3)
  check_chunk(x=x, index=index, chunk_size=4, chunk_step=4)
  check_chunk(x=x, index=index, chunk_size=10, chunk_step=4)
  check_chunk(x=x, index=index, chunk_size=20, chunk_step=4)


def test_chunk():
  n_batch = 3
  n_time = 17
  n_dim = 5
  x = tf_compat.v1.random_normal((n_time, n_batch, n_dim))
  index = [[1.] * n_time] * n_batch
  index[-1][-1] = 0.
  index[-1][-2] = 0.
  index[-2][-1] = 0.
  index = tf.convert_to_tensor(index)
  index = tf.transpose(index)
  check_chunk(x=x, index=index, chunk_size=4, chunk_step=2)
  check_chunk(x=x, index=index, chunk_size=4, chunk_step=3)
  check_chunk(x=x, index=index, chunk_size=4, chunk_step=4)
  check_chunk(x=x, index=index, chunk_size=10, chunk_step=4)
  check_chunk(x=x, index=index, chunk_size=20, chunk_step=4)


def pure_tf_unchunk(x, index, chunk_size, chunk_step, n_time, n_batch):
  """
  :param tf.Tensor x: output e.g. from :func:`chunk`
  :param tf.Tensor index:
  :param int|tf.Tensor chunk_size:
  :param int|tf.Tensor chunk_step:
  :param tf.Tensor n_time:
  :param tf.Tensor n_batch:
  :return: out, oindex, ofactors
  :rtype: (tf.Tensor,tf.Tensor,tf.Tensor)
  """
  from returnn.tf.util.basic import where_bc
  assert x.get_shape().ndims == 3
  n_dim = tf.shape(x)[2]
  # Hack via tf.gradients.
  chunk_x_n = tf.cast(tf.reshape(tf.range(n_time * n_batch), [n_time, n_batch, 1]), dtype=x.dtype)
  chunk_idx0 = tf.ones((n_time, n_batch))
  chunk_y0_n, _ = pure_tf_chunk(chunk_x_n, index=chunk_idx0, chunk_size=chunk_size, chunk_step=chunk_step)
  dx0n, = tf.gradients(ys=[chunk_y0_n], xs=[chunk_x_n], grad_ys=[tf.expand_dims(index, axis=2)])
  chunk_idx = where_bc(tf.greater_equal(tf.squeeze(dx0n, axis=2), 1.0), 1.0, 0.0)
  chunk_y_n, _ = pure_tf_chunk(chunk_x_n, index=chunk_idx, chunk_size=chunk_size, chunk_step=chunk_step)
  dxn, = tf.gradients(ys=[chunk_y_n], xs=[chunk_x_n], grad_ys=[tf.ones_like(chunk_y_n)])
  factors = where_bc(tf.equal(chunk_idx, 1.), 1. / tf.squeeze(dxn, axis=2), 0.)
  chunk_x = tf.zeros((n_time, n_batch, n_dim))
  chunk_y, _ = pure_tf_chunk(chunk_x, index=chunk_idx, chunk_size=chunk_size, chunk_step=chunk_step)
  dx, = tf.gradients(ys=[chunk_y], xs=[chunk_x], grad_ys=[x])
  return dx * tf.expand_dims(factors, axis=2), chunk_idx, factors


def check_unchunk(x, index, chunk_size, chunk_step):
  """
  :param tf.Tensor x: (time,batch,dim)
  :param tf.Tensor index:
  :param int|tf.Tensor chunk_size:
  :param int|tf.Tensor chunk_step:

  Expected return from op:
    out, oindex.
    out is of shape (chunk_size, n_batch * n_chunks, n_dim), oindex of shape (chunk_size, n_batch * n_chunks).
  """
  n_time, n_batch = tf.shape(x)[0], tf.shape(x)[1]
  x, index = chunk(x, index, chunk_size=chunk_size, chunk_step=chunk_step)
  x = tf_compat.v1.random_normal(tf.shape(x))
  out1, oindex1, ofactors1 = pure_tf_unchunk(
    x, index=index, chunk_size=chunk_size, chunk_step=chunk_step, n_time=n_time, n_batch=n_batch)
  dout = tf_compat.v1.random_normal(tf.shape(out1))
  dx1, = tf.gradients(ys=[out1], xs=[x], grad_ys=[dout])
  out2, oindex2, ofactors2 = unchunk(
    x, index=index, chunk_size=chunk_size, chunk_step=chunk_step, n_time=n_time, n_batch=n_batch)
  dx2, = tf.gradients(ys=[out2], xs=[x], grad_ys=[dout])
  index, out1, oindex1, ofactors1, dx1, out2, oindex2, ofactors2, dx2 = session.run(
    (index, out1, oindex1, ofactors1, dx1, out2, oindex2, ofactors2, dx2))
  time_, batch_, dim = out1.shape
  assert out1.shape == out2.shape == (time_, batch_, dim)
  assert oindex1.shape == ofactors1.shape == oindex2.shape == ofactors2.shape == (time_, batch_)
  for t in range(time_):
    for b in range(batch_):
      assert_almost_equal(oindex1[t, b], oindex2[t, b])
      if oindex1[t, b] < 0.5:
        continue
      assert_almost_equal(ofactors1[t, b], ofactors2[t, b])
      assert_almost_equal(out1[t, b], out2[t, b], decimal=5)
  time, batch, _ = dx1.shape
  assert dx1.shape == dx2.shape == (time, batch, dim)
  assert index.shape == (time, batch)
  for t in range(time):
    for b in range(batch):
      if index[t, b] < 0.5:
        continue
      assert_almost_equal(dx1[t, b], dx2[t, b], decimal=5)


def test_unchunk():
  n_batch = 3
  n_time = 17
  n_dim = 5
  x = tf_compat.v1.random_normal((n_time, n_batch, n_dim))
  index = [[1.] * n_time] * n_batch
  index[-1][-1] = 0.
  index[-1][-2] = 0.
  index[-2][-1] = 0.
  index = tf.convert_to_tensor(index)
  index = tf.transpose(index)
  check_unchunk(x=x, index=index, chunk_size=4, chunk_step=2)
  check_unchunk(x=x, index=index, chunk_size=4, chunk_step=3)
  check_unchunk(x=x, index=index, chunk_size=4, chunk_step=4)
  check_unchunk(x=x, index=index, chunk_size=10, chunk_step=4)
  check_unchunk(x=x, index=index, chunk_size=20, chunk_step=4)


def _py_baum_welch(am_scores, float_idx, edges, weights, start_end_states):
  """
  Pure Python Forward-backward (Baum Welch) algorithm.
  The parameters are in the same format as our native fast_baum_welch op.

  :param numpy.ndarray am_scores: (time, batch, dim), in -log space
  :param numpy.ndarray float_idx: (time, batch) -> 0 or 1 (index mask, via seq lens)
  :param numpy.ndarray edges: (4,num_edges), edges of the graph (from,to,emission_idx,sequence_idx)
  :param numpy.ndarray weights: (num_edges,), weights of the edges
  :param numpy.ndarray start_end_states: (2, batch), (start,end) state idx in FSA. there is only one single FSA.
  :return: (fwdbwd, obs_scores), fwdbwd is (time, batch, dim), obs_scores is (time, batch), in -log space
  :rtype: (numpy.ndarray, numpy.ndarray)
  """
  # We get it in -log space, but we calculate in +log space.
  am_scores = -am_scores
  weights = -weights
  n_time, n_batch, dim = am_scores.shape
  assert float_idx.shape == (n_time, n_batch)
  assert edges.ndim == 2 and weights.ndim == 1
  n_edges, = weights.shape
  assert edges.shape == (4, n_edges)
  assert start_end_states.shape == (2, n_batch)
  from collections import defaultdict
  from scipy.special import logsumexp
  zero_score = float("-inf")
  fwdbwd = numpy.zeros((n_time, n_batch, dim), dtype=am_scores.dtype) + zero_score
  obs_scores = numpy.zeros((n_time, n_batch), dtype=am_scores.dtype) + zero_score

  def collect_scores(forward):
    """
    :param bool forward:
    :rtype: list[dict[int,float]]
    """
    start_idx, end_idx = start_end_states[:, sequence_idx]
    states = defaultdict(lambda: zero_score)  # type: typing.Dict[int,float]  # state-idx -> score.
    states[start_idx if forward else end_idx] = 0.0
    scores_over_t = [None] * (n_time + 1)  # type: typing.List[typing.Optional[typing.Dict[int,float]]]
    scores_over_t[0 if forward else -1] = dict(states)
    for t in (range(n_time) if forward else reversed(range(n_time))):
      if float_idx[t, sequence_idx] == 1:
        scores = defaultdict(list)  # type: typing.Dict[int,typing.List[float]]  # state-idx -> list[score]
        for edge_idx in range(n_edges):
          from_idx, to_idx, emission_idx, sequence_idx_ = edges[:, edge_idx]
          if not forward:
            from_idx, to_idx = to_idx, from_idx
          if sequence_idx_ != sequence_idx:
            continue
          if from_idx not in states or states[from_idx] == zero_score:
            continue
          assert 0 <= emission_idx < dim
          score = states[from_idx] + weights[edge_idx] + am_scores[t, sequence_idx, emission_idx]
          scores[to_idx].append(score)
        states.clear()
        for state_idx in scores.keys():
          states[state_idx] = float(logsumexp(scores[state_idx]))
      scores_over_t[(t + 1) if forward else t] = dict(states)
    return scores_over_t

  def gamma():
    """
    :return: nothing, fill fwdbwd and obs_scores
    """
    for t in range(n_time):
      if float_idx[t, sequence_idx] == 1:
        scores = defaultdict(list)  # type: typing.Dict[int,typing.List[float]]  # emission-idx -> list[score]
        all_scores = []  # type: typing.List[float]
        for edge_idx in range(n_edges):
          from_idx, to_idx, emission_idx, sequence_idx_ = edges[:, edge_idx]
          if sequence_idx_ != sequence_idx:
            continue
          assert 0 <= emission_idx < dim
          if from_idx not in fwd_scores[t]:
            continue
          if to_idx not in bwd_scores[t + 1]:
            continue
          score = (
            fwd_scores[t][from_idx] +
            weights[edge_idx] + am_scores[t, sequence_idx, emission_idx] +
            bwd_scores[t + 1][to_idx])
          scores[emission_idx].append(score)
          all_scores.append(score)
        obs_scores[t, sequence_idx] = logsumexp(all_scores) if all_scores else zero_score
        for emission_idx, values in scores.items():
          if not values:
            fwdbwd[t, sequence_idx, emission_idx] = zero_score
          else:
            fwdbwd[t, sequence_idx, emission_idx] = float(logsumexp(values)) - obs_scores[t, sequence_idx]

  for sequence_idx in range(n_batch):
    fwd_scores = collect_scores(forward=True)
    bwd_scores = collect_scores(forward=False)
    gamma()

  # -log space
  return -fwdbwd, -obs_scores


def test_py_baum_welch():
  n_batch = 3
  seq_len = 7
  n_classes = 5
  from returnn.util.fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  for i in range(n_classes):
    fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
    fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
  assert n_classes <= seq_len
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = fast_bw_fsa.edges
  weights = fast_bw_fsa.weights
  start_end_states = fast_bw_fsa.start_end_states
  am_scores = numpy.ones((seq_len, n_batch, n_classes), dtype="float32") * numpy.float32(1.0 / n_classes)
  am_scores = -numpy.log(am_scores)  # in -log space
  float_idx = numpy.ones((seq_len, n_batch), dtype="float32")
  print("Construct call...")
  fwdbwd, obs_scores = _py_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("Done.")
  print("score:")
  print(repr(obs_scores))
  assert_equal(obs_scores.shape, (seq_len, n_batch))
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
    assert_allclose(obs_scores, 8.55801582, rtol=1e-5)  # should be the same everywhere
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


def test_FastBaumWelch():
  print("Make op...")
  op = make_fast_baum_welch_op(compiler_opts=dict(verbose=True))  # will be cached, used inside :func:`fast_baum_welch`
  print("Op:", op)
  n_batch = 3
  seq_len = 5
  n_classes = 10
  from returnn.util.fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  fsa.add_inf_loop(state_idx=0, num_emission_labels=n_classes)
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = tf.constant(fast_bw_fsa.edges, dtype=tf.int32)
  weights = tf.constant(fast_bw_fsa.weights, dtype=tf.float32)
  start_end_states = tf.constant(fast_bw_fsa.start_end_states, dtype=tf.int32)
  am_scores_np = numpy.random.RandomState(42).normal(size=(seq_len, n_batch, n_classes)).astype("float32")
  am_scores = tf.constant(am_scores_np, dtype=tf.float32)  # in -log space
  float_idx_np = numpy.ones((seq_len, n_batch), dtype="float32")
  float_idx = tf.ones((seq_len, n_batch), dtype=tf.float32)
  print("Construct call...")
  fwdbwd, obs_scores = fast_baum_welch(
    am_scores=am_scores, float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("Done.")
  print("Eval:")
  fwdbwd_np, score = session.run([fwdbwd, obs_scores])
  print("score:", score)
  print("Baum-Welch soft alignment:")
  print(repr(fwdbwd_np))
  fwdbwd_np2, score2 = _py_baum_welch(
    am_scores=am_scores_np, float_idx=float_idx_np,
    edges=fast_bw_fsa.edges, weights=fast_bw_fsa.weights, start_end_states=fast_bw_fsa.start_end_states)
  print("ref score:", score2)
  print("ref Baum-Welch soft alignment:")
  print(repr(fwdbwd_np2))
  numpy.testing.assert_allclose(score, score2, rtol=1e-5)
  numpy.testing.assert_allclose(fwdbwd_np, fwdbwd_np2, rtol=1e-5)


def test_fast_bw_uniform():
  print("Make op...")
  op = make_fast_baum_welch_op(compiler_opts=dict(verbose=True))  # will be cached, used inside :func:`fast_baum_welch`
  # args: (am_scores, edges, weights, start_end_states, float_idx, state_buffer)
  print("Op:", op)
  n_batch = 3
  seq_len = 7
  n_classes = 5
  from returnn.util.fsa import FastBwFsaShared
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
  # from returnn.tf.util.basic import sequence_mask_time_major
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


def get_ctc_fsa_fast_bw_via_python(targets, seq_lens, blank_idx):
  """
  :param tf.Tensor targets: shape (batch,time)
  :param tf.Tensor seq_lens: shape (batch,)
  :param int blank_idx:
  :return: edges, weights, start_end_states
  :rtype: (tf.Tensor, tf.Tensor, tf.Tensor)
  """
  from returnn.util.fsa import get_ctc_fsa_fast_bw

  def py_fast_bw_fsa_ctc_wrapper(targets_, seq_lens_):
    """
    :param numpy.ndarray targets_:
    :param numpy.ndarray seq_lens_:
    :rtype: (numpy.ndarray,numpy.ndarray,numpy.ndarray)
    """
    fsa = get_ctc_fsa_fast_bw(targets=targets_, seq_lens=seq_lens_, blank_idx=blank_idx)
    assert fsa.start_end_states.shape == (2, len(seq_lens_)), "shape mismatch %r, n_batch %r, seq lens %r" % (
      fsa.start_end_states.shape, len(seq_lens_), seq_lens_)
    return fsa.edges.astype("int32"), fsa.weights.astype("float32"), fsa.start_end_states.astype("int32")

  edges, weights, start_end_states = tf_compat.v1.py_func(
    py_fast_bw_fsa_ctc_wrapper,
    [targets, seq_lens],
    [tf.int32, tf.float32, tf.int32],
    stateful=False)
  # edges: (4, num_edges), edges of the graph (from,to,emission_idx,sequence_idx)
  # weights: (num_edges,), weights of the edges
  # start_end_states: (2, batch), (start,end) state idx in automaton.
  edges.set_shape((4, None))
  weights.set_shape((None,))
  start_end_states.set_shape((2, None))
  return edges, weights, start_end_states


def ctc_loss_via_python_fsa(logits, logits_seq_lens, time_major, targets, targets_seq_lens):
  """
  Similar to :func:`tf.nn.ctc_loss`.
  We use our :func:`fast_baum_welch`.
  Also see :class:`FastBaumWelchLoss`.

  :param tf.Tensor logits: (time,batch,dim) or (batch,time,dim). unnormalized (before softmax)
  :param tf.Tensor logits_seq_lens: shape (batch,) of int32|int64
  :param bool time_major:
  :param tf.Tensor targets: batch-major, [batch,time]
  :param tf.Tensor targets_seq_lens: (batch,)
  :return: loss, shape (batch,)
  :rtype: tf.Tensor
  """
  assert logits.get_shape().ndims == 3 and logits.get_shape().dims[-1].value
  dim = logits.get_shape().dims[-1].value
  if not time_major:
    logits = tf.transpose(logits, [1, 0, 2])  # (time,batch,dim)
  log_sm = tf.nn.log_softmax(logits)  # (time,batch,dim)
  from returnn.tf.util.basic import sequence_mask_time_major
  seq_mask = sequence_mask_time_major(logits_seq_lens)  # (time,batch)

  edges, weights, start_end_states = get_ctc_fsa_fast_bw_via_python(
    targets=targets, seq_lens=targets_seq_lens, blank_idx=dim - 1)
  fwdbwd, obs_scores = fast_baum_welch(
    am_scores=-log_sm, float_idx=seq_mask,
    edges=edges, weights=weights, start_end_states=start_end_states)
  loss = obs_scores[0]  # (batch,)
  bw = tf.exp(-fwdbwd)  # (time,batch,dim)
  grad_x = (tf.exp(log_sm) - bw) * tf.expand_dims(seq_mask, 2)  # (time,batch,dim)
  from returnn.tf.util.basic import custom_gradient
  loss = custom_gradient.generic_loss_and_error_signal(loss=loss, x=logits, grad_x=grad_x)
  return loss


def _log_softmax(x, axis=-1):
  assert isinstance(x, numpy.ndarray)
  xdev = x - x.max(axis=axis, keepdims=True)
  lsm = xdev - numpy.log(numpy.sum(numpy.exp(xdev), axis=axis, keepdims=True))
  return lsm


def check_ctc_fsa(targets, target_seq_lens, n_classes, with_native_fsa=False, label_loop=True):
  """
  :param numpy.ndarray targets:
  :param numpy.ndarray target_seq_lens:
  :param int n_classes:
  :param bool with_native_fsa:
  :param bool label_loop: ctc_merge_repeated in tf.nn.ctc_loss
  :return: nothing, just checks
  """
  n_batch, n_target_time = targets.shape
  assert n_batch == len(target_seq_lens) and n_target_time == max(target_seq_lens)
  n_time = n_target_time * 3
  # am_scores are logits, unnormalized, i.e. the values before softmax.
  am_scores = numpy.random.RandomState(42).normal(size=(n_time, n_batch, n_classes)).astype("float32")
  # am_scores = numpy.zeros((n_time, n_batch, n_classes), dtype="float32")
  int_idx = numpy.zeros((n_time, n_batch), dtype="int32")
  seq_lens = numpy.array([
    n_time,
    max(n_time - 4, (target_seq_lens[1] if (len(target_seq_lens) >= 2) else 0) + 1, 1),
    max(n_time - 5, 1), max(n_time - 5, 1)],
    dtype="int32")[:n_batch]
  for t in range(n_time):
    int_idx[t] = t < seq_lens
  float_idx = int_idx.astype("float32")
  blank_idx = n_classes - 1

  targets_tf = tf.constant(targets)
  targets_seq_lens_tf = tf.constant(target_seq_lens)

  if label_loop:
    import returnn.util.fsa as fsa
    fsa = fsa.get_ctc_fsa_fast_bw(targets=targets, seq_lens=target_seq_lens, blank_idx=blank_idx)
    assert fsa.start_end_states.shape == (2, len(target_seq_lens))
    edges = fsa.edges.astype("int32")
    weights = fsa.weights.astype("float32")
    start_end_states = fsa.start_end_states.astype("int32")
    if with_native_fsa:
      print("python edges:")
      print(edges)
      print("python start_end_states:")
      print(start_end_states)
  else:
    import returnn.tf.native_op as tf_native_op
    native_edges_tf, native_weights_tf, native_start_end_states_tf = tf_native_op.get_ctc_fsa_fast_bw(
      targets=targets_tf, seq_lens=targets_seq_lens_tf, blank_idx=blank_idx, label_loop=label_loop)
    edges, weights, start_end_states = session.run(
      (native_edges_tf, native_weights_tf, native_start_end_states_tf))

  fwdbwd, obs_scores = _py_baum_welch(
    am_scores=-_log_softmax(am_scores), float_idx=float_idx,
    edges=edges, weights=weights, start_end_states=start_end_states)
  fwdbwd = numpy.exp(-fwdbwd)  # -log space -> prob space
  print(fwdbwd)
  print(obs_scores)

  if with_native_fsa:
    import returnn.tf.native_op as tf_native_op
    native_edges_tf, native_weights_tf, native_start_end_states_tf = tf_native_op.get_ctc_fsa_fast_bw(
      targets=targets_tf, seq_lens=targets_seq_lens_tf, blank_idx=blank_idx)
    native_edges, native_weights, native_start_end_states = session.run(
      (native_edges_tf, native_weights_tf, native_start_end_states_tf))
    # Note: The native FSA vs the Python FSA are not exactly identical
    # (they just should be equivalent; although almost identical).
    # We introduce a dummy state (last before end state), and some dummy edges.
    print("native edges:")
    print(native_edges)
    print("native_start_end_states:")
    print(native_start_end_states)

    native_fwdbwd, native_obs_scores = _py_baum_welch(
      am_scores=-_log_softmax(am_scores), float_idx=float_idx,
      edges=native_edges, weights=native_weights, start_end_states=native_start_end_states)
    native_fwdbwd = numpy.exp(-native_fwdbwd)  # -log space -> prob space
    print(native_fwdbwd)
    print(native_obs_scores)
    for b in range(n_batch):
      for t in range(seq_lens[b]):
        numpy.testing.assert_almost_equal(fwdbwd[t, b], native_fwdbwd[t, b], decimal=5)
    for b in range(n_batch):
      numpy.testing.assert_almost_equal(obs_scores[b], native_obs_scores[b], decimal=5)
    fwdbwd = native_fwdbwd
    obs_scores = native_obs_scores

  from returnn.tf.util.basic import sparse_labels
  targets_sparse_tf = sparse_labels(targets_tf, targets_seq_lens_tf)
  am_scores_tf = tf.constant(am_scores)
  seq_lens_tf = tf.constant(seq_lens)
  # inputs are unnormalized. tf.nn.ctc_loss does softmax internally.
  ref_ctc_loss_tf = tf_compat.v1.nn.ctc_loss(
    labels=targets_sparse_tf,
    inputs=am_scores_tf, sequence_length=seq_lens_tf, time_major=True, ctc_merge_repeated=label_loop)
  # See grad definition of CTCLoss.
  # The op will calculate the gradient w.r.t. the logits (log softmax).
  # I.e. with y = softmax(z), this is \partial loss / \partial z = y - soft_align.
  # Also see CtcLoss.get_soft_alignment.
  ref_ctc_loss_grad_tf = ref_ctc_loss_tf.op.outputs[1]  # time major, i.e. (time, batch, dim)
  y_tf = tf.nn.softmax(am_scores)  # (time, batch, dim)
  soft_align_tf = y_tf - ref_ctc_loss_grad_tf
  soft_align_tf.set_shape(tf.TensorShape((None, None, n_classes)))
  ref_fwdbwd, ref_obs_score = session.run((soft_align_tf, ref_ctc_loss_tf))
  print(ref_fwdbwd)
  print(ref_obs_score)

  for b in range(n_batch):
    for t in range(seq_lens[b]):
      numpy.testing.assert_almost_equal(fwdbwd[t, b], ref_fwdbwd[t, b], decimal=5)
  for b in range(n_batch):
    numpy.testing.assert_almost_equal(obs_scores[0, b], ref_obs_score[b], decimal=5)


def test_ctc_fsa_batch3_len6_c8():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [1, 3, 4, 2, 1, 0],
    [2, 6, 3, 4, 0, 0],
    [0, 3, 2, 0, 0, 0]], dtype="int32")
  target_seq_lens = numpy.array([6, 4, 3], dtype="int32")
  n_classes = 8  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes)


def test_ctc_fsa_batch1_len2():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [0, 1]], dtype="int32")
  target_seq_lens = numpy.array([2], dtype="int32")
  n_classes = 3  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes)


def test_ctc_fsa_batch1_len1():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [0]], dtype="int32")
  target_seq_lens = numpy.array([1], dtype="int32")
  n_classes = 2  # +1 because of blank

  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes)


def test_ctc_fsa_batch3_len6_c8_native():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [1, 3, 4, 2, 1, 0],
    [2, 6, 3, 4, 0, 0],
    [0, 3, 2, 0, 0, 0]], dtype="int32")
  target_seq_lens = numpy.array([6, 4, 3], dtype="int32")
  n_classes = 8  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch4_len6_c8_native():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [1, 2, 4, 4, 1, 0],
    [2, 6, 3, 4, 0, 0],
    [3, 3, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0]], dtype="int32")
  target_seq_lens = numpy.array([6, 4, 2, 1], dtype="int32")
  n_classes = 8  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch4_len6_c8_native_no_loop():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  No label loop (ctc_merge_repeated False) is equivalent to the Recurrent Neural Aligner (RNA) topology.
  """
  targets = numpy.array([
    [1, 2, 4, 4, 1, 0],
    [2, 6, 3, 4, 0, 0],
    [3, 3, 0, 0, 0, 0],
    [5, 0, 0, 0, 0, 0]], dtype="int32")
  target_seq_lens = numpy.array([6, 4, 2, 1], dtype="int32")
  n_classes = 8  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, label_loop=False)


def test_ctc_fsa_batch2_len2a():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [0, 1],
    [1, 0]], dtype="int32")
  target_seq_lens = numpy.array([2, 1], dtype="int32")
  n_classes = 3  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch2_len2():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [0, 1],
    [0, 0]], dtype="int32")
  target_seq_lens = numpy.array([2, 2], dtype="int32")
  n_classes = 3  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch2_len1():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [0],
    [1]], dtype="int32")
  target_seq_lens = numpy.array([1, 1], dtype="int32")
  n_classes = 3  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch1_len2rep_native():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [0, 0]], dtype="int32")
  target_seq_lens = numpy.array([2], dtype="int32")
  n_classes = 2  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch1_len2_native():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [0, 1]], dtype="int32")
  target_seq_lens = numpy.array([2], dtype="int32")
  n_classes = 3  # +1 because of blank
  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def test_ctc_fsa_batch1_len1_native():
  """
  This (:func:`Fsa.get_ctc_fsa_fast_bw`) is used by :func:`ctc_loss`.
  """
  targets = numpy.array([
    [0]], dtype="int32")
  target_seq_lens = numpy.array([1], dtype="int32")
  n_classes = 2  # +1 because of blank

  check_ctc_fsa(targets=targets, target_seq_lens=target_seq_lens, n_classes=n_classes, with_native_fsa=True)


def _py_viterbi(am_scores, am_seq_len, edges, weights, start_end_states):
  """
  Pure Python Viterbi algorithm, to find the best path/alignment.
  The parameters are in the same format as our native fast_baum_welch op.

  :param numpy.ndarray am_scores: (time, batch, dim), in +log space
  :param numpy.ndarray am_seq_len: (batch,) -> int32
  :param numpy.ndarray edges: (4,num_edges), edges of the graph (from,to,emission_idx,sequence_idx)
  :param numpy.ndarray weights: (num_edges,), weights of the edges
  :param numpy.ndarray start_end_states: (2, batch), (start,end) state idx in FSA. there is only one single FSA.
  :return: (alignment, obs_scores), alignment is (time, batch), obs_scores is (batch,), in +log space
  :rtype: (numpy.ndarray, numpy.ndarray)
  """
  n_time, n_batch, dim = am_scores.shape
  assert am_seq_len.shape == (n_batch,)
  assert edges.ndim == 2 and weights.ndim == 1
  n_edges, = weights.shape
  assert edges.shape == (4, n_edges)
  assert start_end_states.shape == (2, n_batch)
  from collections import defaultdict
  zero_score = float("-inf")
  alignment = numpy.zeros((n_time, n_batch), dtype="int32")
  obs_scores = numpy.zeros((n_batch,), dtype=am_scores.dtype) + zero_score

  def search():
    """
    :rtype: list[dict[int,(float,int)]]
    """
    start_idx, _ = start_end_states[:, sequence_idx]
    states = defaultdict(lambda: (zero_score, -1))  # type: typing.Dict[int,typing.Tuple[float,int]]  # state-idx -> score/edge  # nopep8
    states[start_idx] = (0.0, -1)
    res = []  # type: typing.List[typing.Dict[int,typing.Tuple[float,int]]]
    for t in range(n_time):
      if t >= am_seq_len[sequence_idx]:
        break
      scores = defaultdict(list)  # type: typing.Dict[int,typing.List[typing.Tuple[float,int]]]  # state-idx -> list[score/edge]  # nopep8
      for edge_idx in range(n_edges):
        from_idx, to_idx, emission_idx, sequence_idx_ = edges[:, edge_idx]
        if sequence_idx_ != sequence_idx:
          continue
        if from_idx not in states or states[from_idx] == zero_score:
          continue
        assert 0 <= emission_idx < dim
        score = states[from_idx][0] + weights[edge_idx] + am_scores[t, sequence_idx, emission_idx]
        scores[to_idx].append((score, edge_idx))
      states.clear()
      for state_idx in scores.keys():
        states[state_idx] = max(scores[state_idx], key=lambda _item: (_item[0], -_item[1]))
      res.append(dict(states))
    assert len(res) == am_seq_len[sequence_idx]
    return res

  def select_best():
    """
    :return: nothing, fill alignment and obs_scores
    """
    _, end_idx = start_end_states[:, sequence_idx]
    state_idx = end_idx
    for t in reversed(range(am_seq_len[sequence_idx])):
      if state_idx not in fwd_search_res[t]:  # no path?
        alignment[t, sequence_idx] = 0
        continue
      score, edge_idx = fwd_search_res[t][state_idx]
      if t == am_seq_len[sequence_idx] - 1:
        obs_scores[sequence_idx] = score
      from_idx, to_idx, emission_idx, sequence_idx_ = edges[:, edge_idx]
      assert sequence_idx_ == sequence_idx
      alignment[t, sequence_idx] = emission_idx
      state_idx = from_idx

  for sequence_idx in range(n_batch):
    fwd_search_res = search()
    select_best()

  return alignment, obs_scores


def test_py_viterbi():
  n_batch = 3
  seq_len = 7
  n_classes = 5
  from returnn.util.fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  for i in range(n_classes):
    fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
    fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
  assert n_classes <= seq_len
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = fast_bw_fsa.edges
  weights = fast_bw_fsa.weights
  start_end_states = fast_bw_fsa.start_end_states
  am_scores = numpy.eye(n_classes, n_classes, dtype="float32")  # (dim,dim)
  import scipy.ndimage
  am_scores = scipy.ndimage.zoom(am_scores, zoom=(float(seq_len) / n_classes, 1), order=1, prefilter=False)
  assert am_scores.shape == (seq_len, n_classes)
  am_scores = am_scores[:, None]
  am_scores = am_scores + numpy.zeros((seq_len, n_batch, n_classes), dtype="float32")
  print(am_scores[:, 0])
  # am_scores = numpy.ones((seq_len, n_batch, n_classes), dtype="float32") * numpy.float32(1.0 / n_classes)
  am_scores = numpy.log(am_scores)  # in +log space
  print("Construct call...")
  alignment, obs_scores = _py_viterbi(
    am_scores=am_scores, am_seq_len=numpy.array([seq_len] * n_batch),
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("Done.")
  print("score:")
  print(repr(obs_scores))
  assert_equal(obs_scores.shape, (n_batch,))
  print("Hard alignment:")
  print(repr(alignment))
  assert_equal(alignment.shape, (seq_len, n_batch))
  if seq_len == n_classes:
    print("Extra check identity...")
    for i in range(n_batch):
      for t in range(seq_len):
        assert alignment[t, i] == t
  if seq_len == 7 and n_classes == 5:
    print("Extra check ref_align (7,5)...")
    assert_allclose(obs_scores, -1.6218603, rtol=1e-5)  # should be the same everywhere
    for i in range(n_batch):
      assert_equal(alignment[:, i].tolist(), [0, 1, 1, 2, 3, 3, 4])
  print("Done.")


def test_fast_viterbi():
  n_batch = 3
  seq_len = 7
  n_classes = 5
  from returnn.util.fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  for i in range(n_classes):
    fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
    fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
  assert n_classes <= seq_len
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = fast_bw_fsa.edges
  weights = fast_bw_fsa.weights
  start_end_states = fast_bw_fsa.start_end_states
  am_scores = numpy.eye(n_classes, n_classes, dtype="float32")  # (dim,dim)
  import scipy.ndimage
  am_scores = scipy.ndimage.zoom(am_scores, zoom=(float(seq_len) / n_classes, 1), order=1, prefilter=False)
  assert am_scores.shape == (seq_len, n_classes)
  am_scores = am_scores[:, None]
  am_scores = am_scores + numpy.zeros((seq_len, n_batch, n_classes), dtype="float32")
  print(am_scores[:, 0])
  # am_scores = numpy.ones((seq_len, n_batch, n_classes), dtype="float32") * numpy.float32(1.0 / n_classes)
  am_scores = numpy.log(am_scores)  # in +log space
  print("Construct call...")
  alignment, obs_scores = fast_viterbi(
    am_scores=tf.constant(am_scores), am_seq_len=tf.constant(numpy.array([seq_len] * n_batch, dtype="int32")),
    edges=tf.constant(edges), weights=tf.constant(weights), start_end_states=tf.constant(start_end_states))
  alignment, obs_scores = session.run((alignment, obs_scores))
  print("Done.")
  print("score:")
  print(repr(obs_scores))
  assert_equal(obs_scores.shape, (n_batch,))
  print("Hard alignment:")
  print(repr(alignment))
  assert_equal(alignment.shape, (seq_len, n_batch))
  if seq_len == n_classes:
    print("Extra check identity...")
    for i in range(n_batch):
      for t in range(seq_len):
        assert alignment[t, i] == t
  if seq_len == 7 and n_classes == 5:
    print("Extra check ref_align (7,5)...")
    assert_allclose(obs_scores, -1.6218603, rtol=1e-5)  # should be the same everywhere
    for i in range(n_batch):
      assert_equal(alignment[:, i].tolist(), [0, 1, 1, 2, 3, 3, 4])
  print("Done.")


def test_fast_viterbi_rnd():
  n_batch = 4
  seq_len = 23
  n_classes = 5
  from returnn.util.fsa import FastBwFsaShared
  fsa = FastBwFsaShared()
  for i in range(n_classes):
    fsa.add_edge(i, i + 1, emission_idx=i)  # fwd
    fsa.add_edge(i + 1, i + 1, emission_idx=i)  # loop
  assert n_classes <= seq_len
  fast_bw_fsa = fsa.get_fast_bw_fsa(n_batch=n_batch)
  edges = fast_bw_fsa.edges
  weights = fast_bw_fsa.weights
  start_end_states = fast_bw_fsa.start_end_states
  am_scores = numpy.random.RandomState(42).normal(size=(seq_len, n_batch, n_classes)).astype("float32")
  am_seq_len = numpy.array([seq_len] * n_batch, dtype="int32")
  am_seq_len[1] -= 1
  am_seq_len[-1] -= 2
  am_seq_len[-2] = max(n_classes - 1, 1)  # no path possible
  ref_alignment, ref_scores = _py_viterbi(
    am_scores=am_scores, am_seq_len=am_seq_len,
    edges=edges, weights=weights, start_end_states=start_end_states)
  print("ref score:")
  print(repr(ref_scores))
  assert_equal(ref_scores.shape, (n_batch,))
  print("ref hard alignment:")
  print(repr(ref_alignment))
  assert_equal(ref_alignment.shape, (seq_len, n_batch))
  print("Construct fast_viterbi call...")
  alignment, scores = fast_viterbi(
    am_scores=tf.constant(am_scores), am_seq_len=tf.constant(am_seq_len),
    edges=tf.constant(edges), weights=tf.constant(weights), start_end_states=tf.constant(start_end_states))
  alignment, scores = session.run((alignment, scores))
  print("Done.")
  print("score:")
  print(repr(scores))
  assert_equal(scores.shape, (n_batch,))
  print("Hard alignment:")
  print(repr(alignment))
  assert_equal(alignment.shape, (seq_len, n_batch))
  assert_allclose(scores, ref_scores, rtol=1e-5)
  assert_allclose(alignment, ref_alignment, rtol=1e-5)
  print("Done.")


def test_ctc_viterbi_loss():
  n_batch = 3
  seq_len = 13
  n_input_dim = 6
  n_classes = 5

  x = tf.constant(numpy.random.RandomState(42).normal(size=(seq_len, n_batch, n_input_dim)).astype("float32"))
  x_seq_len = tf.constant([seq_len, seq_len - 1, seq_len - 2])
  weights = tf_compat.v1.get_variable(
    "ctc_viterbi_weights", shape=(n_input_dim, n_classes), initializer=tf.random_normal_initializer())
  bias = tf_compat.v1.get_variable("ctc_viterbi_bias", shape=(n_classes,))
  var_list = [weights, bias]
  session.run(tf_compat.v1.initialize_variables(var_list))
  from returnn.tf.util.basic import dot
  logits = dot(x, weights) + bias
  targets = tf.constant([[0, 1, 2, 0, 0], [3, 2, 4, 1, 1], [2, 0, 1, 2, 0]])
  targets.set_shape((n_batch, None))
  targets_seq_len = tf.constant([3, 5, 4])
  targets_seq_len.set_shape((n_batch,))
  loss = ctc_loss_viterbi(
    logits=logits, logits_seq_lens=x_seq_len, logits_time_major=True,
    targets=targets, targets_seq_lens=targets_seq_len)
  loss.set_shape((n_batch,))
  loss = tf.reduce_mean(loss)
  opt = tf_compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
  minimize = opt.minimize(loss, var_list=var_list)
  loss_vals = []
  for step in range(10):
    loss_val, _ = session.run((loss, minimize))
    print("step %i, loss %f" % (step, loss_val))
    loss_vals.append(loss_val)
  assert loss_vals[-1] < loss_vals[0]


def test_edit_distance():
  rnd = numpy.random.RandomState(42)
  n_batch = 15
  n_a_max_len = 13
  n_b_max_len = 11
  num_classes = 10
  a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
  b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
  a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
  b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
  # Likely some high error. So make some explicit examples.
  expected_results = [None] * n_batch
  i = 0
  # One insertion/deletion.
  a_np[i, :1] = [1]
  a_len_np[i] = 1
  b_len_np[i] = 0
  expected_results[i] = 1
  i += 1
  # One deletion.
  a_np[i, :2] = [1, 2]
  b_np[i, :1] = [1]
  a_len_np[i] = 2
  b_len_np[i] = 1
  expected_results[i] = 1
  i += 1
  # One substitution + deletion.
  a_np[i, :2] = [1, 2]
  b_np[i, :1] = [3]
  a_len_np[i] = 2
  b_len_np[i] = 1
  expected_results[i] = 2
  i += 1
  # One substitution error.
  a_np[i, :4] = [1, 2, 3, 4]
  b_np[i, :4] = [1, 2, 4, 4]
  a_len_np[i] = 4
  b_len_np[i] = 4
  expected_results[i] = 1
  i += 1
  # One deletion error.
  a_np[i, :6] = [1, 2, 3, 3, 4, 5]
  b_np[i, :5] = [1, 2, 3, 4, 5]
  a_len_np[i] = 6
  b_len_np[i] = 5
  expected_results[i] = 1
  i += 1
  # One insertion error.
  a_np[i, :6] = [1, 2, 3, 4, 5, 6]
  b_np[i, :7] = [1, 2, 3, 4, 4, 5, 6]
  a_len_np[i] = 6
  b_len_np[i] = 7
  expected_results[i] = 1
  i += 1
  # Same.
  a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
  b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
  a_len_np[i] = 11
  b_len_np[i] = 11
  expected_results[i] = 0
  i += 1
  # Both full length. Error should be 2.
  a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
  b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
  a_len_np[i] = n_a_max_len
  b_len_np[i] = n_b_max_len
  expected_results[i] = 2
  i += 1
  assert n_batch - i >= 5  # still some random left
  a = tf.constant(a_np)
  b = tf.constant(b_np)
  a_len = tf.constant(a_len_np)
  b_len = tf.constant(b_len_np)
  from returnn.tf.util.basic import sparse_labels
  for i in range(n_batch):
    print("testing batch", i, "/", n_batch)
    _a = a[i:i + 1, :a_len_np[i]]
    _a_len = a_len[i:i + 1]
    _b = b[i:i + 1, :b_len_np[i]]
    _b_len = b_len[i:i + 1]
    print("seq a:", a_np[i, :a_len_np[i]])
    print("seq b:", b_np[i, :b_len_np[i]])
    _a_sparse = sparse_labels(_a, _a_len)
    _b_sparse = sparse_labels(_b, _b_len)
    _tf_edit_dist = tf.edit_distance(_a_sparse, _b_sparse, normalize=False)
    _native_edit_dist = edit_distance(_a, _a_len, _b, _b_len)
    tf_edit_dist_np, native_edit_dist_np = session.run((_tf_edit_dist, _native_edit_dist))
    assert isinstance(tf_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("TF edit dist:", tf_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results[i])
    assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[0] == native_edit_dist_np[0]
    else:
      assert tf_edit_dist_np[0] == native_edit_dist_np[0]
    print("swapped:")
    _tf_edit_dist = tf.edit_distance(_b_sparse, _a_sparse, normalize=False)
    _native_edit_dist = edit_distance(_b, _b_len, _a, _a_len)
    tf_edit_dist_np, native_edit_dist_np = session.run((_tf_edit_dist, _native_edit_dist))
    assert isinstance(tf_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("TF edit dist:", tf_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results[i])
    assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[0] == native_edit_dist_np[0]
    else:
      assert tf_edit_dist_np[0] == native_edit_dist_np[0]
    print()
  print("Now the whole batch.")
  a_sparse = sparse_labels(a, a_len)
  b_sparse = sparse_labels(b, b_len)
  tf_edit_dist = tf.edit_distance(a_sparse, b_sparse, normalize=False)
  native_edit_dist = edit_distance(a, a_len, b, b_len)
  tf_edit_dist_np, native_edit_dist_np = session.run((tf_edit_dist, native_edit_dist))
  assert isinstance(tf_edit_dist_np, numpy.ndarray)
  assert isinstance(native_edit_dist_np, numpy.ndarray)
  print("TF edit dist:", tf_edit_dist_np)
  print("Native edit dist:", native_edit_dist_np)
  print("Expected edit dist:", expected_results)
  assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
  for i in range(n_batch):
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[i] == native_edit_dist_np[i]
    else:
      assert tf_edit_dist_np[i] == native_edit_dist_np[i]
  print()
  print("Now the whole batch, flipped.")
  tf_edit_dist = tf.edit_distance(b_sparse, a_sparse, normalize=False)
  native_edit_dist = edit_distance(b, b_len, a, a_len)
  tf_edit_dist_np, native_edit_dist_np = session.run((tf_edit_dist, native_edit_dist))
  assert isinstance(tf_edit_dist_np, numpy.ndarray)
  assert isinstance(native_edit_dist_np, numpy.ndarray)
  print("TF edit dist:", tf_edit_dist_np)
  print("Native edit dist:", native_edit_dist_np)
  print("Expected edit dist:", expected_results)
  assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
  for i in range(n_batch):
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[i] == native_edit_dist_np[i]
    else:
      assert tf_edit_dist_np[i] == native_edit_dist_np[i]


_wrap_tf_edit_distance_global_placeholders = None


def _wrap_tf_edit_distance(a, b):
  """
  :param list[int] a:
  :param list[int] b:
  :rtype: int
  """
  global _wrap_tf_edit_distance_global_placeholders
  if not _wrap_tf_edit_distance_global_placeholders:
    with tf.name_scope("wrap_tf_edit_distance"):
      a_tf = tf_compat.v1.placeholder(tf.int32, shape=(None,), name="a")
      b_tf = tf_compat.v1.placeholder(tf.int32, shape=(None,), name="b")
      _wrap_tf_edit_distance_global_placeholders = [a_tf, b_tf]
      a_len_tf = tf.convert_to_tensor([tf.shape(a_tf)[0]])
      b_len_tf = tf.convert_to_tensor([tf.shape(b_tf)[0]])
      from returnn.tf.util.basic import sparse_labels
      a_tf = tf.expand_dims(a_tf, axis=0)
      b_tf = tf.expand_dims(b_tf, axis=0)
      a_tf = sparse_labels(a_tf, a_len_tf)
      b_tf = sparse_labels(b_tf, b_len_tf)
      res_tf = tf.edit_distance(a_tf, b_tf, normalize=False)
      res_tf = tf.squeeze(res_tf, axis=0)
      _wrap_tf_edit_distance_global_placeholders.append(res_tf)
  a_tf, b_tf, res_tf = _wrap_tf_edit_distance_global_placeholders
  res = session.run(res_tf, feed_dict={a_tf: a, b_tf: b})
  return int(res)


def test_wrap_tf_edit_distance():
  assert_equal(_wrap_tf_edit_distance([1], []), 1)
  assert_equal(_wrap_tf_edit_distance([1, 2], [1]), 1)
  assert_equal(_wrap_tf_edit_distance([2, 2], [1]), 2)
  assert_equal(_wrap_tf_edit_distance([2, 1], [1]), 1)
  assert_equal(_wrap_tf_edit_distance([2, 1], [1, 1]), 1)
  assert_equal(_wrap_tf_edit_distance([2, 1], [1, 1, 1]), 2)
  assert_equal(_wrap_tf_edit_distance([2, 1], [2, 1, 1]), 1)


def _naive_optimal_completion_edit_distance(a, b):
  """
  :param list[int] a: prefix
  :param list[int] b:
  :rtype: int
  """
  distances = [_wrap_tf_edit_distance(a, b[:n]) for n in range(len(b) + 1)]
  return min(distances)


def test_optimal_completion_edit_distance():
  rnd = numpy.random.RandomState(42)
  n_batch = 15
  n_a_max_len = 11
  n_b_max_len = 13
  num_classes = 10
  a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
  b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
  a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
  b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
  # Likely some high error. So make some explicit examples.
  expected_results = [None] * n_batch
  i = 0
  # One deletion.
  a_np[i, :1] = [1]
  a_len_np[i] = 1
  b_len_np[i] = 0
  expected_results[i] = 1
  i += 1
  # One optional insertion.
  a_np[i, :1] = [1]
  b_np[i, :2] = [1, 2]
  a_len_np[i] = 1
  b_len_np[i] = 2
  expected_results[i] = 0
  i += 1
  # One substitution or deletion.
  a_np[i, :1] = [1]
  b_np[i, :2] = [3, 1]
  a_len_np[i] = 1
  b_len_np[i] = 2
  expected_results[i] = 1
  i += 1
  # One substitution error.
  a_np[i, :4] = [1, 2, 3, 4]
  b_np[i, :4] = [1, 2, 4, 4]
  a_len_np[i] = 4
  b_len_np[i] = 4
  expected_results[i] = 1
  i += 1
  # One insertion error.
  a_np[i, :5] = [1, 2, 3, 4, 5]
  b_np[i, :6] = [1, 2, 3, 3, 4, 5]
  a_len_np[i] = 5
  b_len_np[i] = 6
  expected_results[i] = 1
  i += 1
  # Same.
  a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
  b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
  a_len_np[i] = 11
  b_len_np[i] = 11
  expected_results[i] = 0
  i += 1
  # Both full length.
  a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
  b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
  a_len_np[i] = n_a_max_len
  b_len_np[i] = n_b_max_len
  expected_results[i] = 0
  i += 1
  assert n_batch - i >= 5  # still some random left
  a = tf.constant(a_np)
  b = tf.constant(b_np)
  a_len = tf.constant(a_len_np)
  b_len = tf.constant(b_len_np)
  for i in range(n_batch):
    print("testing batch", i, "/", n_batch)
    _a = a[i:i + 1, :a_len_np[i]]
    _a_len = a_len[i:i + 1]
    _b = b[i:i + 1, :b_len_np[i]]
    _b_len = b_len[i:i + 1]
    print("seq a:", a_np[i, :a_len_np[i]])
    print("seq b:", b_np[i, :b_len_np[i]])
    _native_edit_dist = optimal_completion_edit_distance(_a, _a_len, _b, _b_len)
    native_edit_dist_np = session.run(_native_edit_dist)
    tf_edit_dist_np = numpy.array([
      _naive_optimal_completion_edit_distance(a_np[i, :a_len_np[i]], b_np[i, :b_len_np[i]])])
    assert isinstance(tf_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("TF edit dist:", tf_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results[i])
    assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[0] == native_edit_dist_np[0]
    else:
      assert tf_edit_dist_np[0] == native_edit_dist_np[0]
    print()
  print("Now the whole batch.")
  native_edit_dist = optimal_completion_edit_distance(a, a_len, b, b_len)
  native_edit_dist_np = session.run(native_edit_dist)
  tf_edit_dist_np = numpy.array([
    _naive_optimal_completion_edit_distance(a_np[i, :a_len_np[i]], b_np[i, :b_len_np[i]])
    for i in range(n_batch)])
  assert isinstance(tf_edit_dist_np, numpy.ndarray)
  assert isinstance(native_edit_dist_np, numpy.ndarray)
  print("TF edit dist:", tf_edit_dist_np)
  print("Native edit dist:", native_edit_dist_np)
  print("Expected edit dist:", expected_results)
  assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
  for i in range(n_batch):
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[i] == native_edit_dist_np[i]
    else:
      assert tf_edit_dist_np[i] == native_edit_dist_np[i]
  print()


def test_optimal_completion_edit_distance_per_successor():
  rnd = numpy.random.RandomState(42)
  n_batch = 15
  n_a_max_len = 11
  n_b_max_len = 13
  num_classes = 10
  a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
  b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
  a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
  b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
  # Likely some high error. So make some explicit examples.
  expected_results = [None] * n_batch
  i = 0
  # One deletion.
  a_np[i, :1] = [1]
  a_len_np[i] = 1
  b_len_np[i] = 0
  expected_results[i] = 1
  i += 1
  # One optional insertion.
  a_np[i, :1] = [1]
  b_np[i, :2] = [1, 2]
  a_len_np[i] = 1
  b_len_np[i] = 2
  expected_results[i] = 0
  i += 1
  # One substitution or deletion.
  a_np[i, :1] = [1]
  b_np[i, :2] = [3, 1]
  a_len_np[i] = 1
  b_len_np[i] = 2
  expected_results[i] = 1
  i += 1
  # One substitution error.
  a_np[i, :4] = [1, 2, 3, 4]
  b_np[i, :4] = [1, 2, 4, 4]
  a_len_np[i] = 4
  b_len_np[i] = 4
  expected_results[i] = 1
  i += 1
  # One insertion error.
  a_np[i, :5] = [1, 2, 3, 4, 5]
  b_np[i, :6] = [1, 2, 3, 3, 4, 5]
  a_len_np[i] = 5
  b_len_np[i] = 6
  expected_results[i] = 1
  i += 1
  # Same.
  a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 1, 3]
  b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 1, 3]
  a_len_np[i] = 11
  b_len_np[i] = 11
  expected_results[i] = 0
  i += 1
  # Both full length.
  a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
  b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
  a_len_np[i] = n_a_max_len
  b_len_np[i] = n_b_max_len
  expected_results[i] = 0
  i += 1
  assert n_batch - i >= 5  # still some random left
  a = tf.constant(a_np)
  b = tf.constant(b_np)
  assert all(a_len_np > 0)
  a_len = tf.constant(a_len_np)
  b_len = tf.constant(b_len_np)
  print("Now the whole batch.")
  # a_len - 1 such that we can do the check below.
  native_edit_dist = optimal_completion_edit_distance_per_successor(a, a_len - 1, b, b_len, num_classes)
  native_edit_dist_np = session.run(native_edit_dist)
  tf_edit_dist_np = numpy.array([
    _naive_optimal_completion_edit_distance(a_np[i, :a_len_np[i]], b_np[i, :b_len_np[i]])
    for i in range(n_batch)])
  assert isinstance(tf_edit_dist_np, numpy.ndarray)
  assert isinstance(native_edit_dist_np, numpy.ndarray)
  print("TF edit dist:", tf_edit_dist_np)
  print("Native edit dist:", native_edit_dist_np)
  print("Expected edit dist:", expected_results)
  assert tf_edit_dist_np.shape == (n_batch,)
  assert native_edit_dist_np.shape == (n_batch, num_classes)
  for i in range(n_batch):
    a_last = a_np[i, a_len_np[i] - 1]
    assert 0 <= a_last < num_classes
    native_res = native_edit_dist_np[i, a_last]
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[i] == native_res
    else:
      assert tf_edit_dist_np[i] == native_res
    for j in range(num_classes):
      tf_res = _naive_optimal_completion_edit_distance(list(a_np[i, :a_len_np[i] - 1]) + [j], b_np[i, :b_len_np[i]])
      native_res = native_edit_dist_np[i, j]
      assert tf_res == native_res
  print()


def test_next_edit_distance_row():
  rnd = numpy.random.RandomState(42)
  n_batch = 15
  n_a_max_len = 13
  n_b_max_len = 11
  num_classes = 10
  a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
  b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
  a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
  b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
  # Likely some high error. So make some explicit examples.
  expected_results = [None] * n_batch
  i = 0
  # One insertion/deletion.
  a_np[i, :1] = [1]
  a_len_np[i] = 1
  b_len_np[i] = 0
  expected_results[i] = 1
  i += 1
  # One deletion.
  a_np[i, :2] = [1, 2]
  b_np[i, :1] = [1]
  a_len_np[i] = 2
  b_len_np[i] = 1
  expected_results[i] = 1
  i += 1
  # One substitution + deletion.
  a_np[i, :2] = [1, 2]
  b_np[i, :1] = [3]
  a_len_np[i] = 2
  b_len_np[i] = 1
  expected_results[i] = 2
  i += 1
  # One substitution error.
  a_np[i, :4] = [1, 2, 3, 4]
  b_np[i, :4] = [1, 2, 4, 4]
  a_len_np[i] = 4
  b_len_np[i] = 4
  expected_results[i] = 1
  i += 1
  # One deletion error.
  a_np[i, :6] = [1, 2, 3, 3, 4, 5]
  b_np[i, :5] = [1, 2, 3, 4, 5]
  a_len_np[i] = 6
  b_len_np[i] = 5
  expected_results[i] = 1
  i += 1
  # One insertion error.
  a_np[i, :6] = [1, 2, 3, 4, 5, 6]
  b_np[i, :7] = [1, 2, 3, 4, 4, 5, 6]
  a_len_np[i] = 6
  b_len_np[i] = 7
  expected_results[i] = 1
  i += 1
  # Same.
  a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
  b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
  a_len_np[i] = 11
  b_len_np[i] = 11
  expected_results[i] = 0
  i += 1
  # Both full length. Error should be 2.
  a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
  b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
  a_len_np[i] = n_a_max_len
  b_len_np[i] = n_b_max_len
  expected_results[i] = 2
  i += 1
  assert n_batch - i >= 5  # still some random left
  a = tf.constant(a_np)
  b = tf.constant(b_np)
  a_len = tf.constant(a_len_np)
  b_len = tf.constant(b_len_np)
  from returnn.tf.util.basic import sparse_labels
  for i in range(n_batch):
    print("testing batch", i, "/", n_batch)
    _a = a[i:i + 1, :a_len_np[i]]
    _a_len = a_len[i:i + 1]
    _b = b[i:i + 1, :b_len_np[i]]
    _b_len = b_len[i:i + 1]
    print("seq a:", a_np[i, :a_len_np[i]])
    print("seq b:", b_np[i, :b_len_np[i]])
    _a_sparse = sparse_labels(_a, _a_len)
    _b_sparse = sparse_labels(_b, _b_len)
    _tf_edit_dist = tf.edit_distance(_a_sparse, _b_sparse, normalize=False)
    _native_edit_dist = edit_distance_via_next_edit_distance_row(_a, _a_len, _b, _b_len)
    tf_edit_dist_np, native_edit_dist_np = session.run((_tf_edit_dist, _native_edit_dist))
    assert isinstance(tf_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("TF edit dist:", tf_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results[i])
    assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[0] == native_edit_dist_np[0]
    else:
      assert tf_edit_dist_np[0] == native_edit_dist_np[0]
    print("swapped:")
    _tf_edit_dist = tf.edit_distance(_b_sparse, _a_sparse, normalize=False)
    _native_edit_dist = edit_distance_via_next_edit_distance_row(_b, _b_len, _a, _a_len)
    tf_edit_dist_np, native_edit_dist_np = session.run((_tf_edit_dist, _native_edit_dist))
    assert isinstance(tf_edit_dist_np, numpy.ndarray)
    assert isinstance(native_edit_dist_np, numpy.ndarray)
    print("TF edit dist:", tf_edit_dist_np)
    print("Native edit dist:", native_edit_dist_np)
    print("Expected edit dist:", expected_results[i])
    assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (1,)
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[0] == native_edit_dist_np[0]
    else:
      assert tf_edit_dist_np[0] == native_edit_dist_np[0]
    print()
  print("Now the whole batch.")
  a_sparse = sparse_labels(a, a_len)
  b_sparse = sparse_labels(b, b_len)
  tf_edit_dist = tf.edit_distance(a_sparse, b_sparse, normalize=False)
  native_edit_dist = edit_distance_via_next_edit_distance_row(a, a_len, b, b_len)
  tf_edit_dist_np, native_edit_dist_np = session.run((tf_edit_dist, native_edit_dist))
  assert isinstance(tf_edit_dist_np, numpy.ndarray)
  assert isinstance(native_edit_dist_np, numpy.ndarray)
  print("TF edit dist:", tf_edit_dist_np)
  print("Native edit dist:", native_edit_dist_np)
  print("Expected edit dist:", expected_results)
  assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
  for i in range(n_batch):
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[i] == native_edit_dist_np[i]
    else:
      assert tf_edit_dist_np[i] == native_edit_dist_np[i]
  print()
  print("Now the whole batch, flipped.")
  tf_edit_dist = tf.edit_distance(b_sparse, a_sparse, normalize=False)
  native_edit_dist = edit_distance_via_next_edit_distance_row(b, b_len, a, a_len)
  tf_edit_dist_np, native_edit_dist_np = session.run((tf_edit_dist, native_edit_dist))
  assert isinstance(tf_edit_dist_np, numpy.ndarray)
  assert isinstance(native_edit_dist_np, numpy.ndarray)
  print("TF edit dist:", tf_edit_dist_np)
  print("Native edit dist:", native_edit_dist_np)
  print("Expected edit dist:", expected_results)
  assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
  for i in range(n_batch):
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[i] == native_edit_dist_np[i]
    else:
      assert tf_edit_dist_np[i] == native_edit_dist_np[i]


def test_next_edit_distance_row_optimal_completion():
  rnd = numpy.random.RandomState(42)
  n_batch = 15
  n_a_max_len = 11
  n_b_max_len = 13
  num_classes = 10
  a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
  b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
  a_len_np = rnd.randint(1, n_a_max_len + 1, size=(n_batch,), dtype="int32")
  b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
  # Likely some high error. So make some explicit examples.
  expected_results = [None] * n_batch
  i = 0
  # One deletion.
  a_np[i, :1] = [1]
  a_len_np[i] = 1
  b_len_np[i] = 0
  expected_results[i] = 1
  i += 1
  # One optional insertion.
  a_np[i, :1] = [1]
  b_np[i, :2] = [1, 2]
  a_len_np[i] = 1
  b_len_np[i] = 2
  expected_results[i] = 0
  i += 1
  # One substitution or deletion.
  a_np[i, :1] = [1]
  b_np[i, :2] = [3, 1]
  a_len_np[i] = 1
  b_len_np[i] = 2
  expected_results[i] = 1
  i += 1
  # One substitution error.
  a_np[i, :4] = [1, 2, 3, 4]
  b_np[i, :4] = [1, 2, 4, 4]
  a_len_np[i] = 4
  b_len_np[i] = 4
  expected_results[i] = 1
  i += 1
  # One insertion error.
  a_np[i, :5] = [1, 2, 3, 4, 5]
  b_np[i, :6] = [1, 2, 3, 3, 4, 5]
  a_len_np[i] = 5
  b_len_np[i] = 6
  expected_results[i] = 1
  i += 1
  # Same.
  a_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
  b_np[i, :11] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 11]
  a_len_np[i] = 11
  b_len_np[i] = 11
  expected_results[i] = 0
  i += 1
  # Both full length.
  a_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0]
  b_np[i] = [2, 2, 4, 4, 6, 6, 8, 8, 9, 10, 0, 0, 0]
  a_len_np[i] = n_a_max_len
  b_len_np[i] = n_b_max_len
  expected_results[i] = 0
  i += 1
  assert n_batch - i >= 5  # still some random left
  a = tf.constant(a_np)
  b = tf.constant(b_np)
  a_len = tf.constant(a_len_np)
  b_len = tf.constant(b_len_np)
  print("Now the whole batch.")
  native_edit_dist = edit_distance_via_next_edit_distance_row(a, a_len, b, b_len, optimal_completion=True)
  native_edit_dist_np = session.run(native_edit_dist)
  tf_edit_dist_np = numpy.array([
    _naive_optimal_completion_edit_distance(a_np[i, :a_len_np[i]], b_np[i, :b_len_np[i]])
    for i in range(n_batch)])
  assert isinstance(tf_edit_dist_np, numpy.ndarray)
  assert isinstance(native_edit_dist_np, numpy.ndarray)
  print("TF edit dist:", tf_edit_dist_np)
  print("Native edit dist:", native_edit_dist_np)
  print("Expected edit dist:", expected_results)
  assert tf_edit_dist_np.shape == native_edit_dist_np.shape == (n_batch,)
  for i in range(n_batch):
    if expected_results[i] is not None:
      assert expected_results[i] == tf_edit_dist_np[i] == native_edit_dist_np[i]
    else:
      assert tf_edit_dist_np[i] == native_edit_dist_np[i]
  print()


def test_next_edit_distance_reduce_optimal_completion():
  rnd = numpy.random.RandomState(42)
  n_batch = 15
  n_a_max_len = 7
  n_b_max_len = 13
  num_classes = 10
  a_np = rnd.randint(0, num_classes, size=(n_batch, n_a_max_len), dtype="int32")
  b_np = rnd.randint(0, num_classes, size=(n_batch, n_b_max_len), dtype="int32")
  # Test only for full length seqs in a.
  a_len_np = numpy.array([n_a_max_len] * n_batch, dtype="int32")
  b_len_np = rnd.randint(1, n_b_max_len + 1, size=(n_batch,), dtype="int32")
  a = tf.constant(a_np)
  b = tf.constant(b_np)
  assert all(a_len_np > 0)
  a_len = tf.constant(a_len_np)
  b_len = tf.constant(b_len_np)
  print("Now the whole batch.")
  # a_len - 1 such that we can do the check below.
  native_edit_dist = optimal_completion_edit_distance_per_successor_via_next_edit_distance(
    a, a_len, b, b_len, num_classes)
  native_edit_dist_np = session.run(native_edit_dist)
  assert isinstance(native_edit_dist_np, numpy.ndarray)
  print("Native edit dist:", native_edit_dist_np)
  assert native_edit_dist_np.shape == (n_batch, num_classes)
  for i in range(n_batch):
    for j in range(num_classes):
      tf_res = _naive_optimal_completion_edit_distance(list(a_np[i, :a_len_np[i]]) + [j], b_np[i, :b_len_np[i]])
      native_res = native_edit_dist_np[i, j]
      assert tf_res == native_res
  print()


#@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
#@unittest.skipIf(is_gpu_available() and get_available_gpu_min_compute_capability() < 3.5, "too low compute capability")
@unittest.skipIf(not have_blocksparse_requirements(), "do not have Blocksparse requirements")
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
  x = tf_compat.v1.placeholder(tf.float32, shape=[hidden_size, None])
  x_np = np.ones((hidden_size, minibatch_size), dtype='float32')

  # Initialize block-sparse weights
  w = tf_compat.v1.get_variable("w", bsmm.w_shape, dtype=tf.float32, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=3))

  # Block-sparse matrix multiplication
  y = bsmm(x, w)

  # Run
  print('init vars')
  session.run(tf_compat.v1.global_variables_initializer())
  print('blocksparse matmul')
  result = session.run(y, feed_dict={x: x_np})
  print(result)
  print('test')
  w_np = session.run(w)
  y_test = bsmm.fprop_test(x_np, w_np)
  print(y_test)
  i = numpy.argmax((y_test - result) ** 2)
  print('biggest diff at %i: %r vs %r' % (i, y_test.flatten()[i], result.flatten()[i]))
  assert_allclose(result, y_test, rtol=1e-2)  # rtol=1e-03 still fails


@unittest.skipIf(not have_blocksparse_requirements(), "do not have Blocksparse requirements")
def test_blocksparse_simple_identity():
  init_blocksparse()

  from blocksparse.matmul import BlocksparseMatMul
  import tensorflow as tf
  import numpy

  n_in = 64
  n_out = 32 * 32
  block_size = 32
  # Note: It seems everything less than 4 fails, as well as non-power-of-2.
  n_batch = 4

  # Create a dense sparsity pattern
  mask = numpy.ones((n_in // block_size, n_out // block_size), dtype=numpy.int32)
  # MatMul object
  bsmm = BlocksparseMatMul(mask, block_size=block_size, feature_axis=0, name="bsmm")
  # Input
  x_np = numpy.arange(n_in * n_batch, dtype=numpy.float32).reshape((n_in, n_batch)) + 1.0
  x = tf.constant(x_np, name='x')
  # Block-sparse weights
  w_np = bsmm.identity_init()()
  w = tf.constant(w_np, name="w")
  #for b in range(bsmm.blocks):
  #  cb, kb = bsmm.updat_list[b]
  #  print("block %i/%i, cb %i/%i, kb %i/%i" % (b, bsmm.blocks, cb, bsmm.KB, kb, bsmm.CB))
  # Block-sparse matrix multiplication
  y = bsmm(x, w)
  y.set_shape((n_out, n_batch))
  # Run
  result = session.run(y)
  print(result)
  print('L2:', numpy.sum(result ** 2))
  y_test = bsmm.fprop_test(x_np, w_np)
  print(y_test)
  i = numpy.argmax((y_test - result) ** 2)
  print('biggest diff at %i: %r vs %r' % (i, y_test.flatten()[i], result.flatten()[i]))
  assert_allclose(result, y_test, rtol=1e-2)


@unittest.skip('broken?')
@unittest.skipIf(not have_blocksparse_requirements(), "do not have Blocksparse requirements")
def test_blocksparse_simple_feature_axis1():
  init_blocksparse()

  from blocksparse.matmul import BlocksparseMatMul
  import tensorflow as tf
  import numpy

  n_in = 64
  n_out = 32 * 32
  block_size = 32
  n_batch = 4

  # Create a dense sparsity pattern
  mask = numpy.ones((n_in // block_size, n_out // block_size), dtype=numpy.int32)
  # MatMul object
  bsmm = BlocksparseMatMul(mask, block_size=block_size, feature_axis=1, name="bsmm")
  # Input
  x_np = numpy.arange(n_in * n_batch, dtype=numpy.float32).reshape((n_batch, n_in)) + 1.0
  x = tf.constant(x_np, name='x')
  # Block-sparse weights
  w_np = bsmm.identity_init()()
  w = tf.constant(w_np, name="w")
  # Block-sparse matrix multiplication
  y = bsmm(x, w)
  y.set_shape((n_batch, n_out))
  # Run
  result = session.run(y)
  print(result)
  print('L2:', numpy.sum(result ** 2))
  y_test = bsmm.fprop_test(x_np, w_np)
  print(y_test)
  assert_allclose(result, y_test)


def _run_rnnt(acts, labels, input_lengths, label_lengths,
              expected_costs, expected_grads, blank):
  from returnn.extern.HawkAaronWarpTransducer import rnnt_loss, is_checked_out
  if not is_checked_out():
    raise unittest.SkipTest("HawkAaronWarpTransducer not checked out?")
  assert_equal(acts.shape, expected_grads.shape)
  acts_t = tf.constant(acts)
  labels_t = tf.constant(labels)
  input_lengths_t = tf.constant(input_lengths)
  label_lengths_t = tf.constant(label_lengths)

  logits = tf.nn.log_softmax(acts_t)
  costs = rnnt_loss(logits, labels_t, input_lengths_t, label_lengths_t, blank)

  grads = tf.gradients(costs, [acts_t])[0]

  (tf_costs, tf_grad) = session.run([costs, grads])
  # Max absolute difference: 2.861023e-06
  # Max relative difference: 7.264356e-07
  assert_allclose(tf_costs, expected_costs, atol=1e-6, rtol=1e-6)
  assert_allclose(tf_grad, expected_grads, atol=1e-6, rtol=1e-6)


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_warprnnt_forward():
  from returnn.extern.HawkAaronWarpTransducer import rnnt_loss, is_checked_out
  if not is_checked_out():
    raise unittest.SkipTest("HawkAaronWarpTransducer not checked out?")

  # Softmax activations for the following inputs:
  import numpy as np
  acts = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.1,
                   0.1, 0.6, 0.1, 0.1, 0.1, 0.1,
                   0.2, 0.8, 0.1, 0.1, 0.6, 0.1,
                   0.1, 0.1, 0.1, 0.1, 0.2, 0.1,
                   0.1, 0.7, 0.1, 0.2, 0.1, 0.1], dtype=np.float32).reshape(1, 2, 3, 5)

  labels = np.array([[1, 2]], dtype=np.int32)
  input_lengths = np.array([2], dtype=np.int32)
  label_lengths = np.array([2], dtype=np.int32)

  acts_t = tf.constant(acts)
  labels_t = tf.constant(labels)
  input_lengths_t = tf.constant(input_lengths)
  label_lengths_t = tf.constant(label_lengths)
  acts_t = tf.nn.log_softmax(acts_t)  # NOTE cpu
  costs = rnnt_loss(acts_t, labels_t, input_lengths_t, label_lengths_t)
  print(costs.eval())


@unittest.skipIf(not is_gpu_available(), "no gpu on this system")
def test_warprnnt_multiple_batches():
  import numpy as np
  n_batch = 2
  n_time = 4
  n_target = 3
  n_vocab = 3

  acts = np.array([0.065357, 0.787530, 0.081592, 0.529716, 0.750675, 0.754135,
                   0.609764, 0.868140, 0.622532, 0.668522, 0.858039, 0.164539,
                   0.989780, 0.944298, 0.603168, 0.946783, 0.666203, 0.286882,
                   0.094184, 0.366674, 0.736168, 0.166680, 0.714154, 0.399400,
                   0.535982, 0.291821, 0.612642, 0.324241, 0.800764, 0.524106,
                   0.779195, 0.183314, 0.113745, 0.240222, 0.339470, 0.134160,
                   0.505562, 0.051597, 0.640290, 0.430733, 0.829473, 0.177467,
                   0.320700, 0.042883, 0.302803, 0.675178, 0.569537, 0.558474,
                   0.083132, 0.060165, 0.107958, 0.748615, 0.943918, 0.486356,
                   0.418199, 0.652408, 0.024243, 0.134582, 0.366342, 0.295830,
                   0.923670, 0.689929, 0.741898, 0.250005, 0.603430, 0.987289,
                   0.592606, 0.884672, 0.543450, 0.660770, 0.377128, 0.358021], dtype=np.float32)
  acts = np.reshape(acts, (n_batch, n_time, n_target, n_vocab))

  expected_costs = np.array([4.28065, 3.93844], dtype=np.float32)
  expected_grads = np.array([-0.186844, -0.062555, 0.249399, -0.203377, 0.202399, 0.000977,
                             -0.141016, 0.079123, 0.061893, -0.011552, -0.081280, 0.092832,
                             -0.154257, 0.229433, -0.075176, -0.246593, 0.146405, 0.100188,
                             -0.012918, -0.061593, 0.074512, -0.055986, 0.219831, -0.163845,
                             -0.497627, 0.209240, 0.288387, 0.013605, -0.030220, 0.016615,
                             0.113925, 0.062781, -0.176706, -0.667078, 0.367659, 0.299419,
                             -0.356344, -0.055347, 0.411691, -0.096922, 0.029459, 0.067463,
                             -0.063518, 0.027654, 0.035863, -0.154499, -0.073942, 0.228441,
                             -0.166790, -0.000088, 0.166878, -0.172370, 0.105565, 0.066804,
                             0.023875, -0.118256, 0.094381, -0.104707, -0.108934, 0.213642,
                             -0.369844, 0.180118, 0.189726, 0.025714, -0.079462, 0.053748,
                             0.122328, -0.238789, 0.116460, -0.598687, 0.302203, 0.296484], dtype=np.float32)
  expected_grads = np.reshape(expected_grads, (n_batch, n_time, n_target, n_vocab))

  labels = np.array([[1, 2], [1, 1]], dtype=np.int32)
  input_lengths = np.array([4, 4], dtype=np.int32)
  label_lengths = np.array([2, 2], dtype=np.int32)

  _run_rnnt(acts, labels, input_lengths, label_lengths, expected_costs, expected_grads, 0)


def test_warprna_forward():
  from returnn.extern.WarpRna import rna_loss, is_checked_out
  if not is_checked_out():
    raise unittest.SkipTest("WarpRna not checked out?")
  import numpy as np
  # computed using slow TF implementation (cross-check against numpy reference impl)
  expected_costs = np.array([2.6347387, 2.4651031])
  expected_grads = np.array([[[[-0.34075904, -0.65924096, 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]],
                              [[-0.09434381, -0.24641524, 0.],
                               [-0.4480959, 0., -0.2111451],
                               [0., 0., 0.]],
                              [[0., -0.09434381, 0.],
                               [-0.25838017, 0., -0.43613094],
                               [-0.2111451, 0., 0.]],
                              [[0., 0., 0.],
                               [0., 0., -0.35272402],
                               [-0.64727604, 0., 0.]]],
                             [[[-0.6283351, -0.37166485, 0.],
                               [0., 0., 0.],
                               [0., 0., 0.]],
                              [[-0.26558593, -0.36274916, 0.],
                               [-0.23790276, -0.13376209, 0.],
                               [0., 0., 0.]],
                              [[0., -0.26558593, 0.],
                               [-0.26772842, -0.3329236, 0.],
                               [-0.13376209, 0., 0.]],
                              [[0., 0., 0.],
                               [0., -0.53331435, 0.],
                               [-0.46668565, 0., 0.]]]])

  n_batch, n_time, n_target, n_vocab = 2, 4, 3, 3
  acts = np.array([0.065357, 0.787530, 0.081592, 0.529716, 0.750675, 0.754135,
                   0.609764, 0.868140, 0.622532, 0.668522, 0.858039, 0.164539,
                   0.989780, 0.944298, 0.603168, 0.946783, 0.666203, 0.286882,
                   0.094184, 0.366674, 0.736168, 0.166680, 0.714154, 0.399400,
                   0.535982, 0.291821, 0.612642, 0.324241, 0.800764, 0.524106,
                   0.779195, 0.183314, 0.113745, 0.240222, 0.339470, 0.134160,
                   0.505562, 0.051597, 0.640290, 0.430733, 0.829473, 0.177467,
                   0.320700, 0.042883, 0.302803, 0.675178, 0.569537, 0.558474,
                   0.083132, 0.060165, 0.107958, 0.748615, 0.943918, 0.486356,
                   0.418199, 0.652408, 0.024243, 0.134582, 0.366342, 0.295830,
                   0.923670, 0.689929, 0.741898, 0.250005, 0.603430, 0.987289,
                   0.592606, 0.884672, 0.543450, 0.660770, 0.377128, 0.358021], dtype=np.float32)
  acts = np.reshape(acts, (n_batch, n_time, n_target, n_vocab))

  labels = np.array([[1, 2], [1, 1]], dtype=np.int32)
  input_lengths = np.array([4, 4], dtype=np.int32)
  label_lengths = np.array([2, 2], dtype=np.int32)

  acts_t = tf.convert_to_tensor(acts)
  labels_t = tf.convert_to_tensor(labels)
  input_lengths_t = tf.convert_to_tensor(input_lengths)
  label_lengths_t = tf.convert_to_tensor(label_lengths)
  log_probs = tf.nn.log_softmax(acts_t)
  costs_cuda = rna_loss(log_probs, labels_t, input_lengths_t, label_lengths_t)
  grads_cuda = tf.gradients(costs_cuda, [log_probs])[0]
  (out_costs_cuda, out_grads_cuda) = session.run([costs_cuda, grads_cuda])

  print("[CUDA] costs:", out_costs_cuda)
  np.testing.assert_allclose(out_grads_cuda, expected_grads, rtol=1e-6)
  np.testing.assert_allclose(out_costs_cuda, expected_costs, rtol=1e-6)


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
    import threading
    if len(list(threading.enumerate())) > 1:
      print("Warning, more than one thread at exit:")
      better_exchook.dump_all_thread_tracebacks()
