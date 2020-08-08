
from __future__ import print_function

import _setup_test_env  # noqa
import sys
import os
import threading
from returnn.util.task_system import *
import unittest
import gc

SharedMemNumpyConfig["enabled"] = True
SharedMemNumpyConfig["auto_pickling_min_size"] = 1

try:
  # noinspection PyCompatibility
  from StringIO import StringIO
except ImportError:  # Python 3
  from io import BytesIO as StringIO


def pickle_dumps(obj):
  sio = StringIO()
  p = Pickler(sio)
  p.dump(obj)
  return sio.getvalue()


def pickle_loads(s):
  p = Unpickler(StringIO(s))
  return p.load()


def find_numpy_shared_by_shmid(shmid):
  for sh in SharedNumpyArray.ServerInstances:
    assert isinstance(sh, SharedNumpyArray)
    assert sh.mem is not None
    assert sh.mem.shmid > 0
    if sh.mem.shmid == shmid:
      return sh
  return None


_have_working_shmget = None


def have_working_shmget():
  global _have_working_shmget
  if _have_working_shmget is None:
    _have_working_shmget = SharedMem.is_shmget_functioning()
  print("shmget functioning:", _have_working_shmget)
  return _have_working_shmget


@unittest.skipIf(not have_working_shmget(), "shmget does not work")
def test_shmget_functioning():
  assert SharedMem.is_shmget_functioning()


@unittest.skipIf(not have_working_shmget(), "shmget does not work")
def test_pickle_numpy():
  m = numpy.random.randn(10, 10)
  p = pickle_dumps(m)
  m2 = pickle_loads(p)
  assert isinstance(m2, numpy.ndarray)
  assert numpy.allclose(m, m2)
  assert isinstance(m2.base, SharedNumpyArray)
  shared_client = m2.base
  assert not shared_client.is_server
  shared_server = find_numpy_shared_by_shmid(m2.base.mem.shmid)
  assert shared_server.is_server
  assert numpy.allclose(m, shared_server.create_numpy_array())
  assert numpy.allclose(m, shared_client.create_numpy_array())
  assert shared_server.is_in_use()
  assert shared_client.is_in_use()
  numpy_set_unused(m2)
  assert not shared_server.is_in_use()
  assert shared_client.mem is None


@unittest.skipIf(not have_working_shmget(), "shmget does not work")
def test_pickle_numpy_scalar():
  # Note that a scalar really does not work because numpy.array(float) will always own its data.
  m = numpy.array([numpy.random.randn()])
  assert isinstance(m, numpy.ndarray)
  assert m.shape == (1,)
  assert m.nbytes >= 1
  p = pickle_dumps(m)
  m2 = pickle_loads(p)
  assert isinstance(m2, numpy.ndarray)
  assert numpy.allclose(m, m2)
  assert isinstance(m2.base, SharedNumpyArray)
  shared_client = m2.base
  assert not shared_client.is_server
  shared_server = find_numpy_shared_by_shmid(m2.base.mem.shmid)
  assert shared_server.is_server
  assert numpy.allclose(m, shared_server.create_numpy_array())
  assert numpy.allclose(m, shared_client.create_numpy_array())
  assert shared_server.is_in_use()
  assert shared_client.is_in_use()
  numpy_set_unused(m2)
  assert not shared_server.is_in_use()
  assert shared_client.mem is None


@unittest.skipIf(not have_working_shmget(), "shmget does not work")
def test_pickle_gc_aggressive():
  m = numpy.random.randn(10, 10)
  p = pickle_dumps(m)
  m2 = pickle_loads(p)
  assert isinstance(m2, numpy.ndarray)
  assert numpy.allclose(m, m2)
  assert isinstance(m2.base, SharedNumpyArray)
  print("refcount: %i" % sys.getrefcount(m2.base))
  gc.collect()
  gc.collect()
  print("refcount: %i" % sys.getrefcount(m2.base))
  assert m2.base.is_in_use()
  server = find_numpy_shared_by_shmid(m2.base.mem.shmid)
  m2 = None
  gc.collect()


@unittest.skipIf(not have_working_shmget(), "shmget does not work")
def test_pickle_multiple():
  for i in range(20):
    ms = [numpy.random.randn(10, 10) for i in range(i % 3 + 1)]
    p = pickle_dumps(ms)
    ms2 = pickle_loads(p)
    assert len(ms) == len(ms2)
    for m, m2 in zip(ms, ms2):
      assert numpy.allclose(m, m2)
      assert isinstance(m2.base, SharedNumpyArray)


@unittest.skipIf(not have_working_shmget(), "shmget does not work")
def test_pickle_unpickle_auto_unused():
  old_num_servers = None
  for i in range(10):
    m = numpy.random.randn(i * 2 + 1, i * 3 + 2)
    p = pickle_dumps((m, m, m))
    new_num_servers = len(SharedNumpyArray.ServerInstances)
    if old_num_servers is not None:
      assert old_num_servers == new_num_servers
    old_num_servers = new_num_servers
    m2, m3, m4 = pickle_loads(p)
    assert numpy.allclose(m, m2)
    assert numpy.allclose(m, m3)
    assert numpy.allclose(m, m4)
    assert not m4.base.is_server
    m4.base._get_in_use_flag_ref().value = 42
    assert m4.base._get_in_use_flag_ref().value == 42
    assert find_numpy_shared_by_shmid(m4.base.mem.shmid)._get_in_use_flag_ref().value == 42
    assert numpy.allclose(m, m4)
    ss = list([find_numpy_shared_by_shmid(_m.base.mem.shmid) for _m in (m2, m3, m4)])
    _m = None
    m2 = m3 = m4 = None
    gc.collect()
    for s in ss:
      assert isinstance(s, SharedNumpyArray)
      assert s.is_server
      assert not s.is_in_use()
