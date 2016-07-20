
import threading
from TaskSystem import *
import gc

SharedMemNumpyConfig["enabled"] = True
SharedMemNumpyConfig["auto_pickling_min_size"] = 1

from StringIO import StringIO

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


def test_pickle_unpickle_async():
  for i in range(10):
    m = numpy.random.randn(10, 10)
    p = pickle_dumps((m, m, m))
    m2, m3, m4 = pickle_loads(p)
    assert numpy.allclose(m, m2)
    assert numpy.allclose(m, m3)
    assert numpy.allclose(m, m4)
    ss = list([find_numpy_shared_by_shmid(m.base.mem.shmid) for m in (m2, m3, m4)])
    m2 = m3 = m4 = None
    gc.collect()
    gc.collect()
    gc.collect()
    for s in ss:
      assert isinstance(s, SharedNumpyArray)
      assert s.is_server
      assert not s.is_in_use()
