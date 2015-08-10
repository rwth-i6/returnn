
from nose.tools import assert_equal, assert_is_instance, assert_in, assert_not_in, assert_true, assert_false
import unittest
from Device import have_gpu


def test_have_gpu():
  have_gpu()


@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_cuda():
  import theano.sandbox.cuda as theano_cuda
  assert_true(theano_cuda.cuda_available, "Theano CUDA support not available. Check that nvcc is in $PATH.")
  if theano_cuda.cuda_enabled: # already enabled when $THEANO_FLAGS=device=gpu
    print("CUDA already enabled")
  else:
    print("Call theano_cuda.use")
    theano_cuda.use(device="gpu", force=True)
  try:
    import cuda_ndarray.cuda_ndarray as cuda
  except ImportError as exc:
    raise Exception("Theano CUDA support seems broken: %s" % exc)
  id = cuda.active_device_number(); """ :type: int """
  device_name = cuda.active_device_name(); """ :type: str """
  print("id: %i", id)
  print("dev name: %s" % device_name)
