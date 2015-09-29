
import unittest
from Device import have_gpu


@unittest.skipIf(not have_gpu(), "no gpu on this system")
def test_setup():
  import CustomLSTMFunctions as C
  assert hasattr(C, "attention_dot_fun_fwd")
  assert hasattr(C, "attention_dot_fun_fwd_res1")
