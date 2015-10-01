
import CustomLSTMFunctions as C


def test_setup():
  assert hasattr(C, "attention_dot_fun_fwd")
  assert hasattr(C, "attention_dot_fun_fwd_res1")
