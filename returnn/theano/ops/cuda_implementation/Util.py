import os
from theano.sandbox.cuda.basic_ops import (GpuContiguous, GpuFromHost)


def raw_variable(x):
  while x.owner is not None and (type(x.owner.op) == GpuContiguous or type(x.owner.op) == GpuFromHost):
    x = x.owner.inputs[0]
  return x


def get_c_support_code_common():
  base_path = os.path.dirname(__file__)
  with open(base_path + "/c_support_code_common.cpp") as f:
    return f.read()

def get_c_support_code_mdlstm():
  base_path = os.path.dirname(__file__)
  with open(base_path + "/c_support_code_mdlstm.cpp") as f:
    return f.read()

def get_c_support_code_cudnn():
  base_path = os.path.dirname(__file__)
  with open(base_path + "/c_support_code_cudnn.cpp") as f:
    return f.read()
