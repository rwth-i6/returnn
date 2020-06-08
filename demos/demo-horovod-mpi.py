#!/usr/bin/env python3

import os
print("pid %i: Hello" % os.getpid())

import sys
print("Python version:", sys.version)

print("Env:")
for key, value in sorted(os.environ.items()):
  print("%s=%s" % (key, value))
print()

if os.environ.get("PE_HOSTFILE", ""):
  with open(os.environ["PE_HOSTFILE"], "r") as f:
    print("PE_HOSTFILE, %s:" % os.environ["PE_HOSTFILE"])
    print(f.read())

if os.environ.get("SGE_JOB_SPOOL_DIR", ""):
  print("SGE_JOB_SPOOL_DIR, %s:" % os.environ["SGE_JOB_SPOOL_DIR"])
  for name in os.listdir(os.environ["SGE_JOB_SPOOL_DIR"]):
    print(name)
  print()

# https://github.com/horovod/horovod/issues/1123
try:
  import ctypes
  ctypes.CDLL("libhwloc.so", mode=ctypes.RTLD_GLOBAL)
except Exception as exc:
  print("Exception while loading libhwloc.so, ignoring...", exc)

print("sys.path:")
i = 0
for p in list(sys.path):
  print(p)
  if "/.local/lib/" in p:
    # small workaround if the order is messed up... prefer from .local/lib
    print("(insert at position %i)" % i)
    sys.path.insert(i, p)
    i += 1
print()

import tensorflow as tf
print("TF version:", tf.__version__)

import horovod.tensorflow as hvd

# Initialize Horovod
hvd.init()

print("pid %i: hvd: rank: %i, size: %i, local_rank %i, local_size %i" % (os.getpid(), hvd.rank(), hvd.size(), hvd.local_rank(), hvd.local_size()))

