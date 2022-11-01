#!/usr/bin/env python3

"""
Small demo for Horovod + MPI.
"""

import os
import sys


def main():
  """
  Main entry.
  """
  print("pid %i: Hello" % os.getpid())
  print("Python version:", sys.version)

  print("Env:")
  for key, value in sorted(os.environ.items()):
    print("%s=%s" % (key, value))
  print()

  if os.environ.get("PE_HOSTFILE", ""):
    try:
      print("PE_HOSTFILE, %s:" % os.environ["PE_HOSTFILE"])
      with open(os.environ["PE_HOSTFILE"], "r") as f:
        print(f.read())
    except Exception as exc:
      print(exc)

  if os.environ.get("SGE_JOB_SPOOL_DIR", ""):
    print("SGE_JOB_SPOOL_DIR, %s:" % os.environ["SGE_JOB_SPOOL_DIR"])
    try:
      for name in os.listdir(os.environ["SGE_JOB_SPOOL_DIR"]):
        print(name)
      print()
    except Exception as exc:
      print(exc)

  if os.environ.get("OMPI_FILE_LOCATION", ""):
    print("OMPI_FILE_LOCATION, %s:" % os.environ["OMPI_FILE_LOCATION"])
    d = os.path.dirname(os.path.dirname(os.environ["OMPI_FILE_LOCATION"]))
    try:
      print("dir:", d)
      for name in os.listdir(d):
        print(name)
      print()
      print("contact.txt:")
      with open("%s/contact.txt" % d, "r") as f:
        print(f.read())
      print()
    except Exception as exc:
      print(exc)

  # https://github.com/horovod/horovod/issues/1123
  try:
    import ctypes
    ctypes.CDLL("libhwloc.so", mode=ctypes.RTLD_GLOBAL)
  except Exception as exc:
    print("Exception while loading libhwloc.so, ignoring...", exc)

  print("sys.path:")
  for p in list(sys.path):
    print(p)
  print()

  try:
    from mpi4py import MPI  # noqa
    name = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD
    print("mpi4py:", "name: %s," % name, "rank: %i," % comm.Get_rank(), "size: %i" % comm.Get_size())
    hosts = comm.allgather((comm.Get_rank(), name))  # Get the names of all the other hosts
    print("  all hosts:", {key: item for (key, item) in hosts})
  except ImportError:
    print("mpi4py not available")

  print("Import TF now...")
  import tensorflow as tf
  print("TF version:", tf.__version__)

  import horovod  # noqa
  print("Horovod version:", horovod.__version__)
  import horovod.tensorflow as hvd  # noqa

  # Initialize Horovod
  hvd.init()

  print(
    "pid %i: hvd: rank: %i, size: %i, local_rank %i, local_size %i" % (
      os.getpid(), hvd.rank(), hvd.size(), hvd.local_rank(), hvd.local_size()))


if __name__ == "__main__":
  main()
