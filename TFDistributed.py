"""
This is for distributed TensorFlow support.

https://github.com/rwth-i6/returnn/wiki/Distributed-TensorFlow
https://github.com/rwth-i6/returnn/issues/296
"""

import os
import tensorflow as tf
from tensorflow.python.training.server_lib import ClusterSpec


class MPIClusterResolver(tf.distribute.cluster_resolver.ClusterResolver):
  """
  ClusterResolver for MPI.

  If you use Sun Grid Engine (SGE) / Oracle Grid Engine,
  with the parallel environment (PE) feature
  (i.e. the SGE job was started e.g. via: `qsub -pe mpi 8`),
  then you might run your sub processes via `mpirun`, e.g. like::

    mpirun -np 8 -mca pml ob1 -mca btl ^openib python returnn/rnn.py ...

  SGE provides the `PE_HOSTFILE` env var points to a file which lists all hosts
  and number of slots per host.
  This would be available for the process and node where `mpirun` is run,
  but this might *not* be available on the subprocesses
  which are started by `mpirun` remotely.

  Within such a MPI process, the only reliable way to get information about the other peer processes,
  we must use MPI functions.
  A straight-forward simple way for this is the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ module.
  `mpi4py` can be mixed together with `Horovod <https://github.com/horovod/horovod>`_, so this is a sensible choice.

  This is somewhat similar to :class:`SlurmClusterResolver`.
  Also related: MPIClusterResolver (https://github.com/tensorflow/tensorflow/issues/38356).
  https://github.com/Peidong-Wang/Distributed-TensorFlow-Using-MPI/
  https://stackoverflow.com/questions/10912793/how-are-mpi-processes-started
  """

  def __init__(self):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    self._host = MPI.Get_processor_name()
    self._port = _get_open_port()
    self._rank = comm.Get_rank()
    self._size = comm.Get_size()
    hosts = comm.allgather((self._rank, self._host, self._port))
    self._peers = {rank: "%s:%i" % (host, port) for (rank, host, port) in hosts}

  def cluster_spec(self):
    """Retrieve the current state of the cluster and return a ClusterSpec.

    Returns:
      A ClusterSpec representing the state of the cluster at the moment this
      function is called.

    Implementors of this function must take care in ensuring that the
    ClusterSpec returned is up-to-date at the time of calling this function.
    This usually means retrieving the information from the underlying cluster
    management system every time this function is invoked and reconstructing
    a cluster_spec, rather than attempting to cache anything.
    """
    # TODO
    raise NotImplementedError()

  def master(self, task_type=None, task_id=None, rpc_layer=None):
    """Retrieves the name or URL of the session master.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the master.
      task_id: (Optional) The index of the TensorFlow task of the master.
      rpc_layer: (Optional) The RPC protocol for the given cluster.

    Returns:
      The name or URL of the session master.

    Implementors of this function must take care in ensuring that the master
    returned is up-to-date at the time to calling this function. This usually
    means retrieving the master every time this function is invoked.
    """

    # TODO via local job id, local host, local rank ...
    raise NotImplementedError()


def _get_open_port():
  """
  https://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python

  :rtype: int
  """
  import socket
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind(("", 0))
  s.listen(1)
  port = s.getsockname()[1]
  s.close()
  return port

