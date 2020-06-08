"""
This is for distributed TensorFlow support.

https://github.com/rwth-i6/returnn/wiki/Distributed-TensorFlow
https://github.com/rwth-i6/returnn/issues/296
"""

import os
import tensorflow as tf
from tensorflow.python.training.server_lib import ClusterSpec


class SGEClusterResolver(tf.distribute.cluster_resolver.ClusterResolver):
  """
  ClusterResolver for the Sun Grid Engine (SGE) / Oracle Grid Engine,
  when used with the parallel environment (PE) feature.
  I.e. the SGE job was started e.g. via: `qsub -pe mpi 8`.

  We assume that the SGE `JOB_ID` env var is set.

  The `PE_HOSTFILE` env var points to a file which lists all hosts
  and number of slots per host.

  We also assume that each process (worker) is already started via Open MPI,
  and that `OMPI_COMM_WORLD_LOCAL_RANK` and `OMPI_COMM_WORLD_LOCAL_SIZE` is set.
  E.g. this might be via `mpirun` or via some other way.

  This is somewhat similar to :class:`SlurmClusterResolver`.
  Also related: MPIClusterResolver (https://github.com/tensorflow/tensorflow/issues/38356).
  https://github.com/Peidong-Wang/Distributed-TensorFlow-Using-MPI/
  https://stackoverflow.com/questions/10912793/how-are-mpi-processes-started
  """

  def __init__(self):
    assert os.environ.get("JOB_ID", "") # not in SGE environment?
    # TODO cannot use PE_HOSTFILE in all cases... this is not accessible if we are on another node...
    # If on another host, we could connect to the main host. We could know via OMPI_MCA_orte_hnp_uri...
    assert os.environ.get("PE_HOSTFILE", "")  # not in SGE parallel environment?
    assert os.environ.get("OMPI_COMM_WORLD_LOCAL_SIZE", "")  # not in Open MPI environment? not started via mpirun?
    # TODO we could use mpi4py or Horovod (but maybe not both together...)

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
    # qint.py might overwrite JOB_ID but sets SGE_JOB_ID instead.
    job_id = int(os.environ.get("SGE_JOB_ID") or os.environ.get("JOB_ID") or 0)

    with open(os.environ["PE_HOSTFILE"]) as f:
      # Example content:
      """
      cluster-cn-229.informatik.rwth-aachen.de 1 4-GPU-1080@cluster-cn-229.informatik.rwth-aachen.de UNDEFINED
      cluster-cn-234.informatik.rwth-aachen.de 2 4-GPU-1080-5h@cluster-cn-234.informatik.rwth-aachen.de UNDEFINED
      """
      for line in f.read().splitlines():
        parts = line.split()
        assert len(parts) >= 2
        host_name = parts[0]
        num_slots = int(parts[1])

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
