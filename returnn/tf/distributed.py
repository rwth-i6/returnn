"""
This is for distributed TensorFlow support.
For an overview of the terminology, the concepts and the technology, see
`here <https://github.com/rwth-i6/returnn/wiki/Distributed-TensorFlow>`_.

Distributed TensorFlow covers multiple levels of functionality:

* Low level:
  TF server and client (TF session) connecting to the server via gRPC.
  ClusterSpec describes the collection of all servers.

* High level via strategies (:class:`tf.distribute.Strategy`).
  See the `official guide <https://www.tensorflow.org/guide/distributed_training>`_.
  This can be used by the Keras API, by the Estimator API,
  and by a custom training loop.

* Concepts and terminology.

RETURNN potentially could support all of this, although it will try to be very explicit about it.
Currently we do not use the high level strategy concept but only the low level functionality.
This implementation was originally discussed `here <https://github.com/rwth-i6/returnn/issues/296>`_.

This is also related to `Horovod <https://github.com/horovod/horovod>`_.
Horovod and distributed TF are orthogonal to each other.
They can both be mixed, or used independently.

This is also related to the dataset pipeline :mod:`TFDataPipeline`.
"""

from __future__ import print_function
import os
import typing
import contextlib
import tensorflow as tf
from tensorflow.python.training.server_lib import ClusterSpec
from tensorflow.python.distribute.cluster_resolver.cluster_resolver import format_master_url
# noinspection PyProtectedMember
from tensorflow.python.distribute.distribute_lib import _DefaultDistributionExtended as DefaultDistributionExtended
from returnn.log import log
from returnn.config import Config
from returnn.util.basic import CollectionReadCheckCovered


class MPIClusterResolver(tf.distribute.cluster_resolver.ClusterResolver):
  """
  ClusterResolver for MPI.
  Distributed TF is in general totally independent of MPI.
  We only use MPI here to figure out the ClusterSpec.
  After this is set up, MPI will not be used anymore.
  TF itself will not make use of MPI;
  all communications are handled via gRPC.
  (Although `Horovod <https://github.com/horovod/horovod>`_ would use MPI, but that is another topic.)

  If you use Sun Grid Engine (SGE) / Oracle Grid Engine,
  with the parallel environment (PE) feature
  (`doc <http://gridscheduler.sourceforge.net/htmlman/htmlman5/sge_pe.html>`_)
  (i.e. the SGE job was started e.g. via: `qsub -pe mpi 8`),
  then you might run your sub processes (slots) via `mpirun`, e.g. like::

    mpirun -np 8 -mca pml ob1 -mca btl ^openib python returnn/rnn.py ...

  Open MPI knows about SGE and will correctly start subprocesses (for each PE slot)
  (potentially remotely).
  From the `Open MPI doc <https://www.open-mpi.org/faq/?category=sge>`_:

    Open MPI will automatically detect when it is running inside SGE and will just "do the Right Thing."
    Specifically, if you execute an mpirun command in a SGE job,
    it will automatically use the SGE mechanisms to launch and kill processes.
    There is no need to specify what nodes to run on -
    Open MPI will obtain this information directly from SGE
    and default to a number of processes equal to the slot count specified.

  SGE provides the `PE_HOSTFILE` env var points to a file which lists all hosts
  and number of slots per host.
  This would be available for the SGE main job process, i.e. where `mpirun` is run,
  but this might *not* be available on the subprocesses (slots)
  which are started by `mpirun` remotely.

  Within such a MPI process, the only reliable way to get information about the other peer processes,
  we must use MPI functions.
  A straight-forward simple way for this is the `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_ module.
  `mpi4py` can be mixed together with `Horovod <https://github.com/horovod/horovod>`_, so this is a sensible choice.

  This is somewhat similar to :class:`SlurmClusterResolver`.
  Also related: `MPIClusterResolver PR <https://github.com/tensorflow/tensorflow/issues/38356>`_.
  https://github.com/Peidong-Wang/Distributed-TensorFlow-Using-MPI/
  https://stackoverflow.com/questions/10912793/how-are-mpi-processes-started
  """

  def __init__(self):
    # noinspection PyPackageRequirements,PyUnresolvedReferences
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    self._host = MPI.Get_processor_name()
    # Currently just pick some random open port.
    # We could change this to some deterministic variant which could depend on
    # the pid, GPU-number, job-id, or so.
    # But the current variant might actually be fine.
    self._port = _get_open_port()
    self._rank = comm.Get_rank()
    self._size = comm.Get_size()
    hosts = comm.allgather((self._rank, self._host, self._port))
    peers = {rank: "%s:%i" % (host, port) for (rank, host, port) in hosts}
    assert len(peers) == self._size and 0 in peers and (self._size - 1) in peers
    self._peers = [peers[i] for i in range(self._size)]
    self.task_type = "worker"  # currently we are all equal workers (except id 0, which is the chief)
    self.task_id = self._rank
    self.rpc_layer = "grpc"

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
    # We currently only use the worker task type.
    # We currently do not support in dynamic changes of the cluster environment.
    return ClusterSpec({"worker": self._peers})

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
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id

    if task_type is not None and task_id is not None:
      return format_master_url(
        self.cluster_spec().task_address(task_type, task_id),
        rpc_layer or self.rpc_layer)

    return ''

  def num_accelerators(self, task_type=None, task_id=None, config_proto=None):
    """Returns the number of accelerator cores per worker.

    This returns the number of accelerator cores (such as GPUs and TPUs)
    available per worker.

    Optionally, we allow callers to specify the task_type, and task_id, for
    if they want to target a specific TensorFlow process to query
    the number of accelerators. This is to support heterogenous environments,
    where the number of accelerators cores per host is different.

    Args:
      task_type: (Optional) The type of the TensorFlow task of the machine we
        want to query.
      task_id: (Optional) The index of the TensorFlow task of the machine we
        want to query.
      config_proto: (Optional) Configuration for starting a new session to
        query how many accelerator cores it has.

    Returns:
      A map of accelerator types to number of cores.
    """
    # The default might be fine.
    # It will connect to the remote TF server and query that information.
    return super(MPIClusterResolver, self).num_accelerators(
      task_type=task_type, task_id=task_id, config_proto=config_proto)


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


class LocalOnlyClusterResolver(tf.distribute.cluster_resolver.ClusterResolver):
  """
  Cluster resolver for one local instance.
  """

  def __init__(self):
    self._port = _get_open_port()
    self._host = "localhost:%i" % self._port
    self.task_type = "worker"
    self.task_id = 0
    self.rpc_layer = "grpc"

  def cluster_spec(self):
    """
    :rtype: ClusterSpec
    """
    return ClusterSpec({"worker": [self._host]})

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
    task_type = task_type if task_type is not None else self.task_type
    task_id = task_id if task_id is not None else self.task_id

    if task_type is not None and task_id is not None:
      return format_master_url(
        self.cluster_spec().task_address(task_type, task_id),
        rpc_layer or self.rpc_layer)

    return ''


_controller = None  # type: typing.Optional[_Controller]


class _Controller:
  """
  The controller encapsulates the logic needed for distributed TF in RETURNN.
  It would be setup via :func:`init_distributed_tf`.

  We currently check for the TF_CONFIG env var,
  and if set, use the :class:`TFConfigClusterResolver`.
  Otherwise we assume a MPI setup, and use
  :class:`MPIClusterResolver`.

  If we are using in-graph replication, and this is not the chief,
  this would just run a TF server and wait for commands.
  """

  def __init__(self, config):
    """
    :param Config config:
    """
    print("Initialize distributed TensorFlow", file=log.v2)
    self.config = config
    opts = config.get_of_type("distributed_tf", dict, {})
    opts = CollectionReadCheckCovered(opts)
    self.opts = opts
    if opts.get("local_only", False):  # might be useful for testing
      cluster_resolver = LocalOnlyClusterResolver()
      print("Use local-only cluster resolver,", file=log.v4, end=" ")
    elif os.environ.get("TF_CONFIG", ""):
      cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
      print("Use TF_CONFIG %s," % os.environ["TF_CONFIG"], file=log.v4, end=" ")
    else:
      cluster_resolver = MPIClusterResolver()
      print("Use MPI cluster resolver,", file=log.v4, end=" ")
    print("cluster spec %s, master %s" % (cluster_resolver.cluster_spec(), cluster_resolver.master()), file=log.v4)
    self.cluster_resolver = cluster_resolver
    cluster_spec = cluster_resolver.cluster_spec()
    self.cluster_spec = cluster_spec
    tf_session_opts = config.typed_dict.get("tf_session_opts", {})
    server_config = tf.compat.v1.ConfigProto(**tf_session_opts)
    # Note that there is no clean way currently in TF to uninit the TF server.
    # If we would use this multiple times (e.g. in tests),
    # it might actually be better to cache the server as a singleton...
    server = tf.distribute.Server(
      cluster_spec,
      job_name=cluster_resolver.task_type, task_index=cluster_resolver.task_id,
      config=server_config)
    self.server = server
    self.strategy = ReturnnDefaultStrategy()  # not really used currently...
    self.opts.assert_all_read()

  def is_chief(self):
    """
    :return: whether we are the chief (worker 0)
    :rtype: bool
    """
    if "chief" in self.cluster_spec.jobs:
      return self.cluster_resolver.task_type == "chief"
    if "master" in self.cluster_spec.jobs:
      return self.cluster_resolver.task_type == "master"
    if "worker" in self.cluster_spec.jobs:
      if self.cluster_resolver.task_type != "worker":
        return False
      return self.cluster_resolver.task_id == 0
    raise NotImplementedError("is_chief unknown for cluster spec %r" % self.cluster_spec)


# not really used currently
class ReturnnDefaultStrategy(tf.distribute.Strategy):
  """
  RETURNN default strategy.
  """

  def __init__(self):
    super(ReturnnDefaultStrategy, self).__init__(
      extended=ReturnnDefaultStrategyExtended(self))


# not really used currently
# noinspection PyAbstractClass
class ReturnnDefaultStrategyExtended(DefaultDistributionExtended):
  """
  RETURNN default strategy extended.
  """


def init_distributed_tf(config):
  """
  This is called early in startup of RETURNN.

  :param Config config:
  """
  global _controller
  assert not _controller, "init_distributed_tf called twice?"
  _controller = _Controller(config=config)


def is_enabled():
  """
  :rtype: bool
  """
  return bool(_controller)


def get_session_target():
  """
  This would be called if you have a local custom graph in the current process (replica)
  and want to execute parts of it. This is e.g. the case for between-graph replication.
  After creating the graph, you would create a session
  which connects to the server returned by this function.

  :return: URL of the TF server, where the local session should connect to
  :rtype: str
  """
  assert _controller, "init_distributed_tf not called?"
  return _controller.server.target


@contextlib.contextmanager
def _temporary_init_distributed_tf(config):
  """
  This is useful for tests.

  :param config:
  :return: scope where we have initialized distributed TF, and going out-of-scope will uninit again
  """
  global _controller
  init_distributed_tf(config=config)
  yield
  # Now uninit again.
  # Note that there is no real way to stop the server.
  # If we use this often in tests, it might actually be better to cache the server as a singleton...
  assert _controller
  _controller = None  # type: typing.Optional[_Controller]
