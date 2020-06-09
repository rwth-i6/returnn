.. _multi_gpu:

==================
Multi GPU training
==================

This is about multi GPU training with the TensorFlow backend.

We currently use `Horovod <https://github.com/horovod/horovod>`__.
Please refer to the `Horovod documentation <https://github.com/horovod/horovod>`__.
Horovod provides simple TensorFlow ops for allreduce, allgather and broadcast,
which will internally use the best available method,
i.e. either NCCL for direct GPU transfer (on a single node),
or MPI for any kind of transfer,
including multiple nodes in a cluster network.

------------
Installation
------------

If you want to use NCCL, make sure it's installed and it can be found.

You need to install some MPI.
If you are in a cluster environment, usually you have that already.
Check that you can run ``mpirun``.

You need to install to install Horovod. This usually can be installed via pip::

    pip3 install horovod

For further information, please refer to the
`Horovod documentation <https://github.com/horovod/horovod>`__.

-----
Usage
-----

In general, please refer to the
`Horovod documentation <https://github.com/horovod/horovod>`__.

RETURNN will try to use Horovod when you specify ``use_horovod = True``
in your config (or via command line argument).

The implementation in RETURNN is pretty straight forward
and follows mostly the tutorial.
Try to understand that to get a basic understanding about how it works.

Relevant RETURNN settings
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``use_horovod: bool`` should be ``True``

* ``horovod_reduce_type: str`` can be either ``"grad"`` or ``"param"``.

  * ``"grad"`` means that we reduce the gradient after every step,
    and then use the same summed gradient to update the model in each instance.
  * ``"param"`` means that every instance will do an update individually
    and after some N number of steps, we synchronize the models.
    This reduces the amount of communication and should increase the speed.

* ``horovod_param_sync_step: int``:
  if the reduce type is param, this will specify after how many update steps
  the model parameters will be synchronized (i.e. averaged)

* ``horovod_scale_lr: bool``: whether to multiply the lr by number of instances
  (False by default)
  
* ``horovod_dataset_distribution: str``: ``"shard"`` by default. one of:

  * ``"shard"``: uses sharding for the dataset (via ``batch_slice`` for :class:`FeedDictDataProvider`)
  * ``"random_seed_offset"``: sets the default ``random_seed_offset`` via the rank

Recommendations
~~~~~~~~~~~~~~~

You should use a fast dataset implementation.
We recommend to use ``HDFDataset`` with ``cache_size = 0`` in your config.
You can use ``tools/hdf_dump.py`` to convert any dataset into a HDF dataset.

Single node, multiple GPUs
~~~~~~~~~~~~~~~~~~~~~~~~~~

Example::

    mpirun -np 4 -bind-to none -map-by slot \
           -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_TIMELINE -x DEBUG \
           -mca pml ob1 -mca btl ^openib \
           python3 returnn/rnn.py returnn-config.py ++use_horovod 1

Multiple nodes
~~~~~~~~~~~~~~

Example `SGE <https://en.wikipedia.org/wiki/Oracle_Grid_Engine>`__ ``qsub`` parameters::

    -hard -l h_vmem=15G -l h_rt=10:00:00 -l gpu=1 -l qname='*1080*|*TITAN*' -l num_proc=8 -pe mpi 8

You might need to fix your SSH settings::

    Host cluster-*
       TCPKeepAlive yes
       ForwardAgent yes
       ForwardX11 yes
       Compression yes
       StrictHostKeyChecking no
       HashKnownHosts no

Test::

    mpirun -np 8 -mca pml ob1 -mca btl ^openib python3 returnn/demos/demo-horovod-mpi.py

Debugging / profiling / benchmarking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a starting point, please refer to the
`Horovod documentation <https://github.com/horovod/horovod>`__.
E.g. the Horovod timeline feature might be helpful.

In some cases, the dataset can be a bootleneck.
Try to use ``HDFDataset``.
Look at this output at the end of an epoch::

    train epoch 1, finished after 2941 steps, 0:28:58 elapsed (99.3% computing time)

Look at the ``computing time`` in particular.
That numbers measures how much relative time was spend inside TF ``session.run``.
If this is below 90% or so, it means that you wasted some time elsewhere,
e.g. the dataset loading.

Then, refer to the TensorFlow documentation
about how to do basic benchmarking / profiling.
E.g. the timeline feature might be helpful.

Also look through some of the reported
`RETURNN issues <https://github.com/rwth-i6/returnn/issues/>`__,
e.g. `issue #73 <https://github.com/rwth-i6/returnn/issues/73>`__.
