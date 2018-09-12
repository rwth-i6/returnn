#!/usr/bin/env python3

import os
print("pid %i: Hello" % os.getpid())

import tensorflow as tf
import horovod.tensorflow as hvd


# Initialize Horovod
hvd.init()

print("pid %i: hvd: rank: %i, size: %i, local_rank %i, local_size %i" % (os.getpid(), hvd.rank(), hvd.size(), hvd.local_rank(), hvd.local_size()))
