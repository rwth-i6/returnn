#!/bin/bash

cd ..

mpirun -np 3 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH -x HOROVOD_TIMELINE \
    -mca pml ob1 -mca btl ^openib \
    python3 rnn.py demos/demo-tf-native-lstm2.12ax.config ++use_horovod 1 ++device gpu

