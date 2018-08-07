#!/bin/bash

cd ..

mpirun -np 4 \
    -H localhost,localhost,localhost,localhost \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python3 rnn.py demos/demo-tf-native-lstm2.12ax.config ++use_horovod 1 ++device cpu

