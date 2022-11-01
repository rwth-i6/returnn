#!/bin/bash

# https://returnn.readthedocs.io/en/latest/advanced/multi_gpu.html

# example SGE submit:
# echo bash .../returnn/demos/demo-horovod-mpi.py.sh | qsub -l h_vmem=4G -l h_rt=1:00:00 -l gpu=1 -l num_proc=2 -pe mpi 2

# make sure horovod is installed with MPI and TF support
# https://github.com/rwth-i6/returnn/issues/1196

set -ex

mydir=$(dirname $0)
echo "mydir: $mydir"

type mpirun

mpirun -np 2 \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    python3 $mydir/demo-horovod-mpi.py
