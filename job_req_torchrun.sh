#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 8
#PBS -l elapstim_req=00:10:00
#PBS -v NQSV_MPI_VER=4.1.5/nvhpc23.9-cuda12.2-ucx1.15.0
#PBS -M kanakawapanman@gmail.com
source /work/NBB/yu_mingzhe/venv/bin/activate

/system/apps/ubuntu/20.04-202304/openmpi/4.1.5/nvhpc23.9-cuda12.2-ucx1.15.0/bin/mpirun  -x PATH -np 8 --map-by ppr:1:node:PE=16 --report-bindings $NQSII_MPIOPTS bash /work/NBB/yu_mingzhe/kfac-pytorch/launch_torch_single.sh