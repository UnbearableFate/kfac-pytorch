#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 32
#PBS -l elapstim_req=00:30:00
#PBS -v NQSV_MPI_VER=4.1.6/gcc11.4.0-cuda12.3.2
#PBS -M kanakawapanman@gmail.com
source /work/NBB/yu_mingzhe/.venv/bin/activate
module load openmpi/4.1.6/gcc11.4.0-cuda12.3.2

current_time=$(date "+%Y%m%d%H%M")

mpirun -x PATH -np 16 --map-by ppr:1:node:PE=24 --report-bindings $NQSII_MPIOPTS \
 /work/NBB/yu_mingzhe/.venv/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/mlp_fashion_minst_sick.py \
 --timestamp=$current_time