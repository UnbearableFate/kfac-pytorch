#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 8
#PBS -l elapstim_req=00:30:00
#PBS -v NQSV_MPI_VER=4.1.6/nvhpc24.5-cuda12.4
#PBS -M kanakawapanman@gmail.com

module load openmpi/4.1.6/nvhpc24.5-cuda12.4 

current_time=$(date "+%Y%m%d%H%M")

mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
 -x PATH -np 8 --map-by ppr:1:node --report-bindings \
 /work/NBB/yu_mingzhe/miniconda3/envs/py311/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/multi_node_resnet_cifar.py \
 --timestamp="$current_time"