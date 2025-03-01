#!/bin/bash

#PBS -q regular-g
#PBS -W group_list=xg24i002
#PBS -l select=16:mpiprocs=1
#PBS -l walltime=00:45:00
#PBS -j oe

current_time=$(date "+%Y%m%d%H%M")
mpirun --mca mpi_abort_print_stack 1 \
 --report-bindings \
 /work/xg24i002/x10041/miniconda3/envs/py313/bin/python /work/xg24i002/x10041/kfac-pytorch/multi_node_resnet_cifar_ddp.py\
 --timestamp="$current_time"