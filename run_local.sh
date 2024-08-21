#!/bin/bash

module load openmpi/gdr/4.1.6/gcc8.3.1-cuda12.3.2

current_time=$(date "+%Y%m%d%H%M")

mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
    -x PATH -np 4 --map-by ppr:4:node:PE=3 --nooversubscribe --report-bindings \
    /home/NBB/yu_mingzhe/miniconda3/envs/basic_env/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/mobilev3_cifar.py \
    --timestamp="$current_time"
