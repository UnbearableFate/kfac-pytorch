#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 2
#PBS -l elapstim_req=00:20:00
#PBS -v NQSV_MPI_VER=gdr/4.1.6/gcc8.3.1-cuda12.3.2
#PBS -M kanakawapanman@gmail.com

module load openmpi/gdr/4.1.6/gcc8.3.1-cuda12.3.2
#/home/NBB/yu_mingzhe/miniconda3/bin/conda activate basic_env

current_time=$(date "+%Y%m%d%H%M")

mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
 -x PATH -np 8 --map-by ppr:4:node:PE=6 --nooversubscribe --report-bindings \
 /home/NBB/yu_mingzhe/miniconda3/envs/basic_env/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/mobilev3_cifar.py \
 --timestamp="$current_time"
