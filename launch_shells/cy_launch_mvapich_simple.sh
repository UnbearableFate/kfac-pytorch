#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T mvapich
#PBS -b 2
#PBS -l elapstim_req=00:20:00
#PBS -v NQSV_MPI_VER=mvapich/gdr/2.3.6/gcc4.8.5-cuda11.2
#PBS -M kanakawapanman@gmail.com

module load mvapich/gdr/2.3.6/gcc4.8.5-cuda11.2
#/home/NBB/yu_mingzhe/miniconda3/bin/conda activate basic_env

current_time=$(date "+%Y%m%d%H%M")

mpiexec ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
 -genv MV2_NUM_HCAS 4 \
 -np 8 --map-by ppr:4:node:PE=3 --bind-to socket:0 --report-bindings \
 /home/NBB/yu_mingzhe/miniconda3/envs/basic_env/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/rpc_resnet_cifar.py \
 --timestamp="$current_time"