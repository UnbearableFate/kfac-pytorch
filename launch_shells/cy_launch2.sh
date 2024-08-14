#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 4
#PBS -l elapstim_req=00:20:00
#PBS -v NQSV_MPI_VER=gdr/4.1.6/gcc8.3.1-cuda12.3.2
#PBS -M kanakawapanman@gmail.com

module load openmpi/gdr/4.1.6/gcc8.3.1-cuda12.3.2
#/home/NBB/yu_mingzhe/miniconda3/bin/conda activate py310_env

current_time=$(date "+%Y%m%d%H%M")

mpirun ${NQSII_MPIOPTS} -x PATH -np 4 --map-by ppr:1:node --bind-to none --report-bindings \
 /home/NBB/yu_mingzhe/miniconda3/envs/py310_env/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/mlp_fashion_minst.py \
 --timestamp="$current_time"
