#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 4
#PBS -l elapstim_req=00:10:00
#PBS -v NQSV_MPI_VER=gdr/4.1.6/gcc8.3.1-cuda12.3.2
#PBS -M kanakawapanman@gmail.com

module load openmpi/gdr/4.1.6/gcc8.3.1-cuda12.3.2
module load python/3.8
source /work/NBB/yu_mingzhe/basicenv/bin/activate

current_time=$(date "+%Y%m%d%H%M")

mpirun ${NQSII_MPIOPTS} -np 4 --map-by ppr:1:node:PE=16 --report-bindings \
 /work/NBB/yu_mingzhe/basicenv/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/mlp_fashion_minst.py \
 --timestamp="$current_time"