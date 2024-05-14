#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 16
#PBS -l elapstim_req=02:30:00
#PBS -v NQSV_MPI_VER=4.1.6/nvhpc23.1-cuda12.0-ucx1.15.0
#PBS -M kanakawapanman@gmail.com
source /work/NBB/yu_mingzhe/venv/bin/activate

SHAREDFILE=/work/NBB/yu_mingzhe/kfac-pytorch/data/sharedfile
if [ -f $SHAREDFILE ]; then
    rm $SHAREDFILE
fi

current_time=$(date "+%Y%m%d%H%M")

/system/apps/ubuntu/22.04-202404/openmpi/4.1.6/nvhpc23.1-cuda12.0-ucx1.15.0/bin/mpirun \
 -x PATH -np 16 --map-by ppr:1:node:PE=24 --report-bindings $NQSII_MPIOPTS \
 python /work/NBB/yu_mingzhe/kfac-pytorch/pega_resnet_cifar.py \
 --timestamp=$current_time \
 --disconnect_ratio=0.4

/system/apps/ubuntu/22.04-202404/openmpi/4.1.6/nvhpc23.1-cuda12.0-ucx1.15.0/bin/mpirun \
 -x PATH -np 16 --map-by ppr:1:node:PE=24 --report-bindings $NQSII_MPIOPTS \
 python /work/NBB/yu_mingzhe/kfac-pytorch/pega_resnet_cifar.py \
 --timestamp=$current_time \
 --disconnect_ratio=0.6