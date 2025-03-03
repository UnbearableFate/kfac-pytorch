#!/bin/bash

#PBS -q debug-g
#PBS -W group_list=xg24i002
#PBS -l select=16:mpiprocs=1
#PBS -l walltime=00:29:50
#PBS -j oe

current_time=$(date "+%Y%m%d%H%M")
mpirun --mca mpi_abort_print_stack 1 \
 --report-bindings \
 /work/xg24i002/x10041/miniconda3/envs/py313/bin/python /work/xg24i002/x10041/kfac-pytorch/renew_resnet_cifar.py\
 --timestamp="$current_time" \
 --experiment-name="resnet34_rpc" \
 --train-com-method="rpc" \
 --batch-size=256 \
 --base-lr=0.2 \
 --epochs=150 \
 --warmup-epochs=10 \
 --lr-decay 40 80 120 140