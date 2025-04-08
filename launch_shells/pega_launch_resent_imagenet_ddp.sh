#!/bin/bash

#PBS -A NBBG
#PBS -q gen_S
#PBS -T openmpi
#PBS -b 16
#PBS -l elapstim_req=05:00:00
#PBS -v NQSV_MPI_VER=4.1.6/nvhpc24.5-cuda12.4
#PBS -M kanakawapanman@gmail.com

module load openmpi/4.1.6/nvhpc24.5-cuda12.4 

current_time=$(date "+%Y%m%d%H%M")
mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
 -x PATH -np 16 --map-by ppr:1:node --report-bindings \
 /work/NBB/yu_mingzhe/miniconda3/envs/py313/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/resnet_imagenet.py \
 --timestamp="$current_time" \
 --experiment-name="resnet_ddp_test" \
 --model='resnet50' \
 --kfac-inv-update-steps=200 \
 --kfac-factor-update-steps=20 \
 --data-path='/work/NBB/share/datasets/imagenet1k/' \
 --epochs 100 \
 --batch-size 256 \
 --opt adamw \
 --lr 0.002 \
 --weight-decay 0.001 \
 --norm-weight-decay 0.0 \
 --bias-weight-decay 0.0 \
 --transformer-embedding-decay 0.0 \
 --lr-scheduler cosineannealinglr \
 --lr-min 0.00001 \
 --lr-warmup-method linear \
 --lr-warmup-epochs 20 \
 --lr-warmup-decay 0.01 \
 --amp \
 --label-smoothing 0.1 \
 --mixup-alpha 0.8 \
 --clip-grad-norm 5.0 \
 --cutmix-alpha 1.0 \
 --random-erase 0.25 \
 --interpolation bicubic \
 --auto-augment ta_wide \
 --model-ema \
 --ra-sampler \
 --ra-reps 4 \
 --val-resize-size 224 \
 --recover \