#!/bin/bash

#PBS -A NBB
#PBS -q gpu_low
#PBS -T openmpi
#PBS -b 16
#PBS -l elapstim_req=05:00:00
#PBS -v NQSV_MPI_VER=4.1.6/nvhpc24.5-cuda12.4
#PBS -M kanakawapanman@gmail.com

module load openmpi/4.1.6/nvhpc24.5-cuda12.4 

current_time=$(date "+%Y%m%d%H%M")
epochs=120
warmup_epochs=30
mpirun ${NQSII_MPIOPTS} --mca mpi_abort_print_stack 1 \
 -x PATH -np 16 --map-by ppr:1:node --report-bindings \
 /work/NBB/yu_mingzhe/miniconda3/envs/py313/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/swin_imagenet.py \
 --timestamp="$current_time" \
 --experiment-name="swin_test" \
 --model='swin' \
 --train-com-method="rpc" \
 --data-path='/work/NBB/share/datasets/imagenet1k/' \
 --epochs $epochs --batch-size 512 --opt adamw --lr 0.002 --weight-decay 0.0001 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs $warmup_epochs --lr-warmup-decay 0.01 --amp --label-smoothing 0.12 --mixup-alpha 0.96 --clip-grad-norm 5.0 --cutmix-alpha 1.2 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224