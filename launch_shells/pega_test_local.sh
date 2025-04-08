#!/bin/bash

# cd /work/NBB/yu_mingzhe/kfac-pytorch
# module load openmpi/4.1.6/nvhpc24.5-cuda12.4
# conda activate py313

current_time=$(date "+%Y%m%d%H%M")
epochs=100
warmup_epochs=$(($epochs/5))
mpirun -np 2 \
 /work/NBB/yu_mingzhe/miniconda3/envs/py313/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/swin_imagenet.py \
 --timestamp="$current_time"\
 --experiment-name="swin_test" \
 --model='swin' \
 --train-com-method="rpc" \
 --data-path='/work/NBB/share/datasets/imagenet1k/' \
 --epochs $epochs --batch-size 256 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs $warmup_epochs --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224