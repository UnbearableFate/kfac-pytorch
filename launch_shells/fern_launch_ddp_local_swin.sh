#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")
epochs=100
warmup_epochs=$(($epochs/5))
mpirun -np 1 \
 /home/yu/miniconda3/envs/py311/bin/python /home/yu/workspace/kfac-pytorch/swin_imagenet.py \
 --timestamp="$current_time"\
 --experiment-name="swin_test" \
 --kfac-inv-update-steps=100 \
 --kfac-factor-update-steps=20 \
 --model='swin' \
 --data-path='/home/yu/datasets/imagenet' \
 --epochs $epochs --batch-size 256 --opt adamw --lr 0.001 --weight-decay 0.05 --norm-weight-decay 0.0  --bias-weight-decay 0.0 --transformer-embedding-decay 0.0 --lr-scheduler cosineannealinglr --lr-min 0.00001 --lr-warmup-method linear  --lr-warmup-epochs $warmup_epochs --lr-warmup-decay 0.01 --amp --label-smoothing 0.1 --mixup-alpha 0.8 --clip-grad-norm 5.0 --cutmix-alpha 1.0 --random-erase 0.25 --interpolation bicubic --auto-augment ta_wide --model-ema --ra-sampler --ra-reps 4  --val-resize-size 224