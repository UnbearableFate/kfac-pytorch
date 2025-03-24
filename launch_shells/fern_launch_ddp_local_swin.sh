#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")
mpirun -np 1 \
 /home/yu/miniconda3/envs/py311/bin/python /home/yu/workspace/kfac-pytorch/renew_resnet_cifar.py \
 --timestamp="$current_time"\
 --experiment-name="swin" \
 --kfac-inv-update-steps=100 \
 --kfac-factor-update-steps=20 \
 --batch-size=256 \
 --base-lr=0.001 \
 --epochs=2 \
 --model='swin' \
 --warmup-epochs=1 \