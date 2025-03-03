#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")
mpirun -np 3 \
 /home/yu/miniconda3/envs/py311/bin/python /home/yu/workspace/kfac-pytorch/renew_resnet_cifar.py \
 --timestamp="$current_time" \
 --experiment-name="resnet20_rpc" \
 --model="resnet20" \
 --train-com-method="rpc" \
 --batch-size=256 \
 --base-lr=0.2 \
 --epochs=150 \
 --warmup-epochs=10 \
 --lr-decay 40 80 120 140
