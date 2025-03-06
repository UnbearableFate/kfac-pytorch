#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")
mpirun -np 3 \
 /home/yu/miniconda3/envs/py311/bin/python /home/yu/workspace/kfac-pytorch/renew_resnet_cifar.py \
 --timestamp="$current_time" \
 --experiment-name="resnet18_rpc" \
 --model="resnet" \
 --layers=18 \
 --train-com-method="rpc" \
 --batch-size=256 \
 --base-lr=0.001 \
 --epochs=2 \
 --warmup-epochs=5 \
 --optimizer-type="adamw" \
 --lr-decay 20 40 55
