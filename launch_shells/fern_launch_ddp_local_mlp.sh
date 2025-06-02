#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")
mpirun -np 1 \
 /home/yu/miniconda3/envs/py311/bin/python /home/yu/workspace/kfac-pytorch/renew_resnet_cifar.py \
 --timestamp="$current_time"\
 --experiment-name="resnet32_ddp_multi_step_adamw" \
 --kfac-inv-update-steps=1 \
 --kfac-factor-update-steps=1 \
 --batch-size=256 \
 --base-lr=0.001 \
 --epochs=50 \
 --model="mlp" \
 --layers=8 \
 --dataset="FashionMNIST" \
 --optimizer-type="adamw" \