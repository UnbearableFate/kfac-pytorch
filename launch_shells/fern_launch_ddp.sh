#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")
mpirun --host fern01,fern02 \
 -np 2 -map-by ppr:1:node \
 /home/yu/miniconda3/envs/py311/bin/python /home/yu/workspace/kfac-pytorch/renew_resnet_cifar.py \
 --timestamp="$current_time"\
 --experiment-name="resnet32_ddp" \
 --kfac-inv-update-steps=60 \
 --kfac-factor-update-steps=15 \
 --batch-size=256 \
 --base-lr=0.2 \
 --epochs=100