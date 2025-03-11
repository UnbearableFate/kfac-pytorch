#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")
mpirun -np 8 \
 /home/yu/miniconda3/envs/py311/bin/python /home/yu/workspace/kfac-pytorch/renew_resnet_cifar.py \
 --timestamp="$current_time" \
 --experiment-name="mlp_test" \
 --model="mlp" \
 --layers=8 \
 --dataset="FashionMNIST" \
 --train-com-method="rpc" \
 --batch-size=64 \
 --base-lr=0.001 \
 --epochs=3 \
 --optimizer-type="adamw" \
 --lr-scheduler-type="one_cycle"