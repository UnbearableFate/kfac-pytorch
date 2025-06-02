#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")
mpirun -np 1 \
 python renew_resnet_cifar.py \
 --timestamp="$current_time"\
 --experiment-name="swin_kfac" \
 --batch-size=256 \
 --base-lr=0.001 \
 --epochs=50 \
 --lr-scheduler-type="one_cycle" \
 --model="swin" \
 --optimizer-type="adamw" \
 --not-kfac \