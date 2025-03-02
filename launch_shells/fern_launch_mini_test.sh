#!/bin/bash
current_time=$(date "+%Y%m%d%H%M")

mpirun --host fern01,fern02 -np 8 -map-by ppr:4:node \
 /home/yu/miniconda3/envs/py311/bin/python \
 /home/yu/workspace/kfac-pytorch/mlp_fashion_minst.py \
 --timestamp="$current_time" \
 --not_kfac