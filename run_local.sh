#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")

mpirun ${NQSII_MPIOPTS} -x PATH -np 4 python /work/NBB/yu_mingzhe/kfac-pytorch/multi_node_mobilev3_minist_ddp.py \
 --timestamp="$current_time"
