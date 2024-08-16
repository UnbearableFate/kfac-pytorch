#!/bin/bash

current_time=$(date "+%Y%m%d%H%M")

mpirun ${NQSII_MPIOPTS} -x PATH -np 8 python /work/NBB/yu_mingzhe/kfac-pytorch/mobilev3_minist.py \
 --timestamp="$current_time"
