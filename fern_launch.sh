#!/bin/bash
WORLD_SIZE=4

for (( i=0; i<WORLD_SIZE; i++ ))
do
torchrun --nproc_per_node=$WORLD_SIZE --nnodes=1 --node_rank="$i" \
  --master_addr="127.0.0.1" --master_port=12325 \
  fern_resnet18_fashion_minst.py
done

echo "hhh"