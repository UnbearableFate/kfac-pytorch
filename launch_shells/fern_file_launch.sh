#!/bin/bash
echo "hhh"
WORLD_SIZE=4

mpirun -np $WORLD_SIZE -x PATH /home/yu/venv/bin/python /home/yu/workspace/kfac-pytorch/mlp_fashion_minst.py

echo "hhh"