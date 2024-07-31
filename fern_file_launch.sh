#!/bin/bash
echo "hhh"
WORLD_SIZE=4

mpirun -np $WORLD_SIZE -x PATH /home/yu/venv/bin/python /home/yu/workspace/kfac-pytorch/cnn_cifar10.py

echo "hhh"