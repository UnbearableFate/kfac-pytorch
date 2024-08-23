#!/bin/zsh
WORLD_SIZE=4

mpirun -np $WORLD_SIZE -x PATH /opt/homebrew/bin/python3.11 mlp_fashion_minst_ddp.py

echo "hhh"