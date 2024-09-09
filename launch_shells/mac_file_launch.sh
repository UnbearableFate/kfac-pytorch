#!/bin/zsh
WORLD_SIZE=4

mpirun -np $WORLD_SIZE -x PATH python mlp_fashion_minst.py

echo "hhh"