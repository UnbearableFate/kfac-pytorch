#!/bin/zsh
WORLD_SIZE=8

mpirun -np $WORLD_SIZE -x PATH /opt/homebrew/bin/python3.11 mlp_fashion_minst.py

echo "hhh"