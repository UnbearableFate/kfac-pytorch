#!/bin/bash
WORLD_SIZE=4

mpirun -np $WORLD_SIZE -x PATH python3 fern_mlp_fashion_minst.py

echo "hhh"