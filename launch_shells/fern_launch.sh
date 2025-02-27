#!/bin/bash
mpirun --host fern01,fern02 -np 2 -map-by ppr:1:node /home/yu/miniconda3/envs/py311/bin/python /home/yu/workspace/kfac-pytorch/mlp_fashion_minst_ddp.py