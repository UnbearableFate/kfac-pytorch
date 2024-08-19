#!/bin/bash

#PBS -A NBB
#PBS -q gpu
#PBS -T openmpi
#PBS -b 4
#PBS -l elapstim_req=00:40:00
#PBS -v NQSV_MPI_VER=gdr/4.1.6/gcc8.3.1-cuda12.3.2
#PBS -M kanakawapanman@gmail.com

module load openmpi/gdr/4.1.6/gcc8.3.1-cuda12.3.2
#/home/NBB/yu_mingzhe/miniconda3/bin/conda activate basic_env

current_time=$(date "+%Y%m%d%H%M")

mpirun ${NQSII_MPIOPTS} --mca orte_base_help_aggregate 0 --mca plm_base_verbose 100 --mca ras_base_verbose 100 \
 --mca rmaps_base_verbose 100 --mca rml_base_verbose 100 --mca oob_base_verbose 100 \
 --mca btl_base_verbose 100 --mca mpi_abort_print_stack 1 \
 -x PATH -np 4 -npernode 1 --report-bindings \
 /home/NBB/yu_mingzhe/miniconda3/envs/basic_env/bin/python /work/NBB/yu_mingzhe/kfac-pytorch/mlp_fashion_minst.py \
 --timestamp="$current_time"
