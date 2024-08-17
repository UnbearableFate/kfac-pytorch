#!/bin/zsh
WORLD_SIZE=4

mpirun --mca orte_base_help_aggregate 0 --mca plm_base_verbose 100 --mca ras_base_verbose 100 \
  --mca rmaps_base_verbose 100 --mca rml_base_verbose 100 --mca oob_base_verbose 100 \
  --mca btl_base_verbose 100 --mca mpi_abort_print_stack 1 \
  -np $WORLD_SIZE -x PATH /opt/homebrew/bin/python3.11 mlp_fashion_minst.py

echo "hhh"