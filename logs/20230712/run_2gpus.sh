#!/bin/bash

source .env/bin/activate
wandb offline

# bash scripts/mpirun/run_custom.sh -n 1 -g 2 -m 2.7b -f hostfile_2 --tp 1 --pp 1
bash scripts/mpirun/run_custom.sh -n 1 -g 2 -m 2.7b -f hostfile_2 --tp 1 --pp 2
bash scripts/mpirun/run_custom.sh -n 1 -g 2 -m 2.7b -f hostfile_2 --tp 2 --pp 1

wandb sync --sync-all
wandb online
