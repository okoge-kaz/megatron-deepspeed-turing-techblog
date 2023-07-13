#!/bin/bash

source .env/bin/activate
wandb offline

# bash scripts/mpirun/run_custom.sh -n 1 -g 4 -m 6.7b -f hostfile_4 --tp 1 --pp 1
bash scripts/mpirun/run_custom.sh -n 1 -g 4 -m 6.7b -f hostfile_4 --tp 1 --pp 4
bash scripts/mpirun/run_custom.sh -n 1 -g 4 -m 6.7b -f hostfile_4 --tp 4 --pp 1

wandb sync --sync-all
wandb online
