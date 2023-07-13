#!/bin/bash

source .env/bin/activate
wandb offline

bash scripts/mpirun/run_custom.sh -n 1 -g 8 -m 13b -f hostfile_8 --tp 1 --pp 8
bash scripts/mpirun/run_custom.sh -n 1 -g 8 -m 13b -f hostfile_8 --tp 2 --pp 4
bash scripts/mpirun/run_custom.sh -n 1 -g 8 -m 13b -f hostfile_8 --tp 4 --pp 2

wandb sync --sync-all
wandb online
