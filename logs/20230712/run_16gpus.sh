#!/bin/bash

source .env/bin/activate
wandb offline

bash scripts/mpirun/run_custom.sh -n 2 -g 8 -m 26b -f hostfile_16 --tp 1 --pp 16
bash scripts/mpirun/run_custom.sh -n 2 -g 8 -m 26b -f hostfile_16 --tp 2 --pp 8
bash scripts/mpirun/run_custom.sh -n 2 -g 8 -m 26b -f hostfile_16 --tp 4 --pp 4

wandb sync --sync-all
wandb online
