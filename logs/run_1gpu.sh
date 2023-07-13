#!/bin/bash

source .env/bin/activate
wandb offline

bash scripts/mpirun/run_custom.sh -n 1 -g 1 -m 1.3b -f hostfile_1 --tp 1 --pp 1

wandb sync --sync-all
wandb online
