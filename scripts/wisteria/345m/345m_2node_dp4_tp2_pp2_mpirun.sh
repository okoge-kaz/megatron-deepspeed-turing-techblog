#!/bin/bash
#PJM -L rscgrp=tutorial1-a
#PJM -L node=2
#PJM -N gpt
#PJM --mpi proc=16
#PJM -L elapse=06:00:00
#PJM -g gt01
module load cuda/11.8
module use /work/share/modulefiles/nvidia/22.11
module load ompi-cuda/4.1.5-11.8
module load gcc-toolset/9

# Runs the "345M" parameter model

MASTER_ADDR=$(head -n 1 $PJM_O_NODEINF)

# distributed settings
GPUS_PER_NODE=8
NNODES=2
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

TP_SIZE=2
PP_SIZE=2
DP_SIZE=$(($WORLD_SIZE / ($TP_SIZE * $PP_SIZE)))

echo -e "\nTP_SIZE: $TP_SIZE, PP_SIZE: $PP_SIZE, DP_SIZE: $DP_SIZE\n"

nvidia-smi nvlink --status

# load virtualenv
source /work/gt01/t01013/.bashrc
source .env/bin/activate

# dataset, checkpoint path
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2_345m/${NNODES}node-${WORLD_SIZE}gpu-dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-mpirun

mkdir -p $CHECKPOINT_PATH

MICRO_BATCHSIZE=8
GLOBAL_BATCHSIZE=$(($MICRO_BATCHSIZE * $DP_SIZE))

# Open MPI training
GPUS_PER_NODE=`nvidia-smi -L | wc -l`

mpirun -np $WORLD_SIZE -machinefile ${PJM_O_NODEINF} \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=16500 \
  -map-by ppr:${GPUS_PER_NODE}:node -mca pml ob1 \
  -x NCCL_DEBUG=INFO  -x PATH \
  python pretrain_gpt.py \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PP_SIZE \
  --num-layers 24 \
  --hidden-size 1024 \
  --num-attention-heads 16 \
  --micro-batch-size $MICRO_BATCHSIZE \
  --global-batch-size $GLOBAL_BATCHSIZE \
  --seq-length 1024 \
  --max-position-embeddings 1024 \
  --train-iters 500000 \
  --lr-decay-iters 320000 \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $DATA_PATH \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --data-impl mmap \
  --split 949,50,1 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --checkpoint-activations \
  --log-interval 1 \
  --save-interval 3000 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --fp16 \
  --use-mpi \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --wandb-name "gpt2_345m_${NNODES}node_dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-mpirun-wisteria"
