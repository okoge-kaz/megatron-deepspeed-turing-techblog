
# distributed settings
GPUS_PER_NODE=8
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

TP_SIZE=1
PP_SIZE=2
DP_SIZE=$(($WORLD_SIZE / ($TP_SIZE * $PP_SIZE)))

echo -e "\nTP_SIZE: $TP_SIZE, PP_SIZE: $PP_SIZE, DP_SIZE: $DP_SIZE\n"

nvidia-smi nvlink --status

# load virtualenv
source .env/bin/activate

# dataset, checkpoint path
DATA_PATH=dataset/BookCorpusDataset_text_document
CHECKPOINT_PATH=checkpoints/gpt2_1.3b/${NNODES}node-${WORLD_SIZE}gpu-dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-mpirun

mkdir -p $CHECKPOINT_PATH

MICRO_BATCHSIZE=1
GLOBAL_BATCHSIZE=$(($MICRO_BATCHSIZE * $DP_SIZE))

# model parameter (1.3B)
NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=2048


MASTER_ADDR="10.2.72.141"

mpirun -np $WORLD_SIZE --npernode $GPUS_PER_NODE \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=16500 \
  -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO  -x PATH \
  -mca pml ob1 -mca btl ^openib \
  python pretrain_gpt.py \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PP_SIZE \
  --num-layers $NUM_LAYERS \
  --hidden-size $HIDDEN_SIZE \
  --num-attention-heads $NUM_ATTENTION_HEADS \
  --micro-batch-size $MICRO_BATCHSIZE \
  --global-batch-size $GLOBAL_BATCHSIZE \
  --seq-length $SEQ_LENGTH \
  --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
  --train-iters 100 \
  --lr-decay-iters 80 \
  --save $CHECKPOINT_PATH \
  --load $CHECKPOINT_PATH \
  --data-path $DATA_PATH \
  --vocab-file dataset/gpt2-vocab.json \
  --merge-file dataset/gpt2-merges.txt \
  --data-impl mmap \
  --split 900,50,50 \
  --distributed-backend nccl \
  --lr 0.00015 \
  --lr-decay-style cosine \
  --min-lr 1.0e-5 \
  --weight-decay 1e-2 \
  --clip-grad 1.0 \
  --lr-warmup-fraction .01 \
  --checkpoint-activations \
  --log-interval 1 \
  --save-interval 30000 \
  --eval-interval 1000 \
  --eval-iters 10 \
  --fp16 \
  --use-mpi \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --wandb-name "gpt2_1.3b_${NNODES}node_dp${DP_SIZE}-tp${TP_SIZE}-pp${PP_SIZE}-mpirun-mdx"
