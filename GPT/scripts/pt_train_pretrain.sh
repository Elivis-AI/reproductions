#!/bin/bash

LOG_DIR=./logs
export WANDB_MODE="offline"
# export WANDB_MODE="online"


# 设置分布式参数
NNODES=1        # 节点数
NPROC_PER_NODE=1 # 每个节点 GPU 数
NODE_RANK=0     # 节点 rank
MASTER_ADDR="localhost"
MASTER_PORT=29500

# export CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  pt_train_pretrain.py \
  --data_path ./data/pretrain_hq.jsonl \
  --tokenizer ./tokenizer \
  --epochs 3 \
  --batch_size 64 \
  --lr 2e-4 \
  --save_dir ./results/pretrain \
  --max_seq_len 1024 \
  --use_compile