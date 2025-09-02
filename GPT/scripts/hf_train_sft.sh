#!/bin/bash

OUTPUT_DIR=./sft
LOG_DIR=./logs
WANDB_MODE="offline"
# export WANDB_MODE="online"


# 设置分布式参数
NNODES=1        # 节点数
NPROC_PER_NODE=8 # 每个节点 GPU 数
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
  hf_train_sft.py \
  --dataset_path "./data/sft_1024.jsonl" \
  --model_path "./pretrain" \
  --output_dir $OUTPUT_DIR \
  --per_device_train_batch_size 32 \
  --num_train_epochs 5 \
  --max_seq_len 4096 \
  --lr_scheduler_type="cosine" \
  --warmup_ratio=0.0 \
  --eval_strategy "no" \
  --save_strategy "epoch" \
  # --save_steps 10000 
  --logging_dir $LOG_DIR \
  --logging_steps 100 \
  --save_total_limit 2 \
  --report_to "wandb" \
  --run_name "GPT-SFT" \
  --learning_rate 5e-5 \
  --fp16 \
  --remove_unused_columns False \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --save_safetensors False