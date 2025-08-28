#!/bin/bash

OUTPUT_DIR=./pretrain
LOG_DIR=./logs
export WANDB_MODE="offline"
# WANDB_MODE="online"


# 设置分布式参数
NNODES=1        # 节点数
NPROC_PER_NODE=1 # 每个节点 GPU 数
NODE_RANK=0     # 节点 rank
MASTER_ADDR="localhost"
MASTER_PORT=29500
OMP_NUM_THREADS=8

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$NPROC_PER_NODE \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  hf_train_lora.py \
  --dataset_path "./lora_identity.jsonl" \
  --model_path "./sft" \
  --output_dir "./lora" \
  --per_device_train_batch_size 4 \
  --num_train_epochs 20 \
  --max_seq_len 2048 \
  --lr_scheduler_type="cosine" \
  --warmup_ratio=0.0 \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 2000 \
  --logging_dir $LOG_DIR \
  --logging_steps 10 \
  --save_total_limit 2 \
  --report_to "wandb" \
  --run_name "GPT-LoRA" \
  --learning_rate 1e-4 \
  --fp16 \
  --remove_unused_columns False \
  --max_grad_norm 1.0 \
  --weight_decay 0.01 \
  --save_safetensors False