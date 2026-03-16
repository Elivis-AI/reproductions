#!/usr/bin/env bash
set -euo pipefail


python train.py \
  --device=cuda \
  --dataset=cifar10 \
  --data_path=./data/cifar10 \
  --output_dir=./output/cifar10 \
  --batch_size=64 \
  --accum_iter=1 \
  --eval_frequency=5 \
  --epochs=300 \
  --class_drop_prob=1.0 \
  --cfg_scale=0.0 \
  --compute_fid \
  --ode_method=heun2 \
  --ode_options='{"nfe": 50}' \
  --use_ema \
  --edm_schedule \
  --fid_samples 10 \
  --skewed_timesteps
