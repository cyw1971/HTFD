#!/usr/bin/env bash
# HTFD on SP500 (T=32 quick script; prefer script/spx_t128.sh for the paper example)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MPLBACKEND=Agg

model_name=HTFD

python -u run.py \
  --task_name generation \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path SPX.csv \
  --model_id spx_t32 \
  --model $model_name \
  --data spx \
  --seq_len 32 \
  --d_model 64 \
  --train_epochs 200 \
  --batch_size 2000 \
  --learning_rate 0.0001 \
  --norm_mode revin \
  --export 1 \
  --des Exp \
  --itr 1
