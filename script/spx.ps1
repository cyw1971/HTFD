# HTFD on SP500 (Windows)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:MPLBACKEND = "Agg"

python -u run.py `
  --task_name generation `
  --is_training 1 `
  --root_path ./dataset/ `
  --data_path SPX.csv `
  --model_id spx_t32 `
  --model HTFD `
  --data spx `
  --seq_len 32 `
  --d_model 64 `
  --train_epochs 200 `
  --batch_size 2000 `
  --learning_rate 0.0001 `
  --norm_mode revin `
  --export 1 `
  --des Exp `
  --itr 1
