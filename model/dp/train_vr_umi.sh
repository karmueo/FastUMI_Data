#!/usr/bin/env bash
# 使用全部 RM75 末端数据从头训练 120 轮；由用户显式启动，不继承验证实验模型。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
DATASET="$(realpath ../../dataset/vr_target_umi/target.zarr)"
RUN_ROOT="$(realpath ../../dataset/vr_target_umi)/runs"
RUN_DIR="${1:-$RUN_ROOT/full_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname -- "$RUN_DIR")"
mkdir "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"

exec env -u PYTHONPATH \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
  "$SCRIPT_DIR/.venv/bin/python" train.py \
  --config-name=train_diffusion_unet_timm_vr_umi_workspace \
  "task.dataset_path=$DATASET" \
  "hydra.run.dir=$RUN_DIR" \
  training.num_epochs=120 training.resume=false \
  dataloader.batch_size=32 val_dataloader.batch_size=32
