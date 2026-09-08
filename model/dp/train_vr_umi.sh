#!/usr/bin/env bash
# 使用全部 RM75 末端数据从头训练 120 轮；默认双卡 BF16，不继承验证实验模型。
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

for name in NUM_PROCESSES TRAIN_BATCH_SIZE VAL_BATCH_SIZE; do
  value="${!name}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  fi
done
case "$MIXED_PRECISION" in
  no|fp16|bf16) ;;
  *)
    echo "MIXED_PRECISION must be one of: no, fp16, bf16; got: $MIXED_PRECISION" >&2
    exit 2
    ;;
esac

DATASET="$(realpath ../../dataset/vr_target_umi/target.zarr)"
RUN_ROOT="$(realpath ../../dataset/vr_target_umi)/runs"
RUN_DIR="${1:-$RUN_ROOT/full_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname -- "$RUN_DIR")"
mkdir "$RUN_DIR"
RUN_DIR="$(realpath "$RUN_DIR")"

GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * TRAIN_BATCH_SIZE))
if [[ -n "${HF_HUB_CACHE:-}" ]]; then
  RESOLVED_HF_HUB_CACHE="$HF_HUB_CACHE"
elif [[ -n "${HF_HOME:-}" ]]; then
  RESOLVED_HF_HUB_CACHE="$HF_HOME/hub"
else
  RESOLVED_HF_HUB_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface/hub"
fi

printf 'RM75 UMI training configuration:\n'
printf '  processes: %s\n' "$NUM_PROCESSES"
printf '  train batch per GPU: %s\n' "$TRAIN_BATCH_SIZE"
printf '  validation batch per GPU: %s\n' "$VAL_BATCH_SIZE"
printf '  global train batch: %s\n' "$GLOBAL_BATCH_SIZE"
printf '  mixed precision: %s\n' "$MIXED_PRECISION"
printf '  visible GPUs: %s\n' "${CUDA_VISIBLE_DEVICES:-all}"
printf '  Hugging Face Hub cache: %s\n' "$RESOLVED_HF_HUB_CACHE"
printf '  run directory: %s\n' "$RUN_DIR"

ACCELERATE_ARGS=(
  launch
  --num_processes "$NUM_PROCESSES"
  --num_machines 1
  --mixed_precision "$MIXED_PRECISION"
  --dynamo_backend no
)
if ((NUM_PROCESSES > 1)); then
  ACCELERATE_ARGS+=(--multi_gpu)
fi

exec env -u PYTHONPATH \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  HF_HUB_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$RUN_DIR" \
  "$SCRIPT_DIR/.venv/bin/accelerate" "${ACCELERATE_ARGS[@]}" train.py \
  --config-name=train_diffusion_unet_timm_vr_umi_workspace \
  "task.dataset_path=$DATASET" \
  "hydra.run.dir=$RUN_DIR" \
  training.num_epochs=120 training.resume=false \
  "dataloader.batch_size=$TRAIN_BATCH_SIZE" \
  "val_dataloader.batch_size=$VAL_BATCH_SIZE"
