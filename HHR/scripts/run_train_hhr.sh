#!/usr/bin/env bash
set -euo pipefail

PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE="${DEVICE:-0}"
TRAINING_INPUT_ROOT="${TRAINING_INPUT_ROOT:?Set TRAINING_INPUT_ROOT to the prepared HHR training inputs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(dirname "$PACKAGE_ROOT")/hhr_outputs}"
R_OUTPUT="$OUTPUT_ROOT/r_stage"
W_OUTPUT="$OUTPUT_ROOT/w_stage"
FINAL_OUTPUT="$OUTPUT_ROOT/final_weights"

export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HHR_R_OUTPUT="$R_OUTPUT"
export HHR_TRAINING_INPUT_ROOT="$TRAINING_INPUT_ROOT"

mkdir -p "$R_OUTPUT" "$W_OUTPUT" "$FINAL_OUTPUT"

for layer in $(seq 2 31); do
  if [[ ! -f "$R_OUTPUT/layer_$(printf '%02d' "$layer")/report.json" ]]; then
    CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" \
      "$PACKAGE_ROOT/training/source/r/train_hhr_rotation.py" \
      --layer "$layer" --candidate-ratio dynamic --mode r_only
  fi
done

if [[ ! -f "$W_OUTPUT/train_config.json" ]]; then
  CUDA_VISIBLE_DEVICES="$DEVICE" "$PYTHON_BIN" \
    "$PACKAGE_ROOT/training/source/w/train_hhr_projection.py" \
    --page-data "$TRAINING_INPUT_ROOT/hhr_projection_training_data" \
    --initial-weight-dir "$TRAINING_INPUT_ROOT/hhr_initial_hash_weights" \
    --initial-rotation-dir "$TRAINING_INPUT_ROOT/hhr_projection_initial_rotations" \
    --save-dir "$W_OUTPUT" \
    --layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31" \
    --max-samples 21 \
    --exclude-samples "0,2,7,10,16,20" \
    --epochs 20 \
    --page-size 8 \
    --quest-ratio 0.10 \
    --head-budget "$TRAINING_INPUT_ROOT/hhr_projection_head_budget.json" \
    --sparse-ratio 0.015 \
    --num-sink 16 \
    --hash-lr 5e-5 \
    --rotation-lr 2e-5 \
    --transform-mode orthogonal \
    --hash-branch serial \
    --rotation-granularity kv_head \
    --init-transform formal \
    --freeze-rotation \
    --no-rms \
    --loss-profile teacher_exact \
    --weight-paper2-soft 1.0 \
    --weight-token-rank 1.75 \
    --weight-token-listwise 0.7 \
    --weight-quest-miss 1.0 \
    --weight-quest-slack 0.1 \
    --weight-complementarity 0.4 \
    --seed 42 \
    --device cuda:0
fi

for layer in $(seq 2 31); do
  tag="$(printf '%02d' "$layer")"
  cp "$W_OUTPUT/hash_weight_layer_${tag}.pt" "$FINAL_OUTPUT/"
  cp "$R_OUTPUT/layer_${tag}/shared_rotation_layer_${tag}.pt" "$FINAL_OUTPUT/"
done

"$PACKAGE_ROOT/scripts/verify_hhr_training.sh" "$FINAL_OUTPUT"
