#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to the Llama-3.1-8B-Instruct directory}"
DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to the LongBench directory}"
HASH_WEIGHTS_PATH="${HASH_WEIGHTS_PATH:?Set HASH_WEIGHTS_PATH to the trained W directory}"
ROTATION_PATH="${ROTATION_PATH:?Set ROTATION_PATH to the trained R directory}"
DEVICE="${DEVICE:-0}"
TASK="${1:?Usage: run_hhr_longbench.sh TASK OUTPUT_DIR MAX_SAMPLES}"
OUTPUT_DIR="${2:?Usage: run_hhr_longbench.sh TASK OUTPUT_DIR MAX_SAMPLES}"
MAX_SAMPLES="${3:?Usage: run_hhr_longbench.sh TASK OUTPUT_DIR MAX_SAMPLES}"

export QH_PREFILL_CHUNK_SIZE="${QH_PREFILL_CHUNK_SIZE:-4096}"
export QH_QUEST_DYNAMIC="${QH_QUEST_DYNAMIC:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-42}"
export HHR_AUDIT_PATH="${HHR_AUDIT_PATH:-${OUTPUT_DIR}.hhr_audit.jsonl}"

"${PYTHON_BIN}" "${ROOT}/evaluation/run_evaluation.py" \
  --model-family llama31 \
  --model-name Llama-3.1-8B-Instruct \
  --model-path "${MODEL_PATH}" \
  --model-max-length 131072 \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --tasks "${TASK}" \
  --config "${ROOT}/configs/llama31_sparse.ini" \
  --head-budget "${ROOT}/configs/head_budget_q30_page8.json" \
  --hash-weights "${HASH_WEIGHTS_PATH}" \
  --shared-rotation "${ROTATION_PATH}" \
  --topk-ratio 0.015 \
  --candidate-ratio 0.30 \
  --max-samples "${MAX_SAMPLES}" \
  --seed 42 \
  --device "${DEVICE}" \
  --score
