#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${1:?Usage: verify_hhr_training.sh OUTPUT_DIR}"

for layer in $(seq 2 31); do
  tag="$(printf '%02d' "$layer")"
  test -f "$OUTPUT_DIR/hash_weight_layer_${tag}.pt"
  test -f "$OUTPUT_DIR/shared_rotation_layer_${tag}.pt"
done

test "$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'hash_weight_layer_*.pt' | wc -l)" -eq 30
test "$(find "$OUTPUT_DIR" -maxdepth 1 -type f -name 'shared_rotation_layer_*.pt' | wc -l)" -eq 30

echo "HHR training outputs verified."
