# HHR

This anonymous package contains the released HHR training and LongBench
evaluation code. Model weights, prepared training tensors, datasets, and
generated results are not included.

## Environment

Use Linux, Python 3.10 or later, a CUDA-capable PyTorch installation, and a
compatible CUDA toolkit. Install the Python dependencies and build the CUDA
extension from the repository root:

```bash
python3 -m pip install -r requirements.txt
bash scripts/fetch_third_party.sh
bash scripts/build_extension.sh
python3 scripts/check_environment.py
python3 scripts/verify_manifest.py
python3 tests/test_public_configs.py
python3 tests/audit_release.py .
```

## Training inputs

Set `TRAINING_INPUT_ROOT` to a directory containing the prepared tensors used
by the released HHR training workflow:

```text
TRAINING_INPUT_ROOT/
  hhr_rotation_training_manifest.json
  hhr_rotation_head_budget.json
  hhr_projection_head_budget.json
  hhr_initial_hash_weights/
  hhr_initial_rotation/rotation.pt
  hhr_projection_initial_rotations/
  hhr_projection_training_data/layerXX/sampleXXXX.pt
```

The prepared tensors must be produced with the same model revision, tokenizer,
sample ordering, page size, and random seed as the intended experiment. The
package intentionally does not distribute these tensors.

## Train HHR

The release has one supported training entry point:

The released configuration keeps the R-stage procedure unchanged and trains
the W projection for 20 epochs.

```bash
TRAINING_INPUT_ROOT=/path/to/prepared-inputs \
OUTPUT_ROOT=/path/to/hhr-run \
DEVICE=0 \
PYTHON_BIN=python3 \
bash scripts/run_train_hhr.sh
```

The trained projection and rotation tensors are written under
`$OUTPUT_ROOT/final_weights`. Generated files remain outside the source tree.

## Evaluate on LongBench

Run one LongBench task with the trained projection and rotation directories:

```bash
MODEL_PATH=/path/to/model \
DATASET_PATH=/path/to/LongBench \
HASH_WEIGHTS_PATH=/path/to/hhr-run/final_weights \
ROTATION_PATH=/path/to/hhr-run/final_weights \
DEVICE=0 \
PYTHON_BIN=python3 \
bash scripts/run_hhr_longbench.sh TASK /path/to/output MAX_SAMPLES
```

`TASK` is a LongBench dataset identifier and `MAX_SAMPLES` controls the number
of evaluated examples. Predictions, timing records, audit records, and scorer
outputs are written to the selected output directory.

## Source layout

```text
configs/       Released sparse-attention configuration
evaluation/    LongBench runner and scorer
python/        HHR runtime and kernels
scripts/       Build, training, evaluation, and verification entry points
src/           CUDA extension
training/      Released projection and rotation trainers
tests/         Configuration and release-hygiene checks
```

Third-party components retain their upstream notices. See `THIRD_PARTY.md`.
