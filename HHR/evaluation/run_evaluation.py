#!/usr/bin/env python3
"""Run the released HHR method."""

from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LOW_LEVEL_RUNNER = Path(__file__).resolve().parent / "runtime_runner" / "run_pred.py"
SCORER = Path(__file__).resolve().parent / "score_longbench.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise SystemExit(f"{label} does not exist: {path}")
    return path


def rotation_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(directory.glob("shared_rotation_layer_*.pt")):
        layer = path.stem.removeprefix("shared_rotation_layer_")
        if not layer.isdigit():
            raise SystemExit(f"invalid rotation filename: {path.name}")
        hashes[str(int(layer))] = sha256(path)
    if not hashes:
        raise SystemExit(f"no rotation files found in {directory}")
    return hashes


def hash_weight_hashes(directory: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(directory.glob("hash_weight_layer_*.pt")):
        layer = path.stem.removeprefix("hash_weight_layer_")
        if not layer.isdigit():
            raise SystemExit(f"invalid hash-weight filename: {path.name}")
        hashes[str(int(layer))] = sha256(path)
    if not hashes:
        raise SystemExit(f"no hash-weight files found in {directory}")
    return hashes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-family", choices=("llama31", "qwen3_4b"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-max-length", type=int, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--head-budget", type=Path, required=True)
    parser.add_argument("--hash-weights", type=Path, required=True)
    parser.add_argument("--shared-rotation", type=Path, required=True)
    parser.add_argument("--topk-ratio", type=float, default=0.015)
    parser.add_argument("--candidate-ratio", type=float, default=0.30)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pipeline-parallel-gpus", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = require_directory(args.model_path, "model path")
    dataset_path = require_directory(args.dataset_path, "dataset path")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not 0.0 < args.topk_ratio <= 1.0:
        raise SystemExit("topk ratio must be in (0, 1]")
    if not args.topk_ratio <= args.candidate_ratio <= 1.0:
        raise SystemExit("candidate ratio must be between topk ratio and 1")

    config_path = require_file(args.config, "config")
    head_budget_path = require_file(args.head_budget, "head budget")
    hash_weights_path = require_directory(args.hash_weights, "hash weights")
    shared_rotation_path = require_directory(args.shared_rotation, "shared rotation")
    weights = hash_weight_hashes(hash_weights_path)
    rotations = rotation_hashes(shared_rotation_path)

    config = configparser.ConfigParser()
    config.read(config_path)
    configured_topk = config.getfloat("dataset", "TOPK_RATIO")
    configured_candidate = config.getfloat("quest", "RATIO")
    if configured_topk != args.topk_ratio:
        raise SystemExit("topk ratio does not match the config")
    if configured_candidate != args.candidate_ratio:
        raise SystemExit("candidate ratio does not match the config")

    command = [
        sys.executable,
        str(LOW_LEVEL_RUNNER),
        "--model_name",
        args.model_name,
        "--model_name_or_path",
        str(model_path),
        "--model_maxlen",
        str(args.model_max_length),
        "--dataset_name",
        "longbench",
        "--dataset_path",
        str(dataset_path),
        "--output_dir",
        str(output_dir),
        "--method",
        "hash",
        "--write_in_time",
        "--mp_num",
        "1",
        "--pp_num",
        str(args.pipeline_parallel_gpus),
        "--min_seq_len",
        "0",
        "--tasks",
        args.tasks,
        "--max_samples",
        str(args.max_samples),
    ]
    command.extend(("--config_file", str(config_path)))
    if args.resume:
        command.append("--resume")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["QH_SM120_PORTABLE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT), str(ROOT / "python"), str(LOW_LEVEL_RUNNER.parent))
    )
    env["QH_QUEST_DYNAMIC"] = "0"
    env["QH_QUEST_HEAD_BUDGET_PATH"] = str(head_budget_path)
    env["QH_SHARED_ROTATION_PATH"] = str(shared_rotation_path)
    env["HHR_WEIGHTS_PATH"] = str(hash_weights_path)

    manifest = {
        "schema": 1,
        "model_family": args.model_family,
        "model_name": args.model_name,
        "model_path": str(model_path),
        "dataset_path": str(dataset_path),
        "tasks": args.tasks,
        "method": "HHR",
        "topk_ratio": args.topk_ratio,
        "candidate_ratio": args.candidate_ratio,
        "seed": args.seed,
        "config_sha256": sha256(config_path),
        "head_budget_sha256": sha256(head_budget_path),
        "hash_weight_sha256_by_layer": weights,
        "rotation_sha256_by_layer": rotations,
        "state": "running",
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        subprocess.run(command, check=True, env=env, cwd=LOW_LEVEL_RUNNER.parent)
        if args.score:
            subprocess.run(
                (sys.executable, str(SCORER), "--model", str(output_dir)),
                check=True,
                env=env,
                cwd=SCORER.parent,
            )
    except BaseException as error:
        manifest["state"] = "failed"
        manifest["error"] = repr(error)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise

    manifest["state"] = "completed"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
