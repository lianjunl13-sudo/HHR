"""Train the HHR projection on page-aware traces."""

import argparse
import glob
import json
import os
import random

import torch

from hhr_trainer import (
    JointLossWeights,
    JointTrainConfig,
    UnifiedQuestHashTrainer,
)


def scalar_metrics(result):
    return {
        key: float(value.detach().float().cpu())
        for key, value in result.items()
        if value.numel() == 1
    }


def load_head_ratios(path, layer, kv_heads):
    """Read the same per-layer/per-KV-head budget JSON used by inference."""
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("head_axis") not in (None, "kv_head"):
        raise ValueError("head budget must use the kv_head axis")
    ratios = payload.get("layers", {}).get(str(layer), payload.get("default"))
    if ratios is None:
        raise KeyError(f"head budget has no ratios for layer {layer}")
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != kv_heads:
        raise ValueError(
            f"layer {layer} has {len(ratios)} ratios; expected {kv_heads}"
        )
    if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
        raise ValueError(f"invalid head ratio for layer {layer}: {ratios}")
    return ratios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-data", required=True)
    parser.add_argument("--initial-weight-dir", required=True)
    parser.add_argument(
        "--initial-rotation-dir",
        default="",
        help=(
            "Optional directory containing shared_rotation_layer_*.pt. "
            "This keeps the Hash initialization fixed while testing a "
            "specific archived R artifact."
        ),
    )
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--layers", default="8,16,24,31")
    parser.add_argument("--max-samples", type=int, default=13)
    parser.add_argument(
        "--exclude-samples",
        default="",
        help="Comma-separated numeric sample ids reserved for validation",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--page-size", type=int, default=8)
    parser.add_argument("--quest-ratio", type=float, default=0.10)
    parser.add_argument(
        "--head-budget",
        default="",
        help=(
            "Deployment head-budget JSON. Its fixed per-layer/per-KV-head "
            "ratios override --quest-ratio/--quest-ratios."
        ),
    )
    parser.add_argument(
        "--quest-ratios",
        default="",
        help="Optional comma-separated candidate-ratio curriculum",
    )
    parser.add_argument("--sparse-ratio", type=float, default=0.015)
    parser.add_argument("--num-sink", type=int, required=True)
    parser.add_argument("--hash-lr", type=float, default=2e-3)
    parser.add_argument("--rotation-lr", type=float, default=2e-4)
    parser.add_argument(
        "--transform-mode",
        choices=["identity", "orthogonal", "paired_nonorth"],
        required=True,
    )
    parser.add_argument(
        "--hash-branch", choices=["parallel", "serial"], required=True
    )
    parser.add_argument(
        "--rotation-granularity",
        choices=["shared", "kv_head"],
        default="shared",
    )
    parser.add_argument(
        "--allow-orthogonal-parallel",
        action="store_true",
        help=(
            "Explicit opt-in for the experimental split path: R is used only "
            "by QUEST while Hash consumes the unrotated Q/K coordinates."
        ),
    )
    parser.add_argument(
        "--freeze-hash",
        action="store_true",
        help="Freeze the archived Hash projection to isolate R-only learning.",
    )
    parser.add_argument("--nonorth-source-scale", type=float, default=0.05)
    parser.add_argument("--condition-target", type=float, default=4.0)
    parser.add_argument(
        "--init-transform",
        choices=["identity", "formal"],
        default="identity",
        help=(
            "Identity trains from scratch; formal composes a learnable "
            "head-specific delta with the archived rotation."
        ),
    )
    parser.add_argument(
        "--freeze-rotation",
        action="store_true",
        help="Keep QUEST/search coordinates fixed and train only Hash weights.",
    )
    parser.add_argument(
        "--no-rms",
        action="store_true",
        help="Deprecated compatibility flag; strict mainline already disables RMS.",
    )
    parser.add_argument(
        "--use-rms",
        action="store_true",
        help="Enable RMS normalization instead of the released HHR setting.",
    )
    parser.add_argument(
        "--loss-profile",
        choices=["v1", "teacher_joint", "teacher_exact", "ste_conservative", "gqa_mass", "gqa_balanced"],
        default="ste_conservative",
    )
    parser.add_argument("--weight-paper2-soft", type=float, default=1.0)
    parser.add_argument("--weight-token-rank", type=float, default=1.0)
    parser.add_argument("--weight-token-listwise", type=float, default=0.35)
    parser.add_argument("--weight-quest-miss", type=float, default=1.0)
    parser.add_argument("--weight-quest-slack", type=float, default=0.10)
    parser.add_argument("--weight-complementarity", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if (
        args.transform_mode == "orthogonal"
        and args.hash_branch != "serial"
        and not args.allow_orthogonal_parallel
    ):
        raise ValueError(
            "orthogonal+parallel is an experimental split path and requires "
            "--allow-orthogonal-parallel"
        )
    if args.transform_mode == "identity" and args.init_transform != "identity":
        raise ValueError("identity transform cannot load a formal transform")
    if args.use_rms and args.no_rms:
        raise ValueError("--use-rms and --no-rms are mutually exclusive")
    if not args.head_budget:
        raise ValueError(
            "strict training requires --head-budget so training and inference "
            "use identical fixed per-layer/per-KV-head QUEST ratios"
        )

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    all_history = {}
    ratio_curriculum = (
        [float(x) for x in args.quest_ratios.split(",") if x.strip()]
        if args.quest_ratios
        else [args.quest_ratio]
    )

    for layer in [int(x) for x in args.layers.split(",")]:
        excluded = {
            int(x) for x in args.exclude_samples.split(",") if x.strip()
        }
        paths = sorted(
            glob.glob(os.path.join(args.page_data, f"layer{layer:02d}", "sample*.pt"))
        )[: args.max_samples]
        paths = [
            path for path in paths
            if int(os.path.basename(path)[6:10]) not in excluded
        ]
        if not paths:
            raise FileNotFoundError(f"no samples for layer {layer}")
        initial_path = os.path.join(
            args.initial_weight_dir, f"hash_weight_layer_{layer:02d}.pt"
        )
        initial = torch.load(initial_path, map_location="cpu", weights_only=True).float()
        head_ratios = (
            load_head_ratios(args.head_budget, layer, initial.shape[0])
            if args.head_budget
            else None
        )
        rotation_dir = args.initial_rotation_dir or args.initial_weight_dir
        initial_rotation_path = os.path.join(
            rotation_dir, f"shared_rotation_layer_{layer:02d}.pt"
        )
        initial_rotation = (
            torch.load(
                initial_rotation_path, map_location="cpu", weights_only=True
            ).float()
            if args.init_transform == "formal" and os.path.isfile(initial_rotation_path)
            else None
        )
        if args.loss_profile == "teacher_exact":
            # Exact user/teacher objective.  The upstream teacher_joint name
            # also enables oracle_gap and condition_regularizer, which are not
            # present in the declared loss and therefore cannot be used for
            # the controlled fallback experiment.
            weights = JointLossWeights(
                paper2_soft=args.weight_paper2_soft,
                token_rank=args.weight_token_rank,
                token_listwise=args.weight_token_listwise,
                quest_miss=args.weight_quest_miss,
                quest_slack=args.weight_quest_slack,
                complementarity=args.weight_complementarity,
                quantization=0.02,
                bit_balance=0.01,
                decorrelation=0.01,
                hash_orthogonal=0.01,
                rotation_regularizer=0.001,
                oracle_gap=0.0,
                condition_regularizer=0.0,
            )
        elif args.loss_profile == "teacher_joint":
            # Exact teacher-specified joint objective with STE enabled by the
            # named non-v1 profile.  R and W are both trainable unless the
            # caller explicitly requests a frozen rotation.
            weights = JointLossWeights()
        elif args.loss_profile == "gqa_mass":
            # Shared GQA retrieval should maximize the attention probability
            # captured by one common KV-token set, not merely per-head hits.
            weights = JointLossWeights(
                token_rank=1.0,
                token_listwise=0.35,
                quest_miss=0.0,
                quest_slack=0.03,
                complementarity=0.08,
                quantization=0.01,
                bit_balance=0.01,
                decorrelation=0.01,
                hash_orthogonal=0.01,
                rotation_regularizer=0.0001,
                oracle_gap=1.0,
                condition_regularizer=0.01,
            )
        elif args.loss_profile == "gqa_balanced":
            # A recall-oriented alternative with stronger bit health and
            # QUEST/Hash complementarity regularization.
            weights = JointLossWeights(
                token_rank=1.20,
                token_listwise=0.25,
                quest_miss=0.0,
                quest_slack=0.05,
                complementarity=0.10,
                quantization=0.02,
                bit_balance=0.02,
                decorrelation=0.02,
                hash_orthogonal=0.02,
                rotation_regularizer=0.0001,
                oracle_gap=1.0,
                condition_regularizer=0.01,
            )
        elif args.loss_profile == "ste_conservative":
            weights = JointLossWeights(
                token_rank=1.0,
                token_listwise=0.15,
                quest_miss=0.0,
                quest_slack=0.05,
                complementarity=0.05,
                quantization=0.02,
                bit_balance=0.01,
                decorrelation=0.01,
                hash_orthogonal=0.01,
                rotation_regularizer=0.0001,
                oracle_gap=1.0,
                condition_regularizer=0.01,
            )
        else:
            weights = JointLossWeights()
        first_payload = torch.load(paths[0], map_location="cpu", weights_only=True)
        query_heads = int(first_payload["query"].shape[-2])
        del first_payload
        config = JointTrainConfig(
            sparse_ratio=args.sparse_ratio,
            num_sink=args.num_sink,
            quest_train_ratio=args.quest_ratio,
            quest_head_ratios=head_ratios,
            page_size=args.page_size,
            margin=0.5,
            hash_temperature=1.0,
            projection_scale=0.1,
            teacher_temperature=1.0,
            page_teacher_temperature=1.0,
            use_rms=args.use_rms,
            learn_shared_rotation=args.transform_mode != "identity",
            transform_mode=args.transform_mode,
            hash_branch=args.hash_branch,
            rotation_granularity=args.rotation_granularity,
            nonorth_source_scale=args.nonorth_source_scale,
            condition_target=args.condition_target,
            straight_through_binary=args.loss_profile != "v1",
            weights=weights,
        )
        model = UnifiedQuestHashTrainer(
            kv_heads=initial.shape[0],
            query_heads=query_heads,
            head_dim=initial.shape[1],
            hash_bits=initial.shape[2],
            config=config,
            initial_hash_weight=initial,
            initial_rotation=initial_rotation,
        ).to(args.device)
        if args.freeze_hash:
            model.hash_weight.requires_grad_(False)
        parameter_groups = []
        if not args.freeze_hash:
            parameter_groups.append(
                {"params": [model.hash_weight], "lr": args.hash_lr}
            )
        if args.transform_mode != "identity" and not args.freeze_rotation:
            parameter_groups.append(
                {"params": [model.rotation_source], "lr": args.rotation_lr}
            )
        if not parameter_groups and args.epochs > 0:
            raise ValueError("training requires at least one unfrozen parameter group")
        optimizer = (
            torch.optim.AdamW(parameter_groups, weight_decay=1e-5)
            if parameter_groups
            else None
        )
        history = []
        for epoch in range(args.epochs):
            order = paths[:]
            random.shuffle(order)
            for step, path in enumerate(order):
                if head_ratios is None:
                    quest_ratio = ratio_curriculum[
                        (epoch * len(order) + step) % len(ratio_curriculum)
                    ]
                    model.config.quest_train_ratio = quest_ratio
                    logged_quest_ratio = quest_ratio
                else:
                    logged_quest_ratio = list(head_ratios)
                payload = torch.load(path, map_location="cpu", weights_only=True)
                query = payload["query"].unsqueeze(0).to(args.device).float()
                key = payload["key"].unsqueeze(0).to(args.device).float()
                optimizer.zero_grad(set_to_none=True)
                result = model(query, key)
                if not torch.isfinite(result["loss"]):
                    raise FloatingPointError(
                        f"non-finite loss layer={layer} epoch={epoch} sample={path}"
                    )
                result["loss"].backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"non-finite gradient layer={layer} epoch={epoch} sample={path}"
                    )
                previous_rotation = (
                    model.rotation_source.detach().clone()
                    if args.transform_mode == "paired_nonorth"
                    else None
                )
                optimizer.step()
                condition_backtracks = 0
                if previous_rotation is not None:
                    proposed_rotation = model.rotation_source.detach().clone()
                    with torch.no_grad():
                        # Per-KV-head transforms produce one condition number
                        # per head.  The safety gate is defined by the worst
                        # conditioned head, so reduce explicitly before the
                        # scalar comparison/backtracking decision.
                        condition = torch.linalg.cond(
                            model.shared_rotation().float()
                        ).amax()
                        while (
                            float(condition.detach().cpu()) > args.condition_target
                            and condition_backtracks < 12
                        ):
                            proposed_rotation.lerp_(previous_rotation, 0.5)
                            model.rotation_source.copy_(proposed_rotation)
                            condition = torch.linalg.cond(
                                model.shared_rotation().float()
                            ).amax()
                            condition_backtracks += 1
                        if float(condition.detach().cpu()) > args.condition_target:
                            model.rotation_source.copy_(previous_rotation)
                            condition_backtracks += 1
                metrics = {
                    "epoch": epoch,
                    "step": step,
                    "sample": os.path.basename(path),
                    "seq_len": int(key.shape[1]),
                    "quest_ratio": logged_quest_ratio,
                    "grad_norm": float(grad_norm.detach().cpu()),
                    "condition_backtracks": condition_backtracks,
                    **scalar_metrics(result),
                }
                history.append(metrics)
                print(json.dumps({"layer": layer, **metrics}), flush=True)
                del payload, query, key, result
                torch.cuda.empty_cache()
        weight_path = os.path.join(
            args.save_dir, f"hash_weight_layer_{layer:02d}.pt"
        )
        rotation_path = os.path.join(
            args.save_dir, f"shared_rotation_layer_{layer:02d}.pt"
        )
        source_path = os.path.join(
            args.save_dir, f"transform_source_layer_{layer:02d}.pt"
        )
        torch.save(model.hash_weight.detach().bfloat16().cpu(), weight_path)
        torch.save(model.shared_rotation().detach().float().cpu(), rotation_path)
        torch.save(model.rotation_source.detach().float().cpu(), source_path)
        all_history[f"layer{layer:02d}"] = history
        del model, optimizer
        torch.cuda.empty_cache()

    with open(os.path.join(args.save_dir, "train_history.json"), "w") as handle:
        json.dump(all_history, handle, indent=2)
    with open(os.path.join(args.save_dir, "train_config.json"), "w") as handle:
        json.dump(vars(args), handle, indent=2)


if __name__ == "__main__":
    main()
