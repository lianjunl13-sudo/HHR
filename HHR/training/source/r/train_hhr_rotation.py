#!/usr/bin/env python3
"""Original HHR joint loss with one Cayley rotation per KV head."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path(
    os.environ.get("HHR_TRAINING_INPUT_ROOT", PACKAGE_ROOT / "training" / "inputs")
)
sys.path.insert(0, str(SOURCE_ROOT))
from unified_quest_hash import build_page_minmax, project_hash_codes

MANIFEST = INPUT_ROOT / "hhr_rotation_training_manifest.json"
WEIGHTS = INPUT_ROOT / "hhr_initial_hash_weights"
START_R = INPUT_ROOT / "hhr_initial_rotation" / "rotation.pt"
OUT_ROOT = Path(os.environ.get("HHR_R_OUTPUT", PACKAGE_ROOT / "training_outputs" / "trained_r"))
FROZEN_BUDGET = INPUT_ROOT / "hhr_rotation_head_budget.json"
LAYER, PAGE, CANDIDATE_RATIO, FINAL_RATIO = 25, 8, "dynamic", 0.015
HOLDOUT, VALIDATION = {0, 2, 7, 10, 16, 20}, {1, 8, 14}
EPOCHS, MARGIN = 30, 0.5
VARIANTS = ((5e-5, 5e-5), (1e-4, 5e-5), (5e-5, 1e-4))


def mean(values):
    return sum(values) / len(values)


def cayley(source):
    skew = source - source.transpose(-1, -2)
    eye = torch.eye(source.shape[-1], device=source.device, dtype=source.dtype).expand_as(source)
    return torch.linalg.solve(eye + skew, eye - skew)


def inverse_cayley(rotation):
    eye = torch.eye(rotation.shape[-1], device=rotation.device, dtype=rotation.dtype).expand_as(rotation)
    return 0.5 * ((eye - rotation) @ torch.linalg.solve(eye + rotation, eye))


def rotate(query, key, rotation):
    q = query.reshape(1, 8, 4, 128)
    q = torch.einsum("bkgd,kde->bkge", q, rotation).reshape_as(query)
    k = torch.einsum("bskd,kde->bske", key, rotation)
    return q, k


def candidate_ratios():
    if CANDIDATE_RATIO != "dynamic":
        return [float(CANDIDATE_RATIO)] * 8
    budget = json.loads(FROZEN_BUDGET.read_text())
    assert budget["page_size"] == PAGE
    assert abs(budget["sparse_ratio"] - FINAL_RATIO) < 1e-12
    return [float(x) for x in budget["layers"][str(LAYER)]]


def page_selection(upper, length):
    """Select a potentially different number of pages for each KV head."""
    counts = torch.tensor(
        [max(1, math.ceil(length * ratio / PAGE)) for ratio in candidate_ratios()],
        device=upper.device,
        dtype=torch.long,
    )
    order = upper.argsort(dim=-1, descending=True)
    rank = torch.arange(order.shape[-1], device=upper.device).view(1, 1, -1)
    keep = rank < counts.view(1, 8, 1)
    selected_page = torch.zeros_like(upper, dtype=torch.bool)
    selected_page.scatter_(-1, order, keep)
    cutoff = upper.gather(-1, order).gather(
        -1, (counts - 1).view(1, 8, 1)
    )
    return selected_page, cutoff, counts


def retrieval(query, key, rotation, weight):
    """Deployed binary Hash retrieval; original Q/K only define evaluation labels."""
    length = key.shape[1]
    q_search, k_search = rotate(query, key, rotation)
    mins, maxs = build_page_minmax(k_search, PAGE)
    qg = q_search.reshape(1, 8, 4, 128)
    mins, maxs = mins.permute(0, 2, 1, 3), maxs.permute(0, 2, 1, 3)
    upper = torch.einsum("bkgd,bkpd->bkgp", qg.clamp_min(0), maxs)
    upper += torch.einsum("bkgd,bkpd->bkgp", qg.clamp_max(0), mins)
    upper = upper.amax(dim=2)
    selected_page, _, counts = page_selection(upper, length)
    pages = int(counts.max())
    page_idx = upper.topk(pages, dim=-1).indices
    offsets = torch.arange(PAGE, device=key.device)
    candidates = (page_idx.unsqueeze(-1) * PAGE + offsets).flatten(-2)
    candidate_mask = candidates < length
    candidate_mask &= (
        torch.arange(pages, device=key.device).repeat_interleave(PAGE)
        .view(1, 1, -1) < counts.view(1, 8, 1)
    )
    candidates = candidates.clamp_max(length - 1)
    q_codes, k_codes = project_hash_codes(q_search, k_search, weight, use_rms=False)
    selected_codes = k_codes.permute(0, 2, 1, 3).gather(2, candidates.unsqueeze(-1).expand(-1, -1, -1, weight.shape[-1]))
    q_codes = q_codes.reshape(1, 8, 4, weight.shape[-1])
    distance = torch.logical_xor(selected_codes.unsqueeze(2), q_codes.unsqueeze(3)).sum(dim=-1).sum(dim=2).float()
    distance.masked_fill_(~candidate_mask, torch.inf)
    final_k = max(1, math.floor(length * FINAL_RATIO))
    chosen = distance.topk(final_k, dim=-1, largest=False).indices
    final_tokens = candidates.gather(-1, chosen)
    return upper, candidates, candidate_mask, final_tokens


def original_loss(query, key, rotation, weight):
    """The original joint objective, changing only shared R to per-KV-head R_g."""
    length, pages = key.shape[1], math.ceil(key.shape[1] / PAGE)
    qg = query.reshape(1, 8, 4, 128)
    exact = torch.einsum("bkgd,bskd->bkgs", qg, key)
    teacher = (exact / math.sqrt(128)).softmax(dim=-1).mean(dim=2)
    final_k = max(1, math.floor(length * FINAL_RATIO))
    positive_idx = teacher.topk(final_k, dim=-1).indices
    positive_mask = torch.zeros_like(teacher, dtype=torch.bool)
    positive_mask.scatter_(-1, positive_idx, True)

    q_rot, k_rot = rotate(query, key, rotation)
    mins, maxs = build_page_minmax(k_rot, PAGE)
    qrg = q_rot.reshape(1, 8, 4, 128)
    mins, maxs = mins.permute(0, 2, 1, 3), maxs.permute(0, 2, 1, 3)
    upper_h = torch.einsum("bkgd,bkpd->bkgp", qrg.clamp_min(0), maxs)
    upper_h += torch.einsum("bkgd,bkpd->bkgp", qrg.clamp_max(0), mins)
    upper = upper_h.amax(dim=2)
    selected_page, cutoff, _ = page_selection(upper, length)
    token_page = torch.arange(length, device=key.device) // PAGE
    selected_token = selected_page[..., token_page]
    positive_page = torch.zeros_like(selected_page)
    positive_page.scatter_(-1, positive_idx // PAGE, True)
    missed_page = positive_page & ~selected_page
    false_page = selected_page & ~positive_page

    q_logits = torch.einsum("bkgd,kdr->bkgr", qrg, weight)
    k_logits = torch.einsum("bskd,kdr->bskr", k_rot, weight)
    q_soft, k_soft = torch.tanh(0.1 * q_logits / 0.5), torch.tanh(0.1 * k_logits / 0.5)
    q_hard, k_hard = torch.where(q_logits >= 0, 1.0, -1.0), torch.where(k_logits >= 0, 1.0, -1.0)
    q_code, k_code = q_soft + (q_hard - q_soft).detach(), k_soft + (k_hard - k_soft).detach()
    per_head = ((q_code.reshape(1, 32, -1).unsqueeze(2) - k_code.repeat_interleave(4, dim=2).permute(0, 2, 1, 3)) ** 2).mean(dim=-1)
    distance = per_head.reshape(1, 8, 4, length).mean(dim=2)
    negative = selected_token & ~positive_mask
    priority = (-distance.detach()).masked_fill(~negative, -torch.inf)
    neg_idx = priority.topk(min(length - final_k, max(1, final_k * 4)), dim=-1).indices
    neg_valid = negative.gather(-1, neg_idx)
    pos_valid = selected_token.gather(-1, positive_idx)
    pos_dist, neg_dist = distance.gather(-1, positive_idx), distance.gather(-1, neg_idx)
    pair = F.softplus(MARGIN + pos_dist.unsqueeze(-1) - neg_dist.unsqueeze(-2))
    pair_valid = pos_valid.unsqueeze(-1) & neg_valid.unsqueeze(-2)
    pos_weight = teacher.gather(-1, positive_idx)
    token_rank = (pair * pos_weight.unsqueeze(-1) * pair_valid).sum() / (pos_weight.unsqueeze(-1) * pair_valid).sum().clamp_min(1e-12)

    score = (-distance).masked_fill(~selected_token, -torch.inf)
    threshold = score.topk(final_k, dim=-1).values[..., -1:]
    soft_mask = torch.sigmoid(((-distance) - threshold) / 0.1) * selected_token.float()
    survived = positive_mask & selected_token
    survived_count = survived.sum(dim=-1)
    valid = survived_count > 0
    recall = (soft_mask * survived.float()).sum(dim=-1) / survived_count.clamp_min(1)
    paper2 = -torch.log(recall[valid].mean().clamp_min(1e-8)) if bool(valid.any()) else distance.sum() * 0
    paper2 += 0.1 * (soft_mask.sum(dim=-1) / final_k - 1).square().mean()
    student_logp = F.log_softmax(-distance.masked_fill(~selected_token, torch.inf) / 0.5, dim=-1)
    teacher_p = teacher.masked_fill(~selected_token, 0)
    teacher_p = teacher_p / teacher_p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    listwise = (teacher_p * (teacher_p.clamp_min(1e-12).log() - student_logp)).masked_fill(~selected_token, 0).sum(dim=-1).mean()

    exact_page = F.pad(exact.reshape(1, 32, length), (0, pages * PAGE - length), value=-torch.inf).reshape(1, 8, 4, pages, PAGE).amax(dim=-1).amax(dim=2)
    slack = (upper - exact_page).clamp_min(0)
    norm_slack = (slack / exact_page.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)).clamp_max(8)
    quest_miss = (F.relu(cutoff + MARGIN - upper) * missed_page).sum() / missed_page.sum().clamp_min(1)
    fp_weight = false_page.float() * (1 + norm_slack.detach())
    quest_slack = (torch.log1p(norm_slack) * fp_weight).sum() / fp_weight.sum().clamp_min(1)
    hash_page = torch.logsumexp(F.pad(-distance, (0, pages * PAGE - length), value=-torch.inf).reshape(1, 8, pages, PAGE) / 0.5, dim=-1)
    z = lambda x: (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
    complementarity = F.kl_div(F.log_softmax(z(upper) + z(hash_page), dim=-1), F.softmax(z(exact_page), dim=-1), reduction="batchmean")

    code_all = torch.cat((q_soft.reshape(-1, weight.shape[-1]), k_soft.reshape(-1, weight.shape[-1])), dim=0)
    quantization = (code_all.abs() - 1).square().mean()
    balance = code_all.mean(dim=0).square().mean()
    centered = code_all - code_all.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1) @ centered / max(1, centered.shape[0])
    covariance.fill_diagonal_(0)
    decorrelation = covariance.square().mean()
    gram = torch.einsum("kdr,kds->krs", weight, weight)
    scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-6)
    hash_orthogonal = (gram / scale.unsqueeze(-1) - torch.eye(weight.shape[-1], device=weight.device)).square().mean()
    rotation_regularizer = (rotation - torch.eye(128, device=rotation.device)).square().mean()
    total = paper2 + token_rank + .35 * listwise + quest_miss + .10 * quest_slack + .20 * complementarity + .02 * quantization + .01 * balance + .01 * decorrelation + .01 * hash_orthogonal + .001 * rotation_regularizer
    return total, {"paper2": paper2, "rank": token_rank, "listwise": listwise, "miss": quest_miss, "slack": quest_slack, "comp": complementarity}


@torch.no_grad()
def evaluate(paths, rotation, weight, device):
    candidate, final = [], []
    for path in paths:
        data = torch.load(path, map_location=device, weights_only=True)
        query, key = data["query"].float().unsqueeze(0), data["key"].float().unsqueeze(0)
        _, candidates, valid, tokens = retrieval(query, key, rotation, weight)
        qg = query.reshape(1, 8, 4, 128)
        exact = torch.einsum("bkgd,bskd->bkgs", qg, key)
        target_score = (exact / math.sqrt(128)).softmax(dim=-1).mean(dim=2)
        target = torch.zeros_like(target_score, dtype=torch.bool)
        target.scatter_(-1, target_score.topk(max(1, math.floor(key.shape[1] * FINAL_RATIO)), dim=-1).indices, True)
        candidate_mask = torch.zeros_like(target)
        candidate_mask.scatter_(-1, candidates, valid)
        final_mask = torch.zeros_like(target)
        final_mask.scatter_(-1, tokens, True)
        candidate.append(float((candidate_mask & target).sum().float() / target.sum().clamp_min(1)))
        final.append(float((final_mask & target).sum().float() / target.sum().clamp_min(1)))
    return candidate, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-ratio", choices=("0.10", "0.20", "0.30", "dynamic"), default="dynamic")
    parser.add_argument("--mode", choices=("joint", "w_only", "r_only"), default="r_only")
    parser.add_argument("--layer", type=int, choices=range(2, 32), default=25)
    args = parser.parse_args()
    global CANDIDATE_RATIO, LAYER
    LAYER = args.layer
    CANDIDATE_RATIO = args.candidate_ratio if args.candidate_ratio == "dynamic" else float(args.candidate_ratio)
    ratio_tag = "dynamic" if CANDIDATE_RATIO == "dynamic" else f"{int(round(CANDIDATE_RATIO * 100)):02d}"
    out = OUT_ROOT / f"layer_{LAYER:02d}"
    torch.manual_seed(20260912)
    device = torch.device("cuda")
    if (out / "report.json").exists():
        raise RuntimeError(f"refuse overwrite completed result: {out}")
    out.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(MANIFEST.read_text())["mapping"][str(LAYER)]
    mapping = {key: str(PACKAGE_ROOT / value) for key, value in mapping.items()}
    train = [mapping[str(i)] for i in range(21) if i not in HOLDOUT | VALIDATION]
    valid, heldout = [mapping[str(i)] for i in sorted(VALIDATION)], [mapping[str(i)] for i in sorted(HOLDOUT)]
    identity_rotation = torch.eye(128, device=device).expand(8, 128, 128).clone()
    pretrained_rotation = torch.load(START_R, map_location=device, weights_only=True).float()
    start_rotation = pretrained_rotation if args.mode in ("joint", "r_only") else identity_rotation
    start_weight = torch.load(WEIGHTS / f"hash_weight_layer_{LAYER:02d}.pt", map_location=device, weights_only=True).float()
    initial_validation_candidate, initial_validation_final = evaluate(valid, start_rotation, start_weight, device)
    initial_name = "offline_pretrained_R+frozen_original_W" if args.mode == "r_only" else ("offline_pretrained_R+original_hash" if args.mode == "joint" else "identity_R+original_hash")
    best = {"validation_candidate_recall": mean(initial_validation_candidate), "validation_final_recall": mean(initial_validation_final), "epoch": 0, "variant": initial_name, "rotation": start_rotation.clone(), "weight": start_weight.clone()}
    trials = []
    variants = VARIANTS if args.mode == "joint" else (((5e-5, 0.0), (1e-4, 0.0), (3e-4, 0.0)) if args.mode == "r_only" else ((0.0, 5e-5), (0.0, 1e-4)))
    for lr_r, lr_w in variants:
        source = torch.nn.Parameter(inverse_cayley(start_rotation), requires_grad=args.mode in ("joint", "r_only"))
        weight = torch.nn.Parameter(start_weight.clone(), requires_grad=args.mode in ("joint", "w_only"))
        groups = []
        if weight.requires_grad:
            groups.append({"params": [weight], "lr": lr_w})
        if source.requires_grad:
            groups.insert(0, {"params": [source], "lr": lr_r})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        history = []
        for epoch in range(1, EPOCHS + 1):
            logs = []
            for index in torch.randperm(len(train)).tolist():
                data = torch.load(train[index], map_location=device, weights_only=True)
                query, key = data["query"].float().unsqueeze(0), data["key"].float().unsqueeze(0)
                optimizer.zero_grad(set_to_none=True)
                loss, parts = original_loss(query, key, cayley(source), weight)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in (source, weight) if p.requires_grad], 1.0)
                optimizer.step()
                logs.append(float(loss.detach()))
            rotation = cayley(source.detach())
            validation_candidate, validation_final = evaluate(valid, rotation, weight.detach(), device)
            validation_candidate, validation_final = mean(validation_candidate), mean(validation_final)
            history.append({"epoch": epoch, "loss": mean(logs), "validation_candidate_recall": validation_candidate, "validation_final_recall": validation_final})
            if validation_final > best["validation_final_recall"]:
                best = {"validation_candidate_recall": validation_candidate, "validation_final_recall": validation_final, "epoch": epoch, "variant": {"rotation_lr": lr_r, "hash_lr": lr_w}, "rotation": rotation.clone(), "weight": weight.detach().clone()}
        trials.append({"rotation_lr": lr_r, "hash_lr": lr_w, "history": history})
    rotation, weight = best.pop("rotation"), best.pop("weight")
    torch.save(rotation.cpu(), out / f"shared_rotation_layer_{LAYER:02d}.pt")
    if args.mode != "r_only":
        torch.save(weight.cpu(), out / f"hash_weight_layer_{LAYER:02d}.pt")
    candidate, final = evaluate(heldout, rotation, weight, device)
    method_name = {"joint": "offline-pretrained per-KV-head Cayley R + joint W", "w_only": "matched W-only with R=I", "r_only": "offline-pretrained per-KV-head Cayley R with frozen original W"}[args.mode]
    report = {"method": method_name, "selection_rule": "maximize validation final Hash Top-1.5% recall; heldout never used for selection", "deployment": {"layer": LAYER, "rotation_count": 8, "page_size": PAGE, "quest_candidate_ratio": CANDIDATE_RATIO, "quest_candidate_ratio_by_kv_head": candidate_ratios(), "hash_final_ratio": FINAL_RATIO, "rotation_initialization": "offline_pretrained_then_transferred" if args.mode in ("joint", "r_only") else "identity_fixed", "rotation_initialization_path": str(START_R) if args.mode in ("joint", "r_only") else None, "hash_weight_policy": "frozen_original_tensor_exact" if args.mode == "r_only" else "trained", "mode": args.mode}, "selection": best, "heldout": {"candidate_recall": candidate, "final_hash_recall": final, "mean_candidate_recall": mean(candidate), "mean_final_hash_recall": mean(final)}, "orthogonality_max_abs_error": float((rotation @ rotation.transpose(-1, -2) - torch.eye(128, device=device)).abs().max()), "trials": trials}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"selection": best, "heldout": report["heldout"], "orth_error": report["orthogonality_max_abs_error"]}, indent=2))


if __name__ == "__main__":
    main()
