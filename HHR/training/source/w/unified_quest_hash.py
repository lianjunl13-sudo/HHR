"""Reference implementation of HHR: QUEST -> binary Hash -> exact attention.

This module deliberately contains no project-specific CUDA dependency.  It is the
correctness oracle for the fused implementation.  Hash codes are used only to
retrieve token indices; the returned indices must be applied to the untouched
K/V cache before exact attention is evaluated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class QHGuardConfig:
    sparse_ratio: float = 0.015
    page_size: int = 8
    # Formal mainline: one fixed ratio per KV head (and per layer through the
    # caller's layer-specific config). Candidate identities still change with Q.
    budget_mode: str = "fixed_headwise"
    fixed_head_ratios: Optional[Tuple[float, ...]] = None
    probe_ratio: float = 0.08
    base_ratio: float = 0.16
    max_ratio: float = 0.30
    num_sink: int = 0
    num_recent: int = 0
    hash_margin_target: float = 0.025
    hash_margin_temperature: float = 0.0125
    tie_target: float = 0.10
    bound_gap_temperature: float = 0.20
    uncertainty_hash_weight: float = 0.55
    uncertainty_tie_weight: float = 0.20
    uncertainty_bound_weight: float = 0.25
    exact_guard: bool = False
    exact_guard_margin: float = 0.0
    hash_rms: bool = False
    quest_tie_break_strength: float = 0.0
    quest_tie_break_mode: str = "quest"
    final_k_rounding: str = "floor"

    def validate(self) -> None:
        if self.budget_mode not in {"fixed_headwise", "query_dynamic_ablation"}:
            raise ValueError("invalid budget_mode")
        if not 0 < self.sparse_ratio <= self.probe_ratio:
            raise ValueError("require 0 < sparse_ratio <= probe_ratio")
        if not self.probe_ratio <= self.base_ratio <= self.max_ratio <= 1:
            raise ValueError("require probe_ratio <= base_ratio <= max_ratio <= 1")
        if self.page_size <= 0:
            raise ValueError("page_size must be positive")
        if self.final_k_rounding not in {"floor", "ceil"}:
            raise ValueError("final_k_rounding must be floor or ceil")
        if not 0.0 <= self.quest_tie_break_strength < 1.0:
            raise ValueError("quest_tie_break_strength must be in [0, 1)")
        if self.quest_tie_break_mode not in {"quest", "reverse", "token_hash"}:
            raise ValueError("invalid quest_tie_break_mode")
        if self.fixed_head_ratios is not None:
            if any(not self.sparse_ratio <= ratio <= 1.0 for ratio in self.fixed_head_ratios):
                raise ValueError("fixed_head_ratios must cover sparse_ratio and be <= 1")


@dataclass
class QHGuardOutput:
    token_indices: torch.Tensor
    page_indices: torch.Tensor
    page_mask: torch.Tensor
    candidate_mask: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]


def rms_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS-normalize the Hash branch only; never overwrite original K/V."""
    return x * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps).to(x.dtype)


def robust_amplitude_suppress(x: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Smooth clipping for Hash inputs. ``clip`` is calibrated on training data."""
    clip = clip.to(device=x.device, dtype=x.dtype).clamp_min(torch.finfo(x.dtype).eps)
    return clip * torch.tanh(x / clip)


def apply_shared_rotation(
    query: torch.Tensor,
    key: torch.Tensor,
    rotation: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply the same orthogonal map to Q/K, preserving their exact dot product."""
    if rotation is None:
        return query, key
    return query @ rotation, key @ rotation


def quest_upper_bounds(
    page_mins: torch.Tensor,
    page_maxs: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Exact interval upper bounds, aggregated conservatively over GQA heads.

    Args:
        page_mins/page_maxs: [batch, pages, kv_heads, dim]
        query: [batch, query_heads, dim] (or [batch, 1, query_heads, dim])
    Returns:
        [batch, kv_heads, pages]
    """
    if query.ndim == 4:
        if query.shape[1] != 1:
            raise ValueError("4-D query must have singleton sequence dimension")
        query = query[:, 0]
    if query.ndim != 3 or page_mins.ndim != 4:
        raise ValueError("invalid query or page tensor rank")
    batch, _, kv_heads, dim = page_mins.shape
    if page_maxs.shape != page_mins.shape or query.shape[0] != batch or query.shape[-1] != dim:
        raise ValueError("incompatible query/page shapes")
    query_heads = query.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    group = query_heads // kv_heads
    q = query.float().reshape(batch, kv_heads, group, dim)
    mins = page_mins.float().permute(0, 2, 1, 3)
    maxs = page_maxs.float().permute(0, 2, 1, 3)
    upper = torch.einsum("bkgd,bkpd->bkgp", q.clamp_min(0), maxs)
    upper = upper + torch.einsum("bkgd,bkpd->bkgp", q.clamp_max(0), mins)
    return upper.amax(dim=2)


def build_page_minmax(key: torch.Tensor, page_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build page boxes from [B,S,KVH,D] keys, padding the last page safely."""
    if key.ndim != 4:
        raise ValueError("key must be [batch, sequence, kv_heads, dim]")
    batch, seq_len, kv_heads, dim = key.shape
    pages = math.ceil(seq_len / page_size)
    pad = pages * page_size - seq_len
    if pad:
        k_min = F.pad(key, (0, 0, 0, 0, 0, pad), value=torch.inf)
        k_max = F.pad(key, (0, 0, 0, 0, 0, pad), value=-torch.inf)
    else:
        k_min = k_max = key
    mins = k_min.reshape(batch, pages, page_size, kv_heads, dim).amin(dim=2)
    maxs = k_max.reshape(batch, pages, page_size, kv_heads, dim).amax(dim=2)
    return mins, maxs


def project_hash_codes(
    query: torch.Tensor,
    key: torch.Tensor,
    weight: torch.Tensor,
    amplitude_clip: Optional[torch.Tensor] = None,
    use_rms: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return Boolean Q/K codes with per-KV-head projection weights.

    query: [B,H,D], key: [B,S,KVH,D], weight: [KVH,D,bits]
    """
    batch, query_heads, dim = query.shape
    _, _, kv_heads, key_dim = key.shape
    if dim != key_dim or weight.shape[:2] != (kv_heads, dim):
        raise ValueError("incompatible Q/K/projection shapes")
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    group = query_heads // kv_heads
    if use_rms:
        q = rms_normalize(query).reshape(batch, kv_heads, group, dim)
        k = rms_normalize(key)
    else:
        q = query.reshape(batch, kv_heads, group, dim)
        k = key
    if amplitude_clip is not None:
        q = robust_amplitude_suppress(q, amplitude_clip)
        k = robust_amplitude_suppress(k, amplitude_clip)
    q_logits = torch.einsum("bkgd,kdr->bkgr", q, weight.float())
    k_logits = torch.einsum("bskd,kdr->bskr", k, weight.float())
    q_codes = q_logits.reshape(batch, query_heads, -1) >= 0
    return q_codes, k_logits >= 0


def _forced_page_mask(seq_len: int, cfg: QHGuardConfig, device: torch.device) -> torch.Tensor:
    pages = math.ceil(seq_len / cfg.page_size)
    mask = torch.zeros(pages, dtype=torch.bool, device=device)
    sink_pages = math.ceil(min(cfg.num_sink, seq_len) / cfg.page_size)
    mask[:sink_pages] = True
    if cfg.num_recent:
        recent_start = max(0, (seq_len - cfg.num_recent) // cfg.page_size)
        mask[recent_start:] = True
    return mask


def _page_count(seq_len: int, ratio: float, page_size: int, forced: int) -> int:
    active = math.ceil(seq_len / page_size)
    target = math.ceil(max(1, seq_len * ratio) / page_size)
    return min(active, max(target, forced))


def _rank_pages(
    upper: torch.Tensor,
    forced_mask: torch.Tensor,
) -> torch.Tensor:
    forced = torch.nonzero(forced_mask, as_tuple=False).flatten()
    score = upper.clone()
    if forced.numel():
        score[..., forced] = -torch.inf
    ranked = score.argsort(dim=-1, descending=True)
    if not forced.numel():
        return ranked
    forced = forced.view(1, 1, -1).expand(*upper.shape[:2], -1)
    return torch.cat((forced, ranked[..., : upper.shape[-1] - forced.shape[-1]]), dim=-1)


def _pages_to_candidates(
    pages: torch.Tensor,
    count: torch.Tensor,
    seq_len: int,
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize a padded page/token list for variable per-head budgets."""
    max_pages = int(count.max().item())
    chosen = pages[..., :max_pages]
    page_mask = torch.arange(max_pages, device=pages.device).view(1, 1, -1) < count.unsqueeze(-1)
    offsets = torch.arange(page_size, device=pages.device)
    token = (chosen.unsqueeze(-1) * page_size + offsets).flatten(-2)
    valid = page_mask.unsqueeze(-1).expand(*page_mask.shape, page_size).flatten(-2)
    valid = valid & (token < seq_len)
    return token.clamp_max(seq_len - 1), valid


def _hash_distance(
    query_codes: torch.Tensor,
    key_codes: torch.Tensor,
    candidate_idx: torch.Tensor,
    candidate_mask: torch.Tensor,
    num_sink: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match deployed GQA retrieval: one summed distance and Top-K per KV head."""
    batch, _, kv_heads, bits = key_codes.shape
    query_heads = query_codes.shape[1]
    group = query_heads // kv_heads
    key_kv = key_codes.permute(0, 2, 1, 3)
    selected = key_kv.gather(
        2, candidate_idx.unsqueeze(-1).expand(-1, -1, -1, bits)
    )
    query_kv = query_codes.reshape(batch, kv_heads, group, bits)
    distance = torch.logical_xor(
        selected.unsqueeze(2), query_kv.unsqueeze(3)
    ).sum(dim=-1).sum(dim=2).float()
    # Deployed CUDA gives forced sink candidates zero distance so that they
    # survive the final Top-K.  Use token ids here; QUEST orders forced sink
    # pages first, making this equivalent to the CUDA candidate-position rule.
    if num_sink:
        sink_mask = (candidate_idx < num_sink) & candidate_mask
        distance = torch.where(sink_mask, torch.zeros_like(distance), distance)
    distance.masked_fill_(~candidate_mask, torch.inf)
    return distance, candidate_idx, candidate_mask


def _confidence(
    distance: torch.Tensor,
    final_k: int,
    bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_dist = distance.sort(dim=-1).values
    kth = sorted_dist[..., final_k - 1]
    guard_pos = min(sorted_dist.shape[-1] - 1, final_k + max(2, final_k // 4) - 1)
    guard = sorted_dist[..., guard_pos]
    margin = (guard - kth).clamp_min(0) / bits
    valid = torch.isfinite(distance)
    tie = ((distance == kth.unsqueeze(-1)) & valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
    return margin, tie, kth


def _final_k(seq_len: int, cfg: QHGuardConfig) -> int:
    raw = seq_len * cfg.sparse_ratio
    result = math.floor(raw) if cfg.final_k_rounding == "floor" else math.ceil(raw)
    return max(1, min(seq_len, result))


def quest_tie_break_distance(
    distance: torch.Tensor,
    strength: float,
    mode: str = "quest",
    candidate_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Order equal Hamming distances by QUEST candidate rank, never by >=1 bit."""
    if not strength:
        return distance
    if not 0.0 < strength < 1.0:
        raise ValueError("tie-break strength must be in (0, 1)")
    if mode == "token_hash":
        if candidate_idx is None:
            raise ValueError("token_hash tie-break requires candidate_idx")
        token = candidate_idx.long()
        offset = ((token * 1103515245 + 12345) & 0x7FFFFFFF).float()
        offset = offset / float(0x80000000)
    else:
        position = torch.arange(
            distance.shape[-1], device=distance.device, dtype=torch.float32
        )
        offset = position / max(1, distance.shape[-1])
        offset = offset.view(1, 1, -1)
        if mode == "reverse":
            offset = -offset
        elif mode != "quest":
            raise ValueError(f"invalid tie-break mode: {mode}")
    return distance.float() + strength * offset


@torch.no_grad()
def qh_guard_retrieve(
    query: torch.Tensor,
    key: torch.Tensor,
    hash_weight: torch.Tensor,
    cfg: QHGuardConfig = QHGuardConfig(),
    shared_rotation: Optional[torch.Tensor] = None,
    hash_rotation: Optional[torch.Tensor] = None,
    amplitude_clip: Optional[torch.Tensor] = None,
) -> QHGuardOutput:
    """Strict QUEST+Hash retrieval with an exactly fixed final Token Top-K.

    The formal path uses fixed per-KV-head QUEST budgets. Query-level dynamic
    budgeting is retained only behind ``query_dynamic_ablation``.
    """
    cfg.validate()
    if query.ndim == 4:
        query = query[:, 0]
    batch, seq_len, kv_heads, _ = key.shape
    query_heads = query.shape[1]
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if cfg.fixed_head_ratios is not None and len(cfg.fixed_head_ratios) != kv_heads:
        raise ValueError("fixed_head_ratios must contain one ratio per KV head")
    q_search, k_search = apply_shared_rotation(query, key, shared_rotation)
    page_mins, page_maxs = build_page_minmax(k_search, cfg.page_size)
    upper = quest_upper_bounds(page_mins, page_maxs, q_search)
    q_hash, k_hash = apply_shared_rotation(q_search, k_search, hash_rotation)
    q_codes, k_codes = project_hash_codes(
        q_hash,
        k_hash,
        hash_weight,
        amplitude_clip=amplitude_clip,
        use_rms=cfg.hash_rms,
    )

    forced_mask = _forced_page_mask(seq_len, cfg, query.device)
    ranked_pages = _rank_pages(upper, forced_mask)
    forced_count = int(forced_mask.sum().item())
    final_k = _final_k(seq_len, cfg)

    if cfg.budget_mode == "fixed_headwise":
        ratios = (
            cfg.fixed_head_ratios
            if cfg.fixed_head_ratios is not None
            else tuple(cfg.base_ratio for _ in range(kv_heads))
        )
        per_head_count = [
            _page_count(seq_len, ratio, cfg.page_size, forced_count)
            for ratio in ratios
        ]
        dynamic_count = torch.tensor(
            per_head_count, dtype=torch.long, device=query.device
        ).view(1, kv_heads).expand(batch, -1)
        margin = torch.zeros_like(dynamic_count, dtype=torch.float32)
        tie = torch.zeros_like(margin)
        boundary_gap = torch.zeros_like(margin)
        uncertainty = torch.zeros_like(margin)
        kth_h = torch.full_like(margin, torch.nan)
    else:
        # Query-dynamic budget is a named ablation, not the formal method.
        dynamic_count, margin, tie, boundary_gap, uncertainty, kth_h = (
            _query_dynamic_page_count(
                query=query,
                key=key,
                upper=upper,
                ranked_pages=ranked_pages,
                forced_count=forced_count,
                q_codes=q_codes,
                k_codes=k_codes,
                cfg=cfg,
                final_k=final_k,
            )
        )

    candidate_idx, candidate_mask = _pages_to_candidates(
        ranked_pages, dynamic_count, seq_len, cfg.page_size
    )
    final_dist, candidate_idx_kv, candidate_mask_kv = _hash_distance(
        q_codes, k_codes, candidate_idx, candidate_mask, cfg.num_sink
    )
    if bool((candidate_mask_kv.sum(dim=-1) < final_k).any()):
        raise RuntimeError("QUEST candidate set is smaller than final Top-K")
    ranking_distance = quest_tie_break_distance(
        final_dist,
        cfg.quest_tie_break_strength,
        cfg.quest_tie_break_mode,
        candidate_idx_kv,
    )
    local_topk = ranking_distance.topk(final_k, dim=-1, largest=False).indices
    token_indices_kv = candidate_idx_kv.gather(-1, local_topk)
    group = query_heads // kv_heads
    token_indices = token_indices_kv.repeat_interleave(group, dim=1)
    candidate_mask_h = candidate_mask_kv.repeat_interleave(group, dim=1)
    page_indices = ranked_pages[..., : int(dynamic_count.max().item())]
    page_mask = (
        torch.arange(page_indices.shape[-1], device=query.device).view(1, 1, -1)
        < dynamic_count.unsqueeze(-1)
    )
    diagnostics = {
        "quest_upper_bounds": upper,
        "probe_hash_margin": margin,
        "probe_tie_fraction": tie,
        "bound_boundary_gap": boundary_gap,
        "uncertainty": uncertainty,
        "page_count": dynamic_count,
        "candidate_ratio": dynamic_count.float() * cfg.page_size / seq_len,
        "final_k": torch.tensor(final_k, device=query.device),
        "probe_kth_hamming": kth_h,
        "budget_mode_fixed": torch.tensor(
            cfg.budget_mode == "fixed_headwise", device=query.device
        ),
    }
    return QHGuardOutput(
        token_indices=token_indices,
        page_indices=page_indices,
        page_mask=page_mask,
        candidate_mask=candidate_mask_h,
        diagnostics=diagnostics,
    )


def _query_dynamic_page_count(
    query: torch.Tensor,
    key: torch.Tensor,
    upper: torch.Tensor,
    ranked_pages: torch.Tensor,
    forced_count: int,
    q_codes: torch.Tensor,
    k_codes: torch.Tensor,
    cfg: QHGuardConfig,
    final_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Legacy confidence-based budget logic, isolated as an explicit ablation."""
    batch, seq_len, kv_heads, _ = key.shape
    query_heads = query.shape[1]
    probe_count_scalar = _page_count(seq_len, cfg.probe_ratio, cfg.page_size, forced_count)
    probe_count = torch.full(
        upper.shape[:2], probe_count_scalar, dtype=torch.long, device=query.device
    )
    probe_idx, probe_mask = _pages_to_candidates(
        ranked_pages, probe_count, seq_len, cfg.page_size
    )
    probe_dist, _, _ = _hash_distance(
        q_codes, k_codes, probe_idx, probe_mask, cfg.num_sink
    )
    if probe_dist.shape[-1] < final_k:
        raise RuntimeError("probe candidate set is smaller than final Top-K")
    margin, tie, kth_h = _confidence(
        probe_dist, final_k, hash_weight.shape[-1] * (query_heads // kv_heads)
    )

    group = query_heads // kv_heads
    hash_uncertainty = torch.sigmoid(
        (cfg.hash_margin_target - margin) / cfg.hash_margin_temperature
    )
    tie_uncertainty = (tie / cfg.tie_target).clamp(0, 1)

    pages = upper.shape[-1]
    base_count = _page_count(seq_len, cfg.base_ratio, cfg.page_size, forced_count)
    max_count = _page_count(seq_len, cfg.max_ratio, cfg.page_size, forced_count)
    sorted_upper = upper.sort(dim=-1, descending=True).values
    cut = min(pages - 1, max(probe_count_scalar - 1, 0))
    next_cut = min(pages - 1, cut + 1)
    upper_scale = upper.float().std(dim=-1).clamp_min(1e-6)
    boundary_gap = ((sorted_upper[..., cut] - sorted_upper[..., next_cut]) / upper_scale).clamp_min(0)
    bound_uncertainty = torch.exp(-boundary_gap / cfg.bound_gap_temperature)

    uncertainty = (
        cfg.uncertainty_hash_weight * hash_uncertainty
        + cfg.uncertainty_tie_weight * tie_uncertainty
        + cfg.uncertainty_bound_weight * bound_uncertainty
    ).clamp(0, 1)
    dynamic_count = torch.ceil(
        base_count + uncertainty * (max_count - base_count)
    ).long().clamp(min=probe_count_scalar, max=max_count)

    # Optional safe expansion: retrieved exact scores provide a conservative
    # threshold; any page whose strict upper bound crosses it remains plausible.
    if cfg.exact_guard:
        _, probe_idx_kv, probe_valid_kv = _hash_distance(
            q_codes, k_codes, probe_idx, probe_mask, cfg.num_sink
        )
        top_probe = probe_dist.topk(final_k, dim=-1, largest=False).indices
        retrieved_kv = probe_idx_kv.gather(-1, top_probe)
        retrieved = retrieved_kv.repeat_interleave(group, dim=1)
        kv_for_head = torch.arange(query_heads, device=query.device) // group
        k_h = key[:, :, kv_for_head, :].permute(0, 2, 1, 3)
        selected_k = k_h.gather(
            2, retrieved.unsqueeze(-1).expand(-1, -1, -1, key.shape[-1])
        )
        exact = (query.unsqueeze(2).float() * selected_k.float()).sum(dim=-1)
        exact_threshold = exact.amin(dim=-1).reshape(batch, kv_heads, group).amin(dim=-1)
        plausible = upper >= (exact_threshold.unsqueeze(-1) - cfg.exact_guard_margin)
        plausible_count = plausible.sum(dim=-1).clamp(max=max_count)
        dynamic_count = torch.maximum(dynamic_count, plausible_count)

    return dynamic_count, margin, tie, boundary_gap, uncertainty, kth_h


def exact_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    token_indices: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Correctness reference: gather untouched K/V and compute exact attention."""
    if query.ndim == 4:
        query = query[:, 0]
    batch, query_heads, dim = query.shape
    kv_heads = key.shape[2]
    group = query_heads // kv_heads
    kv_for_head = torch.arange(query_heads, device=query.device) // group
    k_h = key[:, :, kv_for_head, :].permute(0, 2, 1, 3)
    v_h = value[:, :, kv_for_head, :].permute(0, 2, 1, 3)
    gather = token_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
    selected_k = k_h.gather(2, gather)
    selected_v = v_h.gather(2, gather)
    logits = (query.unsqueeze(2).float() * selected_k.float()).sum(dim=-1)
    logits *= scale if scale is not None else dim ** -0.5
    probs = logits.softmax(dim=-1).to(selected_v.dtype)
    return (probs.unsqueeze(-1) * selected_v).sum(dim=2)
