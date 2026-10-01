"""HHR joint training objective.

The objective has two teachers that share an orthogonal coordinate system:
the exact dot-product/Page teacher for QUEST and exact token ranking for Hash.
The original K/V tensors are never normalized or overwritten.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from unified_quest_hash import build_page_minmax, rms_normalize


@dataclass
class JointLossWeights:
    paper2_soft: float = 1.0
    token_rank: float = 1.0
    token_listwise: float = 0.35
    quest_miss: float = 1.0
    quest_slack: float = 0.10
    complementarity: float = 0.20
    quantization: float = 0.02
    bit_balance: float = 0.01
    decorrelation: float = 0.01
    hash_orthogonal: float = 0.01
    rotation_regularizer: float = 0.001
    oracle_gap: float = 1.0
    condition_regularizer: float = 0.01


@dataclass
class JointTrainConfig:
    sparse_ratio: float = 0.015
    # Forced prefix tokens consume part of the final sparse budget exactly as
    # in KVLib.hamming_score(..., sink=num_sink) during deployment.
    num_sink: int = 0
    quest_train_ratio: float = 0.16
    # Optional deployment-aligned candidate ratios, one fixed value per KV
    # head. These override quest_train_ratio and stay fixed across Queries.
    quest_head_ratios: Optional[Tuple[float, ...]] = None
    page_size: int = 8
    margin: float = 0.5
    hash_temperature: float = 0.5
    paper2_temperature: float = 0.1
    paper2_budget_weight: float = 0.1
    projection_scale: float = 0.1
    teacher_temperature: float = 1.0
    page_teacher_temperature: float = 1.0
    negatives_per_positive: int = 4
    quest_false_positive_boost: float = 0.75
    slack_cap: float = 8.0
    # The strict magnitude-preserving mainline keeps raw Q/K amplitudes.
    # The released HHR setting keeps raw Q/K amplitudes.
    use_rms: bool = False
    amplitude_clip: Optional[float] = None
    learn_shared_rotation: bool = True
    # identity | orthogonal | paired_nonorth
    transform_mode: str = "paired_nonorth"
    # parallel hashes raw Q/K; serial hashes the transformed search Q/K.
    hash_branch: str = "parallel"
    # shared uses one R per layer; kv_head learns one R per KV head.
    rotation_granularity: str = "shared"
    nonorth_source_scale: float = 0.05
    condition_target: float = 4.0
    straight_through_binary: bool = True
    weights: JointLossWeights = field(default_factory=JointLossWeights)


def _cayley(skew_source: torch.Tensor) -> torch.Tensor:
    """Differentiable exactly-orthogonal rotation from an unconstrained matrix."""
    skew = skew_source - skew_source.transpose(-1, -2)
    eye = torch.eye(skew.shape[-1], device=skew.device, dtype=skew.dtype)
    return torch.linalg.solve(eye + skew, eye - skew)


def _inverse_cayley_source(rotation: torch.Tensor) -> torch.Tensor:
    """Recover a source whose skew part reproduces an orthogonal rotation."""
    eye = torch.eye(
        rotation.shape[-1], device=rotation.device, dtype=rotation.dtype
    )
    # A (R + I) = I - R, and _cayley consumes source-source.T == A.
    skew = torch.linalg.solve(
        (rotation + eye).transpose(-1, -2),
        (eye - rotation).transpose(-1, -2),
    ).transpose(-1, -2)
    skew = 0.5 * (skew - skew.transpose(-1, -2))
    return 0.5 * skew


def _pad_tokens(x: torch.Tensor, pages: int, page_size: int, value: float) -> torch.Tensor:
    pad = pages * page_size - x.shape[-1]
    return F.pad(x, (0, pad), value=value) if pad else x


def _zscore(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return (x - x.mean(dim=dim, keepdim=True)) / x.std(
        dim=dim, keepdim=True, unbiased=False
    ).clamp_min(1e-5)


class UnifiedQuestHashTrainer(nn.Module):
    """Joint QUEST box-tightening and candidate-aware binary Hash training."""

    def __init__(
        self,
        kv_heads: int,
        query_heads: int,
        head_dim: int,
        hash_bits: int = 128,
        config: JointTrainConfig = JointTrainConfig(),
        initial_hash_weight: Optional[torch.Tensor] = None,
        initial_rotation: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if query_heads % kv_heads:
            raise ValueError("query_heads must be divisible by kv_heads")
        self.kv_heads = kv_heads
        self.query_heads = query_heads
        self.group = query_heads // kv_heads
        self.head_dim = head_dim
        self.hash_bits = hash_bits
        self.config = config
        if config.num_sink < 0:
            raise ValueError("num_sink must be nonnegative")
        if config.quest_head_ratios is not None:
            if len(config.quest_head_ratios) != kv_heads:
                raise ValueError("quest_head_ratios must contain one value per KV head")
            if any(not 0.0 < ratio <= 1.0 for ratio in config.quest_head_ratios):
                raise ValueError("quest_head_ratios must be in (0, 1]")
            if any(ratio < config.sparse_ratio for ratio in config.quest_head_ratios):
                raise ValueError("every QUEST ratio must cover final sparse_ratio")
        elif config.quest_train_ratio < config.sparse_ratio:
            raise ValueError("quest_train_ratio must cover final sparse_ratio")
        if initial_hash_weight is None:
            weight = torch.empty(kv_heads, head_dim, hash_bits)
            nn.init.orthogonal_(weight.reshape(kv_heads * head_dim, hash_bits))
            weight.mul_(math.sqrt(head_dim))
        else:
            if tuple(initial_hash_weight.shape) != (kv_heads, head_dim, hash_bits):
                raise ValueError("unexpected initial_hash_weight shape")
            weight = initial_hash_weight.detach().float().clone()
        self.hash_weight = nn.Parameter(weight)
        if config.rotation_granularity not in {"shared", "kv_head"}:
            raise ValueError("rotation_granularity must be shared or kv_head")
        rotation_shape = (
            (head_dim, head_dim)
            if config.rotation_granularity == "shared"
            else (kv_heads, head_dim, head_dim)
        )
        rotation_source = torch.zeros(rotation_shape)
        nonorth_base_rotation = torch.eye(head_dim)
        if config.rotation_granularity == "kv_head":
            nonorth_base_rotation = nonorth_base_rotation.unsqueeze(0).expand(
                kv_heads, -1, -1
            ).clone()
        if initial_rotation is not None:
            initial_rotation = initial_rotation.detach().float()
            if (
                config.rotation_granularity == "kv_head"
                and tuple(initial_rotation.shape) == (head_dim, head_dim)
            ):
                initial_rotation = initial_rotation.unsqueeze(0).expand(
                    kv_heads, -1, -1
                ).clone()
            if tuple(initial_rotation.shape) != rotation_shape:
                raise ValueError("unexpected initial_rotation shape")
            if config.transform_mode == "paired_nonorth":
                # Keep the archived/formal rotation as an explicit invertible
                # base.  Learning starts from that exact matrix and composes a
                # head-specific matrix exponential on its right.  This avoids
                # pretending that an inverse-Cayley source is a matrix log.
                nonorth_base_rotation.copy_(initial_rotation)
            else:
                rotation_source.copy_(
                    _inverse_cayley_source(initial_rotation)
                )
        self.register_buffer(
            "nonorth_base_rotation", nonorth_base_rotation, persistent=True
        )
        self.rotation_source = nn.Parameter(rotation_source)

    def shared_rotation(self) -> torch.Tensor:
        eye = torch.eye(
            self.head_dim, device=self.hash_weight.device, dtype=self.hash_weight.dtype
        )
        if self.config.rotation_granularity == "kv_head":
            eye = eye.unsqueeze(0).expand(self.kv_heads, -1, -1)
        if not self.config.learn_shared_rotation:
            return eye
        if self.config.transform_mode == "identity":
            return eye
        if self.config.transform_mode == "orthogonal":
            return _cayley(self.rotation_source)
        if self.config.transform_mode == "paired_nonorth":
            learned_delta = torch.matrix_exp(
                self.rotation_source.float() * self.config.nonorth_source_scale
            )
            return self.nonorth_base_rotation.float() @ learned_delta
        raise ValueError(f"unsupported transform_mode: {self.config.transform_mode}")

    def _search_transform(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rotation = self.shared_rotation()
        if rotation.ndim == 3:
            q_grouped = query.float().reshape(
                query.shape[0], self.kv_heads, self.group, self.head_dim
            )
            k_search = torch.einsum("bskd,kde->bske", key.float(), rotation)
            if self.config.transform_mode == "paired_nonorth":
                # Row-vector form: Q_h R_h^{-T}, K_h R_h.  Solve every
                # (layer, KV-head) system in a batch instead of materializing
                # inverses; corresponding GQA query heads reuse the same R_h.
                q_search = torch.linalg.solve(
                    rotation.unsqueeze(0), q_grouped.transpose(-1, -2)
                ).transpose(-1, -2)
            else:
                q_search = torch.einsum("bkgd,kde->bkge", q_grouped, rotation)
            return q_search.reshape(query.shape), k_search, rotation
        if self.config.transform_mode != "paired_nonorth":
            return query.float() @ rotation, key.float() @ rotation, rotation
        # For row vectors, Q R^{-T} and K R preserve QK^T even when R is not
        # orthogonal.  Solve avoids materializing an explicit inverse.
        flat_q = query.float().reshape(-1, self.head_dim)
        q_search = torch.linalg.solve(
            rotation, flat_q.transpose(0, 1)
        ).transpose(0, 1).reshape(query.shape)
        k_search = key.float() @ rotation
        return q_search, k_search, rotation

    def _exact_scores(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        q = query.reshape(query.shape[0], self.kv_heads, self.group, self.head_dim)
        score = torch.einsum("bkgd,bskd->bkgs", q.float(), key.float())
        return score.reshape(query.shape[0], self.query_heads, key.shape[1])

    def _upper_per_head(
        self, query: torch.Tensor, page_mins: torch.Tensor, page_maxs: torch.Tensor
    ) -> torch.Tensor:
        batch, pages, _, _ = page_mins.shape
        q = query.reshape(batch, self.kv_heads, self.group, self.head_dim).float()
        mins = page_mins.float().permute(0, 2, 1, 3)
        maxs = page_maxs.float().permute(0, 2, 1, 3)
        upper = torch.einsum("bkgd,bkpd->bkgp", q.clamp_min(0), maxs)
        upper = upper + torch.einsum("bkgd,bkpd->bkgp", q.clamp_max(0), mins)
        return upper

    def _soft_hash(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q = query.reshape(query.shape[0], self.kv_heads, self.group, self.head_dim)
        k = key
        if self.config.use_rms:
            q = rms_normalize(q)
            k = rms_normalize(k)
        if self.config.amplitude_clip is not None:
            clip = torch.as_tensor(
                self.config.amplitude_clip, device=q.device, dtype=q.dtype
            ).clamp_min(1e-4)
            q = clip * torch.tanh(q / clip)
            k = clip * torch.tanh(k / clip)
        q_logits = torch.einsum("bkgd,kdr->bkgr", q.float(), self.hash_weight)
        k_logits = torch.einsum("bskd,kdr->bskr", k.float(), self.hash_weight)
        q_soft = torch.tanh(
            self.config.projection_scale * q_logits / self.config.hash_temperature
        )
        k_soft = torch.tanh(
            self.config.projection_scale * k_logits / self.config.hash_temperature
        )
        if self.config.straight_through_binary:
            q_hard = torch.where(q_logits >= 0, 1.0, -1.0)
            k_hard = torch.where(k_logits >= 0, 1.0, -1.0)
            q_code = q_soft + (q_hard - q_soft).detach()
            k_code = k_soft + (k_hard - k_soft).detach()
        else:
            q_code, k_code = q_soft, k_soft
        shape = (query.shape[0], self.query_heads, self.hash_bits)
        return q_code.reshape(shape), k_code, q_soft.reshape(shape), k_soft

    def _distance(self, q_code: torch.Tensor, k_code: torch.Tensor) -> torch.Tensor:
        k_for_head = k_code.repeat_interleave(self.group, dim=2).permute(0, 2, 1, 3)
        # Relaxed Hamming distance; its hard limit is proportional to XOR popcount.
        per_head = ((q_code.unsqueeze(2) - k_for_head) ** 2).mean(dim=-1)
        # The deployed CUDA kernel sums Hamming distances over all Query heads
        # sharing one KV head and returns one shared token set.  Mean has the
        # same ordering as that sum while preserving the previous loss scale.
        return per_head.reshape(
            q_code.shape[0], self.kv_heads, self.group, k_code.shape[1]
        ).mean(dim=2)

    def _regularizers(
        self, q_code: torch.Tensor, k_code: torch.Tensor, rotation: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        all_codes = torch.cat(
            (q_code.reshape(-1, self.hash_bits), k_code.reshape(-1, self.hash_bits)),
            dim=0,
        )
        if all_codes.shape[0] > 4096:
            pick = torch.randperm(all_codes.shape[0], device=all_codes.device)[:4096]
            all_codes = all_codes[pick]
        quantization = (all_codes.abs() - 1).square().mean()
        bit_balance = all_codes.mean(dim=0).square().mean()
        centered = all_codes - all_codes.mean(dim=0, keepdim=True)
        covariance = centered.transpose(0, 1) @ centered / max(1, centered.shape[0])
        covariance.fill_diagonal_(0)
        decorrelation = covariance.square().mean()
        gram = torch.einsum("kdr,kds->krs", self.hash_weight, self.hash_weight)
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(dim=-1, keepdim=True).clamp_min(1e-6)
        gram = gram / scale.unsqueeze(-1)
        eye_bits = torch.eye(self.hash_bits, device=gram.device, dtype=gram.dtype)
        hash_orthogonal = (gram - eye_bits).square().mean()
        eye_dim = torch.eye(self.head_dim, device=rotation.device, dtype=rotation.dtype)
        rotation_regularizer = (rotation - eye_dim).square().mean()
        singular = torch.linalg.svdvals(rotation.float())
        condition_each = singular.amax(dim=-1) / singular.amin(dim=-1).clamp_min(1e-8)
        condition = condition_each.amax()
        log_singular = singular.log()
        condition_regularizer = (
            log_singular.square().mean()
            + log_singular.mean().square()
            + F.relu(condition - self.config.condition_target).square()
        )
        return {
            "quantization": quantization,
            "bit_balance": bit_balance,
            "decorrelation": decorrelation,
            "hash_orthogonal": hash_orthogonal,
            "rotation_regularizer": rotation_regularizer,
            "condition_regularizer": condition_regularizer,
        }

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        exact_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all joint losses for a page-preserving batch.

        query is [B,H,D], key is [B,S,KVH,D], and optional exact_scores is
        [B,H,S].  Sequence order must be intact.
        """
        if query.ndim != 3 or key.ndim != 4:
            raise ValueError("query/key must be [B,H,D] and [B,S,KVH,D]")
        batch, seq_len, kv_heads, dim = key.shape
        if kv_heads != self.kv_heads or dim != self.head_dim:
            raise ValueError("batch does not match trainer configuration")
        if exact_scores is None:
            exact_scores = self._exact_scores(query, key)
        exact_scores = exact_scores.float()
        attention_scores = exact_scores / math.sqrt(self.head_dim)
        attention_probability = attention_scores.softmax(dim=-1)
        # Shared GQA retrieval teacher: maximize attention mass jointly across
        # the Query heads that consume the same KV-head token list.
        group_teacher = attention_probability.reshape(
            batch, self.kv_heads, self.group, seq_len
        ).mean(dim=2)
        final_k = max(1, math.floor(seq_len * self.config.sparse_ratio))
        sink_count = min(self.config.num_sink, final_k, seq_len)
        learned_k = final_k - sink_count
        if learned_k < 1:
            raise ValueError(
                "training sample is too short for sink-aware learned budget: "
                f"seq_len={seq_len}, final_k={final_k}, sink_count={sink_count}"
            )
        pages = math.ceil(seq_len / self.config.page_size)
        sink_pages = math.ceil(sink_count / self.config.page_size)
        if self.config.quest_head_ratios is None:
            head_ratios = [self.config.quest_train_ratio] * self.kv_heads
        else:
            head_ratios = list(self.config.quest_head_ratios)
        quest_pages_per_head = [
            min(
                pages,
                max(1, math.ceil(seq_len * ratio / self.config.page_size)),
            )
            for ratio in head_ratios
        ]

        q_rot, k_rot, rotation = self._search_transform(query, key)
        page_mins, page_maxs = build_page_minmax(k_rot, self.config.page_size)
        upper_h = self._upper_per_head(q_rot, page_mins, page_maxs)
        upper_kv = upper_h.amax(dim=2)
        selected_page_mask = torch.zeros_like(upper_kv, dtype=torch.bool)
        forced_page_mask = torch.zeros_like(selected_page_mask)
        if sink_pages:
            forced_page_mask[..., :sink_pages] = True
        selection_upper = upper_kv.masked_fill(forced_page_mask, torch.inf)
        quest_cutoff = torch.empty(
            batch, self.kv_heads, 1, device=key.device, dtype=upper_kv.dtype
        )
        # KV-head count is small. Variable per-head Top-K remains explicit;
        # all expensive bound computation is still batched on GPU.
        for head, quest_pages in enumerate(quest_pages_per_head):
            if quest_pages < sink_pages:
                raise RuntimeError("QUEST budget cannot cover forced sink pages")
            values, ranked_pages = selection_upper[:, head].topk(
                quest_pages, dim=-1
            )
            selected_page_mask[:, head].scatter_(-1, ranked_pages, True)
            quest_cutoff[:, head, 0] = values[..., -1]

        if self.config.hash_branch == "parallel":
            q_hash_input, k_hash_input = query, key
        elif self.config.hash_branch == "serial":
            q_hash_input, k_hash_input = q_rot, k_rot
        else:
            raise ValueError(f"unsupported hash_branch: {self.config.hash_branch}")
        q_code, k_code, q_soft, k_soft = self._soft_hash(
            q_hash_input, k_hash_input
        )
        distance = self._distance(q_code, k_code)
        # Deployment label contract: forced prefix sinks occupy sink_count
        # slots; the remaining learned_k positives are the best exact tokens
        # outside that prefix.  This matches the actual final Top-K capacity.
        forced_token_mask = torch.zeros_like(group_teacher, dtype=torch.bool)
        if sink_count:
            forced_token_mask[..., :sink_count] = True
        nonforced_teacher = group_teacher.masked_fill(
            forced_token_mask, -torch.inf
        )
        learned_topk = nonforced_teacher.topk(learned_k, dim=-1).indices
        forced_idx = torch.arange(
            sink_count, device=key.device, dtype=learned_topk.dtype
        ).view(1, 1, sink_count).expand(batch, self.kv_heads, sink_count)
        true_topk = torch.cat((forced_idx, learned_topk), dim=-1)
        positive_mask = torch.zeros_like(group_teacher, dtype=torch.bool)
        positive_mask.scatter_(-1, true_topk, True)

        # Fixed-budget oracle: among all pages, select the attainable set with
        # the largest exact-positive attention mass.  This is an upper bound on
        # page recall at the *same* per-head budget, not a denser oracle.
        positive_token_count = positive_mask.float()
        positive_count_padded = _pad_tokens(
            positive_token_count, pages, self.config.page_size, 0.0
        ).reshape(batch, self.kv_heads, pages, self.config.page_size)
        positive_page_count = positive_count_padded.sum(dim=-1)
        oracle_page_mask = torch.zeros_like(selected_page_mask)
        for head, quest_pages in enumerate(quest_pages_per_head):
            oracle_idx = positive_page_count[:, head].topk(
                quest_pages, dim=-1
            ).indices
            oracle_page_mask[:, head].scatter_(-1, oracle_idx, True)
        # Do not treat zero-positive filler pages as oracle targets.
        oracle_page_mask &= positive_page_count > 0
        total_positive_count = positive_page_count.sum(dim=-1).clamp_min(1e-12)
        oracle_recall_upper = (
            positive_page_count * oracle_page_mask.float()
        ).sum(dim=-1) / total_positive_count
        actual_page_token_recall = (
            positive_page_count * selected_page_mask.float()
        ).sum(dim=-1) / total_positive_count
        oracle_gap = (oracle_recall_upper - actual_page_token_recall).clamp_min(0)
        missed_oracle = oracle_page_mask & ~selected_page_mask
        oracle_weight = positive_page_count / total_positive_count.unsqueeze(-1)

        # Optimize the decision that actually determines fixed-budget recall.
        # For each KV head, the weakest oracle-positive pages must outrank the
        # strongest selected non-oracle pages.  The earlier one-sided cutoff
        # pull could increase every bound together and therefore did not
        # reliably improve the ordering.  Hard-positive/hard-negative mining
        # keeps this O(H * m^2), independent of the number of context pages.
        normalized_upper = _zscore(upper_kv)
        nonoracle_selected = selected_page_mask & ~oracle_page_mask
        hard_count = min(8, pages)
        hard_pos_values, hard_pos_idx = normalized_upper.masked_fill(
            ~oracle_page_mask, torch.inf
        ).topk(hard_count, dim=-1, largest=False)
        hard_neg_values, hard_neg_idx = normalized_upper.masked_fill(
            ~nonoracle_selected, -torch.inf
        ).topk(hard_count, dim=-1, largest=True)
        hard_pos_valid = oracle_page_mask.gather(-1, hard_pos_idx)
        hard_neg_valid = nonoracle_selected.gather(-1, hard_neg_idx)
        hard_pos_weight = oracle_weight.gather(-1, hard_pos_idx)
        pair_valid = hard_pos_valid.unsqueeze(-1) & hard_neg_valid.unsqueeze(-2)
        pair_weight = hard_pos_weight.unsqueeze(-1) * pair_valid.float()
        oracle_pairwise = F.softplus(
            self.config.margin
            + hard_neg_values.unsqueeze(-2)
            - hard_pos_values.unsqueeze(-1)
        )
        oracle_gap_loss = (oracle_pairwise * pair_weight).sum() / (
            pair_weight.sum().clamp_min(1e-12)
        )

        token_page = torch.arange(seq_len, device=key.device) // self.config.page_size
        selected_token_kv = selected_page_mask[..., token_page]
        if bool((selected_token_kv.sum(dim=-1) < final_k).any()):
            raise RuntimeError("QUEST candidate set is smaller than final Token Top-K")
        learned_candidate_mask = selected_token_kv & ~forced_token_mask
        candidate_negative = learned_candidate_mask & ~positive_mask

        # Oracle positive pages are used both to detect QUEST misses and to label
        # QUEST false-positive pages for candidate-aware Hash hard negatives.
        positive_page_kv = torch.zeros(
            batch, self.kv_heads, pages, dtype=torch.bool, device=key.device
        )
        positive_page_kv.scatter_(-1, true_topk // self.config.page_size, True)
        missed_page = positive_page_kv & ~selected_page_mask
        false_positive_page = selected_page_mask & ~positive_page_kv

        exact_padded = _pad_tokens(
            exact_scores, pages, self.config.page_size, -torch.inf
        ).reshape(batch, self.query_heads, pages, self.config.page_size)
        exact_page_h = exact_padded.amax(dim=-1).reshape(
            batch, self.kv_heads, self.group, pages
        )
        exact_page_kv = exact_page_h.amax(dim=2)
        slack = (upper_kv - exact_page_kv).clamp_min(0)
        slack_scale = exact_page_kv.std(dim=-1, keepdim=True).clamp_min(1e-5)
        normalized_slack = (slack / slack_scale).clamp_max(self.config.slack_cap)

        # Hash hard negatives are selected inside QUEST candidates.  QUEST false
        # positives receive an extra boost proportional to bound slack.
        fp_token_kv = false_positive_page[..., token_page]
        slack_token_kv = normalized_slack[..., token_page]
        priority = -distance.detach()
        priority = priority + self.config.quest_false_positive_boost * (
            fp_token_kv.float() * (1 + slack_token_kv)
        )
        priority.masked_fill_(~candidate_negative, -torch.inf)
        negatives = min(
            seq_len - final_k,
            max(1, learned_k * self.config.negatives_per_positive),
        )
        neg_idx = priority.topk(negatives, dim=-1).indices
        neg_valid = candidate_negative.gather(-1, neg_idx)
        pos_dist = distance.gather(-1, true_topk)
        neg_dist = distance.gather(-1, neg_idx)
        # Hash learns only exact positives that survived QUEST. Missed positives
        # are excluded here and train the QUEST projection through quest_miss.
        positive_in_candidate = selected_token_kv.gather(-1, true_topk)
        positive_trainable = ~forced_token_mask.gather(-1, true_topk)
        pair_valid = (
            positive_in_candidate.unsqueeze(-1)
            & positive_trainable.unsqueeze(-1)
            & neg_valid.unsqueeze(-2)
        )
        pairwise = F.softplus(
            self.config.margin + pos_dist.unsqueeze(-1) - neg_dist.unsqueeze(-2)
        )
        pos_weight = group_teacher.gather(-1, true_topk)
        weighted_pair = pairwise * pos_weight.unsqueeze(-1) * pair_valid.float()
        token_rank = weighted_pair.sum() / (
            pos_weight.unsqueeze(-1) * pair_valid.float()
        ).sum().clamp_min(1e-12)

        # Original Paper2 principle, now restricted to the exact deployment
        # candidate set: a differentiable Top-K mask maximizes surviving-positive
        # recall while keeping the expected selected count equal to final_k.
        paper2_score = -distance
        candidate_score = paper2_score.masked_fill(
            ~learned_candidate_mask, -torch.inf
        )
        paper2_threshold = candidate_score.topk(
            learned_k, dim=-1
        ).values[..., -1:]
        learned_soft_mask = torch.sigmoid(
            (paper2_score - paper2_threshold) / self.config.paper2_temperature
        ) * learned_candidate_mask.float()
        paper2_soft_mask = learned_soft_mask + forced_token_mask.float()
        surviving_positive = positive_mask & selected_token_kv
        surviving_count = surviving_positive.sum(dim=-1)
        paper2_recall = (
            paper2_soft_mask * surviving_positive.float()
        ).sum(dim=-1) / surviving_count.clamp_min(1)
        valid_teacher = surviving_count > 0
        paper2_recall_loss = -torch.log(
            paper2_recall[valid_teacher].mean().clamp_min(1e-8)
        ) if bool(valid_teacher.any()) else distance.sum() * 0
        paper2_budget = (
            paper2_soft_mask.sum(dim=-1) / float(final_k) - 1.0
        ).square().mean()
        paper2_soft = (
            paper2_recall_loss
            + self.config.paper2_budget_weight * paper2_budget
        )

        # Listwise distillation is candidate-restricted and uses exact attention
        # scores as teacher.  This corrects ordering among positives, not only recall.
        distill_mask = learned_candidate_mask
        masked_distance = distance.masked_fill(~distill_mask, torch.inf)
        masked_teacher = group_teacher.masked_fill(~distill_mask, 0)
        student_logp = F.log_softmax(
            -masked_distance / self.config.hash_temperature, dim=-1
        )
        teacher_p = masked_teacher / masked_teacher.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        # F.kl_div sees 0 * (log(0) - -inf) outside the candidate set and may
        # produce NaN.  Mask those terms explicitly before reducing.
        teacher_logp = torch.where(
            distill_mask,
            teacher_p.clamp_min(1e-12).log(),
            torch.zeros_like(teacher_p),
        )
        listwise_terms = teacher_p * (teacher_logp - student_logp)
        listwise_terms = listwise_terms.masked_fill(~distill_mask, 0)
        token_listwise = listwise_terms.sum(dim=-1).mean()

        # Any oracle-positive page below the QUEST boundary is explicitly pulled
        # above it.  This loss has gradients through the shared Cayley rotation.
        quest_miss_terms = F.relu(
            quest_cutoff + self.config.margin - upper_kv
        )
        miss_train_mask = missed_page
        quest_miss = (
            quest_miss_terms * miss_train_mask.float()
        ).sum() / miss_train_mask.sum().clamp_min(1)
        fp_weight = false_positive_page.float() * (1 + normalized_slack.detach())
        quest_slack = (
            torch.log1p(normalized_slack) * fp_weight
        ).sum() / fp_weight.sum().clamp_min(1)

        # Complementarity: QUEST magnitude evidence plus best Hash direction
        # evidence should reproduce exact page ordering.  It does not force the
        # two signals to be identical.
        distance_padded = _pad_tokens(
            -distance, pages, self.config.page_size, -torch.inf
        ).reshape(batch, self.kv_heads, pages, self.config.page_size)
        hash_page_kv = torch.logsumexp(
            distance_padded / self.config.hash_temperature, dim=-1
        )
        combined = _zscore(upper_kv) + _zscore(hash_page_kv)
        exact_page_prob = F.softmax(
            _zscore(exact_page_kv) / self.config.page_teacher_temperature, dim=-1
        )
        complementarity = F.kl_div(
            F.log_softmax(combined, dim=-1), exact_page_prob, reduction="batchmean"
        )

        regularizers = self._regularizers(q_soft, k_soft, rotation)
        component = {
            "paper2_soft": paper2_soft,
            "token_rank": token_rank,
            "token_listwise": token_listwise,
            "quest_miss": quest_miss,
            "quest_slack": quest_slack,
            "complementarity": complementarity,
            "oracle_gap": oracle_gap_loss,
            **regularizers,
        }
        weights = self.config.weights
        total = sum(getattr(weights, name) * loss for name, loss in component.items())
        with torch.no_grad():
            # Deployment retrieves only inside QUEST candidates. Metrics must
            # use the same mask instead of an optimistic global Hash Top-K.
            hard_distance = distance.masked_fill(~selected_token_kv, torch.inf)
            hard_distance = hard_distance.masked_fill(
                forced_token_mask, -torch.inf
            )
            retrieved = hard_distance.topk(final_k, dim=-1, largest=False).indices
            token_recall = positive_mask.gather(-1, retrieved).float().mean()
            retrieved_attention_mass = group_teacher.gather(
                -1, retrieved
            ).sum(dim=-1).mean()
            quest_token_recall = (
                (positive_mask & selected_token_kv).sum().float()
                / positive_mask.sum().clamp_min(1)
            )
            quest_page_recall = (
                (positive_page_kv & selected_page_mask).sum().float()
                / positive_page_kv.sum().clamp_min(1)
            )
            fp_rate = false_positive_page.sum().float() / selected_page_mask.sum().clamp_min(1)
        return {
            "loss": total,
            **component,
            "paper2_recall_loss": paper2_recall_loss,
            "paper2_budget_loss": paper2_budget,
            "paper2_candidate_recall": paper2_recall[valid_teacher].mean()
            if bool(valid_teacher.any())
            else distance.new_zeros(()),
            "token_recall": token_recall,
            "retrieved_attention_mass": retrieved_attention_mass,
            "quest_token_recall": quest_token_recall,
            "quest_page_recall": quest_page_recall,
            "quest_false_positive_rate": fp_rate,
            "oracle_page_recall_upper_bound": oracle_recall_upper.mean(),
            "actual_page_token_recall": actual_page_token_recall.mean(),
            "oracle_recall_gap": oracle_gap.mean(),
            "missed_oracle_pages": missed_oracle.sum(),
            "transform_condition_number": torch.linalg.cond(
                rotation.float()
            ).amax(),
            "transform_fro_from_identity": torch.linalg.matrix_norm(
                rotation.float()
                - torch.eye(
                    self.head_dim, device=rotation.device, dtype=torch.float32
                ),
                dim=(-2, -1),
            ).mean(),
            "quest_missed_pages": missed_page.sum(),
            "quest_candidate_ratio": selected_page_mask.float().mean(),
            "final_k": torch.tensor(final_k, device=key.device),
            "sink_count": torch.tensor(sink_count, device=key.device),
            "learned_k": torch.tensor(learned_k, device=key.device),
            # Non-scalar audit tensors are ignored by scalar training logs but
            # make the strict label/candidate contract directly testable.
            "teacher_topk": true_topk.detach(),
            "quest_selected_page_mask": selected_page_mask.detach(),
            "quest_missed_page_mask": missed_page.detach(),
        }
