# -*- coding: utf-8 -*-
"""
Pure PyTorch fallback for missing Triton qk_score.

Expected import:
    from myTransformer.cache.kernels.triton_qk_score import qk_score

Purpose:
    Compute exact QK logits for build_dataset:
        score = q @ k.T / sqrt(head_dim)

Design rule:
    The last dimension of the returned score is ALWAYS token length S.
    This prevents torch.topk(score, k, dim=-1) from accidentally operating
    on the head dimension.
"""

import math
from typing import Optional, Tuple

import torch


def _as_float(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(x):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.float()


def _parse_int_args(args, kwargs):
    """
    Original Triton qk_score wrappers may pass valid_len/query_idx as positional ints.
    This function accepts them without breaking compatibility.
    """
    valid_len = kwargs.pop("valid_len", None)
    query_idx = kwargs.pop("query_idx", None)

    ints = []
    for x in args:
        if isinstance(x, int):
            ints.append(int(x))
        elif torch.is_tensor(x) and x.dim() == 0:
            ints.append(int(x.item()))

    if valid_len is None and len(ints) >= 1:
        valid_len = ints[0]
    if query_idx is None and len(ints) >= 2:
        query_idx = ints[1]

    return valid_len, query_idx, kwargs


def _infer_key_hsd(
    key_states: torch.Tensor,
    head_dim: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert key to [Hk, S, D] for unbatched computation.

    Supported:
        [S, D]
        [S, Hk, D]
        [Hk, S, D]
        [1, S, Hk, D]
        [1, Hk, S, D]
    """
    k = key_states

    if k.dim() == 1:
        # [D] -> [1, 1, D]
        return k.view(1, 1, -1).contiguous()

    if k.dim() == 2:
        # [S, D] -> [1, S, D]
        return k.unsqueeze(0).contiguous()

    if k.dim() == 3:
        # Either [S, H, D] or [H, S, D]
        a, b, d = k.shape

        if head_dim is not None and d != head_dim:
            raise RuntimeError(f"key head_dim mismatch: got {d}, expected {head_dim}")

        if num_kv_heads is not None:
            if b == num_kv_heads:
                # [S, Hk, D] -> [Hk, S, D]
                return k.permute(1, 0, 2).contiguous()
            if a == num_kv_heads:
                # [Hk, S, D]
                return k.contiguous()

        # Heuristic: token length is usually larger than head count.
        if a >= b:
            # [S, H, D]
            return k.permute(1, 0, 2).contiguous()

        # [H, S, D]
        return k.contiguous()

    if k.dim() == 4:
        # Only support batch size 1 safely for this fallback path.
        if k.shape[0] != 1:
            raise RuntimeError(
                f"qk_score fallback received batched key with B={k.shape[0]}. "
                "This fallback expects B=1 for build_dataset."
            )

        k = k[0]

        # Now [S, H, D] or [H, S, D]
        return _infer_key_hsd(k, head_dim=head_dim, num_kv_heads=num_kv_heads)

    raise RuntimeError(f"Unsupported key shape: {tuple(k.shape)}")


def _infer_query_hqd(
    query_states: torch.Tensor,
    head_dim: Optional[int] = None,
    num_heads: Optional[int] = None,
) -> torch.Tensor:
    """
    Convert query to [Hq, Q, D] for unbatched computation.

    Supported:
        [D]
        [1, D]
        [H, D]
        [Q, D]
        [H, Q, D]
        [Q, H, D]
        [1, H, Q, D]
        [1, Q, H, D]
    """
    q = query_states

    if q.dim() == 1:
        # [D] -> [1, 1, D]
        return q.view(1, 1, -1).contiguous()

    if q.dim() == 2:
        a, d = q.shape

        if head_dim is not None and d != head_dim:
            raise RuntimeError(f"query head_dim mismatch: got {d}, expected {head_dim}")

        if num_heads is not None and a == num_heads:
            # [H, D] -> [H, 1, D]
            return q.unsqueeze(1).contiguous()

        # [1, D] or [Q, D] -> [1, Q, D]
        return q.unsqueeze(0).contiguous()

    if q.dim() == 3:
        # Either [H, Q, D] or [Q, H, D] or [B, H, D]
        a, b, d = q.shape

        if head_dim is not None and d != head_dim:
            raise RuntimeError(f"query head_dim mismatch: got {d}, expected {head_dim}")

        if num_heads is not None:
            if a == num_heads:
                # [H, Q, D]
                return q.contiguous()
            if b == num_heads:
                # [Q, H, D] -> [H, Q, D]
                return q.permute(1, 0, 2).contiguous()

        # Heuristic:
        # if first dim is small and second dim may be Q, treat as [H,Q,D]
        if a <= b:
            return q.contiguous()

        # [Q,H,D] -> [H,Q,D]
        return q.permute(1, 0, 2).contiguous()

    if q.dim() == 4:
        if q.shape[0] != 1:
            raise RuntimeError(
                f"qk_score fallback received batched query with B={q.shape[0]}. "
                "This fallback expects B=1 for build_dataset."
            )

        q = q[0]
        return _infer_query_hqd(q, head_dim=head_dim, num_heads=num_heads)

    raise RuntimeError(f"Unsupported query shape: {tuple(q.shape)}")


def _align_hq_hk(
    q_hqd: torch.Tensor,
    k_hsd: torch.Tensor,
    reduce_single_query_head: bool = True,
    reduce_head: str = "max",
) -> Tuple[torch.Tensor, bool]:
    """
    q_hqd: [Hq, Q, D]
    k_hsd: [Hk, S, D]

    Return:
        score_hqs: [Hout, Q, S]
        reduced: whether score was reduced over heads.

    Cases:
        Hq == Hk:
            score [H, Q, S]

        Hq > Hk and Hq % Hk == 0:
            repeat K heads to Hq.

        Hq == 1 and Hk > 1:
            compare single q against all KV heads, then reduce over head -> [1, Q, S].
            This is important because build_dataset usually expects token scores.
    """
    hq, q_len, d = q_hqd.shape
    hk, s_len, kd = k_hsd.shape

    if d != kd:
        raise RuntimeError(f"head_dim mismatch: query D={d}, key D={kd}")

    q = q_hqd.float()
    k = k_hsd.float()

    if hq == hk:
        score = torch.einsum("hqd,hsd->hqs", q, k) / math.sqrt(d)
        return score, False

    if hq > hk and hq % hk == 0:
        repeat_factor = hq // hk
        k_rep = k.repeat_interleave(repeat_factor, dim=0)
        score = torch.einsum("hqd,hsd->hqs", q, k_rep) / math.sqrt(d)
        return score, False

    if hq == 1 and hk > 1:
        q_rep = q.expand(hk, -1, -1).contiguous()
        score = torch.einsum("hqd,hsd->hqs", q_rep, k) / math.sqrt(d)
        # score: [Hk, Q, S]
        if reduce_single_query_head:
            if reduce_head == "mean":
                score = score.mean(dim=0, keepdim=True)
            else:
                score = score.amax(dim=0, keepdim=True)
            return score, True
        return score, False

    if hk == 1 and hq > 1:
        k_rep = k.expand(hq, -1, -1).contiguous()
        score = torch.einsum("hqd,hsd->hqs", q, k_rep) / math.sqrt(d)
        return score, False

    if hk > hq and hk % hq == 0:
        repeat_factor = hk // hq
        q_rep = q.repeat_interleave(repeat_factor, dim=0)
        score = torch.einsum("hqd,hsd->hqs", q_rep, k) / math.sqrt(d)
        if reduce_single_query_head:
            if reduce_head == "mean":
                score = score.mean(dim=0, keepdim=True)
            else:
                score = score.amax(dim=0, keepdim=True)
            return score, True
        return score, False

    raise RuntimeError(f"Cannot align heads: Hq={hq}, Hk={hk}")


@torch.no_grad()
def qk_score(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    *args,
    num_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    valid_len: Optional[int] = None,
    query_idx: Optional[int] = None,
    causal: bool = False,
    return_layout: str = "auto",
    reduce_head: str = "max",
    debug: bool = False,
    **kwargs,
) -> torch.Tensor:
    """
    Compute QK score.

    The last dimension is always sequence length S.

    return_layout:
        auto:
            If single query and single/reduced head -> [S]
            If single query and multiple heads       -> [H, S]
            If multiple query positions             -> [H, Q, S]
        hqs:
            Always [H, Q, S]
        hs:
            Requires Q=1, returns [H, S]
        s:
            Requires Q=1, reduces heads, returns [S]
    """

    parsed_valid_len, parsed_query_idx, kwargs = _parse_int_args(args, kwargs)
    if valid_len is None:
        valid_len = parsed_valid_len
    if query_idx is None:
        query_idx = parsed_query_idx

    original_q_shape = tuple(query_states.shape)
    original_k_shape = tuple(key_states.shape)

    inferred_head_dim = query_states.shape[-1]

    q_hqd = _infer_query_hqd(
        query_states,
        head_dim=inferred_head_dim,
        num_heads=num_heads,
    )

    k_hsd = _infer_key_hsd(
        key_states,
        head_dim=inferred_head_dim,
        num_kv_heads=num_kv_heads,
    )

    if valid_len is not None:
        valid_len = int(valid_len)
        if valid_len > 0:
            k_hsd = k_hsd[:, : min(valid_len, k_hsd.shape[1]), :]

    score_hqs, reduced = _align_hq_hk(
        q_hqd=q_hqd,
        k_hsd=k_hsd,
        reduce_single_query_head=True,
        reduce_head=reduce_head,
    )

    # causal / query_idx mask over sequence dimension S
    s_len = score_hqs.shape[-1]
    q_len = score_hqs.shape[1]

    if query_idx is not None:
        qi = int(query_idx)
        pos = torch.arange(s_len, device=score_hqs.device)
        score_hqs = score_hqs.masked_fill(pos.view(1, 1, -1) > qi, float("-inf"))
    elif causal and q_len > 1:
        q_pos = torch.arange(q_len, device=score_hqs.device).view(1, q_len, 1)
        k_pos = torch.arange(s_len, device=score_hqs.device).view(1, 1, s_len)
        score_hqs = score_hqs.masked_fill(k_pos > q_pos, float("-inf"))

    if debug:
        print(
            "[qk_score debug] "
            f"query={original_q_shape}, key={original_k_shape}, "
            f"q_hqd={tuple(q_hqd.shape)}, k_hsd={tuple(k_hsd.shape)}, "
            f"score_hqs={tuple(score_hqs.shape)}, "
            f"return_layout={return_layout}, reduced={reduced}",
            flush=True,
        )

    if return_layout == "hqs":
        return score_hqs

    if return_layout == "hs":
        if score_hqs.shape[1] != 1:
            raise RuntimeError(f"return_layout='hs' requires Q=1, got Q={score_hqs.shape[1]}")
        return score_hqs[:, 0, :]

    if return_layout == "s":
        if score_hqs.shape[1] != 1:
            raise RuntimeError(f"return_layout='s' requires Q=1, got Q={score_hqs.shape[1]}")
        hs = score_hqs[:, 0, :]
        if reduce_head == "mean":
            return hs.mean(dim=0)
        return hs.amax(dim=0)

    if return_layout != "auto":
        raise ValueError(f"Unknown return_layout={return_layout}")

    # Auto behavior:
    # Q=1 and H=1 -> [S]
    # Q=1 and reduced -> [S]
    # Q=1 and original query is [D] or [1,D] -> [S]
    # Q=1 and H>1 -> [H,S]
    # Q>1 -> [H,Q,S]
    h_out, q_out, s_out = score_hqs.shape

    if q_out == 1:
        hs = score_hqs[:, 0, :]  # [H, S]

        if h_out == 1:
            return hs[0]

        if reduced:
            return hs[0]

        # If original query is [D] or [1,D], caller almost certainly expects [S].
        if len(original_q_shape) == 1:
            if reduce_head == "mean":
                return hs.mean(dim=0)
            return hs.amax(dim=0)

        if len(original_q_shape) == 2 and original_q_shape[0] == 1:
            if reduce_head == "mean":
                return hs.mean(dim=0)
            return hs.amax(dim=0)

        # Otherwise preserve heads: [H, S]
        return hs

    return score_hqs


@torch.no_grad()
def qk_topk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    k: int,
    largest: bool = True,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    score = qk_score(query_states, key_states, *args, **kwargs)
    k = min(int(k), score.shape[-1])
    k = max(1, k)
    return torch.topk(score, k=k, dim=-1, largest=largest)


__all__ = ["qk_score", "qk_topk"]
