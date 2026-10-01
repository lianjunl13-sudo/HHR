import math

import torch

try:
    import KVLib
except (ImportError, ModuleNotFoundError):
    KVLib = None

try:
    from .triton_quest import quest_page_upper_bounds
except (ImportError, ModuleNotFoundError):
    quest_page_upper_bounds = None


def quest_page_select(
    page_mins: torch.Tensor,
    page_maxs: torch.Tensor,
    query: torch.Tensor,
    seq_len: int,
    quest_ratio: float = 0.2,
    page_size: int = 16,
    num_sink: int = 16,
    num_recent: int = 0,
    return_metadata: bool = False,
):
    """Select query-dependent KV pages using QUEST attention upper bounds.

    ``quest_ratio`` may be a scalar or one fixed ratio per KV head.  The
    latter is deliberately *static* for an inference run: heads can have
    different offline-calibrated budgets, but Hash confidence never changes
    the Page budget online.
    """
    batch_size, num_pages, num_kv_heads, head_dim = page_mins.shape
    num_heads = query.shape[1]
    if num_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
        )

    active_pages = math.ceil(seq_len / page_size)
    group_size = num_heads // num_kv_heads
    active_mins = page_mins[:, :active_pages]
    active_maxs = page_maxs[:, :active_pages]
    if quest_page_upper_bounds is not None and query.is_cuda:
        upper_bounds = quest_page_upper_bounds(active_mins, active_maxs, query)
    else:
        mins = active_mins.float().permute(0, 2, 1, 3)
        maxs = active_maxs.float().permute(0, 2, 1, 3)
        grouped_query = query.float().view(
            batch_size, num_kv_heads, group_size, head_dim
        )
        upper_bounds = torch.einsum(
            "bkgd,bkpd->bkgp", grouped_query.clamp_min(0), maxs
        ) + torch.einsum(
            "bkgd,bkpd->bkgp", grouped_query.clamp_max(0), mins
        )
        upper_bounds = upper_bounds.amax(dim=2)
    forced_pages = []
    sink_pages = math.ceil(min(num_sink, seq_len) / page_size)
    if sink_pages:
        forced_pages.extend(range(sink_pages))
    if num_recent:
        recent_start = max(0, (seq_len - num_recent) // page_size)
        forced_pages.extend(range(recent_start, active_pages))
    forced_pages = sorted(set(forced_pages))

    ratio_tensor = torch.as_tensor(
        quest_ratio, device=query.device, dtype=torch.float32
    )
    headwise = ratio_tensor.ndim != 0
    # The experiment protocol treats sink/recent tokens as an *additional*
    # budget.  QUEST therefore allocates ``quest_ratio`` over the searchable
    # non-forced region and then prepends the forced pages.  The legacy code
    # allocated the ratio over the whole sequence and subtracted forced pages,
    # which silently charged sink tokens against the candidate pool.
    forced_token_count = min(num_sink, seq_len)
    forced_token_count += min(num_recent, max(0, seq_len - forced_token_count))
    searchable_tokens = max(0, seq_len - forced_token_count)
    available_extra_pages = max(0, active_pages - len(forced_pages))

    if headwise:
        ratio_tensor = ratio_tensor.flatten()
        if ratio_tensor.numel() != num_kv_heads:
            raise ValueError(
                f"head-wise QUEST requires {num_kv_heads} KV-head ratios, "
                f"got {ratio_tensor.numel()}"
            )
        if not torch.all((ratio_tensor > 0) & (ratio_tensor <= 1)):
            raise ValueError("head-wise QUEST ratios must be in (0, 1]")
        target_tokens = torch.ceil(ratio_tensor * searchable_tokens).clamp_min(1)
        extra_pages = torch.ceil(target_tokens / page_size).long()
        extra_pages = extra_pages.clamp_max(available_extra_pages)
        target_pages = extra_pages + len(forced_pages)
        max_extra_pages = int(extra_pages.max().item())
    else:
        scalar_ratio = float(ratio_tensor.item())
        target_tokens = max(1, math.ceil(searchable_tokens * scalar_ratio))
        extra_pages = min(
            available_extra_pages, math.ceil(target_tokens / page_size)
        )
        target_pages = min(active_pages, len(forced_pages) + extra_pages)
        max_extra_pages = extra_pages

    if (
        not headwise
        and query.is_cuda
        and KVLib is not None
        and hasattr(KVLib, "quest_select_from_upper")
    ):
        (
            candidate_indices,
            valid_mask,
            raw_upper_bounds,
            ranking_upper_bounds,
            selected_pages,
        ) = KVLib.quest_select_from_upper(
            upper_bounds,
            seq_len,
            float(ratio_tensor.item()),
            page_size,
            num_sink,
            num_recent,
        )
        if return_metadata:
            return candidate_indices, valid_mask, {
                "raw_upper_bounds": raw_upper_bounds,
                "ranking_upper_bounds": ranking_upper_bounds,
                "selected_pages": selected_pages,
                "target_pages": target_pages,
                "forced_pages": len(forced_pages),
            }
        return candidate_indices, valid_mask

    raw_upper_bounds = upper_bounds.clone() if return_metadata else None

    if forced_pages:
        forced_page_tensor = torch.tensor(
            forced_pages, device=query.device, dtype=torch.long
        )
        upper_bounds[:, :, forced_page_tensor] = -torch.inf
    else:
        forced_page_tensor = torch.empty(0, device=query.device, dtype=torch.long)

    if max_extra_pages:
        selected_pages = torch.topk(
            upper_bounds, max_extra_pages, dim=-1, largest=True
        ).indices
    else:
        selected_pages = torch.empty(
            batch_size,
            num_kv_heads,
            0,
            device=query.device,
            dtype=torch.long,
        )

    if forced_page_tensor.numel():
        forced = forced_page_tensor.view(1, 1, -1).expand(
            batch_size, num_kv_heads, -1
        )
        selected_pages = torch.cat((forced, selected_pages), dim=-1)

    offsets = torch.arange(page_size, device=query.device, dtype=torch.long)
    candidate_indices = (
        selected_pages.unsqueeze(-1) * page_size + offsets.view(1, 1, 1, -1)
    ).flatten(-2)
    valid_mask = candidate_indices < seq_len
    if headwise and max_extra_pages:
        page_rank = torch.arange(
            selected_pages.shape[-1], device=query.device
        ).view(1, 1, -1)
        if forced_page_tensor.numel():
            selected_valid = torch.ones_like(
                selected_pages, dtype=torch.bool
            )
            selected_valid[..., forced_page_tensor.numel():] = (
                page_rank[..., forced_page_tensor.numel():]
                - forced_page_tensor.numel()
                < extra_pages.view(1, -1, 1)
            )
        else:
            selected_valid = page_rank < extra_pages.view(1, -1, 1)
        valid_mask &= selected_valid.unsqueeze(-1).expand(
            -1, -1, -1, page_size
        ).flatten(-2)
    candidate_indices = candidate_indices.clamp_max(seq_len - 1)
    if return_metadata:
        return candidate_indices, valid_mask, {
            "raw_upper_bounds": raw_upper_bounds,
            "ranking_upper_bounds": upper_bounds,
            "selected_pages": selected_pages,
            "target_pages": target_pages,
            "forced_pages": len(forced_pages),
        }
    return candidate_indices, valid_mask
