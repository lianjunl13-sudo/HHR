import torch
import triton
import triton.language as tl


@triton.jit
def _quest_page_upper_kernel(
    page_mins,
    page_maxs,
    query,
    output,
    stride_min_b: tl.constexpr,
    stride_min_p: tl.constexpr,
    stride_min_k: tl.constexpr,
    stride_min_d: tl.constexpr,
    stride_q_b: tl.constexpr,
    stride_q_h: tl.constexpr,
    stride_q_d: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_k: tl.constexpr,
    stride_o_p: tl.constexpr,
    num_pages: tl.constexpr,
    num_kv_heads: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    page_idx = pid % num_pages
    kv_head = (pid // num_pages) % num_kv_heads
    batch_idx = pid // (num_pages * num_kv_heads)

    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < head_dim
    page_offset = (
        batch_idx * stride_min_b
        + page_idx * stride_min_p
        + kv_head * stride_min_k
        + offsets * stride_min_d
    )
    mins = tl.load(page_mins + page_offset, mask=dim_mask, other=0.0).to(tl.float32)
    maxs = tl.load(page_maxs + page_offset, mask=dim_mask, other=0.0).to(tl.float32)

    best = -float("inf")
    for group_idx in tl.static_range(0, group_size):
        query_head = kv_head * group_size + group_idx
        query_offset = (
            batch_idx * stride_q_b
            + query_head * stride_q_h
            + offsets * stride_q_d
        )
        q = tl.load(query + query_offset, mask=dim_mask, other=0.0).to(tl.float32)
        page_extreme = tl.where(q >= 0, maxs, mins)
        score = tl.sum(q * page_extreme, axis=0)
        best = tl.maximum(best, score)

    output_offset = (
        batch_idx * stride_o_b
        + kv_head * stride_o_k
        + page_idx * stride_o_p
    )
    tl.store(output + output_offset, best)


def quest_page_upper_bounds(
    page_mins: torch.Tensor,
    page_maxs: torch.Tensor,
    query: torch.Tensor,
) -> torch.Tensor:
    """Fused QUEST min/max upper bounds, reduced over GQA query heads."""
    if not (page_mins.is_cuda and page_maxs.is_cuda and query.is_cuda):
        raise ValueError("fused QUEST upper bounds require CUDA tensors")
    if page_mins.shape != page_maxs.shape or page_mins.ndim != 4:
        raise ValueError("page min/max must share [batch, page, kv_head, dim]")
    batch_size, num_pages, num_kv_heads, head_dim = page_mins.shape
    if query.ndim != 3 or query.shape[0] != batch_size or query.shape[2] != head_dim:
        raise ValueError("query must be [batch, query_head, dim]")
    if query.shape[1] % num_kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    group_size = query.shape[1] // num_kv_heads
    output = torch.empty(
        (batch_size, num_kv_heads, num_pages),
        device=query.device,
        dtype=torch.float32,
    )
    block_d = triton.next_power_of_2(head_dim)
    grid = (batch_size * num_kv_heads * num_pages,)
    _quest_page_upper_kernel[grid](
        page_mins,
        page_maxs,
        query,
        output,
        page_mins.stride(0),
        page_mins.stride(1),
        page_mins.stride(2),
        page_mins.stride(3),
        query.stride(0),
        query.stride(1),
        query.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        num_pages,
        num_kv_heads,
        group_size,
        head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )
    return output
