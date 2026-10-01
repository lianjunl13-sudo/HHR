#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

#include "operator.h"

namespace kvlib {

__device__ __forceinline__ bool IsForcedPage(
    int32_t page, int32_t sink_pages, int32_t recent_start,
    int32_t active_pages, bool has_recent) {
  return page < sink_pages
      || (has_recent && page >= recent_start && page < active_pages);
}

__global__ void CopyAndMaskUpperKernel(
    const float* __restrict__ upper, float* __restrict__ ranking,
    int64_t total, int32_t active_pages, int32_t sink_pages,
    int32_t recent_start, bool has_recent) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) return;
  int32_t page = linear % active_pages;
  ranking[linear] = IsForcedPage(
      page, sink_pages, recent_start, active_pages, has_recent)
      ? -INFINITY : upper[linear];
}

__global__ void BuildQuestCandidatesKernel(
    const int64_t* __restrict__ extra_selected,
    int64_t* __restrict__ selected_pages,
    int64_t* __restrict__ candidate_indices,
    bool* __restrict__ valid_mask,
    int64_t total, int32_t target_pages, int32_t extra_pages,
    int32_t page_size, int32_t seq_len, int32_t sink_pages,
    int32_t recent_first, int32_t forced_pages) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) return;
  int32_t candidate_pos = linear % (target_pages * page_size);
  int64_t row = linear / (target_pages * page_size);
  int32_t page_slot = candidate_pos / page_size;
  int32_t offset = candidate_pos % page_size;
  int64_t page;
  if (page_slot < sink_pages) {
    page = page_slot;
  } else if (page_slot < forced_pages) {
    page = recent_first + (page_slot - sink_pages);
  } else {
    int32_t extra_slot = page_slot - forced_pages;
    page = extra_selected[row * extra_pages + extra_slot];
  }
  if (offset == 0)
    selected_pages[row * target_pages + page_slot] = page;
  int64_t token = page * page_size + offset;
  valid_mask[linear] = token < seq_len;
  candidate_indices[linear] = min(token, static_cast<int64_t>(seq_len - 1));
}

std::vector<torch::Tensor> QuestSelectFromUpperCUDA(
    torch::Tensor& upper_bounds, int32_t seq_len, float quest_ratio,
    int32_t page_size, int32_t num_sink, int32_t num_recent) {
  TORCH_CHECK(upper_bounds.is_cuda() && upper_bounds.is_contiguous(),
              "QUEST upper bounds must be a contiguous CUDA tensor");
  TORCH_CHECK(upper_bounds.scalar_type() == torch::kFloat32
                  && upper_bounds.dim() == 3,
              "QUEST upper bounds must be float32 [batch, kv_head, page]");
  TORCH_CHECK(seq_len > 0 && page_size > 0 && quest_ratio > 0.0f,
              "invalid QUEST selection parameters");
  int32_t batch = upper_bounds.size(0);
  int32_t heads = upper_bounds.size(1);
  int32_t active_pages = upper_bounds.size(2);
  TORCH_CHECK(active_pages == (seq_len + page_size - 1) / page_size,
              "QUEST active page count mismatch");
  int32_t sink_pages =
      (std::min(num_sink, seq_len) + page_size - 1) / page_size;
  bool has_recent = num_recent > 0;
  int32_t recent_start = has_recent
      ? std::max(0, (seq_len - num_recent) / page_size) : active_pages;
  int32_t recent_first = std::max(sink_pages, recent_start);
  int32_t recent_count = has_recent
      ? std::max(0, active_pages - recent_first) : 0;
  int32_t forced_pages = sink_pages + recent_count;
  int32_t target_tokens =
      std::max(1, static_cast<int32_t>(ceilf(seq_len * quest_ratio)));
  int32_t target_pages = std::min(
      active_pages, (target_tokens + page_size - 1) / page_size);
  int32_t extra_pages = std::max(0, target_pages - forced_pages);
  int32_t selected_count = forced_pages + extra_pages;

  auto ranking = torch::empty_like(upper_bounds);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
      upper_bounds.device().index());
  constexpr int32_t threads = 256;
  int64_t upper_total = upper_bounds.numel();
  int32_t upper_blocks = static_cast<int32_t>(
      (upper_total + threads - 1) / threads);
  CopyAndMaskUpperKernel<<<upper_blocks, threads, 0, stream>>>(
      upper_bounds.data_ptr<float>(), ranking.data_ptr<float>(), upper_total,
      active_pages, sink_pages, recent_start, has_recent);

  torch::Tensor extra_selected;
  if (extra_pages > 0) {
    extra_selected = std::get<1>(
        at::topk(ranking, extra_pages, -1, true, true));
  } else {
    extra_selected = torch::empty(
        {batch, heads, 0}, upper_bounds.options().dtype(torch::kInt64));
  }
  auto index_options = upper_bounds.options().dtype(torch::kInt64);
  auto selected_pages = torch::empty(
      {batch, heads, selected_count}, index_options);
  auto candidate_indices = torch::empty(
      {batch, heads, selected_count * page_size}, index_options);
  auto valid_mask = torch::empty(
      {batch, heads, selected_count * page_size},
      upper_bounds.options().dtype(torch::kBool));
  int64_t candidate_total = candidate_indices.numel();
  int32_t candidate_blocks = static_cast<int32_t>(
      (candidate_total + threads - 1) / threads);
  BuildQuestCandidatesKernel<<<candidate_blocks, threads, 0, stream>>>(
      extra_selected.data_ptr<int64_t>(), selected_pages.data_ptr<int64_t>(),
      candidate_indices.data_ptr<int64_t>(), valid_mask.data_ptr<bool>(),
      candidate_total, selected_count, extra_pages, page_size, seq_len,
      sink_pages, recent_first, forced_pages);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {candidate_indices, valid_mask, upper_bounds, ranking,
          selected_pages};
}

}  // namespace kvlib
