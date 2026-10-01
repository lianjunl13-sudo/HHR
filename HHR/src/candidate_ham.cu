#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

#include "operator.h"

namespace kvlib {

template <typename index_t>
__global__ void CandidateHammingScoreKernel(
    const int32_t* __restrict__ hash_cache,
    const index_t* __restrict__ candidate_indices,
    const bool* __restrict__ valid_mask,
    const int32_t* __restrict__ query_code,
    half* __restrict__ output, int64_t total, int32_t num_kv_heads,
    int32_t num_candidates, int32_t num_query_heads, int32_t num_chunks,
    int32_t sink, int64_t cache_stride_b, int64_t cache_stride_k,
    int64_t cache_stride_s, int64_t cache_stride_c, int64_t idx_stride_b,
    int64_t idx_stride_k, int64_t idx_stride_n, int64_t valid_stride_b,
    int64_t valid_stride_k, int64_t valid_stride_n, int64_t query_stride_b,
    int64_t query_stride_h, int64_t query_stride_c, int64_t out_stride_b,
    int64_t out_stride_k, int64_t out_stride_n) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) return;

  int32_t candidate_pos = linear % num_candidates;
  int64_t row = linear / num_candidates;
  int32_t kv_head = row % num_kv_heads;
  int32_t batch = row / num_kv_heads;
  int64_t idx_offset = static_cast<int64_t>(batch) * idx_stride_b +
                       static_cast<int64_t>(kv_head) * idx_stride_k +
                       static_cast<int64_t>(candidate_pos) * idx_stride_n;
  int64_t valid_offset = static_cast<int64_t>(batch) * valid_stride_b +
                         static_cast<int64_t>(kv_head) * valid_stride_k +
                         static_cast<int64_t>(candidate_pos) * valid_stride_n;
  int64_t out_offset = static_cast<int64_t>(batch) * out_stride_b +
                       static_cast<int64_t>(kv_head) * out_stride_k +
                       static_cast<int64_t>(candidate_pos) * out_stride_n;
  if (!valid_mask[valid_offset]) {
    output[out_offset] = __float2half(INFINITY);
    return;
  }
  if (candidate_pos < sink) {
    output[out_offset] = __float2half(0.0f);
    return;
  }

  index_t token = candidate_indices[idx_offset];
  int32_t group_size = num_query_heads / num_kv_heads;
  int32_t distance = 0;
  for (int32_t group = 0; group < group_size; ++group) {
    int32_t query_head = kv_head * group_size + group;
    for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
      int64_t cache_offset = static_cast<int64_t>(batch) * cache_stride_b +
                             static_cast<int64_t>(kv_head) * cache_stride_k +
                             static_cast<int64_t>(token) * cache_stride_s +
                             static_cast<int64_t>(chunk) * cache_stride_c;
      int64_t query_offset = static_cast<int64_t>(batch) * query_stride_b +
                             static_cast<int64_t>(query_head) * query_stride_h +
                             static_cast<int64_t>(chunk) * query_stride_c;
      distance += __popc(static_cast<uint32_t>(
          hash_cache[cache_offset] ^ query_code[query_offset]));
    }
  }
  output[out_offset] = __int2half_rn(distance);
}

torch::Tensor CandidateHammingScoreCUDA(
    torch::Tensor& hash_cache, torch::Tensor& candidate_indices,
    torch::Tensor& valid_mask, torch::Tensor& query_code, int32_t sink) {
  TORCH_CHECK(hash_cache.is_cuda() && candidate_indices.is_cuda() &&
                  valid_mask.is_cuda() && query_code.is_cuda(),
              "candidate Hamming inputs must be CUDA tensors");
  TORCH_CHECK(hash_cache.scalar_type() == torch::kInt32,
              "Hash cache must be int32");
  TORCH_CHECK(query_code.scalar_type() == torch::kInt32,
              "query code must be int32");
  TORCH_CHECK(valid_mask.scalar_type() == torch::kBool,
              "valid mask must be bool");
  TORCH_CHECK(hash_cache.dim() == 4 && candidate_indices.dim() == 3 &&
                  valid_mask.sizes() == candidate_indices.sizes(),
              "invalid candidate Hamming shapes");
  if (query_code.dim() == 4) {
    TORCH_CHECK(query_code.size(1) == 1,
                "query code sequence dimension must be one");
  }
  auto query_view = query_code.dim() == 4 ? query_code.select(1, 0) : query_code;
  TORCH_CHECK(query_view.dim() == 3, "query code must have three dimensions");

  int32_t batch = hash_cache.size(0);
  int32_t num_kv_heads = hash_cache.size(1);
  int32_t num_candidates = candidate_indices.size(2);
  int32_t num_query_heads = query_view.size(1);
  int32_t num_chunks = hash_cache.size(3);
  TORCH_CHECK(candidate_indices.size(0) == batch &&
                  candidate_indices.size(1) == num_kv_heads,
              "candidate head shape mismatch");
  TORCH_CHECK(query_view.size(0) == batch && query_view.size(2) == num_chunks &&
                  num_query_heads % num_kv_heads == 0,
              "query code shape mismatch");

  auto output = torch::empty(candidate_indices.sizes(),
                             torch::TensorOptions()
                                 .device(hash_cache.device())
                                 .dtype(torch::kFloat16));
  int64_t total = static_cast<int64_t>(batch) * num_kv_heads * num_candidates;
  constexpr int32_t threads = 256;
  int32_t blocks = static_cast<int32_t>((total + threads - 1) / threads);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(hash_cache.device().index());

  AT_DISPATCH_INDEX_TYPES(candidate_indices.scalar_type(),
                          "CandidateHammingScoreCUDA", [&] {
    CandidateHammingScoreKernel<index_t><<<blocks, threads, 0, stream>>>(
        hash_cache.data_ptr<int32_t>(), candidate_indices.data_ptr<index_t>(),
        valid_mask.data_ptr<bool>(), query_view.data_ptr<int32_t>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()), total,
        num_kv_heads, num_candidates, num_query_heads, num_chunks,
        std::min(sink, num_candidates), hash_cache.stride(0), hash_cache.stride(1),
        hash_cache.stride(2), hash_cache.stride(3), candidate_indices.stride(0),
        candidate_indices.stride(1), candidate_indices.stride(2),
        valid_mask.stride(0), valid_mask.stride(1), valid_mask.stride(2),
        query_view.stride(0), query_view.stride(1), query_view.stride(2),
        output.stride(0), output.stride(1), output.stride(2));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

template <typename index_t>
__global__ void CandidateHammingConfidenceKernel(
    const int32_t* __restrict__ hash_cache,
    const index_t* __restrict__ candidate_indices,
    const bool* __restrict__ valid_mask,
    const int32_t* __restrict__ query_code,
    half* __restrict__ output, float* __restrict__ stats,
    int32_t num_kv_heads, int32_t num_candidates, int32_t num_query_heads,
    int32_t num_chunks, int32_t sink, int32_t probe_limit,
    int32_t fetch_num, int64_t cache_stride_b, int64_t cache_stride_k,
    int64_t cache_stride_s, int64_t cache_stride_c, int64_t idx_stride_b,
    int64_t idx_stride_k, int64_t idx_stride_n, int64_t valid_stride_b,
    int64_t valid_stride_k, int64_t valid_stride_n, int64_t query_stride_b,
    int64_t query_stride_h, int64_t query_stride_c, int64_t out_stride_b,
    int64_t out_stride_k, int64_t out_stride_n) {
  int32_t row = blockIdx.x;
  int32_t kv_head = row % num_kv_heads;
  int32_t batch = row / num_kv_heads;
  int32_t group_size = num_query_heads / num_kv_heads;
  int32_t max_distance = num_chunks * 32 * group_size;
  extern __shared__ int32_t histogram[];
  for (int32_t bucket = threadIdx.x; bucket <= max_distance;
       bucket += blockDim.x) {
    histogram[bucket] = 0;
  }
  __syncthreads();

  for (int32_t candidate_pos = threadIdx.x;
       candidate_pos < num_candidates; candidate_pos += blockDim.x) {
    int64_t idx_offset = static_cast<int64_t>(batch) * idx_stride_b +
                         static_cast<int64_t>(kv_head) * idx_stride_k +
                         static_cast<int64_t>(candidate_pos) * idx_stride_n;
    int64_t valid_offset = static_cast<int64_t>(batch) * valid_stride_b +
                           static_cast<int64_t>(kv_head) * valid_stride_k +
                           static_cast<int64_t>(candidate_pos) * valid_stride_n;
    int64_t out_offset = static_cast<int64_t>(batch) * out_stride_b +
                         static_cast<int64_t>(kv_head) * out_stride_k +
                         static_cast<int64_t>(candidate_pos) * out_stride_n;
    bool valid = valid_mask[valid_offset];
    if (!valid) {
      output[out_offset] = __float2half(INFINITY);
      continue;
    }
    int32_t distance = 0;
    if (candidate_pos >= sink) {
      index_t token = candidate_indices[idx_offset];
      for (int32_t group = 0; group < group_size; ++group) {
        int32_t query_head = kv_head * group_size + group;
        for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
          int64_t cache_offset = static_cast<int64_t>(batch) * cache_stride_b +
                                 static_cast<int64_t>(kv_head) * cache_stride_k +
                                 static_cast<int64_t>(token) * cache_stride_s +
                                 static_cast<int64_t>(chunk) * cache_stride_c;
          int64_t query_offset = static_cast<int64_t>(batch) * query_stride_b +
                                 static_cast<int64_t>(query_head) * query_stride_h +
                                 static_cast<int64_t>(chunk) * query_stride_c;
          distance += __popc(static_cast<uint32_t>(
              hash_cache[cache_offset] ^ query_code[query_offset]));
        }
      }
    }
    output[out_offset] = __int2half_rn(distance);
    if (candidate_pos < probe_limit) atomicAdd(&histogram[distance], 1);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int32_t valid_count = 0;
    for (int32_t bucket = 0; bucket <= max_distance; ++bucket)
      valid_count += histogram[bucket];
    int64_t base = static_cast<int64_t>(row) * 3;
    if (valid_count == 0) {
      stats[base] = INFINITY;
      stats[base + 1] = INFINITY;
      stats[base + 2] = 0.0f;
      return;
    }
    int32_t kth_rank = fetch_num - 1;
    if (kth_rank < 0) kth_rank = 0;
    if (kth_rank >= valid_count) kth_rank = valid_count - 1;
    int32_t guard_step = fetch_num / 4;
    if (guard_step < 2) guard_step = 2;
    int32_t guard_rank = kth_rank + guard_step;
    if (guard_rank >= valid_count) guard_rank = valid_count - 1;
    int32_t cumulative = 0;
    int32_t kth_value = max_distance;
    int32_t guard_value = max_distance;
    for (int32_t bucket = 0; bucket <= max_distance; ++bucket) {
      cumulative += histogram[bucket];
      if (cumulative > kth_rank && kth_value == max_distance)
        kth_value = bucket;
      if (cumulative > guard_rank) {
        guard_value = bucket;
        break;
      }
    }
    stats[base] = static_cast<float>(kth_value);
    stats[base + 1] = static_cast<float>(guard_value);
    stats[base + 2] = valid_count > 0
        ? static_cast<float>(histogram[kth_value]) / valid_count : 0.0f;
  }
}

std::vector<torch::Tensor> CandidateHammingConfidenceCUDA(
    torch::Tensor& hash_cache, torch::Tensor& candidate_indices,
    torch::Tensor& valid_mask, torch::Tensor& query_code, int32_t sink,
    int32_t probe_limit, int32_t fetch_num) {
  TORCH_CHECK(hash_cache.is_cuda() && candidate_indices.is_cuda() &&
                  valid_mask.is_cuda() && query_code.is_cuda(),
              "candidate Hamming inputs must be CUDA tensors");
  TORCH_CHECK(hash_cache.scalar_type() == torch::kInt32 &&
                  query_code.scalar_type() == torch::kInt32 &&
                  valid_mask.scalar_type() == torch::kBool,
              "invalid candidate Hamming dtypes");
  auto query_view = query_code.dim() == 4 ? query_code.select(1, 0) : query_code;
  int32_t batch = hash_cache.size(0);
  int32_t num_kv_heads = hash_cache.size(1);
  int32_t num_candidates = candidate_indices.size(2);
  int32_t num_query_heads = query_view.size(1);
  int32_t num_chunks = hash_cache.size(3);
  TORCH_CHECK(probe_limit > 0 && probe_limit <= num_candidates,
              "invalid probe_limit");
  TORCH_CHECK(fetch_num > 0 && fetch_num <= probe_limit,
              "invalid fetch_num");
  TORCH_CHECK(num_query_heads % num_kv_heads == 0,
              "query/KV head mismatch");
  auto output = torch::empty(candidate_indices.sizes(),
                             hash_cache.options().dtype(torch::kFloat16));
  auto stats = torch::empty({batch, num_kv_heads, 3},
                            hash_cache.options().dtype(torch::kFloat32));
  int32_t group_size = num_query_heads / num_kv_heads;
  int32_t max_distance = num_chunks * 32 * group_size;
  int32_t blocks = batch * num_kv_heads;
  constexpr int32_t threads = 256;
  size_t shared = (max_distance + 1) * sizeof(int32_t);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(hash_cache.device().index());
  AT_DISPATCH_INDEX_TYPES(candidate_indices.scalar_type(),
                          "CandidateHammingConfidenceCUDA", [&] {
    CandidateHammingConfidenceKernel<index_t><<<blocks, threads, shared, stream>>>(
        hash_cache.data_ptr<int32_t>(), candidate_indices.data_ptr<index_t>(),
        valid_mask.data_ptr<bool>(), query_view.data_ptr<int32_t>(),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        stats.data_ptr<float>(), num_kv_heads, num_candidates,
        num_query_heads, num_chunks, std::min(sink, num_candidates),
        probe_limit, fetch_num, hash_cache.stride(0), hash_cache.stride(1),
        hash_cache.stride(2), hash_cache.stride(3), candidate_indices.stride(0),
        candidate_indices.stride(1), candidate_indices.stride(2),
        valid_mask.stride(0), valid_mask.stride(1), valid_mask.stride(2),
        query_view.stride(0), query_view.stride(1), query_view.stride(2),
        output.stride(0), output.stride(1), output.stride(2));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, stats};
}

__global__ void DynamicQuestBudgetKernel(
    const float* __restrict__ confidence_stats,
    const float* __restrict__ gap,
    const float* __restrict__ bound_uncertainty,
    int32_t rows, int32_t seq_len, int32_t page_size,
    int32_t forced_pages, float probe_ratio, float base_ratio,
    float max_ratio, int32_t hash_rbits, float* __restrict__ diagnostics,
    int32_t* __restrict__ final_pages) {
  __shared__ float hash_sum[256];
  __shared__ float margin_sum[256];
  __shared__ float tie_uncertainty_sum[256];
  __shared__ float tie_raw_sum[256];
  __shared__ float gap_sum[256];
  __shared__ float bound_sum[256];
  float local_hash = 0.0f;
  float local_margin = 0.0f;
  float local_tie_uncertainty = 0.0f;
  float local_tie_raw = 0.0f;
  float local_gap = 0.0f;
  float local_bound = 0.0f;
  for (int32_t row = threadIdx.x; row < rows; row += blockDim.x) {
    float kth = confidence_stats[static_cast<int64_t>(row) * 3];
    float guard = confidence_stats[static_cast<int64_t>(row) * 3 + 1];
    float margin = fmaxf(guard - kth, 0.0f) / static_cast<float>(hash_rbits);
    float logit = (0.025f - margin) / 0.0125f;
    local_hash += 1.0f / (1.0f + expf(-logit));
    local_margin += margin;
    float tie = confidence_stats[static_cast<int64_t>(row) * 3 + 2];
    local_tie_raw += tie;
    local_tie_uncertainty += fminf(fmaxf(tie / 0.10f, 0.0f), 1.0f);
    local_gap += gap[row];
    local_bound += bound_uncertainty[row];
  }
  hash_sum[threadIdx.x] = local_hash;
  margin_sum[threadIdx.x] = local_margin;
  tie_uncertainty_sum[threadIdx.x] = local_tie_uncertainty;
  tie_raw_sum[threadIdx.x] = local_tie_raw;
  gap_sum[threadIdx.x] = local_gap;
  bound_sum[threadIdx.x] = local_bound;
  __syncthreads();
  for (int32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      hash_sum[threadIdx.x] += hash_sum[threadIdx.x + offset];
      margin_sum[threadIdx.x] += margin_sum[threadIdx.x + offset];
      tie_uncertainty_sum[threadIdx.x] +=
          tie_uncertainty_sum[threadIdx.x + offset];
      tie_raw_sum[threadIdx.x] += tie_raw_sum[threadIdx.x + offset];
      gap_sum[threadIdx.x] += gap_sum[threadIdx.x + offset];
      bound_sum[threadIdx.x] += bound_sum[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    float inv_rows = 1.0f / static_cast<float>(rows);
    float hash_mean = hash_sum[0] * inv_rows;
    float margin_mean = margin_sum[0] * inv_rows;
    float tie_uncertainty_mean = tie_uncertainty_sum[0] * inv_rows;
    float tie_raw_mean = tie_raw_sum[0] * inv_rows;
    float gap_mean = gap_sum[0] * inv_rows;
    float uncertainty = fminf(fmaxf(
        0.55f * hash_mean + 0.20f * tie_uncertainty_mean
        + 0.25f * bound_sum[0] * inv_rows, 0.0f), 1.0f);
    float ratio = base_ratio + uncertainty * (max_ratio - base_ratio);
    ratio = fminf(fmaxf(ratio, probe_ratio), max_ratio);
    int32_t target_tokens = max(1, static_cast<int32_t>(ceilf(ratio * seq_len)));
    int32_t pages = (target_tokens + page_size - 1) / page_size;
    int32_t active_pages = (seq_len + page_size - 1) / page_size;
    pages = min(active_pages, max(forced_pages, pages));
    diagnostics[0] = ratio;
    diagnostics[1] = margin_mean;
    diagnostics[2] = tie_raw_mean;
    diagnostics[3] = gap_mean;
    diagnostics[4] = uncertainty;
    final_pages[0] = pages;
  }
}

__global__ void MaskDynamicCandidatesKernel(
    half* __restrict__ score, int64_t total, int32_t num_candidates,
    int32_t page_size, const int32_t* __restrict__ final_pages) {
  int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) return;
  int32_t candidate_pos = linear % num_candidates;
  if (candidate_pos >= final_pages[0] * page_size)
    score[linear] = __float2half(INFINITY);
}

std::vector<torch::Tensor> ApplyDynamicQuestBudgetCUDA(
    torch::Tensor& score, torch::Tensor& confidence_stats,
    torch::Tensor& gap, torch::Tensor& bound_uncertainty,
    int32_t seq_len, int32_t page_size, int32_t forced_pages,
    float probe_ratio, float base_ratio, float max_ratio,
    int32_t hash_rbits) {
  TORCH_CHECK(score.is_cuda() && confidence_stats.is_cuda() && gap.is_cuda()
                  && bound_uncertainty.is_cuda(),
              "dynamic QUEST budget inputs must be CUDA tensors");
  TORCH_CHECK(score.scalar_type() == torch::kFloat16
                  && confidence_stats.scalar_type() == torch::kFloat32
                  && gap.scalar_type() == torch::kFloat32
                  && bound_uncertainty.scalar_type() == torch::kFloat32,
              "invalid dynamic QUEST budget dtypes");
  TORCH_CHECK(score.is_contiguous() && confidence_stats.is_contiguous()
                  && gap.is_contiguous() && bound_uncertainty.is_contiguous(),
              "dynamic QUEST budget inputs must be contiguous");
  TORCH_CHECK(score.dim() == 3 && confidence_stats.dim() == 3
                  && confidence_stats.size(2) == 3,
              "invalid dynamic QUEST budget shapes");
  int32_t rows = score.size(0) * score.size(1);
  TORCH_CHECK(confidence_stats.size(0) * confidence_stats.size(1) == rows
                  && gap.numel() == rows && bound_uncertainty.numel() == rows,
              "dynamic QUEST budget row mismatch");
  auto diagnostics = torch::empty({5}, score.options().dtype(torch::kFloat32));
  auto final_pages = torch::empty({1}, score.options().dtype(torch::kInt32));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(score.device().index());
  constexpr int32_t threads = 256;
  DynamicQuestBudgetKernel<<<1, threads, 0, stream>>>(
      confidence_stats.data_ptr<float>(), gap.data_ptr<float>(),
      bound_uncertainty.data_ptr<float>(), rows, seq_len, page_size,
      forced_pages, probe_ratio, base_ratio, max_ratio, hash_rbits,
      diagnostics.data_ptr<float>(), final_pages.data_ptr<int32_t>());
  int64_t total = score.numel();
  int32_t blocks = static_cast<int32_t>((total + threads - 1) / threads);
  MaskDynamicCandidatesKernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<half*>(score.data_ptr<at::Half>()), total,
      score.size(2), page_size, final_pages.data_ptr<int32_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {diagnostics, final_pages};
}

}  // namespace kvlib
