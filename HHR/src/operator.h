#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>

namespace kvlib {

torch::Tensor HammingScoreCUDA(torch::Tensor& key_codes,
                               torch::Tensor& query_code, int32_t rbit,
                               int32_t seq_len, int32_t sink = 0,
                               int32_t recent = 0);
torch::Tensor CandidateHammingScoreCUDA(
    torch::Tensor& hash_cache, torch::Tensor& candidate_indices,
    torch::Tensor& valid_mask, torch::Tensor& query_code, int32_t sink = 0);
std::vector<torch::Tensor> CandidateHammingConfidenceCUDA(
    torch::Tensor& hash_cache, torch::Tensor& candidate_indices,
    torch::Tensor& valid_mask, torch::Tensor& query_code, int32_t sink,
    int32_t probe_limit, int32_t fetch_num);
std::vector<torch::Tensor> ApplyDynamicQuestBudgetCUDA(
    torch::Tensor& score, torch::Tensor& confidence_stats,
    torch::Tensor& gap, torch::Tensor& bound_uncertainty,
    int32_t seq_len, int32_t page_size, int32_t forced_pages,
    float probe_ratio, float base_ratio, float max_ratio,
    int32_t hash_rbits);
std::vector<torch::Tensor> QuestSelectFromUpperCUDA(
    torch::Tensor& upper_bounds, int32_t seq_len, float quest_ratio,
    int32_t page_size, int32_t num_sink, int32_t num_recent);
torch::Tensor TopkCUDA(torch::Tensor& data, int32_t k, bool largest);
void decode_multi_hash_encode(
    torch::Tensor key_data, torch::Tensor hash_weights,
    torch::Tensor key_code_output, torch::Tensor key_norm_output,
    torch::Tensor query_data, torch::Tensor query_code_output,
    torch::Tensor packbit_aux_tensor, int32_t cur_seq);
void KVCacheAppend(torch::Tensor kv_cache_tensor, torch::Tensor key_tensor,
                   torch::Tensor value_tensor, int32_t insert_pos);
void KVCacheAppend2(torch::Tensor dst_kv_cache_tensor,
                    torch::Tensor src_kv_cache_tensor, int32_t dst_pos,
                    int32_t src_pos);

}  // namespace kvlib
