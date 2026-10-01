#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/script.h>
#include <vector>

#include "cuda-attn/flash_api.h"
#include "operator.h"

namespace py = pybind11;

torch::Tensor create_tensor(std::vector<int32_t> size, int dtype) {
  // Compute the total number of tensor elements.
  int64_t num_elements = 1;
  for (auto dim : size) {
    num_elements *= dim;
  }

  if (dtype == 16) {
    // void* buf = aligned_alloc(64, num_elements * sizeof(half));
    void* buf;
    cudaMallocHost(&buf, num_elements * sizeof(half));

    auto tensor = torch::from_blob(buf, {num_elements}, torch::kFloat16);

    // Return a tensor that owns its storage.
    return tensor;
  } else {
    // void* buf = aligned_alloc(64, num_elements * sizeof(float));
    void* buf;
    cudaMallocHost(&buf, num_elements * sizeof(float));

    auto tensor = torch::from_blob(buf, {num_elements}, torch::kFloat32);

    // Return a tensor that owns its storage.
    return tensor;
  }
}

PYBIND11_MODULE(KVLib, m) {
  m.def("hamming_score", &kvlib::HammingScoreCUDA, py::arg("key_code"),
        py::arg("query_code"), py::arg("rbit"), py::arg("seq_len"),
        py::arg("sink") = 0, py::arg("recent") = 0)
      .def("candidate_hamming_score", &kvlib::CandidateHammingScoreCUDA,
           py::arg("hash_cache"), py::arg("candidate_indices"),
           py::arg("valid_mask"), py::arg("query_code"),
           py::arg("sink") = 0)
      .def("candidate_hamming_confidence",
           &kvlib::CandidateHammingConfidenceCUDA,
           py::arg("hash_cache"), py::arg("candidate_indices"),
           py::arg("valid_mask"), py::arg("query_code"),
           py::arg("sink"), py::arg("probe_limit"),
           py::arg("fetch_num"))
      .def("apply_dynamic_quest_budget",
           &kvlib::ApplyDynamicQuestBudgetCUDA,
           py::arg("score"), py::arg("confidence_stats"),
           py::arg("gap"), py::arg("bound_uncertainty"),
           py::arg("seq_len"), py::arg("page_size"),
           py::arg("forced_pages"), py::arg("probe_ratio"),
           py::arg("base_ratio"), py::arg("max_ratio"),
           py::arg("hash_rbits"))
      .def("quest_select_from_upper",
           &kvlib::QuestSelectFromUpperCUDA,
           py::arg("upper_bounds"), py::arg("seq_len"),
           py::arg("quest_ratio"), py::arg("page_size"),
           py::arg("num_sink"), py::arg("num_recent"))
      .def("batch_topk", &kvlib::TopkCUDA)
      .def("decode_multi_hash_encode", &kvlib::decode_multi_hash_encode)
      .def("flash_index_decode", &kvlib::mha_index_decode_fwd)
      .def("flash_decode", &kvlib::mha_decode_fwd)
      .def("kvcache_append", &kvlib::KVCacheAppend)
      .def("kvcache_append2", &kvlib::KVCacheAppend2)
      .def("create_tensor", &create_tensor);
}
