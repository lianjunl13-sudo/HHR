from typing import Dict, Optional, Union, Any
import json
import math
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
from ..kernels.hadamard_utils import hadamard_transform
from ..kernels.quest_utils import quest_page_select
from ..kernels.triton_hash_encode import prefill_multi_hash_encode, decode_multi_hash_encode
from .kvcache_fa import CustomStaticCache
import KVLib
import os


def _select_hash_coordinates(raw_tensor, search_tensor, branch):
    """Keep QUEST search coordinates independent from Hash coordinates."""
    if branch == "parallel":
        return raw_tensor
    if branch == "serial":
        return search_tensor
    raise ValueError(f"unsupported QUEST/Hash branch: {branch}")


class HashStaticCache(CustomStaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        hash_rbits: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_gpu_cache_memory_size: int = 1000000000,  # 0.93 GB
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
        sparse_ratio: float = 0.1,
        quest_ratio: float = 0.2,
        quest_page_size: int = 16,
        quest_dynamic: bool = False,
        quest_probe_ratio: float = 0.04,
        quest_base_ratio: float = 0.05,
        quest_max_ratio: float = 0.08,
        quest_rotation_path: str = None,
        quest_hash_branch: str = "serial",
        use_hadamard: bool = False,
        num_skip_layers: int = 2,
        hash_weights_path: str = None,
        num_sink: int = 0,
        num_recent: int = 0,
        _hash_max_batch_size: int = 16,
    ) -> None:
        super().__init__(config, device, dtype, max_gpu_cache_memory_size,
                         layer_device_map)
        self.hash_rbits = hash_rbits
        self.sparse_ratio = sparse_ratio
        self.quest_ratio = quest_ratio
        self.quest_page_size = quest_page_size
        self.quest_dynamic = quest_dynamic or os.getenv(
            "QH_QUEST_DYNAMIC", "0"
        ) == "1"
        self.quest_probe_ratio = float(os.getenv(
            "QH_QUEST_PROBE_RATIO", str(quest_probe_ratio)
        ))
        self.quest_base_ratio = float(os.getenv(
            "QH_QUEST_BASE_RATIO", str(quest_base_ratio)
        ))
        self.quest_max_ratio = float(os.getenv(
            "QH_QUEST_MAX_RATIO", str(quest_max_ratio)
        ))
        self.quest_low_budget_layers = {
            int(value)
            for value in os.getenv("QH_QUEST_LOW_BUDGET_LAYERS", "").split(",")
            if value.strip()
        }
        self.quest_low_probe_ratio = float(os.getenv(
            "QH_QUEST_LOW_PROBE_RATIO", "0.03"
        ))
        self.quest_low_base_ratio = float(os.getenv(
            "QH_QUEST_LOW_BASE_RATIO", "0.04"
        ))
        self.quest_low_max_ratio = float(os.getenv(
            "QH_QUEST_LOW_MAX_RATIO", "0.06"
        ))
        self.quest_layer_budgets = {}
        for spec in os.getenv("QH_QUEST_LAYER_BUDGETS", "").split(";"):
            if not spec.strip():
                continue
            layer, probe, base, maximum = spec.split(":")
            self.quest_layer_budgets[int(layer)] = (
                float(probe), float(base), float(maximum)
            )
        self.quest_head_budget_path = os.getenv(
            "QH_QUEST_HEAD_BUDGET_PATH", ""
        ).strip() or None
        self.quest_head_budget_default = None
        self.quest_head_budgets = {}
        if self.quest_head_budget_path is not None:
            with open(self.quest_head_budget_path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("head_axis", "kv_head") != "kv_head":
                raise ValueError("QUEST static budget head_axis must be kv_head")
            self.quest_head_budget_default = payload.get("default")
            self.quest_head_budgets = {
                int(layer): ratios
                for layer, ratios in payload.get("layers", {}).items()
            }
        self.quest_bypass_layers = {
            int(value)
            for value in os.getenv("QH_QUEST_BYPASS_LAYERS", "").split(",")
            if value.strip()
        }
        self.route_fallback_weights_path = os.getenv(
            "QH_FALLBACK_HASH_WEIGHTS_PATH", ""
        ).strip() or None
        route_limit = os.getenv("HHR_ROUTE_MAX_SEQ_LEN", "").strip()
        self.route_max_seq_len = int(route_limit) if route_limit else None
        if (self.route_fallback_weights_path is None) != (
            self.route_max_seq_len is None
        ):
            raise ValueError(
                "QH_FALLBACK_HASH_WEIGHTS_PATH and HHR_ROUTE_MAX_SEQ_LEN "
                "must be configured together"
            )
        if self.route_max_seq_len is not None and self.route_max_seq_len < 1:
            raise ValueError("HHR_ROUTE_MAX_SEQ_LEN must be positive")
        self.route_fallback_active = False
        self.route_decided = False
        self.quest_rotation_path = (
            quest_rotation_path or os.getenv("QH_SHARED_ROTATION_PATH")
        )
        self.quest_hash_branch = os.getenv(
            "QH_QUEST_HASH_BRANCH", quest_hash_branch
        ).strip().lower()
        if self.quest_hash_branch not in {"serial", "parallel"}:
            raise ValueError("QH_QUEST_HASH_BRANCH must be serial or parallel")
        self.quest_paired_nonorth = os.getenv(
            "HHR_PAIRED_NONORTH", "0"
        ).strip().lower() in {"1", "true", "yes"}
        if self.quest_paired_nonorth and self.quest_rotation_path is None:
            raise ValueError(
                "HHR_PAIRED_NONORTH requires QH_SHARED_ROTATION_PATH"
            )
        self.quest_parallel_hash_calls = 0
        self.quest_serial_hash_calls = 0
        self.hhr_audit_path = os.getenv("HHR_AUDIT_PATH", "").strip()
        self._hhr_prefill_audited = set()
        self._hhr_decode_audited = set()
        self._extra_sink_selection_audited = False
        amplitude_clip = os.getenv("QH_HASH_AMPLITUDE_CLIP", "").strip()
        self.hash_amplitude_clip = (
            float(amplitude_clip) if amplitude_clip else None
        )
        self.quest_tie_break_strength = float(os.getenv(
            "QH_QUEST_TIE_BREAK_STRENGTH", "0"
        ))
        self.quest_tie_break_mode = os.getenv(
            "QH_QUEST_TIE_BREAK_MODE", "quest"
        ).strip().lower()
        if self.hash_amplitude_clip is not None and self.hash_amplitude_clip <= 0:
            raise ValueError("QH_HASH_AMPLITUDE_CLIP must be positive")
        if not 0.0 <= self.quest_tie_break_strength < 1.0:
            raise ValueError("QH_QUEST_TIE_BREAK_STRENGTH must be in [0, 1)")
        if self.quest_tie_break_mode not in {
            "quest", "reverse", "token_hash"
        }:
            raise ValueError("invalid QH_QUEST_TIE_BREAK_MODE")
        if self.quest_dynamic and not (
            sparse_ratio <= self.quest_probe_ratio
            <= self.quest_base_ratio <= self.quest_max_ratio <= 1.0
        ):
            raise ValueError(
                "dynamic QUEST requires sparse <= probe <= base <= max <= 1"
            )
        if self.quest_low_budget_layers and not (
            sparse_ratio <= self.quest_low_probe_ratio
            <= self.quest_low_base_ratio <= self.quest_low_max_ratio
            <= self.quest_max_ratio
        ):
            raise ValueError(
                "low layer budget requires sparse <= low probe <= low base "
                "<= low max <= global max"
            )
        for layer, ratios in self.quest_layer_budgets.items():
            probe, base, maximum = ratios
            if not (
                sparse_ratio <= probe <= base <= maximum <= 1.0
            ):
                raise ValueError(
                    f"invalid QUEST layer budget for layer {layer}: {ratios}"
                )
        if self.quest_dynamic and self.quest_head_budget_path is not None:
            raise ValueError(
                "online dynamic QUEST and static head-wise budgets are mutually exclusive"
            )
        for label, ratios in ([
            ("default", self.quest_head_budget_default)
        ] + list(self.quest_head_budgets.items())):
            if ratios is None:
                continue
            if len(ratios) != self.num_key_value_heads:
                raise ValueError(
                    f"QUEST budget {label} requires {self.num_key_value_heads} "
                    f"KV-head ratios, got {len(ratios)}"
                )
            if any(
                float(ratio) < sparse_ratio or float(ratio) > 1.0
                for ratio in ratios
            ):
                raise ValueError(
                    f"QUEST budget {label} ratios must satisfy sparse <= ratio <= 1"
                )
        if (
            self.quest_head_budget_path is not None
            and self.quest_head_budget_default is None
            and len(self.quest_head_budgets) < self.num_layers
        ):
            raise ValueError(
                "static head-wise QUEST needs a default or all layer budgets"
            )
        self.quest_enabled = self.quest_dynamic or self.quest_head_budget_path is not None or (
            quest_ratio > sparse_ratio and quest_ratio < 1.0
        )
        self.use_hadamard = use_hadamard
        self.hash_weights_path = hash_weights_path
        self.num_skip_layers = num_skip_layers

        self.num_sink = num_sink
        self.num_recent = num_recent
        self._hash_max_batch_size = _hash_max_batch_size
        self.max_gpu_cache_memory_size -= 2 * self.num_layers * self._hash_max_batch_size * (
            self.num_sink + self.num_recent
        ) * self.num_key_value_heads * self.head_dim * self.dtype.itemsize

        self.gqa_size = self.num_heads // self.num_key_value_heads
        self.hash_packbit_aux_tensors = {}

        self.quest_profile_enabled = os.getenv("QUEST_PROFILE", "0") == "1"
        profile_layers = os.getenv("QUEST_PROFILE_LAYERS", "0,8,16,24,31")
        self.quest_profile_layers = {
            int(layer.strip()) for layer in profile_layers.split(",") if layer.strip()
        }
        self.quest_profile_samples = int(os.getenv("QUEST_PROFILE_SAMPLES", "10"))
        self.quest_profile_decode_tokens = int(
            os.getenv("QUEST_PROFILE_DECODE_TOKENS", "8")
        )
        self.quest_profile_skip_resets = int(
            os.getenv("QUEST_PROFILE_SKIP_RESETS", "1")
        )
        self.quest_profile_output = os.getenv(
            "QUEST_PROFILE_OUTPUT", "/tmp/quest_profile.json"
        )
        self._quest_profile_reset_count = 0
        self._quest_profile_active = False
        self._quest_profile_sample = -1
        self._quest_profile_layer_steps = {}
        self._quest_profile_records = []
        self._quest_profile_written = False

    def build_cache(self):

        self.layer_caches = []
        self.max_layer_caches = []
        self.quest_page_mins = []
        self.quest_page_maxs = []

        self.layer_hash_caches = []
        self.max_layer_hash_caches = []

        self.hash_weights = []
        self.fallback_hash_weights = []
        self.quest_rotations = []

        assert self.hash_rbits % 32 == 0
        self.hash_dim = self.hash_rbits // 32

        per_token_per_head_kv_size = self.dtype.itemsize * self.head_dim * 2
        per_token_per_head_hash_size = torch.int32.itemsize * \
            self.hash_dim + self.dtype.itemsize

        all_layer_per_token_per_head_kv_size = per_token_per_head_kv_size * self.num_layers
        all_layer_per_token_per_head_hash_size = per_token_per_head_hash_size * (
            self.num_layers - self.num_skip_layers)
        all_layer_per_token_per_head_size = all_layer_per_token_per_head_kv_size + \
            all_layer_per_token_per_head_hash_size

        self.max_kv_cache_size = self.max_gpu_cache_memory_size * \
            all_layer_per_token_per_head_kv_size / all_layer_per_token_per_head_size
        self.max_hash_cache_size = self.max_gpu_cache_memory_size * \
            all_layer_per_token_per_head_hash_size / all_layer_per_token_per_head_size

        self.each_layer_max_kv_cache = self.max_kv_cache_size / self.num_layers
        self.each_layer_max_hash_cache = self.max_hash_cache_size / (
            self.num_layers - self.num_skip_layers)
        
        kv_numel = int(self.each_layer_max_kv_cache / self.dtype.itemsize)
        self.each_layer_max_kv_numel = kv_numel
        hash_numel = int(self.each_layer_max_hash_cache / torch.int32.itemsize)
        self.each_layer_max_hash_numel = hash_numel

        for l in range(self.num_layers):
            layer_device = self.layer_devices[l]
            load_device = (
                torch.device(f"cuda:{layer_device}")
                if isinstance(layer_device, int)
                else torch.device(layer_device)
            )
            self.layer_devices.append(layer_device)

            self.layer_caches.append(None)
            self.layer_hash_caches.append(None)
            self.quest_page_mins.append(None)
            self.quest_page_maxs.append(None)
            self.quest_rotations.append(None)

            self.max_layer_caches.append(
                torch.zeros((kv_numel, ),
                            dtype=self.dtype,
                            device=layer_device))

            if l >= self.num_skip_layers:
                self.max_layer_hash_caches.append(
                    torch.zeros((hash_numel, ),
                                dtype=torch.int32,
                                device=layer_device))
                if self.hash_weights_path is None:
                    hash_weight = None
                else:
                    # Load a trained weight file when one is available.
                    weight_file = os.path.join(self.hash_weights_path,
                                               f"hash_weight_layer_{l:02d}.pt")
                    try:
                        hash_weight = torch.load(
                            weight_file, map_location="cpu", weights_only=True
                        ).to(device=load_device, dtype=self.dtype)
                    except FileNotFoundError:
                        # Random initialization is reserved for explicit ablations.
                        print(f"Warning: {weight_file} not found, skipping hash for layer (missing weights) {l}")
                        hash_weight = torch.randn((self.num_key_value_heads,
                                                   self.head_dim, self.hash_rbits),
                                                  dtype=self.dtype,
                                                  device=layer_device)
                self.hash_weights.append(hash_weight)
                if self.route_fallback_weights_path is not None:
                    fallback_file = os.path.join(
                        self.route_fallback_weights_path,
                        f"hash_weight_layer_{l:02d}.pt",
                    )
                    fallback_weight = torch.load(
                        fallback_file, map_location="cpu", weights_only=True
                    ).to(device=load_device, dtype=self.dtype)
                else:
                    fallback_weight = None
                self.fallback_hash_weights.append(fallback_weight)
                if self.quest_rotation_path is not None:
                    rotation_file = os.path.join(
                        self.quest_rotation_path,
                        f"shared_rotation_layer_{l:02d}.pt",
                    )
                    try:
                        rotation = torch.load(
                            rotation_file, map_location="cpu", weights_only=True
                        ).to(dtype=self.dtype, device=load_device)
                        valid_shapes = {
                            (self.head_dim, self.head_dim),
                            (
                                self.num_key_value_heads,
                                self.head_dim,
                                self.head_dim,
                            ),
                        }
                        if tuple(rotation.shape) not in valid_shapes:
                            raise ValueError(
                                f"invalid rotation shape {rotation.shape} in {rotation_file}"
                            )
                        self.quest_rotations[l] = rotation
                    except FileNotFoundError:
                        print(
                            f"Warning: {rotation_file} not found; layer {l} uses identity search coordinates"
                        )
            else:
                self.max_layer_hash_caches.append(None)
                self.hash_weights.append(None)
                self.fallback_hash_weights.append(None)

        self.max_seq_len = 0
        self.curr_batch_size = 0
        self.seq_len = 0
        self.layer_cache_lens = [0 for _ in range(self.num_layers)]

        for device in self.unique_devices:
            self.hash_packbit_aux_tensors[device] = torch.pow(
                2, torch.arange(0, 32, 1, dtype=torch.int32, device=device))

        self.query_code_buffers = None

    def reset(self, batch_size):
        self.route_fallback_active = False
        self.route_decided = False
        if self.quest_profile_enabled:
            profile_end = (
                self.quest_profile_skip_resets + self.quest_profile_samples
            )
            if self._quest_profile_reset_count >= profile_end:
                self._write_quest_profile()
                raise SystemExit(0)

            self._quest_profile_reset_count += 1
            self._quest_profile_active = (
                self._quest_profile_reset_count > self.quest_profile_skip_resets
            )
            self._quest_profile_sample = (
                self._quest_profile_reset_count
                - self.quest_profile_skip_resets
                - 1
            )
            self._quest_profile_layer_steps = {}

        self.curr_batch_size = batch_size
        self.seq_len = 0
        self.layer_cache_lens = [0 for _ in range(self.num_layers)]

        # shape for KVCache: (2, batch_size, max_seq_len, num_heads * head_dim)
        kv_max_seq_len = self.each_layer_max_kv_numel // (
            self.num_key_value_heads * self.head_dim * self.curr_batch_size *
            2)
        hash_max_seq_len = self.each_layer_max_hash_numel // (
            self.num_key_value_heads * self.hash_dim * self.curr_batch_size)
        self.max_seq_len = min(kv_max_seq_len, hash_max_seq_len)

        # print("max_seq_len", self.max_seq_len)

        numel = 2 * batch_size * self.max_seq_len * \
            self.num_key_value_heads * self.head_dim
        hash_numel = batch_size * self.max_seq_len * \
            self.num_key_value_heads * self.hash_dim

        for i in range(self.num_layers):
            self.layer_caches[i] = self.max_layer_caches[i][:numel]
            self.layer_caches[i] = self.layer_caches[i].view(
                2, batch_size, self.max_seq_len, self.num_key_value_heads,
                self.head_dim)

            if i >= self.num_skip_layers:
                self.layer_hash_caches[i] = self.max_layer_hash_caches[
                    i][:hash_numel]
                self.layer_hash_caches[i] = self.layer_hash_caches[i].view(
                    batch_size, self.max_seq_len, self.num_key_value_heads,
                    self.hash_dim)
            if self.quest_enabled and i >= self.num_skip_layers:
                num_pages = (self.max_seq_len + self.quest_page_size - 1) // self.quest_page_size
                page_shape = (
                    batch_size,
                    num_pages,
                    self.num_key_value_heads,
                    self.head_dim,
                )
                self.quest_page_mins[i] = torch.empty(
                    page_shape, dtype=self.dtype, device=self.layer_devices[i]
                )
                self.quest_page_maxs[i] = torch.empty(
                    page_shape, dtype=self.dtype, device=self.layer_devices[i]
                )

        self.query_code_buffers = {}
        self.quest_query_buffers = {}
        for device in self.unique_devices:
            self.query_code_buffers[device] = torch.empty(batch_size,
                                                          1,
                                                          self.num_heads,
                                                          self.hash_dim,
                                                          dtype=torch.int32,
                                                          device=device)
            self.quest_query_buffers[device] = torch.empty(
                batch_size,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=device,
            )

    def _quest_profile_slot(self, layer_idx: int):
        if not self.quest_profile_enabled or not self._quest_profile_active:
            return None
        if layer_idx not in self.quest_profile_layers:
            return None

        decode_step = self._quest_profile_layer_steps.get(layer_idx, 0)
        self._quest_profile_layer_steps[layer_idx] = decode_step + 1
        if decode_step >= self.quest_profile_decode_tokens:
            return None
        return self._quest_profile_sample, decode_step

    def _expand_profile_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4 and tensor.shape[1] == 1:
            tensor = tensor.squeeze(1)
        if tensor.dim() != 3:
            raise RuntimeError(f"Unsupported profile tensor shape: {tuple(tensor.shape)}")
        if tensor.shape[1] == self.num_heads:
            return tensor
        if tensor.shape[1] == self.num_key_value_heads:
            return tensor.repeat_interleave(self.gqa_size, dim=1)
        raise RuntimeError(
            f"Profile head mismatch: got {tensor.shape[1]}, "
            f"expected {self.num_heads} or {self.num_key_value_heads}"
        )

    @torch.no_grad()
    def _record_quest_profile(
        self,
        profile_slot,
        layer_idx: int,
        seq_len: int,
        fetch_num: int,
        encoded_query: torch.Tensor,
        coarse_idx: torch.Tensor,
        valid_mask: torch.Tensor,
        final_idx: torch.Tensor,
        coarse_ms: float,
        fine_hash_ms: float,
    ):
        sample_idx, decode_step = profile_slot
        query = self.quest_query_buffers[self.layer_devices[layer_idx]]
        batch_size = query.shape[0]
        query = query.float().view(
            batch_size,
            self.num_key_value_heads,
            self.gqa_size,
            self.head_dim,
        )
        keys = self.layer_caches[layer_idx][0, :, :seq_len].float()
        exact_scores = torch.einsum("bkgd,bskd->bkgs", query, keys)
        exact_scores = exact_scores.reshape(batch_size, self.num_heads, seq_len)
        exact_scores = exact_scores / math.sqrt(self.head_dim)

        oracle_k = min(fetch_num, seq_len)
        oracle_idx = torch.topk(exact_scores, oracle_k, dim=-1).indices

        coarse_idx_q = self._expand_profile_heads(coarse_idx.long())
        valid_mask_q = self._expand_profile_heads(valid_mask)
        coarse_counts = torch.zeros(
            (batch_size, self.num_heads, seq_len),
            dtype=torch.int16,
            device=coarse_idx.device,
        )
        coarse_counts.scatter_add_(
            -1, coarse_idx_q, valid_mask_q.to(torch.int16)
        )
        coarse_mask = coarse_counts > 0

        final_idx_q = self._expand_profile_heads(final_idx.long())
        final_mask = torch.zeros_like(coarse_mask)
        final_mask.scatter_(-1, final_idx_q, True)

        global_start = torch.cuda.Event(enable_timing=True)
        global_end = torch.cuda.Event(enable_timing=True)
        global_start.record()
        global_score = KVLib.hamming_score(
            self.layer_hash_caches[layer_idx],
            encoded_query,
            self.hash_rbits,
            seq_len,
            sink=self.num_sink,
            recent=self.num_recent,
        )
        global_idx = KVLib.batch_topk(global_score, fetch_num, False)
        global_end.record()
        global_end.synchronize()
        global_hash_ms = global_start.elapsed_time(global_end)

        global_idx_q = self._expand_profile_heads(global_idx.long())
        global_mask = torch.zeros_like(coarse_mask)
        global_mask.scatter_(-1, global_idx_q, True)

        forced_mask = torch.zeros(
            (1, 1, seq_len), dtype=torch.bool, device=exact_scores.device
        )
        if self.num_sink > 0:
            forced_mask[..., :min(self.num_sink, seq_len)] = True
        if self.num_recent > 0:
            forced_mask[..., max(0, seq_len - self.num_recent):] = True

        forced_count = int(forced_mask.sum().item())
        available = seq_len - forced_count
        no_forced_k = min(max(fetch_num - forced_count, 0), max(available, 0))
        if no_forced_k:
            no_forced_scores = exact_scores.masked_fill(forced_mask, -torch.inf)
            oracle_no_forced = torch.topk(
                no_forced_scores, no_forced_k, dim=-1
            ).indices
            coarse_no_forced = coarse_mask & ~forced_mask
            final_no_forced = final_mask & ~forced_mask
            global_no_forced = global_mask & ~forced_mask
            coarse_recall_no_forced = coarse_no_forced.gather(
                -1, oracle_no_forced
            ).float().mean(dim=-1)
            final_recall_no_forced = final_no_forced.gather(
                -1, oracle_no_forced
            ).float().mean(dim=-1)
            global_recall_no_forced = global_no_forced.gather(
                -1, oracle_no_forced
            ).float().mean(dim=-1)
        else:
            zeros = torch.zeros(
                (batch_size, self.num_heads), device=exact_scores.device
            )
            coarse_recall_no_forced = zeros
            final_recall_no_forced = zeros
            global_recall_no_forced = zeros

        coarse_recall = coarse_mask.gather(-1, oracle_idx).float().mean(dim=-1)
        final_recall = final_mask.gather(-1, oracle_idx).float().mean(dim=-1)
        global_recall = global_mask.gather(-1, oracle_idx).float().mean(dim=-1)
        attention_prob = torch.softmax(exact_scores, dim=-1)
        coarse_mass = (attention_prob * coarse_mask).sum(dim=-1)
        final_mass = (attention_prob * final_mask).sum(dim=-1)
        global_mass = (attention_prob * global_mask).sum(dim=-1)

        self._quest_profile_records.append({
            "sample": sample_idx,
            "decode_token": decode_step,
            "layer": layer_idx,
            "seq_len": seq_len,
            "fetch_num": fetch_num,
            "candidate_tokens": int(valid_mask.sum(dim=-1).float().mean().item()),
            "candidate_ratio": coarse_mask.float().mean().item(),
            "coarse_recall": coarse_recall.mean().item(),
            "coarse_recall_min_head": coarse_recall.min().item(),
            "coarse_recall_no_forced": coarse_recall_no_forced.mean().item(),
            "final_recall": final_recall.mean().item(),
            "final_recall_min_head": final_recall.min().item(),
            "final_recall_no_forced": final_recall_no_forced.mean().item(),
            "global_hash_recall": global_recall.mean().item(),
            "global_hash_recall_no_forced": global_recall_no_forced.mean().item(),
            "coarse_attention_mass": coarse_mass.mean().item(),
            "final_attention_mass": final_mass.mean().item(),
            "global_hash_attention_mass": global_mass.mean().item(),
            "coarse_ms": coarse_ms,
            "fine_hash_ms": fine_hash_ms,
            "quest_total_ms": coarse_ms + fine_hash_ms,
            "global_hash_ms": global_hash_ms,
        })

    def _write_quest_profile(self):
        if self._quest_profile_written:
            return
        self._quest_profile_written = True

        metric_names = [
            "candidate_ratio",
            "coarse_recall",
            "coarse_recall_min_head",
            "coarse_recall_no_forced",
            "final_recall",
            "final_recall_min_head",
            "final_recall_no_forced",
            "global_hash_recall",
            "global_hash_recall_no_forced",
            "coarse_attention_mass",
            "final_attention_mass",
            "global_hash_attention_mass",
            "coarse_ms",
            "fine_hash_ms",
            "quest_total_ms",
            "global_hash_ms",
        ]
        summaries = {}
        for layer_idx in sorted(self.quest_profile_layers):
            layer_records = [
                row for row in self._quest_profile_records
                if row["layer"] == layer_idx
            ]
            if layer_idx < self.num_skip_layers:
                summaries[str(layer_idx)] = {
                    "applied": False,
                    "reason": f"full-attention skip layer (<{self.num_skip_layers})",
                    "records": 0,
                    "coarse_recall": 1.0,
                    "final_recall": 1.0,
                }
                continue
            summary = {"applied": True, "records": len(layer_records)}
            for metric in metric_names:
                values = [row[metric] for row in layer_records]
                summary[metric] = sum(values) / len(values) if values else None
            summaries[str(layer_idx)] = summary

        payload = {
            "config": {
                "samples": self.quest_profile_samples,
                "decode_tokens": self.quest_profile_decode_tokens,
                "layers": sorted(self.quest_profile_layers),
                "quest_ratio": self.quest_ratio,
                "page_size": self.quest_page_size,
                "sparse_ratio": self.sparse_ratio,
                "num_sink": self.num_sink,
                "num_recent": self.num_recent,
            },
            "summary_by_layer": summaries,
            "records": self._quest_profile_records,
        }
        output_dir = os.path.dirname(self.quest_profile_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(self.quest_profile_output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"[QUEST-PROFILE] wrote {self.quest_profile_output}", flush=True)
        print(json.dumps(summaries, indent=2), flush=True)

    def append_prefill(self, key_states: torch.Tensor,
                       value_states: torch.Tensor, layer_idx: int):
        q_len = key_states.shape[1]
        seq_start = self.layer_cache_lens[layer_idx]

        # middle
        self.layer_caches[layer_idx][0, :,
                                     seq_start:seq_start+q_len, :, :] = key_states
        self.layer_caches[layer_idx][1, :,
                                     seq_start:seq_start+q_len, :, :] = value_states

        self.layer_cache_lens[layer_idx] += q_len
        if layer_idx == self.num_layers - 1:
            self.seq_len += q_len

        key = self.layer_caches[layer_idx][0]
        value = self.layer_caches[layer_idx][1]
        return key, value, self.layer_cache_lens[layer_idx]

    def append_decode(self, key_states: torch.Tensor,
                      value_states: torch.Tensor, layer_idx: int):

        KVLib.kvcache_append(self.layer_caches[layer_idx], key_states,
                             value_states, self.layer_cache_lens[layer_idx])

        self.layer_cache_lens[layer_idx] += 1
        if layer_idx == self.num_layers - 1:
            self.seq_len += 1

        key = self.layer_caches[layer_idx][0]
        value = self.layer_caches[layer_idx][1]
        return key, value, self.layer_cache_lens[layer_idx]

    def _rotate_search_tensor(
        self, tensor: torch.Tensor, layer_idx: int, is_query: bool = False
    ):
        """Transform search coordinates only; original K/V cache stays untouched.

        The isolated paired-nonorth path uses K_h R_h and Q_h R_h^{-T}.
        All GQA query heads mapped to KV head h reuse that head's inverse-
        transpose transform.  The legacy path remains unchanged by default.
        """
        if self.route_fallback_active:
            return tensor
        rotation = self.quest_rotations[layer_idx]
        if rotation is None:
            return tensor
        if rotation.dim() == 2:
            if not (is_query and self.quest_paired_nonorth):
                return torch.matmul(tensor, rotation)
            flat = tensor.float().reshape(-1, self.head_dim)
            transformed = torch.linalg.solve(
                rotation.float(), flat.transpose(0, 1)
            ).transpose(0, 1)
            return transformed.reshape(tensor.shape).to(dtype=tensor.dtype)
        if rotation.dim() != 3:
            raise RuntimeError(
                f"unsupported rotation rank {rotation.dim()} at layer {layer_idx}"
            )
        tensor_heads = tensor.shape[-2]
        if tensor_heads == self.num_key_value_heads:
            head_rotation = rotation
        elif tensor_heads == self.num_heads:
            head_rotation = rotation.repeat_interleave(self.gqa_size, dim=0)
        else:
            raise RuntimeError(
                f"rotation head mismatch at layer {layer_idx}: got "
                f"{tensor_heads}, expected {self.num_key_value_heads} KV "
                f"heads or {self.num_heads} query heads"
            )
        if not (is_query and self.quest_paired_nonorth):
            return torch.einsum(
                "...hd,hde->...he", tensor, head_rotation
            ).contiguous()
        per_head = tensor.float().movedim(-2, 0)
        original_shape = per_head.shape
        flat = per_head.reshape(tensor_heads, -1, self.head_dim)
        transformed = torch.linalg.solve(
            head_rotation.float(), flat.transpose(-1, -2)
        ).transpose(-1, -2)
        return transformed.reshape(original_shape).movedim(0, -2).to(
            dtype=tensor.dtype
        )

    def _refresh_quest_prefill_pages(
        self,
        layer_idx: int,
        seq_len: int,
        rotated_keys: torch.Tensor = None,
        seq_start: int = 0,
    ):
        if (
            not self.quest_enabled
            or self.route_fallback_active
            or layer_idx in self.quest_bypass_layers
        ):
            return
        page_size = self.quest_page_size
        num_pages = (seq_len + page_size - 1) // page_size
        padded_len = num_pages * page_size
        if (
            rotated_keys is not None
            and seq_start == 0
            and rotated_keys.shape[1] >= seq_len
        ):
            keys = rotated_keys[:, :seq_len]
        else:
            keys = self.layer_caches[layer_idx][0, :, :seq_len]
            keys = self._rotate_search_tensor(keys, layer_idx)
        if padded_len != seq_len:
            padding_shape = (
                keys.shape[0],
                padded_len - seq_len,
                keys.shape[2],
                keys.shape[3],
            )
            min_padding = torch.full(
                padding_shape, torch.inf, dtype=keys.dtype, device=keys.device
            )
            max_padding = torch.full(
                padding_shape, -torch.inf, dtype=keys.dtype, device=keys.device
            )
            min_keys = torch.cat((keys, min_padding), dim=1)
            max_keys = torch.cat((keys, max_padding), dim=1)
        else:
            min_keys = keys
            max_keys = keys

        min_pages = min_keys.view(
            keys.shape[0], num_pages, page_size,
            self.num_key_value_heads, self.head_dim
        ).amin(dim=2)
        max_pages = max_keys.view(
            keys.shape[0], num_pages, page_size,
            self.num_key_value_heads, self.head_dim
        ).amax(dim=2)
        self.quest_page_mins[layer_idx][:, :num_pages].copy_(min_pages)
        self.quest_page_maxs[layer_idx][:, :num_pages].copy_(max_pages)

    def _refresh_quest_decode_page(self, layer_idx: int, token_idx: int):
        if (
            not self.quest_enabled
            or self.route_fallback_active
            or layer_idx in self.quest_bypass_layers
        ):
            return
        page_size = self.quest_page_size
        page_idx = token_idx // page_size
        page_start = page_idx * page_size
        page_end = min(token_idx + 1, page_start + page_size)
        keys = self.layer_caches[layer_idx][0, :, page_start:page_end]
        keys = self._rotate_search_tensor(keys, layer_idx)
        self.quest_page_mins[layer_idx][:, page_idx].copy_(keys.amin(dim=1))
        self.quest_page_maxs[layer_idx][:, page_idx].copy_(keys.amax(dim=1))

    def _update_quest_decode_page(
        self, layer_idx: int, token_idx: int, rotated_key: torch.Tensor
    ):
        """Incrementally update one Page using the newly rotated key only."""
        if (
            not self.quest_enabled
            or self.route_fallback_active
            or layer_idx in self.quest_bypass_layers
        ):
            return
        page_idx = token_idx // self.quest_page_size
        current = rotated_key[:, -1]
        page_mins = self.quest_page_mins[layer_idx][:, page_idx]
        page_maxs = self.quest_page_maxs[layer_idx][:, page_idx]
        if token_idx % self.quest_page_size == 0:
            page_mins.copy_(current)
            page_maxs.copy_(current)
        else:
            torch.minimum(page_mins, current, out=page_mins)
            torch.maximum(page_maxs, current, out=page_maxs)

    def prefill_encode_hash(self, layer_idx, key):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"
        # key = self.layer_caches[layer_idx][
        #     0, :, :self.layer_cache_lens[layer_idx], :, :]
        seq_start = self.layer_cache_lens[layer_idx] - key.shape[1]
        self._decide_route_for_prefill(
            seq_start, self.layer_cache_lens[layer_idx]
        )
        search_key = self._rotate_search_tensor(key, layer_idx)
        hash_key = _select_hash_coordinates(
            key, search_key, self.quest_hash_branch
        )
        if self.quest_hash_branch == "parallel":
            self.quest_parallel_hash_calls += 1
        else:
            self.quest_serial_hash_calls += 1
        if self.hhr_audit_path and int(layer_idx) not in self._hhr_prefill_audited:
            with open(self.hhr_audit_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "hhr_hash_coordinate_branch",
                    "phase": "prefill",
                    "layer_idx": int(layer_idx),
                    "branch": self.quest_hash_branch,
                    "hash_uses_raw_coordinates": self.quest_hash_branch == "parallel",
                    "quest_uses_rotated_coordinates": True,
                    "paired_nonorth": self.quest_paired_nonorth,
                    "key_transform": "K_h R_h",
                    "query_transform": (
                        "Q_h R_h^{-T}"
                        if self.quest_paired_nonorth else "Q_h R_h"
                    ),
                    "rotation_path": self.quest_rotation_path,
                    "hash_weights_path": self.hash_weights_path,
                    "num_sink": int(self.num_sink),
                    "num_recent": int(self.num_recent),
                    "quest_page_size": int(self.quest_page_size),
                    "hash_rbits": int(self.hash_rbits),
                    "rotation_rank": int(self.quest_rotations[layer_idx].dim()),
                }, sort_keys=True) + "\n")
            self._hhr_prefill_audited.add(int(layer_idx))
        hash_key = self._suppress_hash_amplitude(hash_key)
        hash_key = hadamard_transform(hash_key) if self.use_hadamard else hash_key


        prefill_multi_hash_encode(
            hash_key, self._active_hash_weight(layer_idx),
            self.layer_hash_caches[layer_idx],
            self.hash_packbit_aux_tensors[self.layer_devices[layer_idx]], seq_start)
        self._refresh_quest_prefill_pages(
            layer_idx,
            self.layer_cache_lens[layer_idx],
            rotated_keys=search_key,
            seq_start=seq_start,
        )

    def decode_encode_hash(self, key, query, layer_idx):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"

        seq_start = self.layer_cache_lens[layer_idx] - 1
        search_key = self._rotate_search_tensor(key, layer_idx)
        search_query = self._rotate_search_tensor(
            query, layer_idx, is_query=True
        )
        hash_key = _select_hash_coordinates(
            key, search_key, self.quest_hash_branch
        )
        hash_query = _select_hash_coordinates(
            query, search_query, self.quest_hash_branch
        )
        if self.quest_hash_branch == "parallel":
            self.quest_parallel_hash_calls += 1
        else:
            self.quest_serial_hash_calls += 1
        if self.hhr_audit_path and int(layer_idx) not in self._hhr_decode_audited:
            with open(self.hhr_audit_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "hhr_hash_coordinate_branch",
                    "phase": "decode",
                    "layer_idx": int(layer_idx),
                    "branch": self.quest_hash_branch,
                    "hash_uses_raw_coordinates": self.quest_hash_branch == "parallel",
                    "quest_uses_rotated_coordinates": True,
                    "paired_nonorth": self.quest_paired_nonorth,
                    "key_transform": "K_h R_h",
                    "query_transform": (
                        "Q_h R_h^{-T}"
                        if self.quest_paired_nonorth else "Q_h R_h"
                    ),
                    "rotation_path": self.quest_rotation_path,
                    "hash_weights_path": self.hash_weights_path,
                    "num_sink": int(self.num_sink),
                    "num_recent": int(self.num_recent),
                    "quest_page_size": int(self.quest_page_size),
                    "hash_rbits": int(self.hash_rbits),
                    "rotation_rank": int(self.quest_rotations[layer_idx].dim()),
                }, sort_keys=True) + "\n")
            self._hhr_decode_audited.add(int(layer_idx))
        hash_key = self._suppress_hash_amplitude(hash_key)
        hash_query = self._suppress_hash_amplitude(hash_query)
        hash_key = hadamard_transform(hash_key) if self.use_hadamard else hash_key
        hash_query = hadamard_transform(hash_query) if self.use_hadamard else hash_query



        # print("debugs")
        #  KVLib.decode_multi_hash_encode(
        decode_multi_hash_encode(
            hash_key,
            self._active_hash_weight(layer_idx),
            self.layer_hash_caches[layer_idx],
            hash_query,
            self.query_code_buffers[self.layer_devices[layer_idx]],
            self.hash_packbit_aux_tensors[self.layer_devices[layer_idx]],
            seq_start)

        if (
            self.quest_enabled
            and not self.route_fallback_active
            and layer_idx not in self.quest_bypass_layers
        ):
            self.quest_query_buffers[self.layer_devices[layer_idx]].copy_(
                search_query[:, 0]
            )
            self._update_quest_decode_page(layer_idx, seq_start, search_key)

        return self.query_code_buffers[self.layer_devices[layer_idx]]

    def _suppress_hash_amplitude(self, tensor):
        """Smoothly clip search-only Hash inputs without touching QUEST or K/V."""
        if self.route_fallback_active:
            return tensor
        if self.hash_amplitude_clip is None:
            return tensor
        clip = torch.as_tensor(
            self.hash_amplitude_clip, device=tensor.device, dtype=tensor.dtype
        ).clamp_min(torch.finfo(tensor.dtype).eps)
        return clip * torch.tanh(tensor / clip)

    def _decide_route_for_prefill(self, seq_start, total_len):
        if self.route_max_seq_len is None or self.route_decided:
            return
        if seq_start != 0:
            raise RuntimeError(
                "length safety routing requires an unchunked initial prefill"
            )
        self.route_fallback_active = total_len > self.route_max_seq_len
        self.route_decided = True

    def _active_hash_weight(self, layer_idx):
        if self.route_fallback_active:
            weight = self.fallback_hash_weights[layer_idx]
            if weight is None:
                raise RuntimeError(
                    f"missing fallback Hash weight for active layer {layer_idx}"
                )
            return weight
        return self.hash_weights[layer_idx]

    def _quest_hash_score(self, coarse_idx, valid_mask, encoded_query,
                           layer_idx):
        hash_cache = self.layer_hash_caches[layer_idx].permute(0, 2, 1, 3)
        # Newer KVLib builds fuse candidate gather, grouped-query XOR and
        # popcount.  Keep the fallback so old checkpoints/environments remain
        # runnable and Hash still only participates in retrieval.
        if hasattr(KVLib, "candidate_hamming_score"):
            return KVLib.candidate_hamming_score(
                hash_cache,
                coarse_idx,
                valid_mask,
                encoded_query,
                sink=min(self.num_sink, coarse_idx.shape[-1]),
            )
        n = coarse_idx.shape[-1]
        gather_idx = coarse_idx.unsqueeze(-1).expand(
            -1, -1, -1, self.hash_dim
        )
        selected = hash_cache.gather(2, gather_idx)
        selected = selected.permute(0, 2, 1, 3).contiguous()
        score = KVLib.hamming_score(
            selected,
            encoded_query,
            self.hash_rbits,
            n,
            sink=min(self.num_sink, n),
            recent=self.num_recent,
        )
        score.masked_fill_(~valid_mask, torch.inf)
        return score

    def _quest_hash_score_confidence(self, coarse_idx, valid_mask,
                                     encoded_query, layer_idx,
                                     probe_limit, fetch_num):
        """Return candidate scores and fused Hash-boundary statistics.

        The fused CUDA path builds the small integer Hamming histogram while
        producing candidate scores.  This removes the separate float cast,
        partial Top-K and equality reduction from every dynamic QUEST layer.
        """
        if hasattr(KVLib, "candidate_hamming_confidence"):
            hash_cache = self.layer_hash_caches[layer_idx].permute(0, 2, 1, 3)
            return KVLib.candidate_hamming_confidence(
                hash_cache,
                coarse_idx,
                valid_mask,
                encoded_query,
                sink=min(self.num_sink, coarse_idx.shape[-1]),
                probe_limit=probe_limit,
                fetch_num=fetch_num,
            )
        return (
            self._quest_hash_score(
                coarse_idx, valid_mask, encoded_query, layer_idx
            ),
            None,
        )

    def _quest_budget_ratios(self, layer_idx):
        if layer_idx in self.quest_layer_budgets:
            return self.quest_layer_budgets[layer_idx]
        if layer_idx in self.quest_low_budget_layers:
            return (
                self.quest_low_probe_ratio,
                self.quest_low_base_ratio,
                self.quest_low_max_ratio,
            )
        return (
            self.quest_probe_ratio,
            self.quest_base_ratio,
            self.quest_max_ratio,
        )

    def _static_quest_head_ratios(self, layer_idx):
        if self.quest_head_budget_path is None:
            return None
        return self.quest_head_budgets.get(
            layer_idx, self.quest_head_budget_default
        )

    def _dynamic_quest_candidate_ratio(self, score, valid_mask, metadata,
                                       fetch_num, budget_ratios=None,
                                       confidence_stats=None,
                                       score_to_mask=None, seq_len=None):
        """Calibrate Page budget from Hash ambiguity and QUEST boundary density."""
        probe_ratio, base_ratio, max_ratio = budget_ratios or (
            self.quest_probe_ratio,
            self.quest_base_ratio,
            self.quest_max_ratio,
        )
        fused_budget = (
            confidence_stats is not None
            and score_to_mask is not None
            and seq_len is not None
            and hasattr(KVLib, "apply_dynamic_quest_budget")
        )
        if confidence_stats is None:
            kth_pos = min(max(fetch_num - 1, 0), score.shape[-1] - 1)
            guard_pos = min(
                score.shape[-1] - 1,
                kth_pos + max(2, fetch_num // 4),
            )
            # We only need two low-order statistics, not a full candidate sort.
            boundary = torch.topk(
                score.float(), guard_pos + 1, dim=-1,
                largest=False, sorted=True,
            ).values
            kth = boundary[..., kth_pos]
            guard = boundary[..., guard_pos]
            valid = torch.isfinite(score)
            tie_fraction = (
                ((score == kth.unsqueeze(-1)) & valid).sum(dim=-1).float()
                / valid.sum(dim=-1).clamp_min(1)
            )
        elif not fused_budget:
            kth = confidence_stats[..., 0]
            guard = confidence_stats[..., 1]
            tie_fraction = confidence_stats[..., 2]
        if not fused_budget:
            margin = (guard - kth).clamp_min(0) / float(self.hash_rbits)
            hash_uncertainty = torch.sigmoid((0.025 - margin) / 0.0125)
            tie_uncertainty = (tie_fraction / 0.10).clamp(0, 1)

        ranking_upper = metadata["ranking_upper_bounds"].float()
        raw_upper = metadata["raw_upper_bounds"].float()
        probe_target_pages = metadata.get(
            "probe_target_pages", metadata["target_pages"]
        )
        extra_pages = max(
            0, probe_target_pages - metadata["forced_pages"]
        )
        if extra_pages < ranking_upper.shape[-1] and extra_pages > 0:
            selected_pages = metadata.get("selected_pages")
            forced_pages = metadata["forced_pages"]
            boundary_end = forced_pages + extra_pages + 1
            if (
                selected_pages is not None
                and boundary_end <= selected_pages.shape[-1]
            ):
                # The max-budget Page Top-K is already sorted.  Reuse its
                # two boundary Page ids instead of sorting all Pages again.
                boundary_ids = selected_pages[
                    ..., forced_pages + extra_pages - 1:boundary_end
                ]
                boundary = raw_upper.gather(-1, boundary_ids)
            else:
                boundary = torch.topk(
                    ranking_upper, extra_pages + 1, dim=-1
                ).values[..., extra_pages - 1:extra_pages + 1]
            scale = raw_upper.std(dim=-1).clamp_min(1e-6)
            gap = (
                (boundary[..., 0] - boundary[..., 1])
                / scale
            ).clamp_min(0)
            bound_uncertainty = torch.exp(-gap / 0.20)
        else:
            gap = torch.zeros_like(raw_upper[..., 0])
            bound_uncertainty = torch.zeros_like(gap)

        if fused_budget:
            diagnostics, final_pages = KVLib.apply_dynamic_quest_budget(
                score_to_mask,
                confidence_stats,
                gap.contiguous(),
                bound_uncertainty.contiguous(),
                seq_len,
                self.quest_page_size,
                metadata["forced_pages"],
                probe_ratio,
                base_ratio,
                max_ratio,
                self.hash_rbits,
            )
            self.quest_dynamic_final_pages = final_pages
            self.quest_dynamic_score_masked = True
            self.quest_dynamic_last = {
                "ratio": diagnostics[0].detach(),
                "hash_margin": diagnostics[1].detach(),
                "tie_fraction": diagnostics[2].detach(),
                "bound_gap": diagnostics[3].detach(),
                "uncertainty": diagnostics[4].detach(),
            }
            return diagnostics[0]

        uncertainty = (
            0.55 * hash_uncertainty.mean()
            + 0.20 * tie_uncertainty.mean()
            + 0.25 * bound_uncertainty.mean()
        ).clamp(0, 1)
        ratio = base_ratio + uncertainty * (max_ratio - base_ratio)
        ratio = ratio.clamp(
            min=probe_ratio, max=max_ratio
        )
        self.quest_dynamic_last = {
            "ratio": ratio.detach(),
            "hash_margin": margin.mean().detach(),
            "tie_fraction": tie_fraction.mean().detach(),
            "bound_gap": gap.mean().detach(),
            "uncertainty": uncertainty.detach(),
        }
        return ratio

    def _select_extra_sink_indices(
        self,
        score: torch.Tensor,
        candidate_indices: torch.Tensor,
        seq_len: int,
        non_sink_k: int,
        largest: bool = False,
    ) -> torch.Tensor:
        """Select non-sink TopK, then append every sink token explicitly.

        ``non_sink_k`` is the full sparse budget over the searchable region;
        sink tokens never consume those slots.  This implements the locked
        experiment contract ``selected = sink16 UNION Top1.5%(non-sink)``.
        """
        sink_count = min(max(int(self.num_sink), 0), int(seq_len))
        searchable = max(0, int(seq_len) - sink_count)
        non_sink_k = min(max(int(non_sink_k), 0), searchable)
        batch, heads, _ = score.shape

        pieces = []
        if sink_count:
            sink = torch.arange(
                sink_count, device=score.device, dtype=torch.int32
            ).view(1, 1, -1).expand(batch, heads, -1)
            pieces.append(sink)

        if non_sink_k:
            eligible = candidate_indices >= sink_count
            sentinel = -torch.inf if largest else torch.inf
            ranking = score.float().masked_fill(~eligible, sentinel)
            selected_positions = KVLib.batch_topk(
                ranking, non_sink_k, largest
            )
            selected = candidate_indices.gather(
                -1, selected_positions.long()
            ).int()
            if torch.any(selected < sink_count):
                raise RuntimeError("extra-sink TopK selected a sink token twice")
            pieces.append(selected)

        if not pieces:
            return torch.empty(
                batch, heads, 0, device=score.device, dtype=torch.int32
            )
        result = torch.cat(pieces, dim=-1)
        expected = sink_count + non_sink_k
        if result.shape[-1] != expected:
            raise RuntimeError(
                f"extra-sink selection width mismatch: {result.shape[-1]} != {expected}"
            )
        if self.hhr_audit_path and not self._extra_sink_selection_audited:
            with open(self.hhr_audit_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "event": "extra_sink_non_sink_topk_selection",
                    "contract": "sink_union_non_sink_topk",
                    "sink_count": sink_count,
                    "non_sink_topk_count": non_sink_k,
                    "selected_count": int(result.shape[-1]),
                    "seq_len": int(seq_len),
                    "sparse_ratio": float(self.sparse_ratio),
                }) + "\n")
            self._extra_sink_selection_audited = True
        return result

    def compute_topk(self, encoded_query: torch.Tensor, seq_len: int,
                     layer_idx: int):
        assert layer_idx >= self.num_skip_layers, f"hash topk is not enabled in layer{layer_idx}!"
        # Canonical dense-equivalence path. Formal 1.5% sparse runs never
        # enter this branch.
        if self.sparse_ratio >= 1 and int(self.sparse_ratio) >= seq_len:
            return torch.arange(
                seq_len, device=encoded_query.device, dtype=torch.int32
            ).view(1, 1, seq_len).expand(
                self.curr_batch_size, self.num_key_value_heads, seq_len
            )


        if (
            self.quest_enabled
            and not self.route_fallback_active
            and layer_idx not in self.quest_bypass_layers
        ):
            profile_slot = self._quest_profile_slot(layer_idx)
            if profile_slot is not None:
                coarse_start = torch.cuda.Event(enable_timing=True)
                coarse_end = torch.cuda.Event(enable_timing=True)
                fine_end = torch.cuda.Event(enable_timing=True)
                coarse_start.record()
            searchable_tokens = max(0, int(seq_len) - int(self.num_sink))
            if self.sparse_ratio < 1:
                fetch_num = (
                    max(1, int(searchable_tokens * self.sparse_ratio))
                    if searchable_tokens else 0
                )
            else:
                fetch_num = max(
                    0, min(int(self.sparse_ratio), searchable_tokens)
                )
            quest_args = (
                self.quest_page_mins[layer_idx],
                self.quest_page_maxs[layer_idx],
                self.quest_query_buffers[self.layer_devices[layer_idx]],
                seq_len,
            )
            if self.quest_dynamic:
                budget_ratios = self._quest_budget_ratios(layer_idx)
                probe_ratio, _, max_ratio = budget_ratios
                coarse_idx, valid_mask, metadata = quest_page_select(
                    *quest_args,
                    max_ratio,
                    self.quest_page_size,
                    self.num_sink,
                    self.num_recent,
                    return_metadata=True,
                )
                if profile_slot is not None:
                    coarse_end.record()
                active_pages = math.ceil(seq_len / self.quest_page_size)
                probe_target_pages = min(
                    active_pages,
                    math.ceil(
                        max(1, math.ceil(seq_len * probe_ratio))
                        / self.quest_page_size
                    ),
                )
                probe_pages = max(
                    metadata["forced_pages"], probe_target_pages
                )
                metadata["probe_target_pages"] = probe_pages
                probe_limit = min(
                    coarse_idx.shape[-1],
                    probe_pages * self.quest_page_size,
                )
                score, confidence_stats = self._quest_hash_score_confidence(
                    coarse_idx,
                    valid_mask,
                    encoded_query,
                    layer_idx,
                    probe_limit=probe_limit,
                    fetch_num=fetch_num,
                )
                positions = torch.arange(
                    coarse_idx.shape[-1], device=coarse_idx.device
                ).view(1, 1, -1)
                fused_budget = (
                    confidence_stats is not None
                    and hasattr(KVLib, "apply_dynamic_quest_budget")
                )
                if fused_budget:
                    probe_valid_mask = valid_mask
                    probe_score = score
                else:
                    probe_valid_mask = valid_mask & (
                        positions < probe_pages * self.quest_page_size
                    )
                    probe_score = score.masked_fill(
                        ~probe_valid_mask, torch.inf
                    )
                self.quest_dynamic_score_masked = False
                dynamic_ratio = self._dynamic_quest_candidate_ratio(
                    probe_score, probe_valid_mask, metadata, fetch_num,
                    budget_ratios=budget_ratios,
                    confidence_stats=confidence_stats,
                    score_to_mask=score if fused_budget else None,
                    seq_len=seq_len if fused_budget else None,
                )
                if self.quest_dynamic_score_masked:
                    final_pages = self.quest_dynamic_final_pages
                else:
                    final_target_pages = torch.ceil(
                        torch.ceil(dynamic_ratio * seq_len).clamp_min(1)
                        / self.quest_page_size
                    ).clamp_max(active_pages)
                    final_pages = final_target_pages.clamp_min(
                        metadata["forced_pages"]
                    )
                valid_mask = valid_mask & (
                    positions < final_pages * self.quest_page_size
                )
                if not self.quest_dynamic_score_masked:
                    score.masked_fill_(~valid_mask, torch.inf)
            else:
                quest_ratio = self._static_quest_head_ratios(layer_idx)
                if quest_ratio is None:
                    quest_ratio = self.quest_ratio
                coarse_idx, valid_mask = quest_page_select(
                    *quest_args,
                    quest_ratio,
                    self.quest_page_size,
                    self.num_sink,
                    self.num_recent,
                )
                if profile_slot is not None:
                    coarse_end.record()
                score = self._quest_hash_score(
                    coarse_idx, valid_mask, encoded_query, layer_idx
                )
            largest = False
            ranking_score = score
            if self.quest_tie_break_strength:
                if self.quest_tie_break_mode == "token_hash":
                    token = coarse_idx.long()
                    offset = (
                        (token * 1103515245 + 12345) & 0x7FFFFFFF
                    ).float() / float(0x80000000)
                else:
                    position = torch.arange(
                        score.shape[-1], device=score.device,
                        dtype=torch.float32,
                    )
                    offset = position / max(1, score.shape[-1])
                    offset = offset.view(1, 1, -1)
                    if self.quest_tie_break_mode == "reverse":
                        offset = -offset
                ranking_score = score.float() + (
                    self.quest_tie_break_strength * offset
                )
            topk_indices = self._select_extra_sink_indices(
                ranking_score,
                coarse_idx,
                seq_len,
                fetch_num,
                largest=largest,
            )
            if profile_slot is not None:
                fine_end.record()
                fine_end.synchronize()
                self._record_quest_profile(
                    profile_slot=profile_slot,
                    layer_idx=layer_idx,
                    seq_len=seq_len,
                    fetch_num=fetch_num,
                    encoded_query=encoded_query,
                    coarse_idx=coarse_idx,
                    valid_mask=valid_mask,
                    final_idx=topk_indices,
                    coarse_ms=coarse_start.elapsed_time(coarse_end),
                    fine_hash_ms=coarse_end.elapsed_time(fine_end),
                )
            return topk_indices

        score = KVLib.hamming_score(self.layer_hash_caches[layer_idx],
                                    encoded_query,
                                    self.hash_rbits,
                                    seq_len,
                                    sink=self.num_sink,
                                    recent=self.num_recent)


        searchable_tokens = max(0, int(seq_len) - int(self.num_sink))
        if self.sparse_ratio < 1:
            fetch_num = (
                max(1, int(searchable_tokens * self.sparse_ratio))
                if searchable_tokens else 0
            )
        else:
            fetch_num = max(0, min(int(self.sparse_ratio), searchable_tokens))
        largest=False
        all_indices = torch.arange(
            seq_len, device=score.device, dtype=torch.int32
        ).view(1, 1, -1).expand(score.shape[0], score.shape[1], -1)
        topk_indices = self._select_extra_sink_indices(
            score, all_indices, seq_len, fetch_num, largest=largest
        )

        return topk_indices

    def get_num_skip_layers(self):
        return self.num_skip_layers


"""
===================================================
Hugging Face api reload
===================================================
"""


def prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs) -> bool:
    # Compatible: transformers 5.x (3 extra args) and 4.x (4 extra args)
    if len(args) == 3:
        generation_mode, batch_size, max_cache_length = args
        device = self.device if hasattr(self, "device") else None
    elif len(args) == 4:
        assistant_model, batch_size, max_cache_length, device = args
    else:
        batch_size = kwargs.get("batch_size", None)
        max_cache_length = kwargs.get("max_cache_length", None)
        device = kwargs.get("device", None)
        if device is None and hasattr(self, "device"):
            device = self.device
    if not hasattr(self, "_cache"):

        def get_layer_device_map(execution_device_map: Optional[dict] = None):
            if execution_device_map is None or len(execution_device_map) <= 1:
                return None
            layer_device_map = {}
            for layer in execution_device_map:
                for idx in range(self.config.num_hidden_layers):
                    if f".{idx}." in f"{layer}.":
                        layer_device_map[idx] = execution_device_map[layer]
                        break
            for idx in range(self.config.num_hidden_layers):
                if idx not in layer_device_map:
                    raise RuntimeError(
                        f"layer {idx} has not been mapped to a device.")
            return layer_device_map

        execution_device_map = None
        if hasattr(self, "hf_device_map"):
            main_device = [
                d for d in self.hf_device_map.values()
                if d not in ["cpu", "disk"]
            ][0]
            execution_device_map = {
                name: main_device if device in ["cpu", "disk"] else device
                for name, device in self.hf_device_map.items()
            }

        layer_device_map = get_layer_device_map(execution_device_map)
        self._cache = HashStaticCache(
            config=self.config.get_text_config(),
            hash_rbits=generation_config.hash_rbits,
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=self.device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
            sparse_ratio=generation_config.sparse_ratio,
            hash_weights_path=generation_config.hash_weights_path,
            num_skip_layers=getattr(generation_config, 'num_skip_layers', 2),
            num_sink=generation_config.num_sink,
            num_recent=generation_config.num_recent,
            quest_ratio=getattr(generation_config, 'quest_ratio', 0.2),
            quest_page_size=getattr(generation_config, 'quest_page_size', 16),
            quest_dynamic=getattr(generation_config, 'quest_dynamic', False),
            quest_probe_ratio=getattr(
                generation_config, 'quest_probe_ratio', 0.04
            ),
            quest_base_ratio=getattr(
                generation_config, 'quest_base_ratio', 0.05
            ),
            quest_max_ratio=getattr(
                generation_config, 'quest_max_ratio', 0.08
            ),
            quest_rotation_path=getattr(
                generation_config, 'quest_rotation_path', None
            ),
            quest_hash_branch=getattr(
                generation_config, 'quest_hash_branch', 'serial'
            ),
            use_hadamard=getattr(generation_config, 'use_hadamard', False),
        )
        self._cache.build_cache()

    self._cache.reset(batch_size)
    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
