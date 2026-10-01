from typing import Dict, Optional, Union, Any
import torch
from .kvcache_fa import CustomStaticCache
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.generation.configuration_utils import GenerationConfig
import os


class CustomStaticCacheForTraining(CustomStaticCache):

    def __init__(
        self,
        config: PretrainedConfig,
        device: torch.device = None,
        dtype: torch.dtype = torch.float16,
        max_gpu_cache_memory_size: int = 1000000000,  # 0.93 GB
        layer_device_map: Optional[Dict[int, Union[str, torch.device,
                                                   int]]] = None,
    ) -> None:
        super().__init__(config, device, dtype, max_gpu_cache_memory_size,
                         layer_device_map)
        self.build_dataset = False
        # Optional page-preserving sidecar for unified QUEST+Hash training.
        # It is completely inert unless QH_PAGE_DATASET_DIR is set.
        self.qh_page_dataset_dir = os.getenv("QH_PAGE_DATASET_DIR", "")
        requested_layers = os.getenv("QH_PAGE_DATASET_LAYERS", "8,16,24,31")
        self.qh_page_dataset_layers = {
            int(layer.strip()) for layer in requested_layers.split(",")
            if layer.strip()
        }
        self.qh_page_dataset_max_samples = int(
            os.getenv("QH_PAGE_DATASET_MAX_SAMPLES", "16")
        )
        self.qh_page_dataset_only = os.getenv("QH_PAGE_DATASET_ONLY", "0") == "1"
        self.qh_page_dataset_counts = [0 for _ in range(self.num_layers)]

    def prepare_for_building_dataset(self, buffer_size, num_skip_layers,
                                     save_path):
        self.build_dataset = True

        if self.qh_page_dataset_only:
            if not self.qh_page_dataset_dir:
                raise ValueError(
                    "QH_PAGE_DATASET_ONLY=1 requires QH_PAGE_DATASET_DIR"
                )
            self.num_skip_layers = num_skip_layers
            self.buffer_size = buffer_size
            self.layer_save_buffers_q = [None for _ in range(self.num_layers)]
            self.layer_save_buffers_k = [None for _ in range(self.num_layers)]
            self.layer_save_buffers_s = [None for _ in range(self.num_layers)]
            self.layer_buffer_filled = [0 for _ in range(self.num_layers)]
            self.layer_saved_buffer_num = [0 for _ in range(self.num_layers)]
            self.save_path = save_path
            return

        self.layer_save_buffers_q = []
        self.layer_save_buffers_k = []
        self.layer_save_buffers_s = []
        self.num_skip_layers = num_skip_layers
        self.buffer_size = buffer_size
        for l in range(self.num_layers):
            if l < self.num_skip_layers:
                self.layer_save_buffers_q.append(None)
                self.layer_save_buffers_k.append(None)
                self.layer_save_buffers_s.append(None)
            else:
                self.layer_save_buffers_q.append(
                    torch.zeros((self.num_heads, buffer_size, self.head_dim),
                                dtype=self.dtype,
                                device="cpu"))
                self.layer_save_buffers_k.append(
                    torch.zeros(
                        (self.num_key_value_heads, buffer_size, self.head_dim),
                        dtype=self.dtype,
                        device="cpu"))
                self.layer_save_buffers_s.append(
                    torch.zeros((self.num_heads, buffer_size),
                                dtype=self.dtype,
                                device="cpu"))
        self.layer_buffer_filled = [0 for _ in range(self.num_layers)]
        self.layer_saved_buffer_num = [0 for _ in range(self.num_layers)]
        self.save_path = save_path

    def save_to_disk(self, q, k, s, layer_idx, shuffle=True):
        if shuffle:
            rand_idx = torch.randperm(q.shape[1], device=q.device)
            q = q[:, rand_idx, :]
            k = k[:, rand_idx, :]
            s = s[:, rand_idx]

        layer_save_path = os.path.join(self.save_path, f"layer{layer_idx:02d}")
        os.makedirs(layer_save_path, exist_ok=True)

        torch.save(
            q,
            os.path.join(
                layer_save_path,
                f"chunk{self.layer_saved_buffer_num[layer_idx]:03d}_q.pt",
            ),
        )
        torch.save(
            k,
            os.path.join(
                layer_save_path,
                f"chunk{self.layer_saved_buffer_num[layer_idx]:03d}_k.pt",
            ),
        )
        torch.save(
            s,
            os.path.join(
                layer_save_path,
                f"chunk{self.layer_saved_buffer_num[layer_idx]:03d}_s.pt",
            ),
        )

        self.layer_saved_buffer_num[layer_idx] += 1

    def force_save_buffers(self):
        if self.qh_page_dataset_only:
            return
        for l in range(self.num_layers):
            if l < self.num_skip_layers:
                continue
            elif self.layer_buffer_filled[l] > 0:
                # print(l, self.layer_buffer_filled[l])
                self.save_to_disk(
                    self.layer_save_buffers_q[l]
                    [:, :self.layer_buffer_filled[l], :],
                    self.layer_save_buffers_k[l]
                    [:, :self.layer_buffer_filled[l], :],
                    self.layer_save_buffers_s[l]
                    [:, :self.layer_buffer_filled[l]], l)

    def save_data(self, q, k, s, layer_idx):
        self._save_qh_page_sample(q, k, s, layer_idx)
        if self.qh_page_dataset_only:
            return
        seqlen = s.shape[-1]
        offset = 0

        while seqlen > 0:
            # print(layer_idx, seqlen, offset,
            #       self.layer_buffer_filled[layer_idx])
            if seqlen >= self.buffer_size:
                self.save_to_disk(
                    q.view(self.num_heads, 1,
                           self.head_dim).expand(self.num_heads,
                                                 self.buffer_size,
                                                 self.head_dim).cpu(),
                    k[offset:offset + self.buffer_size, :, :].transpose(
                        0, 1).cpu(),
                    s[:, offset:offset + self.buffer_size].cpu(), layer_idx)
                seqlen -= self.buffer_size
                offset += self.buffer_size

            elif self.buffer_size - self.layer_buffer_filled[
                    layer_idx] <= seqlen:
                can_fill = self.buffer_size - self.layer_buffer_filled[
                    layer_idx]
                self.layer_save_buffers_q[
                    layer_idx][:, self.
                               layer_buffer_filled[layer_idx]:, :] = q.view(
                                   self.num_heads, 1, self.head_dim).expand(
                                       self.num_heads, can_fill,
                                       self.head_dim).cpu()
                self.layer_save_buffers_k[
                    layer_idx][:, self.layer_buffer_filled[layer_idx]:, :] = k[
                        offset:offset + can_fill, :, :].transpose(0, 1).cpu()
                self.layer_save_buffers_s[
                    layer_idx][:, self.layer_buffer_filled[
                        layer_idx]:] = s[:, offset:offset + can_fill].cpu()
                self.save_to_disk(self.layer_save_buffers_q[layer_idx],
                                  self.layer_save_buffers_k[layer_idx],
                                  self.layer_save_buffers_s[layer_idx],
                                  layer_idx)
                seqlen -= can_fill
                offset += can_fill
                self.layer_buffer_filled[layer_idx] = 0

            else:
                self.layer_save_buffers_q[
                    layer_idx][:, self.layer_buffer_filled[layer_idx]:self.
                               layer_buffer_filled[layer_idx] +
                               seqlen, :] = q.view(self.num_heads, 1,
                                                   self.head_dim).expand(
                                                       self.num_heads, seqlen,
                                                       self.head_dim).cpu()
                self.layer_save_buffers_k[
                    layer_idx][:, self.layer_buffer_filled[layer_idx]:self.
                               layer_buffer_filled[layer_idx] +
                               seqlen, :] = k[offset:offset +
                                              seqlen, :, :].transpose(0,
                                                                      1).cpu()
                self.layer_save_buffers_s[
                    layer_idx][:, self.layer_buffer_filled[layer_idx]:self.
                               layer_buffer_filled[layer_idx] +
                               seqlen] = s[:, offset:offset + seqlen].cpu()
                self.layer_buffer_filled[layer_idx] += seqlen
                seqlen = 0

    def _save_qh_page_sample(self, q, k, s, layer_idx):
        """Save one intact sequence before the historical token shuffle.

        Q/K are rotary-positioned tensors.  Exact scores can therefore be
        recomputed from Q/K; the legacy rank labels are retained for auditing.
        """
        if not self.qh_page_dataset_dir:
            return
        if layer_idx not in self.qh_page_dataset_layers:
            return
        sample_idx = self.qh_page_dataset_counts[layer_idx]
        if sample_idx >= self.qh_page_dataset_max_samples:
            return
        layer_dir = os.path.join(
            self.qh_page_dataset_dir, f"layer{layer_idx:02d}"
        )
        os.makedirs(layer_dir, exist_ok=True)
        final_path = os.path.join(layer_dir, f"sample{sample_idx:04d}.pt")
        tmp_path = final_path + f".tmp.{os.getpid()}"
        payload = {
            "query": q.detach().reshape(self.num_heads, self.head_dim).cpu().contiguous(),
            "key": k.detach().cpu().contiguous(),
            "legacy_rank_score": s.detach().cpu().contiguous(),
            "seq_len": int(k.shape[0]),
            "layer_idx": int(layer_idx),
            "format_version": 1,
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, final_path)
        self.qh_page_dataset_counts[layer_idx] += 1


"""
===================================================
Hugging Face api reload
===================================================
"""


def prepare_cache_for_generation(
    self,
    generation_config: GenerationConfig,
    model_kwargs: dict,
    *args,
    **kwargs,
) -> bool:
    # transformers 4.45 uses (generation_mode, batch, max_cache), while
    # 4.47+ uses (assistant_model, batch, max_cache, device).
    if len(args) == 3:
        _generation_mode, batch_size, _max_cache_length = args
        device = self.device if hasattr(self, "device") else None
    elif len(args) == 4:
        _assistant_model, batch_size, _max_cache_length, device = args
    else:
        batch_size = kwargs.get("batch_size")
        device = kwargs.get("device")
        if device is None and hasattr(self, "device"):
            device = self.device
    if batch_size is None:
        raise TypeError("prepare_cache_for_generation requires batch_size")
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
        self._cache = CustomStaticCacheForTraining(
            config=self.config.get_text_config(),
            max_gpu_cache_memory_size=generation_config.max_gpu_cache_memory,
            device=device,
            dtype=self.dtype,
            layer_device_map=layer_device_map,
        )
        self._cache.build_cache()

        # for training hash (v1)
        if hasattr(generation_config, "train") and generation_config.train:
            self._cache.rbit = generation_config.hash_rbits
            self._cache.save_path = generation_config.save_path
            self._cache.train_batch_size = generation_config.train_batch_size
            self._cache.train_epochs = generation_config.train_epochs
            self._cache.train_iters = generation_config.train_iters
            self._cache.rep_iters = generation_config.rep_iters
            self._cache.sch_iters = generation_config.sch_iters
            self._cache.lr = generation_config.lr

        # for building training dataset
        if hasattr(generation_config,
                   "build_dataset") and generation_config.build_dataset:
            self._cache.prepare_for_building_dataset(
                generation_config.buffer_size,
                generation_config.num_skip_layers, generation_config.save_path)

    self._cache.reset(batch_size)

    if self._cache.build_dataset:
        self._cache.query_idx = generation_config.query_idx
        self._cache.stop_mask = generation_config.stop_mask
        self._cache.pos_sample_ratio = generation_config.pos_sample_ratio

    cache_name = "past_key_values"
    model_kwargs[cache_name] = self._cache
