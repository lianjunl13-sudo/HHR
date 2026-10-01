from .llama_utils import CustomerLlamaMLP, CustomLlamaRMSNorm, CustomLlamaRotaryEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaDecoderLayer,
    LlamaFlashAttention2,
)
from transformers.utils import logging
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...cache.kvcache_hash import HashStaticCache, prepare_cache_for_generation
import KVLib
import math
import os

logger = logging.get_logger(__name__)


def _to_int_list(x, batch_size: int):
    """
    Convert kvcache_len into a Python int list with length = batch_size.
    """
    if isinstance(x, int):
        return [x for _ in range(batch_size)]

    if torch.is_tensor(x):
        x = x.detach().view(-1).cpu().tolist()
        if len(x) == 1:
            return [int(x[0]) for _ in range(batch_size)]
        return [int(v) for v in x[:batch_size]]

    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return [int(x[0]) for _ in range(batch_size)]
        return [int(v) for v in x[:batch_size]]

    return [int(x) for _ in range(batch_size)]


def _expand_topk_indices_to_query_heads(
    topk_indices: torch.Tensor,
    batch_size: int,
    num_heads: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
):
    """
    Make topk_indices shape become [B, H, K].

    Supported common shapes:
    - [K]
    - [B, K]
    - [H, K]
    - [H_kv, K]
    - [B, H, K]
    - [B, H_kv, K]
    - [B, 1, K]
    """
    idx = topk_indices.detach().long()

    if idx.dim() == 1:
        # [K] -> [B, H, K]
        idx = idx.view(1, 1, -1).expand(batch_size, num_heads, -1)
        return idx

    if idx.dim() == 2:
        dim0, k = idx.shape

        if dim0 == batch_size:
            # [B, K] -> [B, H, K]
            idx = idx.view(batch_size, 1, k).expand(batch_size, num_heads, k)
            return idx

        if dim0 == num_heads:
            # [H, K] -> [B, H, K]
            idx = idx.view(1, num_heads, k).expand(batch_size, num_heads, k)
            return idx

        if dim0 == num_key_value_heads:
            # [H_kv, K] -> [H, K] -> [B, H, K]
            idx = idx.repeat_interleave(num_key_value_groups, dim=0)
            idx = idx[:num_heads]
            idx = idx.view(1, num_heads, k).expand(batch_size, num_heads, k)
            return idx

        if dim0 == 1:
            # [1, K] -> [B, H, K]
            idx = idx.view(1, 1, k).expand(batch_size, num_heads, k)
            return idx

        raise RuntimeError(f"Unsupported topk_indices shape: {tuple(topk_indices.shape)}")

    if idx.dim() == 3:
        b, h_or_kvh, k = idx.shape

        if b == 1 and batch_size > 1:
            idx = idx.expand(batch_size, h_or_kvh, k)
        elif b != batch_size:
            raise RuntimeError(
                f"topk_indices batch size mismatch: got {b}, expected {batch_size}"
            )

        if h_or_kvh == num_heads:
            return idx

        if h_or_kvh == num_key_value_heads:
            # [B, H_kv, K] -> [B, H, K]
            idx = idx.repeat_interleave(num_key_value_groups, dim=1)
            idx = idx[:, :num_heads, :]
            return idx

        if h_or_kvh == 1:
            # [B, 1, K] -> [B, H, K]
            idx = idx.expand(batch_size, num_heads, k)
            return idx

        raise RuntimeError(f"Unsupported topk_indices shape: {tuple(topk_indices.shape)}")

    raise RuntimeError(f"Unsupported topk_indices dim: {idx.dim()}")


def _compute_topk_iou_and_recall(
    query_states: torch.Tensor,
    cached_keys: torch.Tensor,
    topk_indices: torch.Tensor,
    kvcache_len,
    num_heads: int,
    num_key_value_heads: int,
    num_key_value_groups: int,
    scale: float,
):
    """
    Compare hash-selected top-k indices with exact dense-attention top-k indices.

    query_states: [B, q_len, H, D], decode stage usually q_len = 1
    cached_keys:   [B, S, H_kv, D]
    topk_indices:  hash-selected indices, will be expanded to [B, H, K]
    """

    with torch.no_grad():
        batch_size = query_states.shape[0]
        device = query_states.device

        query_last = query_states[:, -1, :, :]  # [B, H, D]

        idx = _expand_topk_indices_to_query_heads(
            topk_indices=topk_indices,
            batch_size=batch_size,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            num_key_value_groups=num_key_value_groups,
        ).to(device)

        cache_lens = _to_int_list(kvcache_len, batch_size)

        iou_values = []
        recall_values = []

        for b in range(batch_size):
            seq_len = int(cache_lens[b])
            if seq_len <= 0:
                continue

            for h in range(num_heads):
                kv_h = h // num_key_value_groups
                kv_h = min(kv_h, num_key_value_heads - 1)

                approx_idx = idx[b, h]
                approx_idx = approx_idx[(approx_idx >= 0) & (approx_idx < seq_len)]
                approx_idx = torch.unique(approx_idx)

                if approx_idx.numel() == 0:
                    iou_values.append(0.0)
                    recall_values.append(0.0)
                    continue

                k = min(int(idx.shape[-1]), seq_len)
                if k <= 0:
                    continue

                q = query_last[b, h].float()                          # [D]
                k_cache = cached_keys[b, :seq_len, kv_h, :].float()    # [S, D]

                scores = torch.matmul(k_cache, q) * scale              # [S]
                oracle_idx = torch.topk(scores, k=k, dim=-1).indices
                oracle_idx = torch.unique(oracle_idx)

                approx_for_cmp = approx_idx
                oracle_for_cmp = oracle_idx

                inter = (approx_for_cmp[:, None] == oracle_for_cmp[None, :]).any(dim=1).sum()
                inter = float(inter.item())

                approx_n = float(approx_for_cmp.numel())
                oracle_n = float(oracle_for_cmp.numel())

                union = approx_n + oracle_n - inter
                iou = inter / union if union > 0 else 0.0
                recall = inter / oracle_n if oracle_n > 0 else 0.0

                iou_values.append(iou)
                recall_values.append(recall)

        if len(iou_values) == 0:
            return {
                "iou_mean": 0.0,
                "iou_min": 0.0,
                "iou_max": 0.0,
                "recall_mean": 0.0,
            }

        iou_tensor = torch.tensor(iou_values, dtype=torch.float32)
        recall_tensor = torch.tensor(recall_values, dtype=torch.float32)

        return {
            "iou_mean": float(iou_tensor.mean().item()),
            "iou_min": float(iou_tensor.min().item()),
            "iou_max": float(iou_tensor.max().item()),
            "recall_mean": float(recall_tensor.mean().item()),
        }


class CustomLlamaAttention(LlamaFlashAttention2):

    _iou_header_printed = False

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.rotary_emb = CustomLlamaRotaryEmbedding(config)

        # Keep the original variable name for compatibility.
        self.sacle = 1 / math.sqrt(self.head_dim)

        # Debug options.
        # IoU logging can be disabled with config.enable_iou_log.
        self.enable_iou_log = bool(getattr(config, "enable_iou_log", False))

        # Log once every configured number of decode steps.
        self.iou_print_interval = int(getattr(config, "iou_print_interval", 1))

        # Per-layer decode counter.
        self._decode_step_for_iou = 0

    @staticmethod
    def _exact_sdpa_with_cache(
        query_states, cached_keys, cached_values, cache_seqlens, *, causal
    ):
        """Exact dense attention for prefill and full-attention layers.

        Inputs keep the existing [batch, sequence, heads, dim] layout.  The
        mask supports per-example cache lengths, while grouped-query attention
        is handled by PyTorch SDPA.  This replaces only the unavailable exact
        FlashAttention backend; HHR sparse retrieval remains in KVLib.
        """
        batch, query_len = query_states.shape[:2]
        if isinstance(cache_seqlens, int):
            lengths = torch.full(
                (batch,), cache_seqlens,
                device=query_states.device, dtype=torch.long,
            )
        elif torch.is_tensor(cache_seqlens):
            lengths = cache_seqlens.to(
                device=query_states.device, dtype=torch.long
            ).reshape(-1)
            if lengths.numel() == 1:
                lengths = lengths.expand(batch)
        else:
            lengths = torch.as_tensor(
                cache_seqlens, device=query_states.device, dtype=torch.long
            ).reshape(-1)
            if lengths.numel() == 1:
                lengths = lengths.expand(batch)
        max_len = int(lengths.max().item())
        cached_keys = cached_keys[:, :max_len]
        cached_values = cached_values[:, :max_len]

        key_pos = torch.arange(max_len, device=query_states.device)
        valid = key_pos.view(1, 1, max_len) < lengths.view(batch, 1, 1)
        if causal:
            query_pos = (
                lengths.view(batch, 1)
                - query_len
                + torch.arange(query_len, device=query_states.device).view(1, -1)
            )
            valid = valid & (
                key_pos.view(1, 1, max_len) <= query_pos.unsqueeze(-1)
            )
        attn_mask = valid.unsqueeze(1)
        output = F.scaled_dot_product_attention(
            query_states.transpose(1, 2),
            cached_keys.transpose(1, 2),
            cached_values.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=query_states.shape[2] != cached_keys.shape[2],
        )
        return output.transpose(1, 2)

    @staticmethod
    def _exact_sdpa_with_indices(
        query_states, cached_keys, cached_values, topk_indices
    ):
        """Exact sparse attention over the already selected original K/V.

        This is a portable execution backend for GPUs unsupported by
        KVLib.flash_index_decode.  It deliberately does not recompute or alter
        retrieval: ``topk_indices`` comes from the unchanged HHR
        QUEST->Hash path, and gather reads the original K/V cache.
        """
        # Canonicalize the selected set before exact attention. Retrieval
        # membership is unchanged; sorting removes rank-order-dependent
        # floating-point reduction differences.
        indices = topk_indices.to(dtype=torch.long).sort(dim=-1).values
        if indices.ndim != 3:
            raise ValueError(
                "topk_indices must be [batch, kv_head, selected_token]"
            )
        keys_by_head = cached_keys.permute(0, 2, 1, 3)
        values_by_head = cached_values.permute(0, 2, 1, 3)
        gather_index = indices.unsqueeze(-1).expand(
            -1, -1, -1, cached_keys.shape[-1]
        )
        selected_keys = keys_by_head.gather(2, gather_index)
        selected_values = values_by_head.gather(2, gather_index)
        output = F.scaled_dot_product_attention(
            query_states.transpose(1, 2),
            selected_keys,
            selected_values,
            dropout_p=0.0,
            is_causal=False,
            enable_gqa=query_states.shape[2] != selected_keys.shape[1],
        )
        return output.transpose(1, 2)

    def _print_iou_log(self, layer_idx, seq_len, stat):
        if not CustomLlamaAttention._iou_header_printed:
            print(
                "| layer | seq_len | final_iou_mean | final_iou_min | final_iou_max | final_recall_mean |",
                flush=True,
            )
            print(
                "| ----: | ------: | -------------: | ------------: | ------------: | ----------------: |",
                flush=True,
            )
            CustomLlamaAttention._iou_header_printed = True

        print(
            f"| {layer_idx:5d} | {seq_len:7d} | "
            f"{stat['iou_mean']:14.6f} | "
            f"{stat['iou_min']:13.6f} | "
            f"{stat['iou_max']:13.6f} | "
            f"{stat['recall_mean']:17.6f} |",
            flush=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HashStaticCache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size = past_key_value.curr_batch_size
        q_len = past_key_value.get_cur_q_len()
        _, hidden_size = hidden_states.size()

        is_prefill = q_len > 1

        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(-1, self.num_heads, self.head_dim)

        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(
            -1, self.num_key_value_heads, self.head_dim
        )

        value_states = self.v_proj(hidden_states)

        query_states, key_states = self.rotary_emb(
            query_states, key_states, past_key_value
        )

        query_states = query_states.view(
            batch_size, -1, self.num_heads, self.head_dim
        )
        key_states = key_states.view(
            batch_size, -1, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            batch_size, -1, self.num_key_value_heads, self.head_dim
        )

        if is_prefill:
            cached_keys, cache_values, kvcache_len = past_key_value.append_prefill(
                key_states, value_states, self.layer_idx
            )

            if self.layer_idx >= past_key_value.get_num_skip_layers():
                # must after append_prefill
                past_key_value.prefill_encode_hash(self.layer_idx, key_states)

            attn_output = self._exact_sdpa_with_cache(
                query_states,
                cached_keys,
                cache_values,
                kvcache_len,
                causal=True,
            )

        else:
            cached_keys, cache_values, kvcache_len = past_key_value.append_decode(
                key_states, value_states, self.layer_idx
            )

            if self.layer_idx >= past_key_value.get_num_skip_layers():
                # must after append_decode
                encoded_query = past_key_value.decode_encode_hash(
                    key_states, query_states, self.layer_idx
                )

                topk_indices = past_key_value.compute_topk(
                    encoded_query, kvcache_len, self.layer_idx
                )

                if self.enable_iou_log:
                    self._decode_step_for_iou += 1

                    if self._decode_step_for_iou % self.iou_print_interval == 0:
                        stat = _compute_topk_iou_and_recall(
                            query_states=query_states,
                            cached_keys=cached_keys,
                            topk_indices=topk_indices,
                            kvcache_len=kvcache_len,
                            num_heads=self.num_heads,
                            num_key_value_heads=self.num_key_value_heads,
                            num_key_value_groups=self.num_key_value_groups,
                            scale=self.sacle,
                        )

                        seq_len_list = _to_int_list(kvcache_len, batch_size)
                        seq_len_to_print = max(seq_len_list) if len(seq_len_list) > 0 else 0

                        self._print_iou_log(
                            layer_idx=self.layer_idx,
                            seq_len=seq_len_to_print,
                            stat=stat,
                        )

                # The cache ranks hash candidates with negative Euclidean distance.
                # The resulting indices are consumed directly by the fused decoder.
                if os.getenv("QH_SM120_PORTABLE", "0") == "1":
                    attn_output = self._exact_sdpa_with_indices(
                        query_states,
                        cached_keys,
                        cache_values,
                        topk_indices,
                    )
                else:
                    attn_output, _ = KVLib.flash_index_decode(
                        query_states,
                        cached_keys,
                        cache_values,
                        topk_indices,
                        self.sacle,
                    )

            else:
                attn_output = self._exact_sdpa_with_cache(
                    query_states,
                    cached_keys,
                    cache_values,
                    kvcache_len,
                    causal=False,
                )

        # PyTorch SDPA can return a non-contiguous tensor on CUDA 12.8/sm120.
        # ``reshape`` preserves the exact logical layout while handling that
        # stride pattern; the legacy ``view`` raised before projection.
        attn_output = attn_output.reshape(-1, hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class CustomLlamaDecoderLayer(LlamaDecoderLayer):

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomLlamaAttention(config, layer_idx)
        self.input_layernorm = CustomLlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = CustomLlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = CustomerLlamaMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[HashStaticCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if hidden_states.device.index != torch.cuda.current_device():
            torch.cuda.set_device(hidden_states.device)

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class CustomLlamaModel(LlamaModel):

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                CustomLlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = CustomLlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HashStaticCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        assert inputs_embeds is None, "inputs_embeds is not supported in CustomLlamaModel"
        output_attentions = False
        use_cache = True

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, "
                "and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            raise ValueError(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`."
            )

        # chunk prefill here
        # Runtime-only memory control. The token order, KV contents, logits,
        # sparse budgets, rotations, and Hash weights are unchanged.
        CHUNK_SIZE = int(os.getenv("QH_PREFILL_CHUNK_SIZE", "4096"))
        if CHUNK_SIZE < 1:
            raise ValueError("QH_PREFILL_CHUNK_SIZE must be positive")

        all_hidden_states = None
        all_self_attns = None
        next_decoder_cache = None

        for chunk_start in range(0, input_ids.shape[1], CHUNK_SIZE):
            chunk_input_ids = input_ids[:, chunk_start:chunk_start + CHUNK_SIZE]

            chunk_inputs_embeds = self.embed_tokens(chunk_input_ids)
            hidden_states = chunk_inputs_embeds
            bsz, q_len, _ = hidden_states.shape

            # all the layers share the same allocation plan
            past_key_values.alloc(q_len)

            kwargs = {}

            hidden_states = hidden_states.view(bsz * q_len, -1)

            for decoder_layer in self.layers:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=None,
                    position_ids=None,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=None,
                    position_embeddings=None,
                    **kwargs,
                )

                hidden_states = layer_outputs[0]

                if use_cache:
                    next_decoder_cache = layer_outputs[
                        2 if output_attentions else 1
                    ]

            # Non-final chunk activations are no longer needed once every
            # layer has appended its K/V state. Release them before the next
            # chunk so allocator reservations cannot accumulate across a long
            # prompt and trip the server's per-GPU safety guard.
            if chunk_start + CHUNK_SIZE < input_ids.shape[1]:
                del chunk_input_ids, chunk_inputs_embeds, layer_outputs, hidden_states
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        # get last hidden state
        hidden_states = hidden_states.view(
            bsz, q_len, -1
        )[:, -1, :].view(bsz, -1)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bsz, 1, -1)

        next_cache = next_decoder_cache

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomLlamaForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.model = CustomLlamaModel(config)
        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = (
            prepare_cache_for_generation
        )
