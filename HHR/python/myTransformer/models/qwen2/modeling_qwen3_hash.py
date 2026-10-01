import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.models.qwen3.modeling_qwen3 import (Qwen3ForCausalLM, apply_rotary_pos_emb, Qwen3Model, Qwen3DecoderLayer, Qwen3Attention, Qwen3RMSNorm)
import transformers
from transformers.utils import logging
from ...cache.kvcache_hash import HashStaticCache, prepare_cache_for_generation

logger = logging.get_logger(__name__)


class CustomQwen3Attention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scale = self.head_dim ** -0.5
        self.num_gqa_groups = self.num_heads // self.num_key_value_heads

    @staticmethod
    def _exact_sdpa(query_states, key_states, value_states, *, causal):
        """Exact dense attention backend for prefill and non-sparse fallback.

        Query/Key/Value use the FlashAttention-compatible [B, S, H, D]
        layout.  PyTorch SDPA handles Qwen3 grouped-query attention directly;
        this changes only the exact-attention kernel, not HHR retrieval.
        """
        output = F.scaled_dot_product_attention(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            dropout_p=0.0,
            is_causal=causal,
            enable_gqa=query_states.shape[2] != key_states.shape[2],
        )
        return output.transpose(1, 2).reshape(
            query_states.shape[0], query_states.shape[1], -1
        )

    def _compute_sparse_attention(
        self, query, past_key_values, layer_idx, key_for_hash, q_for_hash=None
    ):
        """Use the shared QUEST -> candidate Hash -> original K/V route."""
        if (layer_idx < past_key_values.get_num_skip_layers()
                or past_key_values.hash_weights[layer_idx] is None):
            return None
        seq_len = past_key_values.layer_cache_lens[layer_idx]
        cached_keys = past_key_values.layer_caches[layer_idx][0, :, :seq_len, :, :]
        cached_vals = past_key_values.layer_caches[layer_idx][1, :, :seq_len, :, :]
        hash_input = q_for_hash if q_for_hash is not None else query
        encoded_query = past_key_values.decode_encode_hash(
            key_for_hash, hash_input, layer_idx
        )
        final_indices = past_key_values.compute_topk(
            encoded_query, seq_len, layer_idx
        ).long().sort(dim=-1).values
        num_kv = self.num_key_value_heads
        group_size = self.num_heads // num_kv
        cached_keys = cached_keys.permute(0, 2, 1, 3)
        cached_vals = cached_vals.permute(0, 2, 1, 3)
        sel_k = cached_keys.gather(2, final_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        sel_v = cached_vals.gather(2, final_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim))
        sel_k = sel_k.permute(0, 2, 1, 3)
        sel_v = sel_v.permute(0, 2, 1, 3)
        sel_k = sel_k.unsqueeze(-2).expand(-1, -1, -1, group_size, -1).reshape(query.shape[0], -1, self.num_heads, self.head_dim)
        sel_v = sel_v.unsqueeze(-2).expand(-1, -1, -1, group_size, -1).reshape(query.shape[0], -1, self.num_heads, self.head_dim)
        q_t = query.transpose(1, 2)
        k_t = sel_k.transpose(1, 2)
        v_t = sel_v.transpose(1, 2)
        attn = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=False)
        return attn.transpose(1, 2).reshape(query.shape[0], 1, -1)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, cache_position=None, **kwargs):
        bsz, q_len, _ = hidden_states.shape
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_proj_out = self.q_proj(hidden_states).view(hidden_shape)
        k_proj_out = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        q_proj_out = self.q_norm(q_proj_out)
        k_proj_out = self.k_norm(k_proj_out)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(q_proj_out.transpose(1, 2), k_proj_out.transpose(1, 2), cos, sin)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        
        k_for_hash = key_states
        q_for_hash = query_states

        if past_key_values is not None:
            kvcache_len = past_key_values.layer_cache_lens[self.layer_idx]
            if kvcache_len == 0:
                attn_output = self._exact_sdpa(
                    query_states, key_states, value_states, causal=True
                )
                past_key_values.append_prefill(key_states, value_states, self.layer_idx)
                if (past_key_values.hash_weights is not None and
                        self.layer_idx < len(past_key_values.hash_weights) and
                        past_key_values.hash_weights[self.layer_idx] is not None and
                        self.layer_idx >= past_key_values.get_num_skip_layers()):
                    past_key_values.prefill_encode_hash(self.layer_idx, k_for_hash)
            else:
                past_key_values.append_decode(key_states, value_states, self.layer_idx)
                new_len = past_key_values.layer_cache_lens[self.layer_idx]
                if past_key_values.sparse_ratio >= 1.0:
                    cached_keys = past_key_values.layer_caches[self.layer_idx][0, :, :new_len, :, :]
                    cached_vals = past_key_values.layer_caches[self.layer_idx][1, :, :new_len, :, :]
                    attn_output = self._exact_sdpa(
                        query_states, cached_keys, cached_vals, causal=False
                    )
                else:
                    sparse_out = self._compute_sparse_attention(
                        query_states, past_key_values, self.layer_idx,
                        key_for_hash=k_for_hash, q_for_hash=q_for_hash,
                    )
                    if sparse_out is not None:
                        attn_output = sparse_out
                    else:
                        cached_keys = past_key_values.layer_caches[self.layer_idx][0, :, :new_len, :, :]
                        cached_vals = past_key_values.layer_caches[self.layer_idx][1, :, :new_len, :, :]
                        attn_output = self._exact_sdpa(
                            query_states, cached_keys, cached_vals, causal=False
                        )
        else:
            attn_output = self._exact_sdpa(
                query_states, key_states, value_states, causal=True
            )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomQwen3Attention(config, layer_idx)


class CustomQwen3Model(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([CustomQwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])


class CustomQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen3Model(config)
        self.prepare_cache_for_generation = prepare_cache_for_generation

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        from myTransformer.cache.kvcache_hash import prepare_cache_for_generation as _pcfg
        return _pcfg(self, generation_config, model_kwargs, *args, **kwargs)
