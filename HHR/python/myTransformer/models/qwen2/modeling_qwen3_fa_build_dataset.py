import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3DecoderLayer,
    Qwen3Attention,
)
from transformers.utils import logging
from typing import Optional, Tuple, Union
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from ...cache.kvcache_fa_for_training import CustomStaticCacheForTraining, prepare_cache_for_generation
from ..utils import SiLUAndMul

_PORTABLE = os.getenv("QH_SM120_PORTABLE", "0") == "1"
if not _PORTABLE:
    import flashinfer
    from ...cache.kernels.triton_qk_score import qk_score
else:
    flashinfer = None
    def qk_score(query, key, _seq_len):
        q = query[:, 0]
        k = key.permute(0, 2, 1, 3)
        groups = q.shape[1] // k.shape[1]
        k = k.repeat_interleave(groups, dim=1)
        return torch.einsum("bhd,bhsd->bhs", q.float(), k.float()).to(q.dtype)

# Use the same rotary embedding path as the training code.
from .qwen2_utils import CustomQwen2RotaryEmbedding

logger = logging.get_logger(__name__)


class CustomQwen3MLP(Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = config.torch_dtype
        self.hidden_act = config.hidden_act
        self.converted = False
        assert self.hidden_act in ["silu"]

    def convert_fusion_exec(self):
        if not self.converted:
            device = self.down_proj.weight.device
            self.gate_up_proj = nn.Linear(self.hidden_size,
                                          self.intermediate_size * 2,
                                          bias=False,
                                          dtype=self.torch_dtype,
                                          device=device)
            self.gate_up_proj.weight.data[:self.intermediate_size, :] = self.gate_proj.weight.data
            self.gate_up_proj.weight.data[self.intermediate_size:, :] = self.up_proj.weight.data
            self.act_fn = SiLUAndMul()
            del self.gate_proj
            del self.up_proj
            self.converted = True

    def forward(self, x):
        self.convert_fusion_exec()
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class CustomQwen3Attention(Qwen3Attention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scale = 1 / math.sqrt(self.head_dim)
        # Apply rotary embeddings with cache-aware inputs.
        self.rotary_emb = CustomQwen2RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomStaticCacheForTraining] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size = past_key_value.curr_batch_size
        q_len = past_key_value.get_cur_q_len()
        _, hidden_size = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(-1, self.num_heads, self.head_dim)
        key_states = key_states.view(-1, self.num_key_value_heads, self.head_dim)

        # Pass the query, key, and cache to the rotary embedding module.
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        query_states, key_states = self.rotary_emb(query_states, key_states, past_key_value)

        query_states = query_states.view(batch_size, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(batch_size, -1, self.num_key_value_heads, self.head_dim)

        if q_len > 1 and self.layer_idx >= past_key_value.num_skip_layers:
            query_idx = min(past_key_value.query_idx, q_len)
            assert batch_size == 1, "batch size must be 1 when building dataset"
            select_query = query_states[:, query_idx:query_idx + 1, :, :]
            select_key = key_states[:, :query_idx + 1, :, :]
            select_qk_score = qk_score(select_query, select_key, query_idx + 1).squeeze(0)
            topk_indices = torch.topk(select_qk_score,
                                      dim=-1,
                                      k=int(select_qk_score.shape[-1] * past_key_value.pos_sample_ratio),
                                      largest=True).indices
            select_qk_score[:, :] = -1.0
            topk_scores = torch.linspace(20.0, 1.0,
                                         steps=topk_indices.shape[-1],
                                         dtype=select_qk_score.dtype,
                                         device=select_qk_score.device).unsqueeze(0).expand(self.num_heads, -1)
            select_qk_score = torch.scatter(select_qk_score,
                                            dim=-1,
                                            index=topk_indices,
                                            src=topk_scores)
            past_key_value.save_data(select_query.squeeze(0),
                                     select_key.squeeze(0),
                                     select_qk_score,
                                     self.layer_idx)

        if _PORTABLE:
            attn_output = F.scaled_dot_product_attention(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                dropout_p=0.0,
                is_causal=True,
                enable_gqa=True,
            ).transpose(1, 2)
        else:
            attn_output = _flash_attention_forward(
                query_states,
                key_states,
                value_states,
                attention_mask,
                q_len,
                position_ids=position_ids,
                dropout=0,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=False,
                is_causal=self.is_causal,
            )
        attn_output = attn_output.reshape(-1, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


class CustomQwen3DecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.self_attn = CustomQwen3Attention(config, layer_idx)
        self.input_layernorm = CustomQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = CustomQwen3MLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[CustomStaticCacheForTraining] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        if hidden_states.device.index != torch.cuda.current_device():
            torch.cuda.set_device(hidden_states.device)

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

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


class CustomQwen3RMSNorm(Qwen3RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if not _PORTABLE:
            return flashinfer.norm.rmsnorm(hidden_states, self.weight, self.variance_epsilon)
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.variance_epsilon)
        return normalized.to(input_dtype) * self.weight


class CustomQwen3Model(Qwen3Model):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            CustomQwen3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = CustomQwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = None

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens,
                                          past_seen_tokens + inputs_embeds.shape[1],
                                          device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = attention_mask
        hidden_states = inputs_embeds
        bsz, seq_len, _ = hidden_states.shape

        past_key_values.alloc(seq_len)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        kwargs = {}
        hidden_states = hidden_states.view(bsz * seq_len, -1)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=None,
                **kwargs,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.view(bsz, seq_len, -1)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CustomQwen3ForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.model = CustomQwen3Model(config)
        transformers.generation.utils.GenerationMixin._prepare_cache_for_generation = prepare_cache_for_generation
