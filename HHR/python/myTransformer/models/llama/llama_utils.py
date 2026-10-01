from ..utils import SiLUAndMul
import os
import torch.nn as nn
import torch

from transformers.models.llama.modeling_llama import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)


_PORTABLE = os.getenv("QH_SM120_PORTABLE", "0") == "1"
if not _PORTABLE:
    import flashinfer
else:
    flashinfer = None


class CustomerLlamaMLP(LlamaMLP):

    def __init__(self, config):
        super().__init__(config)
        self.torch_dtype = config.torch_dtype
        self.mlp_bias = config.mlp_bias
        self.hidden_act = config.hidden_act
        assert self.hidden_act in ["silu"]

    def convert_fusion_exec(self):
        if not hasattr(self, "gate_up_proj"):
            device = self.down_proj.weight.device
            self.gate_up_proj = nn.Linear(self.hidden_size,
                                          self.intermediate_size * 2,
                                          bias=self.mlp_bias,
                                          dtype=self.torch_dtype,
                                          device=device)
            self.gate_up_proj.weight.data[:self.
                                          intermediate_size, :] = self.gate_proj.weight.data
            self.gate_up_proj.weight.data[
                self.intermediate_size:, :] = self.up_proj.weight.data
            self.act_fn = SiLUAndMul()

            del self.gate_proj
            del self.up_proj

    def forward(self, x):
        self.convert_fusion_exec()
        x = self.gate_up_proj(x)
        x = self.act_fn(x)
        x = self.down_proj(x)
        return x


class CustomLlamaRotaryEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config is not None
        if config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get(
                "rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"

        assert self.rope_type in ["default", "llama3", "linear"]

        self.fn = None
        self.fn_kwargs = {}

        if _PORTABLE:
            self.hf_rotary = LlamaRotaryEmbedding(config=config)
            return

        if self.rope_type == "linear":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['rope_scale'] = config.rope_scaling["factor"]
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn = flashinfer.apply_rope

        elif self.rope_type == "llama3":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['high_freq_factor'] = config.rope_scaling[
                'high_freq_factor']
            self.fn_kwargs['low_freq_factor'] = config.rope_scaling[
                'low_freq_factor']
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn_kwargs['rope_scale'] = config.rope_scaling['factor']
            self.fn_kwargs['old_context_len'] = config.rope_scaling[
                'original_max_position_embeddings']
            self.fn = flashinfer.apply_llama31_rope

        elif self.rope_type == "default":
            self.fn_kwargs['interleave'] = False
            self.fn_kwargs['rope_scale'] = 1
            self.fn_kwargs['rope_theta'] = config.rope_theta
            self.fn = flashinfer.apply_rope

    def forward(self, query_states, key_states, past_key_values):
        if _PORTABLE:
            indptr, offsets = past_key_values.get_rope_metadata(
                query_states.device
            )
            offsets = offsets.to(dtype=torch.long)
            if query_states.dim() == 4:
                q_len = query_states.shape[1]
                positions = offsets.unsqueeze(1) + torch.arange(
                    q_len, device=query_states.device, dtype=torch.long
                ).unsqueeze(0)
                q = query_states.permute(0, 2, 1, 3)
                k = key_states.permute(0, 2, 1, 3)
                cos, sin = self.hf_rotary(q, positions)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
                return q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)

            if query_states.dim() == 3:
                lengths = (indptr[1:] - indptr[:-1]).to(dtype=torch.long)
                positions = torch.cat([
                    offsets[i] + torch.arange(
                        int(length.item()),
                        device=query_states.device,
                        dtype=torch.long,
                    )
                    for i, length in enumerate(lengths)
                ], dim=0).unsqueeze(0)
                if positions.shape[1] != query_states.shape[0]:
                    raise ValueError(
                        "flattened RoPE metadata does not match token count: "
                        f"{positions.shape[1]} != {query_states.shape[0]}"
                    )
                q = query_states.permute(1, 0, 2).unsqueeze(0)
                k = key_states.permute(1, 0, 2).unsqueeze(0)
                cos, sin = self.hf_rotary(q, positions)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
                return (
                    q.squeeze(0).permute(1, 0, 2),
                    k.squeeze(0).permute(1, 0, 2),
                )

            raise ValueError(
                f"unsupported RoPE tensor rank: {query_states.dim()}"
            )
        indptr, offsets = past_key_values.get_rope_metadata(
            query_states.device)
        fl_q, fl_k = self.fn(query_states, key_states, indptr, offsets,
                             **self.fn_kwargs)
        return fl_q, fl_k


class CustomLlamaRMSNorm(LlamaRMSNorm):

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if _PORTABLE:
            input_dtype = hidden_states.dtype
            values = hidden_states.float()
            variance = values.square().mean(dim=-1, keepdim=True)
            values = values * torch.rsqrt(variance + self.variance_epsilon)
            return (self.weight.float() * values).to(input_dtype)
        output = flashinfer.norm.rmsnorm(hidden_states, self.weight,
                                         self.variance_epsilon)
        return output
