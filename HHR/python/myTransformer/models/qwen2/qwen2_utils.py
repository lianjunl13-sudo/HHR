import os
import torch
import torch.nn as nn
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

_PORTABLE = os.getenv("QH_SM120_PORTABLE", "0") == "1"
if not _PORTABLE:
    import flashinfer

class CustomQwen2RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fn_kwargs = {
            "interleave": False,
            "rope_scale": 1.0,
            "rope_theta": getattr(config, "rope_theta", 1000000.0),
        }
        self.fn = None if _PORTABLE else flashinfer.apply_rope
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        inv_freq = 1.0 / (
            self.fn_kwargs["rope_theta"]
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, query_states, key_states, past_key_values):
        indptr, offsets = past_key_values.get_rope_metadata(query_states.device)
        if not _PORTABLE:
            return self.fn(query_states, key_states, indptr, offsets, **self.fn_kwargs)
        positions = []
        for batch_idx in range(indptr.numel() - 1):
            length = int((indptr[batch_idx + 1] - indptr[batch_idx]).item())
            start = int(offsets[batch_idx].item())
            positions.append(torch.arange(start, start + length, device=query_states.device))
        position_ids = torch.cat(positions).to(self.inv_freq.dtype)
        freqs = torch.outer(position_ids, self.inv_freq.to(query_states.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(query_states.dtype).unsqueeze(1)
        sin = emb.sin().to(query_states.dtype).unsqueeze(1)
        def rotate_half(x):
            half = x.shape[-1] // 2
            return torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return (
            query_states * cos + rotate_half(query_states) * sin,
            key_states * cos + rotate_half(key_states) * sin,
        )

class CustomQwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * x

class CustomerQwen2MLP(Qwen2MLP):
    # Keep the unfused MLP implementation for compatibility.
    pass
