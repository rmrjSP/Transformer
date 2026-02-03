from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.rope import apply_rope, build_rope_cache

@dataclass
class BlockConfig:
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    rope_base: float = 10000.0
    max_seq_len: int = 2048

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.cfg = cfg
        self.head_dim = cfg.d_model // cfg.n_heads

        cos, sin = build_rope_cache(
            head_dim=self.head_dim,
            max_seq_len=cfg.max_seq_len,
            base=cfg.rope_base,
            device=torch.device("cpu"),
            dtype=torch.float32
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.attn_drop = nn.Dropout(cfg.attn_dropout)
        self.resid_drop = nn.Dropout(cfg.resid_dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)

        cos = self.rope_cos[:, :, :T, :].to(device=x.device, dtype=q.dtype)
        sin = self.rope_sin[:, :, :T, :].to(device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        try:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.cfg.attn_dropout if self.training else 0.0,
                is_causal=True
            )
        except Exception:
            scale = self.head_dim ** -0.5
            scores = (q @ k.transpose(-2, -1)) * scale
            causal_mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn = scores.softmax(dim=-1)
            attn = self.attn_drop(attn)
            y = attn @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        y = self.resid_drop(y)
        return y
