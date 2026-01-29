from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BlockConfig:
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    rope_base: float = 10000.0
    max_seq_len: int = 2048

#rope (rotation postion embedding helpers)

def rotate_half(x: torch.Tensor) -> torch.Tensor: # split last dim in half and rotate (x1,x2)->(-x2,x1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x*cos) + (rotate_half(x)*sin)

def build_rope_cache(head_dim: int, max_seq_len: int, base: float, device, dtype):
    assert head_dim % 2 == 0
    #inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype) #0..max-1
    freqs = torch.einsum("t,d->td", t, inv_freq)  #(T, head_dim/2), makes matrix of this
    emb = torch.cat((freqs, freqs), dim=-1)  #(T, head_dim), duplicates to match full
    cos = emb.cos()[None, None, :, :]  #(1,1,T,head_dim)
    sin = emb.sin()[None, None, :, :]  #(1,1,T,head_dim)
    return cos, sin

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg
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
        B, T, C = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.cfg.n_heads, self.head_dim).transpose(1, 2)
        #apply rope
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
    # Fallback: manual causal attention
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

class FeedForward(nn.Module):
    def __init__(self, cfg: BlockConfig, mlp_ratio: int = 4):
        super().__init__()
        hidden_dim = cfg.d_model * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.d_model),
            nn.Dropout(cfg.resid_dropout)
        )
    def forward(self, x: torch.Tensor):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = FeedForward(cfg, mlp_ratio=cfg.mlp_ratio)
        self.apply(self._init_weights)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            