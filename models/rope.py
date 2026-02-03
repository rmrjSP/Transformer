import torch

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)

def build_rope_cache(head_dim: int, max_seq_len: int, base: float, device, dtype):
    assert head_dim % 2 == 0, "RoPE requires head_dim to be even"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device, dtype=dtype) / head_dim))
    t = torch.arange(max_seq_len, device=device, dtype=dtype)
    freqs = torch.einsum("t,d->td", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]
    return cos, sin
