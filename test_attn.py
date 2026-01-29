from transformer_block import BlockConfig, MultiHeadSelfAttention
import torch

cfg = BlockConfig(d_model=128, n_heads=4)
attn = MultiHeadSelfAttention(cfg)

x = torch.randn(2, 16, 128)  # (batch, seq_len, d_model)
y = attn(x)

print("Input shape :", x.shape)
print("Output shape:", y.shape)
