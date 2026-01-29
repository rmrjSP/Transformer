from transformer_block import BlockConfig, TransformerBlock
import torch

torch.manual_seed(0)

cfg = BlockConfig(d_model=128, n_heads=4, mlp_ratio=4, attn_dropout=0.1, resid_dropout=0.1)
block = TransformerBlock(cfg)
block.eval()  

B, T, C = 2, 16, 128
x1 = torch.randn(B, T, C)

x2 = x1.clone()

x2[:, 8:, :] = torch.randn(B, T-8, C)

y1 = block(x1)
y2 = block(x2)

diff_early = (y1[:, :8, :] - y2[:, :8, :]).abs().max().item()
diff_late  = (y1[:, 8:, :] - y2[:, 8:, :]).abs().max().item()

print("max abs diff (early positions 0..7):", diff_early)
print("max abs diff (late  positions 8..15):", diff_late)
