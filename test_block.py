from transformer_block import BlockConfig, TransformerBlock
import torch

cfg = BlockConfig(d_model=128, n_heads=4, mlp_ratio=4, attn_dropout=0.1, resid_dropout=0.1)
block = TransformerBlock(cfg)

w = block.attn.q_proj.weight.detach().cpu()
print("q_proj weight mean/std:", w.mean().item(), w.std().item())

x = torch.randn(2, 16, 128)

block.train()
y1 = block(x)
y2 = block(x)

block.eval()
y3 = block(x)
y4 = block(x)

print("Train mode: outputs identical?", torch.allclose(y1, y2))
print("Eval mode : outputs identical?", torch.allclose(y3, y4))
print("Output shape:", y1.shape)
