import torch
from models.attention import BlockConfig
from models.gpt import GPTConfig, GPT

cfg_block = BlockConfig(d_model=128, n_heads=4, max_seq_len=2048)
cfg = GPTConfig(vocab_size=1000, n_layers=2, block=cfg_block)

model = GPT(cfg)

idx = torch.randint(0, cfg.vocab_size, (2, 16))  # (B=2, T=16)
logits = model(idx)

print("idx shape   :", idx.shape)
print("logits shape:", logits.shape)
