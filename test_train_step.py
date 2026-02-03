import torch
import torch.nn.functional as F

from models.attention import BlockConfig
from models.gpt import GPT, GPTConfig

#config setup
device = "cuda" if torch.cuda.is_available() else "cpu"
cfg_block = BlockConfig(d_model=128, n_heads=4, max_seq_len=2048)
cfg = GPTConfig(vocab_size=1000, n_layers=2, block=cfg_block)

model = GPT(cfg).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

#test data
B, T = 2, 16
idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)

x = idx[:, :-1]
y = idx[:, 1:]

logits = model(x) #forward pass

#loss computation
loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1)) #flatten time (T) and batch (B) to 1 dim
print("Loss before step:", loss.item())

#backward pass and optimization step
optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

#forward after one training step
with torch.no_grad():
    logits2 = model(x)
    loss2 = F.cross_entropy(logits2.reshape(-1, cfg.vocab_size), y.reshape(-1))

print("Loss after step :", loss2.item())
print("on device       :", device)