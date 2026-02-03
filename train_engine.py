from dataclasses import dataclass
import torch
import torch.nn.functional as F

from models.attention import BlockConfig
from models.gpt import GPTConfig, GPT


@dataclass
class TrainConfig:
    vocab_size: int = 1000
    seq_len: int = 128
    batch_size: int = 8
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8
    lr: float = 1e-3 #set learning rate
    max_seq_len: int = 2048
    steps: int = 50

def make_model(cfg: TrainConfig, device: str) -> GPT:
    block = BlockConfig(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len
    )
    gpt_cfg = GPTConfig(vocab_size=cfg.vocab_size, n_layers=cfg.n_layers, block=block)
    model = GPT(gpt_cfg).to(device)
    return model

def next_token_loss(model:  GPT, idx: torch.Tensor) -> torch.Tensor:
    #returns cross entropy loss for next token prediction
    x = idx[:, :-1]
    y = idx[:, 1:]
    logits = model(x)  
    loss = F.cross_entropy(logits.reshape(-1, model.cfg.vocab_size), y.reshape(-1))
    return loss

def train_step(model: GPT, optimizer: torch.optim.Optimizer, idx: torch.Tensor) -> float:
    model.train()
    loss = next_token_loss(model, idx) 
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(loss.item())


def run():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = make_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for step in range(cfg.steps):
        #init with random data
        idx = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
        loss = train_step(model, optimizer, idx)

        if step % 10 == 0:
            print(f"Step {step:4d}: loss = {loss:.4f} | device = {device}")


if __name__ == "__main__":
    run()
