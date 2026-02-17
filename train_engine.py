from dataclasses import dataclass
import torch
import torch.nn.functional as F
import os
from models.attention import BlockConfig
from models.gpt import GPTConfig, GPT
from models.tokenizer_wrapper import TokenizerWrapper
from data.lm_dataset import TokenStreamDataset


@dataclass
class TrainConfig:
    seq_len: int = 128
    batch_size: int = 8
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 8
    lr: float = 3e-4
    max_seq_len: int = 2048
    steps: int = 200
    data_path: str = "data/sample.txt"
    tokenizer_path: str = "tokenizer.json"

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

    vocab_size = logits.size(-1)
    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
    return loss

def train_step(model: GPT, optimizer: torch.optim.Optimizer, idx: torch.Tensor) -> float:
    model.train()
    loss = next_token_loss(model, idx) 
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return float(loss.item())

def save_checkpoint(path: str, model: GPT, optimizer: torch.optim.Optimizer, step: int, cfg: TrainConfig, vocab_size: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "train_cfg": cfg.__dict__,
            "vocab_size": vocab_size,
        },
        path,
    )

def load_checkpoint(path: str, model: GPT, optimizer: torch.optim.Optimizer | None = None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optim_state" in ckpt:
        optimizer.load_state_dict(ckpt["optim_state"])
    return ckpt


def run():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = TokenizerWrapper(cfg.tokenizer_path)
    print("Tokenizer vocab_size:", tok.vocab_size)

    dataset = TokenStreamDataset(
        tokenizer=tok,
        path=cfg.data_path,
        seq_len=cfg.seq_len,
        device=device
    )


    block = BlockConfig(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        max_seq_len=cfg.max_seq_len,
    )
    gpt_cfg = GPTConfig(vocab_size=tok.vocab_size, n_layers=cfg.n_layers, block=block)
    model = GPT(gpt_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    for step in range(cfg.steps):
        idx = dataset.sample_batch(cfg.batch_size)  # (B, seq_len+1)
        loss = train_step(model, optimizer, idx)

        if step % 20 == 0:
            print(f"step {step:04d} | loss {loss:.4f} | device {device}")
    
        if step % 50 == 0:
            save_checkpoint(
                "checkpoints/latest.pt",
                model,
                optimizer,
                step,
                cfg,
                vocab_size=tok.vocab_size,
            )
    
    save_checkpoint("checkpoints/latest.pt", model, optimizer, cfg.steps, cfg, vocab_size=tok.vocab_size)
    print("Saved checkpoints/latest.pt")


if __name__ == "__main__":
    run()
