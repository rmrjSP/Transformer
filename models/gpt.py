from dataclasses import dataclass
import torch
import torch.nn as nn

from models.attention import BlockConfig
from models.block import TransformerBlock

@dataclass
class GPTConfig:
    vocab_size: int
    n_layers: int
    block: BlockConfig

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # 1 token embedding, converts token id to vector
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.block.d_model)

        # n transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(cfg.block) for _ in range(cfg.n_layers)])

        # final layer norm + lm head
        self.ln_f = nn.LayerNorm(cfg.block.d_model)
        self.lm_head = nn.Linear(cfg.block.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight # tie weights

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        
        x = self.token_emb(idx)  # (B,T, d_model)
        for block in self.blocks:
            x = block(x)  
        
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T, vocab_size)
        return logits