import torch
import torch.nn as nn

from models.attention import BlockConfig, MultiHeadSelfAttention

class FeedForward(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        hidden_dim = cfg.d_model * cfg.mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, cfg.d_model),
            nn.Dropout(cfg.resid_dropout),
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
        self.mlp = FeedForward(cfg)

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
