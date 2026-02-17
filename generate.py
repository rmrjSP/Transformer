import torch
import torch.nn.functional as F

from models.tokenizer_wrapper import TokenizerWrapper
from models.attention import BlockConfig
from models.gpt import GPTConfig, GPT


@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.LongTensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.LongTensor:
    """
    idx: (B, T) token ids
    returns: (B, T + max_new_tokens)
    """
    model.eval()

    for _ in range(max_new_tokens):
        logits = model(idx)              # (B, T, vocab)
        next_logits = logits[:, -1, :]   # (B, vocab) only last position

        # Temperature: higher -> more random, lower -> more deterministic
        next_logits = next_logits / max(temperature, 1e-8)

        # Top-k filtering: keep only k largest logits
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
            cutoff = v[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, float("-inf")), next_logits)

        probs = F.softmax(next_logits, dim=-1)              # (B, vocab)
        next_id = torch.multinomial(probs, num_samples=1)   # (B, 1)

        idx = torch.cat([idx, next_id], dim=1)              # append token

    return idx


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = TokenizerWrapper("tokenizer.json")

    # IMPORTANT: match the same model sizes you trained with in train_engine.py
    block = BlockConfig(d_model=256, n_heads=8, max_seq_len=2048)
    cfg = GPTConfig(vocab_size=tok.vocab_size, n_layers=4, block=block)

    model = GPT(cfg).to(device)

    # Load weights if you saved a checkpoint
    try:
        ckpt = torch.load("checkpoints/latest.pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        print("Loaded checkpoint at step:", ckpt["step"])
    except FileNotFoundError:
        print("No checkpoint found at 'checkpoints/latest.pt'. Proceeding with untrained model.")

    prompt = 'ERROR 2026-02-03 service=db msg="'
    idx = tok.encode_tensor(prompt, device=device).unsqueeze(0)  # (1, T)

    out = generate(model, idx, max_new_tokens=60, temperature=0.9, top_k=50)
    text = tok.decode(out[0].tolist())

    print(text)


if __name__ == "__main__":
    main()
