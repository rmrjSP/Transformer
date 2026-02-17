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
    top_p: float = None,
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

        # Repetition penalty
        repetition_penalty = 1.5  # try 1.05 to 1.2
        for b in range(idx.size(0)):
            prev = idx[b].tolist()
            next_logits[b, prev] /= repetition_penalty

        probs = F.softmax(next_logits, dim=-1)  # (B, vocab)

        # Top-k (optional)
        if top_k is not None and top_k > 0:
            v, _ = torch.topk(probs, k=min(top_k, probs.size(-1)))
            cutoff = v[:, -1].unsqueeze(-1)
            probs = torch.where(probs < cutoff, torch.zeros_like(probs), probs)


        # Top-p nucleus (optional)
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumsum = torch.cumsum(sorted_probs, dim=-1)

            # mask tokens that push cumulative mass above top_p
            mask = cumsum > top_p
            # always keep at least 1 token
            mask[:, 0] = False

            sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)

            # scatter back to original order
            probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)

        # renormalize
        probs = probs / probs.sum(dim=-1, keepdim=True)

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
        ckpt = torch.load("checkpoints/latest.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        model.eval()
        print("Loaded checkpoint at step:", ckpt["step"])
    except FileNotFoundError:
        print("No checkpoint found at 'checkpoints/latest.pt'. Proceeding with untrained model.")

    prompt = 'ERROR 2026-02-03 service=db msg="'
    idx = tok.encode_tensor(prompt, device=device).unsqueeze(0)  # (1, T)

    out = generate(model, idx, max_new_tokens=60, temperature=0.9, top_k=50, top_p=0.9)
    text = tok.decode(out[0].tolist())

    print(text)


if __name__ == "__main__":
    main()
