import torch
from models.tokenizer_wrapper import TokenizerWrapper

class TokenStreamDataset:
    #turn file of text into stream of token ids

    def __init__ (self, tokenizer: TokenizerWrapper, path: str, seq_len: int, device: str):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.device = device
        
        text = open(path, "r", encoding="utf-8").read()
        ids = tokenizer.encode_ids(text, add_bos=False, add_eos=False)

        if len(ids) < seq_len+1:
            raise ValueError(f"File is too short ({len(ids)} tokens) for the specified sequence length ({seq_len})")
        
        self.tokens = torch.tensor(ids, dtype = torch.long, device=device)

    def sample_batch(self, batch_size: int) -> torch.LongTensor:
        # return idx with shape (batch_size, seq_len+1)
        max_start = len(self.tokens) - self.seq_len - 1
        starts = torch.randint(0, max_start + 1, (batch_size,), device=self.device)
        batch = torch.stack([
            self.tokens[start:start + self.seq_len + 1] for start in starts
        ], dim=0)
        return batch