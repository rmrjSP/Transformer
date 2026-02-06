from dataclasses import dataclass
from typing import List, Optional
import torch
from tokenizers import Tokenizer

@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "[PAD]"
    unk: str = "[UNK]"
    bos: str = "[BOS]"
    eos: str = "[EOS]"

class TokenizerWrapper:
    def __init__(self, tokenizer_path: str, special: SpecialTokens = SpecialTokens()):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.special = special
        self.pad_id = self.tokenizer.token_to_id(self.special.pad)
        self.unk_id = self.tokenizer.token_to_id(self.special.unk)
        self.bos_id = self.tokenizer.token_to_id(self.special.bos)
        self.eos_id = self.tokenizer.token_to_id(self.special.eos)

        missing = [name for name, _id in [
            ("pad", self.pad_id),
            ("unk", self.unk_id),
            ("bos", self.bos_id),
            ("eos", self.eos_id)
        ] if _id is None]
        if missing:
            raise ValueError(f"Missing special tokens: {missing}")
        
    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
    def encode_ids(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self.tokenizer.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids
    
    def encode_tensor(self, text: str, add_bos: bool = False, add_eos: bool = False, device: Optional[str] = None) -> torch.LongTensor:
        ids = self.encode_ids(text, add_bos=add_bos, add_eos=add_eos)
        t = torch.tensor(ids, dtype=torch.long)
        if device is not None:
            t = t.to(device)
        return t
    
    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)