from models.tokenizer_wrapper import TokenizerWrapper

tok = TokenizerWrapper("tokenizer.json")

print("vocab_size:", tok.vocab_size)
print("pad/unk/bos/eos:", tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id)

s = 'ERROR service=db msg="timeout"\n'
ids = tok.encode_ids(s, add_bos=True, add_eos=True)
back = tok.decode(ids[1:-1])  # drop BOS/EOS for decode demonstration

print("ids:", ids)
print("back:", repr(back))
