from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")

text = 'ERROR 2026-02-03 service=db msg="timeout" host=10.0.0.5\n'
enc = tok.encode(text)

print("text :", repr(text))
print("ids  :", enc.ids[:25], "... len=", len(enc.ids))
print("back :", repr(tok.decode(enc.ids)))