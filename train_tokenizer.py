from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelProcessor

tokenizer = Tokenizer(BPE(unk_token="[UNK]")) #define the tokenizer as BPE with unknown token [UNK]
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False) #use byte level pre-tokenizer

trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
)

tokenizer.train(["data/sample.txt"], trainer=trainer)

#decoding and post-processing setup
tokenizer.post_processor = ByteLevelProcessor(trim_offsets=True)
tokenizer.decoder = ByteLevelDecoder()

tokenizer.save("tokenizer.json")
print("Tokenizer trained and saved as tokenizer.json")
print("Vocab size:", tokenizer.get_vocab_size()) 