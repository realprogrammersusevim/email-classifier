import glob
from tokenizers.pre_tokenizers import Whitespace
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d")
parser.add_argument("-v")
parser.add_argument("-m")
args = parser.parse_args()

tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(
    max_token_length=int(args.m),
    vocab_size=int(args.v),
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    show_progress=True,
)
files = glob.glob(f"{args.d}/*/*")
tokenizer.train(files=files, trainer=trainer)
tokenizer.save("tokenizer.json")

# Test it out
tokenizer_loaded = Tokenizer.from_file("tokenizer.json")
encoded = tokenizer_loaded.encode("Hello, how are you doing?")
decoded = tokenizer_loaded.decode(encoded.ids)
print(encoded)
print(encoded.ids)
print(decoded)
