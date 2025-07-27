from pickle import load, dump
from .tokenizer import BPETokenizer
from os.path import join

if __name__ == "__main__":
    # only takes 5 min to tokenize tinystories
    input_filepath = "/Users/jeffreywolberg/Coding/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # input_filepath = "/Users/jeffreywolberg/Coding/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    nprocs = 1
    vocab_size = 1000
    bpe = BPETokenizer(special_tokens)
    print(f"Calling train...")
    bpe.train(input_filepath, vocab_size=vocab_size, num_processes_pretokenizer=nprocs)
    sorted_vocab = sorted(bpe._vocab.values(), key = lambda v : len(v), reverse=True)
    print(sorted_vocab[:5])

    path = join('objdump', 'TinyStoriesV2-GPT4-train_bpe.vocab')
    with open(path, 'wb') as f:
        dump(bpe._vocab, f)
    print(f"Dumped bpe vocab to {path}")

    path = join('objdump', 'TinyStoriesV2-GPT4-train_bpe.merges')
    with open(path, 'wb') as f:
        dump(bpe._merges, f)
    print(f"Dumped bpe merges to {path}")
