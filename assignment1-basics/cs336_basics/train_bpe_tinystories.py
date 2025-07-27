from .tokenizer import BPETokenizer

if __name__ == "__main__":
    input_filepath = "/Users/jeffreywolberg/Coding/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt"
    # input_filepath = "/Users/jeffreywolberg/Coding/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    special_tokens = ["<|endoftext|>"]
    nprocs = 1
    vocab_size = 1000
    bpe = BPETokenizer(special_tokens)
    print(f"Calling train...")
    bpe.train(input_filepath, vocab_size=vocab_size, num_processes_pretokenizer=nprocs)

