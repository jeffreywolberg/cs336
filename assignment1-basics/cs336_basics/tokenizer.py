from abc import ABC
import regex as re
from collections import Counter
from typing import Iterator, Union
import numpy as np

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # pattern

class Tokenizer(ABC):
    def train(self):
        ...

    def encode(text : str) -> list[int]:
        ...

    def decode(tokens : list[int]) -> str:
        ...

class BPETokenizer(Tokenizer):
    def __init__(self):
        self._vocab = {i : (bytes([i])) for i in range(256)} # index to bytes

    def pretokenize(self, text : str) -> Iterator[re.Match[str]]:
        # Pretokenizer splits a block of text into smaller chunks
        return re.finditer(PAT, text)

    def _merge(self, indices : list[int], old_pair : tuple[int, int], new_idx : int, word_lengths : list[int], n_seen : int):
        new_indices = []
        word_length_cumsum = np.cumsum(word_lengths)
        new_word_lengths = [0 for _ in word_lengths]
        n_merged = 0
        w_num = 0

        i = 0
        while i < len(indices):
            if i < len(indices) - 1 and indices[i] == old_pair[0] and indices[i+1] == old_pair[1]:
                new_indices.append(new_idx)
                n_merged += 1
                i += 2
                # TODO: optimization if n_merged == n_seen here by breaking out of loop
            else:
                new_indices.append(indices[i])
                i += 1

            new_word_lengths[w_num] += 1
            w_num += i >= word_length_cumsum[w_num]
        
        assert n_merged == n_seen
        assert np.sum(new_word_lengths) == np.sum(word_lengths) - n_merged
        return new_indices, new_word_lengths


    def train(self, train_data : str, iters=200):
        self._train_data = train_data
        pretokenized_train_data : Iterator[re.Match[str]] = self.pretokenize(train_data)
        pretokenization_bounds = [d.start() for d in pretokenized_train_data] + [len(self._train_data)]
        word_lengths = [i1 - i0 for i0, i1 in zip(pretokenization_bounds[:-1], pretokenization_bounds[1:])]

        indices : list[int] = list(map(int, train_data.encode("utf-8"))) # indices into vocab to lookup byte sequences
        self._merges : dict[tuple[int, int], int] = {} # index1, index2 -> merged index

        for i in range(iters):
            pairs = []
            l_cumsum = 0
            for word_l in word_lengths:
                pairs.extend([(a, b) for a, b in zip(indices[l_cumsum:l_cumsum+word_l-1], indices[l_cumsum+1:l_cumsum+word_l])])
                l_cumsum += word_l
            
            if len(pairs) == 0:
                return 
            
            pairs_counter = Counter(pairs)
            pair_to_merge, nseen = max(pairs_counter.items(), key=lambda kv : (kv[1], *kv[0])) # sort by count first, then lexigraphical priority
            new_idx = 256 + i
            self._vocab[new_idx] = pair_to_merge
            self._merges[pair_to_merge] = new_idx
            indices, word_lengths = self._merge(indices, pair_to_merge, new_idx, word_lengths, nseen)
        
    def encode(text : str) -> list[int]:
        ...

    def decode(self, tokens : Union[list[int], int]) -> str:
        if isinstance(tokens, int):
            if tokens < 256:
                return chr(tokens)
            return self.decode(self._vocab[tokens])
        else:
            return ''.join([self.decode(c) for c in tokens])

if __name__ == "__main__":
    train_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    bpe = BPETokenizer()
    bpe.train(train_text, iters=6000)
                
    for i in range(256, len(bpe._vocab)):
        print(i, '->', bpe.decode([i]))