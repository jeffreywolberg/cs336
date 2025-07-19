from abc import ABC
import regex as re
from collections import Counter
from typing import Iterator, Union
import numpy as np


END_OF_TEXT_STR = "<|endoftext|>"
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
        # + special token + byte values (256 possible ones)
        self._orig_vocab = {0: END_OF_TEXT_STR.encode("utf-8")}
        self._orig_vocab.update({i+1 : bytes([i]) for i in range(256)}) # index to bytes
        self._vocab = dict(self._orig_vocab)

    def strip_special_tokens(self, text: str):
        re.split()

    def pretokenize(self, text : str) -> Iterator[re.Match[str]]:
        # Pretokenizer splits a block of text into smaller chunks
        return re.finditer(PAT, text)

    def _merge(self, indices : list[int], old_pair : tuple[int, int], new_idx : int, words_nbyte : list[int], n_seen : int):
        new_indices = []
        words_nbyte_cumsum = np.cumsum(words_nbyte)
        new_words_nbyte = [0 for _ in words_nbyte]
        n_merged = 0
        w_num = 0

        assert len(words_nbyte) == len(new_words_nbyte) == len(words_nbyte_cumsum)

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

            # print(i, w_num, len(words_nbyte), words_nbyte_cumsum[-1], len(indices))
            new_words_nbyte[w_num] += 1
            w_num += i >= words_nbyte_cumsum[w_num]
        
        assert n_merged == n_seen
        print(np.sum(new_words_nbyte), np.sum(words_nbyte) - n_merged)
        assert np.sum(new_words_nbyte) == np.sum(words_nbyte) - n_merged
        return new_indices, new_words_nbyte


    def train(self, train_data : str, iters=200):
        self._vocab = dict(self._orig_vocab)
        self._train_data = train_data
        pretokenized_train_data : Iterator[re.Match[str]] = self.pretokenize(train_data)
        pretokenization_bounds = [d.start() for d in pretokenized_train_data] + [len(self._train_data)]
        # words_nbyte = [len(bytes(train_data[i0:i1].encode())) for i0, i1 in zip(pretokenization_bounds[:-1], pretokenization_bounds[1:])]
        indices : list[int] = [int(b) + 1 for b in bytes(train_data.encode("utf-8"))] # indices into vocab to lookup byte sequences
        indices = []
        words_nbyte = []
        for st, end in zip(pretokenization_bounds[:-1], pretokenization_bounds[1:]):
            word = train_data[st : end].encode("utf-8")
            word_bytes = [int(b) + 1 for b in bytes(word)]
            indices.extend(word_bytes)
            words_nbyte.append(len(word_bytes))

        self._merges : dict[tuple[int, int], int] = {} # index1, index2 -> merged index

        for i in range(iters):
            pairs = []
            l_cumsum = 0
            for word_l in words_nbyte:
                pairs.extend([(a, b) for a, b in zip(indices[l_cumsum:l_cumsum+word_l-1], indices[l_cumsum+1:l_cumsum+word_l])])
                l_cumsum += word_l
            
            if len(pairs) == 0:
                return 
            
            pairs_counter = Counter(pairs)
            pair_to_merge, nseen = max(pairs_counter.items(), key=lambda kv : (kv[1], *kv[0])) # sort by count first, then lexigraphical priority
            new_idx = len(self._vocab)
            self._vocab[new_idx] = self._vocab[pair_to_merge[0]] + self._vocab[pair_to_merge[1]] # byte concatenation
            self._merges[pair_to_merge] = new_idx
            indices, words_nbyte = self._merge(indices, pair_to_merge, new_idx, words_nbyte, nseen)
        
    def encode(text : str) -> list[int]:
        ...

    def decode(self, tokens : Union[list[int], int]) -> str:
        if isinstance(tokens, (int)):
            if tokens == 0:
                return END_OF_TEXT_STR
            elif tokens < len(self._orig_vocab):
                return chr(int.from_bytes(self._vocab[tokens]))
            else:
                return self._vocab[tokens].decode("utf-8")
        else:
            return ''.join([self.decode(c) for c in tokens])

if __name__ == "__main__":
    train_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest ûÿ"
    # train_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    bpe = BPETokenizer()
    bpe.train(train_text, iters=25)

    for i in range(250, len(bpe._vocab)):
        print(i, '->', bpe.decode([i]))