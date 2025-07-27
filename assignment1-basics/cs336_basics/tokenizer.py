from abc import ABC
from array import array
import regex as re
from collections import Counter, defaultdict
from typing import BinaryIO, Iterator, Union
import numpy as np
from time import time
import multiprocessing
from tqdm import tqdm

from .pretokenization_example import find_chunk_boundaries

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # pattern

L16_MASK = 0x0000ffff
U16_MASK = ~L16_MASK

def pr_time(st : float, name=""):
    print(f"{name} took {round(time() - st, 3)} seconds")

def pack_pair(i1 : np.uint16, i2 : np.uint16) -> np.uint32:
    return (i1 << 16) | i2

def unpack_pair(packed_int : np.uint32) -> tuple[np.uint16, np.uint16]:
    i1 = (packed_int & U16_MASK) >> 16
    i2 = packed_int & L16_MASK
    return i1, i2

class Tokenizer(ABC):
    def train(self):
        ...

    def encode(text : str) -> list[int]:
        ...

    def decode(tokens : list[int]) -> str:
        ...

class Word:
    __slots__ = ("toks", "count")
    def __init__(self, toks : array('I'), count=0):
        self.toks = toks
        self.count = count

    def __repr__(self):
        NSPEC_TOKS = 1
        return f"Word('{''.join([chr(t - NSPEC_TOKS) for t in self.toks])}', count={self.count})"

class TokenNode:
    def __init__(self, vocab_idx, can_pair_forward=True):
        self.vocab_idx = vocab_idx
        self.prev : TokenNode = None
        self.next : TokenNode = None
        self.can_pair_forward = can_pair_forward # set false for splits due to pre-tokenization

    def __repr__(self):
        return f"TokenNode({self.vocab_idx}, prev={id(self.prev) if self.prev is not None else None}, next={id(self.next) if self.next is not None else None}, can_pair_forward={self.can_pair_forward})"

class BPETokenizer(Tokenizer):
    def __init__(self, special_tokens : list[str] =[]):
        # special tokens + byte values (256 possible ones)
        self.special_tokens : list[str] = special_tokens
        self._orig_vocab = {i: tok.encode("utf-8") for i, tok in enumerate(self.special_tokens)}
        self._orig_vocab.update({i+len(self.special_tokens) : bytes([i]) for i in range(256)}) # index to bytes
        self._vocab : dict[int, bytes] = dict(self._orig_vocab)
        self._merges : dict[tuple[bytes, bytes], int] = {} # (bytes1, bytes2) -> new_vocab_idx

        self._packed_pair_to_word_idxes : dict[np.uint32, list[int]] = defaultdict(list)
        self._words : list[Word] = []

    def split_on_special_tokens(self, text: str) -> list[str]:
        if len(self.special_tokens) == 0:
            return [text]
        else:
            special_tokens = [re.escape(tok) for tok in self.special_tokens]
            pat = "|".join(special_tokens)
            chunks = re.split(pat, text)
            return chunks

    def assert_special_tokens_removed(self, train_data : str):
        for stok in self.special_tokens:
            assert stok not in train_data

    def _pretokenization_worker(self, args : tuple) -> list[str]:
        input_path : str; st : int; end : int
        input_path, st, end = args

        with open(input_path, "rb") as f:
            f.seek(st)
            chunk : str = f.read(end - st).decode("utf-8", errors="ignore")

        chunk_list = self.split_on_special_tokens(chunk)
        # use findall and not finditer since multiprocessing canont pickle _regex.Scanner object returned from finditer 
        return [re.findall(PAT, chunk) for chunk in chunk_list]

    def pretokenize(self, input_path : str) -> list[Iterator[re.Match[str]]]:
        # Pretokenizer splits a block of text into smaller chunks
        with open(input_path, "r") as f:
            text = f.read()

        chunks = self.split_on_special_tokens(text)

        return [re.finditer(PAT, chunk) for chunk in chunks] # can use findall but all memory is needed at once

    def index_corpus(self, input_path : str, num_processes_pretokenizer=8):
        st1 = time()
        if num_processes_pretokenizer > 1:
            with open(input_path, "rb") as f:
                chunk_boundaries = find_chunk_boundaries(f, num_processes_pretokenizer*4, "<|endoftext|>".encode("utf-8"))

            with multiprocessing.Pool(num_processes_pretokenizer) as p:
                func_args = [(input_path, st, end) for st, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])]
                pretokenized_train_data_list = p.map(self._pretokenization_worker, func_args)
                pretokenized_train_data_list = [data for worker_result in pretokenized_train_data_list for data in worker_result]
        else:
            pretokenized_train_data_list = self.pretokenize(input_path)
        pr_time(st1, f"pretokenization")

        st2 = time()

        _word_id_to_word_idx : dict[tuple[int], int]= {}
           
        for pretokenized_train_data in tqdm(pretokenized_train_data_list, desc="linked list init"):
            for word in pretokenized_train_data:
                if isinstance(word, re.Match):
                    word = word.group()
                # print(word, len(word))
                toks : list[int] = [b + len(self.special_tokens) for b in word.encode("utf-8")]
                word_id = tuple(toks)
                if word_id in _word_id_to_word_idx:
                    word_idx = _word_id_to_word_idx[word_id]
                    self._words[word_idx].count += 1
                else:
                    word_idx = len(self._words)
                    _word_id_to_word_idx[word_id] = word_idx
                    self._words.append(Word(toks, count=1))
                    for tok1, tok2 in zip(toks[:-1], toks[1:]):
                        packed_pair = pack_pair(tok1, tok2)
                        self._packed_pair_to_word_idxes[packed_pair].append(word_idx)
    
        pr_time(st2, "linked list init")

    def compute_count(self) -> dict[np.uint32, int]:
        st3 = time()
        count = defaultdict(lambda : 0)
        count.update({packed_pair : sum([self._words[i].count for i in word_idxes]) for packed_pair, word_idxes in self._packed_pair_to_word_idxes.items()})
        pr_time(st3, "compute_count")
        return count
    
    def merge(self, packed_pair : np.uint32, new_vocab_idx : np.uint16, packed_pair_to_count : dict[np.uint32, int]):
        word_idxes = self._packed_pair_to_word_idxes[packed_pair]
        n_repeat_pair_in_word = Counter(word_idxes) # how many times does the packed pair appear in each word
        unique_word_indices = list(n_repeat_pair_in_word.keys())
        # print([self._words[i] for i in  word_idxes])
        p1, p2 = unpack_pair(packed_pair)
        # print(packed_pair_to_count[packed_pair])
        n_total_edits = 0
        for w_idx in unique_word_indices:
            word = self._words[w_idx]
            # print(word)
            tok_st_idxes_to_edit = []
            for i, (tok1, tok2) in enumerate(zip(word.toks[:-1], word.toks[1:])):
                if tok1 == p1 and tok2 == p2 and (len(tok_st_idxes_to_edit) == 0 or i - 1 != tok_st_idxes_to_edit[-1]):
                    tok_st_idxes_to_edit.append(i)

            assert len(tok_st_idxes_to_edit) != 0, f"{word}, toks: {word.toks}, p1: {p1}, p2: {p2}"
            assert np.all(np.diff(tok_st_idxes_to_edit) > 1)
            new_toks = []
            n_edits_in_word = 0
            i = 0
            while i < len(word.toks):
                tok = word.toks[i]
                if n_edits_in_word < len(tok_st_idxes_to_edit) and i == tok_st_idxes_to_edit[n_edits_in_word]:
                    new_toks.append(new_vocab_idx)
                    n_edits_in_word += 1
                    i += 2
                else:
                    new_toks.append(tok)
                    i += 1
            assert len(new_toks) + n_edits_in_word == len(word.toks), f"{len(new_toks)} + {n_edits_in_word} != {len(word.toks)}, tok_st_idxes_to_edit: {tok_st_idxes_to_edit}"

            # print(tok_st_idxes_to_edit)
            # print(word, word.toks)
            pair_to_count_inc = defaultdict(lambda : 0)
            for i in tok_st_idxes_to_edit:
                prev_pair = None if i == 0 else (word.toks[i-1], word.toks[i])
                next_pair = None if i == len(word.toks) - 2 else (word.toks[i+1], word.toks[i+2])
                if prev_pair is not None:
                    pair_to_count_inc[prev_pair] -= 1
                    pair_to_count_inc[(word.toks[i-1], new_vocab_idx)] += 1
                    self._packed_pair_to_word_idxes[pack_pair(word.toks[i-1], new_vocab_idx)].extend([w_idx] * n_repeat_pair_in_word[w_idx])
                if next_pair is not None:
                    pair_to_count_inc[next_pair] -= 1
                    pair_to_count_inc[(new_vocab_idx, word.toks[i+2])] += 1
                    self._packed_pair_to_word_idxes[pack_pair(new_vocab_idx, word.toks[i+2])].extend([w_idx] * n_repeat_pair_in_word[w_idx])

            for pair, count in pair_to_count_inc.items():
                packed_pair_to_count[pack_pair(*pair)] += count * word.count * n_repeat_pair_in_word[w_idx]

            word.toks = new_toks
            n_total_edits += n_edits_in_word * n_repeat_pair_in_word[w_idx]

        packed_pair_to_count.pop(packed_pair) # remove count since all instances were replaced

    def train(self, input_path : str, vocab_size : int, num_processes_pretokenizer=8):
        self._vocab = dict(self._orig_vocab)
        assert len(self._vocab) <= vocab_size, f"len(_vocab) {len(self._vocab)} must be <= vocab_size: {vocab_size}"

        st1 = time()
        self.index_corpus(input_path, num_processes_pretokenizer=num_processes_pretokenizer)
        pr_time(st1, "indexing corpus")

        packed_pair_to_count = self.compute_count()

        def packed_pair_to_count_cmp(packed_pair_count_item):
            p1, p2 = unpack_pair(packed_pair_count_item[0])
            count = packed_pair_count_item[1]
            return (count, self._vocab[p1], self._vocab[p2])
        
        niters = vocab_size - len(self._vocab)
        st2 = time()
        for i in tqdm(range(niters)):
            if len(packed_pair_to_count) == 0:
                break

            packed_pair, count = max(packed_pair_to_count.items(), key = packed_pair_to_count_cmp) # sort by frequency of occurence, then lexigraphical greatness
            p1, p2 = unpack_pair(packed_pair)
            new_vocab_idx = len(self._vocab)
            print(f"({p1}, {p2}) -> {new_vocab_idx}: {Word([p1]), Word([p2])}, {count}")

            for w_idx in self._packed_pair_to_word_idxes[packed_pair]:
                assert (p1, p2) in [(_p1, _p2) for (_p1, _p2) in zip(self._words[w_idx].toks[:-1], self._words[w_idx].toks[1:])], f"word: {self._words[w_idx]}, word.toks: {self._words[w_idx].toks}"
                
            self._vocab[new_vocab_idx] = self._vocab[p1] + self._vocab[p2]
            self._merges[(self._vocab[p1], self._vocab[p2])] = new_vocab_idx
            self.merge(packed_pair, new_vocab_idx, packed_pair_to_count)
        
        pr_time(st2, f"All {i} merge operations")

    def encode(text : str) -> list[int]:
        ...

    def decode(self, tokens : Union[list[int], int]) -> str:
        if isinstance(tokens, (int)):
            if tokens < len(self.special_tokens):
                return self.special_tokens[tokens]
            elif tokens < len(self._orig_vocab):
                return chr(int.from_bytes(self._vocab[tokens]))
            else:
                return self._vocab[tokens].decode("utf-8")
        else:
            return ''.join([self.decode(c) for c in tokens])

if __name__ == "__main__":
    test_filepath = "/Users/jeffreywolberg/Coding/cs336/assignment1-basics/data/test_training_data.txt"

    bpe = BPETokenizer(special_tokens=[])
    bpe.train(test_filepath, vocab_size=275)

    for i in range(250, len(bpe._vocab)):
        print(i, '->', bpe.decode([i]), "\t", len(bpe.decode([i])))

    print(bpe._merges)