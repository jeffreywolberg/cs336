from abc import ABC
import regex as re
from collections import Counter, defaultdict
from typing import Iterator, Union
import numpy as np
from time import time
from heapdict import heapdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""" # pattern


def pr_time(st : float, name=""):
    print(f"{name} took {round(time() - st, 3)} seconds")

class Tokenizer(ABC):
    def train(self):
        ...

    def encode(text : str) -> list[int]:
        ...

    def decode(tokens : list[int]) -> str:
        ...


class TokenNode:
    def __init__(self, vocab_idx, can_pair_forward=True):
        self.vocab_idx = vocab_idx
        self.prev : TokenNode = None
        self.next : TokenNode = None
        self.can_pair_forward = can_pair_forward # set false for splits due to pre-tokenization

    def __repr__(self):
        return f"TokenNode({self.vocab_idx}, prev={id(self.prev) if self.prev is not None else None}, next={id(self.next) if self.next is not None else None}, can_pair_forward={self.can_pair_forward})"

HEAD_TN = TokenNode(-1, can_pair_forward=False)
def linked_list_head() -> TokenNode:
    return HEAD_TN.next

class BPETokenizer(Tokenizer):
    def __init__(self, special_tokens=[]):
        # special tokens + byte values (256 possible ones)
        self.special_tokens = special_tokens
        self._orig_vocab = {i: tok.encode("utf-8") for i, tok in enumerate(self.special_tokens)}
        self._orig_vocab.update({i+len(self.special_tokens) : bytes([i]) for i in range(256)}) # index to bytes
        self._vocab = dict(self._orig_vocab)
        self._merges : dict[tuple[bytes, bytes], int] = {} # (bytes1, bytes2) -> new_vocab_idx

    def strip_special_tokens(self, text: str) -> str:
        pat = "|".join(self.special_tokens)
        return ''.join(re.split(pat, text))
        
        for tok in self.special_tokens:
            text = text.replace(tok, '')
        return text

    def assert_pair_is_removed(self, vocab_idxes : tuple[int, int]):
        cur = linked_list_head()
        i = 0
        while cur.next is not None:
            # print(cur)
            assert (cur.vocab_idx, cur.next.vocab_idx) != vocab_idxes, f"On token {i}, pair {vocab_idxes} was found in nodes {cur, cur.next}"
            cur = cur.next
            i += 1
        # print(cur)

    def assert_special_tokens_removed(self, train_data : str):
        for stok in self.special_tokens:
            assert stok not in train_data

    # def pretokenize(self, text : str) -> Iterator[re.Match[str]]:
    #     # Pretokenizer splits a block of text into smaller chunks
    #     return re.finditer(PAT, text)

    def pretokenize(self, text : str) -> list[str]:
        # Pretokenizer splits a block of text into smaller chunks
        toks = re.findall(PAT, text)
        return toks
    


    def prepare_token_node_linked_list(self, train_data):
        st1 = time()
        pretokenized_train_data : list[str] = self.pretokenize(train_data)
        # pretokenization_bounds = [d.start() for d in pretokenized_train_data] + [len(train_data)]
        # pr_time(st1, "pretokenization")

        vocab_idx_to_nodes = defaultdict(list)
        cur_tn = HEAD_TN
        for word_idx, word in enumerate(pretokenized_train_data):
            byte_stream : list[int] = [b + len(self.special_tokens) for b in word.encode("utf-8")]
            for byte_idx, vocab_idx in enumerate(byte_stream):
                tn = TokenNode(vocab_idx)
                vocab_idx_to_nodes[vocab_idx].append(tn)
                cur_tn.next = tn
                tn.prev = cur_tn
                tn.can_pair_forward = byte_idx != len(byte_stream) - 1 # last byte of each pretokenized word cannot be paired with next
                cur_tn = tn

        return vocab_idx_to_nodes
    
    def retrieve_count(self) -> tuple[tuple[int, int], tuple[int, list[TokenNode]]]:
        # (vocab_idx1, vocab_idx2) -> (count, pair_start_nodes)
        pair_to_pair_info : dict[tuple[int, int], tuple[int, list[TokenNode]]] = defaultdict(lambda : [0, []])
        cur_node = linked_list_head()
        i = 1
        while cur_node.next is not None:
            if cur_node.can_pair_forward:
                pair = (cur_node.vocab_idx, cur_node.next.vocab_idx)
                pair_to_pair_info[pair][0] += 1
                pair_to_pair_info[pair][1].append(cur_node)
            cur_node = cur_node.next
            i += 1
        # print(f"Counted {i+1} total tokens in sequence")

        if len(pair_to_pair_info) == 0:
            return (None, None), (None, None)
        
        return max(pair_to_pair_info.items(), key = lambda kv : (kv[1][0], self._vocab[kv[0][0]], self._vocab[kv[0][1]])) # sort by frequency of occurence

    # NOTE: using this merge function does not work, even though I cannot figure out how it is different from correct implementation
    # def merge(self, nodes_to_edit : list[TokenNode], new_vocab_idx : int):
    #     for n in nodes_to_edit:
    #         new_node = TokenNode(new_vocab_idx)
    #         n.prev.next = new_node
    #         new_node.prev = n.prev
    #         assert n.next is not None, f"Pair with start node {n} must have non-none 'next' field"
    #         new_node.can_pair_forward = n.next.can_pair_forward
    #         if n.next.next is not None: # the pair is not at the very end of the list
    #             new_node.next = n.next.next
    #             new_node.next.prev = new_node

    def merge(self, pair_to_merge : tuple[int, int], new_vocab_idx : int):
        n = linked_list_head()
        i = 0
        while n.next is not None:
            if (n.vocab_idx, n.next.vocab_idx) == pair_to_merge and n.can_pair_forward:
                new_node = TokenNode(new_vocab_idx)
                n.prev.next = new_node
                new_node.prev = n.prev
                assert n.next is not None, f"Pair with start node {n} must have non-none 'next' field"
                new_node.can_pair_forward = n.next.can_pair_forward
                if n.next.next is not None: # the pair is not at the very end of the list
                    new_node.next = n.next.next
                    new_node.next.prev = new_node
                n = new_node.next
            else:
                n = n.next
            i += 1


    def train(self, train_data : str, vocab_size : int):
        self._vocab = dict(self._orig_vocab)
        assert len(self._vocab) <= vocab_size, f"len(_vocab) {len(self._vocab)} must be <= vocab_size: {vocab_size}"

        # pretokenized_train_data : Iterator[re.Match[str]] = self.pretokenize(train_data)
        train_data = self.strip_special_tokens(train_data)
        self.assert_special_tokens_removed(train_data)
        vocab_idx_to_nodes = self.prepare_token_node_linked_list(train_data)
        
        niters = vocab_size - len(self._vocab)
        el1 = el2 = 0
        for i in range(niters):
            st1 = time()
            # tuple[int, int], tuple[int, list[TokenNode]]
            pair_to_merge, (nseen, nodes) = self.retrieve_count()
            # print(pair_to_merge, nseen)
            if pair_to_merge == (None, None):
                return
            # el1 += time() - st1
            # st1 = time()
            # # pair_to_merge : tuple[int, int] = next(iter(pair_to_info.keys()))
            # pair_to_merge : tuple[int, int] = list(pair_to_info.keys())[0]
            # nseen, nodes = pair_to_info.pop(pair_to_merge)
            assert nseen == len(nodes)
            new_vocab_idx = len(self._vocab)
            self._vocab[new_vocab_idx] = self._vocab[pair_to_merge[0]] + self._vocab[pair_to_merge[1]]
            self._merges[(self._vocab[pair_to_merge[0]], self._vocab[pair_to_merge[1]])] = new_vocab_idx
            st2 = time()
            # NOTE: using this merge function does not work, even though I cannot figure out how it is different from correct implementation
            # self.merge(nodes, new_vocab_idx)
            self.merge(pair_to_merge, new_vocab_idx)
            self.assert_pair_is_removed(pair_to_merge)
            el2 += time() - st2

        print(f"Counting took {round(el1, 2)} seconds")
        print(f"Merging took {round(el2, 2)} seconds")


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
    train_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest ûÿ"
    # train_text = "000000000"
    # train_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    bpe = BPETokenizer(special_tokens=[])
    bpe.train(train_text, vocab_size=275)

    for i in range(250, len(bpe._vocab)):
        print(i, '->', bpe.decode([i]), "\t", len(bpe.decode([i])))

    print(bpe._merges[:7])