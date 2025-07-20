from abc import ABC
import regex as re
from collections import Counter, defaultdict
from typing import Iterator, Union
import numpy as np
from time import time

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

PREHEAD_TN = TokenNode(-1, can_pair_forward=False)
def linked_list_head() -> TokenNode:
    return PREHEAD_TN.next

class BPETokenizer(Tokenizer):
    def __init__(self, special_tokens=[]):
        # special tokens + byte values (256 possible ones)
        self.special_tokens = special_tokens
        self._orig_vocab = {i: tok.encode("utf-8") for i, tok in enumerate(self.special_tokens)}
        self._orig_vocab.update({i+len(self.special_tokens) : bytes([i]) for i in range(256)}) # index to bytes
        self._vocab = dict(self._orig_vocab)
        self._merges : dict[tuple[bytes, bytes], int] = {} # (bytes1, bytes2) -> new_vocab_idx

    def strip_special_tokens(self, text: str) -> str:
        special_tokens = [re.escape(tok) for tok in self.special_tokens]
        pat = "|".join(special_tokens)
        return ''.join(re.split(pat, text))

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

    def pretokenize(self, text : str) -> Iterator[re.Match[str]]:
        # Pretokenizer splits a block of text into smaller chunks
        return re.finditer(PAT, text)

    # def pretokenize(self, text : str) -> list[str]:
    #     # Pretokenizer splits a block of text into smaller chunks
    #     toks = re.findall(PAT, text)
    #     return toks
    
    def prepare_token_node_linked_list(self, train_data):
        # pretokenized_train_data : list[str] = self.pretokenize(train_data)

        pretokenized_train_data : Iterator[re.Match[str]] = self.pretokenize(train_data)

        vocab_idx_to_nodes = defaultdict(list)
        cur_tn = PREHEAD_TN
        for word in pretokenized_train_data:
            if isinstance(word, re.Match):
                word = train_data[word.start() : word.end()]
            byte_stream : list[int] = [b + len(self.special_tokens) for b in word.encode("utf-8")]
            for byte_idx, vocab_idx in enumerate(byte_stream):
                tn = TokenNode(vocab_idx)
                vocab_idx_to_nodes[vocab_idx].append(tn)
                cur_tn.next = tn
                tn.prev = cur_tn
                tn.can_pair_forward = byte_idx != len(byte_stream) - 1 # last byte of each pretokenized word cannot be paired with next
                cur_tn = tn

        return vocab_idx_to_nodes
    
    def compute_count(self) -> tuple[tuple[int, int], tuple[int, list[TokenNode]]]:
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
        
        return pair_to_pair_info

    def merge(self, pair_st_nodes : list[TokenNode], new_vocab_idx : int, pair_to_pair_info : dict[tuple[int, int], tuple[int, list[TokenNode]]]):
        if len(pair_st_nodes) == 0:
            return
        
        pair_to_merge = (pair_st_nodes[0].vocab_idx,  pair_st_nodes[0].next.vocab_idx)
        is_pair_repeated_token = pair_to_merge[0] == pair_to_merge[1]

        for i, n in enumerate(pair_st_nodes):
            assert n.vocab_idx == pair_to_merge[0]
            assert n.next is not None, f"Pair with start node {n} must have non-none 'next' field"
            assert n.next.vocab_idx == pair_to_merge[1]
            # Handle case of repeated token that is trying to be merged (e.g. (49, 49))
            # Need to ensure that the pairs start nodes that are merged are not consecutive, otherwise when merging in 
            # e.g. if pair_to_merge is (49, 49) for seq [49, 49, 49, 49, 50, 51], the merge operation should perform 
            # two merges, one on node at idx0 and one at node at idx2. Need to skip merging at idx 1 and 3, since they will be deleted.
            if is_pair_repeated_token:
                is_prev_editable = i > 0 and pair_st_nodes[i-1] == n.prev
                is_dprev_editable = i > 1 and pair_st_nodes[i-2] == n.prev.prev
                if is_prev_editable and not is_dprev_editable:
                    continue
            
            new_node = TokenNode(new_vocab_idx)

            prev_pair = (n.prev.vocab_idx, n.vocab_idx)
            
            # edit counts to reflect merge operation below
            if n.prev.can_pair_forward:
                pair_to_pair_info[prev_pair][0] -= 1
                pair_to_pair_info[prev_pair][1].remove(n.prev)
                
                new_pair_left = (n.prev.vocab_idx, new_vocab_idx)
                pair_to_pair_info[new_pair_left][0] += 1
                pair_to_pair_info[new_pair_left][1].append(n.prev)
            
            if n.next.next is not None and n.next.can_pair_forward:
                next_pair = (n.next.vocab_idx, n.next.next.vocab_idx)
                pair_to_pair_info[next_pair][0] -= 1
                pair_to_pair_info[next_pair][1].remove(n.next)
            
                new_pair_right = (new_vocab_idx, n.next.next.vocab_idx)
                pair_to_pair_info[new_pair_right][0] += 1
                pair_to_pair_info[new_pair_right][1].append(new_node)

            # perform merge
            n.prev.next = new_node
            new_node.prev = n.prev
            new_node.can_pair_forward = n.next.can_pair_forward
            if n.next.next is not None: # the pair is not at the very end of the list
                new_node.next = n.next.next
                new_node.next.prev = new_node
        
        pair_to_pair_info.pop(pair_to_merge)

    def train(self, train_data : str, vocab_size : int):
        self._vocab = dict(self._orig_vocab)
        assert len(self._vocab) <= vocab_size, f"len(_vocab) {len(self._vocab)} must be <= vocab_size: {vocab_size}"

        st1 = time()
        train_data = self.strip_special_tokens(train_data)
        self.prepare_token_node_linked_list(train_data)
        pr_time(st1, "pretokenization + preparing linked list")
        
        pair_to_pair_info = self.compute_count()
        
        niters = vocab_size - len(self._vocab)
        st2 = time()
        for i in range(niters):
            if len(pair_to_pair_info) == 0:
                break
            
            # tuple[int, int], tuple[int, list[TokenNode]]
            pair_to_merge, (nseen, nodes) = max(pair_to_pair_info.items(), key = lambda kv :  (kv[1][0], self._vocab[kv[0][0]], self._vocab[kv[0][1]])) # sort by frequency of occurence
            assert nseen == len(nodes)
            new_vocab_idx = len(self._vocab)
            self._vocab[new_vocab_idx] = self._vocab[pair_to_merge[0]] + self._vocab[pair_to_merge[1]]
            self._merges[(self._vocab[pair_to_merge[0]], self._vocab[pair_to_merge[1]])] = new_vocab_idx
            self.merge(nodes, new_vocab_idx, pair_to_pair_info)
        
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
    train_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest ûÿ"
    # train_text = "000000000"
    # train_text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
    bpe = BPETokenizer(special_tokens=[])
    bpe.train(train_text, vocab_size=275)

    for i in range(250, len(bpe._vocab)):
        print(i, '->', bpe.decode([i]), "\t", len(bpe.decode([i])))

    print(bpe._merges[:7])