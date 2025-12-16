
from typing import Optional
import torch

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import BPETokenizer

def sample(probs : torch.Tensor, p_sample : float):
    B, S, V = probs.shape
    assert B == 1, "Only handle batch_size = 1 for now"
    probs = probs[0, -1] 
    n_above = torch.sum(probs >= p_sample)
    vals, args = torch.topk(probs, k=n_above)
    if len(args) == 0:
        return torch.randint(0, V, (1,)).item()

    denom = torch.sum(vals).item()
    new_probs = probs[args] / denom
    tok = args[new_probs.multinomial(1)].item()

    return tok

def generate_text(model : TransformerLM, tokenizer : BPETokenizer, prompt : list[int], max_generated_tokens : Optional[int] = None, temperature : float = 1, p_sample : float = 0.0) -> list[int]:
    special_tokens = set(tokenizer.special_tokens)
    print(f"Generating text...")
    last_generated_tok = None
    TOKS_BUFFER_SIZE = max(8192, len(prompt)) # +1 to fill in room for next tok
    toks_empty_buffer = torch.zeros((TOKS_BUFFER_SIZE), dtype=torch.int32, device=model.device)
    toks = torch.zeros_like(toks_empty_buffer)
    toks[:len(prompt)] = torch.tensor(prompt, dtype=torch.int32, device=model.device)
    token_i = len(prompt)
    
    model.eval()
    while (max_generated_tokens is not None and token_i < max_generated_tokens + len(prompt)) or (last_generated_tok is not None and tokenizer.decode([last_generated_tok]) in special_tokens):
        if token_i == len(toks) == 0:
            toks = torch.cat([toks, toks_empty_buffer], dim=0)
        with torch.no_grad():
            next_tok_logits = model(toks[:token_i][None]) # 1, token_i, vocab
            next_tok_probs = softmax(next_tok_logits, dim=2, temperature=temperature)
            last_generated_tok = sample(next_tok_probs, p_sample)
            toks[token_i] = last_generated_tok
            token_i += 1

    return tokenizer.decode(toks[:token_i].cpu().tolist())
