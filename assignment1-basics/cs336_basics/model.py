import math
from typing import Optional
import einx
import torch
import torch.nn as nn

from cs336_basics.tokenizer import BPETokenizer

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), device=self.device, dtype=self.dtype))
        std = math.sqrt(2/(self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, 0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor):
        """classic matmul over the data (x)"""
        return einx.dot("out [in], b ... [in] -> b ... out", self.weight, x)
        
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device=self.device, dtype=self.dtype))
        std = 1
        nn.init.trunc_normal_(self.weight, 0, std=std, a=-3*std, b=3*std)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Lookup table of embeddings based on token"""
        # token_ids: (batch_size, sequence_length)
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.ones(d_model, device=self.device, dtype=self.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalization of embedding such that result normalized input with gain across each element of d_model"""
        # x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_norm = torch.sqrt(1 / self.d_model * einx.sum('b s [d] -> b s 1', torch.square(x)) + self.eps)
        
        result = self.weight * x / rms_norm

        return result.to(in_dtype)

class SwiGLUFNN(nn.Module):
    def __init__(self, d_model : int, d_ff : int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.device = device
        self.dtype = dtype
        self.w1 = Linear(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=self.device, dtype=self.dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=self.device, dtype=self.dtype)

    def SiLU(self, x : torch.Tensor):
        # x is B, S, D
        return x * torch.sigmoid(x)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """SiLU is an activation function similar to ReLU, but smooth at zero. (w1)
        GLU is an elementwise product of a linear transformation (w1) passed through sigmoid (or SiLU in our case) and another linear transformation (w3). σ(W1x) ⊙ W2x,
        The combination is SwiGLU. The linear transformations used during SwiGLU project from d_model to d_ff.
        Finally, project this output (size d_ff) and project it to d_model"""
        # x is B, S, D
        return self.w2(self.SiLU(self.w1(x)) * self.w3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        K = self.d_k // 2
        thetas = torch.arange(self.max_seq_len)[:, None] / (self.theta ** (2 * torch.arange(K)[None, :] / self.d_k)) # S, K
        thetas = thetas.to(self.device)
        self.c, self.s = torch.cos(thetas), torch.sin(thetas)
        

        # Rs = torch.stack([torch.stack([self.c, -self.s]), torch.stack([self.s, self.c])]) # 2x2xSxK
        # R = torch.zeros((self.max_seq_len, self.d_k, self.d_k))
        # for i in range(self.max_seq_len):
        #     rots = Rs[:, :, i, :].permute(2, 0, 1) # Kx2x2
        #     R[i] = torch.block_diag(*rots) # DxD
        # self.register_buffer('R', R, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Rotate input sequence using an angle proportional to each embedding's position within the input sequence [0, S) and dimension index [0, d_k).
        Absolute positional information is encoded by rotating the embedding by an angle proportional to its position (given by token_positions).
        Moreover, within an embedding, low dimensions rotate slowly, higher dimensions rotate faster. This allows attention layers to recognize relative offsets of position at different granularities (e.g. 10, 100, 1000 tokens apart)
        by looking at different indices within the embedding dimension. e.g. looking at one token offset would require looking at higher dimension indices while looking at 1000 token relative offset would require looking at lower dimension indices.
        This allows earlier dimensions to capture global, slowly changing trends, and higher dimensions
        to capture quicker, local trends.
        """
        # x (..., seq_len, d_k)
        # token_positions (..., seq_len)

        # S, K
        c, s = self.c[token_positions], self.s[token_positions]
        # B, S, K
        x0 = x[..., ::2]
        x1 = x[..., 1::2]

        out = torch.zeros_like(x) # B, S, D

        # Both methods work
        # out[..., ::2] = einx.dot('B S K, S K -> B S K', x0, c) + einx.dot('B S K, S K -> B S K', x1, -s)
        # out[..., 1::2] = einx.dot('B S K, S K -> B S K', x0, s) + einx.dot('B S K, S K -> B S K', x1, c)
        out[..., ::2] = x0 * c + x1 * -s
        out[..., 1::2] = x0 * s + x1 * c
        
        return out
    
def softmax(x : torch.Tensor, dim : int, temperature : float = 1):
    """
    produce a valid probability distribution by raising e to each logit, and normalizing across all values within this dimension.
    """
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - max_val
    x_shifted_exp = torch.exp(x_shifted / temperature)
    return x_shifted_exp / torch.sum(x_shifted_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(keys : torch.Tensor, queries : torch.Tensor, values : torch.Tensor, mask=None):
    """
    Attend each key to the queries to determine the weighting of each value associated with each key.
    Keys that highly attend to a query will have the key's associated value weighted highly in the resulting weighted combination of the values.
    Optionally, a mask can be provided to disallow certain keys to be attended to for each query. e.g. each key can only attend to queries <= its position.

    # Helpful 3b1b Attention video: https://youtu.be/eMlx5fFNoYc?si=Vxmp6Ghf_GDOqFxL
    e.g. in the sentence 'The big brown fox jumped', a query representing the question 'Are there adjectives describing my embedding?' would have the embeddings for the keys of 'big' and 'brown'
    highly attend to it. This would cause the values for 'big' and 'brown' to be highly represented in the resulting output value of 'fox' for the given query. 
    """
    # keys (B, ..., S, d_k)
    # queries (B, ..., S, d_k)
    # values (B, ..., S, d_v)
    # mask = Optional[(B, ..., S, S)]

    
    if len(keys.shape) == 3:
        keys = keys[:, None, ...]
        queries = queries[:, None, ...]
        values = values[:, None, ...]
        mask = mask[:, None, ...]
        is_3d = True
    else:
        is_3d = False

    qk_prod = einx.dot('B h Sq [d_k], B h Sk [d_k] -> B h Sq Sk', queries, keys)
    if mask is not None:
        qk_prod = qk_prod + torch.where(mask, 0.0, -torch.inf)
    d_k = keys.shape[-1]
    attention_vals = softmax(qk_prod / math.sqrt(d_k), dim=3)
    output = einx.dot('B h Sq [Sk], B h [Sk] d_v -> B h Sq d_v', attention_vals, values)

    if is_3d:
        output = output.squeeze(1)

    return output

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model : int, num_heads : int, rope : Optional[RotaryPositionalEmbedding] = None, device = None) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.rope = rope if rope is not None else lambda x, y: x
        self.device = device

        self.d_k = self.d_model // self.num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k, device=self.device)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k, device=self.device)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v, device=self.device)
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model, device=self.device)

    def forward(self, in_features : torch.Tensor, token_positions : Optional[torch.Tensor] = None):
        """
        Run multi-head attention num_head times, producing a value map of d_v * num_heads.
        Then, project this matrix to d_model so resume its flow in the network (e.g. it is added to the input as a residual).
        """
        
        # in_features (... sequence_length d_out)
        head_outs = []

        B, S, d_out = in_features.shape
        assert d_out == self.d_model

        keys = einx.dot('Dproj [Dmodel], B S [Dmodel] -> B S Dproj', self.k_proj.weight, in_features)
        queries = einx.dot('Dproj [Dmodel], B S [Dmodel] -> B S Dproj', self.q_proj.weight, in_features)
        values = einx.dot('Dproj [Dmodel], B S [Dmodel] -> B S Dproj', self.v_proj.weight, in_features)
        mask = torch.tril(torch.ones((B, S, S), dtype=torch.bool, device=self.device))

        for i in range(self.num_heads):
            keys_slice = self.rope(keys[..., i*self.d_k:(i+1)*self.d_k], token_positions)
            queries_slice = self.rope(queries[..., i*self.d_k:(i+1)*self.d_k], token_positions)
            values_slice = values[..., i*self.d_v:(i+1)*self.d_v]
            # B S d_v
            head_out = scaled_dot_product_attention(keys_slice, queries_slice, values_slice, mask=mask)
            head_outs.append(head_out)

        multihead_output = torch.concat(head_outs, dim=2) # B S (d_v*num_heads)

        out_proj = einx.dot('DModel [DProj], B S [DProj] -> B S DModel', self.output_proj.weight, multihead_output)

        return out_proj

class TransformerBlock(nn.Module):
    def __init__(self, d_model : int, num_heads : int, d_ff : int, theta : float, max_seq_len : int, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        self.ln1 = RMSNorm(self.d_model, device=self.device)
        self.ln2 = RMSNorm(self.d_model, device=self.device)

        self.rope = RotaryPositionalEmbedding(theta, self.d_model//self.num_heads, self.max_seq_len, device=self.device)

        self.attn = MultiheadSelfAttention(self.d_model, self.num_heads, self.rope, device=self.device)

        self.ffn = SwiGLUFNN(self.d_model, self.d_ff, device=self.device)

    def forward(self, x : torch.Tensor):
        # x (B, S, d)
        B, S, _ = x.shape
        token_positions = torch.arange(0, S, device=x.device)[None].tile((B, 1))
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.ffn(self.ln2(x))

        return x

class TransformerLM(nn.Module):
    def __init__(self, vocab_size : int, context_length : int, d_model : int, num_layers : int, num_heads : int, d_ff : int, rope_theta : float, device=None) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.device = device

        self.token_embeddings = Embedding(self.vocab_size, self.d_model, self.device)
        layers = [TransformerBlock(self.d_model, self.num_heads, self.d_ff, self.rope_theta, self.context_length, device=self.device) for i in range(self.num_layers)]
        self.layers = nn.Sequential(*layers)
        self.ln_final = RMSNorm(self.d_model, device=self.device)
        self.lm_head = Linear(self.d_model, self.vocab_size, device=self.device)

    
    def forward(self, tokens : torch.Tensor):
        x = self.token_embeddings(tokens)
        for transformer_layer in self.layers:
            x = transformer_layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)

        return x

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, scheduler : torch.optim.lr_scheduler.LRScheduler, iteration : int, out):
    data = {
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration
    }
    torch.save(data, out)

def load_checkpoint(src, model: nn.Module, optimizer: torch.optim.Optimizer, scheduler : Optional[torch.optim.lr_scheduler.LRScheduler], map_location : str):
    data = torch.load(src, map_location=map_location)
    model.load_state_dict(data['model'])
    optimizer.load_state_dict(data['opt'])
    if scheduler is not None:
        scheduler.load_state_dict(data['scheduler']) # epoch is really optimizer step

    iteration = data['iteration']
    print(f"Successfully loaded ckpt from {src}")
    return iteration
    