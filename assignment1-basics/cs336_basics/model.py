import math
import einx
import torch
import torch.nn as nn

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
        return einx.dot("out in, b ... in -> b ... out", self.weight, x)
        
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
        # (batch_size, sequence_length)
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
        # (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_norm = torch.sqrt( (1/self.d_model) * einx.sum('b s [d] -> b s 1', torch.square(x)) + self.eps)
        
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
        self.c, self.s = torch.cos(thetas), torch.sin(thetas)

        # Rs = torch.stack([torch.stack([self.c, -self.s]), torch.stack([self.s, self.c])]) # 2x2xSxK
        # R = torch.zeros((self.max_seq_len, self.d_k, self.d_k))
        # for i in range(self.max_seq_len):
        #     rots = Rs[:, :, i, :].permute(2, 0, 1) # Kx2x2
        #     R[i] = torch.block_diag(*rots) # DxD
        # self.register_buffer('R', R, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
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
    
def softmax(x : torch.Tensor, dim : int):
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_shifted = x - max_val
    x_shifted_exp = torch.exp(x_shifted)
    return x_shifted_exp / torch.sum(x_shifted_exp, dim=dim, keepdim=True)