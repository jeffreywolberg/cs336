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

        self.W = nn.Parameter(torch.empty((self.out_features, self.in_features), device=self.device, dtype=self.dtype))
        std = math.sqrt(2/(self.in_features + self.out_features))
        nn.init.trunc_normal_(self.W, 0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor):
        return einx.dot("out in, b ... in -> b ... out", self.W, x)
        
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.W = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device=self.device, dtype=self.dtype))
        std = 1
        nn.init.trunc_normal_(self.W, 0, std=std, a=-3*std, b=3*std)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # (batch_size, sequence_length)
        return self.W[token_ids]

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
