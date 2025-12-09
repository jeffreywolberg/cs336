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
        