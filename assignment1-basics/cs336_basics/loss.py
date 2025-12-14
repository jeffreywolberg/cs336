import torch
from torch import nn, Tensor

from jaxtyping import Float, Int

from cs336_basics.model import softmax

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
        # inputs B, V
        # targets B

        B, V = inputs.shape
        inputs_shifted = inputs - torch.max(inputs, dim=1, keepdim=True).values
        loss = -(inputs_shifted[torch.arange(B), targets] - torch.logsumexp(inputs_shifted, dim=1))
        return torch.mean(loss)