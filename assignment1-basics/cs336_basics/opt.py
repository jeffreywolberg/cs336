from typing import Callable, Optional, Tuple
import torch
from torch.optim.optimizer import ParamsT

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr = 1.e-3, betas : Tuple[float, float] = (0.9, 0.999), eps=1.e-8, weight_decay=1.e-3) -> None:
        b1, b2 = betas
        defaults = {'lr': lr, 'b1' : b1, 'b2': b2, 'eps': eps, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def step(self, closure : Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1 = group["b1"]
            b2 = group["b2"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p] 
                t = param_state.get("t", 1) 
                grad = p.grad.data
                param_state['m'] = b1 * param_state.get('m', torch.tensor([0.0])) + (1 - b1) * grad
                param_state['v'] = b2 * param_state.get('v', torch.tensor([0.0])) + (1 - b2) * grad * grad
                lr_t = lr * torch.sqrt(torch.tensor([1 - b2 ** t])) / (1 - b1 ** t)
                p.data -= lr_t * param_state['m'] / (torch.sqrt(param_state['v']) + eps)
                p.data -= lr * weight_decay * p.data
                param_state["t"] = t + 1