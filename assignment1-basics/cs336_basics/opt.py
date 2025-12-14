import math
from typing import Callable, Iterable, Optional, Tuple
import torch
from torch.optim.optimizer import ParamsT

def get_lr_with_cosine_sched(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,) -> float:

    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return min_learning_rate + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate

def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1.e-6):
    grad_sum_sq = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad_sum_sq += torch.sum(torch.square(p.grad))
    grad_norm = math.sqrt(grad_sum_sq)
    if grad_norm > max_l2_norm:
        factor = max_l2_norm / (grad_norm + eps)
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.mul_(factor)
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