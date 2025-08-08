# copy from DDPM/ddpm.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DDPM:
    def __init__(self, gamma: torch.Tensor):
        self.device = gamma.device
        # gamma 直接控制噪声水平
        self.gamma = gamma
        self.p = torch.sqrt(gamma)
        self.q = torch.sqrt(1 - gamma)
        # gamma 反推 alpha beta
        gamma = F.pad(gamma, (1, 0), value=1.)
        self.alpha = gamma[1:] / gamma[:-1]
        self.beta = 1 - self.alpha
        self.alpha_sqrt = torch.sqrt(self.alpha)
        self.beta_sqrt = torch.sqrt(self.beta)

    def forward_step(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_0: (B, ...), t: (B,)"""
        t = t.view(-1, *[1] * (x_0.ndim - 1)) # match dims
        noise = torch.randn_like(x_0)  # noise为从正态分布中采样的随机噪声
        x_t = self.p[t] * x_0 + self.q[t] * noise
        return x_t, noise

    def reverse_step(self, x_t: torch.Tensor, eps_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t: (B, ...), eps_t: (like x_t), t: (B,)"""
        t = t.view(-1, *[1] * (x_t.ndim - 1)) # match dims
        coeff = self.beta[t] / self.q[t]
        # 见algorithm-2
        mu = (1 / self.alpha_sqrt[t]) * (x_t - (coeff * eps_t))
        z = torch.randn_like(x_t)
        return mu + self.beta_sqrt[t] * z 


# https://arxiv.org/abs/2102.09672
# http://arxiv.org/abs/2301.10972
# control gamma(alpha_bar, \Pi_i alpha_i), not beta
def constant_beta_schedule(n_steps: int, val: float = None, device='cpu'):
    if val is None:
        val = 1 - math.exp(-8 / n_steps)
    beta = torch.full((n_steps,), val, device=device)
    gamma = torch.cumprod(1 - beta, dim=0)
    return gamma

def linear_schedule(n_steps: int, min: float = 1e-6, max: float = 1., device='cpu'):
    return torch.linspace(min, max, n_steps, device=device).flip(0)  # 从max到min

def sigmoid_schedule(n_steps: int, min: float = 1e-8, max: float = 1., device='cpu'):
    t = torch.linspace(-8, 8, n_steps, device=device)
    return torch.sigmoid(t) * (min - max) + max

def cosine_schedule(
        n_steps: int, 
        min: float = 1e-6,
        max: float = 1.,
        offset: float = 8e-3, 
        exp: int = 2, 
        device='cpu'
    ):
    t = torch.arange(n_steps, device=device)
    gamma = torch.cos((t / (n_steps + 1) + offset) / (1 + offset) * torch.pi / 2) ** exp
    gamma = torch.clip(gamma, min, max)
    return gamma

def sqrt_schedule(
        n_steps: int,
        min: float = 1e-6, 
        max: float = 1., 
        offset = 8e-3, 
        device='cpu'
    ):
    t = torch.linspace(min, max, n_steps, device=device)
    gamma = 1 - torch.sqrt(t + offset)
    gamma = torch.clip(gamma, min, max)
    return gamma


if __name__ == '__main__':
    betas = linear_schedule(100)
    ddpm = DDPM(betas)
    x_0 = torch.randn(2, 3, 32, 32)  # 假设输入图像为2张3通道32x32的图像
    t = torch.tensor([0, 50])  # 假设我们要在第0步和第50步进行前向和反向步骤

    x_t, noise = ddpm.forward_step(x_0, t)
    print("Forward Step Output:", x_t.shape, noise.shape)

    eps_t = torch.randn_like(x_t)  # 模拟模型输出的噪声
    x_reversed = ddpm.reverse_step(x_t, eps_t, t)
    print("Reverse Step Output:", x_reversed.shape)