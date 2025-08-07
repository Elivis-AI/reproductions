import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DDPM:
    def __init__(self, n_steps, device='cpu'):
        self.n_steps = n_steps  # 设置步长
        self.device = device

        self.betas = torch.linspace(-6, 6, n_steps).float().to(device) # 逐渐递增
        self.betas = torch.sigmoid(self.betas)*(0.5e-2 - 1e-5)+1e-5  # β0,β1,...,βt
        self.alphas = 1 - self.betas  # αt = 1 - βt
        self.alphas_sqrt = self.alphas.sqrt()
        self.alphas_prod = self.alphas.cumprod(dim=0)  # αt^ = αt的累乘
        # αt^往右平移一位, 原第t步的值为第t-1步的值, 第0步补1
        self.alphas_prod_p = F.pad(self.alphas_prod[1:], (1, 0), value=1.)  # αt-1 ^
        self.alphas_bar_sqrt = self.alphas_prod.sqrt()  # sqrt(αt^)
        self.one_minus_alphas_bar_log = torch.log(1. - self.alphas_prod)  # log(1-αt^)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1. - self.alphas_prod)  # sqrt(1-αt^)

    def forward_step(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x_0: (B, C, H, W), t: (B,)"""
        t = t.view(-1, *[1] * (x_0.ndim - 1)) # match dims
        noise = torch.randn_like(x_0)  # noise为从正态分布中采样的随机噪声
        x_t = self.alphas_bar_sqrt[t] * x_0 + self.one_minus_alphas_bar_sqrt[t] * noise
        return x_t, noise

    def reverse_step(self, x_t: torch.Tensor, eps_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x_t: (B, C, H, W), eps_t: (B, C, H, W), t: (B,)"""
        t = t.view(-1, *[1] * (x_t.ndim - 1)) # match dims
        coeff = self.betas[t] / self.one_minus_alphas_bar_sqrt[t]  # 模型输出的系数：βt/根号下(1-αt^) = 1-αt/根号下(1-αt^)
        # 见algorithm-2
        mean = (1 / self.alphas_sqrt[t]) * (x_t - (coeff * eps_t))   
        z = torch.randn_like(x_t)  # 对应公式中的 z
        sigma_t = self.betas[t].sqrt()  # 对应公式中的 σt 
        return mean + sigma_t * z 


if __name__ == '__main__':
    ddpm = DDPM(n_steps=100)
    x_0 = torch.randn(2, 3, 32, 32)  # 假设输入图像为2张3通道32x32的图像
    t = torch.tensor([0, 50])  # 假设我们要在第0步和第50步进行前向和反向步骤

    x_t, noise = ddpm.forward_step(x_0, t)
    print("Forward Step Output:", x_t.shape, noise.shape)

    eps_t = torch.randn_like(x_t)  # 模拟模型输出的噪声
    x_reversed = ddpm.reverse_step(x_t, eps_t, t)
    print("Reverse Step Output:", x_reversed.shape)