import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from config import DiTConfig
from model import SelfAttention, FeedForward


class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.attn = SelfAttention(dim, n_heads)
        self.ffn = FeedForward(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.w_cond = nn.Linear(dim, dim * 6)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """ x: (B, L, D), cond: (B, D)"""
        cond = self.w_cond(cond)
        alpha1, alpha2, gamma1, gamma2, beta1, beta2 = cond.unsqueeze(1).chunk(6, dim=-1)

        z = self.norm1(x)
        z = (1 + gamma1) * z + beta1
        z = self.attn(z)
        x = alpha1 * z + x

        z = self.norm2(x)
        z = (1 + gamma2) * z + beta2
        z = self.ffn(z)
        x = alpha2 * z + x
        return x
    

class DiT(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.p_uncond = config.p_uncond
        self.null_label = config.n_classes

        self.blocks = nn.ModuleList([DiTBlock(config.dim, config.n_heads) for _ in range(config.n_blocks)])
        self.norm = nn.LayerNorm(config.dim)

        self.label_emb = nn.Embedding(config.n_classes + 1, config.dim) # +1 for `null` lebel
        self.time_emb = nn.Embedding(config.n_steps, config.dim)
        self.linear = nn.Linear(config.dim, config.n_channels * config.patch_size * config.patch_size)

        # patchify
        self.n_patches = config.img_size // config.patch_size
        self.patch_size = config.patch_size
        self.patch_emb = nn.Conv2d(config.n_channels, config.dim, kernel_size=config.patch_size, stride=config.patch_size) 
        self.patch_pos_emb = nn.Parameter(torch.rand(1, self.n_patches ** 2, config.dim))


    def forward(self, x: torch.Tensor, labels: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """ x: (B, C, H, W), labels: (B), time: (B)"""
        x = self.patch_emb(x) # (B, D, n, n), L = n*n
        x = x.flatten(2).transpose(1, 2) + self.patch_pos_emb

        if self.training:
            mask = torch.rand_like(labels, dtype=torch.float) < self.p_uncond
            labels = mask * self.null_label + ~mask * labels
        cond = self.label_emb(labels) + self.time_emb(time)
        
        for block in self.blocks:
            x = block(x, cond)
        
        x = self.norm(x)
        x = self.linear(x) # (B, n*n, C*p*p)
        # (B, C, n*p, n*p) = (B, C, H, W)
        x = rearrange(x, 'b (h w) (c ph pw) -> b c (h ph) (w pw)', h=self.n_patches, w=self.n_patches, ph=self.patch_size, pw=self.patch_size)
        return x


if __name__ == '__main__':
    config = DiTConfig()
    model = DiT(config)
    
    x = torch.randn(2, 3, 32, 32) # (B, C, H, W)
    labels = torch.randint(0, config.n_classes + 1, (2,)) # (B)
    time = torch.randint(0, config.n_steps, (2,)) # (B)

    out = model(x, labels, time)
    print(out.shape) # should be (B, C, H, W)