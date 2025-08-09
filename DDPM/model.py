import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from unet import UnetFFN, UnetDown, UnetUp


class AdaLN(nn.Module):
    def __init__(self, ch, t_dim, label_dim):
        super().__init__()
        self.w_t = nn.Linear(t_dim, ch, bias=False)
        self.w_label = nn.Linear(label_dim, ch, bias=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor):
        alpha = self.w_t(t).view(t.size(0), -1, 1, 1)  # (B, C, 1, 1)
        beta = self.w_label(label).view(label.size(0), -1, 1, 1)  # (B, C, 1, 1)
        return alpha * x + beta

class TimeEmbedding(nn.Module):
    def __init__(self, n_steps, emb_dim):
        super().__init__()
        
        self.register_buffer('n_steps', torch.tensor([n_steps]))
        self.model = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),    
        )

    def forward(self, t):
        t = t / self.n_steps # 注意这一步是tensor除法
        return self.model(t.reshape(-1, 1))


class ConditionalUnet(nn.Module):
    def __init__(
            self,
            in_size: int = 256,
            in_ch: int = 3,
            chs: List[int] = [32, 64, 128, 256, 512], 
            n_steps: int = 100,
            n_class: int = 10,
            t_dim: int = 32,
            c_dim: int = 32,
            p_uncond: float = 0.1,
        ):
        super().__init__()
        assert in_size % (2 ** (len(chs) - 1)) == 0

        self.p_uncond = p_uncond
        self.null_label = n_class
        self.t_emb = TimeEmbedding(n_steps, t_dim)  # Time embedding
        self.c_emb = nn.Embedding(n_class + 1, c_dim)  # Class embedding

        self.conv_in = UnetFFN(in_ch, chs[0])
        self.downs = nn.ModuleList([UnetDown(chs[i], chs[i+1]) for i in range(len(chs) - 1)])
        self.ups = nn.ModuleList([UnetUp(chs[-i], chs[-(i+1)]) for i in range(1, len(chs))])
        self.cond_downs = nn.ModuleList([AdaLN(chs[i], t_dim, c_dim) for i in range(len(chs) - 1)])
        self.cond_ups = nn.ModuleList([AdaLN(chs[-i], t_dim, c_dim) for i in range(1, len(chs))])
        self.conv_out = nn.Sequential(
            UnetFFN(chs[0] * 2, chs[0]),                                 # UnetFFN ends with activation function
            nn.Conv2d(chs[0], in_ch, kernel_size=3, stride=1, padding=1) # So you need another Conv2d as the end
        )
        # middle layer plays the ratial role of bottleneck
        feat_size = in_size // (2 ** (len(chs) - 1))
        self.middle = nn.Sequential(
            nn.AvgPool2d(feat_size), # output: (..., 1, 1)
            nn.GELU(),
            nn.ConvTranspose2d(chs[-1], chs[-1], feat_size, feat_size),
            nn.GroupNorm(8, chs[-1]),
            nn.ReLU(),
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor | None, t: torch.Tensor) -> torch.Tensor:
        if c is None:
            c = torch.full((x.size(0),), self.null_label, dtype=torch.long, device=x.device)
        elif self.training:
            mask = torch.rand_like(c, dtype=torch.float) < self.p_uncond
            c = mask * self.null_label + ~mask * c
        c = self.c_emb(c)
        t = self.t_emb(t)

        # conv_in
        x = self.conv_in(x)
        x_in = x
        # down
        skips = []
        for down, cond_down in zip(self.downs, self.cond_downs):
            x = cond_down(x, t, c)
            x = down(x)
            skips.append(x)
        # middle
        x = self.middle(x)
        # up
        for up, cond_up in zip(self.ups, self.cond_ups):
            skip = skips.pop()
            x = cond_up(x, t, c)
            x = up(torch.cat([x, skip], dim=1))
        # conv_out
        x = self.conv_out(torch.cat([x, x_in], dim=1))  # Note: concat here is very important
        return x
    

if __name__ == '__main__':
    net = ConditionalUnet(in_size=28, in_ch=1, chs=[32, 64])
    
    # 2 samples of  random imgs with the shape of mnist
    x = torch.randn(2, 1, 28, 28) 
    t = torch.randint(0, 100, size=(2,))
    c = torch.randint(0, 10, size=(2,))

    out = net(x, c, t)
    print(out.shape)