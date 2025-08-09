import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class UnetFFN(nn.Module):
    "CNN that keeps the image size unchanged。"
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.model = nn.Sequential(                 
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class UnetDown(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.down_sample = nn.MaxPool2d(2)
        self.ffn = nn.Sequential(                 
            UnetFFN(ch_in, ch_out),
            UnetFFN(ch_out, ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ffn(x)
        x = self.down_sample(x)
        return x


class UnetUp(nn.Module):
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        # input: torch.cat([x, x_skip], dim=1)
        self.up_sample = nn.ConvTranspose2d(2 * ch_in, ch_out, kernel_size=2, stride=2)
        self.ffn = nn.Sequential(                 
            UnetFFN(ch_out, ch_out),
            UnetFFN(ch_out, ch_out),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_sample(x)
        x = self.ffn(x)                            
        return x
    

class Unet(nn.Module):
    def __init__(
            self,
            in_size = 256,
            in_ch = 3,
            chs: List[int] =[32, 64, 128, 256, 512],
        ):
        super().__init__()
        assert in_size % (2 ** (len(chs) - 1)) == 0
        self.conv_in = UnetFFN(in_ch, chs[0])
        self.downs = nn.ModuleList([UnetDown(chs[i], chs[i+1]) for i in range(len(chs) - 1)])
        self.ups = nn.ModuleList([UnetUp(chs[-i], chs[-(i+1)]) for i in range(1, len(chs))])
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W)
        # conv_in
        x = self.conv_in(x)
        x_in = x
        # down
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        # middle
        x = self.middle(x)
        # up
        for up in self.ups:
            skip = skips.pop()
            # print(x.shape, skip.shape)
            x = up(torch.cat([x, skip], dim=1))
        # conv_out
        x = self.conv_out(torch.cat([x, x_in], dim=1))  # Note: concat here is very important
        return x


if __name__ == "__main__":
    model = Unet()
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M")
    
    # Example input
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(output.shape)  # Should be the same shape as input