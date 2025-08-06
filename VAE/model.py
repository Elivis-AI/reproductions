import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Iterable

class Encoder(nn.Module):
    def __init__(self, chs: List[int] =[3, 32, 64, 128, 256, 512]):
        super().__init__()

        self.layers = nn.ModuleList(
            [  
                nn.Sequential(                 
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU()
                )
                for ch_in, ch_out in zip(chs[:-1], chs[1:])
            ]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, chs=[512, 256, 128, 64, 32, 3]):
        super().__init__()

        self.layers = nn.ModuleList(
            [  
                nn.Sequential(                 
                    nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU()
                )
                for ch_in, ch_out in zip(chs[:-1], chs[1:])
            ]
        )
        self.output_layer = nn.Sequential(
            # nn.ConvTranspose2d(chs[-1], chs[-1], kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.BatchNorm2d(chs[-1]),
            # nn.LeakyReLU(),
            # nn.Conv2d(chs[-1], chs[-1], kernel_size=3, padding=1),
            nn.Conv2d(chs[-1], chs[-1], kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


def build_encoder_decoder(chs: Iterable[int]) -> Tuple[nn.Module, nn.Module]:
    chs = list(chs)
    encoder = Encoder(chs=chs)
    decoder = Decoder(chs=list(reversed(chs)))
    return encoder, decoder



if __name__ == "__main__":
    # Example usage
    encoder, decoder = build_encoder_decoder([3, 16, 32, 64])
    x = torch.randn(1, 3, 32, 32)  # Example input tensor
    encoded = encoder(x)
    decoded = decoder(encoded)
    print(f"Encoded shape: {encoded.shape}, Decoded shape: {decoded.shape}")