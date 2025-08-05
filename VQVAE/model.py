# Modified from:
#   taming-transformers: https://github.com/CompVis/taming-transformers

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLayer(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_dim, h_dim, kernel_size=3, stride=1, padding=1), # s = 1 and k = 2p + 1 => keep image size
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=1, stride=1) # s = 1 and k = 2p + 1 => keep image size
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, in_dim, h_dim, n_layers):
        super().__init__()
        self.n_res_layers = n_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim) for _ in range(n_layers)]
        )

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_dim=3, out_dim=32, n_res_layers=2):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.ReLU(), 
            nn.Conv2d(out_dim, out_dim, 4, 2, 1),
            nn.ReLU(), 
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            ResidualStack(out_dim, out_dim, n_res_layers),
        )

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, in_dim=32, out_dim=3, n_res_layers=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            ResidualStack(in_dim, in_dim, n_res_layers),
            nn.ConvTranspose2d(in_dim, in_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_dim, out_dim, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    # Test the Encoder and Decoder
    # encoder = Encoder(ch_mult=(1,4,4,16), num_res_blocks=1, z_channels=64)
    # decoder = Decoder(ch_mult=(1,4,4,16), num_res_blocks=1, z_channels=64)
    encoder = Encoder()
    decoder = Decoder()
    print(encoder)

    x = torch.randn(1, 3, 32, 32)  # Example input
    z = encoder(x)
    print("Encoded shape:", z.shape)

    x_reconstructed = decoder(z)
    print("Reconstructed shape:", x_reconstructed.shape)
    assert x_reconstructed.shape == (1, 3, 32, 32), "Reconstruction shape mismatch"