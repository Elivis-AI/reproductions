# Modified from https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Tuple

from model import build_encoder_decoder


class VAE(nn.Module):
    def __init__(self, in_size: int, in_ch: int, chs: List[int], latent_dim: int) -> None:
        super().__init__()
        chs = [in_ch] + chs  # Ensure input channels are included
        self.latent_dim = latent_dim 
        self.last_dim = chs[-1]
        self.last_size = in_size // (2 ** (len(chs) - 1))

        self.encoder, self.decoder = build_encoder_decoder(chs)
        # the output of encoder should be (B, last_dim, last_size, last_size)
        # flatten (B, last_dim * last_size * last_size)
        flatten_dim = self.last_dim * self.last_size * self.last_size
        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_var = nn.Linear(flatten_dim, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, flatten_dim)


    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """input: (B, C, H, W)"""
        x = self.encoder(x)
        x = x.flatten(1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (torch.Tensor) [B x D]
        :return: (torch.Tensor) [B x C x H x W]
        """
        x = self.decoder_input(z)
        x = x.view(-1, self.last_dim, self.last_size, self.last_size)
        x = self.decoder(x)
        return x

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(input)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        # Reparameterization trick to sample from N(mu, var) from N(0,1).
        z = eps * std + mu
        return self.decode(z), mu, log_var

    def sample(self, num_samples:int) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=self.fc_mu.weight.device)
        return self.decode(z)


if __name__ == '__main__':
    vae = VAE(in_size=32, in_ch=3, chs=[16, 32, 64], latent_dim=64)
    x = torch.randn(1, 3, 32, 32)  # Example input tensor
    output, mu, log_var = vae(x)
    assert output.shape == x.shape, output.shape
    print(mu.shape, log_var.shape)