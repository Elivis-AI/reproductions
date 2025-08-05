import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, quantizer: VectorQuantizer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer

    def forward(self, x: torch.Tensor):
        # (B, C, H, W)
        z = self.encoder(x)
        z_q, indices, q_loss, codebook_usage = self.quantizer(z)
        x_q = self.decoder(z_q)

        return x_q, indices, q_loss, codebook_usage
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        # (B, C, H, W)
        z = self.encoder(x)
        z_q, *_ = self.quantizer(z)
        return z_q
    
    @torch.no_grad()
    def decode(self, z_q: torch.Tensor):
        # (B, C, H, W)
        x_q = self.decoder(z_q)
        return x_q
    
    @torch.no_grad()
    def decode_code(self, code: torch.Tensor, shape: Tuple):
        # code: (B, k), shape: (C, H, W)
        z_q = self.quantizer.get_codebook_entry(code, shape)
        x_q = self.decoder(z_q)
        return x_q