import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()

        self.n_heads = n_heads

        self.w_qkv = nn.Linear(dim, dim * 3)
        self.w_out = nn.Linear(dim, dim)
        

    def forward(self, x: torch.Tensor):
        # x: (B, L, D)
        qkv = self.w_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        def split(x: torch.Tensor) -> torch.Tensor:
            B, L, D = x.shape
            return x.reshape(B, L, self.n_heads, D // self.n_heads).transpose(1, 2)
        q, k, v = map(split, (q, k, v))

        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v # no causal mask, because image patch tokens are global connection.
        out = out.transpose(1, 2).reshape(out.size(0), out.size(2), -1)

        out = self.w_out(out)
        return out
    

class NewGELU(nn.Module):
    """
    # Copy from minGPT
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class FeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            NewGELU(),
            # nn.Dropout(pdrop),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    

if __name__ == '__main__':
    x = torch.randn(2, 10, 64) # (B, L, D)
    t = torch.randint(0, 1000, (2,)) # (B)

    # Test SelfAttention
    attn = SelfAttention(64, 8)
    ffn = FeedForward(64)
    out = attn(x) # (B, L, D)
    out = ffn(out)
    print(out.shape)
