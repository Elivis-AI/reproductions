# References:
# https://huggingface.co/docs/transformers/model_doc/roberta

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# from utils import print_number_of_parameters

class PositionEmbedding(nn.Embedding):
    def __init__(self, max_len, dim):
        super().__init__(max_len, dim)

    def forward(self, x):
        # x: (B, L)
        pos = torch.arange(x.size(1), dtype=torch.long, device=x.device).repeat(x.size(0), 1)
        return super().forward(pos)


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, attn_pdrop=0.1, out_pdrop=0.1):
        super().__init__()
    
        self.n_heads = n_heads
        self.head_size = dim // n_heads

        self.qkv_proj = nn.Linear(dim, 3 * n_heads * self.head_size)
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(out_pdrop)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = rearrange(q, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        k = rearrange(k, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        v = rearrange(v, pattern="b n (h d) -> b h n d", h=self.n_heads, d=self.head_size)
        attn_score: torch.Tensor = q @ k.transpose(-2, -1) / (self.head_size ** 0.5)
        if mask is not None:
            attn_score.masked_fill_((mask.unsqueeze(1).unsqueeze(2) == False), value=torch.finfo(torch.float16).min) # bf16 bug fix
            # attn_score += (mask.unsqueeze(1).unsqueeze(2) == False) * torch.finfo(torch.float16).min # bf16 bug fix
        attn_weight = F.softmax(attn_score, dim=-1).type_as(q)
        x = attn_weight @ v
        x = rearrange(x, pattern="b h n d -> b n (h d)")
        x = self.attn_drop(x)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, drop_prob=0.1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),  # Using GELU activation as per RoBERTa paper
            nn.Dropout(drop_prob),
            nn.Linear(4 * dim, dim),
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        return self.model(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim, n_heads, pdrop):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, attn_pdrop=pdrop, out_pdrop=pdrop)
        self.ffn = FeedForward(dim, pdrop)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask):
        x = self.norm1(x + self.attn(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x


class RoBERTa(nn.Module):
    def __init__(
        self,
        vocab_size,
        max_len,
        padding_idx,
        n_layers=12,
        n_heads=12,
        dim=768,
        drop_prob=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_len = max_len
        self.dim = dim

        self.pos_emb = PositionEmbedding(max_len, dim)
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=padding_idx)
        self.layers = nn.ModuleList(
            [EncoderBlock(dim, n_heads, drop_prob) for _ in range(n_layers)]
        )

        self.pre_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # (B, L)
        x = self.pre_norm(self.pos_emb(x) + self.token_emb(x))
        
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x


class RoBERTaForPretraining(nn.Module):
    def __init__(self, vocab_size, max_len, n_layers, n_heads, dim, padding_idx):
        super().__init__()

        self.roberta = RoBERTa(
            vocab_size=vocab_size,
            max_len=max_len,
            n_layers=n_layers,
            n_heads=n_heads,
            dim=dim,
            padding_idx=padding_idx
        )

        self.mlm_head = nn.Linear(dim, vocab_size)

    def forward(self, x, pad_mask):
        x = self.roberta(x, pad_mask)  # (B, L)
        x = self.mlm_head(x) # logits
        return x
    

if __name__ == '__main__':
    n_layers = 2        # Number of transformer layers
    n_heads = 4         # Number of attention heads
    hidden_size = 768   # Hidden size of the model
    
    # Example usage
    from transformers import RobertaTokenizerFast
    from mlm import MaskedLanguageModel
    tokenizer = RobertaTokenizerFast.from_pretrained("FacebookAI/roberta-base")
    print(tokenizer.vocab_size)
    mlm = MaskedLanguageModel(
        vocab_size=tokenizer.vocab_size,
        mask_id=tokenizer.mask_token_id,
        no_mask_token_ids=[
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.mask_token_id,
        ],
    )
    model = RoBERTaForPretraining(
        vocab_size=tokenizer.vocab_size,
        max_len=tokenizer.model_max_length,
        n_layers=n_layers,
        n_heads=n_heads,
        dim=hidden_size,
        padding_idx=tokenizer.pad_token_id
    )

    from utils import print_num_parameters
    print_num_parameters(model)
    texts = ["This is an example sentence for masked language modeling.", "Hello World!"]
    output = tokenizer(texts, truncation=True, padding=True, return_tensors='pt')
    masked_token_ids, mask = mlm(output.input_ids)
    print(mask)

    pred = model(masked_token_ids, pad_mask=output.attention_mask)
    print(mlm.cal_loss(pred, masked_token_ids, mask))  # Calculate loss