# Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Generator

from pt_config import ModelConfig

from rope import RotaryEmbedding, apply_rotary_pos_emb

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return ( # 比torch.repeat_interleave更快
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        self.head_dim = args.dim // args.n_heads
        self.w_q = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.w_out = nn.Linear(args.dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache=False
    ):
        
        B, L, _ = x.shape
        
        q = self.w_q(x).view(B, L, self.n_heads, self.head_dim)
        k = self.w_k(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.w_v(x).view(B, L, self.n_kv_heads, self.head_dim)

        # 和绝对位置编码不同的是
        # 相对位置编码需要在每一层都添加编码
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # kv cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)

        if attention_mask is not None:
            attention_mask = attention_mask.bool().view(B, 1, 1, -1).expand(B, self.n_heads, L, -1)
        if self.flash and L != 1:
            # 官方版本内部已经实现了flash attention内核
            # 受限于算子实现，要求序列长度L >= 2
            output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                attn_mask=attention_mask,
                is_causal=True
            )
        else:
            mask: torch.Tensor = self.mask[:, :, :L, :L] # casual mask
            if attention_mask is not None:
                mask = mask.masked_fill(attention_mask == False, float("-inf"))
            # eager attention
            scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5
            scores += mask
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            output = scores @ v

        output = output.transpose(1, 2).reshape(B, L, -1)
        output = self.w_out(output)
        output = self.resid_dropout(output)
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class GPTBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.n_heads
        
        self.attention = Attention(config)
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.feed_forward = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ):
        x_attn, past_kv = self.attention(
            x=self.attention_norm(x),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + x_attn
        out = x + self.feed_forward(self.ffn_norm(x))
        return out, past_kv
    

class MyGPT(nn.Module):

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.token_emb = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.padding_idx)
        self.rotary_emb = RotaryEmbedding(config.max_seq_len, config.dim // config.n_heads, base=config.rope_theta)

        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 分类头和嵌入表权重共享，可以减少模型参数数量，提高训练和推理效率
        # https://arxiv.org/abs/1608.05859
        self.token_emb.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        # KV cache related
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict:

        inputs_embeds: torch.Tensor = self.token_emb(input_ids) # (B, L, D)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        if cache_position is None:
            past_seen_tokens = past_key_values[0][0].size(1) if past_key_values[0] is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        x = self.dropout(inputs_embeds)

        past_kvs = []
        for i, layer in enumerate(self.layers):
            x, past_kv = layer(
                x, 
                position_embeddings = position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
            )
            past_kvs.append(past_kv)
        x = self.norm(x)

        logits: torch.Tensor = self.lm_head(x)

        return dict(
            logits=logits,
            past_key_values=past_kvs,
        )
    
    # 流式生成
    @torch.inference_mode()
    def stream(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 1024, 
        temperature: float = 0.75, 
        top_p: float = 0.90,
        repeat_penalty: float = 1., 
        use_cache: bool = True, 
        pad_token_id: int = 0, 
        eos_token_id: int = 2, 
        **kwargs
    ) -> Generator[torch.Tensor]:
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out = self(
                    input_ids, 
                    past_key_values=past_kvs, 
                    use_cache=use_cache,
                    **kwargs
                )
                first_seq = False
            else:
                out = self(
                    input_ids[..., -1], 
                    past_key_values=past_kvs, 
                    use_cache=use_cache,
                    start_pos=input_ids.shape[1] - 1,
                    **kwargs
                )
            logits, past_kvs = out.logits[..., -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= repeat_penalty
            logits /= (temperature + 1e-9)
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')
            input_ids_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, input_ids_next), dim=1)
            yield input_ids[:, start:]
            if input_ids_next.item() == eos_token_id:
                break

    @torch.inference_mode()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        max_new_tokens: int = 1024, 
        temperature: float = 0.75, 
        top_p: float = 0.90,
        repeat_penalty: float = 1., 
        use_cache: bool = True, 
        pad_token_id: int = 0, 
        eos_token_id: int = 2, 
        **kwargs
    ) -> torch.Tensor:
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self.stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, repeat_penalty, use_cache, **kwargs)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)
        max_length = max(seq.size(1) for seq in generated)
        generated = [
            torch.cat(
                [seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)],
                dim=-1)
            for seq in generated
        ]
        return torch.cat(generated, dim=0)



if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('./weights')
    config = ModelConfig(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=6400,
        hidden_dim=None,
        norm_eps=1e-5,
        max_seq_len=8192,
        rope_theta=1e6,
        dropout=0.1,
        flash_attn=True
    )
    config.padding_idx = tokenizer.pad_token_id
    tokenizer.model_max_length = config.max_seq_len
    model = MyGPT(config)


    text = "Hello, how are you?"
    input_ids: torch.Tensor = tokenizer.encode(text, return_tensors="pt")
    print(input_ids.shape)
    print(tokenizer.decode(input_ids[0]))
    outputs = model(input_ids)
    print(outputs)
