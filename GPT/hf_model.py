# Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Optional, Tuple, List, Generator

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from hf_config import LMConfig
from utils import apply_rotary_emb, repeat_kv, precompute_pos_cis



class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn

        self.head_dim = args.dim // args.n_heads
        self.w_qkv = nn.Linear(args.dim, args.dim * 3, bias=False)
        self.w_out = nn.Linear(args.dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        
        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(
            self,
            x: torch.Tensor,
            pos_cis: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            attention_mask: Optional[torch.Tensor] = None,
            use_cache=False
        ):
        
        B, L, _ = x.shape
        
        q, k, v = self.w_qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim)
        k = k.view(B, L, self.n_kv_heads, self.head_dim)
        v = v.view(B, L, self.n_kv_heads, self.head_dim)

        q, k = apply_rotary_emb(q, k, pos_cis)
        
        # kv_cache实现
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        past_kv = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)

        if self.flash and L != 1:
            # 官方版本内部已经实现了flash attention内核
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / self.head_dim ** 0.5
            scores += self.mask[:, :, :L, :L] * (~attention_mask)
            scores = F.softmax(scores.float(), dim=-1).type_as(q)
            scores = self.attn_dropout(scores)
            output = scores @ v

        output = output.transpose(1, 2).reshape(B, L, -1)
        output = self.resid_dropout(self.w_out(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: LMConfig):
        super().__init__()
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            config.hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
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

    def forward(self, x):
        return self.weight * (x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x)


class GPTBlock(nn.Module):
    def __init__(self, config: LMConfig):
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
        pos_cis: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ):
        x_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        x = x + x_attn
        out = x + self.feed_forward(self.ffn_norm(x))
        return out, past_kv
    

class MyGPT(PreTrainedModel):
    config: LMConfig

    def __init__(self, config: LMConfig):
        super().__init__(config) # 务必传入config
        self.vocab_size = config.vocab_size
        self.n_layers = config.vocab_size, config.n_layers
        
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([GPTBlock(config) for _ in range(self.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(dim=config.dim // config.n_heads, theta=config.rope_theta),
            persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **args
    ) -> CausalLMOutputWithPast:
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = args.get('start_pos', 0)
        h = self.dropout(self.tok_embeddings(input_ids))
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []
        for i, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, pos_cis,
                past_key_value=past_key_values[i],
                use_cache=use_cache
            )
            past_kvs.append(past_kv)
        logits = self.output(self.norm(h))
        
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            hidden_states=h,
            past_key_values=past_kvs,
        )

    # 流式生成
    @torch.inference_mode()
    def stream(
        self, 
        input_ids: torch.Tensor, 
        eos_token_id: int = 2, 
        max_new_tokens: int = 1024, 
        temperature: float = 0.75, 
        top_p: float = 0.90,
        rp: float = 1., 
        use_cache: bool = True, 
        pad_token_id: int = 0, 
        **args
    ) -> Generator[torch.Tensor]:
        start, first_seq, past_kvs = input_ids.shape[1], True, None
        while input_ids.shape[1] < max_new_tokens - 1:
            if first_seq or not use_cache:
                out, first_seq = self(input_ids, past_key_values=past_kvs, use_cache=use_cache, **args), False
            else:
                out = self(input_ids[:, -1:], past_key_values=past_kvs, use_cache=use_cache,
                           start_pos=input_ids.shape[1] - 1, **args)
            logits, past_kvs = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids.tolist()[0]))] /= rp
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
        eos_token_id: int = 2, 
        max_new_tokens: int = 1024, 
        temperature: float = 0.75, 
        top_p: float = 0.90,
        rp: float = 1., 
        use_cache: bool = True, 
        pad_token_id: int = 0, 
        **args
    ) -> torch.Tensor:
        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self.stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, rp, use_cache, **args)
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
