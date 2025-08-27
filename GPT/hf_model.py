# Ref: https://github.com/meta-llama/llama/blob/main/llama/model.py
# Ref: https://github.com/huggingface/transformers/blob/28ca27cb2b78dae855b6e9ce1461fed18a9a9594/src/transformers/models/llama/modeling_llama.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, List

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from hf_config import ModelConfig

from rope import apply_rotary_emb, precompute_pos_cis

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
        pos_cis: torch.Tensor,
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
        q, k = apply_rotary_emb(q, k, pos_cis)
        
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
        pos_cis: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ):
        x_attn, past_kv = self.attention(
            x=self.attention_norm(x),
            attention_mask=attention_mask,
            pos_cis=pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + x_attn
        out = x + self.feed_forward(self.ffn_norm(x))
        return out, past_kv
    

class MyGPTModel(nn.Module):
    pos_cis: torch.Tensor

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.token_emb = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.padding_idx)

        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(dim=config.dim // config.n_heads, theta=config.rope_theta),
            persistent=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> BaseModelOutputWithPast:
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = kwargs.get('start_pos', 0)
        h = self.dropout(self.token_emb(input_ids))
        pos_cis = self.pos_cis[start_pos:start_pos + input_ids.size(1)]
        past_kvs = []
        for i, layer in enumerate(self.layers):
            h, past_kv = layer(
                h, 
                pos_cis = pos_cis,
                attention_mask=attention_mask,
                past_key_value=past_key_values[i],
                use_cache=use_cache,
            )
            past_kvs.append(past_kv)
        hidden_states = self.norm(h)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_kvs,
        )


class MyGPTForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = ModelConfig # 务必定义config_class
    
    def __init__(self, config: ModelConfig):
        super().__init__(config) # 务必传入config
        self.model = MyGPTModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.loss_function = nn.CrossEntropyLoss(reduction='mean')

        # 分类头和嵌入表权重共享
        # https://arxiv.org/abs/1608.05859
        self.model.token_emb.weight = self.lm_head.weight # 这一行的bug查了好久，lm_head权重共享很重要!!!

        # 确保模型初始化完成后，权重被合理初始化；
        # 确保 tied weights（共享权重）被建立；
        # 保证 config 与模型参数一致性。
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs
    ) -> CausalLMOutputWithPast:
        
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        logits: torch.Tensor = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_function(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            logits=logits,
            loss=loss,
            past_key_values=outputs.past_key_values,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        # 使用GenerationMixin的时候
        # forward函数必须显式接受input_ids, token_type_ids, attention_mask三个参数
        # 否则必须额外定义prepare_inputs_for_generation来处理输入
        return {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}


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
    model = MyGPTForCausalLM(config)


    text = "Hello, how are you?"
    input_ids: torch.Tensor = tokenizer.encode(text, return_tensors="pt")
    print(input_ids.shape)
    print(tokenizer.decode(input_ids[0]))
    outputs = model(input_ids)
    print(outputs)