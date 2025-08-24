from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    model_type = "myGPT"

    def __init__(
            self,
            dim: int = 512,
            n_layers: int = 8,
            n_heads: int = 8,
            n_kv_heads: int = 2,
            vocab_size: int = 6400,
            hidden_dim: int = None,
            norm_eps: float = 1e-5,
            max_seq_len: int = 8192,
            padding_idx: int = None,
            rope_theta: int = 1e6,
            dropout: float = 0.0,
            flash_attn: bool = True,
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim or self._default_ffn_dim(dim)
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn

        super().__init__(**kwargs) # 务必保留kwargs
    
    def _default_ffn_dim(self, dim: int, multiple_of: int = 64):
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)


# size ref:
# {
#     # names follow the huggingface naming conventions
#     # GPT-1
#     'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
#     # GPT-2 configs
#     'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
#     'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
#     'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
#     'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
#     # Gophers
#     'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
#     # (there are a number more...)
#     # I made these tiny models up
#     'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
#     'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
#     'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
# }


def compute_params_and_flops(config: ModelConfig):
    dim = config.dim
    n_layers = config.n_layers
    n_heads = config.n_heads
    hidden_dim = config.hidden_dim
    vocab_size = config.vocab_size
    seq_len = config.max_seq_len
    
    # ---- 参数量 ----
    embed_params = vocab_size * dim
    attn_params = n_layers * (3 * dim * dim + dim * dim)  # QKV + out proj
    ffn_params = n_layers * (2 * dim * hidden_dim)        # W1 + W2
    total_params = embed_params + attn_params + ffn_params
    
    # ---- FLOPs ----
    head_dim = dim // n_heads
    # attention: QK^T + softmax + AV ≈ 4 * n_heads * seq_len^2 * head_dim
    attn_flops = n_layers * (4 * n_heads * seq_len * seq_len * head_dim)
    # FFN: 两次 matmul ≈ 2 * dim * hidden_dim * seq_len
    ffn_flops = n_layers * (2 * dim * hidden_dim * seq_len)
    
    max_flops = attn_flops + ffn_flops
    flops_per_token = max_flops / seq_len
    
    print(f"Total params: {total_params / 1e6:.2f} M")
    print(f"FLOPs per token: {flops_per_token / 1e9:.2f} GFLOPs")


if __name__ == "__main__":
    default_config = ModelConfig()
    compute_params_and_flops(default_config)