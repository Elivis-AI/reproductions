from dataclasses import dataclass


# size(dense model) ref:
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


@dataclass
class ModelConfig:
    model_type: str = "myGPT"

    dim: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 4
    vocab_size: int = 6400
    hidden_dim: int | None = None
    norm_eps: float = 1e-5
    max_seq_len: int = 8192
    padding_idx: int | None = None
    rope_theta: float = 1e6
    dropout: float = 0.0
    flash_attn: bool = True


    def __post_init__(self):
        if self.hidden_dim is None:
            self.hidden_dim = self._default_ffn_dim(self.dim)

    def _default_ffn_dim(self, dim: int, multiple_of: int = 64) -> int:
        hidden_dim = 4 * dim
        hidden_dim = int(2 * hidden_dim / 3)
        return multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)