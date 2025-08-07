from dataclasses import dataclass

@dataclass
class DiTConfig:
    # model 
    dim: int = 64
    n_heads: int = 4
    n_blocks: int = 4
    
    # data
    n_classes: int = 10
    img_size: int = 32
    n_channels: int = 3
    patch_size: int = 8

    # ddpm
    n_steps: int = 500
    p_uncond: float = 0.1 # 无条件生成，label置为null