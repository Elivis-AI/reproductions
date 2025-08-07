from dataclasses import dataclass

@dataclass
class DiTConfig:
    # model 
    dim: int = 256
    n_heads: int = 4
    n_blocks: int = 6
    
    # data
    n_classes: int = 10
    img_size: int = 32
    n_channels: int = 3
    patch_size: int = 4

    # ddpm
    n_steps: int = 1000
    p_uncond: float = 0.1 # 无条件生成，label置为null