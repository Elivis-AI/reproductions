from dataclasses import dataclass

@dataclass
class MoEConfig:
    dim: int = 64
    n_experts: int = 8
    top_k: int = 3
    alpha: float = 0.01
    infer_acc: bool = False
