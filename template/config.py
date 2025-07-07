import yaml
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    hidden_dim: int
    layers: int

@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    epochs: int

@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig



def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        raw_cfg: dict[str, Any] = yaml.safe_load(f)
        return Config(
            model=ModelConfig(**raw_cfg["model"]),
            train=TrainConfig(**raw_cfg["train"])
        )

cfg = load_config("config.yaml")
print(cfg.model.hidden_dim)
