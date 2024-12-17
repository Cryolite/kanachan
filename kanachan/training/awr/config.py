from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from kanachan.training.core.config.device import DeviceConfig
from kanachan.training.core.config.encoder import EncoderConfig
from kanachan.training.core.config.decoder import DecoderConfig
from kanachan.training.core.config.optimizer import OptimizerConfig


@dataclass
class Config:
    device: DeviceConfig
    encoder: EncoderConfig
    decoder: DecoderConfig
    optimizer: OptimizerConfig
    training_data: Path = MISSING
    rewrite_rooms: str | int | None = None
    rewrite_grades: str | int | None = None
    num_workers: int | None = None
    initial_model_prefix: Path | None = None
    initial_model_index: int | None = None
    q_model: Path = MISSING
    value_model: Path | None = None
    beta: float = 1.0
    weight_clipping: float = 20.0
    checkpointing: bool = False
    batch_size: int = MISSING
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    snapshot_interval: int = 0
    defaults: list[Any] = field(default_factory=lambda: [
        {"device": "cuda"},
        {"encoder": "bert_base"},
        {"decoder": "single"},
        {"optimizer": "adam"},
        "_self_",
    ])


config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)
