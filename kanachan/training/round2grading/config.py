from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

# pylint: disable=unused-import
import kanachan.training.core.config.device
import kanachan.training.core.config.encoder
import kanachan.training.core.config.optimizer


_defaults = [
    {"device": "cuda"},
    {"encoder": "bert_base"},
    {"decoder": "single"},
    {"optimizer": "adam"},
    "_self_",
]


@dataclass
class Config:
    defaults: list[Any] = field(default_factory=lambda: _defaults)
    training_data: Path = MISSING
    num_workers: int | None = None
    celestial_scale: int = 450
    initial_model_prefix: Path | None = None
    initial_model_index: int | None = None
    checkpointing: bool = False
    batch_size: int = MISSING
    loss_function: str = "symlog"
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    snapshot_interval: int = 0


config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)
