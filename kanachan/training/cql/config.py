from pathlib import Path
from dataclasses import dataclass, field
from typing import Any
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


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
    contiguous_training_data: bool = False
    rewrite_grades: int | None = None
    num_workers: int | None = None
    replay_buffer_size: int = 0
    num_qr_intervals: int = 200
    dueling_architecture: bool = True
    initial_model_prefix: Path | None = None
    initial_model_index: int | None = None
    reward_plugin: Path = MISSING
    discount_factor: float = 0.99
    kappa: float = 1.0
    td_computation_batch_size: int = 0
    alpha: float = 1.0
    checkpointing: bool = False
    batch_size: int = MISSING
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    target_update_interval: int | None = None
    target_update_rate: float | None = None
    snapshot_interval: int = 0


config_store = ConfigStore.instance()
config_store.store(name="config", node=Config)
