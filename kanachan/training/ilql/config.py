from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, List
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class CpuConfig:
    type: Optional[str] = 'cpu'
    dtype: str = 'float64'
    amp_dtype: Optional[str] = None

@dataclass
class CudaConfig:
    type: Optional[str] = 'cuda'
    dtype: str = 'float32'
    amp_dtype: str = 'float16'


@dataclass
class BertBaseEncoderConfig:
    position_encoder: str = 'position_embedding'
    dimension: int = 768
    num_heads: int = 12
    dim_feedforward: Optional[int] = None
    activation_function: str = 'gelu'
    dropout: float = 0.1
    num_layers: int = 12
    load_from: Optional[Path] = None

@dataclass
class BertLargeEncoderConfig:
    position_encoder: str = 'position_embedding'
    dimension: int = 1024
    num_heads: int = 16
    dim_feedforward: Optional[int] = None
    activation_function: str = 'gelu'
    dropout: float = 0.1
    num_layers: int = 24
    load_from: Optional[Path] = None


@dataclass
class SingleDecoderConfig:
    dim_feedforward: Optional[int] = None
    activation_function: str = 'gelu'
    dropout: float = 0.1
    num_layers: int = 1
    load_from: Optional[Path] = None

@dataclass
class DoubleDecoderConfig:
    dim_feedforward: Optional[int] = None
    activation_function: str = 'gelu'
    dropout: float = 0.1
    num_layers: int = 2
    load_from: Optional[Path] = None

@dataclass
class TripleDecoderConfig:
    dim_feedforward: Optional[int] = None
    activation_function: str = 'gelu'
    dropout: float = 0.1
    num_layers: int = 3
    load_from: Optional[Path] = None


@dataclass
class SgdOptimizerConfig:
    type: str = 'sgd'
    momentum: Optional[float] = 0.0
    epsilon: Optional[float] = None
    learning_rate: float = MISSING
    warmup_start_factor: float = 0.00001
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    initialize: bool = False

@dataclass
class AdamOptimizerConfig:
    type: str = 'adam'
    momentum: Optional[float] = None
    epsilon: Optional[float] = 1.0e-8
    learning_rate: float = 0.001
    warmup_start_factor: float = 0.00001
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    initialize: bool = False

@dataclass
class RAdamOptimizerConfig:
    type: str = 'radam'
    momentum: Optional[float] = None
    epsilon: Optional[float] = 1.0e-8
    learning_rate: float = 0.001
    warmup_start_factor: float = 0.00001
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    initialize: bool = False

@dataclass
class MTAdamOptimizerConfig:
    type: str = 'mtadam'
    momentum: Optional[float] = None
    epsilon: Optional[float] = 1.0e-8
    learning_rate: float = 0.001
    warmup_start_factor: float = 0.00001
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    initialize: bool = True

@dataclass
class LambOptimizerConfig:
    type: str = 'lamb'
    momentum: Optional[float] = None
    epsilon: Optional[float] = 1.0e-6
    learning_rate: float = 0.001
    warmup_start_factor: float = 0.00001
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    initialize: bool = False


_defaults = [
    { 'device': 'cuda' },
    { 'encoder': 'bert_base' },
    { 'decoder': 'double' },
    { 'optimizer': 'lamb' },
    '_self_'
]


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: _defaults)
    training_data: Path = MISSING
    num_workers: Optional[int] = None
    initial_model: Optional[Path] = None
    initial_model_prefix: Optional[Path] = None
    initial_model_index: Optional[int] = None
    reward_plugin: Path = MISSING
    discount_factor: float = 1.0
    expectile: float = MISSING
    v_loss_scaling: float = 1.0
    checkpointing: bool = False
    batch_size: int = MISSING
    gradient_accumulation_steps: int = 1
    max_gradient_norm: float = 1.0
    target_update_interval: int = 1
    target_update_rate: float = 0.1
    snapshot_interval: int = 0


config_store = ConfigStore.instance()
config_store.store(name='cpu', node=CpuConfig, group='device')
config_store.store(name='cuda', node=CudaConfig, group='device')
config_store.store(name='bert_base', node=BertBaseEncoderConfig, group='encoder')
config_store.store(name='bert_large', node=BertLargeEncoderConfig, group='encoder')
config_store.store(name='single', node=SingleDecoderConfig, group='decoder')
config_store.store(name='double', node=DoubleDecoderConfig, group='decoder')
config_store.store(name='triple', node=TripleDecoderConfig, group='decoder')
config_store.store(name='sgd', node=SgdOptimizerConfig, group='optimizer')
config_store.store(name='adam', node=AdamOptimizerConfig, group='optimizer')
config_store.store(name='radam', node=RAdamOptimizerConfig, group='optimizer')
config_store.store(name='mtadam', node=MTAdamOptimizerConfig, group='optimizer')
config_store.store(name='lamb', node=LambOptimizerConfig, group='optimizer')
config_store.store(name='config', node=Config)
