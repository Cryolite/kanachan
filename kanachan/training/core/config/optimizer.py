import logging
from typing import Any
from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore
from torch import nn
from torch.optim import Optimizer, SGD, Adam, RAdam
from torch.distributed.optim import ZeroRedundancyOptimizer
import torch.optim.lr_scheduler as lr_scheduler
from apex.optimizers import FusedSGD, FusedAdam, FusedLAMB


@dataclass
class SgdOptimizerConfig:
    type: str = "sgd"
    momentum: float = 0.0
    epsilon: float | None = None
    learning_rate: float = MISSING
    warmup_start_lr: float = 1.0e-8
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    use_zero: bool = False
    initialize: bool = False


@dataclass
class AdamOptimizerConfig:
    type: str = "adam"
    momentum: float | None = None
    epsilon: float = 1.0e-8
    learning_rate: float = 1.0e-4
    warmup_start_lr: float = 1.0e-8
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    use_zero: bool = False
    initialize: bool = False


@dataclass
class RAdamOptimizerConfig:
    type: str = "radam"
    momentum: float | None = None
    epsilon: float = 1.0e-8
    learning_rate: float = 1.0e-4
    warmup_start_lr: float = 1.0e-8
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    use_zero: bool = False
    initialize: bool = False


@dataclass
class LambOptimizerConfig:
    type: str = "lamb"
    momentum: float | None = None
    epsilon: float = 1.0e-6
    learning_rate: float = 1.0e-4
    warmup_start_lr: float = 1.0e-8
    warmup_steps: int = 0
    annealing_steps: int = 0
    annealing_steps_factor: int = 1
    use_zero: bool = False
    initialize: bool = False


config_store = ConfigStore.instance()
config_store.store(name="sgd", node=SgdOptimizerConfig, group="optimizer")
config_store.store(name="adam", node=AdamOptimizerConfig, group="optimizer")
config_store.store(name="radam", node=RAdamOptimizerConfig, group="optimizer")
config_store.store(name="lamb", node=LambOptimizerConfig, group="optimizer")


def validate(config: Any):
    if config.optimizer.type in ("sgd",):
        if config.optimizer.momentum < 0.0 or 1.0 <= config.optimizer.momentum:
            errmsg = (
                f"{config.optimizer.momentum}: `optimizer.momentum`"
                " must be a real value within the range [0.0, 1.0)."
            )
            raise RuntimeError(errmsg)
    else:
        if config.optimizer.momentum is not None:
            errmsg = (
                "`optimizer.momentum` is useless for"
                f" `{config.optimizer.type}`."
            )
            raise RuntimeError(errmsg)

    if config.optimizer.epsilon <= 0.0:
        errmsg = (
            f"{config.optimizer.epsilon}: `optimizer.epsilon` must be a"
            " non-negative real value."
        )
        raise RuntimeError(errmsg)

    if config.optimizer.learning_rate <= 0.0:
        errmsg = (
            f"{config.optimizer.learning_rate}: "
            "`optimizer.learning_rate` must be a positive real value."
        )
        raise RuntimeError(errmsg)

    if config.optimizer.warmup_start_lr <= 0.0:
        errmsg = (
            f"{config.optimizer.warmup_start_lr}: "
            "`optimizer.warmup_start_lr` must be a positive real value."
        )
        raise RuntimeError(errmsg)
    if config.optimizer.warmup_start_lr > config.optimizer.learning_rate:
        errmsg = (
            f"{config.optimizer.warmup_start_lr}:"
            " `config.optimizer.warmup_start_lr` must be less than"
            " `config.optimizer.learning_rate`."
        )
        raise RuntimeError(errmsg)

    if config.optimizer.warmup_steps < 0:
        errmsg = (
            f"{config.optimizer.warmup_steps}: `optimizer.warmup_steps`"
            " must be a non-negative integer."
        )
        raise RuntimeError(errmsg)

    if config.optimizer.annealing_steps < 0:
        errmsg = (
            f"{config.optimizer.annealing_steps}:"
            " `optimizer.annealing_steps` must be a non-negative"
            " integer."
        )
        raise RuntimeError(errmsg)

    if config.optimizer.annealing_steps_factor <= 0:
        errmsg = (
            f"{config.optimizer.annealing_steps_factor}: "
            "`optimizer.annealing_steps_factor` must be a positive"
            " integer."
        )
        raise RuntimeError(errmsg)


def dump(config: Any) -> None:
    logging.info("Optimizer: %s", config.optimizer.type)
    if config.optimizer.type in ("sgd",):
        logging.info("Momentum factor: %f", config.optimizer.momentum)
    if config.optimizer.type in ("adam", "radam", "mtadam", "lamb"):
        logging.info("Epsilon parameter: %E", config.optimizer.epsilon)
    logging.info("Learning rate: %E", config.optimizer.learning_rate)
    if config.optimizer.warmup_steps == 0:
        logging.info("LR warm-up: (disabled)")
    else:
        logging.info(
            "LR warm-up start LR: %E", config.optimizer.warmup_start_lr
        )
        logging.info("LR warm-up steps: %d", config.optimizer.warmup_steps)
    if config.optimizer.annealing_steps == 0:
        logging.info("LR annealing: (disabled)")
    else:
        logging.info(
            "LR annealing steps: %d", config.optimizer.annealing_steps
        )
        logging.info(
            "LR annealing steps factor: %d",
            config.optimizer.annealing_steps_factor,
        )
    logging.info("Use ZeRO: %s", config.optimizer.use_zero)


def create(
    config: Any, module: nn.Module
) -> tuple[Optimizer, lr_scheduler.LRScheduler | None]:
    if config.optimizer.type == "sgd":
        if config.device.type == "cpu":
            optimizer_class = SGD
        else:
            optimizer_class = FusedSGD
        optimizer_kwargs = {
            "lr": config.optimizer.learning_rate,
            "momentum": config.optimizer.momentum,
        }
    elif config.optimizer.type == "adam":
        if config.device.type == "cpu":
            optimizer_class = Adam
        else:
            optimizer_class = FusedAdam
        optimizer_kwargs = {
            "lr": config.optimizer.learning_rate,
            "eps": config.optimizer.epsilon,
        }
    elif config.optimizer.type == "radam":
        optimizer_class = RAdam
        optimizer_kwargs = {
            "lr": config.optimizer.learning_rate,
            "eps": config.optimizer.epsilon,
        }
    elif config.optimizer.type == "lamb":
        optimizer_class = FusedLAMB
        optimizer_kwargs = {
            "lr": config.optimizer.learning_rate,
            "eps": config.optimizer.epsilon,
        }
    else:
        raise NotImplementedError(config.optimizer.type)

    if config.optimizer.use_zero:
        optimizer = ZeroRedundancyOptimizer(
            module.parameters(), optimizer_class, **optimizer_kwargs
        )
    else:
        optimizer = optimizer_class(module.parameters(), **optimizer_kwargs)

    if config.optimizer.warmup_steps == 0:
        warmup_scheduler = None
    else:
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=config.optimizer.warmup_start_lr
            / config.optimizer.learning_rate,
            total_iters=config.optimizer.warmup_steps,
        )
    if config.optimizer.annealing_steps == 0:
        annealing_scheduler = None
    else:
        annealing_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            config.optimizer.annealing_steps,
            config.optimizer.annealing_steps_factor,
        )
    if warmup_scheduler is None and annealing_scheduler is None:
        scheduler = None
    elif warmup_scheduler is not None and annealing_scheduler is not None:
        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            [warmup_scheduler, annealing_scheduler],
            [warmup_scheduler.total_iters],
        )
    elif warmup_scheduler is not None:
        assert annealing_scheduler is None
        scheduler = warmup_scheduler
    else:
        assert warmup_scheduler is None
        assert annealing_scheduler is not None
        scheduler = annealing_scheduler

    return optimizer_class(module.parameters(), **optimizer_kwargs), scheduler
