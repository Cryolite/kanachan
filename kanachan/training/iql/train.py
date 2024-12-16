#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
import logging
import sys
from typing import Callable, Any
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from tensordict import TensorDict  # type: ignore
from tensordict.nn import TensorDictModule, TensorDictSequential  # type: ignore
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.amp.grad_scaler import GradScaler
from torch.distributed import (
    init_process_group,
    broadcast,
    ReduceOp,
    all_reduce,
)
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard.writer import SummaryWriter
import kanachan.training.iql.config  # pylint: disable=unused-import
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES
from kanachan.training.common import (
    get_distributed_environment,
    is_gradient_nan,
    get_gradient,
)
from kanachan.training.core.offline_rl import DataLoader, EpisodeReplayBuffer
from kanachan.nn import Encoder, Decoder, TwinQActor
from kanachan.model_loader import dump_model, dump_object
import kanachan.training.core.config as _config


def _training(
    *,
    training_data: Path,
    contiguous_training_data: bool,
    rewrite_rooms: int | None,
    rewrite_grades: int | None,
    replay_buffer_size: int,
    num_workers: int,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    value_network: nn.Module,
    q1_source_network: nn.Module,
    q2_source_network: nn.Module,
    q1_target_network: nn.Module,
    q2_target_network: nn.Module,
    reward_plugin: Path,
    discount_factor: float,
    expectile: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    v_max_gradient_norm: float,
    q_max_gradient_norm: float,
    value_optimizer: Optimizer,
    q1_optimizer: Optimizer,
    q2_optimizer: Optimizer,
    value_lr_scheduler: LRScheduler | None,
    q1_lr_scheduler: LRScheduler | None,
    q2_lr_scheduler: LRScheduler | None,
    target_update_interval: int,
    target_update_rate: float,
    snapshot_interval: int,
    num_samples: int,
    summary_writer: SummaryWriter,
    snapshot_writer: Callable[[int | None], None],
) -> None:
    start_time = datetime.datetime.now()

    world_size, _, local_rank = get_distributed_environment()

    is_amp_enabled = device != "cpu" and dtype != amp_dtype

    # Load the reward plugin.
    with open(reward_plugin, encoding="UTF-8") as file_pointer:
        exec(file_pointer.read(), globals())  # pylint: disable=exec-used

    if replay_buffer_size >= 1:
        data_loader = EpisodeReplayBuffer(
            training_data=training_data,
            contiguous_training_data=contiguous_training_data,
            num_skip_samples=num_samples,
            rewrite_rooms=rewrite_rooms,
            rewrite_grades=rewrite_grades,
            # pylint: disable=undefined-variable
            get_reward=get_reward,  # type: ignore # noqa: F821
            discount_factor=discount_factor,
            dtype=dtype,
            max_size=replay_buffer_size,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(num_workers >= 1),
        )
    else:
        data_loader = DataLoader(
            path=training_data,
            num_skip_samples=num_samples,
            rewrite_rooms=rewrite_rooms,
            rewrite_grades=rewrite_grades,
            # pylint: disable=undefined-variable
            get_reward=get_reward,  # type: ignore # noqa: F821
            dtype=dtype,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(num_workers >= 1),
            drop_last=(world_size >= 2),
        )

    last_snapshot: int | None = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    is_amp_enabled = device != "cpu" and dtype != amp_dtype
    autocast_kwargs: dict[str, Any] = {
        "device_type": device.type,
        "dtype": amp_dtype,
        "enabled": is_amp_enabled,
    }
    grad_scaler = GradScaler("cuda", enabled=is_amp_enabled)

    batch_count = 0

    for data in data_loader:
        data = data.to(device=device)
        assert isinstance(data, TensorDict)

        local_batch_size: int = data.size(0)  # type: ignore

        action = data["action"]
        assert action.device == device
        assert action.dtype == torch.int32
        assert action.dim() == 1
        assert action.size(0) == data.size(0)

        done = data["next", "done"]
        assert done.device == device
        assert done.dtype == torch.bool
        assert done.dim() == 1
        assert done.size(0) == data.size(0)

        def _compute_q_target(
            q_target_model: nn.Module,
        ) -> tuple[torch.Tensor, float]:
            copy = TensorDict(
                {
                    "sparse": data["sparse"].detach().clone(),
                    "numeric": data["numeric"].detach().clone(),
                    "progression": data["progression"].detach().clone(),
                    "candidates": data["candidates"].detach().clone(),
                },
                batch_size=data.size(0),
                device=device,
            )
            with torch.autocast(**autocast_kwargs):
                q_target_model(copy)
            q = copy["action_value"]
            assert q.dim() == 2
            assert q.size(0) == data.size(0)
            assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
            q = q[torch.arange(local_batch_size), action]
            assert q.dim() == 1
            assert q.size(0) == data.size(0)

            q_batch_mean = q.detach().clone().mean()
            if world_size >= 2:
                all_reduce(q_batch_mean, ReduceOp.AVG)

            return q, q_batch_mean.item()

        # Get the Q target value to compute the loss of the V model.
        with torch.no_grad():
            q1, q1_mean = _compute_q_target(q1_target_network)
            q2, q2_mean = _compute_q_target(q2_target_network)

            q = torch.minimum(q1, q2)
            q = q.detach().clone()
            assert q.dim() == 1
            assert q.size(0) == data.size(0)

        # Backprop for the V model.
        copy = TensorDict(
            {
                "sparse": data["sparse"].detach().clone(),
                "numeric": data["numeric"].detach().clone(),
                "progression": data["progression"].detach().clone(),
                "candidates": data["candidates"].detach().clone(),
            },
            batch_size=data.size(0),
            device=device,
        )
        with torch.autocast(**autocast_kwargs):
            value_network(copy)
        value = copy["state_value"]
        assert value.dim() == 1
        assert value.size(0) == data.size(0)

        value_mean = value.detach().clone().mean()
        if world_size >= 2:
            all_reduce(value_mean, ReduceOp.AVG)

        value_loss = q - value
        value_loss = torch.where(
            value_loss < 0.0,
            (1.0 - expectile) * (value_loss**2.0),
            expectile * (value_loss**2.0),
        )
        value_loss = torch.mean(value_loss)
        if math.isnan(value_loss.item()):
            raise RuntimeError("Value loss becomes NaN.")

        _value_loss = value_loss.detach().clone()
        if world_size >= 2:
            all_reduce(_value_loss, ReduceOp.AVG)
        value_loss_to_display = _value_loss.item()

        value_loss /= gradient_accumulation_steps
        grad_scaler.scale(value_loss).backward()

        # Get the reward to compute the loss of the Q source models.
        reward = data["next", "reward"]

        # Get V to compute the loss of the Q source models.
        copy = TensorDict(
            {
                "sparse": data["next", "sparse"].detach().clone(),
                "numeric": data["next", "numeric"].detach().clone(),
                "progression": data["next", "progression"].detach().clone(),
                "candidates": data["next", "candidates"].detach().clone(),
            },
            batch_size=data.size(0),
            device=device,
        )
        with torch.no_grad(), torch.autocast(**autocast_kwargs):
            value_network(copy)
            value = copy["state_value"]
            assert value.dim() == 1
            assert value.size(0) == data.size(0)
            value = torch.where(done, torch.zeros_like(value), value)
            value *= discount_factor
            value = value.detach().clone()

        def _backprop(q_source_network: nn.Module) -> float:
            copy = TensorDict(
                {
                    "sparse": data["sparse"].detach().clone(),
                    "numeric": data["numeric"].detach().clone(),
                    "progression": data["progression"].detach().clone(),
                    "candidates": data["candidates"].detach().clone(),
                },
                batch_size=data.size(0),
                device=device,
            )
            with torch.autocast(**autocast_kwargs):
                q_source_network(copy)
            q = copy["action_value"]
            assert q.dim() == 2
            assert q.size(0) == data.size(0)
            assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
            q = q[torch.arange(local_batch_size), action]

            q_loss = reward + value - q
            q_loss = q_loss**2.0
            q_loss = torch.mean(q_loss)
            if math.isnan(q_loss.item()):
                raise RuntimeError("Q loss becomes NaN.")

            _q_loss = q_loss.detach().clone()
            if world_size >= 2:
                all_reduce(_q_loss, ReduceOp.AVG)
            q_loss_to_display = _q_loss.item()

            q_loss /= gradient_accumulation_steps
            grad_scaler.scale(q_loss).backward()

            return q_loss_to_display

        # Backprop for the Q1 source network.
        q1_loss_to_display = _backprop(q1_source_network)

        # Backprop for the Q2 source network.
        q2_loss_to_display = _backprop(q2_source_network)

        num_samples += batch_size * world_size
        batch_count += 1

        def _step(
            network: nn.Module,
            max_gradient_norm: float,
            optimizer: Optimizer,
            scheduler,
        ) -> float:
            grad_scaler.unscale_(optimizer)
            gradient = get_gradient(network)
            # pylint: disable=not-callable
            gradient_norm = float(torch.linalg.vector_norm(gradient).item())
            nn.utils.clip_grad_norm_(
                network.parameters(),
                max_gradient_norm,
                error_if_nonfinite=False,
            )
            grad_scaler.step(optimizer)
            if scheduler is not None:
                scheduler.step()

            return gradient_norm

        if batch_count % gradient_accumulation_steps == 0:
            is_grad_nan = (
                is_gradient_nan(value_network)
                or is_gradient_nan(q1_source_network)
                or is_gradient_nan(q2_source_network)
            )
            is_grad_nan = torch.where(
                is_grad_nan,
                torch.ones_like(is_grad_nan),
                torch.zeros_like(is_grad_nan),
            )
            all_reduce(is_grad_nan)
            if is_grad_nan.item() >= 1:
                logging.warning(
                    "Skip an optimization step because of a NaN in the gradient."
                )
                value_optimizer.zero_grad()
                q1_optimizer.zero_grad()
                q2_optimizer.zero_grad()
                continue

            v_gradient_norm = _step(
                value_network,
                v_max_gradient_norm,
                value_optimizer,
                value_lr_scheduler,
            )
            q1_gradient_norm = _step(
                q1_source_network,
                q_max_gradient_norm,
                q1_optimizer,
                q1_lr_scheduler,
            )
            q2_gradient_norm = _step(
                q2_source_network,
                q_max_gradient_norm,
                q2_optimizer,
                q2_lr_scheduler,
            )
            grad_scaler.update()

            value_optimizer.zero_grad()
            q1_optimizer.zero_grad()
            q2_optimizer.zero_grad()

            if (
                batch_count
                % (gradient_accumulation_steps * target_update_interval)
                == 0
            ):
                with torch.no_grad():
                    for _param, _target_param in zip(
                        q1_source_network.parameters(),
                        q1_target_network.parameters(),
                    ):
                        _target_param.data *= 1.0 - target_update_rate
                        _target_param.data += (
                            target_update_rate * _param.data.detach().clone()
                        )
                    for _param, _target_param in zip(
                        q2_source_network.parameters(),
                        q2_target_network.parameters(),
                    ):
                        _target_param.data *= 1.0 - target_update_rate
                        _target_param.data += (
                            target_update_rate * _param.data.detach().clone()
                        )

            if local_rank == 0:
                logging.info(
                    "sample = %d, V loss = %E, Q1 loss = %E, Q2 loss = %E, "
                    "V gradient norm = %E, Q1 gradient norm = %E, "
                    "Q2 gradient norm = %E",
                    num_samples,
                    value_loss_to_display,
                    q1_loss_to_display,
                    q2_loss_to_display,
                    v_gradient_norm,
                    q1_gradient_norm,
                    q2_gradient_norm,
                )
                summary_writer.add_scalars(
                    "Q", {"Q1": q1_mean, "Q2": q2_mean}, num_samples
                )
                summary_writer.add_scalars(
                    "Q Gradient Norm",
                    {"Q1": q1_gradient_norm, "Q2": q2_gradient_norm},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "Q Loss",
                    {"Q1": q1_loss_to_display, "Q2": q2_loss_to_display},
                    num_samples,
                )
                summary_writer.add_scalar(
                    "Value", value_mean.item(), num_samples
                )
                summary_writer.add_scalar(
                    "Value Gradient Norm", v_gradient_norm, num_samples
                )
                summary_writer.add_scalar(
                    "Value Loss", value_loss_to_display, num_samples
                )
        else:
            if local_rank == 0:
                logging.info(
                    "sample = %d, value loss = %E, Q1 loss = %E, Q2 loss = %E",
                    num_samples,
                    value_loss_to_display,
                    q1_loss_to_display,
                    q2_loss_to_display,
                )
                summary_writer.add_scalars(
                    "Q", {"Q1": q1_mean, "Q2": q2_mean}, num_samples
                )
                summary_writer.add_scalars(
                    "Q Loss",
                    {"Q1": q1_loss_to_display, "Q2": q2_loss_to_display},
                    num_samples,
                )
                summary_writer.add_scalar(
                    "Value", value_mean.item(), num_samples
                )
                summary_writer.add_scalar(
                    "Value Loss", value_loss_to_display, num_samples
                )

        if (
            local_rank == 0
            and last_snapshot is not None
            and num_samples - last_snapshot >= snapshot_interval
        ):
            snapshot_writer(num_samples)
            last_snapshot = num_samples

    elapsed_time = datetime.datetime.now() - start_time

    if local_rank == 0:
        logging.info(
            "A training has finished (elapsed time = %s).", elapsed_time
        )
        snapshot_writer(None)


@hydra.main(version_base=None, config_name="config")
def _main(config: DictConfig) -> None:
    (
        world_size,
        rank,
        local_rank,
        device,
        dtype,
        amp_dtype,
    ) = _config.device.validate(config)

    if not config.training_data.exists():
        errmsg = f"{config.training_data}: Does not exist."
        raise RuntimeError(errmsg)
    if not config.training_data.is_file():
        errmsg = f"{config.training_data}: Not a file."
        raise RuntimeError(errmsg)

    if isinstance(config.rewrite_rooms, str):
        config.rewrite_rooms = {
            "bronze": 0,
            "silver": 1,
            "gold": 2,
            "jade": 3,
            "throne": 4,
        }[config.rewrite_rooms]
    if config.rewrite_rooms is not None and (
        config.rewrite_rooms < 0 or 4 < config.rewrite_rooms
    ):
        errmsg = (
            f"{config.rewrite_rooms}: "
            "`rewrite_rooms` must be an integer within the range [0, 4]."
        )
        raise RuntimeError(errmsg)

    if isinstance(config.rewrite_grades, str):
        config.rewrite_grades = {
            "novice1": 0,
            "novice2": 1,
            "novice3": 2,
            "adept1": 3,
            "adept2": 4,
            "adept3": 5,
            "expert1": 6,
            "expert2": 7,
            "expert3": 8,
            "master1": 9,
            "master2": 10,
            "master3": 11,
            "saint1": 12,
            "saint2": 13,
            "saint3": 14,
            "celestial": 15,
        }[config.rewrite_grades]
    if config.rewrite_grades is not None and (
        config.rewrite_grades < 0 or 15 < config.rewrite_grades
    ):
        errmsg = (
            f"{config.rewrite_grades}: "
            "`rewrite_grades` must be an integer within the range [0, 15]."
        )
        raise RuntimeError(errmsg)

    if device.type == "cpu":
        if config.num_workers is None:
            config.num_workers = 0
        if config.num_workers < 0:
            errmsg = f"{config.num_workers}: An invalid number of workers."
            raise RuntimeError(errmsg)
        if config.num_workers > 0:
            errmsg = (
                f"{config.num_workers}: An invalid number of workers"
                " for CPU."
            )
            raise RuntimeError(errmsg)
    else:
        assert device.type == "cuda"
        if config.num_workers is None:
            config.num_workers = 2
        if config.num_workers < 0:
            errmsg = f"{config.num_workers}: An invalid number of workers."
            raise RuntimeError(errmsg)
        if config.num_workers == 0:
            errmsg = (
                f"{config.num_workers}: An invalid number of workers for GPU."
            )
            raise RuntimeError(errmsg)

    if config.replay_buffer_size < 0:
        errmsg = (
            f"{config.replay_buffer_size}: `replay_buffer_size` must be"
            " a non-negative integer."
        )
        raise RuntimeError(errmsg)
    if config.contiguous_training_data and config.replay_buffer_size == 0:
        errmsg = "Use `replay_buffer_size` for `contiguous_training_data`."
        raise RuntimeError(errmsg)

    _config.encoder.validate(config)

    _config.decoder.validate(config)

    if config.initial_model_prefix is not None:
        if config.encoder.load_from is not None:
            errmsg = (
                "`initial_model_prefix` conflicts with `encoder.load_from`."
            )
            raise RuntimeError(errmsg)
        if not config.initial_model_prefix.exists():
            errmsg = f"{config.initial_model_prefix}: Does not exist."
            raise RuntimeError(errmsg)
        if not config.initial_model_prefix.is_dir():
            errmsg = f"{config.initial_model_prefix}: Not a directory."
            raise RuntimeError(errmsg)

    if config.initial_model_index is not None:
        if config.initial_model_prefix is None:
            errmsg = (
                "`initial_model_index` must be combined with"
                " `initial_model_prefix`."
            )
            raise RuntimeError(errmsg)
        if config.initial_model_index < 0:
            errmsg = (
                f"{config.initial_model_index}: An invalid initial model"
                " index."
            )
            raise RuntimeError(errmsg)

    num_samples = 0
    value_encoder_snapshot_path: Path | None = None
    value_decoder_snapshot_path: Path | None = None
    q1_source_encoder_snapshot_path: Path | None = None
    q1_source_decoder_snapshot_path: Path | None = None
    q2_source_encoder_snapshot_path: Path | None = None
    q2_source_decoder_snapshot_path: Path | None = None
    q1_target_encoder_snapshot_path: Path | None = None
    q1_target_decoder_snapshot_path: Path | None = None
    q2_target_encoder_snapshot_path: Path | None = None
    q2_target_decoder_snapshot_path: Path | None = None
    value_optimizer_snapshot_path: Path | None = None
    q1_optimizer_snapshot_path: Path | None = None
    q2_optimizer_snapshot_path: Path | None = None
    value_lr_scheduler_snapshot_path: Path | None = None
    q1_lr_scheduler_snapshot_path: Path | None = None
    q2_lr_scheduler_snapshot_path: Path | None = None

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    "^(?:(?:(?:value|q[12]-source|q[12]-target)-(?:encoder|decoder))|value-optimizer|q[12]-optimizer|value-lr-scheduler|q[12]-lr-scheduler)(?:\\.(\\d+))?\\.pth$",
                    child,
                )
                if match is None:
                    continue
                if match[1] is None:
                    config.initial_model_index = sys.maxsize
                    continue
                if (
                    config.initial_model_index is None
                    or int(match[1]) > config.initial_model_index
                ):
                    config.initial_model_index = int(match[1])
                    continue
        if config.initial_model_index is None:
            errmsg = f"{config.initial_model_prefix}: No model snapshot found."
            raise RuntimeError(errmsg)

        if config.initial_model_index == sys.maxsize:
            config.initial_model_index = 0
            infix = ""
        else:
            num_samples = config.initial_model_index
            infix = f".{num_samples}"

        value_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"value-encoder{infix}.pth"
        )
        if not value_encoder_snapshot_path.exists():
            errmsg = f"{value_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not value_encoder_snapshot_path.is_file():
            errmsg = f"{value_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        value_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"value-decoder{infix}.pth"
        )
        if not value_decoder_snapshot_path.exists():
            errmsg = f"{value_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not value_decoder_snapshot_path.is_file():
            errmsg = f"{value_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q1_source_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"q1-source-encoder{infix}.pth"
        )
        if not q1_source_encoder_snapshot_path.exists():
            errmsg = f"{q1_source_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q1_source_encoder_snapshot_path.is_file():
            errmsg = f"{q1_source_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q1_source_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"q1-source-decoder{infix}.pth"
        )
        if not q1_source_decoder_snapshot_path.exists():
            errmsg = f"{q1_source_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q1_source_decoder_snapshot_path.is_file():
            errmsg = f"{q1_source_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q2_source_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"q2-source-encoder{infix}.pth"
        )
        if not q2_source_encoder_snapshot_path.exists():
            errmsg = f"{q2_source_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q2_source_encoder_snapshot_path.is_file():
            errmsg = f"{q2_source_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q2_source_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"q2-source-decoder{infix}.pth"
        )
        if not q2_source_decoder_snapshot_path.exists():
            errmsg = f"{q2_source_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q2_source_decoder_snapshot_path.is_file():
            errmsg = f"{q2_source_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q1_target_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"q1-target-encoder{infix}.pth"
        )
        if not q1_target_encoder_snapshot_path.exists():
            errmsg = f"{q1_target_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q1_target_encoder_snapshot_path.is_file():
            errmsg = f"{q1_target_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q1_target_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"q1-target-decoder{infix}.pth"
        )
        if not q1_target_decoder_snapshot_path.exists():
            errmsg = f"{q1_target_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q1_target_decoder_snapshot_path.is_file():
            errmsg = f"{q1_target_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q2_target_encoder_snapshot_path = Path(
            config.initial_model_prefix / f"q2-target-encoder{infix}.pth"
        )
        if not q2_target_encoder_snapshot_path.exists():
            errmsg = f"{q2_target_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q2_target_encoder_snapshot_path.is_file():
            errmsg = f"{q2_target_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        q2_target_decoder_snapshot_path = Path(
            config.initial_model_prefix / f"q2-target-decoder{infix}.pth"
        )
        if not q2_target_decoder_snapshot_path.exists():
            errmsg = f"{q2_target_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not q2_target_decoder_snapshot_path.is_file():
            errmsg = f"{q2_target_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        value_optimizer_snapshot_path = Path(
            config.initial_model_prefix / f"value-optimizer{infix}.pth"
        )
        if (
            not value_optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            value_optimizer_snapshot_path = None

        q1_optimizer_snapshot_path = Path(
            config.initial_model_prefix / f"q1-optimizer{infix}.pth"
        )
        if (
            not q1_optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            q1_optimizer_snapshot_path = None

        q2_optimizer_snapshot_path = Path(
            config.initial_model_prefix / f"q2-optimizer{infix}.pth"
        )
        if (
            not q2_optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            q2_optimizer_snapshot_path = None

        value_lr_scheduler_snapshot_path = Path(
            config.initial_model_prefix / f"value-lr-scheduler{infix}.pth"
        )
        if value_optimizer_snapshot_path is None:
            value_lr_scheduler_snapshot_path = None
        else:
            if not value_lr_scheduler_snapshot_path.exists():
                errmsg = f"{value_lr_scheduler_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not value_lr_scheduler_snapshot_path.is_file():
                errmsg = f"{value_lr_scheduler_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

        q1_lr_scheduler_snapshot_path = Path(
            config.initial_model_prefix / f"q1-lr-scheduler{infix}.pth"
        )
        if q1_optimizer_snapshot_path is None:
            q1_lr_scheduler_snapshot_path = None
        else:
            if not q1_lr_scheduler_snapshot_path.exists():
                errmsg = f"{q1_lr_scheduler_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not q1_lr_scheduler_snapshot_path.is_file():
                errmsg = f"{q1_lr_scheduler_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

        q2_lr_scheduler_snapshot_path = Path(
            config.initial_model_prefix / f"q2-lr-scheduler{infix}.pth"
        )
        if q2_optimizer_snapshot_path is None:
            q2_lr_scheduler_snapshot_path = None
        else:
            if not q2_lr_scheduler_snapshot_path.exists():
                errmsg = f"{q2_lr_scheduler_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not q2_lr_scheduler_snapshot_path.is_file():
                errmsg = f"{q2_lr_scheduler_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

    if not config.reward_plugin.exists():
        errmsg = f"{config.reward_plugin}: Does not exist."
        raise RuntimeError(errmsg)
    if not config.reward_plugin.is_file():
        errmsg = f"{config.reward_plugin}: Not a file."
        raise RuntimeError(errmsg)

    if config.discount_factor <= 0.0 or 1.0 < config.discount_factor:
        errmsg = (
            f"{config.discount_factor}: An invalid value for"
            " `discount_factor`."
        )
        raise RuntimeError(errmsg)

    if config.expectile <= 0.0 or 1.0 <= config.expectile:
        errmsg = f"{config.expectile}: An invalid value for `expectile`."
        raise RuntimeError(errmsg)

    if config.batch_size <= 0:
        errmsg = (
            f"{config.batch_size}: `batch_size` must be a positive integer."
        )
        raise RuntimeError(errmsg)
    if config.batch_size % world_size != 0:
        errmsg = (
            f"`batch_size` must be divisible by the world size ({world_size})."
        )
        raise RuntimeError(errmsg)

    if config.gradient_accumulation_steps <= 0:
        errmsg = (
            f"{config.gradient_accumulation_steps}: "
            "`gradient_accumulation_steps` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.q_max_gradient_norm <= 0.0:
        errmsg = (
            f"{config.q_max_gradient_norm}: `q_max_gradient_norm` must be a "
            "positive real number."
        )
        raise RuntimeError(errmsg)

    if config.v_max_gradient_norm <= 0.0:
        errmsg = (
            f"{config.v_max_gradient_norm}: `v_max_gradient_norm` must be a "
            "positive real number."
        )
        raise RuntimeError(errmsg)

    _config.optimizer.validate(config)

    if config.target_update_interval <= 0:
        errmsg = (
            f"{config.target_update_interval}: "
            "`target_update_interval` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.target_update_rate <= 0.0 or config.target_update_rate > 1.0:
        errmsg = (
            f"{config.target_update_rate}: `target_update_rate` must be a "
            "real value within the range (0.0, 1.0]."
        )
        raise RuntimeError(errmsg)

    if config.snapshot_interval < 0:
        errmsg = (
            f"{config.snapshot_interval}: `snapshot_interval` must be a"
            " non-negative integer."
        )
        raise RuntimeError(errmsg)

    output_prefix = Path(HydraConfig.get().runtime.output_dir)

    if local_rank == 0:
        logging.info("Model type: implicit Q-learning (IQL)")

        _config.device.dump(
            world_size=world_size,
            rank=rank,
            local_rank=local_rank,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
        )

        logging.info("Training data: %s", config.training_data)
        if num_samples > 0:
            logging.info(
                "# of training samples consumed so far: %d", num_samples
            )
        logging.info(
            "Training data is contiguous: %s", config.contiguous_training_data
        )
        if config.rewrite_rooms is not None:
            logging.info(
                "Rewrite the rooms in the training data to: %d",
                config.rewrite_rooms,
            )
        if config.rewrite_grades is not None:
            logging.info(
                "Rewrite the grades in the training data to: %d",
                config.rewrite_grades,
            )
        logging.info("# of workers: %d", config.num_workers)
        if config.replay_buffer_size > 0:
            logging.info("Replay buffer size: %d", config.replay_buffer_size)

        _config.encoder.dump(config)

        _config.decoder.dump(config)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial model prefix: %s", config.initial_model_prefix
            )
            logging.info("Initlal model index: %d", config.initial_model_index)
            if config.optimizer.initialize:
                logging.info("(Will not load optimizer)")

        logging.info("Reward plugin: %s", config.reward_plugin)
        logging.info("Double Q-learning: %s", config.double_q_learning)
        logging.info("Discount factor: %f", config.discount_factor)
        logging.info("Expectile: %f", config.expectile)
        logging.info("Checkpointing: %s", config.checkpointing)
        logging.info("Batch size: %d", config.batch_size)
        logging.info(
            "# of steps for gradient accumulation: %d",
            config.gradient_accumulation_steps,
        )
        logging.info(
            "Virtual batch size: %d",
            config.batch_size * config.gradient_accumulation_steps,
        )
        logging.info(
            "Norm threshold for gradient clipping on Q: %E",
            config.q_max_gradient_norm,
        )
        logging.info(
            "Norm threshold for gradient clipping on V: %E",
            config.v_max_gradient_norm,
        )

        _config.optimizer.dump(config)

        logging.info(
            "Target update interval: %d", config.target_update_interval
        )
        logging.info("Target update rate: %f", config.target_update_rate)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial value encoder snapshot: %s",
                value_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q1 source encoder network snapshot: %s",
                q1_source_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q1 source decoder network snapshot: %s",
                q1_source_decoder_snapshot_path,
            )
            logging.info(
                "Initial Q2 source encoder network snapshot: %s",
                q2_source_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q2 source decoder network snapshot: %s",
                q2_source_decoder_snapshot_path,
            )
            logging.info(
                "Initial Q1 target encoder network snapshot: %s",
                q1_target_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q1 target decoder network snapshot: %s",
                q1_target_decoder_snapshot_path,
            )
            logging.info(
                "Initial Q2 target encoder network snapshot: %s",
                q2_target_encoder_snapshot_path,
            )
            logging.info(
                "Initial Q2 target decoder network snapshot: %s",
                q2_target_decoder_snapshot_path,
            )
            if value_optimizer_snapshot_path is not None:
                logging.info(
                    "Initial value optimizer snapshot: %s",
                    value_optimizer_snapshot_path,
                )
                logging.info(
                    "Initial value LR scheduler snapshot: %s",
                    value_lr_scheduler_snapshot_path,
                )
            if q1_optimizer_snapshot_path is not None:
                logging.info(
                    "Initial Q1 optimizer snapshot: %s",
                    q1_optimizer_snapshot_path,
                )
                logging.info(
                    "Initial Q1 LR scheduler snapshot: %s",
                    q1_lr_scheduler_snapshot_path,
                )
            if q2_optimizer_snapshot_path is not None:
                logging.info(
                    "Initial Q2 optimizer snapshot: %s",
                    q2_optimizer_snapshot_path,
                )
                logging.info(
                    "Initial Q2 LR scheduler snapshot: %s",
                    q2_lr_scheduler_snapshot_path,
                )

        logging.info("Output prefix: %s", output_prefix)
        if config.snapshot_interval == 0:
            logging.info("Snapshot interval: N/A")
        else:
            logging.info("Snapshot interval: %d", config.snapshot_interval)

    if world_size >= 2:
        init_process_group(backend="nccl")

    value_encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=device,
        dtype=dtype,
    )
    value_encoder_tdm = TensorDictModule(
        value_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    value_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="state",
        noise_init_std=None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        for _param in value_decoder.parameters():
            _param.zero_()
    value_decoder_tdm = TensorDictModule(
        value_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["state_value"],  # type: ignore
    )
    value_network = TensorDictSequential(value_encoder_tdm, value_decoder_tdm)
    if world_size >= 2:
        value_network.to(device=device)
        for _param in value_network.parameters():
            broadcast(_param.data, src=0)
        value_network.to(device="cpu")

    q1_source_encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=device,
        dtype=dtype,
    )
    q1_source_encoder_tdm = TensorDictModule(
        q1_source_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    q1_source_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="candidates",
        noise_init_std=None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        for _param in q1_source_decoder.parameters():
            _param.zero_()
    q1_source_decoder_tdm = TensorDictModule(
        q1_source_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    q1_source_network = TensorDictSequential(
        q1_source_encoder_tdm, q1_source_decoder_tdm
    )
    if world_size >= 2:
        q1_source_network.to(device=device)
        for _param in q1_source_network.parameters():
            broadcast(_param.data, src=0)
        q1_source_network.to(device="cpu")

    q2_source_encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=device,
        dtype=dtype,
    )
    q2_source_encoder_tdm = TensorDictModule(
        q2_source_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    q2_source_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="candidates",
        noise_init_std=None,
        device=device,
        dtype=dtype,
    )
    with torch.no_grad():
        for _param in q2_source_decoder.parameters():
            _param.zero_()
    q2_source_decoder_tdm = TensorDictModule(
        q2_source_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    q2_source_network = TensorDictSequential(
        q2_source_encoder_tdm, q2_source_decoder_tdm
    )
    if world_size >= 2:
        q2_source_network.to(device=device)
        for _param in q2_source_network.parameters():
            broadcast(_param.data, src=0)
        q2_source_network.to(device="cpu")

    q1_target_encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=device,
        dtype=dtype,
    )
    q1_target_encoder_tdm = TensorDictModule(
        q1_target_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    q1_target_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="candidates",
        noise_init_std=None,
        device=device,
        dtype=dtype,
    )
    q1_target_decoder_tdm = TensorDictModule(
        q1_target_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    q1_target_network = TensorDictSequential(
        q1_target_encoder_tdm, q1_target_decoder_tdm
    )
    with torch.no_grad():
        for _param, _param_target in zip(
            q1_source_network.parameters(), q1_target_network.parameters()
        ):
            _param_target.data = _param.data.detach().clone()

    q2_target_encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=device,
        dtype=dtype,
    )
    q2_target_encoder_tdm = TensorDictModule(
        q2_target_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    q2_target_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="candidates",
        noise_init_std=None,
        device=device,
        dtype=dtype,
    )
    q2_target_decoder_tdm = TensorDictModule(
        q2_target_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    q2_target_network = TensorDictSequential(
        q2_target_encoder_tdm, q2_target_decoder_tdm
    )
    with torch.no_grad():
        for _param, _param_target in zip(
            q2_source_network.parameters(), q2_target_network.parameters()
        ):
            _param_target.data = _param.data.detach().clone()

    model_to_save = TwinQActor(q1_target_network, q2_target_network)
    model_to_save_tdm = TensorDictModule(
        model_to_save,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["action"],  # type: ignore
    )

    value_optimizer, value_lr_scheduler = _config.optimizer.create(
        device.type, config, value_network
    )
    q1_optimizer, q1_lr_scheduler = _config.optimizer.create(
        device.type, config, q1_source_network
    )
    q2_optimizer, q2_lr_scheduler = _config.optimizer.create(
        device.type, config, q2_source_network
    )

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(
            config.encoder.load_from, map_location="cpu", weights_only=True
        )
        value_encoder.load_state_dict(encoder_state_dict)
        q1_source_encoder.load_state_dict(encoder_state_dict)
        q2_source_encoder.load_state_dict(encoder_state_dict)
        q1_target_encoder.load_state_dict(encoder_state_dict)
        q2_target_encoder.load_state_dict(encoder_state_dict)

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert value_encoder_snapshot_path is not None
        assert value_decoder_snapshot_path is not None
        assert q1_source_encoder_snapshot_path is not None
        assert q1_source_decoder_snapshot_path is not None
        assert q2_source_encoder_snapshot_path is not None
        assert q2_source_decoder_snapshot_path is not None
        assert q1_target_encoder_snapshot_path is not None
        assert q1_target_decoder_snapshot_path is not None
        assert q2_target_encoder_snapshot_path is not None
        assert q2_target_decoder_snapshot_path is not None

        value_encoder_state_dict = torch.load(
            value_encoder_snapshot_path, map_location="cpu", weights_only=True
        )
        value_encoder.load_state_dict(value_encoder_state_dict)
        value_decoder_state_dict = torch.load(
            value_decoder_snapshot_path, map_location="cpu", weights_only=True
        )
        value_decoder.load_state_dict(value_decoder_state_dict)

        q1_source_encoder_state_dict = torch.load(
            q1_source_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q1_source_encoder.load_state_dict(q1_source_encoder_state_dict)
        q1_source_decoder_state_dict = torch.load(
            q1_source_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q1_source_decoder.load_state_dict(q1_source_decoder_state_dict)

        q2_source_encoder_state_dict = torch.load(
            q2_source_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q2_source_encoder.load_state_dict(q2_source_encoder_state_dict)
        q2_source_decoder_state_dict = torch.load(
            q2_source_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q2_source_decoder.load_state_dict(q2_source_decoder_state_dict)

        q1_target_encoder_state_dict = torch.load(
            q1_target_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q1_target_encoder.load_state_dict(q1_target_encoder_state_dict)
        q1_target_decoder_state_dict = torch.load(
            q1_target_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q1_target_decoder.load_state_dict(q1_target_decoder_state_dict)

        q2_target_encoder_state_dict = torch.load(
            q2_target_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q2_target_encoder.load_state_dict(q2_target_encoder_state_dict)
        q2_target_decoder_state_dict = torch.load(
            q2_target_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        q2_target_decoder.load_state_dict(q2_target_decoder_state_dict)

        if value_optimizer_snapshot_path is not None:
            assert value_lr_scheduler_snapshot_path is not None
            assert value_lr_scheduler is not None
            value_optimizer_state_dict = torch.load(
                value_optimizer_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            value_optimizer.load_state_dict(value_optimizer_state_dict)
            value_lr_scheduler_state_dict = torch.load(
                value_lr_scheduler_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            value_lr_scheduler.load_state_dict(value_lr_scheduler_state_dict)

        if q1_optimizer_snapshot_path is not None:
            assert q1_lr_scheduler_snapshot_path is not None
            assert q1_lr_scheduler is not None
            q1_optimizer_state_dict = torch.load(
                q1_optimizer_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            q1_optimizer.load_state_dict(q1_optimizer_state_dict)
            q1_lr_scheduler_state_dict = torch.load(
                q1_lr_scheduler_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            q1_lr_scheduler.load_state_dict(q1_lr_scheduler_state_dict)

        if q2_optimizer_snapshot_path is not None:
            assert q2_lr_scheduler_snapshot_path is not None
            assert q2_lr_scheduler is not None
            q2_optimizer_state_dict = torch.load(
                q2_optimizer_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            q2_optimizer.load_state_dict(q2_optimizer_state_dict)
            q2_lr_scheduler_state_dict = torch.load(
                q2_lr_scheduler_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            q2_lr_scheduler.load_state_dict(q2_lr_scheduler_state_dict)

    value_network.requires_grad_(True)
    value_network.train()
    value_network.to(device=device, dtype=dtype)
    if world_size >= 2:
        value_network = DistributedDataParallel(value_network)
        value_network = nn.SyncBatchNorm.convert_sync_batchnorm(value_network)

    q1_source_network.requires_grad_(True)
    q1_source_network.train()
    q1_source_network.to(device=device, dtype=dtype)
    if world_size >= 2:
        q1_source_network = DistributedDataParallel(q1_source_network)
        q1_source_network = nn.SyncBatchNorm.convert_sync_batchnorm(
            q1_source_network
        )

    q2_source_network.requires_grad_(True)
    q2_source_network.train()
    q2_source_network.to(device=device, dtype=dtype)
    if world_size >= 2:
        q2_source_network = DistributedDataParallel(q2_source_network)
        q2_source_network = nn.SyncBatchNorm.convert_sync_batchnorm(
            q2_source_network
        )

    q1_target_network.requires_grad_(False)
    q1_target_network.eval()
    q1_target_network.to(device=device, dtype=dtype)

    q2_target_network.requires_grad_(False)
    q2_target_network.eval()
    q2_target_network.to(device=device, dtype=dtype)

    snapshots_path = output_prefix / "snapshots"

    def snapshot_writer(num_samples: int | None) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)

        infix = "" if num_samples is None else f".{num_samples}"

        torch.save(
            value_encoder.state_dict(),
            snapshots_path / f"value-encoder{infix}.pth",
        )
        torch.save(
            value_decoder.state_dict(),
            snapshots_path / f"value-decoder{infix}.pth",
        )
        torch.save(
            q1_source_encoder.state_dict(),
            snapshots_path / f"q1-source-encoder{infix}.pth",
        )
        torch.save(
            q1_source_decoder.state_dict(),
            snapshots_path / f"q1-source-decoder{infix}.pth",
        )
        torch.save(
            q2_source_encoder.state_dict(),
            snapshots_path / f"q2-source-encoder{infix}.pth",
        )
        torch.save(
            q2_source_decoder.state_dict(),
            snapshots_path / f"q2-source-decoder{infix}.pth",
        )
        torch.save(
            q1_target_encoder.state_dict(),
            snapshots_path / f"q1-target-encoder{infix}.pth",
        )
        torch.save(
            q1_target_decoder.state_dict(),
            snapshots_path / f"q1-target-decoder{infix}.pth",
        )
        torch.save(
            q2_target_encoder.state_dict(),
            snapshots_path / f"q2-target-encoder{infix}.pth",
        )
        torch.save(
            q2_target_decoder.state_dict(),
            snapshots_path / f"q2-target-decoder{infix}.pth",
        )
        torch.save(
            value_optimizer.state_dict(),
            snapshots_path / f"value-optimizer{infix}.pth",
        )
        if value_lr_scheduler is not None:
            torch.save(
                value_lr_scheduler.state_dict(),
                snapshots_path / f"value-lr-scheduler{infix}.pth",
            )
        torch.save(
            q1_optimizer.state_dict(),
            snapshots_path / f"q1-optimizer{infix}.pth",
        )
        if q1_lr_scheduler is not None:
            torch.save(
                q1_lr_scheduler.state_dict(),
                snapshots_path / f"q1_lr_scheduler{infix}.pth",
            )
        torch.save(
            q2_optimizer.state_dict(),
            snapshots_path / f"q2-optimizer{infix}.pth",
        )
        if q2_lr_scheduler is not None:
            torch.save(
                q2_lr_scheduler.state_dict(),
                snapshots_path / f"q2_lr_scheduler{infix}.pth",
            )

        state = dump_object(
            model_to_save_tdm,
            [
                dump_object(
                    model_to_save,
                    [
                        dump_object(
                            q1_target_network,
                            [
                                dump_object(
                                    q1_target_encoder_tdm,
                                    [
                                        dump_model(
                                            q1_target_encoder,
                                            [],
                                            {
                                                "position_encoder": config.encoder.position_encoder,
                                                "dimension": config.encoder.dimension,
                                                "num_heads": config.encoder.num_heads,
                                                "dim_feedforward": config.encoder.dim_feedforward,
                                                "activation_function": config.encoder.activation_function,
                                                "dropout": config.encoder.dropout,
                                                "layer_normalization": config.encoder.layer_normalization,
                                                "num_layers": config.encoder.num_layers,
                                                "checkpointing": config.checkpointing,
                                                "device": torch.device("cpu"),
                                                "dtype": dtype,
                                            },
                                        ),
                                    ],
                                    {
                                        "in_keys": [
                                            "sparse",
                                            "numeric",
                                            "progression",
                                            "candidates",
                                        ],
                                        "out_keys": ["encode"],
                                    },
                                ),
                                dump_object(
                                    q1_target_decoder_tdm,
                                    [
                                        dump_model(
                                            q1_target_decoder,
                                            [],
                                            {
                                                "input_dimension": config.encoder.dimension,
                                                "dimension": config.decoder.dimension,
                                                "activation_function": config.decoder.activation_function,
                                                "dropout": config.decoder.dropout,
                                                "layer_normalization": config.decoder.layer_normalization,
                                                "num_layers": config.decoder.num_layers,
                                                "output_mode": "candidates",
                                                "noise_init_std": None,
                                                "device": torch.device("cpu"),
                                                "dtype": dtype,
                                            },
                                        ),
                                    ],
                                    {
                                        "in_keys": ["encode"],
                                        "out_keys": ["action_value"],
                                    },
                                ),
                            ],
                            {},
                        ),
                        dump_object(
                            q2_target_network,
                            [
                                dump_object(
                                    q2_target_encoder_tdm,
                                    [
                                        dump_model(
                                            q2_target_encoder,
                                            [],
                                            {
                                                "position_encoder": config.encoder.position_encoder,
                                                "dimension": config.encoder.dimension,
                                                "num_heads": config.encoder.num_heads,
                                                "dim_feedforward": config.encoder.dim_feedforward,
                                                "activation_function": config.encoder.activation_function,
                                                "dropout": config.encoder.dropout,
                                                "layer_normalization": config.encoder.layer_normalization,
                                                "num_layers": config.encoder.num_layers,
                                                "checkpointing": config.checkpointing,
                                                "device": torch.device("cpu"),
                                                "dtype": dtype,
                                            },
                                        ),
                                    ],
                                    {
                                        "in_keys": [
                                            "sparse",
                                            "numeric",
                                            "progression",
                                            "candidates",
                                        ],
                                        "out_keys": ["encode"],
                                    },
                                ),
                                dump_object(
                                    q2_target_decoder_tdm,
                                    [
                                        dump_model(
                                            q2_target_decoder,
                                            [],
                                            {
                                                "input_dimension": config.encoder.dimension,
                                                "dimension": config.decoder.dimension,
                                                "activation_function": config.decoder.activation_function,
                                                "dropout": config.decoder.dropout,
                                                "layer_normalization": config.decoder.layer_normalization,
                                                "num_layers": config.decoder.num_layers,
                                                "output_mode": "candidates",
                                                "noise_init_std": None,
                                                "device": torch.device("cpu"),
                                                "dtype": dtype,
                                            },
                                        ),
                                    ],
                                    {
                                        "in_keys": ["encode"],
                                        "out_keys": ["action_value"],
                                    },
                                ),
                            ],
                            {},
                        ),
                    ],
                    {},
                ),
            ],
            {
                "in_keys": ["sparse", "numeric", "progression", "candidates"],
                "out_keys": ["action"],
            },
        )
        torch.save(state, snapshots_path / f"model{infix}.kanachan")

    tensorboard_path = output_prefix / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        _training(
            training_data=config.training_data,
            contiguous_training_data=config.contiguous_training_data,
            rewrite_rooms=config.rewrite_rooms,
            rewrite_grades=config.rewrite_grades,
            replay_buffer_size=config.replay_buffer_size,
            num_workers=config.num_workers,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            value_network=value_network,
            q1_source_network=q1_source_network,
            q2_source_network=q2_source_network,
            q1_target_network=q1_target_network,
            q2_target_network=q2_target_network,
            reward_plugin=config.reward_plugin,
            discount_factor=config.discount_factor,
            expectile=config.expectile,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            v_max_gradient_norm=config.v_max_gradient_norm,
            q_max_gradient_norm=config.q_max_gradient_norm,
            value_optimizer=value_optimizer,
            q1_optimizer=q1_optimizer,
            q2_optimizer=q2_optimizer,
            value_lr_scheduler=value_lr_scheduler,
            q1_lr_scheduler=q1_lr_scheduler,
            q2_lr_scheduler=q2_lr_scheduler,
            target_update_interval=config.target_update_interval,
            target_update_rate=config.target_update_rate,
            snapshot_interval=config.snapshot_interval,
            num_samples=num_samples,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer,
        )


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
    sys.exit(0)
