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
import torch
from torch import Tensor
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.grad_scaler import GradScaler
from torch.distributed import (
    init_process_group,
    broadcast,
    ReduceOp,
    all_reduce,
)
from torch.utils.tensorboard.writer import SummaryWriter
from tensordict import TensorDict  # type: ignore
from tensordict.nn import TensorDictModule, TensorDictSequential  # type: ignore
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES
from kanachan.training.common import (
    get_distributed_environment,
    is_gradient_nan,
    get_gradient,
)
import kanachan.training.ilql.config  # pylint: disable=unused-import
from kanachan.training.core.offline_rl import DataLoader, EpisodeReplayBuffer
from kanachan.nn import Encoder, Decoder, TwinQActor
from kanachan.model_loader import dump_object, dump_model
import kanachan.training.core.config as _config


def _get_q_target(
    *, world_size: int, data: TensorDict, target_model: TensorDictModule
) -> tuple[Tensor, float]:
    batch_size = int(data.batch_size[0])

    copy: TensorDict = data.copy()
    assert isinstance(copy, TensorDict)
    copy = copy.detach()
    target_model(copy)

    q_target: Tensor = copy["action_value"]
    assert isinstance(q_target, Tensor)
    assert q_target.dim() == 2
    assert q_target.size(0) == batch_size
    assert q_target.size(1) == MAX_NUM_ACTION_CANDIDATES

    action: Tensor = copy["action"]
    assert isinstance(action, Tensor)
    assert action.dtype == torch.int32
    assert action.dim() == 1
    assert action.size(0) == batch_size

    q_target = q_target[torch.arange(batch_size), action]
    assert q_target.dim() == 1
    assert q_target.size(0) == batch_size

    q_batch_mean = q_target.detach().clone().mean()
    if world_size >= 2:
        all_reduce(q_batch_mean, ReduceOp.AVG)

    return q_target, q_batch_mean.item()


BackwardResult = tuple[Tensor, Tensor, float, float]


def _backward(
    *,
    world_size: int,
    autocast_kwargs: dict,
    data: TensorDict,
    source_model: nn.Module,
    discount_factor: float,
    expectile: float,
    v_loss_scaling: float,
    gradient_accumulation_steps: int,
    grad_scaler: GradScaler,
    q_target: Tensor,
) -> BackwardResult:
    batch_size = int(data.batch_size[0])

    copy: TensorDict = data.copy()
    assert isinstance(copy, TensorDict)
    copy = copy.detach()
    with torch.autocast(**autocast_kwargs):
        source_model(copy)

    q: Tensor = copy["action_value"]
    assert isinstance(q, Tensor)
    assert q.dim() == 2
    assert q.size(0) == batch_size
    assert q.size(1) == MAX_NUM_ACTION_CANDIDATES

    v: Tensor = copy["state_value"]
    assert isinstance(v, Tensor)
    assert v.dim() == 1
    assert v.size(0) == batch_size

    action: Tensor = copy["action"]
    assert isinstance(action, Tensor)
    assert action.dtype == torch.int32
    assert action.dim() == 1
    assert action.size(0) == batch_size

    q = q[torch.arange(batch_size), action]

    v_batch_mean = v.detach().clone().mean()
    if world_size >= 2:
        all_reduce(v_batch_mean, ReduceOp.AVG)

    _copy: TensorDict = data["next"]  # type: ignore
    assert isinstance(_copy, TensorDict)
    _copy = _copy.copy()
    _copy = _copy.detach()
    with torch.autocast(**autocast_kwargs):
        source_model(_copy)

    vv: Tensor = _copy["state_value"]
    assert isinstance(vv, Tensor)
    assert vv.dim() == 1
    assert vv.size(0) == batch_size

    done: Tensor = _copy["done"]
    assert isinstance(done, Tensor)
    assert done.dim() == 1
    assert done.size(0) == batch_size

    vv = torch.where(done, torch.zeros_like(vv), vv)

    reward: Tensor = _copy["reward"]
    assert isinstance(reward, Tensor)
    assert reward.dim() == 1
    assert reward.size(0) == batch_size

    q_loss = torch.square(reward + discount_factor * vv - q)
    q_loss = torch.mean(q_loss)

    v_loss = q_target - v
    v_loss = torch.where(
        v_loss < 0.0,
        (1.0 - expectile) * torch.square(v_loss),
        expectile * torch.square(v_loss),
    )
    v_loss = torch.mean(v_loss)

    qv_loss = q_loss + v_loss_scaling * v_loss

    qv_batch_loss = qv_loss.detach().clone().mean()
    if world_size >= 2:
        all_reduce(qv_batch_loss, ReduceOp.AVG)

    if math.isnan(qv_loss.item()):
        raise RuntimeError("QV loss becomes NaN.")

    qv_loss /= gradient_accumulation_steps
    grad_scaler.scale(qv_loss).backward()  # type: ignore

    return q_loss, v_loss, v_batch_mean.item(), qv_batch_loss.item()


def _step(
    *,
    source_model: nn.Module,
    grad_scaler: GradScaler,
    max_gradient_norm: float,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
) -> float:
    grad_scaler.unscale_(optimizer)
    qv_gradient = get_gradient(source_model)
    # pylint: disable=not-callable
    qv_gradient_norm = float(torch.linalg.vector_norm(qv_gradient).item())
    nn.utils.clip_grad_norm_(
        source_model.parameters(),
        max_gradient_norm,
        error_if_nonfinite=False,
    )
    grad_scaler.step(optimizer)
    if scheduler is not None:
        scheduler.step()

    return qv_gradient_norm


SnapshotWriter = Callable[[int | None], None]


def _train(
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
    source1_model: nn.Module,
    source2_model: nn.Module,
    target1_model: TensorDictModule,
    target2_model: TensorDictModule,
    reward_plugin: Path,
    double_q_learning: bool,
    discount_factor: float,
    expectile: float,
    target_update_interval: int,
    target_update_rate: float,
    batch_size: int,
    v_loss_scaling: float,
    gradient_accumulation_steps: int,
    max_gradient_norm: float,
    optimizer1: Optimizer,
    lr_scheduler1: LRScheduler | None,
    optimizer2: Optimizer,
    lr_scheduler2: LRScheduler | None,
    snapshot_interval: int,
    num_samples: int,
    summary_writer: SummaryWriter,
    snapshot_writer: SnapshotWriter,
) -> None:
    start_time = datetime.datetime.now()

    world_size, _, local_rank = get_distributed_environment()

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

        # Compute the Q target value.
        with torch.no_grad(), torch.autocast(**autocast_kwargs):
            q1_target, q1_batch_mean = _get_q_target(
                world_size=world_size,
                data=data,
                target_model=target1_model,
            )
            q2_target, q2_batch_mean = _get_q_target(
                world_size=world_size,
                data=data,
                target_model=target2_model,
            )
            q_target = torch.minimum(q1_target, q2_target)
            q_target = q_target.detach().clone()

        # Backprop for the QV1 source model.
        q1_loss, v1_loss, v1_batch_mean, qv1_batch_loss = _backward(
            world_size=world_size,
            autocast_kwargs=autocast_kwargs,
            data=data,
            source_model=source1_model,
            discount_factor=discount_factor,
            expectile=expectile,
            v_loss_scaling=v_loss_scaling,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grad_scaler=grad_scaler,
            q_target=q_target,
        )

        # Backprop for the QV2 source model.
        q2_loss, v2_loss, v2_batch_mean, qv2_batch_loss = _backward(
            world_size=world_size,
            autocast_kwargs=autocast_kwargs,
            data=data,
            source_model=source2_model,
            discount_factor=discount_factor,
            expectile=expectile,
            v_loss_scaling=v_loss_scaling,
            gradient_accumulation_steps=gradient_accumulation_steps,
            grad_scaler=grad_scaler,
            q_target=q_target,
        )

        num_samples += batch_size * world_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            is_grad_nan = is_gradient_nan(source1_model) or is_gradient_nan(
                source2_model
            )
            is_grad_nan = torch.where(
                is_grad_nan,
                torch.ones_like(is_grad_nan),
                torch.zeros_like(is_grad_nan),
            )
            all_reduce(is_grad_nan)
            if is_grad_nan.item() >= 1:
                if local_rank == 0:
                    logging.warning(
                        "Skip an optimization step because of NaN in the gradient."
                    )
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                continue

            qv1_gradient_norm = _step(
                source_model=source1_model,
                grad_scaler=grad_scaler,
                max_gradient_norm=max_gradient_norm,
                optimizer=optimizer1,
                scheduler=lr_scheduler1,
            )
            qv2_gradient_norm = _step(
                source_model=source2_model,
                grad_scaler=grad_scaler,
                max_gradient_norm=max_gradient_norm,
                optimizer=optimizer2,
                scheduler=lr_scheduler2,
            )
            grad_scaler.update()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            if (
                double_q_learning
                and batch_count
                % (gradient_accumulation_steps * target_update_interval)
                == 0
            ):
                with torch.no_grad():
                    for _param, _target_param in zip(
                        source1_model.parameters(),
                        target1_model.parameters(),
                    ):
                        _target_param.data *= 1.0 - target_update_rate
                        _target_param.data += (
                            target_update_rate * _param.data.detach().clone()
                        )
                    for _param, _target_param in zip(
                        source2_model.parameters(),
                        target2_model.parameters(),
                    ):
                        _target_param.data *= 1.0 - target_update_rate
                        _target_param.data += (
                            target_update_rate * _param.data.detach().clone()
                        )

            if local_rank == 0:
                logging.info(
                    "sample = %s, QV1 loss = %s, QV2 loss = %s, "
                    "QV1 gradient norm = %s, QV2 gradient norm = %s",
                    num_samples,
                    qv1_batch_loss,
                    qv2_batch_loss,
                    qv1_gradient_norm,
                    qv2_gradient_norm,
                )
                summary_writer.add_scalars(
                    "Q",
                    {"Q1": q1_batch_mean, "Q2": q2_batch_mean},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "V",
                    {"V1": v1_batch_mean, "V2": v2_batch_mean},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "Q Loss",
                    {"Q1 Loss": q1_loss, "Q2 Loss": q2_loss},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "V Loss",
                    {"V1 Loss": v1_loss, "V2 Loss": v2_loss},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "QV Loss",
                    {"QV1": qv1_batch_loss, "QV2": qv2_batch_loss},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "QV Gradient Norm",
                    {"QV1": qv1_gradient_norm, "QV2": qv2_gradient_norm},
                    num_samples,
                )
                if lr_scheduler1 is not None:
                    summary_writer.add_scalar(
                        "LR",
                        lr_scheduler1.get_last_lr()[0],
                        num_samples,
                    )
        else:
            if local_rank == 0:
                logging.info(
                    "sample = %s, QV1 loss = %s, QV2 loss = %s",
                    num_samples,
                    qv1_batch_loss,
                    qv2_batch_loss,
                )
                summary_writer.add_scalars(
                    "Q",
                    {"Q1": q1_batch_mean, "Q2": q2_batch_mean},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "V",
                    {"V1": v1_batch_mean, "V2": v2_batch_mean},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "Q Loss",
                    {"Q1 Loss": q1_loss, "Q2 Loss": q2_loss},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "V Loss",
                    {"V1 Loss": v1_loss, "V2 Loss": v2_loss},
                    num_samples,
                )
                summary_writer.add_scalars(
                    "QV Loss",
                    {"QV1": qv1_batch_loss, "QV2": qv2_batch_loss},
                    num_samples,
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
            "A training has finished (elapsed time = %s).",
            elapsed_time,
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
    source1_encoder_snapshot_path: Path | None = None
    source1_q_decoder_snapshot_path: Path | None = None
    source1_v_decoder_snapshot_path: Path | None = None
    source2_encoder_snapshot_path: Path | None = None
    source2_q_decoder_snapshot_path: Path | None = None
    source2_v_decoder_snapshot_path: Path | None = None
    target1_encoder_snapshot_path: Path | None = None
    target1_q_decoder_snapshot_path: Path | None = None
    target1_v_decoder_snapshot_path: Path | None = None
    target2_encoder_snapshot_path: Path | None = None
    target2_q_decoder_snapshot_path: Path | None = None
    target2_v_decoder_snapshot_path: Path | None = None
    optimizer1_snapshot_path: Path | None = None
    optimizer2_snapshot_path: Path | None = None
    lr_scheduler1_snapshot_path: Path | None = None
    lr_scheduler2_snapshot_path: Path | None = None

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    "^(?:(?:(?:source|target)[12]-(?:encoder|[qv]-decoder))|optimizer[12]|lr-scheduler[12])(?:\\.(\\d+))?\\.pth$",
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

        source1_encoder_snapshot_path = (
            config.initial_model_prefix / f"source1-encoder{infix}.pth"
        )
        assert source1_encoder_snapshot_path is not None
        if not source1_encoder_snapshot_path.exists():
            errmsg = f"{source1_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not source1_encoder_snapshot_path.is_file():
            errmsg = f"{source1_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        source1_q_decoder_snapshot_path = (
            config.initial_model_prefix / f"source1-q-decoder{infix}.pth"
        )
        assert source1_q_decoder_snapshot_path is not None
        if not source1_q_decoder_snapshot_path.exists():
            errmsg = f"{source1_q_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not source1_q_decoder_snapshot_path.is_file():
            errmsg = f"{source1_q_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        source1_v_decoder_snapshot_path = (
            config.initial_model_prefix / f"source1-v-decoder{infix}.pth"
        )
        assert source1_v_decoder_snapshot_path is not None
        if not source1_v_decoder_snapshot_path.exists():
            errmsg = f"{source1_v_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not source1_v_decoder_snapshot_path.is_file():
            errmsg = f"{source1_v_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        source2_encoder_snapshot_path = (
            config.initial_model_prefix / f"source2-encoder{infix}.pth"
        )
        assert source2_encoder_snapshot_path is not None
        if not source2_encoder_snapshot_path.exists():
            errmsg = f"{source2_encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not source2_encoder_snapshot_path.is_file():
            errmsg = f"{source2_encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        source2_q_decoder_snapshot_path = (
            config.initial_model_prefix / f"source2-q-decoder{infix}.pth"
        )
        assert source2_q_decoder_snapshot_path is not None
        if not source2_q_decoder_snapshot_path.exists():
            errmsg = f"{source2_q_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not source2_q_decoder_snapshot_path.is_file():
            errmsg = f"{source2_q_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        source2_v_decoder_snapshot_path = (
            config.initial_model_prefix / f"source2-v-decoder{infix}.pth"
        )
        assert source2_v_decoder_snapshot_path is not None
        if not source2_v_decoder_snapshot_path.exists():
            errmsg = f"{source2_v_decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not source2_v_decoder_snapshot_path.is_file():
            errmsg = f"{source2_v_decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        if config.double_q_learning:
            target1_encoder_snapshot_path = (
                config.initial_model_prefix / f"target1-encoder{infix}.pth"
            )
            assert target1_encoder_snapshot_path is not None
            if not target1_encoder_snapshot_path.exists():
                errmsg = f"{target1_encoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target1_encoder_snapshot_path.is_file():
                errmsg = f"{target1_encoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target1_q_decoder_snapshot_path = (
                config.initial_model_prefix / f"target1-q-decoder{infix}.pth"
            )
            assert target1_q_decoder_snapshot_path is not None
            if not target1_q_decoder_snapshot_path.exists():
                errmsg = f"{target1_q_decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target1_q_decoder_snapshot_path.is_file():
                errmsg = f"{target1_q_decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target1_v_decoder_snapshot_path = (
                config.initial_model_prefix / f"target1-v-decoder{infix}.pth"
            )
            assert target1_v_decoder_snapshot_path is not None
            if not target1_v_decoder_snapshot_path.exists():
                errmsg = f"{target1_v_decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target1_v_decoder_snapshot_path.is_file():
                errmsg = f"{target1_v_decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target2_encoder_snapshot_path = (
                config.initial_model_prefix / f"target2-encoder{infix}.pth"
            )
            assert target2_encoder_snapshot_path is not None
            if not target2_encoder_snapshot_path.exists():
                errmsg = f"{target2_encoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target2_encoder_snapshot_path.is_file():
                errmsg = f"{target2_encoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target2_q_decoder_snapshot_path = (
                config.initial_model_prefix / f"target2-q-decoder{infix}.pth"
            )
            assert target2_q_decoder_snapshot_path is not None
            if not target2_q_decoder_snapshot_path.exists():
                errmsg = f"{target2_q_decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target2_q_decoder_snapshot_path.is_file():
                errmsg = f"{target2_q_decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target2_v_decoder_snapshot_path = (
                config.initial_model_prefix / f"target2-v-decoder{infix}.pth"
            )
            assert target2_v_decoder_snapshot_path is not None
            if not target2_v_decoder_snapshot_path.exists():
                errmsg = f"{target2_v_decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target2_v_decoder_snapshot_path.is_file():
                errmsg = f"{target2_v_decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

        optimizer1_snapshot_path = (
            config.initial_model_prefix / f"optimizer1{infix}.pth"
        )
        assert optimizer1_snapshot_path is not None
        if (
            not optimizer1_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            optimizer1_snapshot_path = None

        optimizer2_snapshot_path = (
            config.initial_model_prefix / f"optimizer2{infix}.pth"
        )
        assert optimizer2_snapshot_path is not None
        if (
            not optimizer2_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            optimizer2_snapshot_path = None

        lr_scheduler1_snapshot_path = (
            config.initial_model_prefix / f"lr-scheduler1{infix}.pth"
        )
        assert lr_scheduler1_snapshot_path is not None
        if optimizer1_snapshot_path is None:
            lr_scheduler1_snapshot_path = None
        else:
            if not lr_scheduler1_snapshot_path.exists():
                errmsg = f"{lr_scheduler1_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not lr_scheduler1_snapshot_path.is_file():
                errmsg = f"{lr_scheduler1_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

        lr_scheduler2_snapshot_path = (
            config.initial_model_prefix / f"lr-scheduler2{infix}.pth"
        )
        assert lr_scheduler2_snapshot_path is not None
        if optimizer2_snapshot_path is None:
            lr_scheduler2_snapshot_path = None
        else:
            if not lr_scheduler2_snapshot_path.exists():
                errmsg = f"{lr_scheduler2_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not lr_scheduler2_snapshot_path.is_file():
                errmsg = f"{lr_scheduler2_snapshot_path}: Not a file."
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

    if config.v_loss_scaling < 0.0 or 1.0 < config.v_loss_scaling:
        errmsg = (
            f"{config.v_loss_scaling}: An invalid value for `v_loss_scaling`."
        )
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

    if config.max_gradient_norm <= 0.0:
        errmsg = (
            f"{config.max_gradient_norm}: `max_gradient_norm` must be a"
            " positive real value."
        )
        raise RuntimeError(errmsg)

    _config.optimizer.validate(config)

    if config.double_q_learning:
        if config.target_update_interval <= 0:
            errmsg = (
                f"{config.target_update_interval}: "
                "`target_update_interval` must be a positive integer."
            )
            raise RuntimeError(errmsg)
    else:
        if config.target_update_interval != 0:
            errmsg = (
                f"{config.target_update_interval}: "
                "`target_update_interval` must be 0 for the case where"
                " double Q-learning is disabled."
            )
            raise RuntimeError(errmsg)

    if config.double_q_learning:
        if config.target_update_rate <= 0.0 or config.target_update_rate > 1.0:
            errmsg = (
                f"{config.target_update_rate}: `target_update_rate` must be"
                " a real value within the range (0.0, 1.0]."
            )
            raise RuntimeError(errmsg)
    else:
        if config.target_update_rate != 0.0:
            errmsg = (
                f"{config.target_update_rate}: `target_update_rate` must be 0.0"
                " for the case where double Q-learning is disabled."
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
        logging.info("V loss scaling: %E", config.v_loss_scaling)
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
            "Norm threshold for gradient clipping: %E",
            config.max_gradient_norm,
        )

        _config.optimizer.dump(config)

        if config.double_q_learning:
            logging.info(
                "Target update interval: %d",
                config.target_update_interval,
            )
            logging.info("Target update rate: %f", config.target_update_rate)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial source 1 encoder snapshot: %s",
                source1_encoder_snapshot_path,
            )
            logging.info(
                "Initial source 1 Q decoder snapshot: %s",
                source1_q_decoder_snapshot_path,
            )
            logging.info(
                "Initial source 1 V decoder snapshot: %s",
                source1_v_decoder_snapshot_path,
            )
            logging.info(
                "Initial source 2 encoder snapshot: %s",
                source2_encoder_snapshot_path,
            )
            logging.info(
                "Initial source 2 Q decoder snapshot: %s",
                source2_q_decoder_snapshot_path,
            )
            logging.info(
                "Initial source 2 V decoder snapshot: %s",
                source2_v_decoder_snapshot_path,
            )
            if config.double_q_learning:
                logging.info(
                    "Initial target 1 encoder snapshot: %s",
                    source1_encoder_snapshot_path,
                )
                logging.info(
                    "Initial target 1 Q decoder snapshot: %s",
                    source1_q_decoder_snapshot_path,
                )
                logging.info(
                    "Initial target 1 V decoder snapshot: %s",
                    source1_v_decoder_snapshot_path,
                )
                logging.info(
                    "Initial target 2 encoder snapshot: %s",
                    source2_encoder_snapshot_path,
                )
                logging.info(
                    "Initial target 2 Q decoder snapshot: %s",
                    source2_q_decoder_snapshot_path,
                )
                logging.info(
                    "Initial target 2 V decoder snapshot: %s",
                    source2_v_decoder_snapshot_path,
                )
            if optimizer1_snapshot_path is not None:
                logging.info(
                    "Initial optimizer 1 snapshot: %s",
                    optimizer1_snapshot_path,
                )
                logging.info(
                    "Initial LR scheduler 1 snapshot: %s",
                    lr_scheduler1_snapshot_path,
                )
            if optimizer2_snapshot_path is not None:
                logging.info(
                    "Initial optimizer 2 snapshot: %s",
                    optimizer2_snapshot_path,
                )
                logging.info(
                    "Initial LR scheduler 2 snapshot: %s",
                    lr_scheduler2_snapshot_path,
                )

        logging.info("Output prefix: %s", output_prefix)
        if config.snapshot_interval == 0:
            logging.info("Snapshot interval: N/A")
        else:
            logging.info("Snapshot interval: %d", config.snapshot_interval)

    if world_size >= 2:
        init_process_group(backend="nccl")

    source1_encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    source1_encoder_tdm = TensorDictModule(
        source1_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    source1_q_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="candidates",
        noise_init_std=None,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    with torch.no_grad():
        for _param in source1_q_decoder.parameters():
            _param.zero_()
    source1_q_decoder_tdm = TensorDictModule(
        source1_q_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    source1_v_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="state",
        noise_init_std=None,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    with torch.no_grad():
        for _param in source1_v_decoder.parameters():
            _param.zero_()
    source1_v_decoder_tdm = TensorDictModule(
        source1_v_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["state_value"],  # type: ignore
    )
    source1_model = TensorDictSequential(
        source1_encoder_tdm,
        source1_q_decoder_tdm,
        source1_v_decoder_tdm,
    )
    if world_size >= 2:
        source1_model.to(device=device)
        for _param in source1_model.parameters():
            broadcast(_param.data, src=0)
        source1_model.to(device="cpu")

    source2_encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    source2_encoder_tdm = TensorDictModule(
        source2_encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    source2_q_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="candidates",
        noise_init_std=None,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    with torch.no_grad():
        for _param in source2_q_decoder.parameters():
            _param.zero_()
    source2_q_decoder_tdm = TensorDictModule(
        source2_q_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["action_value"],  # type: ignore
    )
    source2_v_decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        layer_normalization=config.decoder.layer_normalization,
        num_layers=config.decoder.num_layers,
        output_mode="state",
        noise_init_std=None,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    with torch.no_grad():
        for _param in source2_v_decoder.parameters():
            _param.zero_()
    source2_v_decoder_tdm = TensorDictModule(
        source2_v_decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["state_value"],  # type: ignore
    )
    source2_model = TensorDictSequential(
        source2_encoder_tdm,
        source2_q_decoder_tdm,
        source2_v_decoder_tdm,
    )
    if world_size >= 2:
        source2_model.to(device=device)
        for _param in source2_model.parameters():
            broadcast(_param.data, src=0)
        source2_model.to(device="cpu")

    if config.double_q_learning:
        target1_encoder = Encoder(
            position_encoder=config.encoder.position_encoder,
            dimension=config.encoder.dimension,
            num_heads=config.encoder.num_heads,
            dim_feedforward=config.encoder.dim_feedforward,
            activation_function=config.encoder.activation_function,
            dropout=config.encoder.dropout,
            layer_normalization=config.encoder.layer_normalization,
            num_layers=config.encoder.num_layers,
            checkpointing=config.checkpointing,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        target1_encoder_tdm = TensorDictModule(
            target1_encoder,
            in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
            out_keys=["encode"],  # type: ignore
        )
        target1_q_decoder = Decoder(
            input_dimension=config.encoder.dimension,
            dimension=config.decoder.dimension,
            activation_function=config.decoder.activation_function,
            dropout=config.decoder.dropout,
            layer_normalization=config.decoder.layer_normalization,
            num_layers=config.decoder.num_layers,
            output_mode="candidates",
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        target1_q_decoder_tdm = TensorDictModule(
            target1_q_decoder,
            in_keys=["encode"],  # type: ignore
            out_keys=["action_value"],  # type: ignore
        )
        target1_v_decoder = Decoder(
            input_dimension=config.encoder.dimension,
            dimension=config.decoder.dimension,
            activation_function=config.decoder.activation_function,
            dropout=config.decoder.dropout,
            layer_normalization=config.decoder.layer_normalization,
            num_layers=config.decoder.num_layers,
            output_mode="state",
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        target1_v_decoder_tdm = TensorDictModule(
            target1_v_decoder,
            in_keys=["encode"],  # type: ignore
            out_keys=["state_value"],  # type: ignore
        )
        target1_model = TensorDictSequential(
            target1_encoder_tdm,
            target1_q_decoder_tdm,
            target1_v_decoder_tdm,
        )
        with torch.no_grad():
            for _param, _target_param in zip(
                source1_model.parameters(), target1_model.parameters()
            ):
                _target_param.data = _param.data.detach().clone()

        target2_encoder = Encoder(
            position_encoder=config.encoder.position_encoder,
            dimension=config.encoder.dimension,
            num_heads=config.encoder.num_heads,
            dim_feedforward=config.encoder.dim_feedforward,
            activation_function=config.encoder.activation_function,
            dropout=config.encoder.dropout,
            layer_normalization=config.encoder.layer_normalization,
            num_layers=config.encoder.num_layers,
            checkpointing=config.checkpointing,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        target2_encoder_tdm = TensorDictModule(
            target2_encoder,
            in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
            out_keys=["encode"],  # type: ignore
        )
        target2_q_decoder = Decoder(
            input_dimension=config.encoder.dimension,
            dimension=config.decoder.dimension,
            activation_function=config.decoder.activation_function,
            dropout=config.decoder.dropout,
            layer_normalization=config.decoder.layer_normalization,
            num_layers=config.decoder.num_layers,
            output_mode="candidates",
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        target2_q_decoder_tdm = TensorDictModule(
            target2_q_decoder,
            in_keys=["encode"],  # type: ignore
            out_keys=["action_value"],  # type: ignore
        )
        target2_v_decoder = Decoder(
            input_dimension=config.encoder.dimension,
            dimension=config.decoder.dimension,
            activation_function=config.decoder.activation_function,
            dropout=config.decoder.dropout,
            layer_normalization=config.decoder.layer_normalization,
            num_layers=config.decoder.num_layers,
            output_mode="state",
            noise_init_std=None,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        target2_v_decoder_tdm = TensorDictModule(
            target2_v_decoder,
            in_keys=["encode"],  # type: ignore
            out_keys=["state_value"],  # type: ignore
        )
        target2_model = TensorDictSequential(
            target2_encoder_tdm,
            target2_q_decoder_tdm,
            target2_v_decoder_tdm,
        )
        with torch.no_grad():
            for _param, _target_param in zip(
                source2_model.parameters(), target2_model.parameters()
            ):
                _target_param.data = _param.data.detach().clone()
    else:
        target1_encoder = source1_encoder
        target1_encoder_tdm = source1_encoder_tdm
        target1_q_decoder = source1_q_decoder
        target1_q_decoder_tdm = source1_q_decoder_tdm
        target1_v_decoder = source1_v_decoder
        target1_v_decoder_tdm = source1_v_decoder_tdm
        target1_model = source1_model
        target2_encoder = source2_encoder
        target2_encoder_tdm = source2_encoder_tdm
        target2_q_decoder = source2_q_decoder
        target2_q_decoder_tdm = source2_q_decoder_tdm
        target2_v_decoder = source2_v_decoder
        target2_v_decoder_tdm = source2_v_decoder_tdm
        target2_model = source2_model

    model_to_save = TwinQActor(target1_model, target2_model)
    model_to_save_tdm = TensorDictModule(
        model_to_save,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["action"],  # type: ignore
    )

    optimizer1, lr_scheduler1 = _config.optimizer.create(
        device.type, config, source1_model
    )
    optimizer2, lr_scheduler2 = _config.optimizer.create(
        device.type, config, source2_model
    )

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(
            config.encoder.load_from,
            map_location="cpu",
            weights_only=True,
        )
        source1_encoder.load_state_dict(encoder_state_dict)
        source2_encoder.load_state_dict(encoder_state_dict)
        if config.double_q_learning:
            assert target1_encoder is not None
            assert target2_encoder is not None
            target1_encoder.load_state_dict(encoder_state_dict)
            target2_encoder.load_state_dict(encoder_state_dict)

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert source1_encoder_snapshot_path is not None
        assert source1_q_decoder_snapshot_path is not None
        assert source1_v_decoder_snapshot_path is not None
        assert source2_encoder_snapshot_path is not None
        assert source2_q_decoder_snapshot_path is not None
        assert source2_v_decoder_snapshot_path is not None
        if config.double_q_learning:
            assert target1_encoder_snapshot_path is not None
            assert target1_q_decoder_snapshot_path is not None
            assert target1_v_decoder_snapshot_path is not None
            assert target2_encoder_snapshot_path is not None
            assert target2_q_decoder_snapshot_path is not None
            assert target2_v_decoder_snapshot_path is not None

        source1_encoder_state_dict = torch.load(
            source1_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        source1_encoder.load_state_dict(source1_encoder_state_dict)
        source1_q_decoder_state_dict = torch.load(
            source1_q_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        source1_q_decoder.load_state_dict(source1_q_decoder_state_dict)
        source1_v_decoder_state_dict = torch.load(
            source1_v_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        source1_v_decoder.load_state_dict(source1_v_decoder_state_dict)

        source2_encoder_state_dict = torch.load(
            source2_encoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        source2_encoder.load_state_dict(source2_encoder_state_dict)
        source2_q_decoder_state_dict = torch.load(
            source2_q_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        source2_q_decoder.load_state_dict(source2_q_decoder_state_dict)
        source2_v_decoder_state_dict = torch.load(
            source2_v_decoder_snapshot_path,
            map_location="cpu",
            weights_only=True,
        )
        source2_v_decoder.load_state_dict(source2_v_decoder_state_dict)

        if config.double_q_learning:
            assert target1_encoder_snapshot_path is not None
            assert target1_encoder is not None
            assert target1_q_decoder_snapshot_path is not None
            assert target1_q_decoder is not None
            assert target1_v_decoder_snapshot_path is not None
            assert target1_v_decoder is not None
            assert target2_encoder_snapshot_path is not None
            assert target2_encoder is not None
            assert target2_q_decoder_snapshot_path is not None
            assert target2_q_decoder is not None
            assert target2_v_decoder_snapshot_path is not None
            assert target2_v_decoder is not None
            target1_encoder_state_dict = torch.load(
                target1_encoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            target1_encoder.load_state_dict(target1_encoder_state_dict)
            target1_q_decoder_state_dict = torch.load(
                target1_q_decoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            target1_q_decoder.load_state_dict(target1_q_decoder_state_dict)
            target1_v_decoder_state_dict = torch.load(
                target1_v_decoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            target1_v_decoder.load_state_dict(target1_v_decoder_state_dict)

            target2_encoder_state_dict = torch.load(
                target2_encoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            target2_encoder.load_state_dict(target2_encoder_state_dict)
            target2_q_decoder_state_dict = torch.load(
                target2_q_decoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            target2_q_decoder.load_state_dict(target2_q_decoder_state_dict)
            target2_v_decoder_state_dict = torch.load(
                target2_v_decoder_snapshot_path,
                map_location="cpu",
                weights_only=True,
            )
            target2_v_decoder.load_state_dict(target2_v_decoder_state_dict)

        if optimizer1_snapshot_path is not None:
            assert lr_scheduler1_snapshot_path is not None
            assert lr_scheduler1 is not None
            optimizer1.load_state_dict(
                torch.load(
                    optimizer1_snapshot_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )
            lr_scheduler1.load_state_dict(
                torch.load(
                    lr_scheduler1_snapshot_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )

        if optimizer2_snapshot_path is not None:
            assert lr_scheduler2_snapshot_path is not None
            assert lr_scheduler2 is not None
            optimizer2.load_state_dict(
                torch.load(
                    optimizer2_snapshot_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )
            lr_scheduler2.load_state_dict(
                torch.load(
                    lr_scheduler2_snapshot_path,
                    map_location="cpu",
                    weights_only=True,
                )
            )

    source1_model.requires_grad_(True)
    source1_model.train()
    source1_model.to(device=device, dtype=dtype)
    if world_size >= 2:
        source1_model = DistributedDataParallel(source1_model)
        source1_model = nn.SyncBatchNorm.convert_sync_batchnorm(source1_model)

    source2_model.requires_grad_(True)
    source2_model.train()
    source2_model.to(device=device, dtype=dtype)
    if world_size >= 2:
        source2_model = DistributedDataParallel(source2_model)
        source2_model = nn.SyncBatchNorm.convert_sync_batchnorm(source2_model)

    if config.double_q_learning:
        target1_model.requires_grad_(False)
        target1_model.eval()
        target1_model.to(device=device, dtype=dtype)

        target2_model.requires_grad_(False)
        target2_model.eval()
        target2_model.to(device=device, dtype=dtype)

    snapshots_path = output_prefix / "snapshots"

    def snapshot_writer(num_samples: int | None) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)

        infix = "" if num_samples is None else f".{num_samples}"

        torch.save(
            source1_encoder.state_dict(),
            snapshots_path / f"source1-encoder{infix}.pth",
        )
        torch.save(
            source1_q_decoder.state_dict(),
            snapshots_path / f"source1-q-decoder{infix}.pth",
        )
        torch.save(
            source1_v_decoder.state_dict(),
            snapshots_path / f"source1-v-decoder{infix}.pth",
        )
        torch.save(
            source2_encoder.state_dict(),
            snapshots_path / f"source2-encoder{infix}.pth",
        )
        torch.save(
            source2_q_decoder.state_dict(),
            snapshots_path / f"source2-q-decoder{infix}.pth",
        )
        torch.save(
            source2_v_decoder.state_dict(),
            snapshots_path / f"source2-v-decoder{infix}.pth",
        )
        if config.double_q_learning:
            assert target1_encoder is not None
            assert target1_q_decoder is not None
            assert target1_v_decoder is not None
            assert target2_encoder is not None
            assert target2_q_decoder is not None
            assert target2_v_decoder is not None
            torch.save(
                target1_encoder.state_dict(),
                snapshots_path / f"target1-encoder{infix}.pth",
            )
            torch.save(
                target1_q_decoder.state_dict(),
                snapshots_path / f"target1-q-decoder{infix}.pth",
            )
            torch.save(
                target1_v_decoder.state_dict(),
                snapshots_path / f"target1-v-decoder{infix}.pth",
            )
            torch.save(
                target2_encoder.state_dict(),
                snapshots_path / f"target2-encoder{infix}.pth",
            )
            torch.save(
                target2_q_decoder.state_dict(),
                snapshots_path / f"target2-q-decoder{infix}.pth",
            )
            torch.save(
                target2_v_decoder.state_dict(),
                snapshots_path / f"target2-v-decoder{infix}.pth",
            )
        torch.save(
            optimizer1.state_dict(), snapshots_path / f"optimizer1{infix}.pth"
        )
        if lr_scheduler1 is not None:
            torch.save(
                lr_scheduler1.state_dict(),
                snapshots_path / f"lr-scheduler1{infix}.pth",
            )
        torch.save(
            optimizer2.state_dict(), snapshots_path / f"optimizer2{infix}.pth"
        )
        if lr_scheduler2 is not None:
            torch.save(
                lr_scheduler2.state_dict(),
                snapshots_path / f"lr-scheduler2{infix}.pth",
            )

        state = dump_object(
            model_to_save_tdm,
            [
                dump_object(
                    model_to_save,
                    [
                        dump_object(
                            target1_model,
                            [
                                dump_object(
                                    target1_encoder_tdm,
                                    [
                                        dump_model(
                                            target1_encoder,
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
                                        )
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
                                    target1_q_decoder_tdm,
                                    [
                                        dump_model(
                                            target1_q_decoder,
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
                                dump_object(
                                    target1_v_decoder_tdm,
                                    [
                                        dump_model(
                                            target1_v_decoder,
                                            [],
                                            {
                                                "input_dimension": config.encoder.dimension,
                                                "dimension": config.decoder.dimension,
                                                "activation_function": config.decoder.activation_function,
                                                "dropout": config.decoder.dropout,
                                                "layer_normalization": config.decoder.layer_normalization,
                                                "num_layers": config.decoder.num_layers,
                                                "output_mode": "state",
                                                "noise_init_std": None,
                                                "device": torch.device("cpu"),
                                                "dtype": dtype,
                                            },
                                        ),
                                    ],
                                    {
                                        "in_keys": ["encode"],
                                        "out_keys": ["state_value"],
                                    },
                                ),
                            ],
                            {},
                        ),
                        dump_object(
                            target2_model,
                            [
                                dump_object(
                                    target2_encoder_tdm,
                                    [
                                        dump_model(
                                            target2_encoder,
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
                                        )
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
                                    target2_q_decoder_tdm,
                                    [
                                        dump_model(
                                            target2_q_decoder,
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
                                        )
                                    ],
                                    {
                                        "in_keys": ["encode"],
                                        "out_keys": ["action_value"],
                                    },
                                ),
                                dump_object(
                                    target2_v_decoder_tdm,
                                    [
                                        dump_model(
                                            target2_v_decoder,
                                            [],
                                            {
                                                "input_dimension": config.encoder.dimension,
                                                "dimension": config.decoder.dimension,
                                                "activation_function": config.decoder.activation_function,
                                                "dropout": config.decoder.dropout,
                                                "layer_normalization": config.decoder.layer_normalization,
                                                "num_layers": config.decoder.num_layers,
                                                "output_mode": "state",
                                                "noise_init_std": None,
                                                "device": torch.device("cpu"),
                                                "dtype": dtype,
                                            },
                                        )
                                    ],
                                    {
                                        "in_keys": ["encode"],
                                        "out_keys": ["state_value"],
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
        torch.autograd.set_detect_anomaly(
            False  # `True` for debbing purpose only.
        )
        assert isinstance(source1_model, nn.Module)
        assert isinstance(source2_model, nn.Module)
        _train(
            training_data=config.training_data,
            contiguous_training_data=config.contiguous_training_data,
            rewrite_rooms=config.rewrite_rooms,
            rewrite_grades=config.rewrite_grades,
            replay_buffer_size=config.replay_buffer_size,
            num_workers=config.num_workers,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            source1_model=source1_model,
            source2_model=source2_model,
            target1_model=target1_model,
            target2_model=target2_model,
            reward_plugin=config.reward_plugin,
            double_q_learning=config.double_q_learning,
            discount_factor=config.discount_factor,
            expectile=config.expectile,
            target_update_interval=config.target_update_interval,
            target_update_rate=config.target_update_rate,
            batch_size=config.batch_size,
            v_loss_scaling=config.v_loss_scaling,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm,
            optimizer1=optimizer1,
            lr_scheduler1=lr_scheduler1,
            optimizer2=optimizer2,
            lr_scheduler2=lr_scheduler2,
            snapshot_interval=config.snapshot_interval,
            num_samples=num_samples,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer,
        )


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
    sys.exit(0)
