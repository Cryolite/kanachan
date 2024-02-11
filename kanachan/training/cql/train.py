#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
import logging
import sys
from typing import Callable
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_scheduler
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group, ReduceOp, all_reduce
from torch.utils.tensorboard.writer import SummaryWriter
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from kanachan.constants import (
    MAX_NUM_ACTIVE_SPARSE_FEATURES,
    MAX_NUM_ACTION_CANDIDATES,
)
from kanachan.training.common import (
    get_distributed_environment,
    get_gradient,
    is_gradient_nan,
)
import kanachan.training.core.config as _config

# pylint: disable=unused-import
import kanachan.training.cql.config  # noqa: F401
from kanachan.training.core.offline_rl import DataLoader, EpisodeReplayBuffer
from kanachan.nn import Encoder, QRDecoder, QDecoder, DecodeConverter
from kanachan.nn.qr_decoder import compute_td_error
from kanachan.model_loader import dump_object, dump_model


SnapshotWriter = Callable[[int | None], None]


def _training(
    *,
    training_data: Path,
    contiguous_training_data: bool,
    rewrite_grades: int | None,
    num_workers: int,
    replay_buffer_size: int,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    source_network: nn.Module,
    target_network: nn.Module | None,
    reward_plugin: Path,
    discount_factor: float,
    kappa: float,
    td_computation_batch_size: int,
    alpha: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_gradient_norm: float,
    optimizer: Optimizer,
    scheduler: lr_scheduler.LRScheduler | None,
    target_update_interval: int | None,
    target_update_rate: int | None,
    snapshot_interval: int,
    num_samples: int,
    summary_writer: SummaryWriter,
    snapshot_writer: SnapshotWriter,
) -> None:
    start_time = datetime.datetime.now()

    world_size, _, local_rank = get_distributed_environment()

    is_amp_enabled = device.type != "cpu" and dtype != amp_dtype
    autocast_kwargs = {
        "device_type": device.type,
        "dtype": amp_dtype,
        "enabled": is_amp_enabled,
    }

    # Load the reward plugin.
    with open(reward_plugin, encoding="UTF-8") as file_pointer:
        exec(file_pointer.read(), globals())  # pylint: disable=exec-used

    if replay_buffer_size >= 1:
        data_loader = EpisodeReplayBuffer(
            training_data=training_data,
            contiguous_training_data=contiguous_training_data,
            num_skip_samples=num_samples,
            # pylint: disable=undefined-variable
            get_reward=get_reward,  # type: ignore # noqa: F821
            max_size=replay_buffer_size,
            batch_size=batch_size,
            drop_last=(world_size >= 2),
        )
    else:
        data_loader = DataLoader(
            path=training_data,
            num_skip_samples=num_samples,
            # pylint: disable=undefined-variable
            get_reward=get_reward,  # type: ignore # noqa: F821
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=(num_workers >= 1),
            drop_last=(world_size >= 2),
        )

    last_snapshot = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    grad_scaler = None
    if is_amp_enabled:
        grad_scaler = GradScaler()

    batch_count = 0

    for data in data_loader:
        if rewrite_grades is not None:
            assert 0 <= rewrite_grades and rewrite_grades <= 15
            sparse: Tensor = data["sparse"]
            assert sparse.dim() == 2
            assert sparse.size(0) == batch_size
            assert sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
            assert sparse.dtype == torch.int32
            sparse[:, 2].fill_(7 + rewrite_grades)
            sparse[:, 3].fill_(23 + rewrite_grades)
            sparse[:, 4].fill_(39 + rewrite_grades)
            sparse[:, 5].fill_(55 + rewrite_grades)
            next_sparse: Tensor = data["next", "sparse"]
            assert next_sparse.dim() == 2
            assert next_sparse.size(0) == batch_size
            assert next_sparse.size(1) == MAX_NUM_ACTIVE_SPARSE_FEATURES
            assert next_sparse.dtype == torch.int32
            next_sparse[:, 2].fill_(7 + rewrite_grades)
            next_sparse[:, 3].fill_(23 + rewrite_grades)
            next_sparse[:, 4].fill_(39 + rewrite_grades)
            next_sparse[:, 5].fill_(55 + rewrite_grades)

        data = data.to(device=device)
        assert isinstance(data, TensorDict)

        with torch.autocast(**autocast_kwargs):
            _chunks: list[TensorDict] = []
            for _chunk in data.split(td_computation_batch_size):
                assert isinstance(_chunk, TensorDict)
                compute_td_error(
                    source_network=source_network,
                    target_network=target_network,
                    data=_chunk,
                    discount_factor=discount_factor,
                    kappa=kappa,
                )
                _chunks.append(_chunk)
            data: TensorDict = torch.cat(_chunks)  # type: ignore
        td_error: Tensor = data["td_error"]
        assert td_error.dim() == 1
        assert td_error.size(0) == batch_size
        td_error = td_error.mean()

        q: Tensor = data["action_value"]
        assert q.dim() == 2
        assert q.size(0) == batch_size
        assert q.size(1) == MAX_NUM_ACTION_CANDIDATES

        action: Tensor = data["action"]
        assert action.dim() == 1
        assert action.size(0) == batch_size

        q_sa = q[torch.arange(batch_size), action]
        assert q_sa.dim() == 1
        assert q_sa.size(0) == batch_size

        _q = q_sa.detach().clone().mean()
        if world_size >= 2:
            all_reduce(_q, ReduceOp.AVG)
        q_to_display = _q.item()

        # Compute the regularization term.
        log_z = torch.logsumexp(q, dim=1)
        regularizer = torch.mean(log_z - q_sa)

        loss_regularizer = alpha * regularizer
        _loss_regularizer = loss_regularizer.detach().clone()
        if world_size >= 2:
            all_reduce(_loss_regularizer, ReduceOp.AVG)
        loss_regularizer_to_display: float = _loss_regularizer.item()

        loss_td_error = 0.5 * td_error
        _loss_td_error = loss_td_error.detach().clone()
        if world_size >= 2:
            all_reduce(_loss_td_error, ReduceOp.AVG)
        loss_td_error_to_display: float = _loss_td_error.item()

        loss = loss_regularizer + loss_td_error
        _loss = loss.detach().clone()
        if world_size >= 2:
            all_reduce(_loss, ReduceOp.AVG)
        loss_to_display: float = _loss.item()

        if math.isnan(loss_to_display):
            errmsg = "Loss becomes NaN."
            raise RuntimeError(errmsg)

        loss /= gradient_accumulation_steps
        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()  # type: ignore

        num_samples += batch_size * world_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            is_grad_nan = is_gradient_nan(source_network)
            if world_size >= 2:
                all_reduce(is_grad_nan)
            if is_grad_nan.item() >= 1:
                if local_rank == 0:
                    logging.warning(
                        "Skip an optimization step because of NaN in the gradient."
                    )
                optimizer.zero_grad()
                continue

            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            gradient = get_gradient(source_network)
            # pylint: disable=not-callable
            gradient_norm: float = torch.linalg.vector_norm(gradient).item()
            nn.utils.clip_grad_norm_(
                source_network.parameters(),
                max_gradient_norm,
                error_if_nonfinite=False,
            )
            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            if (
                target_update_interval is not None
                and batch_count
                % (gradient_accumulation_steps * target_update_interval)
                == 0
            ):
                assert target_update_rate is not None
                assert target_network is not None
                with torch.no_grad():
                    for source_param, target_param in zip(
                        source_network.parameters(),
                        target_network.parameters(),
                    ):
                        target_param.data *= 1.0 - target_update_rate
                        target_param.data += (
                            target_update_rate * source_param.data
                        )

            if local_rank == 0:
                logging.info(
                    "sample = %s, loss = %s, Q = %E, gradient norm = %s",
                    num_samples,
                    loss_to_display,
                    q_to_display,
                    gradient_norm,
                )
                summary_writer.add_scalar("Q", q_to_display, num_samples)
                summary_writer.add_scalar(
                    "Loss (regularizer)",
                    loss_regularizer_to_display,
                    num_samples,
                )
                summary_writer.add_scalar(
                    "Loss (TD error)", loss_td_error_to_display, num_samples
                )
                summary_writer.add_scalar("Loss", loss_to_display, num_samples)
                summary_writer.add_scalar(
                    "Gradient Norm", gradient_norm, num_samples
                )
                if scheduler is not None:
                    summary_writer.add_scalar(
                        "LR", scheduler.get_last_lr()[0], num_samples
                    )
        else:
            if local_rank == 0:
                logging.info(
                    "sample = %s, loss = %s, Q = %E",
                    num_samples,
                    loss_to_display,
                    q_to_display,
                )
                summary_writer.add_scalar("Q", q_to_display, num_samples)
                summary_writer.add_scalar(
                    "Loss (regularizer)",
                    loss_regularizer_to_display,
                    num_samples,
                )
                summary_writer.add_scalar(
                    "Loss (TD error)", loss_td_error_to_display, num_samples
                )
                summary_writer.add_scalar("Loss", loss_to_display, num_samples)

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

    if config.rewrite_grades is not None and (
        config.rewrite_grades < 0 or 15 < config.rewrite_grades
    ):
        errmsg = (
            f"{config.rewrite_grades}: `rewrite_grades` must be an"
            " integer within the range [0, 15]."
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

    if config.num_qr_intervals <= 0:
        errmsg = (
            f"{config.num_qr_intervals}: `num_qr_intervals` must be a"
            " positive integer."
        )
        raise RuntimeError(errmsg)

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

    if (config.target_update_interval is None) != (
        config.target_update_rate is None
    ):
        errmsg = (
            "`target_update_interval` must be consistent with"
            " `target_update_rate`."
        )
        raise RuntimeError(errmsg)
    enable_target_network = config.target_update_interval is not None

    encoder_snapshot_path: Path | None = None
    decoder_snapshot_path: Path | None = None
    source_encoder_snapshot_path: Path | None = None
    source_decoder_snapshot_path: Path | None = None
    target_encoder_snapshot_path: Path | None = None
    target_decoder_snapshot_path: Path | None = None
    optimizer_snapshot_path: Path | None = None
    scheduler_snapshot_path: Path | None = None
    num_samples = 0

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                if enable_target_network:
                    match = re.search(
                        "^(?:(?:source|target)-(?:encoder|decoder)|optimizer|lr-scheduler)(?:\\.(\\d+))?\\.pth$",
                        child,
                    )
                else:
                    match = re.search(
                        "^(?:encoder|decoder|optimizer|lr-scheduler)(?:\\.(\\d+))?\\.pth$",
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

        if enable_target_network:
            source_encoder_snapshot_path = (
                config.initial_model_prefix / f"source-encoder{infix}.pth"
            )
            assert source_encoder_snapshot_path is not None
            if not source_encoder_snapshot_path.exists():
                errmsg = f"{source_encoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not source_encoder_snapshot_path.is_file():
                errmsg = f"{source_encoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            source_decoder_snapshot_path = (
                config.initial_model_prefix / f"source-decoder{infix}.pth"
            )
            assert source_decoder_snapshot_path is not None
            if not source_decoder_snapshot_path.exists():
                errmsg = f"{source_decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not source_decoder_snapshot_path.is_file():
                errmsg = f"{source_decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target_encoder_snapshot_path = (
                config.initial_model_prefix / f"target-encoder{infix}.pth"
            )
            assert target_encoder_snapshot_path is not None
            if not target_encoder_snapshot_path.exists():
                errmsg = f"{target_encoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target_encoder_snapshot_path.is_file():
                errmsg = f"{target_encoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            target_decoder_snapshot_path = (
                config.initial_model_prefix / f"target-decoder{infix}.pth"
            )
            assert target_decoder_snapshot_path is not None
            if not target_decoder_snapshot_path.exists():
                errmsg = f"{target_decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not target_decoder_snapshot_path.is_file():
                errmsg = f"{target_decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)
        else:
            encoder_snapshot_path = (
                config.initial_model_prefix / f"encoder{infix}.pth"
            )
            assert encoder_snapshot_path is not None
            if not encoder_snapshot_path.exists():
                errmsg = f"{encoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not encoder_snapshot_path.is_file():
                errmsg = f"{encoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

            decoder_snapshot_path = (
                config.initial_model_prefix / f"decoder{infix}.pth"
            )
            assert decoder_snapshot_path is not None
            if not decoder_snapshot_path.exists():
                errmsg = f"{decoder_snapshot_path}: Does not exist."
                raise RuntimeError(errmsg)
            if not decoder_snapshot_path.is_file():
                errmsg = f"{decoder_snapshot_path}: Not a file."
                raise RuntimeError(errmsg)

        optimizer_snapshot_path = (
            config.initial_model_prefix / f"optimizer{infix}.pth"
        )
        assert optimizer_snapshot_path is not None
        if (
            not optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            optimizer_snapshot_path = None

        scheduler_snapshot_path = (
            config.initial_model_prefix / f"lr-scheduler{infix}.pth"
        )
        assert scheduler_snapshot_path is not None
        if (
            not scheduler_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            scheduler_snapshot_path = None

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

    if config.kappa < 0.0:
        errmsg = f"{config.kappa}: `kappa` must be a non-negative real value."
        raise RuntimeError(errmsg)

    if config.td_computation_batch_size == 0:
        config.td_computation_batch_size = config.batch_size
    if config.td_computation_batch_size < 0:
        errmsg = (
            f"{config.td_computation_batch_size}: "
            "`td_computation_batch_size` must be a non-negative integer."
        )
        raise RuntimeError(errmsg)

    if config.alpha < 0.0:
        errmsg = f"{config.alpha}: `alpha` must be a non-negative real value."
        raise RuntimeError(errmsg)

    if config.batch_size <= 0:
        errmsg = (
            f"{config.batch_size}: `batch_size` must be a positive integer."
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

    if (
        config.target_update_interval is not None
        and config.target_update_interval <= 0
    ):
        errmsg = (
            f"{config.target_update_interval}: "
            "`target_update_interval` must be a positive integer."
        )
        raise RuntimeError(errmsg)

    if config.target_update_rate is not None and (
        config.target_update_rate <= 0.0 or config.target_update_rate > 1.0
    ):
        errmsg = (
            f"{config.target_update_rate}: `target_update_rate` must be"
            " a real value within the range (0.0, 1.0]."
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

        logging.info("# of QR intervals: %d", config.num_qr_intervals)
        logging.info("Dueling architecture: %s", config.dueling_architecture)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial model prefix: %s", config.initial_model_prefix
            )
            logging.info("Initlal model index: %d", config.initial_model_index)
            if config.optimizer.initialize:
                logging.info("(Will not load optimizer)")

        logging.info("Reward plugin: %s", config.reward_plugin)
        logging.info("Discount factor: %f", config.discount_factor)
        logging.info("Kappa: %f", config.kappa)
        logging.info(
            "Batch size for TD computation: %d",
            config.td_computation_batch_size,
        )
        logging.info("Alpha: %f", config.alpha)
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

        if config.target_update_interval is not None:
            assert config.target_update_rate is not None
            logging.info(
                "Target network update interval: %d",
                config.target_update_interval,
            )
            logging.info(
                "Target network update rate: %f", config.target_update_rate
            )

        if config.initial_model_prefix is not None:
            if enable_target_network:
                logging.info(
                    "Initial source encoder snapshot: %s",
                    source_encoder_snapshot_path,
                )
                logging.info(
                    "Initial source decoder snapshot: %s",
                    source_decoder_snapshot_path,
                )
                logging.info(
                    "Initial target encoder snapshot: %s",
                    target_encoder_snapshot_path,
                )
                logging.info(
                    "Initial target decoder snapshot: %s",
                    target_decoder_snapshot_path,
                )
            else:
                logging.info(
                    "Initial encoder snapshot: %s", encoder_snapshot_path
                )
                logging.info(
                    "Initial decoder snapshot: %s", decoder_snapshot_path
                )
            if optimizer_snapshot_path is not None:
                logging.info(
                    "Initial optimizer snapshot: %s", optimizer_snapshot_path
                )
            if scheduler_snapshot_path is not None:
                logging.info(
                    "Initial LR scheduler snapshot: %s",
                    scheduler_snapshot_path,
                )

        logging.info("Output prefix: %s", output_prefix)
        if config.snapshot_interval == 0:
            logging.info("Snapshot interval: N/A")
        else:
            logging.info("Snapshot interval: %d", config.snapshot_interval)

    encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        num_layers=config.encoder.num_layers,
        checkpointing=config.checkpointing,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    encoder_tdm = TensorDictModule(
        encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],
        out_keys=["encode"],
    )
    decoder = QRDecoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers,
        num_qr_intervals=config.num_qr_intervals,
        dueling_architecture=config.dueling_architecture,
        noise_init_std=0.0,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    for _param in decoder.parameters():
        _param.data.zero_()
    decoder_tdm = TensorDictModule(
        decoder, in_keys=["candidates", "encode"], out_keys=["qr_action_value"]
    )
    network = TensorDictSequential(encoder_tdm, decoder_tdm)
    network = network.to(device=device, dtype=dtype)
    if world_size >= 2:
        init_process_group(backend="nccl")
        network = DistributedDataParallel(network)
        network = nn.SyncBatchNorm.convert_sync_batchnorm(network)

    target_encoder: Encoder | None = None
    target_encoder_tdm: TensorDictModule | None = None
    target_decoder: QRDecoder | None = None
    target_decoder_tdm: TensorDictModule | None = None
    target_network: TensorDictSequential | None = None
    if enable_target_network:
        target_encoder = Encoder(
            position_encoder=config.encoder.position_encoder,
            dimension=config.encoder.dimension,
            num_heads=config.encoder.num_heads,
            dim_feedforward=config.encoder.dim_feedforward,
            activation_function=config.encoder.activation_function,
            dropout=config.encoder.dropout,
            num_layers=config.encoder.num_layers,
            checkpointing=config.checkpointing,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        target_encoder_tdm = TensorDictModule(
            target_encoder,
            in_keys=["sparse", "numeric", "progression", "candidates"],
            out_keys=["encode"],
        )
        target_decoder = QRDecoder(
            input_dimension=config.encoder.dimension,
            dimension=config.decoder.dimension,
            activation_function=config.decoder.activation_function,
            dropout=config.decoder.dropout,
            num_layers=config.decoder.num_layers,
            num_qr_intervals=config.num_qr_intervals,
            dueling_architecture=config.dueling_architecture,
            noise_init_std=0.0,
            device=torch.device("cpu"),
            dtype=dtype,
        )
        target_decoder_tdm = TensorDictModule(
            target_decoder,
            in_keys=["candidates", "encode"],
            out_keys=["qr_action_value"],
        )
        target_network = TensorDictSequential(
            target_encoder_tdm, target_decoder_tdm
        )
        target_network.requires_grad_(False)
        target_network.eval()
        target_network = target_network.to(device=device, dtype=dtype)

        with torch.no_grad():
            for param, target_param in zip(
                network.parameters(), target_network.parameters()
            ):
                target_param.data = param.data.detach().clone()

    q_decoder = QDecoder()
    q_decoder_tdm = TensorDictModule(
        q_decoder, in_keys=["qr_action_value"], out_keys=["action_value"]
    )
    argmax_layer = DecodeConverter("argmax")
    argmax_layer_tdm = TensorDictModule(
        argmax_layer,
        in_keys=["candidates", "action_value"],
        out_keys=["action"],
    )
    if enable_target_network:
        assert target_encoder_tdm is not None
        assert target_decoder_tdm is not None
        network_to_save = TensorDictSequential(
            target_encoder_tdm,
            target_decoder_tdm,
            q_decoder_tdm,
            argmax_layer_tdm,
        )
        network_to_save.requires_grad_(False)
        network_to_save.eval()
        network_to_save.to(device=device, dtype=dtype)
    else:
        network_to_save = TensorDictSequential(
            encoder_tdm, decoder_tdm, q_decoder_tdm, argmax_layer_tdm
        )
        network_to_save.to(device=device, dtype=dtype)

    optimizer, scheduler = _config.optimizer.create(config, network)

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(
            config.encoder.load_from, map_location="cpu"
        )
        encoder.load_state_dict(encoder_state_dict)
        network.to(device=device, dtype=dtype)

        if enable_target_network:
            assert target_encoder is not None
            assert target_network is not None
            target_encoder_state_dict = torch.load(
                config.encoder.load_from, map_location="cpu"
            )
            target_encoder.load_state_dict(target_encoder_state_dict)
            target_network.to(device=device, dtype=dtype)

        network_to_save.to(device=device, dtype=dtype)

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert encoder_snapshot_path is not None
        assert decoder_snapshot_path is not None

        encoder_state_dict = torch.load(
            encoder_snapshot_path, map_location="cpu"
        )
        encoder.load_state_dict(encoder_state_dict)
        decoder_state_dict = torch.load(
            decoder_snapshot_path, map_location="cpu"
        )
        decoder.load_state_dict(decoder_state_dict)
        network.to(device=device, dtype=dtype)

        if enable_target_network:
            assert target_encoder_snapshot_path is not None
            assert target_encoder is not None
            assert target_decoder_snapshot_path is not None
            assert target_decoder is not None
            assert target_network is not None
            target_encoder_state_dict = torch.load(
                target_encoder_snapshot_path, map_location="cpu"
            )
            target_encoder.load_state_dict(target_encoder_state_dict)
            target_decoder_state_dict = torch.load(
                target_decoder_snapshot_path, map_location="cpu"
            )
            target_decoder.load_state_dict(target_decoder_state_dict)
            target_network.to(device=device, dtype=dtype)

        network_to_save.to(device=device, dtype=dtype)

        if optimizer_snapshot_path is not None:
            optimizer.load_state_dict(
                torch.load(optimizer_snapshot_path, map_location="cpu")
            )

        if scheduler is not None and scheduler_snapshot_path is not None:
            scheduler.load_state_dict(
                torch.load(scheduler_snapshot_path, map_location="cpu")
            )

    snapshots_path = output_prefix / "snapshots"

    def snapshot_writer(num_samples: int | None = None) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)

        infix = "" if num_samples is None else f".{num_samples}"

        if enable_target_network:
            assert target_encoder is not None
            assert target_decoder is not None
            torch.save(
                encoder.state_dict(),
                snapshots_path / f"source-encoder{infix}.pth",
            )
            torch.save(
                decoder.state_dict(),
                snapshots_path / f"source-decoder{infix}.pth",
            )
            torch.save(
                target_encoder.state_dict(),
                snapshots_path / f"target-encoder{infix}.pth",
            )
            torch.save(
                target_decoder.state_dict(),
                snapshots_path / f"target-decoder{infix}.pth",
            )
        else:
            torch.save(
                encoder.state_dict(), snapshots_path / f"encoder{infix}.pth"
            )
            torch.save(
                decoder.state_dict(), snapshots_path / f"decoder{infix}.pth"
            )
        torch.save(
            optimizer.state_dict(), snapshots_path / f"optimizer{infix}.pth"
        )
        if scheduler is not None:
            torch.save(
                scheduler.state_dict(),
                snapshots_path / f"lr-scheduler{infix}.pth",
            )

        if enable_target_network:
            assert target_encoder is not None
            assert target_decoder is not None
            encoder_to_save = target_encoder
            encoder_tdm_to_save = target_encoder_tdm
            decoder_to_save = target_decoder
            decoder_tdm_to_save = target_decoder_tdm
        else:
            encoder_to_save = encoder
            encoder_tdm_to_save = encoder_tdm
            decoder_to_save = decoder
            decoder_tdm_to_save = decoder_tdm
        network_state = dump_object(
            network_to_save,
            [
                dump_object(
                    encoder_tdm_to_save,
                    [
                        dump_model(
                            encoder_to_save,
                            [],
                            {
                                "position_encoder": config.encoder.position_encoder,
                                "dimension": config.encoder.dimension,
                                "num_heads": config.encoder.num_heads,
                                "dim_feedforward": config.encoder.dim_feedforward,
                                "activation_function": config.encoder.activation_function,
                                "dropout": config.encoder.dropout,
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
                    decoder_tdm_to_save,
                    [
                        dump_model(
                            decoder_to_save,
                            [],
                            {
                                "input_dimension": config.encoder.dimension,
                                "dimension": config.decoder.dimension,
                                "activation_function": config.decoder.activation_function,
                                "dropout": config.decoder.dropout,
                                "num_layers": config.decoder.num_layers,
                                "num_qr_intervals": config.num_qr_intervals,
                                "dueling_architecture": config.dueling_architecture,
                                "noise_init_std": 0.0,
                                "device": torch.device("cpu"),
                                "dtype": dtype,
                            },
                        )
                    ],
                    {
                        "in_keys": ["candidates", "encode"],
                        "out_keys": ["qr_action_value"],
                    },
                ),
                dump_object(
                    q_decoder_tdm,
                    [dump_model(q_decoder, [], {})],
                    {
                        "in_keys": ["qr_action_value"],
                        "out_keys": ["action_value"],
                    },
                ),
                dump_object(
                    argmax_layer_tdm,
                    [dump_model(argmax_layer, ["argmax"], {})],
                    {
                        "in_keys": ["candidates", "action_value"],
                        "out_keys": ["action"],
                    },
                ),
            ],
            {},
        )
        torch.save(network_state, snapshots_path / f"model{infix}.kanachan")

    tensorboard_path = output_prefix / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        torch.autograd.set_detect_anomaly(
            False
        )  # `True` for debbing purpose only.
        _training(
            training_data=config.training_data,
            contiguous_training_data=config.contiguous_training_data,
            rewrite_grades=config.rewrite_grades,
            num_workers=config.num_workers,
            replay_buffer_size=config.replay_buffer_size,
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            source_network=network,
            target_network=target_network,
            reward_plugin=config.reward_plugin,
            discount_factor=config.discount_factor,
            kappa=config.kappa,
            td_computation_batch_size=config.td_computation_batch_size,
            alpha=config.alpha,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            target_update_interval=config.target_update_interval,
            target_update_rate=config.target_update_rate,
            snapshot_interval=config.snapshot_interval,
            num_samples=num_samples,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer,
        )  # pylint: disable=missing-kwoa


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
    sys.exit(0)
