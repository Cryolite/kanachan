#!/usr/bin/env python3

import datetime
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tensordict import TensorDict  # type: ignore
from tensordict.nn import (  # type: ignore
    TensorDictModule,
    TensorDictSequential,
)
from torch import Tensor, nn
from torch.amp.grad_scaler import GradScaler
from torch.distributed import (
    init_process_group,
    broadcast,
    ReduceOp,
    all_reduce,
)
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard.writer import SummaryWriter

import kanachan.training.awr.config  # pylint: disable=unused-import
import kanachan.training.core.config as _config
from kanachan.constants import MAX_NUM_ACTION_CANDIDATES, NUM_TYPES_OF_ACTIONS
from kanachan.model_loader import dump_model, dump_object, load_model
from kanachan.nn import DecodeConverter, Decoder, Encoder
from kanachan.training.common import (
    get_distributed_environment,
    get_gradient,
    is_gradient_nan,
)
from kanachan.training.core.bc import DataLoader

SnapshotWriter = Callable[[int | None], None]


def _training(
    *,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    training_data: Path,
    rewrite_rooms: int | None,
    rewrite_grades: int | None,
    num_workers: int,
    q_model: nn.Module,
    value_model: nn.Module | None,
    policy_model: nn.Module,
    beta: float,
    weight_clipping: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    max_gradient_norm: float,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    snapshot_interval: int,
    num_samples: int,
    snapshot_writer: SnapshotWriter,
    summary_writer: SummaryWriter,
) -> None:
    start_time = datetime.datetime.now()

    world_size, _, local_rank = get_distributed_environment()

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    data_loader = DataLoader(
        path=training_data,
        num_skip_samples=num_samples,
        rewrite_rooms=rewrite_rooms,
        rewrite_grades=rewrite_grades,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(num_workers >= 1),
        drop_last=(world_size >= 2),
    )

    is_amp_enabled = dtype != amp_dtype
    autocast_kwargs: dict[str, Any] = {
        "device_type": device.type,
        "dtype": amp_dtype,
        "enabled": is_amp_enabled,
    }
    grad_scaler = GradScaler("cuda", enabled=is_amp_enabled)

    last_snapshot = None
    if snapshot_interval > 0:
        last_snapshot = 0

    batch_count = 0

    for data in data_loader:
        data = data.to(device=device)

        candidates = data["candidates"]
        assert isinstance(candidates, Tensor)
        assert candidates.device == device
        assert candidates.dtype == torch.int32
        assert candidates.dim() == 2
        assert candidates.size(0) == batch_size
        assert candidates.size(1) == MAX_NUM_ACTION_CANDIDATES

        action = data["action"]
        assert isinstance(action, Tensor)
        assert action.device == device
        assert action.dtype == torch.int32
        assert action.dim() == 1
        assert action.size(0) == batch_size

        mask = candidates >= NUM_TYPES_OF_ACTIONS

        copy: TensorDict = data.detach().clone()
        assert isinstance(copy, TensorDict)
        with torch.no_grad(), torch.autocast(**autocast_kwargs):
            q_model(copy)

        q = copy["action_value"].detach().clone()
        assert isinstance(q, Tensor)
        assert q.device == device
        assert q.dtype == dtype
        assert q.dim() == 2
        assert q.size(0) == batch_size
        assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
        q = q.masked_fill(mask, 0.0)

        value: torch.Tensor | None = None
        if value_model is None:
            value = None
        else:
            copy = data.detach().clone()
            assert isinstance(copy, TensorDict)
            with torch.no_grad(), torch.autocast(**autocast_kwargs):
                value_model(copy)

            value = copy["state_value"].detach().clone()
            assert isinstance(value, Tensor)
            assert value.device == device
            assert value.dtype == dtype
            assert value.dim() == 1
            assert value.size(0) == batch_size

        copy = data.detach().clone()
        assert isinstance(copy, TensorDict)
        with torch.autocast(**autocast_kwargs):
            policy_model(copy)

        log_probs: torch.Tensor = copy["log_probs"]
        assert isinstance(log_probs, Tensor)
        assert log_probs.device == device
        assert log_probs.dtype == dtype
        assert log_probs.dim() == 2
        assert log_probs.size(0) == batch_size
        assert log_probs.size(1) == MAX_NUM_ACTION_CANDIDATES

        probs = log_probs.exp()

        if value is None:
            value = torch.sum(probs * q, dim=1)

        q = q[torch.arange(batch_size, device=device), action]
        q /= beta
        q = torch.exp(q)
        value /= beta
        value = torch.exp(value)
        weight = q / value
        weight = torch.clamp(weight, max=weight_clipping)

        log_probs = log_probs[torch.arange(batch_size, device=device), action]
        loss = -log_probs * weight
        loss = torch.mean(loss)

        _loss = loss.detach().clone()
        if world_size >= 2:
            all_reduce(_loss, ReduceOp.AVG)
        loss_to_display = float(_loss.item())

        if math.isnan(loss_to_display):
            errmsg = "Training loss becomes NaN."
            raise RuntimeError(errmsg)

        loss = loss / gradient_accumulation_steps
        grad_scaler.scale(loss).backward()  # type: ignore

        num_samples += batch_size * world_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            is_grad_nan = is_gradient_nan(policy_model)
            if world_size >= 2:
                all_reduce(is_grad_nan)
            if is_grad_nan.item() >= 1:
                if local_rank == 0:
                    logging.warning(
                        "Skip an optimization step "
                        "because of NaN in the gradient."
                    )
                optimizer.zero_grad()
                continue

            grad_scaler.unscale_(optimizer)
            gradient = get_gradient(policy_model)
            # pylint: disable=not-callable
            gradient_norm = float(torch.linalg.vector_norm(gradient).item())
            nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                max_gradient_norm,
                error_if_nonfinite=False,
            )
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            if local_rank == 0:
                logging.info(
                    "sample = %d, loss = %E, gradient norm = %E",
                    num_samples,
                    loss_to_display,
                    gradient_norm,
                )
                summary_writer.add_scalar("Loss", loss_to_display, num_samples)
                summary_writer.add_scalar(
                    "Gradient Norm", gradient_norm, num_samples
                )
        else:
            if local_rank == 0:
                logging.info(
                    "sample = %d, loss = %E", num_samples, loss_to_display
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
            "Training has finished (elapsed time = %s).", elapsed_time
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
            "`rewrite_rooms` must be an integer within the range `[0, 4]`."
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
            "`rewrite_grades` must be an integer within the range `[0, 15]`."
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
                f"{config.num_workers}: An invalid number of workers "
                "for CPU."
            )
            raise RuntimeError(errmsg)
    elif device.type == "cuda":
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
    else:
        errmsg = f"{device.type}: An unsupported device type."
        raise ValueError(errmsg)

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
                "`initial_model_index` must be combined with "
                "`initial_model_prefix`."
            )
            raise RuntimeError(errmsg)
        if config.initial_model_index < 0:
            errmsg = (
                f"{config.initial_model_index}: "
                "An invalid initial model index."
            )
            raise RuntimeError(errmsg)

    num_samples = 0
    encoder_snapshot_path: Path | None = None
    decoder_snapshot_path: Path | None = None
    optimizer_snapshot_path: Path | None = None
    scheduler_snapshot_path: Path | None = None

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    "^(?:encoder|decoder|optimizer|scheduler)(?:\\.(\\d+))?\\.pth$",
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

        encoder_snapshot_path = Path(
            config.initial_model_prefix / f"encoder{infix}.pth"
        )
        if not encoder_snapshot_path.exists():
            errmsg = f"{encoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not encoder_snapshot_path.is_file():
            errmsg = f"{encoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        decoder_snapshot_path = Path(
            config.initial_model_prefix / f"decoder{infix}.pth"
        )
        if not decoder_snapshot_path.exists():
            errmsg = f"{decoder_snapshot_path}: Does not exist."
            raise RuntimeError(errmsg)
        if not decoder_snapshot_path.is_file():
            errmsg = f"{decoder_snapshot_path}: Not a file."
            raise RuntimeError(errmsg)

        optimizer_snapshot_path = Path(
            config.initial_model_prefix / f"optimizer{infix}.pth"
        )
        if (
            not optimizer_snapshot_path.is_file()
            or config.optimizer.initialize
        ):
            optimizer_snapshot_path = None

        scheduler_snapshot_path = Path(
            config.initial_model_prefix / f"scheduler{infix}.pth"
        )
        if optimizer_snapshot_path is None:
            scheduler_snapshot_path = None

    if not config.q_model.exists():
        errmsg = f"{config.q_model}: Does not exist."
        raise RuntimeError(errmsg)
    if not config.q_model.is_file():
        errmsg = f"{config.q_model}: Not a file."
        raise RuntimeError(errmsg)

    if config.value_model is not None and not config.value_model.exists():
        errmsg = f"{config.value_model}: Does not exist."
        raise RuntimeError(errmsg)
    if config.value_model is not None and not config.value_model.is_file():
        errmsg = f"{config.value_model}: Not a file."
        raise RuntimeError(errmsg)

    if config.beta <= 0.0:
        errmsg = f"{config.beta}: `beta` must be a positive real number."
        raise RuntimeError(errmsg)

    if config.weight_clipping <= 0.0:
        errmsg = (
            f"{config.weight_clipping}: `weight_clipping` must be a positive "
            "real number."
        )
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
            f"{config.max_gradient_norm}: "
            "`max_gradient_norm` must be a positive real value."
        )
        raise RuntimeError(errmsg)

    _config.optimizer.validate(config)

    if config.snapshot_interval < 0:
        errmsg = (
            f"{config.snapshot_interval}: "
            "`snapshot_interval` must be a non-negative integer."
        )
        raise RuntimeError(errmsg)

    output_prefix = Path(HydraConfig.get().runtime.output_dir)

    if local_rank == 0:
        logging.info("Model type: advantage weighted regression (AWR)")

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

        _config.encoder.dump(config)

        _config.decoder.dump(config)

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial model prefix: %s", config.initial_model_prefix
            )
            logging.info("Initlal model index: %d", config.initial_model_index)
            if config.optimizer.initialize:
                logging.info("(Will not load optimizer)")

        logging.info("Q model: %s", config.q_model)
        if config.value_model is None:
            logging.info("Value model: (N/A)")
        else:
            logging.info("Value model: %s", config.value_model)

        logging.info("Beta: %E", config.beta)
        logging.info("Weight clipping: %E", config.weight_clipping)
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

        if config.initial_model_prefix is not None:
            logging.info(
                "Initial encoder snapshot: %s",
                encoder_snapshot_path,
            )
            logging.info(
                "Initial decoder snapshot: %s",
                decoder_snapshot_path,
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

    if world_size >= 2:
        init_process_group(backend="nccl")

    q_model = load_model(config.q_model, map_location=torch.device("cpu"))
    q_model.to(device=config.device.type, dtype=dtype)
    q_model.requires_grad_(False)
    q_model.eval()

    if config.value_model is None:
        value_model = None
    else:
        value_model = load_model(
            config.value_model, map_location=torch.device("cpu")
        )
        value_model.to(device=config.device.type, dtype=dtype)
        value_model.requires_grad_(False)
        value_model.eval()

    encoder = Encoder(
        position_encoder=config.encoder.position_encoder,
        dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads,
        dim_feedforward=config.encoder.dim_feedforward,
        layer_normalization=config.encoder.layer_normalization,
        num_layers=config.encoder.num_layers,
        activation_function=config.encoder.activation_function,
        dropout=config.encoder.dropout,
        checkpointing=config.checkpointing,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    encoder_tdm = TensorDictModule(
        encoder,
        in_keys=["sparse", "numeric", "progression", "candidates"],  # type: ignore
        out_keys=["encode"],  # type: ignore
    )
    decoder = Decoder(
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
        for _param in decoder.parameters():
            _param.zero_()
    decoder_tdm = TensorDictModule(
        decoder,
        in_keys=["encode"],  # type: ignore
        out_keys=["decode"],  # type: ignore
    )
    decode_converter = DecodeConverter("log_probs")
    decode_converter_tdm = TensorDictModule(
        decode_converter,
        in_keys=["candidates", "decode"],  # type: ignore
        out_keys=["log_probs"],  # type: ignore
    )
    network_tdm = TensorDictSequential(
        encoder_tdm, decoder_tdm, decode_converter_tdm
    )
    if world_size >= 2:
        network_tdm.to(device=device)
        for _param in network_tdm.parameters():
            broadcast(_param.data, src=0)
        network_tdm.to(device="cpu")

    argmax_layer = DecodeConverter("argmax")
    argmax_tdm = TensorDictModule(
        argmax_layer,
        in_keys=["candidates", "log_probs"],  # type: ignore
        out_keys=["action"],  # type: ignore
    )
    network_tdm_to_save = TensorDictSequential(
        encoder_tdm, decoder_tdm, decode_converter_tdm, argmax_tdm
    )

    network_tdm.requires_grad_(True)
    network_tdm.train()
    network_tdm = network_tdm.to(device=device, dtype=dtype)
    if world_size >= 2:
        network_tdm = DistributedDataParallel(network_tdm)
        network_tdm = nn.SyncBatchNorm.convert_sync_batchnorm(network_tdm)
        assert isinstance(network_tdm, nn.Module)

    network_tdm_to_save.requires_grad_(True)
    network_tdm_to_save.train()
    network_tdm_to_save = network_tdm_to_save.to(device=device, dtype=dtype)

    optimizer, scheduler = _config.optimizer.create(
        device.type, config, network_tdm
    )

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(
            config.encoder.load_from, map_location="cpu", weights_only=True
        )
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(device=device, dtype=dtype)

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert encoder_snapshot_path is not None
        assert decoder_snapshot_path is not None

        encoder_state_dict = torch.load(
            encoder_snapshot_path, map_location="cpu"
        )
        encoder.load_state_dict(encoder_state_dict)
        encoder.to(device=device, dtype=dtype)

        decoder_state_dict = torch.load(
            decoder_snapshot_path, map_location="cpu"
        )
        decoder.load_state_dict(decoder_state_dict)
        decoder.to(device=device, dtype=dtype)

        if optimizer_snapshot_path is not None:
            optimizer_state_dict = torch.load(
                optimizer_snapshot_path, map_location="cpu"
            )
            optimizer.load_state_dict(optimizer_state_dict)

            for _optimizer_state in optimizer.state.values():
                assert isinstance(_optimizer_state, dict)
                for key, value in _optimizer_state.items():
                    if isinstance(value, Tensor):
                        _optimizer_state[key] = value.to(
                            device=device, dtype=dtype
                        )

        if scheduler_snapshot_path is not None and scheduler is not None:
            assert scheduler_snapshot_path is not None
            scheduler_state_dict = torch.load(
                scheduler_snapshot_path, map_location="cpu"
            )
            scheduler.load_state_dict(scheduler_state_dict)

    snapshots_path = output_prefix / "snapshots"

    def snapshot_writer(num_samples: int | None = None) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)

        infix = "" if num_samples is None else f".{num_samples}"

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
                snapshots_path / f"scheduler{infix}.pth",
            )

        state = dump_model(
            network_tdm_to_save,
            [
                dump_object(
                    encoder_tdm,
                    [
                        dump_object(
                            encoder,
                            [],
                            {
                                "position_encoder": config.encoder.position_encoder,
                                "dimension": config.encoder.dimension,
                                "num_heads": config.encoder.num_heads,
                                "dim_feedforward": config.encoder.dim_feedforward,
                                "layer_normalization": config.encoder.layer_normalization,
                                "num_layers": config.encoder.num_layers,
                                "activation_function": config.encoder.activation_function,
                                "dropout": config.encoder.dropout,
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
                    decoder_tdm,
                    [
                        dump_object(
                            decoder,
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
                        "out_keys": ["decode"],
                    },
                ),
                dump_object(
                    decode_converter_tdm,
                    [
                        dump_object(
                            decode_converter,
                            [],
                            {},
                        ),
                    ],
                    {
                        "in_keys": ["candidates", "decode"],
                        "out_keys": ["log_probs"],
                    },
                ),
                dump_object(
                    argmax_tdm,
                    [
                        dump_object(argmax_layer, ["argmax"], {}),
                    ],
                    {
                        "in_keys": ["candidates", "log_probs"],
                        "out_keys": ["action"],
                    },
                ),
            ],
            {},
        )
        torch.save(state, snapshots_path / f"model{infix}.kanachan")

    tensorboard_path = output_prefix / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        _training(
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            training_data=config.training_data,
            rewrite_rooms=config.rewrite_rooms,
            rewrite_grades=config.rewrite_grades,
            num_workers=config.num_workers,
            q_model=q_model,
            value_model=value_model,
            policy_model=network_tdm,
            beta=config.beta,
            weight_clipping=config.weight_clipping,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            snapshot_interval=config.snapshot_interval,
            num_samples=num_samples,
            snapshot_writer=snapshot_writer,
            summary_writer=summary_writer,
        )


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
    sys.exit(0)
