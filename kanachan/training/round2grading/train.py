import re
import math
from pathlib import Path
import datetime
import os
import logging
import sys
from typing import Optional, Callable
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp import GradScaler
from torch.distributed import (
    init_process_group,
    is_initialized,
    get_world_size,
    get_rank,
    ReduceOp,
    all_reduce,
)
from torch.utils.tensorboard.writer import SummaryWriter
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from kanachan.common import get_grading_point
import kanachan.training.core.config as _config
import kanachan.training.round2grading.config  # pylint: disable=unused-import
from kanachan.training.core.round import DataLoader
from kanachan.model_loader import dump_object, dump_model
from kanachan.nn import RoundEncoder, Decoder, Symexp, SymlogLoss
from kanachan.training.common import get_gradient, is_gradient_nan


Progress = Callable[[], None]
SnapshotWriter = Callable[[int | None], None]


def _training(
    *,
    device: torch.device,
    dtype: torch.dtype,
    amp_dtype: torch.dtype,
    training_data: Path,
    num_workers: int,
    num_samples: int,
    celestial_scale: int,
    network_tdm: nn.Module,
    batch_size: int,
    gradient_accumulation_steps: int,
    loss_function: str,
    max_gradient_norm: float,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    snapshot_interval: int,
    summary_writer: SummaryWriter,
    snapshot_writer: SnapshotWriter,
) -> None:
    start_time = datetime.datetime.now()

    if is_initialized():
        world_size = get_world_size()
        rank = get_rank()
    else:
        world_size = 1
        rank = 0

    is_amp_enabled = device.type != "cpu" and dtype != amp_dtype
    autocast_kwargs = {
        "device_type": device.type,
        "dtype": amp_dtype,
        "enabled": is_amp_enabled,
    }

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    data_loader = DataLoader(
        path=training_data,
        num_skip_samples=num_samples,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(num_workers >= 1),
        drop_last=True,
    )

    if loss_function == "l1":
        _loss_function = nn.L1Loss()
    elif loss_function == "l2":
        _loss_function = nn.MSELoss()
    elif loss_function == "smooth_l1":
        _loss_function = nn.HuberLoss()
    else:
        assert loss_function == "symlog"
        _loss_function = SymlogLoss()

    grad_scaler = None
    if is_amp_enabled:
        grad_scaler = GradScaler()

    last_snapshot = None
    if snapshot_interval > 0:
        last_snapshot = 0

    batch_count = 0

    for data in data_loader:

        def get_target(_data: TensorDict) -> Tensor:
            sparse: Tensor = _data["sparse"]
            results: Tensor = _data["results"]

            room: list[int] = sparse[:, 0].tolist()
            game_style: list[int] = (sparse[:, 1] - 5).tolist()
            _grades: Tensor = sparse[:, 2:6]
            _grades[:, 0] -= 7
            _grades[:, 1] -= 23
            _grades[:, 2] -= 39
            _grades[:, 3] -= 55
            grades: list[list[int]] = _grades.tolist()
            scores: list[list[int]] = results[:, 4:].tolist()

            targets: list[Tensor] = []
            for i in range(batch_size):
                all_celestial = all(map(lambda grade: grade == 15, grades[i]))
                target = torch.zeros(
                    4, device=torch.device("cpu"), dtype=dtype
                )
                for seat in range(4):
                    grade = grades[i][seat]
                    score = scores[i][seat]
                    ranking = 0
                    for j in range(seat):
                        if scores[i][j] >= score:
                            ranking += 1
                    for j in range(seat + 1, 4):
                        if scores[i][j] > score:
                            ranking += 1
                    target[seat] = get_grading_point(
                        room[i],
                        game_style[i],
                        grade,
                        ranking,
                        score,
                        celestial_scale,
                        all_celestial,
                    )
                targets.append(target)
            return torch.stack(targets)

        target = get_target(data)

        data = data.to(device=device)
        target = target.to(device=device)

        with torch.autocast(**autocast_kwargs):
            network_tdm(data)
        prediction: Tensor = data["decode"]
        loss: Tensor = _loss_function(prediction, target)

        _loss = loss
        if world_size >= 2:
            all_reduce(_loss, ReduceOp.AVG)
        loss_to_display = _loss.item()

        if math.isnan(loss_to_display):
            raise RuntimeError("Training loss becomes NaN.")

        loss = loss / gradient_accumulation_steps
        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()  # type: ignore

        num_samples += batch_size * world_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            if is_gradient_nan(network_tdm):
                logging.warning(
                    "Skip an optimization step because of NaN in the gradient."
                )
                optimizer.zero_grad()
                continue

            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            gradient = get_gradient(network_tdm)
            # pylint: disable=not-callable
            gradient_norm: float = torch.linalg.vector_norm(gradient).item()
            nn.utils.clip_grad_norm_(
                network_tdm.parameters(),
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

            if rank == 0:
                logging.info(
                    "sample = %d, loss = %E, gradient norm = %E",
                    num_samples,
                    loss_to_display,
                    gradient_norm,
                )
            summary_writer.add_scalar("Loss", loss_to_display, num_samples)
            summary_writer.add_scalar(
                "Gradient norm", gradient_norm, num_samples
            )
            if scheduler is not None:
                summary_writer.add_scalar(
                    "LR", scheduler.get_last_lr()[0], num_samples
                )
        else:
            if rank == 0:
                logging.info(
                    "sample = %d, loss = %E", num_samples, loss_to_display
                )
            summary_writer.add_scalar("Loss", loss_to_display, num_samples)

        if (
            rank == 0
            and last_snapshot is not None
            and num_samples - last_snapshot >= snapshot_interval
        ):
            snapshot_writer(num_samples)
            last_snapshot = num_samples

    elapsed_time = datetime.datetime.now() - start_time

    if rank == 0:
        logging.info(
            "A training epoch has finished (elapsed time = %s).", elapsed_time
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
        raise RuntimeError(f"{config.training_data}: Does not exist.")
    if not config.training_data.is_file():
        raise RuntimeError(f"{config.training_data}: Not a file.")

    if device.type == "cpu":
        if config.num_workers is None:
            config.num_workers = 0
        if config.num_workers < 0:
            raise RuntimeError(
                f"{config.num_workers}: An invalid number of workers."
            )
        if config.num_workers > 0:
            raise RuntimeError(
                f"{config.num_workers}: An invalid number of workers for CPU."
            )
    else:
        assert device.type == "cuda"
        if config.num_workers is None:
            config.num_workers = 2
        if config.num_workers < 0:
            raise RuntimeError(
                f"{config.num_workers}: An invalid number of workers."
            )
        if config.num_workers == 0:
            raise RuntimeError(
                f"{config.num_workers}: An invalid number of workers for GPU."
            )

    if config.celestial_scale <= 0:
        raise RuntimeError(
            f"{config.celestial_scale}: `celestial_scale` must be a positive inter."
        )

    _config.encoder.validate(config)

    _config.decoder.validate(config)

    if config.initial_model_prefix is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError(
                "`initial_model_prefix` conflicts with `encoder.load_from`."
            )
        if not config.initial_model_prefix.exists():
            raise RuntimeError(
                f"{config.initial_model_prefix}: Does not exist."
            )
        if not config.initial_model_prefix.is_dir():
            raise RuntimeError(
                f"{config.initial_model_prefix}: Not a directory."
            )

    if config.initial_model_index is not None:
        if config.initial_model_prefix is None:
            raise RuntimeError(
                "`initial_model_index` must be combined with `initial_model_prefix`."
            )
        if config.initial_model_index < 0:
            raise RuntimeError(
                f"{config.initial_model_index}: An invalid initial model index."
            )

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
            raise RuntimeError(
                f"{config.initial_model_prefix}: No model snapshot found."
            )

        if config.initial_model_index == sys.maxsize:
            config.initial_model_index = 0
            infix = ""
        else:
            num_samples = config.initial_model_index
            infix = f".{num_samples}"

        encoder_snapshot_path = (
            config.initial_model_prefix / f"encoder{infix}.pth"
        )
        assert encoder_snapshot_path is not None
        if not encoder_snapshot_path.exists():
            raise RuntimeError(f"{encoder_snapshot_path}: Does not exist.")
        if not encoder_snapshot_path.is_file():
            raise RuntimeError(f"{encoder_snapshot_path}: Not a file.")

        decoder_snapshot_path = (
            config.initial_model_prefix / f"decoder{infix}.pth"
        )
        assert decoder_snapshot_path is not None
        if not decoder_snapshot_path.exists():
            raise RuntimeError(f"{decoder_snapshot_path}: Does not exist.")

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

    if config.batch_size <= 0:
        raise RuntimeError(
            f"{config.batch_size}: `batch_size` must be a positive integer."
        )

    if config.loss_function not in ("l1", "l2", "smooth_l1", "symlog"):
        raise RuntimeError(
            f"{config.loss_function}: An invalid value for `loss_function`."
        )

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f"{config.gradient_accumulation_steps}: "
            "`gradient_accumulation_steps` must be an integer greater than 0."
        )

    if config.max_gradient_norm <= 0.0:
        raise RuntimeError(
            f"{config.max_gradient_norm}: `max_gradient_norm` must be a positive real value."
        )

    _config.optimizer.validate(config)

    if config.snapshot_interval < 0:
        raise RuntimeError(
            f"{config.snapshot_interval}: `snapshot_interval` must be a non-negative integer."
        )

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

        logging.info("Checkpointing: %s", config.checkpointing)
        logging.info("Batch size: %d", config.batch_size)
        logging.info("Loss function: %s", config.loss_function)
        logging.info(
            "# of steps for gradient accumulation: %d",
            config.gradient_accumulation_steps,
        )
        logging.info(
            "Norm threshold for gradient clipping: %E",
            config.max_gradient_norm,
        )

        _config.optimizer.dump(config)

        if config.initial_model_prefix is not None:
            logging.info("Initial encoder snapshot: %s", encoder_snapshot_path)
            logging.info("Initial decoder snapshot: %s", decoder_snapshot_path)
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

    encoder = RoundEncoder(
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
        encoder, in_keys=["sparse", "numeric"], out_keys=["encode"]
    )
    decoder = Decoder(
        input_dimension=config.encoder.dimension,
        dimension=config.decoder.dimension,
        activation_function=config.decoder.activation_function,
        dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers,
        output_mode="scores",
        noise_init_std=0.0,
        device=torch.device("cpu"),
        dtype=dtype,
    )
    decoder_tdm = TensorDictModule(
        decoder, in_keys=["encode"], out_keys=["decode"]
    )
    network_tdm = TensorDictSequential(encoder_tdm, decoder_tdm)
    network_tdm.to(device=device, dtype=dtype)
    if world_size >= 2:
        init_process_group(backend="nccl")
        network_tdm = DistributedDataParallel(network_tdm)
        network_tdm = nn.SyncBatchNorm.convert_sync_batchnorm(network_tdm)

    identity = nn.Identity()
    identity_tdm = TensorDictModule(
        identity, in_keys=["decode"], out_keys=["grading_points"]
    )
    symexp_module = Symexp()
    symexp_tdm = TensorDictModule(
        symexp_module, in_keys=["decode"], out_keys=["grading_points"]
    )
    if config.loss_function == "symlog":
        network_tdm_to_save = TensorDictSequential(
            encoder_tdm, decoder_tdm, symexp_tdm
        )
    else:
        network_tdm_to_save = TensorDictSequential(
            encoder_tdm, decoder_tdm, identity_tdm
        )

    optimizer, scheduler = _config.optimizer.create(config, network_tdm)

    if config.encoder.load_from is not None:
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(
            config.encoder.load_from, map_location="cpu"
        )
        encoder.load_state_dict(encoder_state_dict)
        if device.type == "cuda":
            encoder.cuda()

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert encoder_snapshot_path is not None
        assert decoder_snapshot_path is not None

        encoder_state_dict = torch.load(
            encoder_snapshot_path, map_location="cpu"
        )
        encoder.load_state_dict(encoder_state_dict)
        if device.type == "cuda":
            encoder.cuda()

        decoder_state_dict = torch.load(
            decoder_snapshot_path, map_location="cpu"
        )
        decoder.load_state_dict(decoder_state_dict)
        if device.type == "cuda":
            decoder.cuda()

        if optimizer_snapshot_path is not None:
            optimizer.load_state_dict(
                torch.load(optimizer_snapshot_path, map_location="cpu")
            )

        if scheduler_snapshot_path is not None:
            assert scheduler is not None
            scheduler.load_state_dict(
                torch.load(scheduler_snapshot_path, map_location="cpu")
            )

    snapshots_path = output_prefix / "snapshots"

    def snapshot_writer(num_samples: Optional[int] = None) -> None:
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
                snapshots_path / f"lr-scheduler{infix}.pth",
            )

        _encoder_state_dict = dump_object(
            encoder_tdm,
            [
                dump_model(
                    encoder,
                    [],
                    {
                        "position_encoder": config.encoder.position_encoder,
                        "dimension": config.encoder.dimension,
                        "num_heads": config.encoder.num_heads,
                        "dim_feedforward": config.encoder.dim_feedforward,
                        "activation_function": config.encoder.activation_function,
                        "dropout": config.encoder.dropout,
                        "num_layers": config.encoder.num_layers,
                        "checkpointing": False,
                        "device": torch.device("cpu"),
                        "dtype": dtype,
                    },
                )
            ],
            {"in_keys": ["sparse", "numeric"], "out_keys": ["encode"]},
        )
        _decoder_state_dict = dump_object(
            decoder_tdm,
            [
                dump_model(
                    decoder,
                    [],
                    {
                        "input_dimension": config.encoder.dimension,
                        "dimension": config.decoder.dimension,
                        "activation_function": config.decoder.activation_function,
                        "dropout": config.decoder.dropout,
                        "num_layers": config.decoder.num_layers,
                        "output_mode": "scores",
                        "noise_init_std": 0.0,
                        "device": torch.device("cpu"),
                        "dtype": dtype,
                    },
                )
            ],
            {"in_keys": ["encode"], "out_keys": ["decode"]},
        )
        if config.loss_function == "symlog":
            _decode_transformer_state_dict = dump_object(
                symexp_tdm,
                [dump_model(Symexp(), [], {})],
                {"in_keys": ["decode"], "out_keys": ["grading_points"]},
            )
        else:
            _decode_transformer_state_dict = dump_object(
                identity_tdm,
                [dump_model(nn.Identity(), [], {})],
                {"in_keys": ["decode"], "out_keys": ["grading_points"]},
            )
        network_tdm_state = dump_object(
            network_tdm_to_save,
            [
                _encoder_state_dict,
                _decoder_state_dict,
                _decode_transformer_state_dict,
            ],
            {},
        )
        torch.save(
            network_tdm_state, snapshots_path / f"model{infix}.kanachan"
        )

    tensorboard_path = output_prefix / "tensorboard"
    tensorboard_path.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        torch.autograd.set_detect_anomaly(
            False
        )  # `True` for debbing purpose only.
        _training(
            device=device,
            dtype=dtype,
            amp_dtype=amp_dtype,
            training_data=config.training_data,
            num_workers=config.num_workers,
            num_samples=num_samples,
            celestial_scale=config.celestial_scale,
            network_tdm=network_tdm,
            batch_size=config.batch_size,
            loss_function=config.loss_function,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm,
            optimizer=optimizer,
            scheduler=scheduler,
            snapshot_interval=config.snapshot_interval,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer,
        )


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
    sys.exit(0)
