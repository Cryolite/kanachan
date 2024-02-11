#!/usr/bin/env python3

import re
import datetime
import math
from pathlib import Path
import os
import logging
import sys
from typing import Callable, Optional
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
from torch import backends
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer, SGD, Adam, RAdam
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.distributed import init_process_group, all_reduce
from torch.utils.tensorboard.writer import SummaryWriter
from apex.optimizers import FusedAdam, FusedSGD, FusedLAMB
from kanachan.constants import NUM_TYPES_OF_ACTIONS, MAX_NUM_ACTION_CANDIDATES
from kanachan.training.common import Dataset, get_gradient, is_gradient_nan
import kanachan.training.awr.config # pylint: disable=unused-import
from kanachan.nn import Encoder
from kanachan.model_loader import load_model, dump_object, dump_model
from kanachan.training.awr.policy_model import PolicyDecoder, PolicyModel
from kanachan.training.core.offline_rl import DataIterator


SnapshotWriter = Callable[[Optional[int]], None]


def _training(
        *, is_multiprocess: bool, world_size: Optional[int], rank: Optional[int],
        is_main_process: bool, training_data: Path, num_workers: int, device: torch.device,
        dtype: torch.dtype, amp_dtype: torch.dtype, value_model: Optional[nn.Module],
        q_model: nn.Module, policy_model: nn.Module, beta: float, weight_clipping: float,
        batch_size: int, gradient_accumulation_steps: int, max_gradient_norm: float,
        optimizer: Optimizer, scheduler, snapshot_interval: int,
        num_samples: int, snapshot_writer: SnapshotWriter, summary_writer: SummaryWriter) -> None:
    start_time = datetime.datetime.now()

    is_amp_enabled = (device != 'cpu' and dtype != amp_dtype)

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    dataset = Dataset(training_data, DataIterator)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(num_workers >= 1),
        drop_last=is_multiprocess)

    last_snapshot = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    num_consumed_samples = 0
    batch_count = 0

    grad_scaler = None
    if is_amp_enabled:
        grad_scaler = GradScaler()

    for annotation in data_loader:
        if num_consumed_samples < num_samples:
            num_consumed_samples += batch_size
            continue

        if is_multiprocess:
            assert world_size is not None
            assert rank is not None
            if batch_size % world_size != 0:
                raise RuntimeError(
                    'Batch size must be divisible by the world size.')
            first = (batch_size // world_size) * rank
            last = (batch_size // world_size) * (rank + 1)
            annotation = tuple(x[first:last] for x in annotation)

        if device != 'cpu':
            annotation = tuple(x.cuda() for x in annotation)

        local_batch_size = annotation[0].size(0)
        world_batch_size = batch_size

        with torch.no_grad():
            q: torch.Tensor = q_model(*annotation[:4])
            assert q.dim() == 2
            assert q.size(0) == local_batch_size
            assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
            q = q.detach()
            if value_model is None:
                value = None
            else:
                value: torch.Tensor = value_model(annotation[:4])
                assert value.dim() == 1
                assert value.size(0) == local_batch_size
                value = value.detach()
        policy: torch.Tensor = policy_model(*annotation[:4])
        assert policy.dim() == 2
        assert policy.size(0) == local_batch_size
        assert policy.size(1) == MAX_NUM_ACTION_CANDIDATES
        if value is None:
            q_ = torch.where(annotation[3] < NUM_TYPES_OF_ACTIONS, q, 0.0)
            value = torch.sum(policy * q_, dim=1)
            assert value.dim() == 1
            assert value.size(0) == local_batch_size
        q = q[torch.arange(local_batch_size), annotation[4]]
        q /= beta
        q = torch.exp(q)
        value /= beta
        value = torch.exp(value)
        weight = q / value
        weight = torch.clamp(weight, max=weight_clipping)
        policy = policy[torch.arange(local_batch_size), annotation[4]]
        policy = torch.log(policy)
        loss = -weight * policy
        loss = torch.mean(loss)
        if math.isnan(loss.item()):
            raise RuntimeError('Loss becomes NaN.')

        batch_loss = loss
        if is_multiprocess:
            all_reduce(batch_loss)
            batch_loss /= world_size
        loss_to_display: float = batch_loss.item()

        loss /= gradient_accumulation_steps
        if grad_scaler is None:
            loss.backward()
        else:
            grad_scaler.scale(loss).backward()

        num_samples += world_batch_size
        num_consumed_samples += world_batch_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            is_grad_nan = is_gradient_nan(policy_model)
            is_grad_nan = torch.where(
                is_grad_nan, torch.ones_like(is_grad_nan), torch.zeros_like(is_grad_nan))
            all_reduce(is_grad_nan)
            if is_grad_nan.item() >= 1:
                if is_main_process:
                    logging.warning('NaN in the gradient.')

            if grad_scaler is not None:
                grad_scaler.unscale_(optimizer)
            gradient = get_gradient(policy_model)
            gradient_norm: float = torch.linalg.vector_norm(gradient).item()
            nn.utils.clip_grad_norm_(
                policy_model.parameters(), max_gradient_norm, error_if_nonfinite=False)
            if grad_scaler is None:
                optimizer.step()
            else:
                grad_scaler.step(optimizer)
                grad_scaler.update()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()

            if is_main_process:
                logging.info(
                    'sample = %d, loss = %E, gradient norm = %E',
                    num_samples, loss_to_display, gradient_norm)
                summary_writer.add_scalar('Loss', loss_to_display, num_samples)
                summary_writer.add_scalar('Gradient Norm', gradient_norm, num_samples)
        else:
            if is_main_process:
                logging.info('sample = %d, loss = %E', num_samples, loss_to_display)
                summary_writer.add_scalar('Loss', loss_to_display, num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            snapshot_writer(num_samples)
            last_snapshot = num_samples

    elapsed_time = datetime.datetime.now() - start_time

    if is_main_process:
        logging.info('Training has finished (elapsed time = %s).', elapsed_time)
        snapshot_writer()


@hydra.main(version_base=None, config_name='config')
def _main(config: DictConfig) -> None:
    if 'LOCAL_RANK' in os.environ:
        if os.environ['WORLD_SIZE'] != os.environ['LOCAL_WORLD_SIZE']:
            raise RuntimeError('Multi-node not supported.')
        world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        is_multiprocess = True
        is_main_process = rank == 0
    else:
        world_size = None
        rank = None
        is_multiprocess = False
        is_main_process = True

    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: Does not exist.')
    if not config.training_data.is_file():
        raise RuntimeError(f'{config.training_data}: Not a file.')

    if config.device.type == 'cpu':
        if config.num_workers is None:
            config.num_workers = 0
        if config.num_workers < 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers.')
        if config.num_workers > 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers for CPU.')
    else:
        if config.num_workers is None:
            config.num_workers = 2
        if config.num_workers < 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers.')
        if config.num_workers == 0:
            raise RuntimeError(f'{config.num_workers}: An invalid number of workers for GPU.')

    if config.device.type is not None:
        match = re.search('^(?:cpu|cuda(?::\\d+)?)$', config.device.type)
        if match is None:
            raise RuntimeError(f'{config.device.type}: An invalid device.')
    elif backends.cuda.is_built():
        config.device.type = 'cuda'
    else:
        config.device.type = 'cpu'
    if config.device.type == 'cuda':
        torch.cuda.set_device(rank)

    if config.device.dtype not in ('float64', 'double', 'float32', 'float', 'float16', 'half', 'bfloat16'):
        raise RuntimeError(f'{config.device.dtype}: An invalid dtype.')
    dtype = {
        'float64': torch.float64, 'double': torch.float64,
        'float32': torch.float32, 'float': torch.float32,
        'float16': torch.float16, 'half': torch.float16
    }[config.device.dtype]

    if config.device.type == 'cpu':
        if config.device.amp_dtype is not None:
            raise RuntimeError('AMP is not supported on CPU.')
        config.device.amp_dtype = config.device.dtype
    if config.device.amp_dtype is None:
        config.device.amp_dtype = config.device.dtype
    if config.device.amp_dtype not in ('float64', 'double', 'float32', 'float', 'float16', 'half'):
        raise RuntimeError(f'{config.device.amp_dtype}: An invalid AMP dtype.')
    amp_dtype = {
        'float64': torch.float64, 'double': torch.float64,
        'float32': torch.float32, 'float': torch.float32,
        'float16': torch.float16, 'half': torch.float16
    }[config.device.amp_dtype]
    if dtype == torch.float32 and amp_dtype == torch.float64:
        raise RuntimeError(
            f'An invalid combination of `device.dtype` (`{config.device.dtype}`) and '
            f'`device.amp_dtype` (`{config.device.amp_dtype}`).')
    if dtype == torch.float16 and amp_dtype in (torch.float64, torch.float32):
        raise RuntimeError(
            f'An invalid combination of `device.dtype` (`{config.device.dtype}`) and '
            f'`device.amp_dtype` (`{config.device.amp_dtype}`).')

    if backends.cudnn.is_available():
        backends.cudnn.benchmark = True

    if config.encoder.position_encoder not in ('positional_encoding', 'position_embedding'):
        raise RuntimeError(f'{config.encoder.position_encoder}: An invalid position encoder.')

    if config.encoder.dimension < 1:
        raise RuntimeError(
            f'{config.encoder.dimension}: `encoder.dimension` must be a positive integer.')

    if config.encoder.num_heads < 1:
        raise RuntimeError(
            f'{config.encoder.num_heads}: `encoder.num_heads` must be a positive integer.')

    if config.encoder.dim_feedforward is None:
        config.encoder.dim_feedforward = 4 * config.encoder.dimension
    if config.encoder.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.encoder.dim_feedforward}: '
            '`encoder.dim_feedforward` must be a positive integer.')

    if config.encoder.activation_function not in ('relu', 'gelu'):
        raise RuntimeError(
            f'{config.encoder.activation_function}: '
            'An invalid activation function for the encoder.')

    if config.encoder.dropout < 0.0 or 1.0 <= config.encoder.dropout:
        raise RuntimeError(
            f'{config.encoder.dropout}:'
            ' `encoder.dropout` must be a real number within the range [0.0, 1.0).')

    if config.encoder.num_layers < 1:
        raise RuntimeError(
            f'{config.encoder.num_layers}: `encoder.num_layers` must be a positive integer.')

    if config.encoder.load_from is not None:
        if not config.encoder.load_from.exists():
            raise RuntimeError(f'{config.encoder.load_from}: Does not exist.')
        if not config.encoder.load_from.is_file():
            raise RuntimeError(f'{config.encoder.load_from}: Not a file.')

    if config.decoder.dim_feedforward is None:
        config.decoder.dim_feedforward = config.encoder.dim_feedforward
    if config.decoder.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.decoder.dim_feedforward}:'
            ' `decoder.dim_feedforward` must be a positive integer.')

    if config.decoder.activation_function not in ('relu', 'gelu'):
        raise RuntimeError(
            f'{config.decoder.activation_function}: '
            'An invalid activation function for the decoder.')

    if config.decoder.dropout < 0.0 or 1.0 <= config.decoder.dropout:
        raise RuntimeError(
            f'{config.decoder.dropout}:'
            ' `decoder.dropout` must be a real number within the range [0.0, 1.0).')

    if config.decoder.num_layers < 1:
        raise RuntimeError(
            f'{config.decoder.num_layers}: `decoder.num_layers` must be a positive integer.')

    if config.decoder.load_from is not None:
        if not config.decoder.load_from.exists():
            raise RuntimeError(f'{config.decoder.load_from}: Does not exist.')
        if not config.decoder.load_from.is_file():
            raise RuntimeError(f'{config.decoder.load_from}: Not a file.')

    if config.initial_model is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError('`initial_model` conflicts with `encoder.load_from`.')
        if config.decoder.load_from is not None:
            raise RuntimeError('`initial_model` conflicts with `decoder.load_from`.')
        if not config.initial_model.exists():
            raise RuntimeError(f'{config.initial_model}: Does not exist.')
        if not config.initial_model.is_file():
            raise RuntimeError(f'{config.initial_model}: Not a file.')

    if config.initial_model_prefix is not None:
        if config.encoder.load_from is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `encoder.load_from`.')
        if config.decoder.load_from is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `decoder.load_from`.')
        if config.initial_model is not None:
            raise RuntimeError('`initial_model_prefix` conflicts with `initial_model`.')
        if not config.initial_model_prefix.exists():
            raise RuntimeError(f'{config.initial_model_prefix}: Does not exist.')
        if not config.initial_model_prefix.is_dir():
            raise RuntimeError(f'{config.initial_model_prefix}: Not a directory.')

    if config.initial_model_index is not None:
        if config.initial_model_prefix is None:
            raise RuntimeError('`initial_model_index` must be combined with `initial_model_prefix`.')
        if config.initial_model_index < 0:
            raise RuntimeError(f'{config.initial_model_index}: An invalid initial model index.')

    num_samples = 0

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model is None

        if config.initial_model_index is None:
            for child in os.listdir(config.initial_model_prefix):
                match = re.search(
                    '^(?:policy-(?:encoder|decoder)|optimizer|scheduler)(?:\\.(\\d+))?\\.pth$',
                    child)
                if match is None:
                    continue
                if match[1] is None:
                    config.initial_model_index = sys.maxsize
                    continue
                if config.initial_model_index is None or int(match[1]) > config.initial_model_index:
                    config.initial_model_index = int(match[1])
                    continue
        if config.initial_model_index is None:
            raise RuntimeError(f'{config.initial_model_prefix}: No model snapshot found.')

        if config.initial_model_index == sys.maxsize:
            config.initial_model_index = 0
            infix = ''
        else:
            num_samples = config.initial_model_index
            infix = f'.{num_samples}'

        policy_encoder_snapshot_path: Path = config.initial_model_prefix / f'policy-encoder{infix}.pth'
        if not policy_encoder_snapshot_path.exists():
            raise RuntimeError(f'{policy_encoder_snapshot_path}: Does not exist.')
        if not policy_encoder_snapshot_path.is_file():
            raise RuntimeError(f'{policy_encoder_snapshot_path}: Not a file.')

        policy_decoder_snapshot_path: Path = config.initial_model_prefix / f'policy-decoder{infix}.pth'
        if not policy_decoder_snapshot_path.exists():
            raise RuntimeError(f'{policy_decoder_snapshot_path}: Does not exist.')
        if not policy_decoder_snapshot_path.is_file():
            raise RuntimeError(f'{policy_decoder_snapshot_path}: Not a file.')

        optimizer_snapshot_path: Path = config.initial_model_prefix / f'optimizer{infix}.pth'
        if not optimizer_snapshot_path.is_file() or config.optimizer.initialize:
            optimizer_snapshot_path = None

        scheduler_snapshot_path: Path = config.initial_model_prefix / f'scheduler{infix}.pth'
        if optimizer_snapshot_path is None:
            scheduler_snapshot_path = None
        if scheduler_snapshot_path.exists() and not scheduler_snapshot_path.is_file():
            raise RuntimeError(f'{scheduler_snapshot_path}: Not a file.')

    if config.value_model is not None and not config.value_model.exists():
        raise RuntimeError(f'{config.value_model}: Does not exist.')
    if config.value_model is not None and not config.value_model.is_file():
        raise RuntimeError(f'{config.value_model}: Not a file.')

    if not config.q_model.exists():
        raise RuntimeError(f'{config.q_model}: Does not exist.')
    if not config.q_model.is_file():
        raise RuntimeError(f'{config.q_model}: Not a file.')

    if config.beta <= 0.0:
        raise RuntimeError(f'{config.beta}: `beta` must be a positive real number.')

    if config.weight_clipping <= 0.0:
        raise RuntimeError(
            f'{config.weight_clipping}: `weight_clipping` must be a positive real number.')

    if config.batch_size < 1:
        raise RuntimeError(f'{config.batch_size}: `batch_size` must be a positive integer.')
    if config.batch_size % world_size != 0:
        raise RuntimeError(f'`batch_size` must be divisible by the world size ({world_size}).')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: '
            '`gradient_accumulation_steps` must be a positive integer.')

    if config.max_gradient_norm <= 0.0:
        raise RuntimeError(
            f'{config.max_gradient_norm}: `max_gradient_norm` must be a positive real value.')

    if config.optimizer.type in ('sgd',):
        if config.optimizer.momentum is None:
            raise RuntimeError('`optimizer.momentum` must be specified for `sgd`.')
        if config.optimizer.momentum < 0.0 or 1.0 <= config.optimizer.momentum:
            raise RuntimeError(
                f'{config.optimizer.momentum}: '
                '`optimizer.momentum` must be a real value within the range [0.0, 1.0).')
    else:
        if config.optimizer.momentum is not None:
            raise RuntimeError(f'`optimizer.momentum` is useless for `{config.optimizer.type}`.')

    if config.optimizer.type in ('sgd',):
        if config.optimizer.epsilon is not None:
            raise RuntimeError(f'`optimizer.epsilon` is useless for `{config.optimizer.type}`.')
    else:
        if config.optimizer.epsilon is None:
            if config.optimizer.type in ('adam', 'radam',):
                config.optimizer.epsilon = 1.0e-8
            elif config.optimizer in ('lamb',):
                config.optimizer.epsilon = 1.0e-6
            else:
                raise NotImplementedError(config.optimizer.type)
    if config.optimizer.epsilon is not None and config.optimizer.epsilon <= 0.0:
        raise RuntimeError(
            f'{config.optimizer.epsilon}: `optimizer.epsilon` must be a positive real value.')

    if config.optimizer.learning_rate <= 0.0:
        raise RuntimeError(
            f'{config.optimizer.learning_rate}: '
            '`optimizer.learning_rate` must be a positive real value.')

    if config.optimizer.warmup_steps < 0:
        raise RuntimeError(
            f'{config.optimizer.warmup_steps}: '
            '`optimizer.warmup_steps` must be a non-negative integer.')

    if config.optimizer.annealing_steps < 0:
        raise RuntimeError(
            f'{config.optimizer.annealing_steps}: '
            '`optimizer.annealing_steps` must be a non-negative integer.')

    if config.optimizer.annealing_steps_factor <= 0:
        raise RuntimeError(
            f'{config.optimizer.annealing_steps_factor}: '
            '`optimizer.annealing_steps_factor` must be a positive integer.')

    if config.snapshot_interval < 0:
        raise RuntimeError(
            f'{config.snapshot_interval}: `snapshot_interval` must be a non-negative integer.')

    output_prefix = Path(HydraConfig.get().runtime.output_dir)
    tensorboard_path = output_prefix / 'tensorboard'
    snapshots_path = output_prefix / 'snapshots'

    if is_main_process:
        if world_size is None:
            assert rank is None
            logging.info('World size: N/A (single process)')
            logging.info('Process rank: N/A (single process)')
        else:
            assert rank is not None
            logging.info('World size: %d', world_size)
            logging.info('Process rank: %d', rank)
        logging.info('Training data: %s', config.training_data)
        if num_samples > 0:
            logging.info('# of training samples consumed so far: %d', num_samples)
        logging.info('# of workers: %d', config.num_workers)
        logging.info('Device: %s', config.device.type)
        if backends.cudnn.is_available():
            logging.info('cuDNN: available')
        else:
            logging.info('cuDNN: N/A')
        logging.info('dtype: %s', dtype)
        if dtype == amp_dtype:
            logging.info('AMP dtype: (AMP is disabled)')
        else:
            logging.info('AMP dtype: %s', amp_dtype)
        logging.info('Position encoder: %s', config.encoder.position_encoder)
        logging.info('Encoder dimension: %d', config.encoder.dimension)
        logging.info('# of heads for encoder: %d', config.encoder.num_heads)
        logging.info(
            'Dimension of feedforward networks for encoder: %d', config.encoder.dim_feedforward)
        logging.info('Activation function for encoder: %s', config.encoder.activation_function)
        logging.info('Dropout for encoder: %f', config.encoder.dropout)
        logging.info('# of encoder layers: %d', config.encoder.num_layers)
        if config.encoder.load_from is not None:
            logging.info('Load encoder from: %s', config.encoder.load_from)
        if config.decoder.num_layers >= 2:
            logging.info(
                'Dimension of feedforward networks for decoder: %d', config.decoder.dim_feedforward)
        logging.info('Activation function for decoder: %s', config.decoder.activation_function)
        logging.info('Dropout for decoder: %f', config.decoder.dropout)
        logging.info('# of decoder layers: %d', config.decoder.num_layers)
        if config.decoder.load_from is not None:
            logging.info('Load decoder from: %s', config.decoder.load_from)
        if config.initial_model is not None:
            logging.info('Load model from: %s', config.initial_model)
        if config.initial_model_prefix is not None:
            logging.info('Initial model prefix: %s', config.initial_model_prefix)
            logging.info('Initlal model index: %d', config.initial_model_index)
            if config.optimizer.initialize:
                logging.info('(Will not load optimizer)')
        if config.value_model is None:
            logging.info('Value model: (N/A)')
        else:
            logging.info('Value model: %s', config.value_model)
        logging.info('Q model: %s', config.q_model)
        logging.info('Beta: %E', config.beta)
        logging.info('Weight clipping: %E', config.weight_clipping)
        logging.info('Checkpointing: %s', config.checkpointing)
        if world_size is None:
            logging.info('Batch size: %d', config.batch_size)
        else:
            logging.info('Local batch size: %d', config.batch_size // world_size)
            logging.info('World batch size: %d', config.batch_size)
        logging.info('# of steps for gradient accumulation: %d', config.gradient_accumulation_steps)
        logging.info(
            'Virtual batch size: %d', config.batch_size * config.gradient_accumulation_steps)
        logging.info('Norm threshold for gradient clipping: %E', config.max_gradient_norm)
        logging.info('Optimizer: %s', config.optimizer.type)
        if config.optimizer in ('sgd',):
            logging.info('Momentum factor: %f', config.optimizer.momentum)
        if config.optimizer in ('adam', 'radam', 'lamb',):
            logging.info('Epsilon parameter: %E', config.optimizer.epsilon)
        logging.info('Learning rate: %E', config.optimizer.learning_rate)
        if config.optimizer.warmup_steps == 0:
            logging.info('LR warm-up: (disabled)')
        else:
            logging.info('# of steps for LR warm-up: %d', config.optimizer.warmup_steps)
        if config.optimizer.annealing_steps == 0:
            logging.info('Cosine annealing: (disabled)')
        else:
            logging.info('# of steps for cosine annealing: %d', config.optimizer.annealing_steps)
            logging.info(
                'Step factor for cosine annealing: %d', config.optimizer.annealing_steps_factor)
        if config.initial_model_prefix is not None:
            logging.info('Initial policy encoder snapshot: %s', policy_encoder_snapshot_path)
            logging.info('Initial policy decoder snapshot: %s', policy_decoder_snapshot_path)
            if optimizer_snapshot_path is not None:
                logging.info('Initial optimizer snapshot: %s', optimizer_snapshot_path)
            if scheduler_snapshot_path is not None:
                logging.info('Initial LR scheduler snapshot: %s', scheduler_snapshot_path)
        logging.info('Output prefix: %s', output_prefix)
        if config.snapshot_interval == 0:
            logging.info('Snapshot interval: N/A')
        else:
            logging.info('Snapshot interval: %d', config.snapshot_interval)

    if config.value_model is None:
        value_model = None
    else:
        value_model = load_model(config.value_model, map_location='cpu')
        value_model.to(device=config.device.type, dtype=dtype)
        value_model.requires_grad_(False)
        value_model.eval()

    q_model = load_model(config.q_model, map_location='cpu')
    q_model.to(device=config.device.type, dtype=dtype)
    q_model.requires_grad_(False)
    q_model.eval()

    policy_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        num_layers=config.encoder.num_layers,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        checkpointing=config.checkpointing, device=config.device.type, dtype=dtype)
    policy_decoder = PolicyDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    policy_model = PolicyModel(policy_encoder, policy_decoder)
    policy_model.to(device=config.device.type, dtype=dtype)

    if config.optimizer.type == 'sgd':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            if config.device.type == 'cpu':
                return SGD(
                    model.parameters(), lr=config.optimizer.learning_rate,
                    momentum=config.optimizer.momentum)
            return FusedSGD(
                model.parameters(), lr=config.optimizer.learning_rate,
                momentum=config.optimizer.momentum)
    elif config.optimizer.type == 'adam':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            if config.device.type == 'cpu':
                return Adam(
                    model.parameters(), lr=config.optimizer.learning_rate,
                    eps=config.optimizer.epsilon)
            return FusedAdam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'radam':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            return RAdam(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'lamb':
        def construct_optimizer(model: nn.Module) -> Optimizer:
            return FusedLAMB(
                model.parameters(), lr=config.optimizer.learning_rate, eps=config.optimizer.epsilon)
    else:
        raise NotImplementedError(config.optimizer.type)
    optimizer = construct_optimizer(policy_model)

    if config.optimizer.warmup_steps == 0:
        warmup_scheduler = None
    else:
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer, start_factor=config.optimizer.warmup_start_factor,
            total_iters=config.optimizer.warmup_steps)
    if config.optimizer.annealing_steps == 0:
        annealing_scheduler = None
    else:
        annealing_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, config.optimizer.annealing_steps, config.optimizer.annealing_steps_factor)
    if warmup_scheduler is None and annealing_scheduler is None:
        scheduler = None
    elif warmup_scheduler is not None and annealing_scheduler is not None:
        scheduler = lr_scheduler.SequentialLR(
            optimizer, [warmup_scheduler, annealing_scheduler], [warmup_scheduler.total_iters])
    elif warmup_scheduler is None:
        assert annealing_scheduler is not None
        scheduler = annealing_scheduler
    else:
        assert warmup_scheduler is not None
        assert annealing_scheduler is None
        scheduler = warmup_scheduler

    if config.encoder.load_from is not None:
        assert config.initial_model is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        policy_encoder_state_dict = torch.load(config.encoder.load_from, map_location='cpu')
        policy_encoder.load_state_dict(policy_encoder_state_dict)
        if config.device.type != 'cpu':
            policy_encoder.cuda()

    if config.decoder.load_from is not None:
        assert config.initial_model is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        policy_decoder_state_dict = torch.load(config.decoder.load_from, map_location='cpu')
        policy_decoder.load_state_dict(policy_decoder_state_dict)
        if config.device.type != 'cpu':
            policy_decoder.cuda()

    if config.initial_model is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        model_state_dict = torch.load(config.initial_model, map_location='cpu')
        policy_model.load_state_dict(model_state_dict)
        if config.device.type != 'cpu':
            policy_model.cuda()

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model is None

        policy_encoder_state_dict = torch.load(policy_encoder_snapshot_path, map_location='cpu')
        policy_encoder.load_state_dict(policy_encoder_state_dict)
        if config.device.type != 'cpu':
            policy_encoder.cuda()

        policy_decoder_state_dict = torch.load(policy_decoder_snapshot_path, map_location='cpu')
        policy_decoder.load_state_dict(policy_decoder_state_dict)
        if config.device.type != 'cpu':
            policy_decoder.cuda()

        if optimizer_snapshot_path is not None:
            optimizer_state_dict = torch.load(optimizer_snapshot_path, map_location='cpu')
            optimizer.load_state_dict(optimizer_state_dict)

        if scheduler_snapshot_path is not None and scheduler is not None:
            scheduler_state_dict = torch.load(scheduler_snapshot_path, map_location='cpu')
            scheduler.load_state_dict(scheduler_state_dict)

    if is_multiprocess:
        init_process_group(backend='nccl')
        policy_model = DistributedDataParallel(policy_model)
        policy_model = nn.SyncBatchNorm.convert_sync_batchnorm(policy_model)

    def snapshot_writer(num_samples: Optional[int]=None) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)
        infix = '' if num_samples is None else f'.{num_samples}'

        torch.save(policy_encoder.state_dict(), snapshots_path / f'policy-encoder{infix}.pth')
        torch.save(policy_decoder.state_dict(), snapshots_path / f'policy-decoder{infix}.pth')
        torch.save(optimizer.state_dict(), snapshots_path / f'optimizer{infix}.pth')
        if scheduler is not None:
            torch.save(scheduler.state_dict(), snapshots_path / f'scheduler{infix}.pth')

        policy_model = PolicyModel(encoder=policy_encoder, decoder=policy_decoder)
        state = dump_object(
            policy_model,
            [
                dump_model(
                    policy_model.encoder,
                    [],
                    {
                        'position_encoder': config.encoder.position_encoder,
                        'dimension': config.encoder.dimension,
                        'num_heads': config.encoder.num_heads,
                        'dim_feedforward': config.encoder.dim_feedforward,
                        'num_layers': config.encoder.num_layers,
                        'activation_function': config.encoder.activation_function,
                        'dropout': config.encoder.dropout,
                        'checkpointing': config.checkpointing,
                        'device': config.device.type,
                        'dtype': dtype
                    }
                ),
                dump_model(
                    policy_model.decoder,
                    [],
                    {
                        'dimension': config.encoder.dimension,
                        'dim_feedforward': config.decoder.dim_feedforward,
                        'activation_function': config.decoder.activation_function,
                        'dropout': config.decoder.dropout,
                        'num_layers': config.decoder.num_layers,
                        'device': config.device.type,
                        'dtype': dtype
                    }
                )
            ],
            {}
        )
        torch.save(state, snapshots_path / f'policy{infix}.kanachan')

    tensorboard_path.mkdir(parents=True, exist_ok=True)
    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        _training(
            is_multiprocess=is_multiprocess, world_size=world_size, rank=rank,
            is_main_process=is_main_process, training_data=config.training_data,
            num_workers=config.num_workers, device=config.device.type, dtype=dtype, amp_dtype=amp_dtype,
            value_model=value_model, q_model=q_model, policy_model=policy_model, beta=config.beta,
            weight_clipping=config.weight_clipping, batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm, optimizer=optimizer, scheduler=scheduler,
            snapshot_interval=config.snapshot_interval, num_samples=num_samples,
            snapshot_writer=snapshot_writer, summary_writer=summary_writer)


if __name__ == '__main__':
    _main() # pylint: disable=no-value-for-parameter
    sys.exit(0)
