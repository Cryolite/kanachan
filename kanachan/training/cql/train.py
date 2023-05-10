#!/usr/bin/env python3

import re
import datetime
from pathlib import Path
import os
import logging
import sys
from typing import Optional, Callable
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
from torch.distributed import init_process_group, all_reduce, barrier
from torch.utils.tensorboard.writer import SummaryWriter
from apex.optimizers import FusedAdam, FusedSGD, FusedLAMB
from kanachan.training.constants import NUM_TYPES_OF_SPARSE_FEATURES, MAX_NUM_ACTION_CANDIDATES
from kanachan.training.common import Dataset
import kanachan.training.cql.config # pylint: disable=unused-import
from kanachan.training.iql.iterator_adaptor import IteratorAdaptor
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.q_model import QDecoder, QModel
from kanachan.model_loader import load_model, dump_object, dump_model


SnapshotWriter = Callable[[Optional[int]], None]


def _training(
        *, is_multiprocess: bool, world_size: Optional[int], rank: Optional[int],
        is_main_process: bool, training_data: Path, num_workers: int, device: torch.device,
        dtype: torch.dtype, amp_dtype: torch.dtype, q_source_model: QModel, q_target_model: QModel,
        reward_plugin: Path, discount_factor: float, policy_model: nn.Module,
        policy_model_requires_softmax: bool, alpha: float, batch_size: int,
        gradient_accumulation_steps: int, max_gradient_norm: float, q_optimizer: Optimizer,
        scheduler #: lr_scheduler.LRScheduler
        , target_update_interval: int, target_update_rate: float,
        snapshot_interval: int, num_samples: int, summary_writer: SummaryWriter,
        snapshot_writer: SnapshotWriter) -> None:
    start_time = datetime.datetime.now()

    # Load the reward plugin.
    with open(reward_plugin, encoding='UTF-8') as file_pointer:
        exec(file_pointer.read(), globals()) # pylint: disable=exec-used

    # Prepare the training data loader. Note that this data loader must iterate
    # the training data set only once.
    def iterator_adaptor(path: Path) -> IteratorAdaptor:
        return IteratorAdaptor(path, get_reward) # type: ignore pylint: disable=undefined-variable
    dataset = Dataset(training_data, iterator_adaptor)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(num_workers >= 1),
        drop_last=is_multiprocess)

    last_snapshot = None
    if snapshot_interval > 0:
        last_snapshot = num_samples

    num_consumed_samples = 0
    batch_count = 0

    grad_scaler = None
    if device != 'cpu':
        grad_scaler = GradScaler()

    for annotation in data_loader:
        if num_consumed_samples < num_samples:
            num_consumed_samples += batch_size
            continue

        if is_multiprocess:
            barrier()

        if is_multiprocess:
            assert world_size is not None
            assert rank is not None
            assert batch_size % world_size == 0
            first = (batch_size // world_size) * rank
            last = (batch_size // world_size) * (rank + 1)
            annotation = tuple(x[first:last] for x in annotation)

        if device != 'cpu':
            annotation = tuple(x.cuda() for x in annotation)

        local_batch_size = annotation[0].size(0)
        world_batch_size = batch_size

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device != 'cpu' and dtype != amp_dtype)):
            q: torch.Tensor = q_source_model(*(annotation[:4]))
            assert q.dim() == 2
            assert q.size(0) == local_batch_size
            assert q.size(1) == MAX_NUM_ACTION_CANDIDATES
            q_sa = q[torch.arange(local_batch_size), annotation[4]]

            _q = q_sa.detach().clone().mean()
            if is_multiprocess:
                all_reduce(_q)
                _q /= world_size
            q_to_display = _q.item()

            policy: torch.Tensor = policy_model(*(annotation[:4]))
            assert policy.dim() == 2
            assert policy.size(0) == local_batch_size
            assert policy.size(1) == MAX_NUM_ACTION_CANDIDATES
            if policy_model_requires_softmax:
                policy = nn.Softmax(dim=1)(policy)
            policy = policy[torch.arange(local_batch_size), annotation[4]]

            q_next: torch.Tensor = q_source_model(*(annotation[5:9]))
            assert q_next.dim() == 2
            assert q_next.size(0) == local_batch_size
            assert q_next.size(1) == MAX_NUM_ACTION_CANDIDATES
            q_max = torch.max(q_next, dim=1).values
            is_terminal_state = annotation[5][:, 0] == NUM_TYPES_OF_SPARSE_FEATURES
            q_max = torch.where(is_terminal_state, torch.zeros_like(q_max), q_max)

            # Compute the regularization term.
            log_z = torch.logsumexp(q, dim=1)
            regularizer = torch.mean(log_z - policy * q_sa)

            # Compute the TD error.
            td_error = (annotation[9] + discount_factor * q_max - q_sa) ** 2.0
            td_error = torch.mean(td_error) / 2.0

            loss = alpha * regularizer + td_error

            _loss = loss.detach().clone()
            if is_multiprocess:
                all_reduce(_loss)
                _loss /= world_size
            loss_to_display = _loss.item()

            loss /= gradient_accumulation_steps
            if grad_scaler is None:
                loss.backward()
            else:
                grad_scaler.scale(loss).backward()

        num_samples += world_batch_size
        num_consumed_samples += world_batch_size
        batch_count += 1

        if batch_count % gradient_accumulation_steps == 0:
            if grad_scaler is not None:
                grad_scaler.unscale_(q_optimizer)
            gradient = nn.utils.parameters_to_vector(q_source_model.parameters())
            gradient_norm: float = torch.linalg.vector_norm(gradient).item()
            nn.utils.clip_grad_norm_(
                q_source_model.parameters(), max_gradient_norm, error_if_nonfinite=False)
            if grad_scaler is None:
                q_optimizer.step()
            else:
                grad_scaler.step(q_optimizer)
                grad_scaler.update()
            scheduler.step()
            q_optimizer.zero_grad()

            if batch_count % (gradient_accumulation_steps * target_update_interval) == 0:
                with torch.no_grad():
                    param_source = nn.utils.parameters_to_vector(q_source_model.parameters())
                    param_target = nn.utils.parameters_to_vector(q_target_model.parameters())
                    param_target *= (1.0 - target_update_rate)
                    param_target += target_update_rate * param_source
                    nn.utils.vector_to_parameters(param_target, q_target_model.parameters())

            if is_main_process:
                logging.info(
                    'sample = %s, loss = %s, gradient norm = %s',
                    num_samples, loss_to_display, gradient_norm)
                summary_writer.add_scalar('Q', q_to_display, num_samples)
                summary_writer.add_scalar('Loss', loss_to_display, num_samples)
                summary_writer.add_scalar('Gradient Norm', gradient_norm, num_samples)
                summary_writer.add_scalar('LR', scheduler.get_last_lr()[0], num_samples)
        else:
            if is_main_process:
                logging.info('sample = %s, loss = %s', num_samples, loss_to_display)
                summary_writer.add_scalar('Q', q_to_display, num_samples)
                summary_writer.add_scalar('Loss', loss_to_display, num_samples)

        if is_main_process and last_snapshot is not None and num_samples - last_snapshot >= snapshot_interval:
            snapshot_writer(num_samples)
            last_snapshot = num_samples

    if is_multiprocess:
        barrier()

    elapsed_time = datetime.datetime.now() - start_time

    if is_main_process:
        logging.info('A training has finished (elapsed time = %s).', elapsed_time)
        snapshot_writer()


@hydra.main(version_base=None, config_name='config')
def _main(config: DictConfig) -> None:
    if 'LOCAL_RANK' in os.environ:
        if os.environ['WORLD_SIZE'] != os.environ['LOCAL_WORLD_SIZE']:
            raise RuntimeError('Multi-node not supported.')
        world_size = int(os.environ['LOCAL_WORLD_SIZE'])
        rank = int(os.environ['LOCAL_RANK'])
        is_multiprocess = True
        is_main_process = (rank == 0)
    else:
        world_size = None
        rank = None
        is_multiprocess = False
        is_main_process = True

    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: Does not exist.')
    if not config.training_data.is_file():
        raise RuntimeError(f'{config.training_data}: Not a file.')

    if config.num_workers is None:
        if config.device.type == 'cpu':
            config.num_workers = 0
        else:
            config.num_workers = 2
    if config.num_workers < 0:
        raise RuntimeError(f'{config.num_workers}: `num_workers` must be a non-negative integer.')

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

    if config.device.dtype not in ('float64', 'double', 'float32', 'float', 'float16', 'half'):
        raise RuntimeError(f'{config.device.dtype}: An invalid dtype.')
    dtype = {
        'float64': torch.float64, 'double': torch.float64,
        'float32': torch.float32, 'float': torch.float32,
        'float16': torch.float16, 'half': torch.float16,
        'bfloat16': torch.bfloat16
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
        'float16': torch.float16, 'half': torch.float16,
        'bfloat16': torch.bfloat16
    }[config.device.amp_dtype]
    if amp_dtype == torch.float64 and dtype in (torch.float32, torch.float16):
        raise RuntimeError(
            f'An invalid combination of `dtype` (`{dtype}`) and `amp_dtype` (`{amp_dtype}`).')
    if amp_dtype == torch.float32 and dtype == torch.float16:
        raise RuntimeError(
            f'An invalid combination of `dtype` (`{dtype}`) and `amp_dtype` (`{amp_dtype}`).')

    if backends.cudnn.is_available():
        backends.cudnn.benchmark = True

    if config.encoder.position_encoder not in ('positional_encoding', 'position_embedding'):
        raise RuntimeError(f'{config.encoder.position_encoder}: An invalid position encoder.')

    if config.encoder.dimension < 1:
        raise RuntimeError(
            f'{config.encoder.dimension}: '
            '`encoder.dimension` must be an integer greater than 0.')

    if config.encoder.num_heads < 1:
        raise RuntimeError(
            f'{config.encoder.num_heads}: `encoder.num_heads` must be an integer greater than 0.')

    if config.encoder.dim_feedforward is None:
        config.encoder.dim_feedforward = 4 * config.encoder.dimension
    if config.encoder.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.encoder.dim_feedforward}: '
            '`encoder.dim_feedforward` must be an integer greater than 0.')

    if config.encoder.activation_function not in ('relu', 'gelu'):
        raise RuntimeError(
            f'{config.encoder.activation_function}: '
            'An invalid activation function for the encoder.')

    if config.encoder.dropout < 0.0 or 1.0 <= config.encoder.dropout:
        raise RuntimeError(
            f'{config.encoder.dropout}: '
            '`encoder.dropout` must be a real value within the range [0.0, 1.0).')

    if config.encoder.num_layers < 1:
        raise RuntimeError(
            f'{config.encoder.num_layers}: `encoder.num_layers` must be an integer greater than 0.')

    if config.encoder.load_from is not None:
        if not config.encoder.load_from.exists():
            raise RuntimeError(f'{config.encoder.load_from}: Does not exist.')
        if not config.encoder.load_from.is_file():
            raise RuntimeError(f'{config.encoder.load_from}: Not a file.')

    if config.decoder.dim_feedforward is None:
        config.decoder.dim_feedforward = config.encoder.dim_feedforward
    if config.decoder.dim_feedforward < 1:
        raise RuntimeError(
            f'{config.decoder.dim_feedforward}: '
            '`decoder.dim_feedforward` must be an integer greater than 0.')

    if config.decoder.activation_function not in ('relu', 'gelu'):
        raise RuntimeError(
            f'{config.decoder.activation_function}: '
            'An invalid activation function for the decoder.')

    if config.decoder.dropout < 0.0 or 1.0 <= config.decoder.dropout:
        raise RuntimeError(
            f'{config.decoder.dropout}: '
            '`decoder.dropout` must be a real value within the range [0.0, 1.0).')

    if config.decoder.num_layers < 1:
        raise RuntimeError(
            f'{config.decoder.num_layers}: `decoder.num_layers` must be an integer greater than 0.')

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
                    '^(?:q-(?:source|target)-(?:encoder|decoder)|q-optimizer|lr-scheduler)(?:\\.(\\d+))?\\.pth$',
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

        q_source_encoder_snapshot_path: Path = config.initial_model_prefix / f'q-source-encoder{infix}.pth'
        if not q_source_encoder_snapshot_path.exists():
            raise RuntimeError(f'{q_source_encoder_snapshot_path}: Does not exist.')
        if not q_source_encoder_snapshot_path.is_file():
            raise RuntimeError(f'{q_source_encoder_snapshot_path}: Not a file.')

        q_source_decoder_snapshot_path: Path = config.initial_model_prefix / f'q-source-decoder{infix}.pth'
        if not q_source_decoder_snapshot_path.exists():
            raise RuntimeError(f'{q_source_decoder_snapshot_path}: Does not exist.')
        if not q_source_decoder_snapshot_path.is_file():
            raise RuntimeError(f'{q_source_decoder_snapshot_path}: Not a file.')

        q_target_encoder_snapshot_path: Path = config.initial_model_prefix / f'q-target-encoder{infix}.pth'
        if not q_target_encoder_snapshot_path.exists():
            raise RuntimeError(f'{q_target_encoder_snapshot_path}: Does not exist.')
        if not q_target_encoder_snapshot_path.is_file():
            raise RuntimeError(f'{q_target_encoder_snapshot_path}: Not a file.')

        q_target_decoder_snapshot_path: Path = config.initial_model_prefix / f'q-target-decoder{infix}.pth'
        if not q_target_decoder_snapshot_path.exists():
            raise RuntimeError(f'{q_target_decoder_snapshot_path}: Does not exist.')
        if not q_target_decoder_snapshot_path.is_file():
            raise RuntimeError(f'{q_target_decoder_snapshot_path}: Not a file.')

        q_optimizer_snapshot_path: Path = config.initial_model_prefix / f'q-optimizer{infix}.pth'
        if not q_optimizer_snapshot_path.is_file() or config.optimizer.initialize:
            q_optimizer_snapshot_path = None

        scheduler_snapshot_path: Path = config.initial_model_prefix / f'lr-scheduler{infix}.pth'
        if q_optimizer_snapshot_path is None:
            scheduler_snapshot_path = None
        else:
            if not scheduler_snapshot_path.exists():
                raise RuntimeError(f'{scheduler_snapshot_path}: Does not exist.')
            if not scheduler_snapshot_path.is_file():
                raise RuntimeError(f'{scheduler_snapshot_path}: Not a file.')

    if not config.reward_plugin.exists():
        raise RuntimeError(f'{config.reward_plugin}: Does not exist.')
    if not config.reward_plugin.is_file():
        raise RuntimeError(f'{config.reward_plugin}: Not a file.')

    if config.discount_factor <= 0.0 or 1.0 < config.discount_factor:
        raise RuntimeError(f'{config.discount_factor}: An invalid value for `discount_factor`.')

    if not config.policy_model.exists():
        raise RuntimeError(f'{config.policy_model}: Does not exist.')
    if not config.policy_model.is_file():
        raise RuntimeError(f'{config.policy_model}: Not a file')

    if config.alpha < 0.0:
        raise RuntimeError(f'{config.alpha}: `alpha` must be a non-negative real value.')

    if config.batch_size < 1:
        raise RuntimeError(f'{config.batch_size}: `batch_size` must be an integer greater than 0.')
    if config.batch_size % world_size != 0:
        raise RuntimeError(f'`batch_size` must be divisible by the world size ({world_size}).')

    if config.gradient_accumulation_steps < 1:
        raise RuntimeError(
            f'{config.gradient_accumulation_steps}: '
            '`gradient_accumulation_steps` must be an integer greater than 0.')

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
            if config.optimizer.type in ('adam', 'radam', 'mtadam'):
                config.optimizer.epsilon = 1.0e-8
            elif config.optimizer in ('lamb',):
                config.optimizer.epsilon = 1.0e-6
            else:
                raise NotImplementedError(config.optimizer.type)
    if config.optimizer.epsilon is not None and config.optimizer.epsilon <= 0.0:
        raise RuntimeError(
            f'{config.optimizer.epsilon}: `optimizer.epsilon` must be a non-negative real value.')

    if config.optimizer.learning_rate <= 0.0:
        raise RuntimeError(
            f'{config.optimizer.learning_rate}: '
            '`optimizer.learning_rate` must be a positive real value.')

    if config.optimizer.warmup_start_factor <= 0.0 or 1.0 <= config.optimizer.warmup_start_factor:
        raise RuntimeError(
            f'{config.optimizer.warmup_start_factor}: '
            '`optimizer.warmup_start_factor` must be a real value within the range (0,0, 1.0)')

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

    if config.target_update_interval <= 0:
        raise RuntimeError(
            f'{config.target_update_interval}: '
            '`target_update_interval` must be a positive integer.')

    if config.target_update_rate <= 0.0 or 1.0 < config.target_update_rate:
        raise RuntimeError(
            f'{config.target_update_rate}: '
            '`target_update_rate` must be a real value within the range (0.0, 1.0].')

    if config.snapshot_interval < 0:
        raise RuntimeError(
            f'{config.snapshot_interval}: `snapshot_interval` must be a non-negative integer.')

    output_prefix = Path(HydraConfig.get().runtime.output_dir)

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
        logging.info('Reward plugin: %s', config.reward_plugin)
        logging.info('Discount factor: %f', config.discount_factor)
        logging.info('Policy model: %s', config.policy_model)
        logging.info('Policy model requires softmax: %s', config.policy_model_requires_softmax)
        logging.info('Alpha: %f', config.alpha)
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
        if config.optimizer in ('adam', 'radam', 'mtadam', 'lamb'):
            logging.info('Epsilon parameter: %E', config.optimizer.epsilon)
        logging.info('Learning rate: %E', config.optimizer.learning_rate)
        if config.optimizer.warmup_steps == 0:
            logging.info('LR warm-up: (disabled)')
        else:
            logging.info('Warm-up start factor: %E', config.optimizer.warmup_start_factor)
            logging.info('# of steps for LR warm-up: %d', config.optimizer.warmup_steps)
        if config.optimizer.annealing_steps is None:
            logging.info('Annealing: (disabled)')
        else:
            logging.info('# of steps for Annealing: %d', config.optimizer.annealing_steps)
            logging.info('Step factor for Annealing: %d', config.optimizer.annealing_steps_factor)
        logging.info('Target update interval: %d', config.target_update_interval)
        logging.info('Target update rate: %f', config.target_update_rate)
        if config.initial_model_prefix is not None:
            logging.info('Initial Q source encoder snapshot: %s', q_source_encoder_snapshot_path)
            logging.info('Initial Q source decoder snapshot: %s', q_source_decoder_snapshot_path)
            logging.info('Initial Q target encoder snapshot: %s', q_target_encoder_snapshot_path)
            logging.info('Initial Q target decoder snapshot: %s', q_target_decoder_snapshot_path)
            if q_optimizer_snapshot_path is not None:
                logging.info('Initial optimizer snapshot: %s', q_optimizer_snapshot_path)
                logging.info('Initial LR scheduler snapshot: %s', scheduler_snapshot_path)
        logging.info('Output prefix: %s', output_prefix)
        if config.snapshot_interval == 0:
            logging.info('Snapshot interval: N/A')
        else:
            logging.info('Snapshot interval: %d', config.snapshot_interval)

    q_source_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        num_layers=config.encoder.num_layers, checkpointing=config.checkpointing,
        device=config.device.type, dtype=dtype)
    q_source_decoder = QDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    q_source_model = QModel(q_source_encoder, q_source_decoder)
    q_source_model.to(device=config.device.type, dtype=dtype)

    q_target_encoder = Encoder(
        position_encoder=config.encoder.position_encoder, dimension=config.encoder.dimension,
        num_heads=config.encoder.num_heads, dim_feedforward=config.encoder.dim_feedforward,
        activation_function=config.encoder.activation_function, dropout=config.encoder.dropout,
        num_layers=config.encoder.num_layers, checkpointing=config.checkpointing,
        device=config.device.type, dtype=dtype)
    q_target_decoder = QDecoder(
        dimension=config.encoder.dimension, dim_feedforward=config.decoder.dim_feedforward,
        activation_function=config.decoder.activation_function, dropout=config.decoder.dropout,
        num_layers=config.decoder.num_layers, device=config.device.type, dtype=dtype)
    q_target_model = QModel(q_target_encoder, q_target_decoder)
    q_target_model.requires_grad_(False)
    q_target_model.to(device=config.device.type, dtype=dtype)

    if config.optimizer.type == 'sgd':
        if config.device.type == 'cpu':
            q_optimizer = SGD(
                q_source_model.parameters(), lr=config.optimizer.learning_rate,
                momentum=config.optimizer.momentum)
        else:
            q_optimizer = FusedSGD(
                q_source_model.parameters(), lr=config.optimizer.learning_rate,
                momentum=config.optimizer.momentum)
    elif config.optimizer.type == 'adam':
        if config.device.type == 'cpu':
            q_optimizer = Adam(
                q_source_model.parameters(), lr=config.optimizer.learning_rate,
                eps=config.optimizer.epsilon)
        else:
            q_optimizer = FusedAdam(
                q_source_model.parameters(), lr=config.optimizer.learning_rate,
                eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'radam':
        q_optimizer = RAdam(
            q_source_model.parameters(), lr=config.optimizer.learning_rate,
            eps=config.optimizer.epsilon)
    elif config.optimizer.type == 'lamb':
        q_optimizer = FusedLAMB(
            q_source_model.parameters(), lr=config.optimizer.learning_rate,
            eps=config.optimizer.epsilon)
    else:
        raise NotImplementedError(config.optimizer.type)

    warmup_scheduler = lr_scheduler.LinearLR(
        q_optimizer, start_factor=config.optimizer.warmup_start_factor,
        total_iters=config.optimizer.warmup_steps)
    annealing_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        q_optimizer, config.optimizer.annealing_steps, config.optimizer.annealing_steps_factor)
    scheduler = lr_scheduler.SequentialLR(
        q_optimizer, [warmup_scheduler, annealing_scheduler], [warmup_scheduler.total_iters])

    if config.encoder.load_from is not None:
        assert config.initial_model is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        encoder_state_dict = torch.load(config.encoder.load_from, map_location='cpu')
        q_source_encoder.load_state_dict(encoder_state_dict)
        q_target_encoder.load_state_dict(encoder_state_dict)
        if config.device.type != 'cpu':
            q_source_encoder.cuda()
            q_target_encoder.cuda()

    if config.decoder.load_from is not None:
        assert config.initial_model is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        decoder_state_dict = torch.load(config.decoder.load_from, map_location='cpu')
        q_source_decoder.load_state_dict(decoder_state_dict)
        q_target_decoder.load_state_dict(decoder_state_dict)
        if config.device.type != 'cpu':
            q_source_decoder.cuda()
            q_target_decoder.cuda()

    if config.initial_model is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model_prefix is None
        assert config.initial_model_index is None

        model_state_dict = torch.load(config.initial_model, map_location='cpu')
        q_source_model.load_state_dict(model_state_dict)
        q_target_model.load_state_dict(model_state_dict)
        if config.device.type != 'cpu':
            q_source_model.cuda()
            q_target_model.cuda()

    if config.initial_model_prefix is not None:
        assert config.encoder.load_from is None
        assert config.decoder.load_from is None
        assert config.initial_model is None

        q_source_encoder_state_dict = torch.load(q_source_encoder_snapshot_path, map_location='cpu')
        q_source_encoder.load_state_dict(q_source_encoder_state_dict)
        q_source_decoder_state_dict = torch.load(q_source_decoder_snapshot_path, map_location='cpu')
        q_source_decoder.load_state_dict(q_source_decoder_state_dict)
        if config.device.type != 'cpu':
            q_source_model.cuda()

        q_target_encoder_state_dict = torch.load(q_target_encoder_snapshot_path, map_location='cpu')
        q_target_encoder.load_state_dict(q_target_encoder_state_dict)
        q_target_decoder_state_dict = torch.load(q_target_decoder_snapshot_path, map_location='cpu')
        q_target_decoder.load_state_dict(q_target_decoder_state_dict)
        if config.device.type != 'cpu':
            q_target_model.cuda()

        if q_optimizer_snapshot_path is not None:
            q_optimizer.load_state_dict(
                torch.load(q_optimizer_snapshot_path, map_location='cpu'))
            scheduler.load_state_dict(
                torch.load(scheduler_snapshot_path, map_location='cpu'))

    policy_model = load_model(config.policy_model, map_location='cpu')
    if config.device.type != 'cpu':
        policy_model.cuda()

    if is_multiprocess:
        init_process_group(backend='nccl')
        q_source_model = DistributedDataParallel(q_source_model)
        q_source_model = nn.SyncBatchNorm.convert_sync_batchnorm(q_source_model)

    tensorboard_path = output_prefix / 'tensorboard'
    tensorboard_path.mkdir(parents=True, exist_ok=True)

    snapshots_path = output_prefix / 'snapshots'

    def snapshot_writer(num_samples: Optional[int]=None) -> None:
        snapshots_path.mkdir(parents=True, exist_ok=True)

        infix = '' if num_samples is None else f'.{num_samples}'

        torch.save(q_source_encoder.state_dict(), snapshots_path / f'q-source-encoder{infix}.pth')
        torch.save(q_source_decoder.state_dict(), snapshots_path / f'q-source-decoder{infix}.pth')
        torch.save(q_target_encoder.state_dict(), snapshots_path / f'q-target-encoder{infix}.pth')
        torch.save(q_target_decoder.state_dict(), snapshots_path / f'q-target-decoder{infix}.pth')
        torch.save(q_optimizer.state_dict(), snapshots_path / f'q-optimizer{infix}.pth')
        torch.save(scheduler.state_dict(), snapshots_path / f'lr-scheduler{infix}.pth')

        q_model = QModel(q_target_encoder, q_target_decoder)
        q_model_state = dump_object(
            q_model,
            [
                dump_model(
                    q_target_encoder,
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
                    q_target_decoder,
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
        torch.save(q_model_state, snapshots_path / f'q-model{infix}.kanachan')

    with SummaryWriter(log_dir=tensorboard_path) as summary_writer:
        _training(
            is_multiprocess=is_multiprocess, world_size=world_size, rank=rank,
            is_main_process=is_main_process, training_data=config.training_data,
            num_workers=config.num_workers, device=config.device.type, dtype=dtype,
            amp_dtype=amp_dtype, q_source_model=q_source_model, q_target_model=q_target_model,
            reward_plugin=config.reward_plugin, discount_factor=config.discount_factor,
            policy_model=policy_model,
            policy_model_requires_softmax=config.policy_model_requires_softmax, alpha=config.alpha,
            batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            max_gradient_norm=config.max_gradient_norm, q_optimizer=q_optimizer,
            scheduler=scheduler, target_update_interval=config.target_update_interval,
            target_update_rate=config.target_update_rate,
            snapshot_interval=config.snapshot_interval, num_samples=num_samples,
            summary_writer=summary_writer,
            snapshot_writer=snapshot_writer) # pylint: disable=missing-kwoa


if __name__ == '__main__':
    _main() # pylint: disable=no-value-for-parameter
    sys.exit(0)
