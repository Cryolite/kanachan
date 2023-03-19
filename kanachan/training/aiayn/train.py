#!/usr/bin/env python3

import re
import datetime
import math
import pathlib
import os
from argparse import ArgumentParser
import logging
import sys
from typing import Optional
import torch
from torch import backends
from torch import nn
from torch.utils.data import DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.utils import tensorboard
from apex import amp
from apex.optimizers import FusedAdam, FusedSGD, FusedLAMB
from kanachan.training.common import initialize_logging, Dataset
from kanachan.training.aiayn.iterator_adaptor import IteratorAdaptor
from kanachan.training.aiayn.encoder import Encoder
from kanachan.training.aiayn.decoder import Decoder
from kanachan.training.aiayn.model import Model


def _training_epoch(
        data_loader: DataLoader, encoder: Encoder, decoder: Decoder, model: Model, optimizer,
        device: str, writer, batch_count: int, snapshot_interval: int, is_main_process: bool,
        is_multiprocess: bool, rank: Optional[int], snapshots_path: pathlib.Path) -> None:
    start_time = datetime.datetime.now()

    loss_function = nn.CrossEntropyLoss()
    training_loss = 0.0

    for annotation in data_loader:
        if device != 'cpu':
            if is_multiprocess:
                local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
                batch_size = len(annotation[0])
                if batch_size % local_world_size != 0:
                    raise RuntimeError(
                        'Batch size must be divisible by the world size.')
                first = (batch_size // local_world_size) * rank
                last = (batch_size // local_world_size) * (rank + 1)
                annotation = tuple(x[first:last].cuda() for x in annotation)
            else:
                annotation = tuple(x.cuda() for x in annotation)

        prediction = model(annotation[:-1])
        loss = loss_function(prediction, annotation[-1])
        if math.isnan(loss.item()):
            raise RuntimeError('Training loss becomes NaN in training.')

        batch_loss = loss.item()
        training_loss += batch_loss

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if is_main_process:
            logging.info('batch = %d, training loss = %E', batch_count, loss.item())
            writer.add_scalar('Training batch loss', batch_loss, batch_count)

        batch_count += 1

        if is_main_process and batch_count % snapshot_interval == 0:
            snapshots_path.mkdir(parents=False, exist_ok=True)
            torch.save(
                encoder.state_dict(),
                snapshots_path / f'encoder.{batch_count}.pth')
            torch.save(
                decoder.state_dict(),
                snapshots_path / f'decoder.{batch_count}.pth')
            torch.save(
                optimizer.state_dict(),
                snapshots_path / f'optimizer.{batch_count}.pth')

    elapsed_time = datetime.datetime.now() - start_time
    logging.info('Pretraining has finished (elapsed time = %f).', elapsed_time)
    logging.info('The final training loss = %E', training_loss / batch_count)

    if is_main_process:
        snapshots_path.mkdir(parents=False, exist_ok=True)
        torch.save(
            encoder.state_dict(), snapshots_path / f'encoder.{batch_count}.pth')
        torch.save(
            decoder.state_dict(), snapshots_path / f'decoder.{batch_count}.pth')
        torch.save(
            optimizer.state_dict(), snapshots_path / f'optimizer.{batch_count}.pth')


def _main() -> None:
    ap = ArgumentParser(
        description='Pre-train the model by imitating human players.')
    ap_data = ap.add_argument_group(title='Data')
    ap_data.add_argument(
        '--training-data', type=pathlib.Path, required=True,
        help='path to training data', metavar='PATH')
    ap_data.add_argument(
        '--num-workers', default=2, type=int,
        help='# of worker processes in data loading (defaults to 2)',
        metavar='NWORKERS')
    ap_device = ap.add_argument_group(title='Device')
    ap_device.add_argument('--device', help='device', metavar='DEV')
    ap_device.add_argument(
        '--dtype', default='float32', choices=('float16','float32'),
        help='floating point type (defaults to `float32`)')
    ap_model = ap.add_argument_group(title='Model')
    ap_model.add_argument(
        '--num-dimensions', default=512, type=int,
        help='# of embedding dimensions (defaults to 512)', metavar='DIM')
    ap_model.add_argument(
        '--num-heads', default=8, type=int, help='# of heads (defaults to 8)',
        metavar='NHEAD')
    ap_model.add_argument(
        '--num-layers', default=6, type=int,
        help='# of layers (defaults to 6)', metavar='NLAYERS')
    ap_model.add_argument(
        '--num-final-dimensions', default=2048, type=int,
        help='# of dimensions of the final feed-forward network (defaults to 2048)',
        metavar='DIM')
    ap_model.add_argument(
        '--initial-encoder', type=pathlib.Path,
        help='path to the initial encoder; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_model.add_argument(
        '--initial-decoder', type=pathlib.Path,
        help='path to the initial decoder; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_training = ap.add_argument_group(title='Training')
    ap_training.add_argument(
        '--batch-size', default=32, type=int, help='batch size (defaults to 32)',
        metavar='N')
    ap_training.add_argument(
        '--optimizer', default='adam', choices=('adam', 'sgd', 'lamb',),
        help='optimizer (defaults to `adam`)')
    ap_training.add_argument(
        '--learning-rate', type=float,
        help='learning rate (defaults to 0.001 for `adam` and `lamb`, 0.1 for `sgd`)',
        metavar='LR')
    ap_training.add_argument(
        '--epsilon', default=1.0e-5, type=float,
        help='epsilon parameter; only meaningful for Adam and LAMB (defaults to 1.0e-5)',
        metavar='EPS')
    ap_training.add_argument(
        '--momentum', default=0.9, type=float,
        help='momentum factor; only meaningful for SGD (defaults to 0.9)',
        metavar='MOMENTUM')
    ap_training.add_argument(
        '--dropout', default=0.1, type=float, help='defaults to 0.1',
        metavar='DROPOUT')
    ap_training.add_argument(
        '--initial-optimizer', type=pathlib.Path,
        help='path to the initial optimizer state; mutually exclusive to `--resume`',
        metavar='PATH')
    ap_output = ap.add_argument_group(title='Output')
    ap_output.add_argument(
        '--output-prefix', type=pathlib.Path, required=True, metavar='PATH')
    ap_output.add_argument('--experiment-name', metavar='NAME')
    ap_output.add_argument(
        '--snapshot-interval', default=10000, type=int,
        help='take a snapshot every specified number of batches',
        metavar='NBATCHES')
    ap_output.add_argument('--resume', action='store_true')

    config = ap.parse_args()

    if 'LOCAL_RANK' in os.environ:
        if os.environ['WORLD_SIZE'] != os.environ['LOCAL_WORLD_SIZE']:
            raise RuntimeError('Multi-node not supported')
        rank = int(os.environ['LOCAL_RANK'])
        is_main_process = rank == 0
        is_multiprocess = int(os.environ['LOCAL_WORLD_SIZE']) >= 2
        torch.cuda.set_device(rank)
    else:
        rank = None
        is_main_process = True
        is_multiprocess = False

    if not config.training_data.exists():
        raise RuntimeError(f'{config.training_data}: does not exist')
    if config.num_workers < 0:
        raise RuntimeError(
            f'{config.num_workers}: invalid number of workers')

    if config.device is not None:
        m = re.search('^(?:cpu)|(?:cuda(\\d+)?)', config.device)
        if m is None:
            raise RuntimeError(f'{config.device}: invalid device')
        if is_multiprocess and m[1] != '':
            raise RuntimeError(
                'Must not specify any device number in multi-process mode')
        device = config.device
    elif backends.cuda.is_built():
        device = 'cuda'
    else:
        device = 'cpu'
    if config.dtype == 'float16':
        dtype = torch.float16
    elif config.dtype == 'float32':
        dtype = torch.float32
    else:
        raise RuntimeError(f'{config.dtype}: invalid value for `--dtype`')

    if config.num_dimensions < 1:
        raise RuntimeError(
            f'{config.num_dimensions}: invalid number of dimensions')
    if config.num_heads < 1:
        raise RuntimeError(f'{config.num_heads}: invalid number of heads')
    if config.num_layers < 1:
        raise RuntimeError(f'{config.num_layers}: invalid number of layers')
    if config.num_final_dimensions < 1:
        raise RuntimeError(
            f'{config.num_final_dimensions}: invalid number of dimensions')
    if config.initial_encoder is not None and not config.initial_encoder.exists():
        raise RuntimeError(f'{config.initial_encoder}: does not exist')
    if config.initial_encoder is not None and config.resume:
        raise RuntimeError('`--initial-encoder` conflicts with `--resume`')
    if config.initial_decoder is not None and not config.initial_decoder.exists():
        raise RuntimeError(f'{config.initial_decoder}: does not exist')
    if config.initial_decoder is not None and config.resume:
        raise RuntimeError('`--initial-decoder` conflicts with `--resume`')

    if config.batch_size < 1:
        raise RuntimeError(f'{config.batch_size}: invalid batch size')
    if config.optimizer == 'sgd':
        if config.momentum == 0.0:
            sparse = True
        else:
            # See https://github.com/pytorch/pytorch/issues/29814
            sparse = False
    else:
        assert(config.optimizer in ('adam', 'lamb',))
        sparse = False
    if config.optimizer in ('adam', 'lamb',):
        if config.learning_rate is None:
            learning_rate = 0.001
        else:
            learning_rate = config.learning_rate
    else:
        assert(config.optimizer == 'sgd')
        if config.learning_rate is None:
            learning_rate = 0.1
        else:
            learning_rate = config.learning_rate
    if config.initial_optimizer is not None and not config.initial_optimizer.exists():
        raise RuntimeError(f'{config.initial_optimizer}: does not exist')
    if config.initial_optimizer is not None and config.resume:
        raise RuntimeError('`--initial-optimizer` conflicts with `--resume`')

    if config.experiment_name is None:
        now = datetime.datetime.now()
        experiment_name = now.strftime('%Y-%m-%d-%H-%M-%S')
    else:
        experiment_name = config.experiment_name

    experiment_path = pathlib.Path(config.output_prefix / experiment_name)
    if rank is None and (experiment_path / 'training.log').exists() and not config.resume:
        raise RuntimeError(
            f'{experiment_path}: already exists; did you mean `--resume`?')
    if rank == 0 and (experiment_path / 'training.0.log').exists() and not config.resume:
        raise RuntimeError(
            f'{experiment_path}: already exists; did you mean `--resume`?')
    snapshots_path = experiment_path / 'snapshots'
    tensorboard_path = config.output_prefix / 'tensorboard' / experiment_name

    if config.snapshot_interval < 1:
        raise RuntimeError(
            f'{config.snapshot_interval}: invalid value for `--snapshot-interval`')

    experiment_path.mkdir(parents=True, exist_ok=True)
    initialize_logging(experiment_path, rank)

    logging.info('Training data: %s', str(config.training_data))
    logging.info('# of workers: %d', config.num_workers)
    logging.info('Device: %s', device)
    if backends.cudnn.is_available():
        logging.info('cuDNN: available')
        backends.cudnn.benchmark = True
    else:
        logging.info('cuDNN: N/A')
    logging.info('dtype: %s', dtype)
    logging.info('# of dimensions: %d', config.num_dimensions)
    logging.info('# of heads: %d', config.num_heads)
    logging.info('# of layers: %d', config.num_layers)
    logging.info('# of final dimensions: %d', config.num_final_dimensions)
    if config.initial_encoder is None and not config.resume:
        logging.info('Initial encoder: (initialized randomly)')
    elif config.initial_encoder is not None:
        logging.info('Initial encoder: %s', str(config.initial_encoder))
    if config.initial_decoder is None and not config.resume:
        logging.info('Initial decoder: (initialized randomly)')
    elif config.initial_decoder is not None:
        logging.info('Initial decoder: %s', str(config.initial_decoder))
    logging.info('Batch size: %d', config.batch_size)
    logging.info('Optimizer: %s', config.optimizer)
    if config.optimizer in ('adam', 'lamb',):
        logging.info('Epsilon parameter: %E', config.epsilon)
    if config.optimizer == 'sgd':
        logging.info('Momentum factor: %f', config.momentum)
    logging.info('Sparse: %s', sparse)
    logging.info('Learning rate: %E', learning_rate)
    logging.info('Dropout: %f', config.dropout)
    if config.initial_optimizer is None and not config.resume:
        logging.info('Initial optimizer: (initialized normally)')
    elif config.initial_optimizer is not None:
        logging.info('Initial optimizer: %s', str(config.initial_optimizer))
    if rank is None:
        logging.info('Process rank: N/A (single process)')
    else:
        logging.info('Process rank: %d', rank)
    logging.info('Snapshot interval: %d', config.snapshot_interval)
    if config.resume:
        logging.info('Resume from %s', str(experiment_path))
    else:
        logging.info('Experiment output: %s', str(experiment_path))

    config = {
        'training_data': config.training_data,
        'num_workers': config.num_workers,
        'device': device,
        'dtype': dtype,
        'num_dimensions': config.num_dimensions,
        'num_heads': config.num_heads,
        'num_layers': config.num_layers,
        'num_final_dimensions': config.num_final_dimensions,
        'initial_encoder': config.initial_encoder,
        'initial_decoder': config.initial_decoder,
        'batch_size': config.batch_size,
        'optimizer': config.optimizer,
        'epsilon': config.epsilon,
        'momentum': config.momentum,
        'sparse': sparse,
        'learning_rate': learning_rate,
        'dropout': config.dropout,
        'initial_optimizer': config.initial_optimizer,
        'experiment_name': experiment_name,
        'experiment_path': experiment_path,
        'snapshots_path': snapshots_path,
        'tensorboard_path': tensorboard_path,
        'snapshot_interval': config.snapshot_interval,
        'resume': config.resume,
    }

    encoder = Encoder(
        config['num_dimensions'], config['num_heads'], config['num_layers'],
        dropout=config['dropout'], sparse=config['sparse'])
    decoder = Decoder(
        config['num_dimensions'], config['num_heads'], config['num_layers'],
        num_final_dimensions=config['num_final_dimensions'],
        dropout=config['dropout'], sparse=config['sparse'])
    model = Model(encoder, decoder)
    if config['device'] == 'cpu':
        model.to(dtype=config['dtype'])
    else:
        model.to(device=config['device'], dtype=config['dtype'])

    if config['optimizer'] == 'adam':
        optimizer = FusedAdam(
            model.parameters(), lr=config['learning_rate'],
            eps=config['epsilon'])
    elif config['optimizer'] == 'sgd':
        optimizer = FusedSGD(
            model.parameters(), lr=config['learning_rate'],
            momentum=config['momentum'])
    else:
        assert(config['optimizer'] == 'lamb')
        optimizer = FusedLAMB(
            model.parameters(), lr=config['learning_rate'],
            eps=config['epsilon'])

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

    if is_multiprocess:
        init_process_group(backend='nccl')
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank)

    batch_count = 0
    if config['resume']:
        if not config['snapshots_path'].exists():
            raise RuntimeError(f'{config["snapshots_path"]}: does not exist')
        for child in os.listdir(config['snapshots_path']):
            m = re.search('^encoder\\.(\\d+)\\.pth$', child)
            if m is None:
                continue
            if int(m[1]) > batch_count:
                batch_count = int(m[1])
        latest_encoder_snapshot_path \
            = config['snapshots_path'] / f'encoder.{batch_count}.pth'
        encoder.load_state_dict(torch.load(latest_encoder_snapshot_path))
        latest_decoder_snapshot_path \
            = config['snapshots_path'] / f'decoder.{batch_count}.pth'
        decoder.load_state_dict(torch.load(latest_decoder_snapshot_path))
        latest_optimizer_snapshot_path \
            = config['snapshots_path'] / f'optimizer.{batch_count}.pth'
        optimizer.load_state_dict(torch.load(latest_optimizer_snapshot_path))

    if config['initial_encoder'] is not None:
        encoder.load_state_dict(torch.load(config['initial_encoder']))
    if config['initial_decoder'] is not None:
        decoder.load_state_dict(torch.load(config['initial_decoder']))
    if config['initial_optimizer'] is not None:
        optimizer.load_state_dict(torch.load(config['initial_optimizer']))

    with tensorboard.SummaryWriter(log_dir=config['tensorboard_path']) as writer:
        config['tensorboard_path'].mkdir(parents=True, exist_ok=True)
        if is_main_process and not config['resume']:
            (config['experiment_path'] / 'tensorboard').symlink_to(
                f'../tensorboard/{config["experiment_name"]}',
                target_is_directory=True)

        # Prepare the data loader. Note that this data loader must iterate
        # the data set only once.
        iterator_adaptor = lambda fp: IteratorAdaptor(
            fp, config['num_dimensions'], config['dtype'])
        dataset = Dataset(config['training_data'], iterator_adaptor)
        data_loader = DataLoader(
            dataset, batch_size=config['batch_size'],
            num_workers=config['num_workers'], pin_memory=True,
            drop_last=is_multiprocess)

        _training_epoch(
            data_loader, encoder, decoder, model, optimizer, config['device'], writer, batch_count,
            config['snapshot_interval'], is_main_process, is_multiprocess, rank,
            config['snapshots_path'])


if __name__ == '__main__':
    _main()
    sys.exit(0)
