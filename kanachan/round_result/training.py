#!/usr/bin/env python3

import re
import datetime
import pathlib
import os
from argparse import ArgumentParser
import logging
import sys
import torch
from torch import backends
from torch import nn
from torch.cuda import amp
from torch.utils import tensorboard
from kanachan import common
from kanachan import option
from kanachan.iterator_adaptor_base import IteratorAdaptor


class Decoder(common.DecoderBase):
    def __init__(
            self, dimension, num_heads, num_layers, sparse,
            target_num_classes) -> None:
        super(Decoder, self).__init__(dimension, num_heads, num_layers, sparse)
        self.__is_regression = target_num_classes is None
        if self.__is_regression:
            # Regression
            self.__full_connection = nn.Linear(dimension, 1)
        else:
            # Classification
            self.__activation = nn.ReLU()
            self.__full_connection = nn.Linear(dimension, target_num_classes)

    def forward(self, encode, action):
        decode = super(Decoder, self).forward(encode, action)
        if self.__is_regression:
            # Regression
            decode = torch.squeeze(decode, dim=1)
            prediction = self.__full_connection(decode)
            prediction = torch.squeeze(prediction, dim=1)
        else:
            # Classification
            decode = self.__activation(decode)
            decode = torch.squeeze(decode, dim=1)
            prediction = self.__full_connection(decode)
        return prediction


class Model(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(Model, self).__init__()
        self.__encoder = encoder
        self.__decoder = decoder

    def forward(self, x):
        encode = self.__encoder(x[:-1])
        prediction = self.__decoder(encode, x[-1])
        return prediction


if __name__ == '__main__':
    config = option.parse_argument(
        description='Train a model predicting the round result by action.')

    iterator_adaptor = lambda fp: IteratorAdaptor(
        fp, config['dimension'], config['target_index'],
        config['target_num_classes'], config['dtype'])

    encoder = common.Encoder(
        config['dimension'], config['num_heads'], config['num_layers'],
        config['sparse'])
    decoder = Decoder(
        config['dimension'], config['num_heads'], config['num_layers'],
        config['sparse'], config['target_num_classes'])
    model = Model(encoder, decoder)
    model.to(device=config['device'], dtype=config['dtype'])

    if config['target_num_classes'] is None:
        # Regression
        loss_function = nn.MSELoss()
    else:
        # Classification
        loss_function = nn.CrossEntropyLoss()

    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['learning_rate'])
    else:
        assert(config['optimizer'] == 'sgd')
        optimizer = torch.optim.SGD(
            model.parameters(), lr=config['learning_rate'])

    grad_scaler = amp.GradScaler()

    epoch = 0
    if config['resume']:
        if not config['snapshots_path'].exists():
            raise RuntimeError(f'{config["snapshots_path"]}: does not exist.')
        for child in os.listdir(config['snapshots_path']):
            m = re.search('^encoder\\.(\\d+)\\.pth$', child)
            if m is None:
                continue
            if epoch is None or int(m[1]) > epoch:
                epoch = int(m[1])
        latest_encoder_snapshot_path = config['snapshots_path'] / f'encoder.{epoch}.pth'
        latest_decoder_snapshot_path = config['snapshots_path'] / f'decoder.{epoch}.pth'
        if not latest_decoder_snapshot_path.exists():
            raise RuntimeError(
                f'{latest_decoder_snapshot_path}: does not exist.')
        encoder.load_state_dict(torch.load(latest_encoder_snapshot_path))
        decoder.load_state_dict(torch.load(latest_decoder_snapshot_path))

    if config['initial_encoder'] is not None:
        encoder.load_state_dict(torch.load(config['initial_encoder']))

    if config['initial_decoder'] is not None:
        decoder.load_state_dict(torch.load(config['initial_decoder']))

    with tensorboard.SummaryWriter(log_dir=config['tensorboard_path']) as writer:
        config['tensorboard_path'].mkdir(parents=True, exist_ok=True)
        if not config['resume']:
            (config['experiment_path'] / 'tensorboard').symlink_to(
                f'../tensorboard/{config["experiment_name"]}',
                target_is_directory=True)

        if config['validation_data'] is not None:
            validation_loss = common.get_validation_loss(
                config['validation_data'], iterator_adaptor,
                config['num_workers'], model, loss_function,
                config['batch_size'], config['device'])
            logging.info(
                f'epoch = {epoch}, validation loss = {validation_loss}')
            writer.add_scalar('Validation epoch loss', validation_loss, epoch)

        epoch += 1

        for i in range(epoch, config['max_epoch'] + 1):
            validation_loss = common.training_epoch(
                config['training_data'], config['validation_data'],
                iterator_adaptor, config['num_workers'], model, loss_function,
                optimizer, grad_scaler, config['batch_size'], config['device'],
                config['dtype'], writer, i)
            config['snapshots_path'].mkdir(parents=False, exist_ok=True)
            torch.save(encoder.state_dict(), config['snapshots_path'] / f'encoder.{i}.pth')
            torch.save(decoder.state_dict(), config['snapshots_path'] / f'decoder.{i}.pth')

            if validation_loss is not None:
                best_tsv_path = config['experiment_path'] / 'best.tsv'
                if not best_tsv_path.exists():
                    with open(best_tsv_path, 'w') as f:
                        f.write(f'{i}\t{validation_loss}\n')
                    best_epoch = i
                else:
                    with open(best_tsv_path, 'r') as f:
                        best_epoch, best_loss = f.read().rstrip('\n').split('\t')
                    best_epoch, best_loss = int(best_epoch), float(best_loss)
                    if validation_loss < best_loss:
                        with open(best_tsv_path, 'w') as f:
                            f.write(f'{i}\t{validation_loss}\n')
                        best_epoch = i
                (config['experiment_path'] / 'encoder.best.pth').unlink(missing_ok=True)
                (config['experiment_path'] / 'decoder.best.pth').unlink(missing_ok=True)
                (config['experiment_path'] / 'encoder.best.pth').symlink_to(
                    f'snapshots/encoder.{best_epoch}.pth', target_is_directory=False)
                (config['experiment_path'] / 'decoder.best.pth').symlink_to(
                    f'snapshots/decoder.{best_epoch}.pth', target_is_directory=False)

    sys.exit(0)
