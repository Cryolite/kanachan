#!/usr/bin/env python3

from typing import Optional
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.qv_model import QVDecoder, QVModel


class QModel(nn.Module):
    def __init__(
            self, *,
            position_encoder: Optional[str]=None,
            dimension: Optional[int]=None,
            encoder_num_heads: Optional[int]=None,
            encoder_dim_feedforward: Optional[int]=None,
            encoder_activation_function: Optional[str]=None,
            encoder_dropout: Optional[float]=None,
            encoder_num_layers: Optional[int]=None,
            decoder_dim_feedforward: Optional[int]=None,
            decoder_activation_function: Optional[str]=None,
            decoder_dropout: Optional[float]=None,
            decoder_num_layers: Optional[int]=None,
            checkpointing: Optional[bool]=None,
            qv1_model: Optional[QVModel]=None,
            qv2_model: Optional[QVModel]=None,
            **_) -> None:
        super(QModel, self).__init__()

        if position_encoder is not None:
            if dimension is None:
                raise ValueError('`dimension` must be specified.')
            if encoder_num_heads is None:
                raise ValueError('`encoder_num_heads` must be specified.')
            if encoder_dim_feedforward is None:
                raise ValueError('`encoder_dim_feedforward` must be specified.')
            if encoder_activation_function is None:
                raise ValueError('`encoder_activation_function` must be specified.')
            if encoder_dropout is None:
                raise ValueError('`encoder_dropout` must be specified.')
            if encoder_num_layers is None:
                raise ValueError('`encoder_num_layers` must be specified.')
            if decoder_dim_feedforward is None:
                raise ValueError('`decoder_dim_feedforward` must be specified.')
            if decoder_activation_function is None:
                raise ValueError('`decoder_activation_function` must be specified.')
            if decoder_dropout is None:
                raise ValueError('`decoder_dropout` must be specified.')
            if checkpointing is None:
                raise ValueError('`checkpointing` must be specified.')
            if qv1_model is not None:
                raise ValueError('`qv1_model` must not be specified.')
            if qv2_model is not None:
                raise ValueError('`qv2_model` must not be specified.')

            qv1_encoder = Encoder(
                position_encoder=position_encoder, dimension=dimension, num_heads=encoder_num_heads,
                dim_feedforward=encoder_dim_feedforward,
                activation_function=encoder_activation_function, dropout=encoder_dropout,
                num_layers=encoder_num_layers, checkpointing=checkpointing)
            qv1_decoder = QVDecoder(
                dimension=dimension, dim_feedforward=decoder_dim_feedforward,
                activation_function=decoder_activation_function, dropout=decoder_dropout,
                num_layers=decoder_num_layers)
            self._qv1_model = QVModel(qv1_encoder, qv1_decoder)

            qv2_encoder = Encoder(
                position_encoder=position_encoder, dimension=dimension, num_heads=encoder_num_heads,
                dim_feedforward=encoder_dim_feedforward,
                activation_function=encoder_activation_function, dropout=encoder_dropout,
                num_layers=encoder_num_layers, checkpointing=checkpointing)
            qv2_decoder = QVDecoder(
                dimension=dimension, dim_feedforward=decoder_dim_feedforward,
                activation_function=decoder_activation_function, dropout=decoder_dropout,
                num_layers=decoder_num_layers)
            self._qv2_model = QVModel(qv2_encoder, qv2_decoder)
        else:
            if dimension is not None:
                raise ValueError('`dimension` must not be specified.')
            if encoder_num_heads is not None:
                raise ValueError('`encoder_num_heads` must not be specified.')
            if encoder_dim_feedforward is not None:
                raise ValueError('`encoder_dim_feedforward` must not be specified.')
            if encoder_activation_function is not None:
                raise ValueError('`encoder_activation_function` must not be specified.')
            if encoder_dropout is not None:
                raise ValueError('`encoder_dropout` must not be specified.')
            if encoder_num_layers is not None:
                raise ValueError('`encoder_num_layers` must not be specified.')
            if decoder_dim_feedforward is not None:
                raise ValueError('`decoder_dim_feedforward` must not be specified.')
            if decoder_activation_function is not None:
                raise ValueError('`decoder_activation_function` must not be specified.')
            if decoder_dropout is not None:
                raise ValueError('`decoder_dropout` must not be specified.')
            if decoder_num_layers is not None:
                raise ValueError('`decoder_num_layers` must not be specified.')
            if checkpointing is not None:
                raise ValueError('`checkpointing` must not be specified.')
            if qv1_model is None:
                raise ValueError('`qv1_model` must be specified.')
            if qv2_model is None:
                raise ValueError('`qv2_model` must be specified.')

            assert isinstance(qv1_model, DistributedDataParallel) == isinstance(qv2_model, DistributedDataParallel)
            if isinstance(qv1_model, DistributedDataParallel):
                self._qv1_model = qv1_model.module
            else:
                self._qv1_model = qv1_model
            if isinstance(qv2_model, DistributedDataParallel):
                self._qv2_model = qv2_model.module
            else:
                self._qv2_model = qv2_model

    def mode(self, mode: str) -> None:
        if mode not in ('training', 'validation', 'prediction'):
            raise ValueError(mode)

    def forward(self, x) -> torch.Tensor:
        q1, _ = self._qv1_model(x)
        q2, _ = self._qv2_model(x)
        return torch.minimum(q1, q2)
