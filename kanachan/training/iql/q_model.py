#!/usr/bin/env python3

from typing import Optional
import torch
from torch import nn
from kanachan.training.bert.encoder import Encoder
from kanachan.training.iql.qv_model import (QVDecoder, QVModel,)


class QModel(nn.Module):
    def __init__(
            self, *,
            dimension: Optional[int]=None,
            num_heads: Optional[int]=None,
            num_layers: Optional[int]=None,
            dim_feedforward: Optional[int]=None,
            dim_final_feedforward: Optional[int]=None,
            activation_function: Optional[str]=None,
            dropout: Optional[float]=None,
            checkpointing: Optional[bool]=None,
            qv1_model: Optional[QVModel]=None,
            qv2_model: Optional[QVModel]=None,
            **kwargs) -> None:
        super(QModel, self).__init__()

        if dimension is not None:
            if num_heads is None:
                raise ValueError('`num_heads` must be specified.')
            if num_layers is None:
                raise ValueError('`num_layers` must be specified.')
            if dim_feedforward is None:
                raise ValueError('`dim_feedforward` must be specified.')
            if dim_final_feedforward is None:
                raise ValueError('`dim_final_feedforward` must be specified.')
            if activation_function is None:
                raise ValueError('`activation_function` must be specified.')
            if dropout is None:
                raise ValueError('`dropout` must be specified.')
            if checkpointing is None:
                raise ValueError('`checkpointing` must be specified.')
            if qv1_model is not None:
                raise ValueError('`qv1_model` must not be specified.')
            if qv2_model is not None:
                raise ValueError('`qv2_model` must not be specified.')

            qv1_encoder = Encoder(
                dimension=dimension, num_heads=num_heads, num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                activation_function=activation_function, dropout=dropout,
                checkpointing=checkpointing)
            qv1_decoder = QVDecoder(
                dimension=dimension, dim_final_feedforward=dim_final_feedforward,
                activation_function=activation_function, dropout=dropout)
            self._qv1_model = QVModel(qv1_encoder, qv1_decoder)

            qv2_encoder = Encoder(
                dimension=dimension, num_heads=num_heads, num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                activation_function=activation_function, dropout=dropout,
                checkpointing=checkpointing)
            qv2_decoder = QVDecoder(
                dimension=dimension, dim_final_feedforward=dim_final_feedforward,
                activation_function=activation_function, dropout=dropout)
            self._qv2_model = QVModel(qv2_encoder, qv2_decoder)
        else:
            if num_heads is not None:
                raise ValueError('`num_heads` must not be specified.')
            if num_layers is not None:
                raise ValueError('`num_layers` must not be specified.')
            if dim_feedforward is not None:
                raise ValueError('`dim_feedforward` must not be specified.')
            if dim_final_feedforward is not None:
                raise ValueError(
                    '`dim_final_feedforward` must not be specified.')
            if activation_function is not None:
                raise ValueError(
                    '`activation_function` must not be specified.')
            if dropout is not None:
                raise ValueError('`dropout` must not be specified.')
            if checkpointing is not None:
                raise ValueError('`checkpointing` must not be specified.')
            if qv1_model is None:
                raise ValueError('`qv1_model` must be specified.')
            if qv2_model is None:
                raise ValueError('`qv2_model` must be specified.')

            # TODO: Replace the following conditions with `isinstance(_, DistributedDataParallel)`
            #       when `torch.nn.parallel.DistributedDataParallel` is introduced.
            if hasattr(self._qv1_model, 'module'):
                self._qv1_model = qv1_model.module
            else:
                self._qv1_model = qv1_model
            if hasattr(self._qv2_model, 'module'):
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
