#!/usr/bin/env python3

from torch import nn


class ModelMode(object):
    def __init__(self, model: nn.Module, mode: str) -> None:
        if mode not in ('training', 'validation', 'prediction'):
            raise ValueError(mode)
        self.__model = model
        self.__mode = mode

    def __enter__(self) -> None:
        self.__model.mode(self.__mode)
        if self.__mode != 'training':
            self.__model.eval()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.__model.train()
        self.__model.mode('training')
