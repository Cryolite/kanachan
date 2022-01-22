#!/usr/bin/env python3

import sys
from torch import nn
from kanachan.training.bert.phase2.decoder import Decoder
from kanachan.training.bert.phase2.model import Model
from kanachan.training.bert.phase2.iterator_adaptor import IteratorAdaptor
from kanachan.training.bert import training


if __name__ == '__main__':
    loss_function = nn.HuberLoss()
    training.main(
        program_description='BERT training phase 2 - maximize round delta of score.',
        decoder_type=Decoder, model_type=Model, default_optimizer='lamb',
        iterator_adaptor_type=IteratorAdaptor, loss_function=loss_function)
    sys.exit(0)
