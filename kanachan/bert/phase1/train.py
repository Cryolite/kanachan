#!/usr/bin/env python3

import sys
from torch import nn
from kanachan.bert.phase1.decoder import Decoder
from kanachan.bert.phase1.model import Model
from kanachan.bert.phase1.iterator_adaptor import IteratorAdaptor
from kanachan.bert import training


if __name__ == '__main__':
    loss_function = nn.CrossEntropyLoss()
    training.main(
        program_description='BERT training phase1 - imitate human players.',
        decoder_type=Decoder, model_type=Model, default_optimizer='lamb',
        iterator_adaptor_type=IteratorAdaptor, loss_function=loss_function)
    sys.exit(0)
