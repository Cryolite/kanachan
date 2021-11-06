#!/usr/bin/env python3

import sys
from torch import nn
from kanachan.bert import oneshot_training
from kanachan.bert.phase0.decoder import Decoder
from kanachan.bert.phase0.model import Model
from kanachan.bert.phase0.iterator_adaptor import IteratorAdaptor


if __name__ == '__main__':
    loss_function = nn.CrossEntropyLoss()
    oneshot_training.main(
        program_description='BERT training phase 0 - oneshot training to imitate human players.',
        decoder_type=Decoder, model_type=Model, default_optimizer='lamb',
        iterator_adaptor_type=IteratorAdaptor, loss_function=loss_function)
    sys.exit(0)
