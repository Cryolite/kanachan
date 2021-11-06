#!/usr/bin/env python3

import sys
from torch import nn
from kanachan.bert import oneshot_training
from kanachan.bert.phase2.training_decoder import TrainingDecoder
from kanachan.bert.phase2.model import Model
from kanachan.bert.phase2.iterator_adaptor import IteratorAdaptor


if __name__ == '__main__':
    loss_function = nn.MSELoss()
    oneshot_training.main(
        program_description='BERT training phase 2 - oneshot training to maximize round delta of score.',
        decoder_type=TrainingDecoder, model_type=Model,
        default_optimizer='adam', iterator_adaptor_type=IteratorAdaptor,
        loss_function=loss_function)
    sys.exit(0)
