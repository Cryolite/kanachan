#!/usr/bin/env python3

import pathlib
from kanachan.training.iterator_adaptor_base import IteratorAdaptorBase


class IteratorAdaptor(IteratorAdaptorBase):
    def __init__(self, path: pathlib.Path, dimension: int) -> None:
        super(IteratorAdaptor, self).__init__(path, dimension)

    def __next__(self):
        return super(IteratorAdaptor, self).__next__()[:-1]
