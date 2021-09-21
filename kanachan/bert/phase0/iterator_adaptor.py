#!/usr/bin/env python3

import pathlib
from kanachan.iterator_adaptor_base import IteratorAdaptorBase


class IteratorAdaptor(IteratorAdaptorBase):
    def __init__(self, path: pathlib.Path, num_dimensions: int, dtype) -> None:
        super(IteratorAdaptor, self).__init__(path, num_dimensions, dtype)

    def __next__(self):
        return super(IteratorAdaptor, self).__next__()[:-1]
