#!/usr/bin/env python3

from pathlib import Path
from kanachan.training.iterator_adaptor_base import IteratorAdaptorBase


class IteratorAdaptor(IteratorAdaptorBase):
    def __init__(self, path: Path) -> None:
        super(IteratorAdaptor, self).__init__(path)

    def __next__(self):
        return super(IteratorAdaptor, self).__next__()[:-1]
