#!/usr/bin/env python3

from kanachan.iterator_adaptor_base import IteratorAdaptorBase


NUM_ROUND_RESULT_CATEGORIES = 9


class IteratorAdaptor(IteratorAdaptorBase):
    def __init__(
            self, fp, dimension, prefetch_size, device=None, dtype=None) -> None:
        super(IteratorAdaptor, self).__init__(
            fp, dimension, NUM_ROUND_RESULT_CATEGORIES, 0, prefetch_size,
            device=device, dtype=dtype)
