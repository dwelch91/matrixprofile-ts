# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import random
from six.moves import range


class Order:
    """
    These objects define the order in which the distance profiles are calculated for a given matrix profile
    """
    def next(self):
        raise NotImplementedError("next() not implemented")


class LinearOrder(Order):
    def __init__(self, m):
        self.m = m
        self.idx = -1


    def next(self):
        self.idx += 1
        return self.idx if self.idx < self.m else None


class RandomOrder(Order):
    def __init__(self, m):
        self.idx = -1
        self.indices = list(range(m))
        random.shuffle(self.indices)


    def next(self):
        self.idx += 1
        try:
            return self.indices[self.idx]
        except IndexError:
            return
