# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import range

import sys
import numpy as np


def discords(mp, ex_zone, k=3):
    """
    Computes the top k discords from a matrix profile
    :param mp: matrix profile numpy array
    :param ex_zone: the number of samples to exclude and set to Inf on either side of a found discord
    :param k: the number of discords to discover
    :return: list of discord indexes
    Returns a list of indexes represent the discord starting locations. MaxInt indicates there
    were no more discords that could be found due to too many exclusions or profile being too
    small. Discord start indices are sorted by highest matrix profile value.
    """

    k = len(mp) if k > len(mp) else k
    mp_current = np.copy(mp)
    d = np.zeros(k)

    for i in range(k):
        max_val = 0
        max_idx = sys.maxsize
        for j, val in enumerate(mp_current):
            if not np.isinf(val) and val > max_val:
                max_val, max_idx = val, j

        d[i] = max_idx
        mp_current[max([max_idx - ex_zone, 0]):min([max_idx + ex_zone, len(mp_current)])] = np.inf

    return d
