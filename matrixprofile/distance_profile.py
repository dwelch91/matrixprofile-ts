# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from six.moves import range

from .utils import *
import numpy as np


def naive_distance_profile(ts_a, idx, m, ts_b=None):
    """
    Return the distance profile of a query within tsA against the time series tsB.
    Uses the naive all-pairs comparison. idx defines the starting index of the query
    within tsA and m is the length of the query.
    """

    self_join = False
    if ts_b is None:
        self_join = True
        ts_b = ts_a

    query = ts_a[idx: (idx + m)]
    distance_profile = []
    n = len(ts_b)

    for i in range(n - m + 1):
        distance_profile.append(z_normalize_euclidian(query, ts_b[i:i + m]))

    dp = np.array(distance_profile)

    if self_join:
        trivial_match_range = (int(max(0, idx - np.round(m / 2, 0))), int(min(idx + np.round(m / 2 + 1, 0), n)))

        dp[trivial_match_range[0]: trivial_match_range[1]] = np.inf

    return dp, np.full(n - m + 1, idx, dtype=float)


def mass_distance_profile(ts_a, idx, m, ts_b=None):
    """
    Return the distance profile of a query within tsA against the time series tsB.
    Uses the more efficient MASS comparison. idx defines the starting index of the query
    within tsA and m is the length of the query.
    """

    self_join = False
    if ts_b is None:
        self_join = True
        ts_b = ts_a

    query = ts_a[idx:(idx + m)]
    n = len(ts_b)
    distance_profile = np.real(np.sqrt(mass(query, ts_b).astype(complex)))
    if self_join:
        trivial_match_range = (int(max(0, idx - np.round(m / 2, 0))), int(min(idx + np.round(m / 2 + 1, 0), n)))
        distance_profile[trivial_match_range[0]:trivial_match_range[1]] = np.inf

    # Both the distance profile and corresponding matrix profile index (which should just have the current index)
    return distance_profile, np.full(n - m + 1, idx, dtype=float)


def stomp_distance_profile(ts_a, idx, m, ts_b, dot_first, dp, mean, std):
    """
    Return the distance profile of a query within tsA against the time series tsB.
    Uses the more efficient MASS comparison. idx defines the starting index of the
    query within tsA and m is the length of the query.
    :param ts_a:
    :param idx:
    :param m:
    :param ts_b:
    :param dot_first:
    :param dp:
    :param mean:
    :param std:
    :return:
    """

    self_join = False
    if ts_b is None:
        self_join = True
        ts_b = ts_a

    query = ts_a[idx:(idx + m)]
    n = len(ts_b)

    # Calculate the first distance profile via MASS
    if idx == 0:
        distance_profile = np.real(np.sqrt(mass(query, ts_b).astype(complex)))

        # Currently re-calculating the dot product separately as opposed to updating all of the mass function...
        dot = sliding_dot_product(query, ts_b)

    # Calculate all subsequent distance profiles using the STOMP dot product shortcut
    else:
        res, dot = mass_stomp(query, ts_b, dot_first, dp, idx, mean, std)
        distance_profile = np.real(np.sqrt(res.astype(complex)))

    if self_join:
        trivial_match_range = (int(max(0, idx - np.round(m / 2, 0))), int(min(idx + np.round(m / 2 + 1, 0), n)))
        distance_profile[trivial_match_range[0]:trivial_match_range[1]] = np.inf

    # Both the distance profile and corresponding matrix profile index (which should just have the current index)
    return (distance_profile, np.full(n - m + 1, idx, dtype=float)), dot


if __name__ == "__main__":
    import doctest

    doctest.method()
