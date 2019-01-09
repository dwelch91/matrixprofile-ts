# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .distance_profile import naive_distance_profile, mass_distance_profile, stomp_distance_profile
from . import order
from .utils import mov_mean_std
import numpy as np


def _matrix_profile(ts_a, m, order_class, distance_profile_function, ts_b=None):
    """

    :param ts_a:
    :param m:
    :param order_class:
    :param distance_profile_function:
    :param ts_b:
    :return:
    """
    order = order_class(len(ts_a) - m + 1)

    # Account for the case where ts_b is None (note that ts_b = None triggers a self matrix profile)
    if ts_b is None:
        mp = np.full(len(ts_a) - m + 1, np.inf)
        mp_index = np.full(len(ts_a) - m + 1, np.inf)

    else:
        mp = np.full(len(ts_b) - m + 1, np.inf)
        mp_index = np.full(len(ts_b) - m + 1, np.inf)

    idx = order.next()
    while idx is not None:
        distance_profile, query_segments_id = distance_profile_function(ts_a, idx, m, ts_b)

        # Check which of the indices have found a new minimum
        ids_to_update = distance_profile < mp

        # Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mp_index[ids_to_update] = query_segments_id[ids_to_update]

        # Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp, distance_profile)
        idx = order.next()

    return mp, mp_index


def _matrix_profile_sampling(ts_a, m, order_class, distance_profile_function, ts_b=None, sampling=0.2):
    """

    :param ts_a:
    :param m:
    :param order_class:
    :param distance_profile_function:
    :param ts_b:
    :param sampling:
    :return:
    """
    order_ = order_class(len(ts_a) - m + 1)

    # Account for the case where ts_b is None (note that ts_b = None triggers a self matrix profile)
    if ts_b is None:
        mp = np.full(len(ts_a) - m + 1, np.inf)
        mp_index = np.full(len(ts_a) - m + 1, np.inf)

    else:
        mp = np.full(len(ts_b) - m + 1, np.inf)
        mp_index = np.full(len(ts_b) - m + 1, np.inf)

    idx = order_.next()

    # Define max numbers of iterations to sample
    iters = (len(ts_a) - m + 1) * sampling

    iter_val = 0

    while iter_val < iters:
        distance_profile, query_segments_id = distance_profile_function(ts_a, idx, m, ts_b)

        # Check which of the indices have found a new minimum
        ids_to_update = distance_profile < mp

        # Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mp_index[ids_to_update] = query_segments_id[ids_to_update]

        # Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp, distance_profile)
        idx = order_.next()

        iter_val += 1

    return mp, mp_index


def _matrix_profile_stomp(ts_a, m, order_class, distance_profile_function, ts_b=None):
    """
    Write matrix profile function for STOMP and then consolidate later! (aka link to the previous distance profile)
    :param ts_a:
    :param m:
    :param order_class:
    :param distance_profile_function:
    :param ts_b:
    :return:
    """
    order = order_class(len(ts_a) - m + 1)

    # Account for the case where ts_b is None (note that ts_b = None triggers a self matrix profile)
    if ts_b is None:
        mp = np.full(len(ts_a) - m + 1, np.inf)
        mp_index = np.full(len(ts_a) - m + 1, np.inf)

    else:
        mp = np.full(len(ts_b) - m + 1, np.inf)
        mp_index = np.full(len(ts_b) - m + 1, np.inf)

    idx = order.next()

    # Get moving mean and standard deviation
    mean, std = mov_mean_std(ts_a, m)

    # Initialize code to set dot_prev to None for the first pass
    dp = None

    # Initialize dot_first to None for the first pass
    dot_first = None

    while idx is not None:
        # Need to pass in the previous sliding dot product for subsequent distance profile calculations
        (distance_profile, query_segments_id), dot_prev = distance_profile_function(ts_a, idx, m, ts_b, dot_first, dp, mean, std)

        if idx == 0:
            dot_first = dot_prev

        # Check which of the indices have found a new minimum
        ids_to_update = distance_profile < mp

        # Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
        mp_index[ids_to_update] = query_segments_id[ids_to_update]

        # Update the matrix profile to include the new minimum values (where appropriate)
        mp = np.minimum(mp, distance_profile)
        idx = order.next()

        dp = dot_prev

    return mp, mp_index


def stampi_update(ts_a, m, mp, mp_index, newval, ts_b=None, distance_profile_function=mass_distance_profile):
    """
    Updates the self-matched matrix profile for a time series Ts_a with the arrival of a new data point newval.
    Note that comparison of two separate time-series with new data arriving will be built later -> currently,
    ts_b should be set to ts_a
    :param ts_a:
    :param m:
    :param mp:
    :param mp_index:
    :param newval:
    :param ts_b:
    :param distance_profile_function:
    :return:
    """

    # Update time-series array with recent value
    ts_a_new = np.append(np.copy(ts_a), newval)

    # Expand matrix profile and matrix profile index to include space for latest point
    mp_new = np.append(np.copy(mp), np.inf)
    mp_index_new = np.append(np.copy(mp_index), np.inf)

    # Determine new index value
    idx = len(ts_a_new) - m

    distance_profile, query_segments_id = distance_profile_function(ts_a_new, idx, m, ts_b)

    # Check which of the indices have found a new minimum
    ids_to_update = distance_profile < mp_new

    # Update the Matrix Profile Index to indicate that the current index is the minimum location for the aforementioned indices
    mp_index_new[ids_to_update] = query_segments_id[ids_to_update]

    # Update the matrix profile to include the new minimum values (where appropriate)
    mp_final = np.minimum(np.copy(mp_new), distance_profile)

    # Finally, set the last value in the matrix profile to the minimum of the distance profile (with corresponding index)
    mp_final[-1] = np.min(distance_profile)
    mp_index_new[-1] = np.argmin(distance_profile)

    return mp_final, mp_index_new


def naive_mp(ts_a, m, ts_b=None):
    """
    Naive matrix profile
    :param ts_a:
    :param m:
    :param ts_b:
    :return:
    """
    return _matrix_profile(ts_a, m, order.LinearOrder, naive_distance_profile, ts_b)


def stmp(ts_a, m, ts_b=None):
    """

    :param ts_a:
    :param m:
    :param ts_b:
    :return:
    """
    return _matrix_profile(ts_a, m, order.LinearOrder, mass_distance_profile, ts_b)


def stamp(ts_a, m, ts_b=None, sampling=0.2):
    """
    STAMP
    :param ts_a:
    :param m:
    :param ts_b:
    :param sampling:
    :return:
    """
    return _matrix_profile_sampling(ts_a, m, order.RandomOrder, mass_distance_profile, ts_b, sampling=sampling)


def stomp(ts_a, m, ts_b=None):
    """
    STOMP
    :param ts_a:
    :param m:
    :param ts_b:
    :return:
    """
    return _matrix_profile_stomp(ts_a, m, order.LinearOrder, stomp_distance_profile, ts_b)


if __name__ == "__main__":
    import doctest
    doctest.method()
