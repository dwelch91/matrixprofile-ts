# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.fft as fft


def z_normalize(ts):
    """
    Return a z-normalized version of the time series
    :param ts: Timeseries
    :return: Z-normalized timeseries
    """

    ts -= np.mean(ts)
    std = np.std(ts)

    if std == 0:
        raise ValueError("The Standard Deviation cannot be zero")

    #ts /= std
    return ts / std


def z_normalize_euclidian(ts_a, ts_b):
    """
    Return the z-normalized Euclidian distance between the time series ts_a and ts_b
    :param ts_a:
    :param ts_b:
    :return:
    """

    if len(ts_a) != len(ts_b):
        raise ValueError("ts_a and ts_b must be the same length")

    return np.linalg.norm(z_normalize(ts_a.astype("float64")) - z_normalize(ts_b.astype("float64")))


def mov_mean_std(ts, m):
    """
    Calculate the mean and standard deviation within a moving window of width m passing across the time series ts
    :param ts:
    :param m:
    :return: (moving mean, moving std dev)
    """

    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")

    # Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts), 0, 0)

    # Add zero to the beginning of the cumsum of ts ** 2
    s_sq = np.insert(np.cumsum(ts ** 2), 0, 0)
    seg_sum = s[m:] - s[:-m]
    seg_sum_sq = s_sq[m:] - s_sq[:-m]
    return seg_sum / m, np.sqrt(seg_sum_sq / m - (seg_sum / m) ** 2)


def mov_std(ts, m):
    """
    Calculate the standard deviation within a moving window of width m passing across the time series ts
    :param ts: Timeseries
    :param m: Window width
    :return: Std dev
    """

    if m <= 1:
        raise ValueError("Query length must be longer than one")

    ts = ts.astype("float")
    # Add zero to the beginning of the cumsum of ts
    s = np.insert(np.cumsum(ts), 0, 0)
    # Add zero to the beginning of the cumsum of ts ** 2
    s_sq = np.insert(np.cumsum(ts ** 2), 0, 0)
    seg_sum = s[m:] - s[:-m]
    seg_sum_sq = s_sq[m:] - s_sq[:-m]
    return np.sqrt(seg_sum_sq / m - (seg_sum / m) ** 2)


def sliding_dot_product(query, ts):
    """
    Calculate the dot product between the query and all subsequences of length(query)
    in the timeseries ts. Note that we use Numpy's rfft method instead of fft.
    :param query:
    :param ts:
    :return:
    """

    m = len(query)
    n = len(ts)

    # If length is odd, zero-pad time time series
    ts_add = 0
    if n % 2 == 1:
        ts = np.insert(ts, 0, 0)
        ts_add = 1

    q_add = 0
    # If length is odd, zero-pad query
    if m % 2 == 1:
        query = np.insert(query, 0, 0)
        q_add = 1

    # This reverses the array
    query = query[::-1]

    query = np.pad(query, (0, n - m + ts_add - q_add), 'constant')

    # Determine trim length for dot product. Note that zero-padding of the query has no effect on array length,
    # which is solely determined by the longest vector
    trim = m - 1 + ts_add

    dot_product = fft.irfft(fft.rfft(ts) * fft.rfft(query))

    # Note that we only care about the dot product results from index m-1 onwards, as the first few values aren't
    # true dot products (due to the way the FFT works for dot products)
    return dot_product[trim:]


def dot_product_stomp(ts, m, dot_first, dot_prev, order):
    """
    Updates the sliding dot product for time series ts from the previous dot product dot_prev.
    QT(1,1) is pulled from the initial dot product as dot_first
    :param ts: Timeseries
    :param m: Length
    :param dot_first:
    :param dot_prev:
    :param order:
    :return:
    """

    length = len(ts) - m + 1
    dot = np.roll(dot_prev, 1)
    dot += ts[order + m - 1] * ts[m - 1:length + m] - ts[order - 1] * np.roll(ts[:length], 1)
    dot[0] = dot_first[order]
    return dot


def mass(query, ts):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS) between a query and timeseries.
    MASS is a Euclidian distance similarity search algorithm. Note that we are returning the square of MASS.
    :param query: Query
    :param ts: Timeseries
    :return: Square of MASS
    """

    m = len(query)
    q_mean = np.mean(query)
    q_std = np.std(query)
    mean, std = mov_mean_std(ts, m)
    dot = sliding_dot_product(query, ts)
    return 2 * m * (1 - (dot - m * mean * q_mean) / (m * std * q_std))


def mass_stomp(query, ts, dot_first, dot_prev, index, mean, std):
    """
    Calculates Mueen's ultra-fast Algorithm for Similarity Search (MASS) between a query and timeseries
    using the STOMP dot product speedup. Note that we are returning the square of MASS.
    :param query:
    :param ts:
    :param dot_first:
    :param dot_prev:
    :param index:
    :param mean:
    :param std:
    :return:
    """

    m = len(query)
    dot = dot_product_stomp(ts, m, dot_first, dot_prev, index)
    res = 2 * m * (1 - (dot - m * mean[index] * mean) / (m * std[index] * std))

    # Return both the MASS calculation and the dot product
    return res, dot


def apply_av(mp, av=None):
    """
    Applies annotation vector 'av' to the original matrix profile and matrix profile index contained in tuple mp,
    and returns the corrected MP/MPI as a new tuple
    :param mp: Matrix profile
    :param av: Annotation vector
    :return: Corrected matrix profile
    """
    av = [1.0] if av is None else av

    if len(mp[0]) != len(av):
        raise ValueError("Annotation Vector must be the same length as the matrix profile")

    return mp[0] * np.array(av)
