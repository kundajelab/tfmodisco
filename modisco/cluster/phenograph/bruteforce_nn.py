#copied from
#https://github.com/jacoblevine/PhenoGraph/blob/master/phenograph/bruteforce_nn.py
"""
Compute k-nearest neighbors using brute force search in parallel
via scipy.spatial.distance.cdist and multiprocessing.Pool
psutil is used to evaluate available memory and minimize the number
of parallel jobs for the available resources
"""
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from scipy.spatial.distance import cdist
from multiprocessing import Pool
from contextlib import closing
from functools import partial
import psutil


def process_chunk(chunk, data, k, metric):
    d = cdist(chunk, data, metric=metric).astype('float32')
    p = np.argpartition(d, k).astype('int32')[:, :k]
    rows = np.arange(chunk.shape[0])[:, None]
    d = d[rows, p]
    i = np.argsort(d)
    return d[rows, i], p[rows, i]


def determine_n_chunks(n, k):
    """Assuming 32 bit representations for distances and indices"""

    # available memory
    available = psutil.virtual_memory().available

    # memory needed to store final knn data (d, idx)
    final = 2 * (n * k * 32) / 8

    # total memory usable for subprocesses
    usable = available - final

    # usable per subprocess
    usable_per_subprocess = usable / psutil.cpu_count()

    # chunk size - number of n-dimensional distance arrays that can be held in memory by each subprocess simultaneously
    chunk_size = usable_per_subprocess // (n * 32)

    return int(n // chunk_size)


def knnsearch(data, k, metric):
    """k-nearest neighbor search via parallelized brute force
    Parameters
    ----------
    data : ndarray
        n observations in d dimensions
    k : int
        number of neighbors (including self)
    metric : str
        see cdist documentation http://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    Returns
    -------
    d : ndarray
        distances to k nearest neighbors
    idx : ndarray
        indices of k nearest neighbors
    Notes
    -----
    This implementation uses np.array_split to pass the data to subprocesses. This uses views and does not copy the data
    in the subprocesses
    """

    f = partial(process_chunk, **{'data': data, 'k': k, 'metric': metric})

    n_chunks = determine_n_chunks(len(data), k)

    if n_chunks > 2:

        with closing(Pool()) as pool:
            result = pool.map(f, np.array_split(data, n_chunks))

        d, idx = zip(*result)

        d, idx = np.vstack(d), np.vstack(idx)

    else:

        d, idx = process_chunk(data, data, k, metric)

    return d, idx
