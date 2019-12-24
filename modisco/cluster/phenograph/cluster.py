#copied from https://github.com/jacoblevine/PhenoGraph/blob/master/phenograph/cluster.py
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from scipy import sparse as sp
from .core import (gaussian_kernel, parallel_jaccard_kernel, jaccard_kernel,
                   find_neighbors, neighbor_graph, graph2binary, runlouvain,
                   runlouvain_average_runs)
import time
import re
import os
import uuid


def sort_by_size(clusters, min_size):
    """
    Relabel clustering in order of descending cluster size.
    New labels are consecutive integers beginning at 0
    Clusters that are smaller than min_size are assigned to -1
    :param clusters:
    :param min_size:
    :return: relabeled
    """
    relabeled = np.zeros(clusters.shape, dtype=np.int)
    sizes = [sum(clusters == x) for x in np.unique(clusters)]
    o = np.argsort(sizes)[::-1]
    for i, c in enumerate(o):
        if sizes[c] > min_size:
            relabeled[clusters == c] = i
        else:
            relabeled[clusters == c] = -1
    return relabeled


def runlouvain_given_graph(graph, level_to_return, q_tol, louvain_time_limit,
                           min_cluster_size, max_clusters=-1,
                           contin_runs=20, tic=None,
                           seed=1234):
    if (not sp.isspmatrix_coo(graph)):
        graph = sp.coo_matrix(graph) 
    # write to file with unique id
    uid = uuid.uuid1().hex
    graph2binary(uid, graph)
    communities, Q, =\
     runlouvain(uid, level_to_return=level_to_return,
                tol=q_tol, max_clusters=max_clusters,
                contin_runs=contin_runs, 
                time_limit=louvain_time_limit, seed=seed)
    if (tic is not None):
        print("PhenoGraph complete in {} seconds".format(time.time() - tic))
    communities = sort_by_size(communities, min_size=0)
    # clean up
    for f in os.listdir(os.getcwd()):
        if re.search(uid, f):
            os.remove(f)

    return communities, graph, Q


def runlouvain_average_runs_given_graph(
        graph, n_runs, level_to_return, parallel_threads, verbose,
        max_clusters=-1, tic=None, seed=1234):

    if (not sp.isspmatrix_coo(graph)):
        graph = sp.coo_matrix(graph) 
    # write to file with unique id
    uid = uuid.uuid1().hex
    graph2binary(uid, graph)
    coocc_count = runlouvain_average_runs(
                    uid, level_to_return=level_to_return,
                    max_clusters=max_clusters, n_runs=n_runs,
                    seed=seed, parallel_threads=parallel_threads,
                    verbose=verbose)
    if (tic is not None):
        print("PhenoGraph complete in {} seconds".format(time.time() - tic))
    # clean up
    for f in os.listdir(os.getcwd()):
        if re.search(uid, f):
            os.remove(f)

    return coocc_count

