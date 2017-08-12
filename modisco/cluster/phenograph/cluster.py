#copied from https://github.com/jacoblevine/PhenoGraph/blob/master/phenograph/cluster.py

import numpy as np
from scipy import sparse as sp
from phenograph.core import (gaussian_kernel, parallel_jaccard_kernel, jaccard_kernel,
                             find_neighbors, neighbor_graph, graph2binary, runlouvain)
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


def cluster(data, k=30, directed=False, prune=False, min_cluster_size=10, jaccard=True,
            primary_metric='euclidean', n_jobs=-1, q_tol=1e-3, louvain_time_limit=2000,
            nn_method='kdtree'):
    """
    PhenoGraph clustering
    :param data: Numpy ndarray of data to cluster, or sparse matrix of k-nearest neighbor graph
        If ndarray, n-by-d array of n cells in d dimensions
        If sparse matrix, n-by-n adjacency matrix
    :param k: Number of nearest neighbors to use in first step of graph construction
    :param directed: Whether to use a symmetric (default) or asymmetric ("directed") graph
        The graph construction process produces a directed graph, which is symmetrized by one of two methods (see below)
    :param prune: Whether to symmetrize by taking the average (prune=False) or product (prune=True) between the graph
        and its transpose
    :param min_cluster_size: Cells that end up in a cluster smaller than min_cluster_size are considered outliers
        and are assigned to -1 in the cluster labels
    :param jaccard: If True, use Jaccard metric between k-neighborhoods to build graph.
        If False, use a Gaussian kernel.
    :param primary_metric: Distance metric to define nearest neighbors.
        Options include: {'euclidean', 'manhattan', 'correlation', 'cosine'}
        Note that performance will be slower for correlation and cosine.
    :param n_jobs: Nearest Neighbors and Jaccard coefficients will be computed in parallel using n_jobs. If n_jobs=-1,
        the number of jobs is determined automatically
    :param q_tol: Tolerance (i.e., precision) for monitoring modularity optimization
    :param louvain_time_limit: Maximum number of seconds to run modularity optimization. If exceeded
        the best result so far is returned
    :param nn_method: Whether to use brute force or kdtree for nearest neighbor search. For very large high-dimensional
        data sets, brute force (with parallel computation) performs faster than kdtree.
    :return communities: numpy integer array of community assignments for each row in data
    :return graph: numpy sparse array of the graph that was used for clustering
    :return Q: the modularity score for communities on graph
    """

    # NB if prune=True, graph must be undirected, and the prune setting takes precedence
    if prune:
        print("Setting directed=False because prune=True")
        directed = False

    if n_jobs == 1:
        kernel = jaccard_kernel
    else:
        kernel = parallel_jaccard_kernel
    kernelargs = {}

    # Start timer
    tic = time.time()
    # Go!
    if isinstance(data, sp.spmatrix) and data.shape[0] == data.shape[1]:
        print("Using neighbor information from provided graph, rather than computing neighbors directly", flush=True)
        lilmatrix = data.tolil()
        d = np.vstack(lilmatrix.data).astype('float32')  # distances
        idx = np.vstack(lilmatrix.rows).astype('int32')  # neighbor indices by row
        del lilmatrix
        assert idx.shape[0] == data.shape[0]
        k = idx.shape[1]
    else:
        d, idx = find_neighbors(data, k=k, metric=primary_metric, method=nn_method, n_jobs=n_jobs)
        print("Neighbors computed in {} seconds".format(time.time() - tic), flush=True)

    subtic = time.time()
    kernelargs['idx'] = idx
    # if not using jaccard kernel, use gaussian
    if not jaccard:
        kernelargs['d'] = d
        kernelargs['sigma'] = 1.
        kernel = gaussian_kernel
        graph = neighbor_graph(kernel, kernelargs)
        print("Gaussian kernel graph constructed in {} seconds".format(time.time() - subtic), flush=True)
    else:
        del d
        graph = neighbor_graph(kernel, kernelargs)
        print("Jaccard graph constructed in {} seconds".format(time.time() - subtic), flush=True)
    if not directed:
        if not prune:
            # symmetrize graph by averaging with transpose
            sg = (graph + graph.transpose()).multiply(.5)
        else:
            # symmetrize graph by multiplying with transpose
            sg = graph.multiply(graph.transpose())
        # retain lower triangle (for efficiency)
        graph = sp.tril(sg, -1)
    # write to file with unique id
    uid = uuid.uuid1().hex
    graph2binary(uid, graph)
    communities, Q = runlouvain(uid, tol=q_tol, time_limit=louvain_time_limit)
    print("PhenoGraph complete in {} seconds".format(time.time() - tic), flush=True)
    communities = sort_by_size(communities, min_cluster_size)
    # clean up
    for f in os.listdir():
        if re.search(uid, f):
            os.remove(f)

    return communities, graph, Q



def knn_jaccard_dist(affinitymat):
    #TODO
    pass


def jaccard_kernel(idx):
    """
    Compute Jaccard coefficient between nearest-neighbor sets
    :param idx: numpy array of nearest-neighbor indices
    :return (i, j, s): tuple of indices and jaccard coefficients, suitable for constructing COO matrix
    """
    n, k = idx.shape
    s = list()
    for i in range(n):
        shared_neighbors = np.fromiter((len(set(idx[i]).intersection(set(idx[j]))) for j in idx[i]), dtype=float)
        s.extend(shared_neighbors / (2 * k - shared_neighbors))
    i = np.concatenate(np.array([np.tile(x, (k, )) for x in range(n)]))
    j = np.concatenate(idx)
    return i, j, s


def calc_jaccard(i, idx):
    """Compute the Jaccard coefficient between i and i's direct neighbors"""
    coefficients = np.fromiter((len(set(idx[i]).intersection(set(idx[j]))) for j in idx[i]), dtype=float)
    coefficients /= (2 * idx.shape[1] - coefficients)
    return idx[i], coefficients


def parallel_jaccard_kernel(idx):
    """Compute Jaccard coefficient between nearest-neighbor sets in parallel
    :param idx: n-by-k integer matrix of k-nearest neighbors
    :return (i, j, s): row indices, column indices, and nonzero values for a sparse adjacency matrix
    """
    n = len(idx)
    with closing(Pool()) as pool:
        jaccard_values = pool.starmap(calc_jaccard, zip(range(n), repeat(idx)))

    graph = sp.lil_matrix((n, n), dtype=float)
    for i, tup in enumerate(jaccard_values):
        graph.rows[i] = tup[0]
        graph.data[i] = tup[1]

    i, j = graph.nonzero()
    s = graph.tocoo().data
    return i, j, s[s > 0]



#copied from
#https://github.com/jacoblevine/PhenoGraph/blob/master/phenograph/core.py
def find_neighbors(data, k=30, metric='minkowski', p=2, method='brute', n_jobs=-1):
    """
    Wraps sklearn.neighbors.NearestNeighbors
    Find k nearest neighbors of every point in data and delete self-distances
    :param data: n-by-d data matrix
    :param k: number for nearest neighbors search
    :param metric: string naming distance metric used to define neighbors
    :param p: if metric == "minkowski", p=2 --> euclidean, p=1 --> manhattan; otherwise ignored.
    :param method: 'brute' or 'kdtree'
    :param n_jobs:
    :return d: n-by-k matrix of distances
    :return idx: n-by-k matrix of neighbor indices
    """
    if metric.lower() == "euclidean":
        metric = "minkowski"
        p = 2
    if metric.lower() == "manhattan":
        metric = "minkowski"
        p = 1
    if metric.lower() == "minkowski":
        algorithm = "auto"
    elif metric.lower() == "cosine" or metric.lower() == "correlation":
        algorithm = "brute"
    else:
        algorithm = "auto"

    print("Finding {} nearest neighbors using {} metric and '{}' algorithm".format(k, metric, algorithm),
          flush=True)
    if method == 'kdtree':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k+1, # k+1 because results include self
                                n_jobs=n_jobs,              # use multiple cores if possible
                                metric=metric,          # primary metric
                                p=p,                    # if metric == "minkowski", 2 --> euclidean, 1 --> manhattan
                                algorithm=algorithm     # kd_tree is fastest for minkowski metrics
                                ).fit(data)
        d, idx = nbrs.kneighbors(data)

    elif method == 'brute':
        d, idx = knnsearch(data, k+1, metric)

    else:
        raise ValueError("Invalid argument to `method` parameters: {}".format(method))

    # Remove self-distances if these are in fact included
    if idx[0, 0] == 0:
        idx = np.delete(idx, 0, axis=1)
        d = np.delete(d, 0, axis=1)
    else:  # Otherwise delete the _last_ column of d and idx
        idx = np.delete(idx, -1, axis=1)
        d = np.delete(d, -1, axis=1)
    return d, idx
        

#copied from
#https://github.com/jacoblevine/PhenoGraph/blob/master/phenograph/core.py
def graph2binary(filename, graph):
    """
    Write (weighted) graph to binary file filename.bin
    :param filename:
    :param graph:
    :return None: graph is written to filename.bin
    """
    tic = time.time()
    # Unpack values in graph
    i, j = graph.nonzero()
    s = graph.data
    # place i and j in single array as edge list
    ij = np.hstack((i[:, np.newaxis], j[:, np.newaxis]))
    # add dummy self-edges for vertices at the END of the list with no neighbors
    ijmax = np.union1d(i, j).max()
    n = graph.shape[0]
    missing = np.arange(ijmax+1, n)
    for q in missing:
        ij = np.append(ij, [[q, q]], axis=0)
        s = np.append(s, [0.], axis=0)
    # Check data types: int32 for indices, float64 for weights
    if ij.dtype != np.int32:
        ij = ij.astype('int32')
    if s.dtype != np.float64:
        s = s.astype('float64')
    # write to file (NB f.writelines is ~10x faster than np.tofile(f))
    with open(filename + '.bin', 'w+b') as f:
        f.writelines([e for t in zip(ij, s) for e in t])
    print("Wrote graph to binary file in {} seconds".format(time.time() - tic))


#copied from
#https://github.com/jacoblevine/PhenoGraph/blob/master/phenograph/core.py
def runlouvain(self, filename, max_runs=100, time_limit=2000, tol=1e-3):
    """
    From binary graph file filename.bin, optimize modularity by running multiple random re-starts of
    the Louvain C++ code.
    Louvain is run repeatedly until modularity has not increased in some number (20) of runs
    or if the total number of runs exceeds some larger number (max_runs) OR if a time limit (time_limit) is exceeded
    :param filename: *.bin file generated by graph2binary
    :param max_runs: maximum number of times to repeat Louvain before ending iterations and taking best result
    :param time_limit: maximum number of seconds to repeat Louvain before ending iterations and taking best result
    :param tol: precision for evaluating modularity increase
    :return communities: community assignments
    :return Q: modularity score corresponding to `communities`
    """
    def get_modularity(msg):
        # pattern = re.compile('modularity increased from -*0.\d+ to 0.\d+')
        pattern = re.compile('modularity increased from -*\d.\d+e*-*\d+ to \d.\d+')
        matches = pattern.findall(msg.decode())
        q = list()
        for line in matches:
            q.append(line.split(sep=" ")[-1])
        return list(map(float, q))

    print('Running Louvain modularity optimization', flush=True)

    # Use package location to find Louvain code
    # lpath = os.path.abspath(resource_filename(Requirement.parse("PhenoGraph"), 'louvain'))
    lpath = os.path.join(os.path.dirname(__file__), 'louvain')
    try:
        assert os.path.isdir(lpath)
    except AssertionError:
        print("Could not find Louvain code, tried: {}".format(lpath), flush=True)

    # Determine if we're using Windows, Mac, or Linux
    if sys.platform == "win32" or sys.platform == "cygwin":
        convert_binary = "convert.exe"
        community_binary = "community.exe"
        hierarchy_binary = "hierarchy.exe"
    elif sys.platform.startswith("linux"):
        convert_binary = "linux-convert"
        community_binary = "linux-community"
        hierarchy_binary = "linux-hierarchy"
    elif sys.platform == "darwin":
        convert_binary = "convert"
        community_binary = "community"
        hierarchy_binary = "hierarchy"
    else:
        raise RuntimeError("Operating system could not be determined or is not supported. "
                           "sys.platform == {}".format(sys.platform), flush=True)
    # Prepend appropriate path separator
    convert_binary = os.path.sep + convert_binary
    community_binary = os.path.sep + community_binary
    hierarchy_binary = os.path.sep + hierarchy_binary

    tic = time.time()

    # run convert
    args = [lpath + convert_binary, '-i', filename + '.bin', '-o',
            filename + '_graph.bin', '-w', filename + '_graph.weights']
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    # check for errors from convert
    if bool(out) or bool(err):
        print("stdout from convert: {}".format(out.decode()))
        print("stderr from convert: {}".format(err.decode()))

    Q = 0
    run = 0
    updated = 0
    while run - updated < 20 and run < max_runs and (time.time() - tic) < time_limit:

        # run community
        fout = open(filename + '.tree', 'w')
        args = [lpath + community_binary, filename + '_graph.bin', '-l', '-1', '-v', '-w', filename + '_graph.weights']
        p = subprocess.Popen(args, stdout=fout, stderr=subprocess.PIPE)
        # Here, we print communities to filename.tree and retain the modularity scores reported piped to stderr
        _, msg = p.communicate()
        fout.close()
        # get modularity from err msg
        q = get_modularity(msg)
        run += 1

        # continue only if we've reached a higher modularity than before
        if q[-1] - Q > tol:

            Q = q[-1]
            updated = run

            # run hierarchy
            args = [lpath + hierarchy_binary, filename + '.tree']
            p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out, err = p.communicate()
            # find number of levels in hierarchy and number of nodes in graph
            nlevels = int(re.findall('\d+', out.decode())[0])
            nnodes = int(re.findall('level 0: \d+', out.decode())[0].split(sep=" ")[-1])

            # get community assignments at each level in hierarchy
            hierarchy = np.empty((nnodes, nlevels), dtype='int')
            for level in range(nlevels):
                    args = [lpath + hierarchy_binary, filename + '.tree', '-l', str(level)]
                    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out, err = p.communicate()
                    h = np.empty((nnodes,))
                    for i, line in enumerate(out.decode().splitlines()):
                        h[i] = int(line.split(sep=' ')[-1])
                    hierarchy[:, level] = h

            communities = hierarchy[:, nlevels-1]

            print("After {} runs, maximum modularity is Q = {}".format(run, Q), flush=True)

    print("Louvain completed {} runs in {} seconds".format(run, time.time() - tic), flush=True)

    return communities, Q 
