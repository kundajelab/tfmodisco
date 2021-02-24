#copied from https://github.com/jacoblevine/PhenoGraph/blob/master/phenograph/core.py
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from contextlib import closing
from itertools import repeat
from scipy import sparse as sp
import subprocess
import time
import re
import os
import sys
from collections import defaultdict, Counter
from joblib import Parallel, delayed
import itertools
from tqdm import tqdm
from ...util import print_memory_use


def find_neighbors(data, k=30, metric='minkowski', p=2, method='brute', n_jobs=-1):
    from .bruteforce_nn import knnsearch
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

    print("Finding {} nearest neighbors using {} metric and '{}' algorithm".format(k, metric, algorithm))
    if method == 'kdtree':
        nbrs = NearestNeighbors(n_neighbors=k+1,        # k+1 because results include self
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


def neighbor_graph(kernel, kernelargs):
    """
    Apply kernel (i.e. affinity function) to kernelargs (containing information about the data)
    and return graph as a sparse COO matrix
    :param kernel: affinity function
    :param kernelargs: dictionary of keyword arguments for kernel
    :return graph: n-by-n COO sparse matrix
    """
    i, j, s = kernel(**kernelargs)
    n, k = kernelargs['idx'].shape
    graph = sp.coo_matrix((s, (i, j)), shape=(n, n))
    return graph


def gaussian_kernel(idx, d, sigma):
    """
    For truncated list of k-nearest distances, apply Gaussian kernel
    Assume distances in d are Euclidean
    :param idx:
    :param d:
    :param sigma:
    :return:
    """
    n, k = idx.shape
    i = [np.tile(x, (k,)) for x in range(n)]
    i = np.concatenate(np.array(i))
    j = np.concatenate(idx)
    d = np.concatenate(d)
    f = np.vectorize(lambda x: 1/(sigma * (2 * np.pi) ** .5) * np.exp(-.5 * (x / sigma) ** 2))
    # apply vectorized gaussian function
    p = f(d)
    return i, j, p


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


def calc_jaccard(i_and_idx): #for python 2 map compatibility, single argument
    i, idx = i_and_idx
    """Compute the Jaccard coefficient between i and i's direct neighbors"""
    coefficients = np.fromiter((len(set(idx[i]).intersection(set(idx[j]))) for j in idx[i]), dtype=float)
    coefficients /= (2 * idx.shape[1] - coefficients)
    return (idx[i], coefficients)


def parallel_jaccard_kernel(idx):
    """Compute Jaccard coefficient between nearest-neighbor sets in parallel
    :param idx: n-by-k integer matrix of k-nearest neighbors
    :return (i, j, s): row indices, column indices, and nonzero values for a sparse adjacency matrix
    """
    n = len(idx)
    with closing(Pool()) as pool:
        if (sys.version_info[0]==3):
            jaccard_values = pool.starmap(calc_jaccard,
                                          zip(range(n), repeat(idx)))
        else:
            jaccard_values = pool.map(calc_jaccard, zip(range(n),
                                                     [idx for i in range(n)]))

    graph = sp.lil_matrix((n, n), dtype=float)
    for i, tup in enumerate(jaccard_values):
        graph.rows[i] = tup[0]
        graph.data[i] = tup[1]

    i, j = graph.nonzero()
    s = graph.tocoo().data
    return i, j, s[s > 0]


def graph2binary(filename, graph):
    """
    Write (weighted) graph to binary file filename.bin
    :param filename:
    :param graph:
    :return None: graph is written to filename.bin
    """
    assert sp.issparse(graph) #expecting a coo matrix
    tic = time.time()
    # Unpack values in graph
    i, j = graph.nonzero()
    s = graph.data
    assert len(graph.data)==len(i)
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
        #f.writelines([e for t in zip(ij, s) for e in t]) # this step creates a 60G list for a 4G sparse matrix, avoid
        for idx in range(len(s)):
            f.write(ij[idx])
            f.write( s[idx])
    print("Wrote graph to binary file in {} seconds".format(time.time() - tic))
    print_memory_use()
    sys.stdout.flush()


def get_modularity(msg):
    # pattern = re.compile('modularity increased from -*0.\d+ to 0.\d+')
    pattern = re.compile('modularity increased from -?\d.?\d*e?-?\d* to -?\d.?\d*e?-?\d*')
    matches = pattern.findall(msg.decode())
    q = list()
    for line in matches:
        q.append(line.split(" ")[-1])
    return list(map(float, q))


def get_paths_and_run_convert(filename):
    
    # Use package location to find Louvain code
    # lpath = os.path.abspath(resource_filename(Requirement.parse("PhenoGraph"), 'louvain'))
    lpath = os.path.join(os.path.dirname(__file__), 'louvain')
    try:
        assert os.path.isdir(lpath)
    except AssertionError:
        print("Could not find Louvain code, tried: {}".format(lpath))

    # Determine if we're using Mac or Linux
    if sys.platform.startswith("linux"):
        convert_binary = "linux-convert"
        community_binary = "linux-community"
        hierarchy_binary = "linux-hierarchy"
    elif sys.platform == "darwin":
        convert_binary = "osx-convert"
        community_binary = "osx-community"
        hierarchy_binary = "osx-hierarchy"
    else:
        raise RuntimeError("Operating system could not be determined or is not supported. "
                           "sys.platform == {}".format(sys.platform))
    # Prepend appropriate path separator
    convert_binary = os.path.sep + convert_binary
    community_binary = os.path.sep + community_binary
    hierarchy_binary = os.path.sep + hierarchy_binary

    # run convert
    args = [lpath + convert_binary, '-i', filename + '.bin', '-o',
            filename + '_graph.bin', '-w', filename + '_graph.weights']
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    # check for errors from convert
    if bool(out) or bool(err):
        print("stdout from convert: {}".format(out.decode()))
        print("stderr from convert: {}".format(err.decode()))

    return lpath, community_binary, hierarchy_binary


def parse_l1_clusters(stdout):
    max_idx = 0
    idx_to_cluster = {}
    for i,line in enumerate(stdout.splitlines()):
        idx,cluster = line.split(" ")
        idx,cluster = int(idx),int(cluster)
        max_idx = max(idx, max_idx)
        idx_to_cluster[idx] = cluster
    communities = []
    for i in range(max_idx+1):
        communities.append(idx_to_cluster[i])
    return np.array(communities)


def runlouvain(filename, level_to_return=-1, tol=1e-3,
                         max_clusters=-1, contin_runs=50,
                         max_runs=500, time_limit=2000, seed=1234):
    """
    From binary graph file filename.bin, optimize modularity by running multiple random re-starts of
    the Louvain C++ code.
    Louvain is run repeatedly until modularity has not increased in some number (contin_runs) of runs
    or if the total number of runs exceeds some larger number (max_runs) OR if a time limit (time_limit) is exceeded
    :param filename: *.bin file generated by graph2binary
    :param max_runs: maximum number of times to repeat Louvain before ending iterations and taking best result
    :param time_limit: maximum number of seconds to repeat Louvain before ending iterations and taking best result
    :return communities: community assignments
    :return Q: modularity score corresponding to `communities`
    """
    assert level_to_return==-1 or level_to_return==1

    rng = np.random.RandomState(seed)

    print('Running Louvain modularity optimization')
    tic = time.time()

    (lpath, community_binary, hierarchy_binary) =\
        get_paths_and_run_convert(filename) 

    Q = -np.inf
    run = 0
    updated = 0
    while run - updated < contin_runs and run < max_runs and (time.time() - tic) < time_limit:

        # run community
        fout = open(filename + '.tree', 'w')
        args = [lpath + community_binary, filename + '_graph.bin',
                str(rng.random_integers(0,9999)), '-l',
                str(level_to_return), "-m", str(max_clusters),
                '-v', '-w', filename + '_graph.weights']
        p = subprocess.Popen(args, stdout=fout, stderr=subprocess.PIPE)
        # Here, we print communities to filename.tree and retain the modularity scores reported piped to stderr
        _, msg = p.communicate()
        fout.close()
        # get modularity from err msg
        q = get_modularity(msg)
        run += 1

        if (len(q)==0):
            print(msg)
            sys.stdout.flush()
            raise RuntimeError("No levels found with louvain; stderr above")

        # continue only if we've reached a higher modularity than before
        if q[-1] - Q > tol:

            Q = q[-1]
            updated = run
            if (level_to_return==-1):
                nlevels = len(q)
                #get the topmost level
                args = [lpath + hierarchy_binary,
                        filename + '.tree', '-l', str(nlevels-1)]
                p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)
                out, err = p.communicate()
                communities = []
                for i, line in enumerate(out.decode().splitlines()):
                    communities.append(int(line.split(' ')[-1]))
                communities = np.array(communities)
            else:
                args = ['cat', filename + '.tree']
                p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                out, err = p.communicate()
                communities = parse_l1_clusters(out.decode())

            print("After {} runs, maximum modularity is Q = {}".format(run, Q))

    print("Louvain completed {} runs in {} seconds".format(run, time.time() - tic))
    sys.stdout.flush()

    return communities, Q


def single_louvain_run(lpath, community_binary, hierarchy_binary,
                       filename, level_to_return, max_clusters, seed):

    # run community
    fout = open(filename + '.tree_'+str(seed), 'w')
    args = [lpath + community_binary, filename + '_graph.bin',
            str(seed), '-l',
            str(level_to_return), "-m", str(max_clusters),
            '-v', '-w', filename + '_graph.weights']
    p = subprocess.Popen(args, stdout=fout, stderr=subprocess.PIPE)
    # Here, we print communities to filename.tree and retain the modularity scores reported piped to stderr
    _, msg = p.communicate()
    fout.close()
    # get modularity from err msg
    q = get_modularity(msg)

    if (len(q)==0):
        print(msg)
        sys.stdout.flush()
        raise RuntimeError("No levels found with louvain; stderr above")

    if (level_to_return==-1):
        nlevels = len(q)
        #get the topmost level
        args = [lpath + hierarchy_binary,
                filename + '.tree_'+str(seed), '-l', str(nlevels-1)]
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        out, err = p.communicate()
        communities = []
        for i, line in enumerate(out.decode().splitlines()):
            communities.append(int(line.split(' ')[-1]))
        communities = np.array(communities)
    else:
        args = ['cat', filename + '.tree_'+str(seed)]
        p = subprocess.Popen(args, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        out, err = p.communicate()
        communities = parse_l1_clusters(out.decode())

    return communities


def runlouvain_average_runs(filename, n_runs,
                            level_to_return, verbose,
                            max_clusters=-1,
                            seed=1234, parallel_threads=1):

    assert level_to_return==-1 or level_to_return==1

    rng = np.random.RandomState(seed)

    print('Running Louvain modularity optimization')
    tic = time.time()

    (lpath, community_binary, hierarchy_binary) =\
        get_paths_and_run_convert(filename) 

    coocc_count = None
    chosen_seeds = rng.choice(np.arange(9999), size=n_runs) 
    sys.stdout.flush()
    communities_list = (Parallel(n_jobs=parallel_threads, verbose=verbose) 
                           (delayed(single_louvain_run)
                                   (lpath, community_binary, hierarchy_binary,
                                    filename, level_to_return, max_clusters,
                                    chosen_seed)
                            for i,chosen_seed in
                                zip(range(n_runs),chosen_seeds)))

    #for fault tolerance, sometimes some louvain runs return no
    #communities - filter these out
    communities_list = [x for x in communities_list if len(x) > 0]
    if (len(communities_list) < n_runs):
        sys.stderr.flush()
        print("WARNING!!! only "
              +str(len(communities_list))+" louvain runs"
              +" worked, out of "+str(n_runs), file=sys.stderr)
        sys.stderr.flush()

    print("Louvain completed {} runs in {} seconds".format(
          n_runs, time.time() - tic))
    sys.stdout.flush()

    print_memory_use()
    sys.stdout.flush()

    print("Preparing sparse coo_matrix")
    sparse_coo_start = time.time()
    #rewrite to be a sparse matrix
    cooc_count = sp.coo_matrix((len(communities_list[0]),
                                len(communities_list[0])),
                               dtype="float32").tocsr()

    sys.stdout.flush()
    sys.stderr.flush()
    for communities_idx,communities in tqdm(enumerate(communities_list)):
        cooc_mat = communities[:,None]==communities[None,:]
        csr_to_add = sp.csr_matrix(cooc_mat)
        cooc_count = (cooc_count + csr_to_add)

    sys.stdout.flush()
    sys.stderr.flush()

    cooc_count.multiply(1.0/len(communities_list))
    cooc_count = cooc_count.tocoo()

    print("Prepared sparse coo_matrix in ",time.time()-sparse_coo_start,"s")
    print_memory_use()
    sys.stdout.flush()

    return cooc_count
