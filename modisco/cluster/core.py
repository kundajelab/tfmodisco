from __future__ import division, print_function, absolute_import
import sklearn
from . import phenograph as ph
import numpy as np
import time


class ClusterResults(object):

    def __init__(self, cluster_indices):
        self.cluster_indices = cluster_indices 

    def remap(self, mapping):
        return ClusterResults(cluster_indices=
                np.array([mapping[x] if x in mapping else x
                 for x in self.cluster_indices]))


class LouvainClusterResults(ClusterResults):

    def __init__(self, cluster_indices, hierarchy, Q):
        super(LouvainClusterResults, self).__init__(
         cluster_indices=cluster_indices)
        self.hierarchy = hierarchy
        self.Q = Q


class AbstractAffinityMatClusterer(object):

    def cluster(self, affinity_mat):
        raise NotImplementedError()


class PhenographCluster(AbstractAffinityMatClusterer):

    def __init__(self, k=30, min_cluster_size=10, jaccard=True,
                       primary_metric='euclidean',
                       n_jobs=-1, q_tol=1e-3, louvain_time_limit=2000,
                       nn_method='kdtree'):
        self.k = k
        self.min_cluster_size = min_cluster_size
        self.jaccard = jaccard
        self.primary_metric = primary_metric
        self.n_jobs = n_jobs
        self.q_tol = q_tol
        self.louvain_time_limit = louvain_time_limit
        self.nn_method = nn_method
    
    def cluster(self, affinity_mat):
        communities, graph, Q, hierarchy = ph.cluster.cluster(
            data=affinity_mat,
            k=self.k, min_cluster_size=self.min_cluster_size,
            jaccard=self.jaccard, primary_metric=self.primary_metric,
            n_jobs=self.n_jobs, q_tol=self.q_tol,
            louvain_time_limit=self.louvain_time_limit,
            nn_method=self.nn_method)
        return LouvainClusterResults(
                cluster_indices=communities,
                hierarchy=hierarchy,
                Q=Q)
        

class LouvainCluster(AbstractAffinityMatClusterer):

    def __init__(self, affmat_transformer=None, min_cluster_size=10,
                       q_tol=1e-3, louvain_time_limit=2000,
                       verbose=True):
        self.affmat_transformer = affmat_transformer
        self.min_cluster_size = min_cluster_size
        self.q_tol = q_tol
        self.louvain_time_limit = louvain_time_limit
        self.verbose = verbose
    
    def cluster(self, orig_affinity_mat):

        if (self.verbose):
            print("Beginning preprocessing + Louvain")
        all_start = time.time()
        if (self.affmat_transformer is not None):
            affinity_mat = self.affmat_transformer(orig_affinity_mat)
        else:
            affinity_mat = orig_affinity_mat

        communities, graph, Q, hierarchy =\
            ph.cluster.runlouvain_given_graph(
                graph=affinity_mat,
                min_cluster_size=self.min_cluster_size,
                q_tol=self.q_tol,
                louvain_time_limit=self.louvain_time_limit)

        cluster_results = LouvainClusterResults(
                cluster_indices=communities,
                hierarchy=hierarchy,
                Q=Q)

        if (self.verbose):
            print("Preproc + Louvain took "+str(time.time()-all_start)+" s")
        return cluster_results
 
