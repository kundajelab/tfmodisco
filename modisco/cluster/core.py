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


class AbstractClusterer(object):

    def cluster(self, affinity_mat):
        raise NotImplementedError()


class PhenographCluster(AbstractClusterer):

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
        

class LouvainCluster(AbstractClusterer):

    def __init__(self, affmat_transformer=None, min_cluster_size=10,
                       q_tol=1e-3, louvain_time_limit=2000,
                       verbose=True, min_nonneg=0, max_nonneg=None,
                       second_transformer = None):
        self.affmat_transformer = affmat_transformer
        self.min_cluster_size = min_cluster_size
        self.q_tol = q_tol
        self.louvain_time_limit = louvain_time_limit
        self.verbose = verbose
        self.min_nonneg = min_nonneg
        self.max_nonneg = max_nonneg
        self.second_transformer = second_transformer
    
    def cluster(self, orig_affinity_mat):

        self.max_nonneg = len(orig_affinity_mat)
        if (self.verbose):
            print("Beginning preprocessing + Louvain")
        all_start = time.time()
        if (self.affmat_transformer is not None):
            affinity_mat = self.affmat_transformer(orig_affinity_mat)
        else:
            affinity_mat = orig_affinity_mat

        #subset the affinity mat to rows that have at least
        #self.min_nonneg nonnegative connections; the ones that don't pass
        #this filter as are assigned a cluster of -2 
        #the -2 is to discount the self-connection
        num_connected_neighbors = np.sum(affinity_mat > 0, axis=1)-1
        filtered_mask = ((num_connected_neighbors >= self.min_nonneg)*
                         (num_connected_neighbors <= self.max_nonneg))

        if (self.verbose):
            print(str(sum(filtered_mask))+" points remain after subsetting")

        if (self.second_transformer is not None):
            if (self.verbose):
                print("Beginning second round preprocessing")
            subsetted_affmat = orig_affinity_mat[filtered_mask]
            subsetted_affmat = subsetted_affmat[:,filtered_mask]
            subsetted_affmat = self.second_transformer(subsetted_affmat)
        else:
            subsetted_affmat = affinity_mat[filtered_mask]
            subsetted_affmat = subsetted_affmat[:,filtered_mask]

        subset_communities, graph, Q, hierarchy =\
            ph.cluster.runlouvain_given_graph(
                graph=subsetted_affmat,
                min_cluster_size=self.min_cluster_size,
                q_tol=self.q_tol,
                louvain_time_limit=self.louvain_time_limit)

        #fill in the communities that are -2
        communities = (np.ones(len(affinity_mat))*-2).astype("int")
        communities[filtered_mask] = subset_communities

        cluster_results = LouvainClusterResults(
                cluster_indices=communities,
                hierarchy=hierarchy,
                Q=Q)
        if (self.verbose):
            print("Preproc + Louvain took "+str(time.time()-all_start)+" s")
        return cluster_results
 
