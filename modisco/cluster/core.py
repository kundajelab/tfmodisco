from __future__ import division, print_function, absolute_import
import sklearn
from . import phenograph as ph
import numpy as np
import time


class ClusterResults(object):

    def __init__(self, cluster_indices):
        self.cluster_indices = cluster_indices 


class LouvainClusterResults(ClusterResults):

    def __init__(self, cluster_indices, hierarchy, Q):
        super(LouvainClusterResults, self).__init__(
         cluster_indices=cluster_indices)
        self.hierarchy = hierarchy
        self.Q = Q


class AbstractClusterer(object):

    def cluster(self, affinity_mat):
        raise NotImplementedError()


class AbstractThresholder(object):

    def __call__(self, values):
        raise NotImplementedError()


def firstd(x_values, y_values):
    x_differences = x_values[1:] - x_values[:-1]
    x_midpoints = 0.5*(x_values[1:] + x_values[:-1])
    y_differences = y_values[1:] - y_values[:-1]
    rise_over_run = y_differences/x_differences
    return x_midpoints, rise_over_run


class CurvatureChangeAfterMax(AbstractThresholder):

    def __init__(self, bins):
        self.bins = bins

    def __call__(self, values):
        hist_y, hist_x = np.histogram(values, bins=self.bins)
        #get midpoints for hist_x
        hist_x = 0.5*(hist_x[:-1] + hist_x[1:])
        firstd_x, firstd_y = firstd(hist_x, hist_y)
        secondd_x, secondd_y = firstd(x_values=firstd_x, y_values=firstd_y)
        (x_first_neg_firstd, y_first_neg_firstd) =\
            next(x for x in zip(firstd_x, firstd_y) if x[1] < 0)
        (x_second_cross_0, y_secondd_cross_0) =\
            next(x for x in zip(secondd_x, secondd_y)
                 if x[0] > x_first_neg_firstd and x[1] >= 0)
        return x_second_cross_0


class AbstractAffMatPostProcessor(object):

    #binarizes an affinity matrix
    def __call__(self, affinity_mat):
        raise NotImplementedError()

    def chain(self, other_affmat_post_processor):
        return AdhocAffMatPostProcessor(
                func = lambda x: other_affmat_post_processor(self(x))) 


class AdhocAffMatPostProcessor(AbstractAffMatPostProcessor):

    def __init__(self, func):
        self.func = func 

    def __call__(self, affinity_mat):
        return self.func(affinity_mat)


class SimilarityToDistance(AbstractAffMatPostProcessor):

    def __call__(self, affinity_mat):
        return np.max(affinity_mat)-affinity_mat


class PerNodeThresholdBinarizer(AbstractAffMatPostProcessor):

    def __init__(self, thresholder, verbose=True):
        self.thresholder = thresholder
        self.verbose = verbose
    
    def __call__(self, affinity_mat):
        if (self.verbose):
            print("Starting thresholding preprocessing")
        start = time.time()
        thresholds = np.array([self.thresholder(x) for x in affinity_mat])
        to_return = (affinity_mat > thresholds[:,None]).astype("int") 
        if (self.verbose):
            print("Thresholding preproc took "+str(time.time()-start)+" s")
        return to_return


class JaccardSimCPU(AbstractAffMatPostProcessor):

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, affinity_mat):

        if (self.verbose):
            print("Starting Jaccard preprocessing via CPU matmul")
        start = time.time()
 
        #perform a sanity check to ensure max is 1 and min is 0
        assert np.max(affinity_mat)==1 and np.min(affinity_mat)==0,\
               ("max is "+str(np.max(affinity_mat))
                +" and min is "+str(np.min(affinity_mat)))
        intersections = np.dot(affinity_mat,
                               affinity_mat.transpose(1,0))
        one_minus_affinity_mat = 1 - affinity_mat
        unions_complement = np.dot(one_minus_affinity_mat,
                                   one_minus_affinity_mat.transpose(1,0))
        unions = len(affinity_mat) - unions_complement
        jaccard_sim = intersections.astype("float")/unions.astype("float")

        if (self.verbose):
            print("Jaccard preproc took "+str(time.time()-start)+" s")

        return jaccard_sim


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

    def __init__(self, affmat_preprocessor=None, min_cluster_size=10,
                       q_tol=1e-3, louvain_time_limit=2000, verbose=True):
        self.affmat_preprocessor = affmat_preprocessor
        self.min_cluster_size = min_cluster_size
        self.q_tol = q_tol
        self.louvain_time_limit = louvain_time_limit
        self.verbose = verbose
    
    def cluster(self, affinity_mat):
        if (self.verbose):
            print("Beginning preprocessing + Louvain")
        all_start = time.time()
        if (self.affmat_preprocessor is not None):
            affinity_mat = self.affmat_preprocessor(affinity_mat)
        communities, graph, Q, hierarchy = ph.cluster.runlouvain_given_graph(
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
 