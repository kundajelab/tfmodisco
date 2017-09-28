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


class CurvatureBasedThreshold(AbstractThresholder):

    def __init__(self, bins):
        self.bins = bins

    def __call__(self, values):
        droppped_zeros = [x for x in values if x != 0]
        hist_y, hist_x = np.histogram(droppped_zeros, bins=self.bins)
        cumsum = np.cumsum(hist_y)
        #get midpoints for hist_x
        hist_x = 0.5*(hist_x[:-1] + hist_x[1:])
        median_x = next(hist_x[i] for i in range(len(hist_x)) if
                        (cumsum[i] > len(values)*0.5)) 
        firstd_x, firstd_y = firstd(hist_x, hist_y)
        secondd_x, secondd_y = firstd(x_values=firstd_x, y_values=firstd_y) 
        try:

            #look at the fastest curvature change for the secondd vals
            #below the median
            secondd_vals_below_median = [x for x in zip(secondd_x, secondd_y)
                                         if x[0] < median_x]
            fastest_secondd_threshold =\
                max(secondd_vals_below_median, key=lambda x: x[1])[0]

            #if the median is concentrated at the first bar, this if condition
            #will be triggered
            if (len(secondd_vals_below_median)==0):
                return 0

            #find the first curvature change after the max
            (x_first_neg_firstd, y_first_neg_firstd) =\
                next(x for x in zip(firstd_x, firstd_y) if x[1] < 0)
            (x_second_cross_0, y_secondd_cross_0) =\
                next(x for x in zip(secondd_x, secondd_y)
                     if x[0] > x_first_neg_firstd and x[1] >= 0
                     and x[0] < median_x)

            #return the more conservative threshold
            return min(x_second_cross_0, fastest_secondd_threshold)

        except StopIteration:
            return fastest_secondd_threshold 


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
        #ignore affinity to self
        affinity_mat_zero_d = affinity_mat*(1-np.eye(len(affinity_mat)))
        thresholds = np.array([self.thresholder(x)
                               for x in affinity_mat_zero_d])
        to_return = (affinity_mat <= thresholds[:,None]).astype("int") 
        #each node is attached to itself
        to_return = np.maximum(to_return, np.eye(len(affinity_mat)))
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


class MinVal(AbstractAffMatPostProcessor):

    def __init__(self, min_val):
        self.min_val = min_val

    def __call__(self, affinity_mat):
        return affinity_mat*(affinity_mat >= self.min_val)


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
                       q_tol=1e-3, louvain_time_limit=2000,
                       verbose=True, min_nonneg=0, max_nonneg=np.inf,
                       second_preprocessor = None):
        self.affmat_preprocessor = affmat_preprocessor
        self.min_cluster_size = min_cluster_size
        self.q_tol = q_tol
        self.louvain_time_limit = louvain_time_limit
        self.verbose = verbose
        self.min_nonneg = min_nonneg
        self.max_nonneg = max_nonneg
        self.second_preprocessor = second_preprocessor
    
    def cluster(self, orig_affinity_mat):

        if (self.verbose):
            print("Beginning preprocessing + Louvain")
        all_start = time.time()
        if (self.affmat_preprocessor is not None):
            affinity_mat = self.affmat_preprocessor(orig_affinity_mat)
        else:
            affinity_mat = orig_affinity_mat

        #subset the affinity mat to rows that have at least
        #self.min_nonneg nonnegative connections; the ones that don't pass
        #this filter as are assigned a cluster of -1        
        #the -1 is to discount the self-connection
        num_connected_neighbors = np.sum(affinity_mat > 0, axis=1)-1
        filtered_mask = ((num_connected_neighbors >= self.min_nonneg)*
                         (num_connected_neighbors <= self.max_nonneg))

        if (self.verbose):
            print(str(sum(filtered_mask))+" points remain after subsetting")

        if (self.second_preprocessor is not None):
            if (self.verbose):
                print("Beginning second round preprocessing")
            subsetted_affmat = orig_affinity_mat[filtered_mask]
            subsetted_affmat = subsetted_affmat[:,filtered_mask]
            subsetted_affmat = self.second_preprocessor(subsetted_affmat)
        else:
            subsetted_affmat = affinity_mat[filtered_mask]
            subsetted_affmat = subsetted_affmat[:,filtered_mask]

        subset_communities, graph, Q, hierarchy =\
            ph.cluster.runlouvain_given_graph(
                graph=subsetted_affmat,
                min_cluster_size=self.min_cluster_size,
                q_tol=self.q_tol,
                louvain_time_limit=self.louvain_time_limit)

        #fill in the communities that are -1
        communities = (np.ones(len(affinity_mat))*-1).astype("int")
        communities[filtered_mask] = subset_communities

        cluster_results = LouvainClusterResults(
                cluster_indices=communities,
                hierarchy=hierarchy,
                Q=Q)
        if (self.verbose):
            print("Preproc + Louvain took "+str(time.time()-all_start)+" s")
        return cluster_results
 
