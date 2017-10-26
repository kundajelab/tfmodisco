from __future__ import division, print_function, absolute_import
import numpy as np
import time
from modisco import util


class AbstractThresholder(object):

    def __call__(self, values):
        raise NotImplementedError()


class NonzeroMeanThreshold(AbstractThresholder):

    def __init__(self, expected_nonzeros=None):
        self.expected_nonzeros = expected_nonzeros 

    def __call__(self, values):
        if (self.expected_nonzeros is None):
            return np.sum(values)/np.sum(values > 0)
        else:
            return np.sum(values)/self.expected_nonzeros


class CurvatureBasedThreshold(AbstractThresholder):

    def __init__(self, bins):
        self.bins = bins

    def __call__(self, values):
        values = np.max(values)-values #convert similarity to distance
        droppped_zeros = [x for x in values if x != 0]
        hist_y, hist_x = np.histogram(droppped_zeros, bins=self.bins)
        cumsum = np.cumsum(hist_y)
        #get midpoints for hist_x
        hist_x = 0.5*(hist_x[:-1] + hist_x[1:])
        firstd_x, firstd_y = util.angle_firstd(hist_x, hist_y)
        secondd_x, secondd_y = util.firstd(x_values=firstd_x,
                                           y_values=firstd_y) 
        try:
            secondd_vals = [x for x in zip(secondd_x, secondd_y)]

            fastest_secondd_threshold =\
                max(secondd_vals, key=lambda x: x[1])[0]

            #find the first curvature change after the max
            (x_first_neg_firstd, y_first_neg_firstd) =\
                next(x for x in zip(firstd_x, firstd_y) if x[1] < 0)
            (x_second_cross_0, y_secondd_cross_0) =\
                next(x for x in zip(secondd_x, secondd_y)
                     if x[0] > x_first_neg_firstd and x[1] >= 0)

            if (fastest_secondd_threshold >= x_first_neg_firstd):
                #return the more conservative threshold
                return min(x_second_cross_0, fastest_secondd_threshold)
            else:
                return x_second_cross_0
        except StopIteration:
            return fastest_secondd_threshold 


class AbstractAffMatTransformer(object):

    #binarizes an affinity matrix
    def __call__(self, affinity_mat):
        raise NotImplementedError()

    def chain(self, other_affmat_post_processor):
        return AdhocAffMatTransformer(
                func = lambda x: other_affmat_post_processor(self(x))) 


class AdhocAffMatTransformer(AbstractAffMatTransformer):

    def __init__(self, func):
        self.func = func 

    def __call__(self, affinity_mat):
        return self.func(affinity_mat)


class PerNodeThresholdBinarizer(AbstractAffMatTransformer):

    def __init__(self, thresholder, verbose=True):
        self.thresholder = thresholder
        self.verbose = verbose
    
    def __call__(self, affinity_mat):
        if (self.verbose):
            print("Starting thresholding preprocessing")
        start = time.time()
        #ignore affinity to self
        thresholds = np.array([self.thresholder(x) for x in affinity_mat])
        thresholds = np.sum(affinity_mat,axis=1)/150
        to_return = (affinity_mat >= thresholds[:,None]).astype("int") 
        if (self.verbose):
            print("Thresholding preproc took "+str(time.time()-start)+" s")
        return to_return


class JaccardSimCPU(AbstractAffMatTransformer):

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
        jaccard_sim = intersections.astype("float")/(
                       unions.astype("float") + 0.0000001*(unions==0))

        if (self.verbose):
            print("Jaccard preproc took "+str(time.time()-start)+" s")

        return jaccard_sim


class SymmetrizeByMultiplying(AbstractAffMatTransformer):

    def __call__(self, affinity_mat):
        return affinity_mat*affinity_mat.T


class MinVal(AbstractAffMatTransformer):

    def __init__(self, min_val):
        self.min_val = min_val

    def __call__(self, affinity_mat):
        return affinity_mat*(affinity_mat >= self.min_val)


class TsneJointProbs(AbstractAffMatTransformer):

    def __init__(self, perplexity, verbose=1):
        self.perplexity = perplexity 
        self.verbose=verbose
    
    def __call__(self, affinity_mat):
        import sklearn
        import sklearn.manifold
        import scipy
        from scipy.sparse import csr_matrix
        from sklearn.neighbors import NearestNeighbors

        #make the affinity mat a distance mat
        dist_mat = (np.max(affinity_mat) - affinity_mat)
        dist_mat = sklearn.utils.check_array(dist_mat, ensure_min_samples=2,
                                             dtype=[np.float32, np.float64])
        n_samples = dist_mat.shape[0]


        #copied from https://github.com/scikit-learn/scikit-learn/blob/45dc891c96eebdb3b81bf14c2737d8f6540fabfe/sklearn/manifold/t_sne.py

        # Compute the number of nearest neighbors to find.
        # LvdM uses 3 * perplexity as the number of neighbors.
        # In the event that we have very small # of points
        # set the neighbors to n - 1.
        k = min(n_samples - 1, int(3. * self.perplexity + 1))

        if self.verbose:
            print("[t-SNE] Computing {} nearest neighbors...".format(k))

        # Find the nearest neighbors for every point
        knn = NearestNeighbors(algorithm='brute', n_neighbors=k,
                               metric='precomputed')
        t0 = time.time()
        knn.fit(dist_mat)
        duration = time.time() - t0
        if self.verbose:
            print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
                n_samples, duration))

        t0 = time.time()
        distances_nn, neighbors_nn = knn.kneighbors(
            None, n_neighbors=k)
        duration = time.time() - t0
        if self.verbose:
            print("[t-SNE] Computed neighbors for {} samples in {:.3f}s..."
                  .format(n_samples, duration))

        # Free the memory
        del knn

        P = sklearn.manifold.t_sne._joint_probabilities_nn(
                                    distances_nn, neighbors_nn,
                                    self.perplexity, self.verbose)
        return np.array(P.todense())


