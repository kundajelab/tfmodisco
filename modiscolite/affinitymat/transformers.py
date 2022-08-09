from __future__ import division, print_function, absolute_import
import numpy as np
import time
from modisco import util
import sklearn
import sklearn.manifold
import scipy
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.neighbors import NearestNeighbors
import sys
from ..cluster import phenograph as ph


class AbstractThresholder(object):

    def __call__(self, values):
        raise NotImplementedError()


class FixedValueThreshold(AbstractThresholder):

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, values=None):
        return self.threshold


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
            sys.stdout.flush()
            
        start = time.time()
        #ignore affinity to self
        thresholds = np.array([self.thresholder(x) for x in affinity_mat])
        to_return = (affinity_mat >= thresholds[:,None]).astype("int") 
        if (self.verbose):
            print("Thresholding preproc took "+str(time.time()-start)+" s")
            sys.stdout.flush()
        return to_return


class NearestNeighborsBinarizer(AbstractAffMatTransformer):

    def __init__(self, n_neighbors, nearest_neighbors_object):
        self.nearest_neighbors_object = nearest_neighbors_object 
        self.n_neighbors = n_neighbors

    def __call__(self, affinity_mat):
        seqlet_neighbors = (self.nearest_neighbors_object.fit(-affinity_mat).
                                 kneighbors(X=-affinity_mat,
                                            n_neighbors=self.n_neighbors,
                                            return_distance=False)) 
        to_return = np.zeros_like(affinity_mat)
        for i, neighbors in enumerate(seqlet_neighbors):
            to_return[i,neighbors] = 1 
        return to_return 


class ProductOfTransformations(AbstractAffMatTransformer):

    def __init__(self, transformer1, transformer2):
        self.transformer1 = transformer1
        self.transformer2 = transformer2

    def __call__(self, affinity_mat):
        return self.transformer1(affinity_mat)*self.transformer2(affinity_mat)


class JaccardSimCPU(AbstractAffMatTransformer):

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, affinity_mat):

        if (self.verbose):
            print("Starting Jaccard preprocessing via CPU matmul")
            sys.stdout.flush()
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
            sys.stdout.flush()

        return jaccard_sim


class SymmetrizeByElemwiseGeomMean(AbstractAffMatTransformer):

    def __call__(self, affinity_mat):
        return np.sqrt(affinity_mat*affinity_mat.T)


class SymmetrizeByElemwiseMultiplying(AbstractAffMatTransformer):

    def __call__(self, affinity_mat):
        return affinity_mat*affinity_mat.T


class SymmetrizeByAddition(AbstractAffMatTransformer):

    def __init__(self, probability_normalize=False):
        self.probability_normalize = probability_normalize

    def __call__(self, affinity_mat):
        to_return = affinity_mat + affinity_mat.T
        if (self.probability_normalize):
            to_return = to_return/np.sum(to_return).astype("float")
        return to_return


class MinVal(AbstractAffMatTransformer):

    def __init__(self, min_val):
        self.min_val = min_val

    def __call__(self, affinity_mat):
        return affinity_mat*(affinity_mat >= self.min_val)


class DistToSymm(AbstractAffMatTransformer):

    def __call__(self, affinity_mat):
        return np.max(affinity_mat)-affinity_mat


class ApplyTransitions(AbstractAffMatTransformer):

    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __call__(self, affinity_mat):
        return np.dot(np.linalg.matrix_power(affinity_mat.T,
                                             self.num_steps),affinity_mat)


class AbstractAffToDistMat(object):

    def __call__(self, affinity_mat):
        raise NotImplementedError()


class MaxToMin(AbstractAffToDistMat):

    def __call__(self, affinity_mat):
        return (np.max(affinity_mat) - affinity_mat)


class AffToDistViaInvLogistic(AbstractAffToDistMat):

    def __call__(self, affinity_mat):
        to_return = np.log((1.0/
                           (0.5*np.maximum(affinity_mat, 0.0000001)))-1)
        to_return = np.maximum(to_return, 0.0) #eliminate tiny neg floats
        return to_return


class AbstractNNTsneProbs(AbstractAffMatTransformer):

    def __init__(self, perplexity, aff_to_dist_mat, verbose=1):
        self.perplexity = perplexity 
        self.verbose=verbose
        self.aff_to_dist_mat = aff_to_dist_mat

    def __call__(self, affinity_mat, nearest_neighbors):

        #pad the affmat and the nearest neighbors so they all have the
        # same length...
        max_neighbors = max([len(x) for x in nearest_neighbors])
        affinity_mat = np.array([list(row)+[-np.inf for x in
                                 range(max_neighbors-len(row))]
                                 for row in affinity_mat])
        nearest_neighbors = [list(row)+[np.nan for x in
                             range(max_neighbors-len(row))]
                             for row in nearest_neighbors]

        #assert that affinity_mat as the same dims as nearest_neighbors
        assert affinity_mat.shape==(len(nearest_neighbors),
                                    len(nearest_neighbors[0]))
        #assert all rows of nearest_neighbors have the same length (i.e.
        # they have been padded)
        assert len(set([len(x) for x in nearest_neighbors]))==1

        distmat_nn = self.aff_to_dist_mat(
                        affinity_mat=affinity_mat) 
        #assert that the distances are increasing to the right
        assert np.min(distmat_nn[:,1:] - distmat_nn[:,:-1]) >= 0.0
        #assert that the self-distances are 0
        assert np.min(distmat_nn[:,0])==0
        assert np.max(distmat_nn[:,0])==0
        #assert that each idx is its own nearest neighbor
        if not all([i==nearest_neighbors[i][0] for
                    i in range(len(nearest_neighbors))]):
            print("Warning: each seqlet is not its own nearest neighbor;"
                  +" there may be duplicates")
        # Compute the number of nearest neighbors to find.
        # LvdM uses 3 * perplexity as the number of neighbors.
        # In the event that we have very small # of points
        # set the neighbors to n - 1.
        n_samples = distmat_nn.shape[0]
        k = min(n_samples - 1, int(3. * self.perplexity + 1))
        assert k < distmat_nn.shape[1],(
            "Not enough neighbors for perplexity calc! Need over"
            +" "+str(k)+" but have "+str(distmat_nn.shape[1]))
        P = self.tsne_probs_calc(distances_nn=distmat_nn[:,1:(k+1)],
                                 neighbors_nn=[row[1:(k+1)] for row in 
                                               nearest_neighbors])
        return P


class NNTsneConditionalProbs(AbstractNNTsneProbs):

    def tsne_probs_calc(self, distances_nn, neighbors_nn):
        t0 = time.time()
        # Compute conditional probabilities such that they approximately match
        # the desired perplexity
        assert len(set([len(row) for row in neighbors_nn]))==1
        n_samples, k = len(neighbors_nn),len(neighbors_nn[0])
        distances = distances_nn.astype(np.float32, copy=False)
        neighbors = neighbors_nn
        try:
            conditional_P = sklearn.manifold._utils._binary_search_perplexity(
                distances, np.array(neighbors).astype("int"),
                self.perplexity, self.verbose)
        except:
            #API changed in v0.22 to not require the redundant neighbors
            # argument
            conditional_P = sklearn.manifold._utils._binary_search_perplexity(
                distances, self.perplexity, self.verbose)
        #for some reason, likely a sklearn bug, a few of
        #the rows don't sum to 1...for now, fix by making them sum to 1
        assert np.all(np.isfinite(conditional_P)), \
            "All probabilities should be finite"
        #normalize the conditional_P to sum to 1 across the rows
        conditional_P = conditional_P/np.sum(conditional_P, axis=-1)[:,None]

        data = []
        rows = []
        cols = []
        for row_idx,(ps,neigh_row) in enumerate(zip(conditional_P, neighbors)):
            data.extend([p for p,neighbor in zip(ps, neigh_row)
                         if np.isnan(neighbor)==False])
            rows.extend([row_idx for neighbor in neigh_row
                         if np.isnan(neighbor)==False])
            cols.extend([neighbor for neighbor in neigh_row
                         if np.isnan(neighbor)==False])
        P = coo_matrix((data, (rows, cols)),
                       shape=(len(neighbors), len(neighbors)))
        return P


class AbstractTsneProbs(AbstractAffMatTransformer):

    def __init__(self, perplexity, aff_to_dist_mat, verbose=1):
        self.perplexity = perplexity 
        self.verbose=verbose
        self.aff_to_dist_mat = aff_to_dist_mat
    
    def __call__(self, affinity_mat):

        #make the affinity mat a distance mat
        dist_mat = self.aff_to_dist_mat(affinity_mat)
        #make sure self-distances are 0
        dist_mat = dist_mat*(1-np.eye(len(dist_mat)))
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

        P = self.tsne_probs_calc(distances_nn, neighbors_nn)
        return P

    def tsne_probs_calc(self, distances_nn, neighbors_nn):
        raise NotImplementedError()


class TsneConditionalProbs(AbstractTsneProbs):

    def tsne_probs_calc(self, distances_nn, neighbors_nn):
        t0 = time.time()
        # Compute conditional probabilities such that they approximately match
        # the desired perplexity
        n_samples, k = neighbors_nn.shape
        distances = distances_nn.astype(np.float32, copy=False)
        neighbors = neighbors_nn.astype(np.int64, copy=False)
        try:
            conditional_P = sklearn.manifold._utils._binary_search_perplexity(
                distances, neighbors, self.perplexity, self.verbose)
        except:
            #API changed in v0.22 to not require the redundant neighbors
            # argument
            conditional_P = sklearn.manifold._utils._binary_search_perplexity(
                distances, self.perplexity, self.verbose)
        #for some reason, likely a sklearn bug, a few of
        #the rows don't sum to 1...for now, fix by making them sum to 1
        #print(np.sum(np.sum(conditional_P, axis=1) > 1.1))
        #print(np.sum(np.sum(conditional_P, axis=1) < 0.9))
        assert np.all(np.isfinite(conditional_P)), \
            "All probabilities should be finite"

        # Symmetrize the joint probability distribution using sparse operations
        P = csr_matrix((conditional_P.ravel(), neighbors.ravel(),
                        range(0, n_samples * k + 1, k)),
                       shape=(n_samples, n_samples))
        to_return = np.array(P.todense())
        to_return = to_return/np.sum(to_return,axis=1)[:,None]
        return to_return


class TsneJointProbs(AbstractTsneProbs):

    def tsne_probs_calc(self, distances_nn, neighbors_nn):
        P = sklearn.manifold.t_sne._joint_probabilities_nn(
                                    distances_nn, neighbors_nn,
                                    self.perplexity, self.verbose)
        return np.array(P.todense())


class LouvainMembershipAverage(AbstractAffMatTransformer):

    def __init__(self, n_runs, level_to_return, parallel_threads,
                 verbose=True, seed=1234):
        self.n_runs = n_runs
        self.level_to_return = level_to_return
        self.parallel_threads = parallel_threads
        self.verbose = verbose
        self.seed=seed
    
    def __call__(self, affinity_mat):

        return ph.cluster.runlouvain_average_runs_given_graph(
                graph=affinity_mat,
                n_runs=self.n_runs, level_to_return=self.level_to_return,
                parallel_threads=self.parallel_threads,
                seed=self.seed,
                verbose=self.verbose)
