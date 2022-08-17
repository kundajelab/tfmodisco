import numpy as np
import sklearn
import sklearn.manifold
import scipy
from scipy.sparse import coo_matrix

class SymmetrizeByAddition():
    def __init__(self, probability_normalize=False):
        self.probability_normalize = probability_normalize

    def __call__(self, affinity_mat):
        to_return = affinity_mat + affinity_mat.T
        if (self.probability_normalize):
            to_return = to_return/np.sum(to_return).astype("float")
        return to_return

class AffToDistViaInvLogistic():
    def __call__(self, affinity_mat):
        to_return = np.log((1.0/(0.5*np.maximum(affinity_mat, 0.0000001)))-1)
        to_return = np.maximum(to_return, 0.0) #eliminate tiny neg floats
        return to_return

class NNTsneConditionalProbs():
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

    def tsne_probs_calc(self, distances_nn, neighbors_nn):

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

