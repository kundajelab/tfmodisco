import sklearn
import sklearn.manifold

import numpy as np

import scipy
from scipy.sparse import coo_matrix

from tqdm import tqdm
from numba import njit
from numba import prange

from . import core

class MagnitudeNormalizer():
	def __call__(self, inp):
		inp = inp - np.mean(inp)
		return (inp / (np.linalg.norm(inp.ravel())+0.0000001))

class L1Normalizer():
	def __call__(self, inp):
		abs_sum = np.sum(np.abs(inp))
		if (abs_sum==0):
			return inp
		else:
			return (inp/abs_sum)


from numba import njit
from numba import prange

@njit('double[:](float64[:], int64[:], int64[:], float64[:], int64[:], int64[:], int64)', parallel=True)
def _sparse_vm_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, d):
	dot = np.zeros(d)
	n_rows = len(Y_indptr) - 1

	for j in prange(n_rows):
		xj = 0
		yj = Y_indptr[j]

		while xj < X_indptr[-1] and yj < Y_indptr[j+1]:
			x_col = X_indices[xj]
			x_data = X_data[xj]

			y_col = Y_indices[yj]
			y_data = Y_data[yj]

			if x_col == y_col:
				dot[j] += x_data * y_data
				xj += 1
				yj += 1

			elif x_col < y_col:
				xj += 1

			else:
				yj += 1

	return dot

@njit('Tuple((float64[:, :], int32[:,:]))(float64[:], int64[:], int64[:], float64[:], int64[:], int64[:], int64, int64, int64)', parallel=True)
def _sparse_mm_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, n, d, n_neighbors):
	sims = np.empty((n, n_neighbors), dtype='float64')
	neighbors = np.empty((n, n_neighbors), dtype='int32')

	for i in prange(n):
		xdot = np.zeros(d, dtype='float32')
		ydot = np.zeros(d, dtype='float32')

		for j in range(n):
			xj = X_indptr[i]
			yj = X_indptr[j]

			while xj < X_indptr[i+1] and yj < X_indptr[j+1]:
				x_col = X_indices[xj]
				x_data = X_data[xj]

				y_col = X_indices[yj]
				y_data = X_data[yj]

				if x_col == y_col:
					xdot[j] += x_data * y_data
					xj += 1
					yj += 1

				elif x_col < y_col:
					xj += 1

				else:
					yj += 1

		for j in range(n):
			xj = X_indptr[i]
			yj = Y_indptr[j]

			while xj < X_indptr[i+1] and yj < Y_indptr[j+1]:
				x_col = X_indices[xj]
				x_data = X_data[xj]

				y_col = Y_indices[yj]
				y_data = Y_data[yj]

				if x_col == y_col:
					ydot[j] += x_data * y_data
					xj += 1
					yj += 1

				elif x_col < y_col:
					xj += 1

				else:
					yj += 1

		dot = np.maximum(xdot, ydot)
		neighbors[i] = np.argsort(-dot)[:n_neighbors]
		sims[i] = dot[neighbors[i]]

	return sims, neighbors


def sparse_cosine_similarity(X, Y, n_neighbors):
	#normalize the vectors 
	X = sklearn.preprocessing.normalize(X, norm='l2', axis=1)
	Y = sklearn.preprocessing.normalize(Y, norm='l2', axis=1)

	n, d = X.shape
	k = min(n_neighbors+1, n)

	sims = np.empty((n, k), dtype='float64')
	neighbors = np.empty((n, k), dtype='int32')

	for i in tqdm(range(n)):
		Xi = X[i]
		
		XX_dot = _sparse_vm_dot(Xi.data, Xi.indices, Xi.indptr, X.data, X.indices, X.indptr, X.shape[0])
		XY_dot = _sparse_vm_dot(Xi.data, Xi.indices, Xi.indptr, Y.data, Y.indices, Y.indptr, X.shape[0])

		dotprod = np.maximum(XX_dot, XY_dot) 
		dotprod_argsort = np.argsort(-dotprod) 

		neighbors[i] = dotprod_argsort[:k] 
		sims[i] = dotprod[dotprod_argsort[:k]]

	#import time

	#tic = time.time()
	#sims, neighbors = _sparse_mm_dot(X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, n, d, n_neighbors)
	#print(time.time() - tic, "c")

	return sims, neighbors


def jaccard_from_seqlets(seqlets, track_names, transformer, min_overlap,
		filter_seqlets=None, seqlet_neighbors=None, 
		return_sparse=False, min_overlap_override=None, n_cores=1):

	all_fwd_data, all_rev_data = core.get_2d_data_from_patterns(seqlets,
		track_names=track_names, track_transformer=transformer)

	if filter_seqlets is None:
		filter_seqlets = seqlets
		filters_all_fwd_data = all_fwd_data
		filters_all_rev_data = all_rev_data
	else:
		filters_all_fwd_data, filters_all_rev_data = core.get_2d_data_from_patterns(
			filter_seqlets, track_names=track_names,
			track_transformer=transformer)

	if seqlet_neighbors is None:
		seqlet_neighbors = [list(range(len(filter_seqlets)))
							for x in seqlets] 

	min_overlap = (min_overlap 
		if min_overlap_override is None else min_overlap_override)

	#apply the cross metric
	affmat_fwd = jaccard(seqlet_neighbors=seqlet_neighbors, 
		X=filters_all_fwd_data,
		Y=all_fwd_data, min_overlap=min_overlap, func=int, 
		return_sparse=return_sparse)

	affmat_rev = jaccard(seqlet_neighbors=seqlet_neighbors,
		X=filters_all_rev_data, Y=all_fwd_data,
		min_overlap=min_overlap, func=int, return_sparse=return_sparse) 

	if return_sparse == False:
		if len(affmat_fwd.shape) == 3:
			#will return something that's N x M x 3, where the third
			# entry in last dim is is_fwd 
			#is_fwd==False means the alignment and sim returned is
			# for the reverse-complement filter
			is_fwd = (affmat_fwd[:,:,0] > affmat_rev[:,:,0])*1.0
			affmat = np.zeros((affmat_fwd.shape[0],
							   affmat_fwd.shape[1],3))
			affmat[:,:,0:2] = (affmat_fwd*is_fwd[:,:,None]
							   + affmat_rev*(1-is_fwd[:,:,None]))
			affmat[:,:,2] = is_fwd 

		else:
			affmat = (np.maximum(affmat_fwd, affmat_rev) if
					  (affmat_rev is not None) else np.array(affmat_fwd))
	else:
		if (len(affmat_fwd[0].shape)==2):
			affmat = [] 
			for fwd, rev in zip(affmat_fwd, affmat_rev):
				is_fwd = (fwd[:,0] > rev[:,0])*1.0 
				new_row = np.zeros((fwd.shape[0],3))
				new_row[:,0:2] = (fwd*is_fwd[:,None] +
								  rev*(1-is_fwd[:,None]))
				new_row[:,2] = is_fwd 
				affmat.append(new_row)
		else:
			affmat = ([np.maximum(x,y) for (x,y)
					   in zip(affmat_fwd, affmat_rev)]
					  if affmat_rev is not None else affmat_fwd)
			 
	return np.array(affmat)  


def jaccard(X, Y, min_overlap=None, seqlet_neighbors=None, return_sparse=False, 
	func=np.ceil, verbose=True):

	if seqlet_neighbors is None:
		seqlet_neighbors = np.tile(np.arange(X.shape[0]), (Y.shape[0], 1))

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0,0), (n_pad, n_pad), (0,0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")
	else:
		n_pad = 0 

	X = X.astype('float32')
	Y = Y.astype('float32')
	seqlet_neighbors = seqlet_neighbors.astype('int32')
	len_output = 1 + Y.shape[1] - X.shape[1] 

	scores = np.zeros((Y.shape[0], seqlet_neighbors.shape[1], len_output), dtype='float32')
	_jaccard(X, Y, seqlet_neighbors, scores)
	scores = np.nan_to_num(scores)

	argmaxs = np.argmax(scores, axis=-1)
	idxs = np.arange(seqlet_neighbors.shape[1])

	results = np.zeros((Y.shape[0], 2, seqlet_neighbors.shape[1]))
	for i in range(Y.shape[0]):
		results[i, 0] = scores[i][idxs, argmaxs[i]]
		results[i, 1] = argmaxs[i] - n_pad


	if return_sparse == True:
		to_return = [x for x, _ in results]
	else:
		to_return = np.zeros((Y.shape[0], X.shape[0], 2))

		for i, (result, neighbor_indices) in enumerate(zip(results, seqlet_neighbors)):
			to_return[i, neighbor_indices] = np.transpose(result, (1,0))

	return to_return

@njit('void(float32[:, :, :], float32[:, :, :], int32[:, :], float32[:, :, :])', parallel=True)
def _jaccard(X, Y, neighbors, scores):
	X_abs = np.abs(X)
	Y_abs = np.abs(Y)

	X_sign = np.sign(X)
	Y_sign = np.sign(Y)

	nx, d, m = X.shape
	ny = Y.shape[0]
	len_output = scores.shape[-1]

	for l in prange(ny):
		for idx in range(len_output):
			for i in range(neighbors.shape[1]):
				min_sum = 0.0
				max_sum = 0.0
				neighbor_li = neighbors[l, i]

				for j in range(idx, idx+d):
					j_idx = j - idx

					for k in range(m):
						sign = X_sign[neighbor_li, j_idx, k] * Y_sign[l, j, k]

						x = X_abs[neighbor_li, j_idx, k]
						y = Y_abs[l, j, k]

						if y > x:
							min_sum += x * sign
							max_sum += y
						else:
							min_sum += y * sign
							max_sum += x

				scores[l, i, idx] = min_sum / max_sum



def pearson_correlation(X, Y, min_overlap=None, func=np.ceil):
	if X.ndim == 2:
		X = X[None, :, :]
	if Y.ndim == 2:
		Y = Y[None, :, :]

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0, 0), (n_pad, n_pad), (0, 0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")

	n, d, _ = X.shape
	len_output = 1 + Y.shape[1] - d 
	scores = np.zeros((n, len_output))

	for idx in range(len_output):
		Y_ = Y[:, idx:idx+d]

		scores_ = np.dot((X / np.linalg.norm(X)).ravel(),
                  (Y_ / np.linalg.norm(Y_)).ravel()) 
		scores_ = np.nan_to_num(scores_)
		scores[:,idx] = scores_

	argmaxs = np.argmax(scores, axis=1)
	idxs = np.arange(len(scores))
	return np.array([[scores[idxs, argmaxs], argmaxs - n_pad]]).transpose(0, 2, 1)

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


        distmat_nn = self.aff_to_dist_mat(
                        affinity_mat=affinity_mat) 
        #assert that the distances are increasing to the right


        # Compute the number of nearest neighbors to find.
        # LvdM uses 3 * perplexity as the number of neighbors.
        # In the event that we have very small # of points
        # set the neighbors to n - 1.
        n_samples = distmat_nn.shape[0]
        k = min(n_samples - 1, int(3. * self.perplexity + 1))

        P = self.tsne_probs_calc(distances_nn=distmat_nn[:,1:(k+1)],
                                 neighbors_nn=[row[1:(k+1)] for row in 
                                               nearest_neighbors])
        return P

    def tsne_probs_calc(self, distances_nn, neighbors_nn):

        # Compute conditional probabilities such that they approximately match
        # the desired perplexity
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

