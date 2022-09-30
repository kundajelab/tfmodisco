# affinitymat.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import sklearn
import sklearn.manifold

import numpy as np

import scipy
from scipy.sparse import coo_matrix

from numba import njit
from numba import prange

from . import util
from . import gapped_kmer


@njit('float64(float64[:], int64[:], int64[:], float64[:], int64[:], int64[:], int64, int64)')
def _sparse_vv_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, i, j):
	xi = X_indptr[i]
	yj = Y_indptr[j]
	dot = 0.0

	while xi < X_indptr[i+1] and yj < Y_indptr[j+1]:
		x_col = X_indices[xi]
		x_data = X_data[xi]

		y_col = Y_indices[yj]
		y_data = Y_data[yj]

		if x_col == y_col:
			dot += x_data * y_data
			xi += 1
			yj += 1

		elif x_col < y_col:
			xi += 1

		else:
			yj += 1

	return dot

@njit(parallel=True)
def _sparse_mm_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, k):
	n_rows = len(Y_indptr) - 1

	neighbors = np.empty((n_rows, k), dtype='int32')
	sims = np.empty((n_rows, k), dtype='float64')

	for i in prange(n_rows):
		dot = np.zeros(n_rows, dtype='float64')

		for j in range(n_rows):
			xdot = _sparse_vv_dot(X_data, X_indices, X_indptr, X_data, X_indices, X_indptr, i, j)
			ydot = _sparse_vv_dot(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, i, j)
			dot[j] = max(xdot, ydot)

		dot_argsort = np.argsort(-dot, kind='mergesort')[:k]
		neighbors[i] = dot_argsort
		sims[i] = dot[dot_argsort]

	return sims, neighbors

def cosine_similarity_from_seqlets(seqlets, n_neighbors, sign, topn=20, 
	min_k=4, max_k=6, max_gap=15, max_len=15, max_entries=500, 
	alphabet_size=4):

	X_fwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, 
		min_k, max_k, max_gap, max_len, max_entries, True, sign)

	X_bwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, min_k, max_k, max_gap, 
			max_len, max_entries, False, sign)

	X = sklearn.preprocessing.normalize(X_fwd, norm='l2', axis=1)
	Y = sklearn.preprocessing.normalize(X_bwd, norm='l2', axis=1)

	n, d = X.shape
	k = min(n_neighbors+1, n)
	return _sparse_mm_dot(X.data, X.indices, X.indptr, Y.data, Y.indices, Y.indptr, k)


def jaccard_from_seqlets(seqlets, min_overlap, filter_seqlets=None, 
	seqlet_neighbors=None):

	all_fwd_data, all_rev_data = util.get_2d_data_from_patterns(seqlets)

	if filter_seqlets is None:
		filter_seqlets = seqlets
		filters_all_fwd_data = all_fwd_data
		filters_all_rev_data = all_rev_data
	else:
		filters_all_fwd_data, filters_all_rev_data = util.get_2d_data_from_patterns(filter_seqlets)

	if seqlet_neighbors is None:
		seqlet_neighbors = [list(range(len(filter_seqlets)))
							for x in seqlets] 

	#apply the cross metric
	affmat_fwd = jaccard(seqlet_neighbors=seqlet_neighbors, 
		X=filters_all_fwd_data,
		Y=all_fwd_data, min_overlap=min_overlap, func=int, 
		return_sparse=True)

	affmat_rev = jaccard(seqlet_neighbors=seqlet_neighbors,
		X=filters_all_rev_data, Y=all_fwd_data,
		min_overlap=min_overlap, func=int,
		return_sparse=True) 

	affmat = np.maximum(affmat_fwd, affmat_rev)
	return affmat


def jaccard(X, Y, min_overlap=None, seqlet_neighbors=None, func=np.ceil, 
	return_sparse=False):

	if seqlet_neighbors is None:
		seqlet_neighbors = np.tile(np.arange(X.shape[0]), (Y.shape[0], 1))

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0,0), (n_pad, n_pad), (0,0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")
	else:
		n_pad = 0 

	len_output = 1 + Y.shape[1] - X.shape[1] 

	X = X.astype('float32')
	Y = Y.astype('float32')

	seqlet_neighbors = seqlet_neighbors.astype('int32')
	scores = np.zeros((Y.shape[0], seqlet_neighbors.shape[1], len_output), dtype='float32')
	_jaccard(X, Y, seqlet_neighbors, scores)

	if return_sparse == True:
		return scores.max(axis=-1)

	argmaxs = np.argmax(scores, axis=-1)
	idxs = np.arange(seqlet_neighbors.shape[1])
	results = np.zeros((Y.shape[0], seqlet_neighbors.shape[1], 2))
	for i in range(Y.shape[0]):
		results[i, :, 0] = scores[i][idxs, argmaxs[i]]
		results[i, :, 1] = argmaxs[i] - n_pad

	return results

@njit(parallel=True)
def pairwise_jaccard(X, k):
	n, m = X.shape

	jaccards = np.empty((n, k), dtype='float64')
	neighbors = np.empty((n, k), dtype='int32')

	for i in prange(n):
		jaccard_ = np.empty(n, dtype='float64')

		for j in range(n):
			min_sum = 0.0
			max_sum = 0.0

			for l in range(m):
				sign = np.sign(X[i, l]) * np.sign(X[j, l])
				xi = abs(X[i, l])
				xj = abs(X[j, l])

				if xi > xj:
					min_sum += xj * sign
					max_sum += xi
				else:
					min_sum += xi * sign
					max_sum += xj 

			jaccard_[j] = min_sum / max_sum

		idxs = np.argsort(-jaccard_, kind='mergesort')[:k]

		jaccards[i] = jaccard_[idxs]
		neighbors[i] = idxs

	return jaccards, neighbors


@njit('void(float32[:, :, :], float32[:, :, :], int32[:, :], float32[:, :, :])', parallel=True)
def _jaccard(X, Y, neighbors, scores):
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
						sign = np.sign(X[neighbor_li, j_idx, k]) * np.sign(Y[l, j, k])

						x = abs(X[neighbor_li, j_idx, k])
						y = abs(Y[l, j, k])

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


class NNTsneConditionalProbs():
	def __init__(self, perplexity):
		self.perplexity = perplexity 

	def __call__(self, affinity_mat, nearest_neighbors):
		distmat_nn = np.log((1.0/(0.5*np.maximum(affinity_mat, 0.0000001)))-1)
		distmat_nn = np.maximum(distmat_nn, 0.0) #eliminate tiny neg floats

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
		
		conditional_P = sklearn.manifold._utils._binary_search_perplexity(
			distances, self.perplexity, verbose=False)

		eps = 1e-8
		marginal_sum = conditional_P.sum(axis=-1)
		marginal_sum[marginal_sum < eps] = eps

		#normalize the conditional_P to sum to 1 across the rows
		conditional_P = conditional_P / marginal_sum[:,None]

		data = []
		rows = []
		cols = []
		for row_idx,(ps,neigh_row) in enumerate(zip(conditional_P, neighbors)):
			data.extend([p for p,neighbor in zip(ps, neigh_row)])
			rows.extend([row_idx for neighbor in neigh_row])
			cols.extend([neighbor for neighbor in neigh_row])

		P = coo_matrix((data, (rows, cols)),
					   shape=(len(neighbors), len(neighbors)))
		return P

