from __future__ import division, print_function, absolute_import
import numpy as np
from .. import util as modiscoutil
from .. import core as modiscocore
from . import transformers
import sys
import time
import itertools
import scipy.stats
import gc
import sklearn
from joblib import Parallel, delayed
from tqdm import tqdm
from datetime import datetime

class AbstractTrackTransformer(object):
	def __init__(self, func=None):
		self.func = func

	def __call__(self, inp):
		return self.func(inp)

	def chain(self, other_normalizer):        
		return AbstractTrackTransformer(
				func=(lambda x: other_normalizer(
								self(x))))

class MeanNormalizer(AbstractTrackTransformer):
	def __call__(self, inp):
		return inp - np.mean(inp)


class MagnitudeNormalizer(AbstractTrackTransformer):
	def __call__(self, inp):
		return (inp / (np.linalg.norm(inp.ravel())+0.0000001))

class L1Normalizer(AbstractTrackTransformer):
	def __call__(self, inp):
		abs_sum = np.sum(np.abs(inp))
		if (abs_sum==0):
			return inp
		else:
			return (inp/abs_sum)

class PatternComparisonSettings(object):
	def __init__(self, track_names, track_transformer, min_overlap):
		assert hasattr(track_names, '__iter__')
		self.track_names = track_names
		self.track_transformer = track_transformer
		self.min_overlap = min_overlap


def sparse_cosine_similarity(fwd_vecs, rev_vecs, n_neighbors, 
	memory_cap_gb=1.0):
	#fwd_vecs2 is used when you don't just want to compute self-similarities

	#normalize the vectors 
	fwd_vecs = sklearn.preprocessing.normalize(fwd_vecs, norm='l2', axis=1)
	rev_vecs = sklearn.preprocessing.normalize(rev_vecs, norm='l2', axis=1)

	#assuming float64 for the affinity matrix, figure out the batch size
	# to use given the memory cap
	memory_cap_gb = (memory_cap_gb if rev_vecs is None else memory_cap_gb/2.0)
	batch_size = int(memory_cap_gb*(2**30)/(fwd_vecs.shape[0]*8))
	batch_size = min(max(1,batch_size),fwd_vecs.shape[0])

	k = min(n_neighbors+1, fwd_vecs.shape[0])

	neighbors, sims = [], []
	for i in tqdm(range(0, fwd_vecs.shape[0], batch_size)):
		fwd_vecs_slice = fwd_vecs[i:i+batch_size]

		fwd_dot = fwd_vecs_slice.dot(fwd_vecs.T).toarray()
		rev_dot = fwd_vecs_slice.dot(rev_vecs.T).toarray()
		dotprod = np.maximum(fwd_dot, rev_dot)

		dotprod_argsort = np.argsort(-dotprod, axis=-1) 

		for row_idx, argsort_row in enumerate(dotprod_argsort): 
			combined_neighbor_row = [] 
			neighbor_row_topnn = argsort_row[:k] 

			combined_neighbor_row.extend(neighbor_row_topnn) 

			neighbors.append(np.array(combined_neighbor_row).astype("int")) 
			sims.append(dotprod[row_idx][combined_neighbor_row])

	return np.array(sims), np.array(neighbors)

def _get_attr_from_seqlets(seqlets, track_names, transformer):
	fwd_data, rev_data = [], []

	for seqlet in seqlets:
		snippets = [seqlet[track_name] for track_name in track_names]

		fwd_data.append(np.concatenate([transformer(
				 np.reshape(snippet.fwd, (len(snippet.fwd), -1)))
				for snippet in snippets], axis=1))

		rev_data.append(np.concatenate([transformer(
				np.reshape(snippet.rev, (len(snippet.rev), -1)))
				for snippet in snippets], axis=1))

	return np.array(fwd_data), np.array(rev_data)


def AffmatFromSeqletsWithNNpairs(seqlets, track_names, transformer, min_overlap,
		filter_seqlets=None, seqlet_neighbors=None, 
		return_sparse=False, min_overlap_override=None, n_cores=1):

	all_fwd_data, all_rev_data = _get_attr_from_seqlets(seqlets=seqlets,
		track_names=track_names, transformer=transformer)

	if filter_seqlets is None:
		filter_seqlets = seqlets
		filters_all_fwd_data = all_fwd_data
		filters_all_rev_data = all_rev_data
	else:
		filters_all_fwd_data, filters_all_rev_data = _get_attr_from_seqlets(
			seqlets=filter_seqlets, track_names=track_names,
			transformer=transformer)

	if seqlet_neighbors is None:
		seqlet_neighbors = [list(range(len(filter_seqlets)))
							for x in seqlets] 

	min_overlap = (min_overlap 
		if min_overlap_override is None else min_overlap_override)

	#apply the cross metric
	affmat_fwd = ParallelCpuCrossMetricOnNNpairs(
		seqlet_neighbors=seqlet_neighbors, X=filters_all_fwd_data,
		Y=all_fwd_data, min_overlap=min_overlap, 
		return_sparse=return_sparse, n_cores=n_cores)

	affmat_rev = ParallelCpuCrossMetricOnNNpairs(
		seqlet_neighbors=seqlet_neighbors,
		X=filters_all_rev_data, Y=all_fwd_data,
		min_overlap=min_overlap, return_sparse=return_sparse,
		n_cores=n_cores) 

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


def ParallelCpuCrossMetricOnNNpairs(X, Y, min_overlap,
	seqlet_neighbors=None, return_sparse=False, n_cores=1, verbose=True):
	#n_pad = int(X.shape[1]*(1-min_overlap))
	#pad_width = ((0,0), (n_pad, n_pad), (0,0)) 
	#Y = np.pad(array=Y, pad_width=pad_width, mode="constant")

	f = delayed(jaccard)
	results = Parallel(n_jobs=n_cores, backend="threading")(
		(f(X[seqlet_neighbors[i]], Y[i], min_overlap, func=int) for i in range(len(Y))))

	if return_sparse == True:
		to_return = [x for x, _ in results]
	else:
		to_return = np.zeros((Y.shape[0], X.shape[0], 2))

		for i, (result, neighbor_indices) in enumerate(zip(results, seqlet_neighbors)):
			to_return[i, neighbor_indices] = np.transpose(result, (1,0))

	return to_return



def jaccard(X, Y, min_overlap=None, func=np.ceil):
	if X.ndim == 2:
		X = X[None, :, :]
	if Y.ndim == 2:
		Y = Y[None, :, :]

	if X.shape[1] > Y.shape[1]:
		X, Y = Y, X

	n_pad = 0
	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0, 0), (n_pad, n_pad), (0, 0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")

	n, d, _ = X.shape
	len_output = 1 + Y.shape[1] - d 
	scores = np.zeros((n, len_output))

	for idx in range(len_output):
		Y_ = Y[:, idx:idx+d]

		mins = np.minimum(np.abs(Y_), np.abs(X))
		signs = np.sign(Y_) * np.sign(X)
		maxs = np.maximum(np.abs(Y_), np.abs(X))

		scores_ = np.sum(mins*signs, axis=(1, 2)) / np.sum(maxs, axis=(1, 2))
		scores_ = np.nan_to_num(scores_)
		scores[:,idx] = scores_

	argmaxs = np.argmax(scores, axis=1)
	idxs = np.arange(len(scores))
	return np.array([scores[idxs, argmaxs], argmaxs - n_pad])

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
	return np.array([scores[idxs, argmaxs], argmaxs - n_pad])
