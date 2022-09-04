# util.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import numpy as np
from numba import njit

def cpu_sliding_window_sum(arr, window_size):
	to_return = np.zeros(len(arr)-window_size+1)
	current_sum = np.sum(arr[0:window_size])
	to_return[0] = current_sum
	idx_to_include = window_size
	idx_to_exclude = 0
	while idx_to_include < len(arr):
		current_sum += (arr[idx_to_include] - arr[idx_to_exclude]) 
		to_return[idx_to_exclude+1] = current_sum
		idx_to_include += 1
		idx_to_exclude += 1
	return to_return


@njit('float64(float64, float64[:])')
def binary_search_perplexity(desired_perplexity, distances):
	EPSILON_DBL = 1e-8
	PERPLEXITY_TOLERANCE = 1e-5
	n_steps = 100
	
	desired_entropy = np.log(desired_perplexity)
	
	beta_min = -np.inf
	beta_max = np.inf
	beta = 1.0
	
	for l in range(n_steps):
		ps = np.exp(-distances * beta)
		sum_ps = np.sum(ps) + 1
		ps = ps/(max(sum_ps,EPSILON_DBL))
		sum_disti_Pi = np.sum(distances*ps)
		entropy = np.log(sum_ps) + beta * sum_disti_Pi
		
		entropy_diff = entropy - desired_entropy
		if np.abs(entropy_diff) <= PERPLEXITY_TOLERANCE:
			break
		
		if entropy_diff > 0.0:
			beta_min = beta
			if beta_max == np.inf:
				beta *= 2.0
			else:
				beta = (beta + beta_max) / 2.0
		else:
			beta_max = beta
			if beta_min == -np.inf:
				beta /= 2.0
			else:
				beta = (beta + beta_min) / 2.0
	
	return beta


def compute_per_position_ic(ppm, background, pseudocount):
	"""Compute information content at each position of ppm.

	Arguments:
		ppm: should have dimensions of length x alphabet. Entries along the
			alphabet axis should sum to 1.
		background: the background base frequencies
		pseudocount: pseudocount to be added to the probabilities of the ppm
			to prevent overflow/underflow.

	Returns:
		total information content at each positon of the ppm.
	"""

	if (not np.allclose(np.sum(ppm, axis=1), 1.0, atol=1.0e-5)):
		ppm = ppm/np.sum(ppm, axis=1)[:,None]

	alphabet_len = len(background)
	ic = ((np.log((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/np.log(2))
		  *ppm - (np.log(background)*background/np.log(2))[None,:])
	return np.sum(ic,axis=1)


#rolling_window is from this blog post by Erik Rigtorp:
# https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
#The last axis of a will be subject to the windowing
def rolling_window(a, window):
	shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
	strides = a.strides + (a.strides[-1],)
	return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def magnitude(X):
	X = X - np.mean(X)
	return (X / (np.linalg.norm(X.ravel())+0.0000001))

def l1(X):
	abs_sum = np.sum(np.abs(X))
	if abs_sum == 0:
		return X
	return (X/abs_sum)

def get_2d_data_from_patterns(patterns, transformer='l1', include_hypothetical=True):
	func = l1 if transformer == 'l1' else magnitude
	tracks = ['hypothetical_contribs', 'contrib_scores']
	if not include_hypothetical:
		tracks = tracks[1:]

	all_fwd_data, all_rev_data = [], []

	for pattern in patterns:
		snippets = [getattr(pattern, track) for track in tracks]

		fwd_data = np.concatenate([func(snippet) for snippet in snippets], axis=1)
		rev_data = np.concatenate([func(snippet[::-1, ::-1]) for snippet in snippets], axis=1)

		all_fwd_data.append(fwd_data)
		all_rev_data.append(rev_data)

	return np.array(all_fwd_data), np.array(all_rev_data)
