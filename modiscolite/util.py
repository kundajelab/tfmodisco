# util.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>, Ivy Raine <ivy.ember.raine@gmail.com>
# adapted from code written by Avanti Shrikumar 

from enum import Enum
import textwrap
from typing import List
import numpy as np
from numba import njit


class MemeDataType(Enum):
	PFM = "PFM"
	CWM = "CWM"
	hCWM = "hCWM"
	CWM_PFM = "CWM-PFM"
	hCWM_PFM = "hCWM-PFM"

	def __str__(self):
		return self.value



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


def calculate_window_offsets(center: int, window_size: int) -> tuple:
	return (center - window_size // 2, center + window_size // 2)


def filter_bed_rows_by_chrom(peak_rows: List[str], valid_chroms: List[str]):
	"""This function filters the rows of a bed file by chromosome.
	
	Parameters:
		peak_rows: list of strings, where each string is a row of a bed file.
		valid_chroms: list of chars, where each string is a valid chromosome.
			Example: ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'X']

	Returns:
		list of row strings, where each sublist is a row of a bed file.

	Usage:
		>>> peak_rows = ['chr1\t1\t2\tpeak1', 'chr2\t1\t2\tpeak2']
		>>> valid_chroms = ['1']
		>>> result = filter_bed_rows_by_chrom(peak_rows, valid_chroms)
	"""
	try:
		return [row for row in peak_rows if row.split('\t')[0] in valid_chroms]
	except IndexError:
		raise IndexError(textwrap.dedent(f'''\
			An error occurred while processing the BED file rows.
			Verify that the BED file is correct.'''))
