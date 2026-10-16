# extract_seqlets.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import os
import math
import mmap
import signal
import warnings

from concurrent.futures import ThreadPoolExecutor

import numba
import numpy as np

from . import core
from . import precompile

def _bin_mode(values, bins=1000):
	hist, bin_edges = np.histogram(values, bins=bins)
	peak = np.argmax(hist)
	l_edge = bin_edges[peak]
	r_edge = bin_edges[peak+1]
	return l_edge, r_edge, values[(l_edge < values) & (values < r_edge)]

def _laplacian_null(track, window_size, num_to_samp, random_seed=1234):
	percentiles_to_use = np.array([5*(x+1) for x in range(19)])

	rng = np.random.RandomState()
	values = track.ravel()

	# first estimate mu, using two level histogram to get to 1e-6
	_, _, top_values = _bin_mode(values)
	l_edge, r_edge, _ = _bin_mode(top_values)
	mu = (l_edge + r_edge) / 2

	# The negative side's values and percentiles are found on a thread, when
	# numba's thread count allows one, while this thread finds the positive
	# side's: numpy's masking, copying and partitioning release the GIL, and
	# each side's arithmetic is unchanged.
	def negative_side():
		neg_values = values[values <=mu] 
		return len(neg_values), np.percentile(a=neg_values,
			q=100-percentiles_to_use)

	with ThreadPoolExecutor(1) as pool:
		negative = (pool.submit(negative_side) if numba.get_num_threads() > 1
			else None)
		pos_values = values[values >= mu]
		pos_percentiles = np.percentile(a=pos_values, q=percentiles_to_use)
		n_neg_values, neg_percentiles = (negative_side() if negative is None
			else negative.result())

	#Take the most aggressive lambda over all percentiles
	pos_laplace_lambda = np.max(
		-np.log(1-(percentiles_to_use/100.0))/
		(pos_percentiles-mu))

	neg_laplace_lambda = np.max(
		-np.log(1-(percentiles_to_use/100.0))/
		(np.abs(neg_percentiles-mu)))

	rng.seed(random_seed)
	prob_pos = float(len(pos_values))/(len(pos_values)+n_neg_values) 

	# Each sample draws two uniforms, the sign's and then the CDF's; one call
	# for all of them gives the same stream as a call per draw. np.log gives
	# the same bits for an array as for one float.
	uniforms = rng.uniform(size=(num_to_samp, 2))
	sign_positive = uniforms[:, 0] < prob_pos
	sampled_cdf = uniforms[:, 1]
	sampled_vals = np.where(sign_positive,
		-np.log(1-sampled_cdf)/pos_laplace_lambda + mu,
		mu + np.log(1-sampled_cdf)/neg_laplace_lambda)

	return sampled_vals[sampled_vals >= 0], sampled_vals[sampled_vals < 0]


def _iterative_extract_seqlets(score_track, window_size, flank, suppress):
	n, d = score_track.shape
	seqlets = []
	neg_inf = -np.inf

	# A row with nothing above -inf would stop at its first argmax.
	rows = np.flatnonzero((score_track > neg_inf).any(axis=1)).tolist()
	for example_idx in rows:
		single_score_track = score_track[example_idx]
		row_argmax = single_score_track.argmax
		while True:
			argmax = row_argmax()

			#bail if exhausted everything that passed the threshold
			#and was not suppressed
			if single_score_track[argmax] == neg_inf:
				break

			#need to be able to expand without going off the edge
			if argmax >= flank and argmax < (d-flank): 
				seqlet = core.Seqlet(example_idx=example_idx, 
					start=argmax-flank, end=argmax+window_size+flank,
					is_revcomp=False)

				seqlets.append(seqlet)

			#suppress the chunks within +- suppress; the same float64
			#arithmetic as numpy's, on a Python int
			position = int(argmax)
			l_idx = int(max(math.floor(position+0.5-suppress),0))
			r_idx = int(min(math.ceil(position+0.5+suppress), d))
			single_score_track[l_idx:r_idx] = neg_inf

	return seqlets


def _smooth(tracks, window_size):
	n = len(tracks)

	tracks = np.hstack([np.zeros((n, 1)), np.cumsum(tracks, axis=-1)])
	return tracks[:, window_size:] - tracks[:, :-window_size]


def _n_windows(tracks, window_size):
	"""The number of values _smooth(tracks, window_size) returns, from
	the shapes alone, or None when it is not simply n * (length + 1 -
	window_size)."""

	if not (isinstance(tracks, np.ndarray) and tracks.ndim == 2 and
			isinstance(window_size, (int, np.integer)) and
			0 < window_size <= tracks.shape[1]):
		return None

	return tracks.shape[0] * (tracks.shape[1] + 1 - int(window_size))


def _subsample_indices(n, subsample_cap):
	"""The indices that RandomState(1234).choice(a, size=subsample_cap,
	replace=False) takes from an array `a` of n values: the first
	subsample_cap of RandomState(1234).permutation(n), which is how the
	legacy generator draws a sample without replacement, and part of its
	frozen stream (NEP 19). They depend on n alone, not on the values."""

	return np.random.RandomState(1234).permutation(n)[:subsample_cap].copy()


class _ForkedSubsampleIndices:
	"""_subsample_indices(n, subsample_cap), computed in a forked child
	while this process gets on with other work. The legacy generator's
	shuffle holds the GIL throughout (about 0.4 s for the 17 million windows
	of 44,000 regions), so a thread could not run it beside anything else.

	The child is forked only when precompile._can_fork() allows it (Linux,
	no other thread running, a fork-safe numba threading layer or none yet)
	and numba's thread count leaves a thread for it. It writes the indices
	into memory shared with this process and reports over a pipe. result()
	returns them, or computes them here when no child was forked or it did
	not report them, and reaps the child; close() kills and reaps a child
	that result() has not reaped."""

	def __init__(self, n, subsample_cap):
		self.n = n
		self.subsample_cap = subsample_cap
		self._pid = None
		self._fd = None
		self._buffer = None

		if numba.get_num_threads() < 2 or not precompile._can_fork():
			return

		try:
			buffer = mmap.mmap(-1, 8 * max(1, min(n, subsample_cap)))
		except (OSError, ValueError):
			return

		try:
			read_fd, write_fd = os.pipe()
		except OSError:
			buffer.close()
			return

		try:
			with warnings.catch_warnings():
				# Python >= 3.12 warns when a process with native threads
				# forks (precompile.start).
				warnings.simplefilter('ignore', DeprecationWarning)
				pid = os.fork()
		except OSError:
			os.close(read_fd)
			os.close(write_fd)
			buffer.close()
			return

		if pid == 0:
			code = 1
			try:
				os.close(read_fd)
				indices = _subsample_indices(n, subsample_cap)
				np.frombuffer(buffer, dtype=np.int64,
					count=len(indices))[:] = indices
				os.write(write_fd, b'.')
				code = 0
			finally:
				os._exit(code)

		os.close(write_fd)
		self._pid, self._fd, self._buffer = pid, read_fd, buffer

	def result(self):
		indices = None
		if self._pid is not None:
			try:
				if os.read(self._fd, 1) == b'.':
					indices = np.frombuffer(self._buffer, dtype=np.int64,
						count=min(self.n, self.subsample_cap)).copy()
			except OSError:
				pass

			# The child exits once it has reported, or has died.
			try:
				os.waitpid(self._pid, 0)
			except OSError:
				pass
			self._pid = None
			self.close()

		if indices is None:
			indices = _subsample_indices(self.n, self.subsample_cap)
		return indices

	def close(self):
		if self._pid is not None:
			try:
				os.kill(self._pid, signal.SIGKILL)
				os.waitpid(self._pid, 0)
			except OSError:
				pass
			self._pid = None

		if self._fd is not None:
			os.close(self._fd)
			self._buffer.close()
			self._fd = self._buffer = None


def _split(values):
	pos_values = values[values >= 0]
	pos_values = np.sort(pos_values)

	neg_values = values[values < 0]
	neg_values = np.sort(neg_values)[::-1]
	return pos_values, neg_values


def _smooth_and_split(tracks, window_size, subsample_cap=1000000):
	tracks = _smooth(tracks, window_size)

	values = np.concatenate(tracks, axis=0)
	if len(values) > subsample_cap:
		values = np.random.RandomState(1234).choice(
			a=values, size=subsample_cap, replace=False)
	
	pos_values, neg_values = _split(values)
	return pos_values, neg_values, tracks

def _isotonic_thresholds(values, null_values, increasing, target_fdr, 
	min_frac_neg=0.95):
	n1, n2 = len(values), len(null_values)

	X = np.concatenate([values, null_values], axis=0)
	y = np.concatenate([np.ones(n1), np.zeros(n2)], axis=0)

	w = len(values) / len(null_values)
	sample_weight = np.concatenate([np.ones(n1), np.ones(n2)*w], axis=0)

	from sklearn.isotonic import IsotonicRegression
	model = IsotonicRegression(out_of_bounds='clip', increasing=increasing)
	model.fit(X, y, sample_weight=sample_weight)

	min_prec_x = model.X_min_ if increasing else model.X_max_
	min_precision = model.transform([min_prec_x])[0]
	implied_frac_neg = -1/(1-(1/max(min_precision, 1e-7)))
	if (implied_frac_neg > 1.0 or implied_frac_neg < min_frac_neg):
		implied_frac_neg = max(min(1.0,implied_frac_neg), min_frac_neg)

	precisions = np.minimum(np.maximum(1 + implied_frac_neg*(
		1 - (1 / np.maximum(model.transform(values), 1e-7))), 0.0), 1.0)
	precisions[-1] = 1
	return values[precisions >= (1 - target_fdr)][0]

def _refine_thresholds(vals, pos_threshold, neg_threshold,
	min_passing_windows_frac, max_passing_windows_frac):
	frac_passing_windows =(
		np.count_nonzero(vals >= pos_threshold)
		 + np.count_nonzero(vals <= neg_threshold))/float(len(vals))

	if frac_passing_windows < min_passing_windows_frac:
		pos_threshold = np.percentile(
			a=np.abs(vals),
			q=100*(1-min_passing_windows_frac)) 
		neg_threshold = -pos_threshold

	if frac_passing_windows > max_passing_windows_frac:
		pos_threshold = np.percentile(
			a=np.abs(vals),
			q=100*(1-max_passing_windows_frac)) 
		neg_threshold = -pos_threshold

	return pos_threshold, neg_threshold


def extract_seqlets(attribution_scores, window_size, flank, suppress, 
	target_fdr, min_passing_windows_frac, max_passing_windows_frac, 
	weak_threshold_for_counting_sign):

	# _smooth_and_split's values, computed in another order. The indices of
	# its subsample depend only on the number of windows, so they are found
	# in a forked child while this process estimates the null. Every value
	# is computed as before.
	subsample_cap = 1000000
	n_windows = _n_windows(attribution_scores, window_size)
	indices = None
	if n_windows is not None and n_windows > subsample_cap:
		indices = _ForkedSubsampleIndices(n_windows, subsample_cap)

	try:
		smoothed_tracks = _smooth(attribution_scores, window_size)

		pos_null_values, neg_null_values = _laplacian_null(
			track=smoothed_tracks, window_size=window_size, num_to_samp=10000)

		# _laplacian_null has raised on any NaN or inf in the tracks, and
		# np.abs leaves no -0.0, so equal values have equal bits and an
		# in-place sort gives exactly the array sorted() gave, in a small
		# fraction of the time.
		distribution = np.abs(smoothed_tracks).ravel()
		distribution.sort()

		# The values np.concatenate(smoothed_tracks, axis=0) gave, uncopied.
		values = smoothed_tracks.ravel()
		if len(values) > subsample_cap:
			if indices is not None and len(values) == n_windows:
				values = values[indices.result()]
			else:
				values = np.random.RandomState(1234).choice(
					a=values, size=subsample_cap, replace=False)
	finally:
		if indices is not None:
			indices.close()

	pos_values, neg_values = _split(values)
	del values

	pos_threshold = _isotonic_thresholds(pos_values, pos_null_values, 
		increasing=True, target_fdr=target_fdr)
	neg_threshold = _isotonic_thresholds(neg_values, neg_null_values,
		increasing=False, target_fdr=target_fdr)

	pos_threshold, neg_threshold = _refine_thresholds(
		  vals=np.concatenate([pos_values, neg_values], axis=0),
		  pos_threshold=pos_threshold,
		  neg_threshold=neg_threshold,
		  min_passing_windows_frac=min_passing_windows_frac,
		  max_passing_windows_frac=max_passing_windows_frac) 

	transformed_pos_threshold = np.sign(pos_threshold)*np.searchsorted(
		a=distribution, v=abs(pos_threshold))/len(distribution)

	transformed_neg_threshold = np.sign(neg_threshold)*np.searchsorted(
		a=distribution, v=abs(neg_threshold))/len(distribution)

	idxs = (smoothed_tracks >= pos_threshold) | (smoothed_tracks <= neg_threshold)

	smoothed_tracks[idxs] = np.abs(smoothed_tracks[idxs])
	smoothed_tracks[~idxs] = -np.inf

	# Filter out the flanks
	smoothed_tracks[:, :flank] = -np.inf
	smoothed_tracks[:, -flank:] = -np.inf

	seqlets = _iterative_extract_seqlets(score_track=smoothed_tracks,
		window_size=window_size,
		flank=flank,
		suppress=suppress)

	#find the weakest transformed threshold used across all tasks
	weak_thresh = min(min(transformed_pos_threshold, 
		abs(transformed_neg_threshold)) - 0.0001, 
			weak_threshold_for_counting_sign)

	threshold = distribution[int(weak_thresh * len(distribution))]

	return seqlets, threshold
