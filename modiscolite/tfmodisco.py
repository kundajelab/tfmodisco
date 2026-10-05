# tfmodisco.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import mmap
import pickle

import numba
import numpy as np

import scipy
import scipy.sparse

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import affinitymat
from . import aggregator
from . import extract_seqlets
from . import core
from . import util
from . import cluster
from . import forked
from . import precompile

def _density_adaptation(affmat_nn, seqlet_neighbors, tsne_perplexity):
	n = len(affmat_nn)

	# The (row, col, value) triples in row order and, within a row, in the
	# order of zip(seqlet_neighbors[row], affmat_nn[row]): row r holds
	# lengths[r] of them.
	if (isinstance(affmat_nn, np.ndarray) and affmat_nn.ndim == 2 and
		isinstance(seqlet_neighbors, np.ndarray) and seqlet_neighbors.ndim == 2):
		k = min(affmat_nn.shape[1], seqlet_neighbors.shape[1])
		data = affmat_nn[:, :k].ravel()
		cols = seqlet_neighbors[:n, :k].ravel()
		lengths = np.full(n, k, dtype='int64')
	else:
		lengths = np.array([min(len(seqlet_neighbors[row]), len(affmat_nn[row]))
			for row in range(n)], dtype='int64')
		data = np.concatenate([np.asarray(affmat_nn[row][:m])
			for row, m in enumerate(lengths)] + [np.empty(0)])
		cols = np.concatenate([np.asarray(seqlet_neighbors[row][:m], dtype='int64')
			for row, m in enumerate(lengths)] + [np.empty(0, dtype='int64')])

	# The kernels build the matrix with 32-bit indices, as scipy does below
	# this size; other inputs, and triples the kernels turn down, go through
	# scipy.
	affmat = None
	if (n > 0 and data.dtype.kind == 'f' and data.dtype.itemsize <= 8 and
		cols.dtype.kind in 'iu' and len(data) == len(cols) == lengths.sum() and
		2 * len(data) + n <= np.iinfo(np.int32).max):
		affmat = _density_adaptation_kernels(data, cols, lengths, n,
			tsne_perplexity)

	if affmat is None:
		rows = np.repeat(np.arange(n), lengths)
		del lengths
		affmat = _density_adaptation_scipy(data, rows, cols, n,
			tsne_perplexity)

	return affmat


def _betas(tsne_perplexity, data, indptr):
	"""util.binary_search_perplexity over each row of a CSR matrix. Rows are
	independent, so contiguous blocks of them are shared among
	numba.get_num_threads() threads running util._perplexity_betas, which
	releases the GIL. The result does not depend on the number of threads or
	blocks."""

	n = len(indptr) - 1
	betas = np.empty(n)
	n_threads = numba.get_num_threads()
	n_blocks = max(1, min(n, 16 * n_threads))
	bounds = [n * b // n_blocks for b in range(n_blocks + 1)]

	def run(b):
		util._perplexity_betas(tsne_perplexity, data, indptr, bounds[b],
			bounds[b+1], betas)

	with ThreadPoolExecutor(min(n_threads, n_blocks)) as pool:
		list(pool.map(run, range(n_blocks)))

	return betas


def _density_adaptation_kernels(data, cols, lengths, n, tsne_perplexity):
	"""_density_adaptation_scipy's matrix, built without sorting: every
	value is computed by the same operations in the same order, and every
	entry lands in the same place. Returns None, having changed nothing,
	when a column index lies outside [0, n) or a row holds one twice, which
	scipy would reject or sum, or when a log-transformed affinity is NaN
	(util._transpose_triples)."""

	eps = 0.0000001

	offsets = np.zeros(n + 1, dtype='int64')
	np.cumsum(lengths, out=offsets[1:])
	cols = np.ascontiguousarray(cols, dtype='int64')

	# The transform is elementwise, so applying it before the entries are
	# put in column order, and dropping its zeros after, changes no value.
	data = np.asarray(data, dtype='float64')
	data = np.maximum(np.log((1.0/(0.5*np.maximum(data, eps)))-1), 0)

	# B is the transpose of the log-transformed matrix, with sorted indices
	# and its zeros dropped; A, its transpose, is that matrix itself, as
	# scipy builds it (coo -> csr, sort_indices, eliminate_zeros).
	nnz = len(cols)
	b_indptr = np.empty(n + 1, dtype='int64')
	b_indices = np.empty(nnz, dtype='int32')
	b_data = np.empty(nnz, dtype='float64')
	if util._transpose_triples(offsets, cols, data, n, b_indptr, b_indices,
			b_data) != 0:
		return None

	del offsets, cols, data
	nnz = int(b_indptr[n])
	b_indices, b_data = b_indices[:nnz], b_data[:nnz]

	a_indptr = np.empty(n + 1, dtype='int64')
	a_indices = np.empty(nnz, dtype='int32')
	a_data = np.empty(nnz, dtype='float64')
	util._transpose(b_indptr, b_indices, b_data, a_indptr, a_indices, a_data)

	# (A + A.T) / (C + C.T), C holding A's pattern: the symmetrized matrix.
	indptr = np.empty(n + 1, dtype='int64')
	indices = np.empty(2 * nnz, dtype='int32')
	data = np.empty(2 * nnz, dtype='float64')
	nnz = util._symmetrize(a_indptr, a_indices, a_data, b_indptr, b_indices,
		b_data, indptr, indices, data)
	del a_indptr, a_indices, a_data, b_indptr, b_indices, b_data
	indices, data = indices[:nnz], data[:nnz]

	betas = _betas(tsne_perplexity, data, indptr)

	# As in _density_adaptation_scipy, with each row's normalizer summed in
	# numpy's order by util._row_sums_plus_one.
	row_idx = np.repeat(np.arange(n), np.diff(indptr))
	neg_data = -data

	rbf_i = betas[row_idx]
	np.divide(neg_data, rbf_i, out=rbf_i)
	np.exp(rbf_i, out=rbf_i)
	normfactors = np.empty(n, dtype='float64')
	util._row_sums_plus_one(rbf_i, indptr, normfactors)

	rbf_i /= normfactors[row_idx]
	del row_idx

	rbf_j = betas[indices]
	np.divide(neg_data, rbf_j, out=rbf_j)
	del neg_data
	np.exp(rbf_j, out=rbf_j)
	rbf_j /= normfactors[indices]

	rbf_i *= rbf_j
	del rbf_j
	np.sqrt(rbf_i, out=data)
	del rbf_i

	# + scipy.sparse.diags(1.0 / normfactors), which also drops zeros.
	out_indptr = np.empty(n + 1, dtype='int32')
	out_indices = np.empty(nnz + n, dtype='int32')
	out_data = np.empty(nnz + n, dtype='float64')
	nnz = util._add_diagonal(indptr, indices, data, 1.0 / normfactors,
		out_indptr, out_indices, out_data)

	return scipy.sparse.csr_matrix((out_data[:nnz], out_indices[:nnz],
		out_indptr), shape=(n, n))


def _density_adaptation_scipy(data, rows, cols, n, tsne_perplexity):
	eps = 0.0000001

	affmat_nn = scipy.sparse.csr_matrix((data, (rows, cols)), 
		shape=(n, n), dtype='float64')
	del data, rows, cols
	
	affmat_nn.data = np.maximum(np.log((1.0/(0.5*np.maximum(affmat_nn.data, eps)))-1), 0)
	affmat_nn.eliminate_zeros()

	counts_nn = scipy.sparse.csr_matrix((np.ones_like(affmat_nn.data), 
		affmat_nn.indices, affmat_nn.indptr), shape=affmat_nn.shape, dtype='float64')

	affmat_nn += affmat_nn.T
	counts_nn += counts_nn.T
	affmat_nn.data /= counts_nn.data
	del counts_nn

	data, indices, indptr = affmat_nn.data, affmat_nn.indices, affmat_nn.indptr
	betas = np.array([util.binary_search_perplexity(tsne_perplexity,
		data[indptr[i]:indptr[i+1]]) for i in range(n)])

	# Elementwise over all stored entries: the same operations, in the same
	# order, as a loop over entries, and each row's normalizer is numpy's sum
	# over that row's values, so every value keeps its bits. The ufuncs
	# write into arrays that already exist and each array is dropped after
	# its last use, so at most four arrays of nnz floats are alive at once
	# instead of six; writing a ufunc's result in place changes no value.
	row_idx = np.repeat(np.arange(n), np.diff(indptr))
	neg_data = -data

	# rbf_i = exp(neg_data / betas[row_idx])
	rbf_i = betas[row_idx]
	np.divide(neg_data, rbf_i, out=rbf_i)
	np.exp(rbf_i, out=rbf_i)
	normfactors = np.array([rbf_i[indptr[i]:indptr[i+1]].sum() + 1
		for i in range(n)])

	rbf_i /= normfactors[row_idx]
	del row_idx

	# rbf_j = exp(neg_data / betas[indices]) / normfactors[indices]
	rbf_j = betas[indices]
	np.divide(neg_data, rbf_j, out=rbf_j)
	del neg_data
	np.exp(rbf_j, out=rbf_j)
	rbf_j /= normfactors[indices]

	# data = sqrt(rbf_i * rbf_j)
	rbf_i *= rbf_j
	del rbf_j
	np.sqrt(rbf_i, out=data)
	del rbf_i

	affmat_diags = scipy.sparse.diags(1.0 / normfactors)
	affmat_nn += affmat_diags
	return affmat_nn


def _filter_patterns(patterns, min_seqlet_support, window_size, 
	min_ic_in_window, background, ppm_pseudocount):
	passing_patterns = []
	for pattern in patterns:
		if len(pattern.seqlets) < min_seqlet_support:
			continue

		ppm = pattern.sequence
		per_position_ic = util.compute_per_position_ic(ppm=ppm, 
			background=background, pseudocount=ppm_pseudocount)

		if len(per_position_ic) < window_size:       
			if np.sum(per_position_ic) < min_ic_in_window:
				continue
		else:
			#do the sliding window sum rearrangement
			windowed_ic = np.sum(util.rolling_window(
				a=per_position_ic, window=window_size), axis=-1)

			if np.max(windowed_ic) < min_ic_in_window:
				continue

		passing_patterns.append(pattern)

	return passing_patterns


# A share of clusters goes to a forked child only when it holds at least this
# many seqlets. On the 20k input, merging costs 0.05-0.1 ms per seqlet with
# one thread (PATTERN_THREADS), against 10-20 ms to fork a child and 10-20
# ms more for it to exit, which this process waits for when the child
# finishes last.
MIN_CHILD_SEQLETS = 500

# Bytes of shared memory set aside per seqlet of a child's share for the
# pickled coordinates of its patterns, which hold no more seqlets than the
# share: about 45 bytes per seqlet on the 20k input.
SHARED_BYTES_PER_SEQLET = 128

# Numba threads per process while patterns are built. Merging aligns one
# seqlet at a time with small numba kernels, and each call pays for waking
# every thread it has: on the 20k input, the largest round-1 cluster built
# in 193 ms with 1 thread, 258 ms with 6 and 311 ms with 16, with the same
# pattern each time.
PATTERN_THREADS = 1


def _pattern_from_cluster(seqlets, track_set, min_overlap, min_frac, min_num,
	flank_to_add, window_size, bg_freq, track_sign):
	"""The pattern built from one cluster's seqlets, or None when polishing
	drops it or the sign of its contributions is not track_sign."""

	seqlet_sort_metric = lambda x: -np.sum(np.abs(x.contrib_scores))
	sorted_seqlets = sorted(seqlets, key=seqlet_sort_metric)
	pattern = core.SeqletSet([sorted_seqlets[0]])

	if len(sorted_seqlets) > 1:
		pattern = aggregator.merge_in_seqlets_filledges(
			parent_pattern=pattern, seqlets_to_merge=sorted_seqlets[1:],
			track_set=track_set, metric=affinitymat.jaccard,
			min_overlap=min_overlap)

	pattern = aggregator.polish_pattern(pattern, min_frac=min_frac,
		min_num=min_num, track_set=track_set, flank=flank_to_add,
		window_size=window_size, bg_freq=bg_freq)

	if pattern is not None:
		if np.sign(np.sum(pattern.contrib_scores)) == track_sign:
			return pattern

	return None


def _plan_clusters(sizes, n_budget):
	"""Group clusters into shares, one per process, or return None to build
	every pattern here.

	A cluster's cost is about linear in its number of seqlets, since its
	seqlets are merged into the pattern one at a time. The largest cluster
	goes to share 0, which the parent runs, and the rest are packed longest
	first (forked._lpt) into the fewest shares that leave no child more
	seqlets than the parent. There is at most one share per cluster, one per
	thread of the budget, and one child for every MIN_CHILD_SEQLETS seqlets
	outside the largest cluster.
	"""

	n = len(sizes)
	if n < 2:
		return None

	order = sorted(range(n), key=lambda j: -sizes[j])
	largest = sizes[order[0]]
	rest = sum(sizes) - largest
	max_bins = min(n, n_budget, 1 + rest // MIN_CHILD_SEQLETS)
	if max_bins < 2:
		return None

	for n_bins in range(2, max_bins + 1):
		bins, loads = forked._lpt(sizes, order, n_bins)
		if max(loads[1:]) <= largest:
			break

	return bins, loads


def _patterns_from_clusters(seqlets, track_set, min_overlap,
	min_frac, min_num, flank_to_add, window_size, bg_freq, cluster_indices,
	track_sign):

	num_clusters = max(cluster_indices+1)
	cluster_to_seqlets = defaultdict(list)

	for seqlet, idx in zip(seqlets, cluster_indices):
		cluster_to_seqlets[idx].append(seqlet)

	def build(i):
		return _pattern_from_cluster(cluster_to_seqlets[i], track_set=track_set,
			min_overlap=min_overlap, min_frac=min_frac, min_num=min_num,
			flank_to_add=flank_to_add, window_size=window_size,
			bg_freq=bg_freq, track_sign=track_sign)

	# The clusters' patterns are built independently of each other, so
	# shares of them are built in forked children (forked.forked_map). A
	# child sends back the coordinates of each pattern's seqlets, in order,
	# and the pattern is made again here from new seqlets over the track set:
	# views of the same arrays with the same strides as the child's, added
	# in the same order, so every sum in the pattern has the same bits.
	def build_share(share):
		return share, [build(i) for i in share]

	n_budget = forked.budget()
	plan = _plan_clusters([len(cluster_to_seqlets[i])
		for i in range(num_clusters)], n_budget)

	n_threads = min(n_budget, PATTERN_THREADS)
	if plan is None or not forked.numba_fork_safe():
		saved = numba.get_num_threads()
		numba.set_num_threads(min(saved, n_threads))
		try:
			built = [build(i) for i in range(num_clusters)]
		finally:
			numba.set_num_threads(saved)
	else:
		bins, loads = plan

		# The pickled coordinates travel through a shared anonymous mapping,
		# each child writing its own region, and only their place in it goes
		# through the pipe. A pickle larger than the pipe (8 KB here) blocks
		# its child until this process reads it, after building its own
		# share, and the children then exit one after another, each reaped
		# before the next is read. With the pipe unblocked, a child exits
		# as soon as it is done, while this process still builds. A child
		# whose pickle does not fit its region sends the coordinates through
		# the pipe instead. Either way this process loads the same pickle.
		regions, total = {}, 0
		for share, load in zip(bins[1:], loads[1:]):
			if share:
				size = SHARED_BYTES_PER_SEQLET * load + 4096
				regions[tuple(share)] = (total, size)
				total += size

		with mmap.mmap(-1, max(total, 1)) as shared:
			def pack_share(result):
				share, patterns = result
				coordinates = [None if pattern is None else [(
					seqlet.example_idx, seqlet.start, seqlet.end,
					seqlet.is_revcomp) for seqlet in pattern.seqlets]
					for pattern in patterns]

				data = pickle.dumps(coordinates,
					protocol=pickle.HIGHEST_PROTOCOL)
				offset, size = regions.get(tuple(share), (0, -1))
				if len(data) > size:
					return share, coordinates

				shared[offset:offset + len(data)] = data
				return share, offset, len(data)

			def unpack_share(packed):
				if len(packed) == 3:
					share, offset, length = packed
					coordinates = pickle.loads(shared[offset:offset + length])
				else:
					share, coordinates = packed

				return share, [None if pattern is None else core.SeqletSet(
					track_set.create_seqlets([core.Seqlet(
						example_idx=example_idx, start=start, end=end,
						is_revcomp=is_revcomp) for example_idx, start, end,
						is_revcomp in pattern])) for pattern in coordinates]

			shares = forked.forked_map(build_share, bins,
				budgets=[n_threads] * len(bins), pack=pack_share,
				unpack=unpack_share)

		built = [None] * num_clusters
		for share, (_, share_patterns) in zip(bins, shares):
			for i, pattern in zip(share, share_patterns):
				built[i] = pattern

	return [pattern for pattern in built if pattern is not None]


def _average_ranks(values, mask):
	"""The 1-based ranks of each row's masked entries among themselves, ties
	given their average rank, as scipy.stats.rankdata gives them. Entries
	outside the mask get meaningless ranks.

	A rank depends only on how many entries of its row are below it and
	equal to it, so any sort gives the same ranks: the rows are sorted by
	numpy's float64 argsort (float32 values widen to float64 exactly), or
	not at all when every row already runs from high to low, as rows of
	nearest-neighbour affinities do. Only rows holding a tie need the
	average over their tie groups; in every other row the rank is the
	sorted position plus one.
	"""

	n, k = values.shape
	positions = np.arange(k)

	# Entries outside the mask become NaN, which sorts after every number,
	# so each row's masked entries come first and in order. NaN never equals
	# anything, so it never joins a group of ties, and never passes the
	# descending check.
	x = np.where(mask, values, np.nan)
	if np.less_equal(x[:, 1:], x[:, :-1]).all():
		order = None
		y = x[:, ::-1]
	else:
		order = np.argsort(x.astype(np.float64, copy=False), axis=1)
		order += (np.arange(n) * k)[:, None]
		y = x.reshape(-1)[order]

	starts = np.empty((n, k), dtype=bool)
	starts[:, :1] = True
	np.not_equal(y[:, 1:], y[:, :-1], out=starts[:, 1:])

	sorted_ranks = np.empty((n, k), dtype='float64')
	sorted_ranks[:] = positions + 1.0

	tied = np.flatnonzero(~starts.all(axis=1))
	if len(tied) > 0:
		starts = starts[tied]
		ends = np.ones((len(tied), k), dtype=bool)
		ends[:, :-1] = starts[:, 1:]

		first = np.maximum.accumulate(np.where(starts, positions, 0), axis=1)
		last = np.minimum.accumulate(np.where(ends, positions, k)[:, ::-1],
			axis=1)[:, ::-1]

		# rankdata's ordinal rank of the first tie plus (count - 1) / 2: both
		# are exact, so these are the same half-integers. Where a row has no
		# tie, first == last and this is the position plus one.
		sorted_ranks[tied] = (first + 1) + (last - first) / 2

	if order is None:
		return np.ascontiguousarray(sorted_ranks[:, ::-1])

	ranks = np.empty((n, k), dtype='float64')
	ranks.reshape(-1)[order] = sorted_ranks
	return ranks


def _spearman_rows(fine_affmat_nn, coarse_affmat_nn, mask):
	"""For every row i, scipy.stats.spearmanr(fine_affmat_nn[i][mask[i]],
	coarse_affmat_nn[i][mask[i]]).correlation, with the same bits.

	spearmanr ranks both columns and hands the ranks to np.corrcoef. Ranks
	are half-integers with mean (m+1)/2 over m entries, so the centred
	ranks, their products and every partial sum of those products are
	multiples of 0.25 far below 2**53: the three sums corrcoef takes with a
	BLAS product are exact in any order. What remains is the scaling, done
	here in corrcoef's order: each sum times 1/(m-1), then divided by the
	coarse and then by the fine standard deviation, then clipped to [-1, 1].
	spearmanr returns NaN for fewer than two entries, for a constant input
	and for an input holding a NaN.
	"""

	counts = mask.sum(axis=1)
	ranks_fine = _average_ranks(fine_affmat_nn, mask)
	ranks_coarse = _average_ranks(coarse_affmat_nn, mask)

	# With every entry masked, np.where(mask, a, 0) is a itself.
	full = bool(mask.all())

	with np.errstate(divide='ignore', invalid='ignore'):
		if full:
			mean_fine = ranks_fine.sum(axis=1) / counts
			mean_coarse = ranks_coarse.sum(axis=1) / counts

			centred_fine = ranks_fine
			centred_fine -= mean_fine[:, None]
			centred_coarse = ranks_coarse
			centred_coarse -= mean_coarse[:, None]
		else:
			mean_fine = np.where(mask, ranks_fine, 0).sum(axis=1) / counts
			mean_coarse = np.where(mask, ranks_coarse, 0).sum(axis=1) / counts

			centred_fine = np.where(mask, ranks_fine - mean_fine[:, None], 0)
			centred_coarse = np.where(mask, ranks_coarse - mean_coarse[:, None],
				0)
		del ranks_fine, ranks_coarse

		c_ff = (centred_fine * centred_fine).sum(axis=1)
		c_cc = (centred_coarse * centred_coarse).sum(axis=1)
		c_cf = (centred_coarse * centred_fine).sum(axis=1)

		scale = 1 / (counts - 1)
		std_fine = np.sqrt(c_ff * scale)
		std_coarse = np.sqrt(c_cc * scale)
		correlations = ((c_cf * scale) / std_coarse) / std_fine
		np.clip(correlations, -1, 1, out=correlations)

	# A constant input has every rank tied, so its exact sum of squares is 0.
	# Masked fine entries are nonzero and so never NaN; coarse ones can be.
	masked_coarse = coarse_affmat_nn if full else np.where(mask,
		coarse_affmat_nn, 0)
	undefined = ((counts <= 1) | (c_ff == 0) | (c_cc == 0) |
		np.isnan(masked_coarse).any(axis=1))
	correlations[undefined] = np.nan
	return correlations


# Entries per block of rows in _correlations: each thread's temporaries are
# arrays of about this many float64s, around a megabyte. Threads other than
# the main one allocate from their own malloc arenas, which keep what is
# freed mapped, so the output is allocated in the calling thread, and at
# most SPEARMAN_THREADS threads run: on the 20k input 8 threads were as
# fast as 16 (the allocations contend) and kept 30 MB mapped against 74.
SPEARMAN_BLOCK = 1 << 17
SPEARMAN_THREADS = 8


def _correlations(fine_affmat_nn, coarse_affmat_nn):
	"""_spearman_rows over every row, with the mask |fine| > 0. Rows are
	independent, so blocks of rows are shared among up to
	numba.get_num_threads() threads (numpy releases the GIL in the sorts and
	the elementwise work), each writing its own slice of the output. No
	value depends on the blocks or the number of threads."""

	n, k = fine_affmat_nn.shape
	correlations = np.empty(n, dtype='float64')
	step = max(1, SPEARMAN_BLOCK // max(k, 1))

	def run(start):
		block = slice(start, start + step)
		fine = fine_affmat_nn[block]
		correlations[block] = _spearman_rows(fine, coarse_affmat_nn[block],
			np.abs(fine) > 0)

	starts = range(0, n, step)
	n_threads = min(numba.get_num_threads(), SPEARMAN_THREADS, len(starts))
	if n_threads < 2:
		for start in starts:
			run(start)
	else:
		with ThreadPoolExecutor(n_threads) as pool:
			list(pool.map(run, starts))

	return correlations


def _filter_by_correlation(seqlets, seqlet_neighbors, coarse_affmat_nn, 
	fine_affmat_nn, correlation_threshold):

	arrays = (seqlet_neighbors, coarse_affmat_nn, fine_affmat_nn)
	if (all(isinstance(a, np.ndarray) and a.ndim == 2 for a in arrays) and
		seqlet_neighbors.shape == fine_affmat_nn.shape == coarse_affmat_nn.shape
		and np.issubdtype(seqlet_neighbors.dtype, np.integer) and
		len(seqlets) == len(fine_affmat_nn)):

		n = len(fine_affmat_nn)
		correlations = _correlations(fine_affmat_nn, coarse_affmat_nn)

		filtered_rows_mask = correlations > correlation_threshold
		rows = np.flatnonzero(filtered_rows_mask)

		seqlets = list(seqlets)
		filtered_seqlets = [seqlets[i] for i in rows.tolist()]
		if len(filtered_seqlets) == 0:
			return filtered_seqlets, [], []

		# Each retained row keeps, in order, the neighbours that are
		# retained rows, renumbered to the retained rows' positions, and
		# their affinities: int64 and fine_affmat_nn's dtype, as before.
		# Each row is a view of one flat array, as np.split gives.
		new_idx_mapping = np.cumsum(filtered_rows_mask) - 1
		neighbors = seqlet_neighbors[rows]
		if neighbors.size and (neighbors.min() < 0 or neighbors.max() >= n):
			in_range = (neighbors >= 0) & (neighbors < n)
			keep = in_range & filtered_rows_mask[np.where(in_range, neighbors,
				0)]
		else:
			keep = np.take(filtered_rows_mask, neighbors)

		# np.compress over the flattened rows is a[keep] for a 2-D a, in the
		# same row-major order.
		bounds = [0] + np.cumsum(keep.sum(axis=1)).tolist()
		bounds = list(zip(bounds[:-1], bounds[1:]))
		keep = keep.ravel()
		flat_neighbors = np.take(new_idx_mapping, np.compress(keep,
			neighbors.ravel()))
		del neighbors
		flat_affmat_nn = np.compress(keep, fine_affmat_nn[rows].ravel())
		filtered_neighbors = [flat_neighbors[a:b] for a, b in bounds]
		filtered_affmat_nn = [flat_affmat_nn[a:b] for a, b in bounds]
		return filtered_seqlets, filtered_neighbors, filtered_affmat_nn

	import scipy.stats
	correlations = []
	for fine_affmat_row, coarse_affmat_row in zip(fine_affmat_nn, coarse_affmat_nn):
		to_compare_mask = np.abs(fine_affmat_row) > 0
		corr = scipy.stats.spearmanr(fine_affmat_row[to_compare_mask],
			coarse_affmat_row[to_compare_mask])
		correlations.append(corr.correlation)

	correlations = np.array(correlations)
	filtered_rows_mask = np.array(correlations) > correlation_threshold

	filtered_seqlets = [seqlet for seqlet, mask in zip(seqlets, 
		filtered_rows_mask) if mask == True]

	#figure out a mapping from pre-filtering to the
	# post-filtering indices
	new_idx_mapping = np.cumsum(filtered_rows_mask) - 1
	retained_indices = set(np.where(filtered_rows_mask == True)[0])

	filtered_neighbors = []
	filtered_affmat_nn = []
	for old_row_idx, (old_neighbors, affmat_row) in enumerate(zip(seqlet_neighbors, fine_affmat_nn)):
		if old_row_idx in retained_indices:
			filtered_old_neighbors = [neighbor for neighbor in old_neighbors if neighbor in retained_indices]
			filtered_affmat_row = [affmatval for affmatval, neighbor in zip(affmat_row,old_neighbors) if neighbor in retained_indices]
			filtered_neighbors_row = [new_idx_mapping[neighbor] for neighbor in filtered_old_neighbors]
			filtered_neighbors.append(filtered_neighbors_row)
			filtered_affmat_nn.append(filtered_affmat_row)

	return filtered_seqlets, filtered_neighbors, filtered_affmat_nn


def seqlets_to_patterns(seqlets, track_set, track_signs=None, 
	min_overlap_while_sliding=0.7, nearest_neighbors_to_compute=500, 
	affmat_correlation_threshold=0.15, tsne_perplexity=10.0, 
	n_leiden_iterations=-1, n_leiden_runs=50, frac_support_to_trim_to=0.2,
	min_num_to_trim_to=30, trim_to_window_size=20, initial_flank_to_add=5,
	final_flank_to_add=0,
	prob_and_pertrack_sim_merge_thresholds=[(0.8,0.8), (0.5, 0.85), (0.2, 0.9)],
	prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],
	subcluster_perplexity=50, merging_max_seqlets_subsample=1000,
	final_min_cluster_size=20,min_ic_in_window=0.6, min_ic_windowsize=6,
	ppm_pseudocount=0.001, beside_clustering=None):

	bg_freq = np.mean([seqlet.sequence for seqlet in seqlets], axis=(0, 1)) 

	seqlets_sorter = (lambda arr: sorted(arr, key=lambda x:
		-np.sum(np.abs(x.contrib_scores))))

	seqlets = seqlets_sorter(seqlets)

	for round_idx in range(2):
		if len(seqlets) == 0:
			return None

		# Step 1: Generate coarse resolution
		coarse_affmat_nn, seqlet_neighbors = affinitymat.cosine_similarity_from_seqlets(
			seqlets=seqlets, n_neighbors=nearest_neighbors_to_compute, sign=track_signs)

		# Step 2: Generate fine representation
		fine_affmat_nn = affinitymat.jaccard_from_seqlets(
			seqlets=seqlets, seqlet_neighbors=seqlet_neighbors,
			min_overlap=min_overlap_while_sliding)

		if round_idx == 0:
			filtered_seqlets, seqlet_neighbors, filtered_affmat_nn = (
				_filter_by_correlation(seqlets, seqlet_neighbors, 
					coarse_affmat_nn, fine_affmat_nn, 
					affmat_correlation_threshold))
		else:
			filtered_seqlets = seqlets
			filtered_affmat_nn = fine_affmat_nn

		del coarse_affmat_nn
		del fine_affmat_nn
		del seqlets

		# Step 4: Density adaptation
		csr_density_adapted_affmat = _density_adaptation(
			filtered_affmat_nn, seqlet_neighbors, tsne_perplexity)

		del filtered_affmat_nn
		del seqlet_neighbors

		# Step 5: Clustering. Leiden runs on one thread in each of its
		# processes, so the first clustering leaves the rest of the thread
		# budget idle: a forked.Background given as beside_clustering runs
		# in a child with those threads, and is waited for before the next
		# step, which uses the whole budget again.
		beside = beside_clustering if round_idx == 0 else None
		if beside is not None:
			n_workers = cluster._n_workers(n_leiden_runs,
				len(csr_density_adapted_affmat.data))
			beside.start(min(forked.budget(), numba.get_num_threads()) -
				n_workers)

		try:
			cluster_indices = cluster.LeidenCluster(
				csr_density_adapted_affmat,
				n_seeds=n_leiden_runs,
				n_leiden_iterations=n_leiden_iterations)
		except BaseException:
			if beside is not None:
				beside.kill()
			raise

		if beside is not None:
			beside.join()

		del csr_density_adapted_affmat

		patterns = _patterns_from_clusters(filtered_seqlets, 
			track_set=track_set, 
			min_overlap=min_overlap_while_sliding, 
			min_frac=frac_support_to_trim_to, 
			min_num=min_num_to_trim_to, 
			flank_to_add=initial_flank_to_add, 
			window_size=trim_to_window_size, 
			bg_freq=bg_freq, 
			cluster_indices=cluster_indices, 
			track_sign=track_signs)

		#obtain unique seqlets from adjusted motifs
		seqlets = list(dict([(y.string, y)
						 for x in patterns for y in x.seqlets]).values())

	del seqlets

	merged_patterns, pattern_merge_hierarchy = aggregator._detect_spurious_merging(
		patterns=patterns, track_set=track_set, perplexity=subcluster_perplexity, 
		min_in_subcluster=max(final_min_cluster_size, subcluster_perplexity), 
		min_overlap=min_overlap_while_sliding,
		prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
		prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
		min_frac=frac_support_to_trim_to, min_num=min_num_to_trim_to,
		flank_to_add=initial_flank_to_add,
		window_size=trim_to_window_size, bg_freq=bg_freq,
		max_seqlets_subsample=merging_max_seqlets_subsample,
		n_seeds=n_leiden_runs)

	#Now start merging patterns 
	merged_patterns = sorted(merged_patterns, key=lambda x: -len(x.seqlets))

	patterns = _filter_patterns(merged_patterns, 
		min_seqlet_support=final_min_cluster_size, 
		window_size=min_ic_windowsize, min_ic_in_window=min_ic_in_window, 
		background=bg_freq, ppm_pseudocount=ppm_pseudocount)

	#apply subclustering procedure on the final patterns
	for patternidx, pattern in enumerate(patterns):
		patterns[patternidx] = aggregator._expand_seqlets_to_fill_pattern(
			pattern, track_set, left_flank_to_add=final_flank_to_add,
			right_flank_to_add=final_flank_to_add)

	# The patterns are subclustered independently of each other, so this runs
	# in forked workers (forked.map_tasks). A worker sends back each
	# pattern's membership, and the subpatterns are built from it here, as
	# compute_subpatterns would. When the largest pattern's Leiden runs are
	# expected to finish last (forked.finishes_last), its affinity matrix is
	# computed first, with the whole thread budget, before the workers are
	# forked (prepare).
	affinities = {}

	def subcluster(patternidx):
		pattern = patterns[patternidx]
		pattern.compute_subpatterns(subcluster_perplexity,
			n_seeds=n_leiden_runs, n_iterations=n_leiden_iterations,
			affinity=affinities.pop(patternidx, None))
		return pattern.subclusters

	def attach(packed):
		for patternidx, subclusters in packed.items():
			patterns[patternidx]._set_subclusters(subclusters)
		return packed

	def prepare(patternidx, loads, budgets):
		if forked.finishes_last(loads, budgets, n_leiden_runs):
			affinities[patternidx] = patterns[patternidx]._subpattern_affinity(
				subcluster_perplexity)

	forked.map_tasks(subcluster, range(len(patterns)),
		weights=[len(pattern.seqlets) ** 2 for pattern in patterns],
		unpack=attach,
		before_fork=lambda: patterns[0]._compile_subpattern_kernels(),
		prepare=prepare)

	return patterns


def _pack_patterns(patterns):
	"""What seqlets_to_patterns returned, as a forked child sends it: each
	pattern's seqlet coordinates, in order, and its subclusters."""

	if patterns is None:
		return None

	return [([(seqlet.example_idx, seqlet.start, seqlet.end, seqlet.is_revcomp)
		for seqlet in pattern.seqlets], pattern.subclusters)
		for pattern in patterns]


def _unpack_patterns(packed, track_set):
	"""The patterns _pack_patterns packed, made again here. seqlets_to_patterns
	makes each final pattern from new seqlets over the track set
	(aggregator._expand_seqlets_to_fill_pattern) and then attaches its
	subclusters (core.SeqletSet._set_subclusters). The same is done here with
	the same coordinates, in the same order: views of the same arrays with
	the same strides, summed in the same order, and the same subpatterns."""

	if packed is None:
		return None

	patterns = []
	for coordinates, subclusters in packed:
		pattern = core.SeqletSet(track_set.create_seqlets([core.Seqlet(
			example_idx=example_idx, start=start, end=end,
			is_revcomp=is_revcomp)
			for example_idx, start, end, is_revcomp in coordinates]))
		if subclusters is not None:
			pattern._set_subclusters(subclusters)
		patterns.append(pattern)

	return patterns


def TFMoDISco(one_hot, hypothetical_contribs, sliding_window_size=21, 
	flank_size=10, min_metacluster_size=100,
	weak_threshold_for_counting_sign=0.8, max_seqlets_per_metacluster=20000,
	target_seqlet_fdr=0.2, min_passing_windows_frac=0.03,
	max_passing_windows_frac=0.2, n_leiden_runs=50, n_leiden_iterations=-1, 
	min_overlap_while_sliding=0.7, nearest_neighbors_to_compute=500, 
	affmat_correlation_threshold=0.15, tsne_perplexity=10.0, 
	frac_support_to_trim_to=0.2, min_num_to_trim_to=30, trim_to_window_size=30, 
	initial_flank_to_add=10, final_flank_to_add=0,
	prob_and_pertrack_sim_merge_thresholds=[(0.8,0.8), (0.5, 0.85), (0.2, 0.9)],
	prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],
	subcluster_perplexity=50, merging_max_seqlets_subsample=1000,
	final_min_cluster_size=20, min_ic_in_window=0.6, min_ic_windowsize=6,
	ppm_pseudocount=0.001, verbose=False):

	# Compile the numba kernels in forked children while this runs. The CLI
	# has already called this before reading its inputs, and a second call
	# does nothing.
	precompile.start()

	# The package imports sklearn, igraph and leidenalg where they are used,
	# not when it is imported, so that the children above are forked sooner.
	# Importing them here, while the children compile, puts that time before
	# the run's first wait for a compiled kernel instead of later in the run.
	import sklearn.isotonic
	import sklearn.preprocessing
	import sklearn.manifold
	import igraph
	import leidenalg

	contrib_scores = np.multiply(one_hot, hypothetical_contribs)

	track_set = core.TrackSet(one_hot=one_hot, 
		contrib_scores=contrib_scores,
		hypothetical_contribs=hypothetical_contribs)

	seqlet_coords, threshold = extract_seqlets.extract_seqlets(
		attribution_scores=contrib_scores.sum(axis=2),
		window_size=sliding_window_size,
		flank=flank_size,
		suppress=(int(0.5*sliding_window_size) + flank_size),
		target_fdr=target_seqlet_fdr,
		min_passing_windows_frac=min_passing_windows_frac,
		max_passing_windows_frac=max_passing_windows_frac,
		weak_threshold_for_counting_sign=weak_threshold_for_counting_sign) 

	seqlets = track_set.create_seqlets(seqlet_coords) 

	pos_seqlets, neg_seqlets = [], []
	for seqlet in seqlets:
		flank = int(0.5*(len(seqlet)-sliding_window_size))
		attr = np.sum(seqlet.contrib_scores[flank:-flank])

		if attr > threshold:
			pos_seqlets.append(seqlet)
		elif attr < -threshold:
			neg_seqlets.append(seqlet)

	del seqlets

	# The negative metacluster's patterns depend only on its seqlets and the
	# track set, not on the positive metacluster's, so they are found in a
	# forked child while the positive metacluster's first clustering leaves
	# threads idle (seqlets_to_patterns, beside_clustering), or here, after
	# the positive metacluster, when no child ran or it failed.
	negative = None
	if len(neg_seqlets) > min_metacluster_size:
		neg_seqlets = neg_seqlets[:max_seqlets_per_metacluster]

		def find_negative():
			return seqlets_to_patterns(seqlets=neg_seqlets,
				track_set=track_set, 
				track_signs=-1,
				min_overlap_while_sliding=min_overlap_while_sliding,
				nearest_neighbors_to_compute=nearest_neighbors_to_compute,
				affmat_correlation_threshold=affmat_correlation_threshold,
				tsne_perplexity=tsne_perplexity,
				n_leiden_iterations=n_leiden_iterations,
				n_leiden_runs=n_leiden_runs,
				frac_support_to_trim_to=frac_support_to_trim_to,
				min_num_to_trim_to=min_num_to_trim_to,
				trim_to_window_size=trim_to_window_size,
				initial_flank_to_add=initial_flank_to_add,
				final_flank_to_add=final_flank_to_add,
				prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
				prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
				subcluster_perplexity=subcluster_perplexity,
				merging_max_seqlets_subsample=merging_max_seqlets_subsample,
				final_min_cluster_size=final_min_cluster_size,
				min_ic_in_window=min_ic_in_window,
				min_ic_windowsize=min_ic_windowsize,
				ppm_pseudocount=ppm_pseudocount)

		negative = forked.Background(find_negative, pack=_pack_patterns,
			unpack=lambda packed: _unpack_patterns(packed, track_set))

	if len(pos_seqlets) > min_metacluster_size:
		pos_seqlets = pos_seqlets[:max_seqlets_per_metacluster]
		if verbose:
			print("Using {} positive seqlets".format(len(pos_seqlets)))

		pos_patterns = seqlets_to_patterns(seqlets=pos_seqlets,
			track_set=track_set, 
			track_signs=1,
			min_overlap_while_sliding=min_overlap_while_sliding,
			nearest_neighbors_to_compute=nearest_neighbors_to_compute,
			affmat_correlation_threshold=affmat_correlation_threshold,
			tsne_perplexity=tsne_perplexity,
			n_leiden_iterations=n_leiden_iterations,
			n_leiden_runs=n_leiden_runs,
			frac_support_to_trim_to=frac_support_to_trim_to,
			min_num_to_trim_to=min_num_to_trim_to,
			trim_to_window_size=trim_to_window_size,
			initial_flank_to_add=initial_flank_to_add,
			final_flank_to_add=final_flank_to_add,
			prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
			prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
			subcluster_perplexity=subcluster_perplexity,
			merging_max_seqlets_subsample=merging_max_seqlets_subsample,
			final_min_cluster_size=final_min_cluster_size,
			min_ic_in_window=min_ic_in_window,
			min_ic_windowsize=min_ic_windowsize,
			ppm_pseudocount=ppm_pseudocount,
			beside_clustering=negative)
	else:
		pos_patterns = None

	if negative is not None:
		if verbose:
			print("Extracted {} negative seqlets".format(len(neg_seqlets)))

		neg_patterns = negative.result()
	else:
		neg_patterns = None

	return pos_patterns, neg_patterns
