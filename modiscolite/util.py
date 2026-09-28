# util.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>, Ivy Raine <ivy.ember.raine@gmail.com>
# adapted from code written by Avanti Shrikumar 

from enum import Enum
from operator import attrgetter
import textwrap
from typing import List
import numpy as np
from numba import njit

from . import precompile


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


@precompile.signature('float64(float64, float64[:])')
@njit
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


# The kernels below build the density-adapted affinity matrix of
# tfmodisco._density_adaptation. Each one keeps every value's bits: they move
# values and apply the IEEE operations scipy's and numpy's code applied, in
# the same order. The exponentials and logarithms stay in numpy (numba's
# np.exp differs from numpy's in the last bit for some arguments), except
# those inside binary_search_perplexity, which was already compiled.

@precompile.signature('int64(int64[::1], int64[::1], float64[::1], int64, '
	'int64[::1], int32[::1], float64[::1])')
@njit(nogil=True)
def _transpose_triples(offsets, cols, values, n, t_indptr, t_indices, t_data):
	"""The transpose, as a CSR matrix with sorted indices and no stored
	zeros, of the n x n matrix whose row r holds values[e] in column cols[e]
	for e in offsets[r]:offsets[r+1].

	Returns 0, or 1 when a column lies outside [0, n), or 2 when a row holds
	one column twice (scipy would sum those entries), or 3 when a value is
	NaN. The outputs are then undefined. Zeros are dropped only after the
	duplicate check, which looks at every entry. t_indptr has n+1 entries,
	t_indices and t_data len(cols).

	Which of two NaNs an addition returns depends on the order in which the
	compiler put its operands, so NaNs of both signs in one sum could come
	out with another sign bit than scipy's and numpy's. Without a NaN here,
	none arises later: every value stays finite."""

	nnz = cols.shape[0]
	for e in range(nnz):
		if cols[e] < 0 or cols[e] >= n:
			return 1
		if values[e] != values[e]:
			return 3

	for c in range(n + 1):
		t_indptr[c] = 0
	for e in range(nnz):
		t_indptr[cols[e] + 1] += 1
	for c in range(n):
		t_indptr[c + 1] += t_indptr[c]

	# A stable counting sort by column: each column's rows come out in
	# increasing order.
	fill = t_indptr[:n].copy()
	for r in range(n):
		for e in range(offsets[r], offsets[r + 1]):
			c = cols[e]
			p = fill[c]
			t_indices[p] = r
			t_data[p] = values[e]
			fill[c] = p + 1

	for c in range(n):
		for p in range(t_indptr[c] + 1, t_indptr[c + 1]):
			if t_indices[p] == t_indices[p - 1]:
				return 2

	# Drop the zeros (scipy's eliminate_zeros), keeping the order.
	w = 0
	for c in range(n):
		start, end = t_indptr[c], t_indptr[c + 1]
		t_indptr[c] = w
		for p in range(start, end):
			if t_data[p] != 0:
				t_indices[w] = t_indices[p]
				t_data[w] = t_data[p]
				w += 1
	t_indptr[n] = w
	return 0


@precompile.signature('void(int64[::1], int32[::1], float64[::1], '
	'int64[::1], int32[::1], float64[::1])')
@njit(nogil=True)
def _transpose(indptr, indices, data, t_indptr, t_indices, t_data):
	"""The transpose of a square CSR matrix, with sorted indices (a stable
	counting sort by column, as scipy's csr_tocsc)."""

	n = indptr.shape[0] - 1
	for c in range(n + 1):
		t_indptr[c] = 0
	for p in range(indptr[n]):
		t_indptr[indices[p] + 1] += 1
	for c in range(n):
		t_indptr[c + 1] += t_indptr[c]

	fill = t_indptr[:n].copy()
	for r in range(n):
		for p in range(indptr[r], indptr[r + 1]):
			c = indices[p]
			q = fill[c]
			t_indices[q] = r
			t_data[q] = data[p]
			fill[c] = q + 1


@precompile.signature('int64(int64[::1], int32[::1], float64[::1], '
	'int64[::1], int32[::1], float64[::1], int64[::1], int32[::1], '
	'float64[::1])')
@njit(nogil=True)
def _symmetrize(a_indptr, a_indices, a_data, b_indptr, b_indices, b_data,
	s_indptr, s_indices, s_data):
	"""(A + B) / (C + C_B), where B is A's transpose and C and C_B hold a
	1.0 wherever A and B store an entry: the value of
	`A += A.T; C += C.T; A.data /= C.data` in scipy, for A and B with sorted
	indices. scipy's csr_binop_csr_canonical merges the two rows, computing
	a + b where both store an entry, a + 0 or 0 + b where one does, and the
	counts the same way; this does the same additions and the one division.
	scipy drops sums that are zero; A's entries are positive or NaN, so no
	sum is. Returns the number of entries written."""

	n = a_indptr.shape[0] - 1
	w = 0
	s_indptr[0] = 0
	for i in range(n):
		p, p_end = a_indptr[i], a_indptr[i + 1]
		q, q_end = b_indptr[i], b_indptr[i + 1]
		while p < p_end and q < q_end:
			ja, jb = a_indices[p], b_indices[q]
			if ja == jb:
				s_indices[w] = ja
				s_data[w] = (a_data[p] + b_data[q]) / (1.0 + 1.0)
				p += 1
				q += 1
			elif ja < jb:
				s_indices[w] = ja
				s_data[w] = (a_data[p] + 0.0) / (1.0 + 0.0)
				p += 1
			else:
				s_indices[w] = jb
				s_data[w] = (0.0 + b_data[q]) / (0.0 + 1.0)
				q += 1
			w += 1

		while p < p_end:
			s_indices[w] = a_indices[p]
			s_data[w] = (a_data[p] + 0.0) / (1.0 + 0.0)
			p += 1
			w += 1

		while q < q_end:
			s_indices[w] = b_indices[q]
			s_data[w] = (0.0 + b_data[q]) / (0.0 + 1.0)
			q += 1
			w += 1

		s_indptr[i + 1] = w
	return w


@precompile.signature('void(float64, float64[:], int64[::1], int64, int64, '
	'float64[::1])')
@njit(nogil=True)
def _perplexity_betas(desired_perplexity, data, indptr, start, stop, betas):
	"""betas[i] = binary_search_perplexity(desired_perplexity, row i of the
	CSR matrix's data) for rows start to stop. The rows are passed with the
	layout binary_search_perplexity was declared with, so this runs its
	code. It releases the GIL, so threads can share the rows."""

	for i in range(start, stop):
		betas[i] = binary_search_perplexity(desired_perplexity,
			data[indptr[i]:indptr[i + 1]])


@njit(nogil=True)
def _pairwise_sum(a, lo, n):
	"""numpy's pairwise summation of a[lo:lo+n] (pairwise_sum in numpy's
	loops_utils.h.src): below 8 values a plain loop, up to 128 values eight
	accumulators combined as ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7)) and then
	the remainder, and above that two halves, the first a multiple of 8."""

	if n < 8:
		res = -0.0
		for i in range(n):
			res += a[lo + i]
		return res
	elif n <= 128:
		r0 = a[lo]
		r1 = a[lo + 1]
		r2 = a[lo + 2]
		r3 = a[lo + 3]
		r4 = a[lo + 4]
		r5 = a[lo + 5]
		r6 = a[lo + 6]
		r7 = a[lo + 7]
		m = n - n % 8
		i = 8
		while i < m:
			r0 += a[lo + i]
			r1 += a[lo + i + 1]
			r2 += a[lo + i + 2]
			r3 += a[lo + i + 3]
			r4 += a[lo + i + 4]
			r5 += a[lo + i + 5]
			r6 += a[lo + i + 6]
			r7 += a[lo + i + 7]
			i += 8
		res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
		while i < n:
			res += a[lo + i]
			i += 1
		return res
	else:
		n2 = n // 2
		n2 -= n2 % 8
		return _pairwise_sum(a, lo, n2) + _pairwise_sum(a, lo + n2, n - n2)


@precompile.signature('void(float64[::1], int64[::1], float64[::1])')
@njit(nogil=True)
def _row_sums_plus_one(values, indptr, out):
	"""out[i] = values[indptr[i]:indptr[i+1]].sum() + 1, with numpy's order
	of additions: its reduction starts from 0 and adds the pairwise sum of
	each 8,192-value block of the row in turn."""

	for i in range(out.shape[0]):
		lo = indptr[i]
		n = indptr[i + 1] - lo
		s = 0.0
		for b in range(0, n, 8192):
			s += _pairwise_sum(values, lo + b, min(8192, n - b))
		out[i] = s + 1


@precompile.signature('int64(int64[::1], int32[::1], float64[::1], '
	'float64[::1], int32[::1], int32[::1], float64[::1])')
@njit(nogil=True)
def _add_diagonal(indptr, indices, data, diag, out_indptr, out_indices,
	out_data):
	"""A + D for a CSR matrix A with sorted indices and D the diagonal
	matrix of diag, as scipy's `A + scipy.sparse.diags(diag)`: D stores
	only diag's nonzero values, and csr_binop_csr_canonical merges each row,
	computing a + d where both store an entry, a + 0 or 0 + d where one
	does, and drops the sums that are zero. Returns the number of entries
	written; out_indices and out_data need room for A's entries plus n."""

	n = indptr.shape[0] - 1
	w = 0
	out_indptr[0] = 0
	for i in range(n):
		d = diag[i]
		pending = d != 0
		for p in range(indptr[i], indptr[i + 1]):
			j = indices[p]
			if pending and j >= i:
				pending = False
				if j == i:
					x = data[p] + d
					if x != 0:
						out_indices[w] = j
						out_data[w] = x
						w += 1
					continue

				x = 0.0 + d
				if x != 0:
					out_indices[w] = i
					out_data[w] = x
					w += 1

			x = data[p] + 0.0
			if x != 0:
				out_indices[w] = j
				out_data[w] = x
				w += 1

		if pending:
			x = 0.0 + d
			if x != 0:
				out_indices[w] = i
				out_data[w] = x
				w += 1

		out_indptr[i + 1] = w
	return w


@precompile.signature('uint8(float32[:, :, :], float32[:, :, :], '
	'float32[:, :, :], int64[::1], int64[::1], uint8[::1], float64[:, ::1], '
	'float64[:, ::1], float64[:, ::1])')
@njit(nogil=True)
def _seqlet_sums(one_hot, contrib_scores, hypothetical_contribs, example_idx,
	starts, revcomp, sequence_sum, contrib_sum, hypothetical_sum):
	"""Add the three tracks of each seqlet, in seqlet order, into the
	running sums, which start at zero: the seqlet at (example_idx[s],
	starts[s], revcomp[s]) and as long as the sums, read from the track
	arrays as `core.TrackSet.create_seqlets` makes its views (the reverse
	strand as [::-1, ::-1]). Each sum adds its values one seqlet after
	another, each converted exactly to float64, which is the order of
	`SeqletSet.__init__`'s reductions over the stacked seqlets. Returns 0
	when a value is a NaN, whose payload could then depend on the order of
	each addition's operands; the sums are then unfinished."""

	n, k = sequence_sum.shape
	for s in range(example_idx.shape[0]):
		e = example_idx[s]
		start = starts[s]
		for c in range(k):
			if revcomp[s]:
				ci = k - 1 - c
				for j in range(n):
					p = start + n - 1 - j
					a = one_hot[e, p, ci]
					b = contrib_scores[e, p, ci]
					h = hypothetical_contribs[e, p, ci]
					if a != a or b != b or h != h:
						return 0
					sequence_sum[j, c] += a
					contrib_sum[j, c] += b
					hypothetical_sum[j, c] += h
			else:
				for j in range(n):
					p = start + j
					a = one_hot[e, p, c]
					b = contrib_scores[e, p, c]
					h = hypothetical_contribs[e, p, c]
					if a != a or b != b or h != h:
						return 0
					sequence_sum[j, c] += a
					contrib_sum[j, c] += b
					hypothetical_sum[j, c] += h
	return 1


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

# Below this many patterns the per-pattern loop is faster than stacking.
_MIN_STACKED = 2

_get_shape = attrgetter('shape')
_get_dtype = attrgetter('dtype')
_get_strides = attrgetter('strides')


def _l1_stacked(patterns, tracks):
	"""Apply `l1` to every pattern's snippets at once.

	Returns the same two arrays as the per-pattern loop in
	`get_2d_data_from_patterns`, bit for bit, or None when the snippets differ
	in shape, dtype or memory order and cannot be stacked.

	`l1` sums |X| over an (L, w) snippet in the memory order of `np.abs(X)`,
	which numpy lays out by the snippet's absolute strides: position-major
	when the snippet is C-like, base-major for the transposed layout the CLI
	makes from npz input. Each row of `flat` below holds one snippet's |X| in
	that order, so reducing a row runs numpy's pairwise sum over the same
	sequence of values. The reverse complement X[::-1, ::-1] has the same
	absolute strides, so its sum runs over the same row read backwards.
	"""

	n = len(patterns)
	shape, dtype = None, None
	per_track = []

	for track in tracks:
		snippets = [getattr(pattern, track) for pattern in patterns]

		shapes = set(map(_get_shape, snippets))
		dtypes = set(map(_get_dtype, snippets))
		if len(shapes) != 1 or len(dtypes) != 1:
			return None

		track_shape, track_dtype = shapes.pop(), dtypes.pop()
		if shape is None:
			shape, dtype = track_shape, track_dtype
		elif track_shape != shape or track_dtype != dtype:
			return None

		if len(shape) != 2 or shape[0] == 0 or shape[1] == 0:
			return None

		if shape[0] == 1 or shape[1] == 1:
			base_major = False
		else:
			orders = set()
			for strides in set(map(_get_strides, snippets)):
				if abs(strides[0]) == abs(strides[1]):
					return None
				orders.add(abs(strides[0]) < abs(strides[1]))

			if len(orders) != 1:
				return None

			base_major = orders.pop()

		# Row k of `stacked` is snippet k laid out as np.abs lays it out:
		# (w, L) when base-major, (L, w) otherwise.
		if base_major:
			stacked = np.array([snippet.T for snippet in snippets])
		else:
			stacked = np.array(snippets)

		flat = np.abs(stacked).reshape(n, -1)
		fwd_sums = np.sum(flat, axis=1)
		rev_sums = np.sum(np.ascontiguousarray(flat[:, ::-1]), axis=1)

		# `l1` returns X unchanged when its sum is zero; X / 1 is X.
		fwd_sums[fwd_sums == 0] = 1
		rev_sums[rev_sums == 0] = 1
		per_track.append((stacked, base_major, fwd_sums, rev_sums))

	length, width = shape
	all_fwd_data = np.empty((n, length, width * len(tracks)), dtype=dtype)
	all_rev_data = np.empty((n, length, width * len(tracks)), dtype=dtype)

	# Divide in the contiguous stacked layout, then copy into (n, L, w) order.
	for i, (stacked, base_major, fwd_sums, rev_sums) in enumerate(per_track):
		columns = slice(i * width, (i + 1) * width)

		fwd = stacked / fwd_sums[:, None, None]
		rev = (stacked / rev_sums[:, None, None])[:, ::-1, ::-1]
		if base_major:
			fwd, rev = fwd.transpose(0, 2, 1), rev.transpose(0, 2, 1)

		all_fwd_data[:, :, columns] = fwd
		all_rev_data[:, :, columns] = rev

	return all_fwd_data, all_rev_data


def get_2d_data_from_patterns(patterns, transformer='l1', include_hypothetical=True):
	func = l1 if transformer == 'l1' else magnitude
	tracks = ['hypothetical_contribs', 'contrib_scores']
	if not include_hypothetical:
		tracks = tracks[1:]

	if transformer == 'l1' and isinstance(patterns, list) and len(patterns) >= _MIN_STACKED:
		stacked = _l1_stacked(patterns, tracks)
		if stacked is not None:
			return stacked

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
