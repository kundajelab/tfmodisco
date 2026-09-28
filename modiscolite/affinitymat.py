# affinitymat.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import sklearn
import sklearn.manifold

import math

import numpy as np

import scipy
from scipy.sparse import coo_matrix

from numba import njit
from numba import prange

from . import util
from . import gapped_kmer
from . import precompile


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
def _sparse_mm_dot_merge(X_data, X_indices, X_indptr, Y_data, Y_indices, Y_indptr, k):
	"""The pairwise merge-join version of `_sparse_mm_dot`, used only when a
	row of X or Y is not in canonical form (sorted, no duplicate columns),
	where the inverted index would not reproduce the merge join."""

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


@njit
def _lt(a, b):
	"""numba's float `<` for sorting: NaN after every number."""
	return a < b or (np.isnan(b) and not np.isnan(a))


@njit
def _after(a, b, dot):
	"""Whether index `a` comes after index `b` in the stable argsort of -dot."""
	va = -dot[a]
	vb = -dot[b]
	return _lt(vb, va) or (not _lt(va, vb) and a > b)


@njit
def _sift_down(heap, size, dot):
	pos = 0
	while True:
		child = 2 * pos + 1
		if child >= size:
			break

		if child + 1 < size and _after(heap[child + 1], heap[child], dot):
			child += 1

		if _after(heap[child], heap[pos], dot):
			tmp = heap[pos]
			heap[pos] = heap[child]
			heap[child] = tmp
			pos = child
		else:
			break


@njit(parallel=True)
def _sparse_mm_dot(X_data, X_indptr, X_cols, Xt_indptr, Xt_rows, Xt_data,
	Yt_indptr, Yt_rows, Yt_data, k):
	"""For each row i of X, the k rows j with the largest max(X_i.X_j,
	X_i.Y_j) and those values, in the order of a stable argsort of the
	negated values.

	X is in CSR form with its columns renumbered to a compact range
	(`X_cols`); Xt and Yt are X and Y in CSC form over the same compact
	columns. Every product X_i[c] * X_j[c] is added to dot[j] while walking
	row i's columns in ascending order, so each dot is summed over the
	shared columns in the same order, from the same +0.0, as the pairwise
	merge join of two sorted rows, and is bitwise equal to it. Only pairs
	that share a column cost anything.

	The top k are kept in a heap whose root is the entry that comes last in
	the order the argsort would give (value descending, NaN last, ties by
	ascending index), then written out best first.
	"""

	n_rows = len(X_indptr) - 1

	neighbors = np.empty((n_rows, k), dtype='int32')
	sims = np.empty((n_rows, k), dtype='float64')

	for i in prange(n_rows):
		dot = np.zeros(n_rows, dtype='float64')
		ydot = np.zeros(n_rows, dtype='float64')

		for p in range(X_indptr[i], X_indptr[i+1]):
			c = X_cols[p]
			x_data = X_data[p]

			for q in range(Xt_indptr[c], Xt_indptr[c+1]):
				dot[Xt_rows[q]] += x_data * Xt_data[q]

			for q in range(Yt_indptr[c], Yt_indptr[c+1]):
				ydot[Yt_rows[q]] += x_data * Yt_data[q]

		for j in range(n_rows):
			dot[j] = max(dot[j], ydot[j])

		heap = np.empty(k, dtype=np.int64)
		for j in range(k):
			heap[j] = j
			pos = j
			while pos > 0:
				parent = (pos - 1) // 2
				if _after(heap[pos], heap[parent], dot):
					tmp = heap[pos]
					heap[pos] = heap[parent]
					heap[parent] = tmp
					pos = parent
				else:
					break

		# j is larger than every index in the heap, so it enters exactly when
		# its value sorts strictly before the root's.
		for j in range(k, n_rows):
			if _lt(-dot[j], -dot[heap[0]]):
				heap[0] = j
				_sift_down(heap, k, dot)

		for size in range(k, 0, -1):
			neighbors[i, size-1] = heap[0]
			sims[i, size-1] = dot[heap[0]]
			heap[0] = heap[size-1]
			_sift_down(heap, size-1, dot)

	return sims, neighbors


def _sparse_knn(matrices, k):
	"""The k nearest rows of X for each row of X by max(X_i.X_j, X_i.Y_j),
	for two CSR matrices with the same number of rows, given as the list
	[X, Y].

	The list is emptied, and each of X's and Y's arrays is dropped as soon
	as it is no longer needed: the index arrays once they are sorted, and
	all of Y once its CSC form is built. A caller that keeps no other
	reference to X and Y has that memory back before the kernel runs. Y's
	CSC form is built before X's, so that its temporaries are not alive
	alongside X's CSC arrays.
	"""

	X, Y = matrices
	matrices.clear()

	if not (X.has_canonical_format and Y.has_canonical_format):
		return _sparse_mm_dot_merge(X.data, X.indices, X.indptr, Y.data, 
			Y.indices, Y.indptr, k)

	n, nnz = X.shape[0], X.nnz
	X_data, X_indptr, X_indices = X.data, X.indptr, X.indices
	Y_data, Y_indptr, Y_indices = Y.data, Y.indptr, Y.indices
	del X, Y

	# Sort X's entries by column, which gives X in CSC form and renumbers
	# the hashed columns (up to 5**15) to 0..len(cols)-1. A row holds each
	# column at most once, so the order of the rows inside one column
	# changes no sum and the sort need not be stable.
	order_x = np.argsort(X_indices)
	sorted_cols = X_indices[order_x]
	del X_indices
	first = np.empty(nnz, dtype=bool)
	first[:1] = True
	np.not_equal(sorted_cols[1:], sorted_cols[:-1], out=first[1:])
	cols = sorted_cols[first]
	del sorted_cols

	# Y in CSC form over the same columns, without the columns X lacks,
	# which no row of X visits. -1 is below every column index.
	order = np.argsort(Y_indices)
	sorted_cols = Y_indices[order]
	del Y_indices
	pos = np.searchsorted(cols, sorted_cols)
	keep = np.append(cols, -1)[pos] == sorted_cols
	del sorted_cols

	order = order[keep]
	Yt_indptr = np.searchsorted(pos[keep], np.arange(len(cols) + 1))
	del pos, keep

	rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(Y_indptr))
	Yt_rows = rows[order]
	Yt_data = Y_data[order]
	del order, rows, Y_data, Y_indptr

	# X in CSC form, from the sort above.
	X_cols = np.empty(nnz, dtype=np.int64)
	ranks = np.cumsum(first)
	ranks -= 1
	X_cols[order_x] = ranks
	del ranks
	Xt_indptr = np.append(np.flatnonzero(first), nnz)
	del first

	rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(X_indptr))
	Xt_rows = rows[order_x]
	Xt_data = X_data[order_x]
	del order_x, rows

	return _sparse_mm_dot(X_data, np.asarray(X_indptr, dtype=np.int64), 
		X_cols, Xt_indptr, Xt_rows, Xt_data, Yt_indptr, Yt_rows, Yt_data, k)


def cosine_similarity_from_seqlets(seqlets, n_neighbors, sign, topn=20, 
	min_k=4, max_k=6, max_gap=15, max_len=15, max_entries=500, 
	alphabet_size=4):

	X_fwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, 
		min_k, max_k, max_gap, max_len, max_entries, True, sign)

	X_bwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, min_k, max_k, max_gap, 
			max_len, max_entries, False, sign)

	X = sklearn.preprocessing.normalize(X_fwd, norm='l2', axis=1)
	Y = sklearn.preprocessing.normalize(X_bwd, norm='l2', axis=1)
	del X_fwd, X_bwd

	n, d = X.shape
	k = min(n_neighbors+1, n)

	# _sparse_knn frees X's and Y's arrays as it goes, once nothing here
	# holds them.
	matrices = [X, Y]
	del X, Y
	return _sparse_knn(matrices, k)


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

	all_neighbors = seqlet_neighbors is None

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		pad_width = ((0,0), (n_pad, n_pad), (0,0)) 
		Y = np.pad(array=Y, pad_width=pad_width, mode="constant")
	else:
		n_pad = 0 

	len_output = 1 + Y.shape[1] - X.shape[1] 

	X = np.ascontiguousarray(X, dtype='float32')
	Y = np.ascontiguousarray(Y, dtype='float32')

	size = X.shape[1] * X.shape[2]
	if all_neighbors and len_output == 1 and X.shape[2] == Y.shape[2] and size > 0:
		# Every row of Y against every row of X at the one offset, as
		# SimilarPatternsCollapser calls it: the scores `_jaccard_best`
		# would give with neighbours 0..nx-1, from a kernel whose loop over
		# the neighbours vectorizes. A row flattens to its positions and
		# channels in (j, k) order, the order each score is summed in.
		# Empty rows, and inputs holding a NaN or an infinity (the kernel
		# returns 0), stay on the path below.
		n, k = Y.shape[0], X.shape[0]
		best = np.empty((n, k), dtype='float32')
		if _jaccard_all(X.reshape(k, size), Y.reshape(n, size), best):
			if return_sparse != True:
				results = np.zeros((n, k, 2))
				results[:, :, 0] = best
				results[:, :, 1] = -n_pad
				return results

			# As below: a NaN or -0.0 score sends the call to numpy's max.
			if not (np.isnan(best) | ((best == 0) & np.signbit(best))).any():
				return best

	if all_neighbors:
		seqlet_neighbors = np.tile(np.arange(X.shape[0]), (Y.shape[0], 1))

	seqlet_neighbors = np.ascontiguousarray(seqlet_neighbors.astype('int32'))
	n, k = Y.shape[0], seqlet_neighbors.shape[1]

	if len_output == 1 or (len_output > 1 and X.shape[2] % 2 == 0):
		best = np.empty((n, k), dtype='float32')
		best_idx = np.empty((n, k), dtype='int64')
		flags = np.zeros(n, dtype='uint8')
		_jaccard_best(X, Y, seqlet_neighbors, best, best_idx, flags)

		if return_sparse != True:
			results = np.zeros((n, k, 2))
			results[:, :, 0] = best
			results[:, :, 1] = best_idx - n_pad
			return results

		# The largest score is unique in its bits unless a row holds a NaN
		# or a -0.0; for those, numpy's max decides which bits come back.
		if not flags.any():
			return best

	scores = np.zeros((n, k, len_output), dtype='float32')
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


# Every score at every offset. `jaccard` uses it only where `_jaccard_best`
# cannot stand in (a NaN or -0.0 score under return_sparse, an odd number of
# channels), so it is compiled on first use.
@njit(parallel=True)
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


@precompile.signature('void(float32[:, :, ::1], float32[:, :, ::1], int32[:, ::1], float32[:, ::1], int64[:, ::1], uint8[::1])')
@njit(parallel=True)
def _jaccard_best(X, Y, neighbors, best, best_idx, flags):
	"""For each (l, i), the largest score over all offsets and the first
	offset that reaches it, as `np.argmax` picks it (a NaN wins), without
	storing every offset's score. Each score is summed in the same order as
	in `_jaccard` and rounded to float32 before the comparison, so both are
	bit-identical to taking them from `_jaccard`'s scores. flags[l] is set
	when row l holds a NaN or a -0.0 score. With more than one offset, the
	number of channels must be even."""

	nx, d, m = X.shape
	ny, L, _ = Y.shape
	n_neighbors = neighbors.shape[1]
	len_output = L - d + 1

	# The offsets are padded to a multiple of 8 so the vectorized loop over
	# them has no scalar remainder; the padding offsets are never read.
	n_offsets = (len_output + 7) // 8 * 8
	L_pad = d - 1 + n_offsets

	for l in prange(ny):
		flag = 0

		if len_output == 1:
			for i in range(n_neighbors):
				min_sum = 0.0
				max_sum = 0.0
				neighbor_li = neighbors[l, i]

				for j in range(d):
					for k in range(m):
						sign = np.sign(X[neighbor_li, j, k]) * np.sign(Y[l, j, k])

						x = abs(X[neighbor_li, j, k])
						y = abs(Y[l, j, k])

						if y > x:
							min_sum += x * sign
							max_sum += y
						else:
							min_sum += y * sign
							max_sum += x

				score = np.float32(min_sum / max_sum)
				if score != score or (score == 0 and math.copysign(1.0, score) < 0):
					flag = 1

				best[l, i] = score
				best_idx[l, i] = 0

		else:
			# |Y| and sign(Y) for this row, channel-major, so that the
			# offsets of one (position, channel) pair are contiguous. Zero
			# past the end.
			abs_Y = np.empty((m, L_pad), dtype=np.float32)
			sign_Y = np.empty((m, L_pad), dtype=np.float32)
			for j in range(L):
				for k in range(m):
					abs_Y[k, j] = abs(Y[l, j, k])
					sign_Y[k, j] = np.sign(Y[l, j, k])
			for k in range(m):
				for j in range(L, L_pad):
					abs_Y[k, j] = 0.0
					sign_Y[k, j] = 0.0

			min_sums = np.empty(n_offsets, dtype=np.float64)
			max_sums = np.empty(n_offsets, dtype=np.float64)

			for i in range(n_neighbors):
				neighbor_li = neighbors[l, i]
				for idx in range(n_offsets):
					min_sums[idx] = 0.0
					max_sums[idx] = 0.0

				# Offsets innermost, two channels per pass: each offset's
				# sums still take their terms in (j, k) order.
				for j in range(d):
					for k in range(0, m, 2):
						x0 = abs(X[neighbor_li, j, k])
						x1 = abs(X[neighbor_li, j, k+1])
						sign_x0 = np.sign(X[neighbor_li, j, k])
						sign_x1 = np.sign(X[neighbor_li, j, k+1])

						for idx in range(n_offsets):
							min_sum = min_sums[idx]
							max_sum = max_sums[idx]

							y = abs_Y[k, j + idx]
							sign = sign_x0 * sign_Y[k, j + idx]
							gt = y > x0
							min_sum += (x0 if gt else y) * sign
							max_sum += y if gt else x0

							y = abs_Y[k+1, j + idx]
							sign = sign_x1 * sign_Y[k+1, j + idx]
							gt = y > x1
							min_sum += (x1 if gt else y) * sign
							max_sum += y if gt else x1

							min_sums[idx] = min_sum
							max_sums[idx] = max_sum

				best_score = np.float32(0.0)
				best_offset = 0
				for idx in range(len_output):
					score = np.float32(min_sums[idx] / max_sums[idx])
					if score != score or (score == 0 and math.copysign(1.0, score) < 0):
						flag = 1

					if idx == 0 or (best_score == best_score and not (score <= best_score)):
						best_score = score
						best_offset = idx

				best[l, i] = best_score
				best_idx[l, i] = best_offset

		flags[l] = flag


@precompile.signature('uint8(float32[:, ::1], float32[:, ::1], float32[:, ::1])')
@njit(parallel=True)
def _jaccard_all(X, Y, best):
	"""best[l, i]: the score of row l of Y against row i of X, for every
	row of X, at the one offset, with each row flattened to its positions
	and channels in (j, k) order. Bit-identical to `_jaccard_best`'s
	single-offset branch with neighbours 0..nx-1: each score's float64 sums
	take the same terms in the same order, followed by the same division and
	float32 rounding. The neighbours are innermost, one pair of sums each,
	so that loop vectorizes; each score is written by one thread.

	Returns 0, leaving best unfinished, when X or Y holds a NaN or an
	infinity. Only a NaN's payload and sign could then differ from
	`_jaccard_best`'s (which operand's NaN an operation returns depends on
	the generated code); with finite values the only NaN is 0/0's."""

	nx, D = X.shape
	ny = Y.shape[0]

	# X transposed, so that one position's values across the neighbours are
	# contiguous, and padded with zeros to a multiple of 16 neighbours so
	# that the vector loop has no scalar remainder. The padding's scores are
	# computed and never read. (np.empty and a loop: np.zeros of a 2-D
	# array costs about 0.25 s more to compile.)
	n_pad = (nx + 15) // 16 * 16
	X_t = np.empty((D, n_pad), dtype=np.float32)
	finite = True
	for j in range(D):
		for i in range(nx):
			v = X[i, j]
			if not math.isfinite(v):
				finite = False
			X_t[j, i] = v
		for i in range(nx, n_pad):
			X_t[j, i] = 0.0

	if not finite:
		return 0

	finite_rows = np.empty(ny, dtype=np.uint8)
	for l in prange(ny):
		min_sums = np.zeros(n_pad, dtype=np.float64)
		max_sums = np.zeros(n_pad, dtype=np.float64)
		row_finite = 1

		for j in range(D):
			if not math.isfinite(Y[l, j]):
				row_finite = 0
			y = abs(Y[l, j])
			sign_y = np.sign(Y[l, j])

			for i in range(n_pad):
				v = X_t[j, i]
				x = abs(v)
				sign = np.sign(v) * sign_y
				gt = y > x
				min_sums[i] += (x if gt else y) * sign
				max_sums[i] += y if gt else x

		for i in range(nx):
			best[l, i] = np.float32(min_sums[i] / max_sums[i])
		finite_rows[l] = row_finite

	for l in range(ny):
		if finite_rows[l] == 0:
			return 0
	return 1


def pearson_correlation(X, Y, min_overlap=None, func=np.ceil):
	if X.ndim == 2:
		X = X[None, :, :]
	if Y.ndim == 2:
		Y = Y[None, :, :]

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		if n_pad >= 0:
			# np.pad(mode="constant") without its Python overhead: a new
			# C-ordered array of Y's dtype, zeros around a copy of Y.
			m, l, k = Y.shape
			Y_padded = np.zeros((m, l + 2*n_pad, k), dtype=Y.dtype)
			Y_padded[:, n_pad:n_pad+l] = Y
			Y = Y_padded
		else:
			pad_width = ((0, 0), (n_pad, n_pad), (0, 0))
			Y = np.pad(array=Y, pad_width=pad_width, mode="constant")

	n, d, _ = X.shape
	len_output = 1 + Y.shape[1] - d 
	scores = np.zeros((n, len_output))

	if (n == 1 and Y.shape[0] == 1 and X.shape[2] == Y.shape[2]
			and len_output > 0 and X.dtype in (np.float32, np.float64)
			and Y.dtype in (np.float32, np.float64)
			and Y.flags.c_contiguous):
		# The loop below with its loop-invariant work taken out. Every
		# value comes from the same operation on the same operands:
		# np.linalg.norm(Y_) is sqrt(Y_.ravel().dot(itself)), so each
		# offset's window (a contiguous view, as Y_.ravel() is) is dotted
		# with itself; sqrt and the division are elementwise and exact;
		# np.dot casts both vectors to their common dtype, as done here,
		# and each score is one np.dot of the same two contiguous vectors
		# of the same length; nan_to_num maps each element as on a scalar.
		k = Y.shape[2]
		Y_flat = Y.reshape(-1)
		windows = np.lib.stride_tricks.as_strided(Y_flat,
			shape=(len_output, d*k), strides=(k*Y_flat.itemsize,
			Y_flat.itemsize), writeable=False)

		sqnorms = np.array(list(map(np.ndarray.dot, windows, windows)),
			dtype=Y.dtype)
		normed = windows / np.sqrt(sqnorms)[:, None]

		X_normed = (X / np.linalg.norm(X)).ravel()
		dtype = np.result_type(X_normed, normed)
		X_normed = X_normed.astype(dtype, copy=False)
		normed = normed.astype(dtype, copy=False)

		scores_ = np.array(list(map(X_normed.dot, normed)), dtype=dtype)
		scores[0] = np.nan_to_num(scores_)
	else:
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

