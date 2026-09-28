# affinitymat.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import math

from concurrent.futures import ThreadPoolExecutor

import numpy as np

import scipy
from scipy.sparse import coo_matrix

import numba
from numba import njit
from numba import prange
from numba import uint64

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

		# Every index here is non-negative; unsigned casts spare numba's
		# negative-index fix-up on each access.
		for p in range(X_indptr[i], X_indptr[i+1]):
			p = uint64(p)
			c = uint64(X_cols[p])
			x_data = X_data[p]

			for q in range(Xt_indptr[c], Xt_indptr[c + uint64(1)]):
				q = uint64(q)
				dot[uint64(Xt_rows[q])] += x_data * Xt_data[q]

			for q in range(Yt_indptr[c], Yt_indptr[c + uint64(1)]):
				q = uint64(q)
				ydot[uint64(Yt_rows[q])] += x_data * Yt_data[q]

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


def _sort_by_column(holder, n_cols, packed, order):
	"""Sort the column indices (below n_cols) in the one-element list
	`holder`, and find the permutation that sorts them, in the caller's
	arrays: `order` holds 0..len(indices)-1 on entry and the permutation on
	return, and `packed` the sorted indices. Each index is packed with its
	position into one int64, index in the high bits, and the packed values
	are sorted in place, which is several times faster than an argsort;
	ties come out in position order. The list is emptied once the indices
	are copied, so that they can be freed. Returns False, having changed
	nothing, when the two do not fit in 63 bits. Callers must not depend on
	the order of equal indices."""

	nnz = len(holder[0])
	shift = max(1, int(nnz).bit_length())
	if nnz == 0 or max(1, int(n_cols - 1).bit_length()) + shift > 63:
		return False

	np.copyto(packed, holder.pop())
	np.left_shift(packed, shift, out=packed)
	np.bitwise_or(packed, order, out=packed)
	packed.sort()

	np.bitwise_and(packed, (1 << shift) - 1, out=order)
	np.right_shift(packed, shift, out=packed)
	return True


def _sorted_by_column(holder, n_cols, pool=None):
	"""(order, sorted indices) for the column indices in `holder`, by
	`_sort_by_column`, run in `pool` when one is given (a Future is then
	returned), or by np.argsort when they cannot be packed."""

	nnz = len(holder[0])
	packed, order = np.empty(nnz, dtype=np.int64), np.arange(nnz)

	def run():
		if _sort_by_column(holder, n_cols, packed, order):
			return order, packed

		indices = holder.pop()
		order_ = np.argsort(indices)
		return order_, indices[order_]

	return run() if pool is None else pool.submit(run)


def _block_offsets(mask, step):
	"""The number of True entries of `mask` before each block of `step`
	entries, and the total, as an array of len(range(0, len(mask), step))
	+ 1 values."""

	starts = np.arange(0, len(mask), step)
	offsets = np.zeros(len(starts) + 1, dtype=np.int64)
	if len(starts):
		np.cumsum(np.add.reduceat(mask, starts, dtype=np.int64), 
			out=offsets[1:])
	return offsets


def _sparse_knn(matrices, k):
	"""The k nearest rows of X for each row of X by max(X_i.X_j, X_i.Y_j),
	for two CSR matrices with the same number of rows, given as the list
	[X, Y].

	The list is emptied, and each of X's and Y's arrays is dropped as soon
	as it is no longer needed. A caller that keeps no other reference to X
	and Y has that memory back before the kernel runs.

	The CSC forms are built with numpy calls that release the GIL, run by
	numba.get_num_threads() threads: Y's column sort beside X's, then
	searches and gathers over blocks of the sorted entries, each writing
	its own slice of arrays allocated here (gapped_kmer.BLOCK says why), so
	every array is the one a single thread would build. Y's CSC form is
	built before X's, so that Y's sort is freed before X's CSC arrays are
	made.
	"""

	X, Y = matrices
	matrices.clear()

	if not (X.has_canonical_format and Y.has_canonical_format):
		return _sparse_mm_dot_merge(X.data, X.indices, X.indptr, Y.data, 
			Y.indices, Y.indptr, k)

	n, nnz, nnz_y, n_cols = X.shape[0], X.nnz, Y.nnz, X.shape[1]
	X_data, X_indptr, x_indices = X.data, X.indptr, [X.indices]
	Y_data, Y_indptr, y_indices = Y.data, Y.indptr, [Y.indices]
	del X, Y

	step = gapped_kmer.BLOCK

	with ThreadPoolExecutor(max(2, numba.get_num_threads())) as pool:
		def map_blocks(f, size):
			list(pool.map(f, range(0, size, step)))

		# Sort X's and Y's entries by column. A row holds each column at
		# most once, so the order of the rows inside one column changes no
		# sum and the sorts need not be stable.
		y_sort = _sorted_by_column(y_indices, n_cols, pool)
		order_x, sorted_x = _sorted_by_column(x_indices, n_cols)

		# X's distinct columns renumber the hashed columns (up to 5**15) to
		# 0..len(cols)-1.
		first = np.empty(nnz, dtype=bool)
		first[:1] = True
		np.not_equal(sorted_x[1:], sorted_x[:-1], out=first[1:])
		cols = sorted_x[first]
		del sorted_x

		# Y in CSC form over the same columns, without the columns X lacks,
		# which no row of X visits: each entry's place among X's columns
		# and whether it is there, then the kept entries.
		order_y, sorted_y = y_sort.result()
		del y_sort

		m = len(cols)
		pos = np.empty(nnz_y, dtype=np.int32 if m < 2**31 else np.int64)
		keep = np.empty(nnz_y, dtype=bool)
		# -1 is below every column index.
		cols_end = np.append(cols, -1)

		def y_search(lo):
			hi = min(lo + step, nnz_y)
			pos[lo:hi] = np.searchsorted(cols, sorted_y[lo:hi])
			np.equal(cols_end[pos[lo:hi]], sorted_y[lo:hi], out=keep[lo:hi])

		map_blocks(y_search, nnz_y)
		del sorted_y, cols_end

		y_offsets = _block_offsets(keep, step)
		n_kept = y_offsets[-1]
		rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(Y_indptr))
		Yt_rows = np.empty(n_kept, dtype=np.int32)
		Yt_data = np.empty(n_kept, dtype=np.float64)
		Yt_cols = np.empty(n_kept, dtype=pos.dtype)

		def y_gather(lo):
			hi = min(lo + step, nnz_y)
			a, z = y_offsets[lo // step], y_offsets[lo // step + 1]
			order = order_y[lo:hi][keep[lo:hi]]
			np.take(rows, order, out=Yt_rows[a:z], mode='clip')
			np.take(Y_data, order, out=Yt_data[a:z], mode='clip')
			np.compress(keep[lo:hi], pos[lo:hi], out=Yt_cols[a:z])

		map_blocks(y_gather, nnz_y)
		del order_y, pos, keep, rows, Y_data, Y_indptr

		# Yt_indptr[c] is the number of kept entries whose column is below
		# c; Yt_cols is sorted.
		Yt_indptr = np.empty(m + 1, dtype=np.int64)
		needles = np.arange(m + 1, dtype=Yt_cols.dtype)

		def y_indptr(lo):
			hi = min(lo + step, m + 1)
			Yt_indptr[lo:hi] = np.searchsorted(Yt_cols, needles[lo:hi])

		map_blocks(y_indptr, m + 1)
		del Yt_cols, needles

		# X in CSR form over the new columns (X_cols) and in CSC form. The
		# new column of the sorted entry q is the number of first entries
		# up to q, less one.
		x_offsets = _block_offsets(first, step)
		Xt_indptr = np.append(np.flatnonzero(first), nnz)
		rows = np.repeat(np.arange(n, dtype=np.int32), np.diff(X_indptr))
		X_cols = np.empty(nnz, dtype=np.int64)
		Xt_rows = np.empty(nnz, dtype=np.int32)
		Xt_data = np.empty(nnz, dtype=np.float64)

		def x_block(lo):
			hi = min(lo + step, nnz)
			order = order_x[lo:hi]
			ranks = np.cumsum(first[lo:hi], dtype=np.int64)
			ranks += x_offsets[lo // step] - 1
			X_cols[order] = ranks
			np.take(rows, order, out=Xt_rows[lo:hi], mode='clip')
			np.take(X_data, order, out=Xt_data[lo:hi], mode='clip')

		map_blocks(x_block, nnz)
		del order_x, first, rows

	# Rows differ widely in cost; small chunks keep every thread busy. Each
	# row is computed by one thread on its own, so the result does not
	# depend on the chunk size or the thread count.
	with numba.parallel_chunksize(4):
		return _sparse_mm_dot(X_data, np.asarray(X_indptr, dtype=np.int64), 
			X_cols, Xt_indptr, Xt_rows, Xt_data, Yt_indptr, Yt_rows, 
			Yt_data, k)


def cosine_similarity_from_seqlets(seqlets, n_neighbors, sign, topn=20, 
	min_k=4, max_k=6, max_gap=15, max_len=15, max_entries=500, 
	alphabet_size=4):

	# The inputs of both passes, built for all seqlets at once when that is
	# exact, else seqlet by seqlet.
	Xs = gapped_kmer._gkmer_inputs(seqlets, topn, sign)
	if Xs is None:
		Xs = (gapped_kmer._gkmer_inputs_loop(seqlets, topn, True, sign), 
			gapped_kmer._gkmer_inputs_loop(seqlets, topn, False, sign))

	X_fwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, 
		min_k, max_k, max_gap, max_len, max_entries, True, sign, X=Xs[0])

	X_bwd = gapped_kmer._seqlet_to_gkmers(seqlets, topn, min_k, max_k, max_gap, 
			max_len, max_entries, False, sign, X=Xs[1])
	del Xs

	import sklearn.preprocessing
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

	affmat = _jaccard_fwd_rev(filters_all_fwd_data, filters_all_rev_data,
		all_fwd_data, seqlet_neighbors, min_overlap)
	if affmat is not None:
		return affmat

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


# `_fwd_rev_best` keeps each row of Y in a buffer whose rows are this many
# float64 apart, a constant, so that every load of one pass is at a
# constant distance from one base address. Offsets that need longer rows
# take the two `jaccard` calls instead.
_FWD_REV_STRIDE = 128

# Rows of Y per parallel task, and so per slice of the scratch arrays.
_FWD_REV_ROWS = 16


def _noalias(dispatcher):
	"""Have numba compile the dispatcher with its array arguments marked
	noalias, as it marks those of the function it makes of a parallel
	loop's body. Otherwise LLVM would have to check at run time that no
	array a vector loop writes overlaps one it reads, and with more than a
	few such pairs it leaves the loop scalar. The flag is not one of
	numba's public options, so it is set through the dispatcher's
	compiler, when this numba has both; the dispatcher is then marked
	`_modisco_noalias`."""

	try:
		from numba.core.compiler import Flags
	except ImportError:
		return dispatcher

	compiler = getattr(dispatcher, '_compiler', None)
	customize = getattr(compiler, '_customize_flags', None)
	if customize is None or not hasattr(Flags, 'noalias'):
		return dispatcher

	def _customize_flags(flags):
		flags = customize(flags)
		flags.noalias = True
		return flags

	compiler._customize_flags = _customize_flags
	dispatcher._modisco_noalias = True
	return dispatcher


def _one_hot_contribs(X):
	"""Whether no position of any row of X, 8 channels, holds more than one
	nonzero value among channels 4 to 7."""

	nonzero = np.ascontiguousarray(X[:, :, 4:] != 0)
	words = nonzero.view(np.uint32)
	return bool(((words & (words - np.uint32(1))) == 0).all())


def _fwd_rev_threads(X_fwd, X_rev, Y, neighbors, n_pad, best, flags, sums):
	"""Run `_fwd_rev_best8` over every task, sharing blocks of tasks among
	numba.get_num_threads() threads."""

	# A first call on no tasks, here in the main thread, waits for the
	# kernel's compile and hands back the threads of the compile children
	# that have finished, before the thread count is read.
	_fwd_rev_best8(X_fwd, X_rev, Y, neighbors, n_pad, best, flags, *sums, 0, 0)

	n_tasks = sums[0].shape[0]
	n_threads = numba.get_num_threads()
	n_blocks = max(1, min(n_tasks, 32 * n_threads))
	bounds = [n_tasks * b // n_blocks for b in range(n_blocks + 1)]

	def run(b):
		_fwd_rev_best8(X_fwd, X_rev, Y, neighbors, n_pad, best, flags, *sums,
			bounds[b], bounds[b+1])

	if n_threads == 1 or n_blocks == 1:
		for b in range(n_blocks):
			run(b)
	else:
		with ThreadPoolExecutor(min(n_threads, n_blocks)) as pool:
			list(pool.map(run, range(n_blocks)))


def _jaccard_fwd_rev(X_fwd, X_rev, Y, seqlet_neighbors, min_overlap):
	"""The array jaccard_from_seqlets returns, np.maximum of
	jaccard(X_fwd, Y) and jaccard(X_rev, Y) with func=int and
	return_sparse=True, computed by one kernel that scores both strands of
	each neighbour against a row of Y while that row is in cache:
	`_fwd_rev_best8`, on a pool of threads, for the 8 channels of
	`util.get_2d_data_from_patterns` whose channels 4 to 7 are one-hot
	sparse, else `_fwd_rev_best`. Returns None when neither applies, and
	the caller then makes the two calls: channels not a multiple of 4, offsets that do not fit the
	kernel's row buffer, a value that is not finite, any score that is a
	NaN or a -0.0 (for those, as in `jaccard`, numpy's max decides which
	bits come back), and the arguments on which `jaccard` itself raises.

	Without a NaN or a -0.0 among the scores the largest score is unique in
	its bits, so the maximum over offsets and strands is the value the two
	calls and np.maximum give."""

	if not isinstance(seqlet_neighbors, np.ndarray) or seqlet_neighbors.ndim != 2:
		return None

	if X_fwd.ndim != 3 or X_rev.shape != X_fwd.shape or Y.ndim != 3:
		return None

	nx, d, m = X_fwd.shape
	ny, Ly, my = Y.shape
	n_neighbors = seqlet_neighbors.shape[1]

	# As `jaccard` pads Y with func=int.
	n_pad = 0 if min_overlap is None else int(int(d*(1-min_overlap)))
	len_output = Ly + 2*n_pad - d + 1
	n_offsets = (len_output + 7) // 8 * 8

	if (n_pad < 0 or len_output < 1 or m == 0 or m % 4 != 0 or my != m
			or d - 1 + n_offsets > _FWD_REV_STRIDE or ny == 0 or nx == 0
			or n_neighbors == 0 or seqlet_neighbors.shape[0] != ny):
		return None

	X_fwd = np.ascontiguousarray(X_fwd, dtype='float32')
	X_rev = np.ascontiguousarray(X_rev, dtype='float32')
	Y = X_fwd if Y is X_fwd else np.ascontiguousarray(Y, dtype='float32')
	for array in (X_fwd, X_rev) if Y is X_fwd else (X_fwd, X_rev, Y):
		if not np.isfinite(array).all():
			return None

	neighbors = np.ascontiguousarray(seqlet_neighbors.astype('int32'))

	# One row of each accumulator per task, 64 bytes longer than it needs
	# to be so that no two tasks write to one cache line.
	n_tasks = (ny + _FWD_REV_ROWS - 1) // _FWD_REV_ROWS
	sums = [np.empty((n_tasks, n_offsets + 8)) for _ in range(4)]

	best = np.empty((ny, n_neighbors), dtype='float32')
	flags = np.zeros(ny, dtype='uint8')
	if (_FWD_REV_SERIAL and m == 8 and _one_hot_contribs(X_fwd)
			and _one_hot_contribs(X_rev)):
		_fwd_rev_threads(X_fwd, X_rev, Y, neighbors, n_pad, best, flags, sums)
	else:
		_fwd_rev_best(X_fwd, X_rev, Y, neighbors, n_pad, best, flags, *sums)

	if flags.any():
		return None
	return best


def jaccard(X, Y, min_overlap=None, seqlet_neighbors=None, func=np.ceil, 
	return_sparse=False):

	all_neighbors = seqlet_neighbors is None

	if min_overlap is not None:
		n_pad = int(func(X.shape[1]*(1-min_overlap)))
		if n_pad >= 0 and isinstance(Y, np.ndarray) and Y.ndim == 3:
			# np.pad(mode="constant") without its Python overhead, as in
			# pearson_correlations: zeros of Y's dtype around a copy of Y.
			m, l, k = Y.shape
			Y_padded = np.zeros((m, l + 2*n_pad, k), dtype=Y.dtype)
			Y_padded[:, n_pad:n_pad+l] = Y
			Y = Y_padded
		else:
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
		# One row of Y (aggregator._best_alignment, once per seqlet) runs
		# on this thread: the same scores without a parallel launch.
		kernel = _jaccard_best_serial if n == 1 else _jaccard_best
		kernel(X, Y, seqlet_neighbors, best, best_idx, flags)

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

# The columns of pairwise_jaccard's inner loop taken per pass: a tile's two
# float64 running sums (16 KB) and its slices of the signs and magnitudes
# (8 KB) stay in L1 while the loop over features runs.
_PAIRWISE_TILE = 1024


@njit
def _ranks_after(values, a, b):
	"""Whether index a comes after index b in the stable descending order of
	values: a smaller value, or an equal one at a later index."""

	return values[a] < values[b] or (values[a] == values[b] and a > b)


@njit
def _top_k_sift_down(values, heap, size, pos, index):
	"""Place index at heap[pos], moving it down past children that come
	after it, so that the root of heap[:size] is the one that comes last."""

	while True:
		child = 2 * pos + 1
		if child >= size:
			break
		if (child + 1 < size and
				_ranks_after(values, heap[child + 1], heap[child])):
			child += 1
		if _ranks_after(values, heap[child], index):
			heap[pos] = heap[child]
			pos = child
		else:
			break
	heap[pos] = index


@njit
def _top_k(values, k):
	"""np.argsort(-values, kind='mergesort')[:k] for values holding no NaN
	and 0 < k <= len(values): the indices of the k largest values, largest
	first, equal values in index order.

	That stable sort orders the indices by (value descending, index
	ascending), a strict order on distinct indices (-0.0 and 0.0 compare
	equal, as in the sort), so its first k are the k first in that order,
	in that order. A heap of k holds the best seen so far with the one that
	comes last at its root; an index read later replaces the root only with
	a larger value, since among equal values the earlier index comes first.
	Taking the root out k times gives the k from last to first."""

	n = values.shape[0]
	heap = np.empty(k, dtype=np.int64)

	for size in range(k):
		pos = size
		while pos > 0:
			parent = (pos - 1) >> 1
			if _ranks_after(values, size, heap[parent]):
				heap[pos] = heap[parent]
				pos = parent
			else:
				break
		heap[pos] = size

	for j in range(k, n):
		if values[j] > values[heap[0]]:
			_top_k_sift_down(values, heap, k, 0, j)

	top = np.empty(k, dtype=np.int64)
	for size in range(k, 0, -1):
		top[size - 1] = heap[0]
		if size > 1:
			_top_k_sift_down(values, heap, size - 1, 0, heap[size - 1])

	return top


@njit(nogil=True)
def _signs_magnitudes(X, signs, magnitudes):
	"""np.sign and abs of every element of X, transposed."""

	n, m = X.shape
	for j in range(n):
		for l in range(m):
			signs[l, j] = np.sign(X[j, l])
			magnitudes[l, j] = abs(X[j, l])


# The numpy error model, as numba uses in the parallel loop this replaced:
# 0 / 0 gives NaN rather than raising.
@njit(nogil=True, error_model='numpy')
def _pairwise_jaccard_rows(X, signs, magnitudes, k, start, stop, jaccards,
	neighbors):
	"""pairwise_jaccard's rows start to stop, written into jaccards and
	neighbors."""

	n, m = X.shape
	min_sums = np.empty(n, dtype=np.float64)
	max_sums = np.empty(n, dtype=np.float64)
	jaccard_ = np.empty(n, dtype=np.float64)

	for i in range(start, stop):
		min_sums[:] = 0.0
		max_sums[:] = 0.0

		for j0 in range(0, n, _PAIRWISE_TILE):
			j1 = min(j0 + _PAIRWISE_TILE, n)
			min_tile = min_sums[j0:j1]
			max_tile = max_sums[j0:j1]

			for l in range(m):
				si = np.sign(X[i, l])
				xi = abs(X[i, l])
				signs_l = signs[l, j0:j1]
				magnitudes_l = magnitudes[l, j0:j1]

				for j in range(j1 - j0):
					sign = si * signs_l[j]
					xj = magnitudes_l[j]

					larger = xi > xj
					lo = xj if larger else xi
					hi = xi if larger else xj
					min_tile[j] += lo * sign
					max_tile[j] += hi

		has_nan = False
		for j in range(n):
			jaccard_[j] = min_sums[j] / max_sums[j]
			has_nan |= np.isnan(jaccard_[j])

		if has_nan or k >= n:
			idxs = np.argsort(-jaccard_, kind='mergesort')[:k]
		else:
			idxs = _top_k(jaccard_, k)

		for c in range(k):
			jaccards[i, c] = jaccard_[idxs[c]]
			neighbors[i, c] = idxs[c]


def pairwise_jaccard(X, k):
	"""The continuous Jaccard similarity of every row of X with every row,
	and for each row the k largest, largest first (ties in row order), with
	their row indices: (n, k) float64 similarities and int32 indices.

	Row i's similarity with row j sums, over the features l in order, the
	smaller magnitude times the product of the two signs, and the larger
	magnitude, in two float64 running sums, and divides them. Here row i
	is scored against every row j at once: the loop over j is innermost, so
	it vectorizes, and each j keeps its own two running sums, which receive
	the same terms in the same order of l as one pair at a time would. The
	signs and magnitudes of X's columns are laid out once, transposed, for
	that loop. The k largest come from `_top_k`, or from the stable sort
	they stand in for when a row holds a NaN or k is every row.

	Rows are independent, so contiguous blocks of them are shared among
	numba.get_num_threads() threads running `_pairwise_jaccard_rows`, which
	releases the GIL (a serial kernel compiles in a fraction of the time a
	parallel one takes). The result does not depend on the number of
	threads or blocks.
	"""

	n, m = X.shape
	jaccards = np.empty((n, k), dtype='float64')
	neighbors = np.empty((n, k), dtype='int32')
	signs = np.empty((m, n), dtype=X.dtype)
	magnitudes = np.empty((m, n), dtype=X.dtype)
	_signs_magnitudes(X, signs, magnitudes)

	n_threads = numba.get_num_threads()
	n_blocks = max(1, min(n, 4 * n_threads))
	bounds = [n * b // n_blocks for b in range(n_blocks + 1)]

	def run(b):
		_pairwise_jaccard_rows(X, signs, magnitudes, k, bounds[b],
			bounds[b+1], jaccards, neighbors)

	if n_threads == 1 or n_blocks == 1:
		for b in range(n_blocks):
			run(b)
	else:
		with ThreadPoolExecutor(min(n_threads, n_blocks)) as pool:
			list(pool.map(run, range(n_blocks)))

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


@precompile.signature('void(float32[:, :, ::1], float32[:, :, ::1], int32[:, ::1], float32[:, ::1], int64[:, ::1], uint8[::1])')
@njit(nogil=True, error_model='numpy')
def _jaccard_best_serial(X, Y, neighbors, best, best_idx, flags):
	"""`_jaccard_best` without `prange`: the same statements, rows one
	after another on the calling thread. For the one-row calls of
	`aggregator._best_alignment` (one per seqlet merged into a pattern), a
	parallel launch for a single row costs more than the row's work. Each
	score is the same float64 sums, taken in the same order, as in
	`_jaccard_best`, so the outputs are bit-identical to its outputs. The
	numpy error model is the one numba gives a prange body, so 0 / 0 is a
	NaN here too, not a ZeroDivisionError."""

	nx, d, m = X.shape
	ny, L, _ = Y.shape
	n_neighbors = neighbors.shape[1]
	len_output = L - d + 1

	# The offsets are padded to a multiple of 8 so the vectorized loop over
	# them has no scalar remainder; the padding offsets are never read.
	n_offsets = (len_output + 7) // 8 * 8
	L_pad = d - 1 + n_offsets

	for l in range(ny):
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


@precompile.signature('void(float32[:, :, ::1], float32[:, :, ::1], '
	'float32[:, :, ::1], int32[:, ::1], int64, float32[:, ::1], uint8[::1], '
	'float64[:, ::1], float64[:, ::1], float64[:, ::1], float64[:, ::1])')
@njit(parallel=True, error_model='numpy')
def _fwd_rev_best(Xf, Xr, Y, neighbors, n_pad, best, flags, min_f, max_f,
	min_r, max_r):
	"""best[l, i]: the largest score, over every offset and both strands,
	of row l of Y (zero-padded by n_pad on each side) against rows
	neighbors[l, i] of Xf and of Xr. flags[l] is set when row l holds a NaN
	or a -0.0 score. Each score is the one `_jaccard_best` gives, bit for
	bit, for finite inputs whose number of channels is a multiple of 4:

	`_jaccard_best` adds, for each offset and in (j, k) order, the float32
	values min(|x|, |y|) * sign(x) * sign(y) and max(|x|, |y|), each
	converted exactly to float64. Here the same float64 values are added in
	the same order, formed from float64 copies. The first is
	clamp(sign(x) * y, -|x|, |x|), which equals it whenever it is nonzero;
	when it is zero its sign may differ, and adding a zero of either sign
	to a sum that starts at +0.0, and so is never -0.0, changes no bit.
	sign(x) * y is y or -y, read from one of two rows of the buffer.

	The offsets are innermost and padded to a multiple of 8, as in
	`_jaccard_best`; four channels of both strands are added per pass over
	them. The per-task accumulators are arguments rather than allocated
	here, so the compiler knows they share no memory with the row buffer
	and the inputs, and the vector loop needs no runtime overlap checks.
	Each row is scored by one task and each task owns its scratch, so the
	result does not depend on the number of threads."""

	nx, d, m = Xf.shape
	ny, Ly, _ = Y.shape
	n_neighbors = neighbors.shape[1]
	len_output = Ly + 2 * n_pad - d + 1
	n_offsets = (len_output + 7) // 8 * 8
	n_vectors = n_offsets // 8
	L_pad = d - 1 + n_offsets
	W = _FWD_REV_STRIDE
	R = _FWD_REV_ROWS
	n_tasks = (ny + R - 1) // R

	for b in prange(n_tasks):
		# Rows 3k, 3k+1 and 3k+2 of the buffer hold y, -y and |y| for
		# channel k, zero past either end of the row.
		ybuf = np.empty(3 * m * W, dtype=np.float64)

		for l in range(b * R, min(ny, (b + 1) * R)):
			for k in range(m):
				for p in range(L_pad):
					j = p - n_pad
					v = 0.0
					if j >= 0 and j < Ly:
						v = np.float64(Y[l, j, k])
					ybuf[(3*k)*W + p] = v
					ybuf[(3*k+1)*W + p] = -v
					ybuf[(3*k+2)*W + p] = abs(v)

			flag = 0
			for i in range(n_neighbors):
				nb = neighbors[l, i]
				for idx in range(n_offsets):
					min_f[b, idx] = 0.0
					max_f[b, idx] = 0.0
					min_r[b, idx] = 0.0
					max_r[b, idx] = 0.0

				for j in range(d):
					for k in range(0, m, 4):
						# Per channel and strand: |x|, -|x|, and the buffer
						# row holding sign(x) * y, less the offset index.
						# (Unsigned indices spare the negative-index checks.)
						base = 3*k*W + j

						x = np.float64(Xf[nb, j, np.uint64(k)])
						af0 = abs(x)
						nf0 = -af0
						of0 = base + (1 if x < 0 else 0)*W
						x = np.float64(Xr[nb, j, np.uint64(k)])
						ar0 = abs(x)
						nr0 = -ar0
						or0 = base + (1 if x < 0 else 0)*W

						x = np.float64(Xf[nb, j, np.uint64(k+1)])
						af1 = abs(x)
						nf1 = -af1
						of1 = base + (3 + (1 if x < 0 else 0))*W
						x = np.float64(Xr[nb, j, np.uint64(k+1)])
						ar1 = abs(x)
						nr1 = -ar1
						or1 = base + (3 + (1 if x < 0 else 0))*W

						x = np.float64(Xf[nb, j, np.uint64(k+2)])
						af2 = abs(x)
						nf2 = -af2
						of2 = base + (6 + (1 if x < 0 else 0))*W
						x = np.float64(Xr[nb, j, np.uint64(k+2)])
						ar2 = abs(x)
						nr2 = -ar2
						or2 = base + (6 + (1 if x < 0 else 0))*W

						x = np.float64(Xf[nb, j, np.uint64(k+3)])
						af3 = abs(x)
						nf3 = -af3
						of3 = base + (9 + (1 if x < 0 else 0))*W
						x = np.float64(Xr[nb, j, np.uint64(k+3)])
						ar3 = abs(x)
						nr3 = -ar3
						or3 = base + (9 + (1 if x < 0 else 0))*W

						# Split into vectors of 8 offsets, so that the loop has no
						# remainder to keep scalars alive for.
						for vec in range(n_vectors):
							for q in range(8):
								idx = vec * 8 + q
								mf = min_f[b, idx]
								Mf = max_f[b, idx]
								mr = min_r[b, idx]
								Mr = max_r[b, idx]

								ya = ybuf[np.uint64(base + 2*W + idx)]
								t = ybuf[np.uint64(of0 + idx)]
								t = t if t > nf0 else nf0
								mf += t if t < af0 else af0
								Mf += ya if ya > af0 else af0
								t = ybuf[np.uint64(or0 + idx)]
								t = t if t > nr0 else nr0
								mr += t if t < ar0 else ar0
								Mr += ya if ya > ar0 else ar0

								ya = ybuf[np.uint64(base + 5*W + idx)]
								t = ybuf[np.uint64(of1 + idx)]
								t = t if t > nf1 else nf1
								mf += t if t < af1 else af1
								Mf += ya if ya > af1 else af1
								t = ybuf[np.uint64(or1 + idx)]
								t = t if t > nr1 else nr1
								mr += t if t < ar1 else ar1
								Mr += ya if ya > ar1 else ar1

								ya = ybuf[np.uint64(base + 8*W + idx)]
								t = ybuf[np.uint64(of2 + idx)]
								t = t if t > nf2 else nf2
								mf += t if t < af2 else af2
								Mf += ya if ya > af2 else af2
								t = ybuf[np.uint64(or2 + idx)]
								t = t if t > nr2 else nr2
								mr += t if t < ar2 else ar2
								Mr += ya if ya > ar2 else ar2

								ya = ybuf[np.uint64(base + 11*W + idx)]
								t = ybuf[np.uint64(of3 + idx)]
								t = t if t > nf3 else nf3
								mf += t if t < af3 else af3
								Mf += ya if ya > af3 else af3
								t = ybuf[np.uint64(or3 + idx)]
								t = t if t > nr3 else nr3
								mr += t if t < ar3 else ar3
								Mr += ya if ya > ar3 else ar3

								min_f[b, idx] = mf
								max_f[b, idx] = Mf
								min_r[b, idx] = mr
								max_r[b, idx] = Mr

				# With no NaN and no -0.0 among the scores, equal scores
				# have equal bits, so any maximum is the one numpy's gives.
				best_score = np.float32(0.0)
				for idx in range(len_output):
					s_f = np.float32(min_f[b, idx] / max_f[b, idx])
					s_r = np.float32(min_r[b, idx] / max_r[b, idx])
					if s_f != s_f or (s_f == 0 and math.copysign(1.0, s_f) < 0):
						flag = 1
					if s_r != s_r or (s_r == 0 and math.copysign(1.0, s_r) < 0):
						flag = 1

					s = s_f if s_f > s_r else s_r
					if idx == 0 or s > best_score:
						best_score = s

				best[l, i] = best_score
			flags[l] = flag


@precompile.signature('void(float32[:, :, ::1], float32[:, :, ::1], '
	'float32[:, :, ::1], int32[:, ::1], int64, float32[:, ::1], uint8[::1], '
	'float64[:, ::1], float64[:, ::1], float64[:, ::1], float64[:, ::1], '
	'int64, int64)')
@_noalias
@njit(nogil=True, error_model='numpy')
def _fwd_rev_best8(Xf, Xr, Y, neighbors, n_pad, best, flags, min_f, max_f,
	min_r, max_r, b0, b1):
	"""`_fwd_rev_best`, for the rows of Y in tasks b0 to b1 - 1, when there
	are 8 channels and in channels 4 to 7 no position of any row of Xf or
	Xr holds more than one nonzero value: the layout of
	`util.get_2d_data_from_patterns`, whose channels 4 to 7 are the
	contribution scores, the hypothetical ones masked by a one-hot
	sequence. Serial and without the GIL, for a pool of threads to share
	the tasks: it compiles in half the time a parallel loop takes. Each
	score's two sums take the float64 values `_fwd_rev_best`'s take, in
	the same order, less zeros:

	- One pass over the offsets per position adds all 8 channels' terms of
	  both strands, in channel order.
	- Channels 0 to 3: the min term is copysign(min(|y|, |x|), t), where
	  t = sign(x) * y is read from the y or the -y row. Its magnitude is
	  min(|t|, |x|) and its sign that of t, so it equals clamp(t, -|x|,
	  |x|), the term `_fwd_rev_best` adds, but for the sign of a zero.
	- Channels 4 to 7: the max terms of all four are added. Of the min
	  terms, only the nonzero channel's is added (none is, and channel 4's
	  row serves, when all four are zero): each other one is
	  clamp(t, -0, +0), a zero.
	- A sum starts at +0.0, so it is never -0.0, and adding a zero of
	  either sign to it changes no bit.

	The result depends neither on the number of threads nor on how the
	tasks are shared: each row is scored by one task, and each task owns
	its rows of the accumulators."""

	nx, d, m = Xf.shape
	ny, Ly, _ = Y.shape
	n_neighbors = neighbors.shape[1]
	len_output = Ly + 2 * n_pad - d + 1
	n_offsets = (len_output + 7) // 8 * 8
	n_vectors = n_offsets // 8
	L_pad = d - 1 + n_offsets
	W = _FWD_REV_STRIDE
	R = _FWD_REV_ROWS

	for b in range(b0, b1):
		# Rows 3k, 3k+1 and 3k+2 of the buffer hold y, -y and |y| for
		# channel k, zero past either end of the row.
		ybuf = np.empty(3 * m * W, dtype=np.float64)

		for l in range(b * R, min(ny, (b + 1) * R)):
			for k in range(m):
				for p in range(L_pad):
					j = p - n_pad
					v = 0.0
					if j >= 0 and j < Ly:
						v = np.float64(Y[l, j, k])
					ybuf[(3*k)*W + p] = v
					ybuf[(3*k+1)*W + p] = -v
					ybuf[(3*k+2)*W + p] = abs(v)

			flag = 0
			for i in range(n_neighbors):
				nb = neighbors[l, i]
				for idx in range(n_offsets):
					min_f[b, idx] = 0.0
					max_f[b, idx] = 0.0
					min_r[b, idx] = 0.0
					max_r[b, idx] = 0.0

				for j in range(d):
					# Channels 0 to 3, both strands: |x| and the buffer row
					# holding sign(x) * y. (Unsigned indices spare the
					# negative-index checks.)
					base = j
					x = np.float64(Xf[nb, j, np.uint64(0)])
					af0 = abs(x)
					of0 = base + (0 + (1 if x < 0 else 0))*W
					x = np.float64(Xr[nb, j, np.uint64(0)])
					ar0 = abs(x)
					or0 = base + (0 + (1 if x < 0 else 0))*W
					x = np.float64(Xf[nb, j, np.uint64(1)])
					af1 = abs(x)
					of1 = base + (3 + (1 if x < 0 else 0))*W
					x = np.float64(Xr[nb, j, np.uint64(1)])
					ar1 = abs(x)
					or1 = base + (3 + (1 if x < 0 else 0))*W
					x = np.float64(Xf[nb, j, np.uint64(2)])
					af2 = abs(x)
					of2 = base + (6 + (1 if x < 0 else 0))*W
					x = np.float64(Xr[nb, j, np.uint64(2)])
					ar2 = abs(x)
					or2 = base + (6 + (1 if x < 0 else 0))*W
					x = np.float64(Xf[nb, j, np.uint64(3)])
					af3 = abs(x)
					of3 = base + (9 + (1 if x < 0 else 0))*W
					x = np.float64(Xr[nb, j, np.uint64(3)])
					ar3 = abs(x)
					or3 = base + (9 + (1 if x < 0 else 0))*W
					# Channels 4 to 7: |x| of each, and of the one that may
					# be nonzero, v (its value: a sum of it and zeros,
					# exact), its channel c and the row holding sign(v) * y.
					x0 = np.float64(Xf[nb, j, np.uint64(4)])
					x1 = np.float64(Xf[nb, j, np.uint64(5)])
					x2 = np.float64(Xf[nb, j, np.uint64(6)])
					x3 = np.float64(Xf[nb, j, np.uint64(7)])
					bf0 = abs(x0)
					bf1 = abs(x1)
					bf2 = abs(x2)
					bf3 = abs(x3)
					v = x0 + x1 + x2 + x3
					c = (1 if x1 != 0 else 0) + (2 if x2 != 0 else 0) + (3 if x3 != 0 else 0)
					avf = abs(v)
					nvf = -avf
					ovf = base + (12 + 3*c + (1 if v < 0 else 0))*W
					x0 = np.float64(Xr[nb, j, np.uint64(4)])
					x1 = np.float64(Xr[nb, j, np.uint64(5)])
					x2 = np.float64(Xr[nb, j, np.uint64(6)])
					x3 = np.float64(Xr[nb, j, np.uint64(7)])
					br0 = abs(x0)
					br1 = abs(x1)
					br2 = abs(x2)
					br3 = abs(x3)
					v = x0 + x1 + x2 + x3
					c = (1 if x1 != 0 else 0) + (2 if x2 != 0 else 0) + (3 if x3 != 0 else 0)
					avr = abs(v)
					nvr = -avr
					ovr = base + (12 + 3*c + (1 if v < 0 else 0))*W

					for vec in range(n_vectors):
						for q in range(8):
							idx = vec * 8 + q
							mf = min_f[b, idx]
							Mf = max_f[b, idx]
							mr = min_r[b, idx]
							Mr = max_r[b, idx]

							ya = ybuf[np.uint64(base + 2*W + idx)]
							t = ybuf[np.uint64(of0 + idx)]
							u = ya if ya < af0 else af0
							mf += math.copysign(u, t)
							Mf += ya if ya > af0 else af0
							t = ybuf[np.uint64(or0 + idx)]
							u = ya if ya < ar0 else ar0
							mr += math.copysign(u, t)
							Mr += ya if ya > ar0 else ar0

							ya = ybuf[np.uint64(base + 5*W + idx)]
							t = ybuf[np.uint64(of1 + idx)]
							u = ya if ya < af1 else af1
							mf += math.copysign(u, t)
							Mf += ya if ya > af1 else af1
							t = ybuf[np.uint64(or1 + idx)]
							u = ya if ya < ar1 else ar1
							mr += math.copysign(u, t)
							Mr += ya if ya > ar1 else ar1

							ya = ybuf[np.uint64(base + 8*W + idx)]
							t = ybuf[np.uint64(of2 + idx)]
							u = ya if ya < af2 else af2
							mf += math.copysign(u, t)
							Mf += ya if ya > af2 else af2
							t = ybuf[np.uint64(or2 + idx)]
							u = ya if ya < ar2 else ar2
							mr += math.copysign(u, t)
							Mr += ya if ya > ar2 else ar2

							ya = ybuf[np.uint64(base + 11*W + idx)]
							t = ybuf[np.uint64(of3 + idx)]
							u = ya if ya < af3 else af3
							mf += math.copysign(u, t)
							Mf += ya if ya > af3 else af3
							t = ybuf[np.uint64(or3 + idx)]
							u = ya if ya < ar3 else ar3
							mr += math.copysign(u, t)
							Mr += ya if ya > ar3 else ar3

							ya = ybuf[np.uint64(base + 14*W + idx)]
							Mf += ya if ya > bf0 else bf0
							Mr += ya if ya > br0 else br0

							ya = ybuf[np.uint64(base + 17*W + idx)]
							Mf += ya if ya > bf1 else bf1
							Mr += ya if ya > br1 else br1

							ya = ybuf[np.uint64(base + 20*W + idx)]
							Mf += ya if ya > bf2 else bf2
							Mr += ya if ya > br2 else br2

							ya = ybuf[np.uint64(base + 23*W + idx)]
							Mf += ya if ya > bf3 else bf3
							Mr += ya if ya > br3 else br3
							t = ybuf[np.uint64(ovf + idx)]
							t = t if t > nvf else nvf
							mf += t if t < avf else avf
							t = ybuf[np.uint64(ovr + idx)]
							t = t if t > nvr else nvr
							mr += t if t < avr else avr

							min_f[b, idx] = mf
							max_f[b, idx] = Mf
							min_r[b, idx] = mr
							max_r[b, idx] = Mr

				# With no NaN and no -0.0 among the scores, equal scores
				# have equal bits, so any maximum is the one numpy's gives.
				best_score = np.float32(0.0)
				for idx in range(len_output):
					s_f = np.float32(min_f[b, idx] / max_f[b, idx])
					s_r = np.float32(min_r[b, idx] / max_r[b, idx])
					if s_f != s_f or (s_f == 0 and math.copysign(1.0, s_f) < 0):
						flag = 1
					if s_r != s_r or (s_r == 0 and math.copysign(1.0, s_r) < 0):
						flag = 1

					s = s_f if s_f > s_r else s_r
					if idx == 0 or s > best_score:
						best_score = s

				best[l, i] = best_score
			flags[l] = flag


# Whether `_fwd_rev_best8` compiles with its arrays marked noalias; without
# that its vector loops stay scalar, and `_fwd_rev_best` is used instead.
_FWD_REV_SERIAL = getattr(_fwd_rev_best8, '_modisco_noalias', False)


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
	return pearson_correlations([X], Y, min_overlap=min_overlap, func=func)[0]


def pearson_correlations(Xs, Y, min_overlap=None, func=np.ceil):
	"""[pearson_correlation(X, Y, min_overlap, func) for X in Xs].

	For arrays X of one shape, Y is padded once, and on the fast path its
	windows are normalized once, for all of them; each X's scores are then
	the same operations on the same operands as in a call of its own.
	"""

	Xs = [X[None, :, :] if X.ndim == 2 else X for X in Xs]
	if Y.ndim == 2:
		Y = Y[None, :, :]

	if len(set(X.shape for X in Xs)) > 1:
		return [pearson_correlations([X], Y, min_overlap=min_overlap,
			func=func)[0] for X in Xs]

	if min_overlap is not None:
		n_pad = int(func(Xs[0].shape[1]*(1-min_overlap)))
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

	n, d, _ = Xs[0].shape
	len_output = 1 + Y.shape[1] - d 
	Y_normed = None

	results = []
	for X in Xs:
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
			# The normalized windows depend only on Y and d, so they are
			# made for the first X that needs them and kept for the rest.
			if Y_normed is None:
				k = Y.shape[2]
				Y_flat = Y.reshape(-1)
				windows = np.lib.stride_tricks.as_strided(Y_flat,
					shape=(len_output, d*k), strides=(k*Y_flat.itemsize,
					Y_flat.itemsize), writeable=False)

				sqnorms = np.array(list(map(np.ndarray.dot, windows, windows)),
					dtype=Y.dtype)
				Y_normed = windows / np.sqrt(sqnorms)[:, None]

			X_normed = (X / np.linalg.norm(X)).ravel()
			dtype = np.result_type(X_normed, Y_normed)
			X_normed = X_normed.astype(dtype, copy=False)
			normed = Y_normed.astype(dtype, copy=False)

			scores_ = np.array(list(map(X_normed.dot, normed)), dtype=dtype)
			# nan_to_num changes only NaN and infinite values, so when there
			# are none it returns an identical copy, which is skipped.
			if np.isfinite(scores_).all():
				scores[0] = scores_
			else:
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
		results.append(np.array([[scores[idxs, argmaxs], argmaxs - n_pad]]
			).transpose(0, 2, 1))

	return results


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
		
		import sklearn.manifold
		conditional_P = sklearn.manifold._utils._binary_search_perplexity(
			distances, self.perplexity, verbose=False)

		eps = 1e-8
		marginal_sum = conditional_P.sum(axis=-1)
		marginal_sum[marginal_sum < eps] = eps

		#normalize the conditional_P to sum to 1 across the rows
		conditional_P = conditional_P / marginal_sum[:,None]

		n_rows, n_cols = conditional_P.shape
		if n_rows == n_samples and n_rows > 0 and all(
				len(neigh_row) == n_cols for neigh_row in neighbors):
			# Every row of neighbours is as long as its row of
			# probabilities, so the loop below would list the probabilities
			# row by row, each row's index once per neighbour, and the
			# neighbours row by row: the same values, in the same order, as
			# these arrays. coo_matrix picks the index dtype from the values
			# and converts both index lists to it, and the data keeps
			# conditional_P's dtype either way.
			data = conditional_P.ravel()
			rows = np.repeat(np.arange(n_rows), n_cols)
			cols = np.concatenate(neighbors)
		else:
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

