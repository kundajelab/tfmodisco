# gapped_kmer.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import scipy
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from numba import njit
from numba import uint64
import numba

from . import precompile

_FIBONACCI = np.uint64(11400714819323198485)


@njit(nogil=True)
def _gkmer_rows(X, lo, hi, min_k, max_k, max_gap, max_len, keys, scores):
	"""Rows lo to hi of `_extract_gkmers`, written into keys and scores.

	This computes what the earlier typed-Dict version computed, bit for bit,
	without a typed Dict, lists of arrays or a parallel loop, all of which
	were slow to compile. The (k-1)-gkmers of a row are kept grouped by their
	first position j, in the order the lists held them, and are extended in
	the same (k, j, i, g) order, so each key's sum is the same sequence of
	float64 additions, starting from 0.0 as `dict.get(h, 0)` did. Keys are
	kept in first-insertion order, the order the dict iterated them in.
	Buffers are grown before the loop over g, never inside it.
	"""

	n = X.shape[1]
	max_entries = keys.shape[1]

	prev_ptr = np.empty(n + 1, dtype=np.int64)
	next_ptr = np.empty(n + 1, dtype=np.int64)
	prev_idx = np.empty(16 * n + 16, dtype=np.int64)
	prev_attr = np.empty(16 * n + 16, dtype=np.float64)
	prev_hash = np.empty(16 * n + 16, dtype=np.int64)
	next_idx = np.empty(16 * n + 16, dtype=np.int64)
	next_attr = np.empty(16 * n + 16, dtype=np.float64)
	next_hash = np.empty(16 * n + 16, dtype=np.int64)

	# An open-addressing table: table[slot] is 1 + the insertion rank of the
	# key in that slot, or 0 when it is empty.
	tbits = 12
	table = np.zeros(1 << tbits, dtype=np.int64)
	ukeys = np.empty(2048, dtype=np.int64)
	usums = np.empty(2048, dtype=np.float64)
	uslot = np.empty(2048, dtype=np.int64)
	order = np.empty(2048, dtype=np.float64)
	heap = np.empty(max_entries, dtype=np.int64)

	# Every index below is non-negative; unsigned casts spare numba's
	# negative-index fix-up on each access.
	for xi in range(lo, hi):
		xi = uint64(xi)
		for i in range(n):
			prev_ptr[i] = i
			prev_idx[i] = i
			prev_attr[i] = X[xi, i, 2]
			prev_hash[i] = int(X[xi, i, 1]) + 1
		prev_ptr[n] = n

		nu = 0
		for k in range(2, max_k+1):
			cnt = 0
			for j in range(n):
				next_ptr[j] = cnt
				start_position = X[xi, j, 0]
				n_prev = prev_ptr[j+1] - prev_ptr[j]

				for i in range(j+1, n):
					i = uint64(i)
					position = X[xi, i, 0]
					base = int(X[xi, i, 1])
					attr = X[xi, i, 2]

					if (position - start_position) >= max_len:
						break

					# Room for n_prev more gkmers and keys.
					if k < max_k and cnt + n_prev > next_idx.shape[0]:
						size = max(2 * next_idx.shape[0], cnt + n_prev)
						t_idx = np.empty(size, dtype=np.int64)
						t_attr = np.empty(size, dtype=np.float64)
						t_hash = np.empty(size, dtype=np.int64)
						for q in range(cnt):
							t_idx[q] = next_idx[q]
							t_attr[q] = next_attr[q]
							t_hash[q] = next_hash[q]
						next_idx = t_idx
						next_attr = t_attr
						next_hash = t_hash

					if k >= min_k and nu + n_prev > ukeys.shape[0]:
						size = max(2 * ukeys.shape[0], nu + n_prev)
						t_keys = np.empty(size, dtype=np.int64)
						t_sums = np.empty(size, dtype=np.float64)
						t_slot = np.empty(size, dtype=np.int64)
						for q in range(nu):
							t_keys[q] = ukeys[q]
							t_sums[q] = usums[q]
							t_slot[q] = uslot[q]
						ukeys = t_keys
						usums = t_sums
						uslot = t_slot

					if k >= min_k and 2 * (nu + n_prev) > table.shape[0]:
						while 2 * (nu + n_prev) > (1 << tbits):
							tbits += 1
						table = np.zeros(1 << tbits, dtype=np.int64)
						shift = np.uint64(64 - tbits)
						for q in range(nu):
							s = np.int64((np.uint64(ukeys[q]) * _FIBONACCI) >> shift)
							while table[s] != 0:
								s = (s + 1) & (table.shape[0] - 1)
							table[s] = q + 1
							uslot[q] = s

					mask = table.shape[0] - 1
					shift = np.uint64(64 - tbits)
					length = int(position - start_position)
					step = (base+1) * (5 ** length)

					for g in range(prev_ptr[j], prev_ptr[j+1]):
						g = uint64(g)
						last_position = X[xi, uint64(prev_idx[g]), 0]
						if last_position >= position:
							break

						if (position - last_position) > max_gap:
							continue

						new_gkmer_hash = prev_hash[g] + step
						new_gkmer_attr = prev_attr[g] + attr

						if k < max_k:
							next_idx[uint64(cnt)] = i
							next_attr[uint64(cnt)] = new_gkmer_attr
							next_hash[uint64(cnt)] = new_gkmer_hash
							cnt += 1

						if k >= min_k:
							value = new_gkmer_attr / k
							slot = np.int64((np.uint64(new_gkmer_hash) * _FIBONACCI) >> shift)
							while True:
								rank = table[uint64(slot)]
								if rank == 0:
									ukeys[uint64(nu)] = new_gkmer_hash
									usums[uint64(nu)] = 0.0 + value
									uslot[uint64(nu)] = slot
									nu += 1
									table[uint64(slot)] = nu
									break

								if ukeys[uint64(rank - 1)] == new_gkmer_hash:
									usums[uint64(rank - 1)] = usums[uint64(rank - 1)] + value
									break

								slot = (slot + 1) & mask

			next_ptr[n] = cnt
			prev_ptr, next_ptr = next_ptr, prev_ptr
			prev_idx, next_idx = next_idx, prev_idx
			prev_attr, next_attr = next_attr, prev_attr
			prev_hash, next_hash = next_hash, prev_hash

		# The first num_entries of the stable argsort of -|sums| that the
		# dict version took, which orders by |sum| descending, NaNs last and
		# ties by insertion rank. order[q] is |sum|, or -1 for a NaN, so rank
		# r comes after rank q when order[r] < order[q], or when they are
		# equal and r > q. The heap keeps the num_entries ranks that come
		# first, with the one that comes last at its root.
		if order.shape[0] < nu:
			order = np.empty(max(nu, 2 * order.shape[0]), dtype=np.float64)
		for q in range(nu):
			order[q] = -1.0 if np.isnan(usums[q]) else abs(usums[q])

		num_entries = min(nu, max_entries)
		hn = 0
		for q in range(nu):
			wq = order[q]
			if hn < num_entries:
				c = hn
				hn += 1
				while c > 0:
					p = (c - 1) >> 1
					r = heap[p]
					if order[r] < wq or (order[r] == wq and r > q):
						break
					heap[c] = r
					c = p
				heap[c] = q

			elif order[heap[0]] < wq:
				c = 0
				while True:
					child = 2 * c + 1
					if child >= hn:
						break
					r = heap[child]
					if child + 1 < hn:
						s = heap[child + 1]
						if order[s] < order[r] or (order[s] == order[r] and s > r):
							child += 1
							r = s
					if not (order[r] < wq or (order[r] == wq and r > q)):
						break
					heap[c] = r
					c = child
				heap[c] = q

		for pos in range(num_entries - 1, -1, -1):
			r = heap[0]
			keys[xi, pos] = ukeys[r]
			scores[xi, pos] = usums[r]

			hn -= 1
			q = heap[hn]
			wq = order[q]
			c = 0
			while True:
				child = 2 * c + 1
				if child >= hn:
					break
				r = heap[child]
				if child + 1 < hn:
					s = heap[child + 1]
					if order[s] < order[r] or (order[s] == order[r] and s > r):
						child += 1
						r = s
				if not (order[r] < wq or (order[r] == wq and r > q)):
					break
				heap[c] = r
				c = child
			heap[c] = q

		for q in range(nu):
			table[uslot[q]] = 0


def _extract_gkmers(X, min_k, max_k, max_gap, max_len, max_entries):
	"""The top max_entries gapped k-mers of each row of X, by |score|.

	Rows are independent, so they are split into contiguous blocks that
	numba.get_num_threads() threads run through `_gkmer_rows`, which releases
	the GIL. The result does not depend on the number of threads or blocks.
	"""

	nx = X.shape[0]
	keys = np.zeros((nx, max_entries), dtype='int64')
	scores = np.zeros((nx, max_entries), dtype='float64')

	# Give back the threads of compile children that have exited
	# (precompile lowers this process's thread count while they run).
	precompile._reap()
	n_threads = numba.get_num_threads()
	n_blocks = max(1, min(nx, 16 * n_threads))
	bounds = [nx * b // n_blocks for b in range(n_blocks + 1)]

	def run(b):
		_gkmer_rows(X, bounds[b], bounds[b+1], min_k, max_k, max_gap, max_len,
			keys, scores)

	with ThreadPoolExecutor(n_threads) as pool:
		list(pool.map(run, range(n_blocks)))

	return keys, scores

def _gkmer_inputs(seqlets, topn, sign):
	"""The arrays `_gkmer_inputs_loop` builds for the forward and for the
	reverse pass, (X_fwd, X_bwd), built for all seqlets at once. Returns
	None, and the caller takes the per-seqlet loop, unless the seqlets have
	one shape, float32 or float64 contributions, finite products and at
	most one nonzero base per position.

	Those conditions make every value the loop computes come out the same:
	a position's four products hold at most one nonzero value, so their sum
	is that value (or +0.0) whatever order numpy adds them in; argmax and
	the elementwise products do not depend on the layout; and np.argsort
	along the last axis runs, row by row, the same sort as np.argsort on
	each row's own copy, so ties at the top-n boundary are broken the same
	way. Positions are distinct, so sorting the (pos, base, imp) tuples by
	pos is sorting the positions, and np.array of those tuples is float64.
	"""

	if len(seqlets) == 0 or topn < 1:
		return None

	shape = np.shape(seqlets[0].sequence)
	if len(shape) != 2 or shape[0] < 1:
		return None

	for seqlet in seqlets:
		if (np.shape(seqlet.sequence) != shape or
				np.shape(seqlet.hypothetical_contribs) != shape):
			return None

	onehot = np.stack([seqlet.sequence for seqlet in seqlets])
	with np.errstate(all='ignore'):
		# Non-finite products send the seqlets to the loop below, which
		# warns about them as it always has.
		contrib_scores = np.stack([seqlet.hypothetical_contribs for seqlet 
			in seqlets]) * onehot * sign

	if contrib_scores.dtype not in (np.float32, np.float64):
		return None

	if not np.isfinite(contrib_scores).all():
		return None

	if (np.count_nonzero(onehot, axis=-1) > 1).any():
		return None

	per_pos_imp = np.sum(contrib_scores, axis=-1)
	del contrib_scores

	Xs = []
	for take_fwd in (True, False):
		if take_fwd:
			imp = per_pos_imp
			bases = np.argmax(onehot, axis=-1)
		else:
			imp = np.ascontiguousarray(per_pos_imp[:, ::-1])
			bases = np.argmax(onehot[:, ::-1, ::-1], axis=-1)

		topn_pos = np.argsort(-imp, axis=-1)[:, :topn]
		topn_pos.sort(axis=-1)

		X = np.empty(topn_pos.shape + (3,), dtype=np.float64)
		X[:, :, 0] = topn_pos
		X[:, :, 1] = np.take_along_axis(bases, topn_pos, axis=-1)
		X[:, :, 2] = np.take_along_axis(imp, topn_pos, axis=-1)
		Xs.append(X)

	return tuple(Xs)


# Entries per block of the numpy work split over threads. Each thread other
# than the main one allocates from its own malloc arena, which keeps what is
# freed mapped, so the outputs are allocated in the calling thread and the
# threads' own temporaries are kept to about a megabyte.
BLOCK = 1 << 17


def _gkmer_csr(keys, scores, n_cols):
	"""The CSR matrix scipy builds from the COO triplets (row, keys[row, e],
	scores[row, e]) followed by eliminate_zeros, without scipy's per-row
	sort of (index, value) pairs: each row's keys are sorted as packed
	(key, slot) integers, and entries whose score is zero (the padding of
	rows with fewer than max_entries gkmers, and gkmers whose sum is zero)
	are dropped. Rows are independent: blocks of rows are sorted, then
	compacted into their place in the output, by numba.get_num_threads()
	threads (numpy releases the GIL in each step). Returns None, and the
	caller builds it through scipy, when the keys cannot be packed or a row
	holds a key twice (scipy would sum the two entries).
	"""

	n, n_entries = keys.shape
	slot_bits = max(1, (n_entries - 1).bit_length())
	if (keys.dtype != np.int64 or n == 0 or n_entries == 0 or
			(n_cols - 1).bit_length() + slot_bits > 63):
		return None

	slot_ids = np.arange(n_entries, dtype=np.int64)
	mask = (1 << slot_bits) - 1
	packed = np.empty((n, n_entries), dtype=np.int64)
	values = np.empty((n, n_entries), dtype=np.float64)
	counts = np.empty(n, dtype=np.int64)
	in_range = [True]

	step = max(1, BLOCK // n_entries)
	starts = list(range(0, n, step))

	def sort_rows(lo):
		hi = min(lo + step, n)
		block = keys[lo:hi]
		if block.min() < 0 or block.max() >= n_cols:
			in_range[0] = False
			return

		rows = packed[lo:hi]
		np.left_shift(block, slot_bits, out=rows)
		np.bitwise_or(rows, slot_ids, out=rows)
		rows.sort(axis=1)

		slots = rows & mask
		np.right_shift(rows, slot_bits, out=rows)
		values[lo:hi] = np.take_along_axis(scores[lo:hi], slots, axis=1)
		counts[lo:hi] = np.count_nonzero(values[lo:hi], axis=1)

	n_threads = max(1, min(numba.get_num_threads(), len(starts)))
	with ThreadPoolExecutor(n_threads) as pool:
		list(pool.map(sort_rows, starts))
		if not in_range[0]:
			return None

		indptr = np.zeros(n + 1, dtype=np.int64)
		np.cumsum(counts, out=indptr[1:])

		if indptr[-1] == packed.size:
			indices, data = packed.reshape(-1), values.reshape(-1)
		else:
			indices = np.empty(indptr[-1], dtype=np.int64)
			data = np.empty(indptr[-1], dtype=np.float64)

			def compact(lo):
				hi = min(lo + step, n)
				keep = values[lo:hi] != 0
				indices[indptr[lo]:indptr[hi]] = packed[lo:hi][keep]
				data[indptr[lo]:indptr[hi]] = values[lo:hi][keep]

			list(pool.map(compact, starts))
			del packed, values

	csr_mat = scipy.sparse.csr_matrix((data, indices, indptr), 
		shape=(n, n_cols))
	if not csr_mat.has_canonical_format:
		return None

	return csr_mat


def _gkmer_inputs_loop(seqlets, topn, take_fwd, sign):
	"""The array `_extract_gkmers` takes for one pass, built seqlet by
	seqlet: each seqlet's topn positions by contribution, in position
	order, as (position, base, contribution) rows."""

	Xs = []
	for seqlet in seqlets:
		onehot = seqlet.sequence
		contrib_scores = seqlet.hypothetical_contribs*onehot*sign

		if not take_fwd:
			onehot = onehot[::-1, ::-1]
			contrib_scores = contrib_scores[::-1, ::-1]

		#get the top n positiosn
		per_pos_imp = np.sum(contrib_scores, axis=-1)
		per_pos_bases = np.argmax(onehot, axis=-1)

		#get the top n positions
		topn_pos = np.argsort(-per_pos_imp)[:topn]

		X_ = sorted([(pos, per_pos_bases[pos], per_pos_imp[pos]) for pos in topn_pos], key=lambda x:x[0])
		X_ = np.array(X_)
		Xs.append(X_)

	return np.array(Xs)


def _seqlet_to_gkmers(seqlets, topn, min_k, max_k, max_gap, max_len, 
	max_entries, take_fwd, sign, X=None):
	"""A CSR matrix of each seqlet's top max_entries gapped k-mers. X, when
	given, is the input array for this pass, from `_gkmer_inputs` or
	`_gkmer_inputs_loop`."""

	if X is None:
		Xs = _gkmer_inputs(seqlets, topn, sign)
		if Xs is not None:
			X = Xs[0] if take_fwd else Xs[1]
		else:
			X = _gkmer_inputs_loop(seqlets, topn, take_fwd, sign)
		del Xs

	keys, scores = _extract_gkmers(X, min_k=min_k, max_k=max_k, 
		max_gap=max_gap, max_len=max_len, max_entries=max_entries)

	csr_mat = _gkmer_csr(keys, scores, 5**max_len)
	if csr_mat is not None:
		return csr_mat

	row_idxs = np.repeat(range(keys.shape[0]), keys.shape[1])
	csr_mat = scipy.sparse.csr_matrix((scores.flatten(), 
		(row_idxs, keys.flatten())), shape=(len(keys), 5**max_len))
	
	scipy.sparse.csr_matrix.eliminate_zeros(csr_mat)

	return csr_mat
