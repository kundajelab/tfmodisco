# gapped_kmer.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import scipy
import numpy as np

from concurrent.futures import ThreadPoolExecutor

from numba import njit
import numba

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

	for xi in range(lo, hi):
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
						last_position = X[xi, prev_idx[g], 0]
						if last_position >= position:
							break

						if (position - last_position) > max_gap:
							continue

						new_gkmer_hash = prev_hash[g] + step
						new_gkmer_attr = prev_attr[g] + attr

						if k < max_k:
							next_idx[cnt] = i
							next_attr[cnt] = new_gkmer_attr
							next_hash[cnt] = new_gkmer_hash
							cnt += 1

						if k >= min_k:
							value = new_gkmer_attr / k
							slot = np.int64((np.uint64(new_gkmer_hash) * _FIBONACCI) >> shift)
							while True:
								rank = table[slot]
								if rank == 0:
									ukeys[nu] = new_gkmer_hash
									usums[nu] = 0.0 + value
									uslot[nu] = slot
									nu += 1
									table[slot] = nu
									break

								if ukeys[rank - 1] == new_gkmer_hash:
									usums[rank - 1] = usums[rank - 1] + value
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

	n_threads = numba.get_num_threads()
	n_blocks = max(1, min(nx, 16 * n_threads))
	bounds = [nx * b // n_blocks for b in range(n_blocks + 1)]

	def run(b):
		_gkmer_rows(X, bounds[b], bounds[b+1], min_k, max_k, max_gap, max_len,
			keys, scores)

	with ThreadPoolExecutor(n_threads) as pool:
		list(pool.map(run, range(n_blocks)))

	return keys, scores

def _seqlet_to_gkmers(seqlets, topn, min_k, max_k, max_gap, max_len, 
	max_entries, take_fwd, sign):

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

	X = np.array(Xs)
	keys, scores = _extract_gkmers(X, min_k=min_k, max_k=max_k, 
		max_gap=max_gap, max_len=max_len, max_entries=max_entries)
	
	row_idxs = np.repeat(range(keys.shape[0]), keys.shape[1])
	csr_mat = scipy.sparse.csr_matrix((scores.flatten(), 
		(row_idxs, keys.flatten())), shape=(len(keys), 5**max_len))
	
	scipy.sparse.csr_matrix.eliminate_zeros(csr_mat)

	return csr_mat
