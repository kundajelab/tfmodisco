# gapped_kmer.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import scipy
import numpy as np

from numba import njit, prange
import numba

key_type = numba.types.int64
value_type = numba.types.float64

@njit(parallel=True)
def _extract_gkmers(X, min_k, max_k, max_gap, max_len, max_entries):
	nx = X.shape[0]
	keys = np.zeros((nx, max_entries), dtype='int64')
	scores = np.zeros((nx, max_entries), dtype='float64')

	for xi in prange(nx):
		n = X.shape[1]
		gkmer_attrs = numba.typed.Dict.empty(key_type=key_type, value_type=value_type)

		last_k_gkmers = []
		last_k_gkmers_attrs = []
		last_k_gkmers_hashes = []

		for i in range(n):
			base = int(X[xi, i, 1])
			attr = X[xi, i, 2]

			last_k_gkmers.append(np.array([i], dtype='int32'))
			last_k_gkmers_attrs.append(np.array([attr], dtype='float64'))
			last_k_gkmers_hashes.append(np.array([base+1], dtype='int64'))

		for k in range(2, max_k+1):
			for j in range(n):
				start_position = X[xi, j, 0]

				gkmers_ = []
				gkmer_attrs_ = []
				gkmer_hashes_ = []

				for i in range(j+1, n):
					position = X[xi, i, 0]
					base = int(X[xi, i, 1])
					attr = X[xi, i, 2]

					if (position - start_position) >= max_len:
						break

					for g in range(len(last_k_gkmers[j])):
						gkmer = last_k_gkmers[j][g]
						gkmer_attr = last_k_gkmers_attrs[j][g]
						gkmer_hash = last_k_gkmers_hashes[j][g]

						last_position = X[xi, gkmer, 0]
						if last_position >= position:
							break

						if (position - last_position) > max_gap:
							continue

						diff = int(position - last_position - 1)
						length = int(position - start_position)

						new_gkmer_hash = gkmer_hash + (base+1) * (5 ** length)
						new_gkmer_attr = gkmer_attr + attr
						
						gkmers_.append(i)
						gkmer_attrs_.append(new_gkmer_attr)
						gkmer_hashes_.append(new_gkmer_hash)

						if k >= min_k:
							gkmer_attrs[new_gkmer_hash] = gkmer_attrs.get(new_gkmer_hash, 0) + new_gkmer_attr / k

				if len(gkmers_) == 0:
					last_k_gkmers[j] = np.zeros(0, dtype='int32')
					last_k_gkmers_attrs[j] = np.zeros(0, dtype='float64')
					last_k_gkmers_hashes[j] = np.zeros(0, dtype='int64')
				else:
					last_k_gkmers[j] = np.array(gkmers_, dtype='int32')
					last_k_gkmers_attrs[j] = np.array(gkmer_attrs_, dtype='float64')
					last_k_gkmers_hashes[j] = np.array(gkmer_hashes_, dtype='int64')
		
		ny = len(gkmer_attrs)
		keys_ = np.empty(ny, dtype='int64')
		scores_ = np.empty(ny, dtype='float64')

		for i, key in enumerate(gkmer_attrs.keys()):
			keys_[i] = key
			scores_[i] = gkmer_attrs[key]

		idxs = np.argsort(-np.abs(scores_), kind='mergesort')[:max_entries]

		keys[xi] = keys_[idxs]
		scores[xi] = scores_[idxs]

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

	return csr_mat
