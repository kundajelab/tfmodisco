import scipy
import itertools
import numpy as np

from joblib import Parallel, delayed

import time

from numba import jit, njit
import numba

key_type = numba.types.int64
value_type = numba.types.float64

@njit
def _extract_gkmers(posbaseimptuples, min_k, max_k, max_gap, max_len):
	n = len(posbaseimptuples)
	gkmer_attrs = numba.typed.Dict.empty(key_type=key_type, value_type=value_type)
	#gkmer_attrs = {}

	last_k_gkmers = []
	last_k_gkmers_attrs = []
	last_k_gkmers_hashes = []

	for i in range(n):
		base = posbaseimptuples[i, 1]
		attr = posbaseimptuples[i, 2]

		last_k_gkmers.append(np.array([i], dtype='int32'))
		last_k_gkmers_attrs.append(np.array([attr], dtype='float64'))
		last_k_gkmers_hashes.append(np.array([base+1], dtype='int64'))

		#if min_k <= 1:
		#	gkmer_hash = hash(posbaseimptuples[i])
		#	gkmer_attrs[gkmer_hash] = gkmer_attrs.get(gkmer_hash, 0) + attr

	for k in range(2, max_k+1):
		for j in range(n):
			start_position = posbaseimptuples[j][0]

			gkmers_ = []
			gkmer_attrs_ = []
			gkmer_hashes_ = []

			for i in range(j+1, n):
				position = posbaseimptuples[i, 0]
				base = int(posbaseimptuples[i, 1])
				attr = posbaseimptuples[i, 2]

				if (position - start_position) >= max_len:
					break

				for g in range(len(last_k_gkmers[j])):
					gkmer = last_k_gkmers[j][g]
					gkmer_attr = last_k_gkmers_attrs[j][g]
					gkmer_hash = last_k_gkmers_hashes[j][g]

					last_position = posbaseimptuples[gkmer][0]
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

	#return gkmer_attrs
	
	n = len(gkmer_attrs)
	keys = np.empty(n, dtype='int64')
	scores = np.empty(n, dtype='float64')

	for i, key in enumerate(gkmer_attrs.keys()):
		keys[i] = key
		scores[i] = gkmer_attrs[key]

	idxs = np.argsort(-np.abs(scores), kind='mergesort')[:500]
	return keys[idxs], scores[idxs]



def _seqlet_to_gkmers(seqlet, topn, min_k, max_k, max_gap, max_len, 
	max_entries, take_fwd, sign):
	
	tic = time.time()

	attr = "fwd" if take_fwd else "rev"

	onehot = getattr(seqlet["sequence"], attr)
	contrib_scores = getattr(seqlet["task0_hypothetical_contribs"], attr)*onehot*sign

	#get the top n positiosn
	per_pos_imp = np.sum(contrib_scores, axis=-1)
	per_pos_bases = np.argmax(contrib_scores, axis=-1)
	#per_pos_bases = np.argmax(onehot, axis=-1) # <- THIS IS CORRECT BUT KEEP COMMENTED
	#get the top n positions
	topn_pos = np.argsort(-per_pos_imp)[:topn]

	posbaseimptuples = sorted([(pos, per_pos_bases[pos], per_pos_imp[pos])
								 for pos in topn_pos ], key=lambda x:x[0])
	posbaseimptuples = np.array(posbaseimptuples)

	#gapped_kmer_to_totalseqimp = _extract_gkmers(posbaseimptuples, min_k=min_k, 
	#	max_k=max_k, max_gap=max_gap, max_len=max_len)

	keys, scores = _extract_gkmers(posbaseimptuples, min_k=min_k, 
		max_k=max_k, max_gap=max_gap, max_len=max_len)

	#only retain the top max_entries entries
	#gapped_kmer_to_totalseqimp = sorted(gapped_kmer_to_totalseqimp.items(),
	#			  key=lambda x: -abs(x[1]))[:max_entries]

	#for i in range(500):
	#	print(keys[i], scores[i], gapped_kmer_to_totalseqimp[i])

	#awdawddwa

	#return gapped_kmer_to_totalseqimp
	return keys, scores 


def _rc_gkmer(gkmer):
	gkmer_, prev_gap = [], 0
	for gap, base in reversed(gkmer):
		gkmer_.append((prev_gap, 3-base))
		prev_gap = gap
	return gkmer_

def _gkmers_to_csr(embedded_seqlets, min_k, max_k, max_len, alphabet_size=4):
	# Create csr matrix
	rows, cols, data = [], [], []

	for row_idx, emb_seqlet in enumerate(embedded_seqlets):
		for col_idx, attr in zip(*emb_seqlet):
			rows.append(row_idx)
			cols.append(col_idx)
			data.append(attr)

	csr_mat = scipy.sparse.csr_matrix(
		(data, (np.array(rows).astype("int64"),
				np.array(cols).astype("int64"))),
		shape=(len(embedded_seqlets), 5**max_len))

	return csr_mat

def _gkmers_to_csr2(embedded_seqlets, min_k, max_k, max_len, alphabet_size=4):
	rows, data, fwd_cols, bwd_cols = [], [], [], []

	for row_idx, emb_seqlet in enumerate(embedded_seqlets):
		for gkmer, attr in zip(emb_seqlet):
			rows.append(row_idx)
			data.append(attr)

			fwd_col_idx = _gkmer_to_col_idx(gkmer, template_to_startidx)
			bwd_col_idx = _gkmer_to_col_idx(_rc_gkmer(gkmer), template_to_startidx)

			fwd_cols.append(fwd_col_idx)
			bwd_cols.append(bwd_col_idx)

	fwd_csr_mat = scipy.sparse.csr_matrix(
		(data, (np.array(rows).astype("int64"),
				np.array(fwd_cols).astype("int64"))),
		shape=(len(embedded_seqlets), embedding_size))

	bwd_csr_mat = scipy.sparse.csr_matrix(
		(data, (np.array(rows).astype("int64"),
				np.array(bwd_cols).astype("int64"))),
		shape=(len(embedded_seqlets), embedding_size))

	return fwd_csr_mat, bwd_csr_mat


def AdvancedGappedKmerEmbedder(seqlets, sign, topn=20, min_k=4, max_k=6, 
	max_gap=15, max_len=15, max_entries=500, alphabet_size=4):

	print("start")
	tic = time.time()

	embedded_seqlets_fwd = [_seqlet_to_gkmers(seqlet, topn, min_k, max_k, max_gap, 
			max_len, max_entries, True, sign) for seqlet in seqlets]

	print(time.time() - tic, "a")

	tic = time.time()

	embedded_seqlets_bwd = [_seqlet_to_gkmers(seqlet, topn, min_k, max_k, max_gap, 
			max_len, max_entries, False, sign) for seqlet in seqlets]
	print(time.time() - tic, "b")

	sparse_agkm_embeddings_fwd = _gkmers_to_csr(
		embedded_seqlets=embedded_seqlets_fwd, min_k=min_k, max_k=max_k, 
		max_len=max_len, alphabet_size=4)


	sparse_agkm_embeddings_rev = _gkmers_to_csr(
		embedded_seqlets=embedded_seqlets_bwd, min_k=min_k, max_k=max_k, 
		max_len=max_len, alphabet_size=4)

	#a0, b0 = _gkmers_to_csr2(
	#	embedded_seqlets=embedded_seqlets_fwd, min_k=min_k, max_k=max_k, 
	#	max_len=max_len, alphabet_size=4)

	return sparse_agkm_embeddings_fwd, sparse_agkm_embeddings_rev
