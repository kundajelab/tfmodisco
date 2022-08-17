import scipy
import itertools
import numpy as np

from joblib import Parallel, delayed

def _extract_gkmers(posbaseimptuples, min_k, max_k, max_gap, max_len):
	n = len(posbaseimptuples)

	gkmers = [[[((tup,), ((0, tup[1]),), tup[2])]] + [[] for i in range(
		max_k+1)] for j, tup in enumerate(posbaseimptuples)]
	gkmer_attrs = {}

	for i, (_, base, attr) in enumerate(posbaseimptuples):
		gkmer = ((0, base),)
		gkmer_attrs[gkmer] = gkmer_attrs.get(gkmer, 0) + attr

	for k in range(2, max_k+1):
		for i, (position, base, attr) in enumerate(posbaseimptuples):
			for j in range(i-1, -1, -1):
				start_position = posbaseimptuples[j][0]
				if (position - start_position) >= max_len:
					break

				for gkmer, gkmer_idxs, gkmer_attr in gkmers[j][k-2]:
					last_position = gkmer[-1][0]
					if last_position >= position:
						break

					if (position - last_position) > max_gap:
						continue

					diff = position - last_position - 1

					new_gkmer = gkmer + (posbaseimptuples[i],)
					new_gkmer_idxs = gkmer_idxs + ((diff, base),)
					new_gkmer_attr = gkmer_attr + attr

					tup = (new_gkmer, new_gkmer_idxs, new_gkmer_attr)

					gkmers[j][k-1].append(tup)
					gkmer_attrs[new_gkmer_idxs] = (new_gkmer_attr / 
						len(new_gkmer) + gkmer_attrs.get(new_gkmer_idxs, 0))


	return {gkmer: attr for gkmer, attr in gkmer_attrs.items() 
		if len(gkmer) >= min_k}


def _seqlet_to_gkmers(seqlet, topn, min_k, max_k, max_gap, max_len, 
	max_entries, take_fwd, sign):
	
	attr = "fwd" if take_fwd else "rev"

	onehot = getattr(seqlet["sequence"], attr)
	contrib_scores = getattr(seqlet["task0_hypothetical_contribs"], attr)*onehot*sign

	#get the top n positiosn
	per_pos_imp = np.sum(contrib_scores, axis=-1)
	per_pos_bases = np.argmax(contrib_scores, axis=-1)
	#per_pos_bases = np.argmax(onehot, axis=-1) # <- THIS IS CORRECT BUT KEEP COMMENTED
	#get the top n positions
	topn_pos = np.argsort(-per_pos_imp)[:topn]

	posbaseimptuples = sorted([ (pos, per_pos_bases[pos], per_pos_imp[pos])
								 for pos in topn_pos ], key=lambda x:x[0])

	gapped_kmer_to_totalseqimp = _extract_gkmers(posbaseimptuples, min_k=min_k, 
		max_k=max_k, max_gap=max_gap, max_len=max_len)

	#only retain the top max_entries entries
	gapped_kmer_to_totalseqimp = sorted(gapped_kmer_to_totalseqimp.items(),
				  key=lambda x: -abs(x[1]))[:max_entries]


	return gapped_kmer_to_totalseqimp

def _gkmer_to_col_idx(gkmer, template_to_startidx):
	template = []
	offset = 0
	for letternum, (gapbefore, letteridx) in enumerate(gkmer):
		template.extend([False]*gapbefore)
		template.append(True)
		offset += letteridx*(4**letternum)

	return template_to_startidx[tuple(template)] + offset

def _gkmers_to_csr(embedded_seqlets, min_k, max_k, max_len, alphabet_size=4):
	# Create lookup table
	template_to_startidx = {}
	embedding_size = 0
	for a_len in range(min_k, max_len+1):
		for num_nongap in range(min_k-2, min(a_len-2, max_k)+1):
			nongap_pos_combos = itertools.combinations(range(a_len-2), num_nongap)
			for nongap_pos_combo in nongap_pos_combos:
				template = [False]*(a_len-2)
				for nongap_pos in nongap_pos_combo:
					template[nongap_pos] = True

				template = tuple([True]+template+[True])
				template_to_startidx[template] = embedding_size
				embedding_size += alphabet_size**(num_nongap+2)

	# Create csr matrix
	rows, cols, data = [], [], []

	for row_idx, emb_seqlet in enumerate(embedded_seqlets):
		for gkmer, attr in emb_seqlet:
			col_idx = _gkmer_to_col_idx(gkmer, template_to_startidx)

			rows.append(row_idx)
			cols.append(col_idx)
			data.append(attr)

	csr_mat = scipy.sparse.csr_matrix(
		(data, (np.array(rows).astype("int64"),
				np.array(cols).astype("int64"))),
		shape=(len(embedded_seqlets), embedding_size))

	return csr_mat

def AdvancedGappedKmerEmbedder(seqlets, sign, topn=20, min_k=4, max_k=6, 
	max_gap=15, max_len=15, max_entries=500, alphabet_size=4, n_jobs=1):

	embedded_seqlets_fwd = Parallel(n_jobs=n_jobs, verbose=True)(
		delayed(_seqlet_to_gkmers)(seqlet, topn, min_k, max_k, max_gap, 
			max_len, max_entries, True, sign) for seqlet in seqlets)

	embedded_seqlets_bwd = Parallel(n_jobs=n_jobs, verbose=True)(
		delayed(_seqlet_to_gkmers)(seqlet, topn, min_k, max_k, max_gap, 
			max_len, max_entries, False, sign) for seqlet in seqlets)


	sparse_agkm_embeddings_fwd = _gkmers_to_csr(
		embedded_seqlets=embedded_seqlets_fwd, min_k=min_k, max_k=max_k, 
		max_len=max_len, alphabet_size=4)

	sparse_agkm_embeddings_rev = _gkmers_to_csr(
		embedded_seqlets=embedded_seqlets_bwd, min_k=min_k, max_k=max_k, 
		max_len=max_len, alphabet_size=4)

	return sparse_agkm_embeddings_fwd, sparse_agkm_embeddings_rev
