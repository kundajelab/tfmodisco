from __future__ import division, print_function, absolute_import
from ..affinitymat.core import MeanNormalizer
from .. import backend as B
from .core import (AbstractSeqletsToOnedEmbedder,
                   AbstractSeqletsToOnedEmbedderFactory)
from .. import core as modiscocore
import itertools
import numpy as np
import sys
from joblib import Parallel, delayed
import scipy
import time
from .cython_advanced_gapped_kmer import (
    fast_recursively_get_gappedkmersandimp,
    unravel_fast_recursively_get_gappedkmersandimp,
    get_agkmer_to_totalseqimp)


def prepare_gapped_kmer_from_contribs(contrib_scores, topn, min_k,
                                      max_k, max_gap, max_len):
    #get the top n positiosn
    per_pos_imp = np.sum(contrib_scores, axis=-1)
    per_pos_bases = np.argmax(contrib_scores, axis=-1)
    #get the top n positions
    topn_pos = np.argsort(-per_pos_imp)[:topn]
    #prepare 'posbaseimp tuples':
    posbaseimptuples = sorted([ (pos, per_pos_bases[pos], per_pos_imp[pos])
                                 for pos in topn_pos ], key=lambda x:x[0])
    gappedkmersandimp = unravel_fast_recursively_get_gappedkmersandimp(
            posbaseimptuples=posbaseimptuples, max_k=max_k,
            max_gap=max_gap, max_len=max_len)
    
    #condense this by total imp on gapped gkmers across sequence
    gapped_kmer_to_totalseqimp = get_agkmer_to_totalseqimp(
                                  gappedkmersandimp, min_k=min_k)
    return gapped_kmer_to_totalseqimp


def prepare_gapped_kmer_from_seqlet(seqlet, topn, min_k,
                                    max_k, max_gap, max_len,
                                    take_fwd,
                                    onehot_track_name,
                                    toscore_track_names_and_signs):
    if (take_fwd):
        attr = "fwd"
    else:
        attr = "rev"
    onehot = getattr(seqlet[onehot_track_name], attr)
    contrib_scores = np.sum([getattr(seqlet[track_name], attr)*onehot*sign
            for track_name, sign in toscore_track_names_and_signs ], axis=0)
    return prepare_gapped_kmer_from_contribs(
            contrib_scores=contrib_scores, topn=topn, min_k=min_k,
            max_k=max_k, max_gap=max_gap, max_len=max_len) 


class AdvancedGappedKmerEmbedderFactory(object):

    def __init__(self, topn=20, min_k=4, max_k=6, max_gap=15, max_len=15, 
                       alphabet_size=4, n_jobs=10,
                       num_filters_to_retain=None):
        self.topn = topn
        self.min_k = min_k
        self.max_k = max_k
        self.max_gap = max_gap
        self.max_len = max_len
        self.alphabet_size = alphabet_size
        self.n_jobs = n_jobs

    def get_jsonable_config(self):
        return OrderedDict([
                ('topn', self.topn),
                ('min_k', self.min_k),
                ('max_k', self.max_k),
                ('max_gap', self.max_gap),
                ('max_len', self.max_len)
                ('alphabet_size', self.alphabet_size)
                ('n_jobs', self.n_jobs)
                ])

    def __call__(self, onehot_track_name, toscore_track_names_and_signs):
        return AdvancedGappedKmerEmbedder(
                topn=self.topn, min_k=self.min_k, max_k=self.max_k,
                max_gap=self.max_gap, max_len=self.max_len,
                alphabet_size=self.alphabet_size,
                n_jobs=self.n_jobs,
                onehot_track_name=onehot_track_name,
                toscore_track_names_and_signs=toscore_track_names_and_signs) 


class AdvancedGappedKmerEmbedder(AbstractSeqletsToOnedEmbedder):
    
    def __init__(self, topn, min_k, max_k, max_gap, max_len, 
                       alphabet_size,
                       n_jobs,
                       onehot_track_name,
                       toscore_track_names_and_signs):
        self.topn = topn
        self.min_k = min_k
        self.max_k = max_k
        self.max_gap = max_gap
        self.max_len = max_len
        self.alphabet_size = alphabet_size
        self.n_jobs = n_jobs
        self.onehot_track_name = onehot_track_name
        self.toscore_track_names_and_signs = toscore_track_names_and_signs

    def __call__(self, seqlets):
        advanced_gappedkmer_embeddings_fwd =\
            Parallel(n_jobs=self.n_jobs, verbose=True)(
                delayed(prepare_gapped_kmer_from_seqlet)(
                    seqlets[i],
                    self.topn, self.min_k,
                    self.max_k, self.max_gap,
                    self.max_len, True,
                    self.onehot_track_name,
                    self.toscore_track_names_and_signs)
                   for i in range(len(seqlets)))
        advanced_gappedkmer_embeddings_rev =\
            Parallel(n_jobs=self.n_jobs, verbose=True)(
                delayed(prepare_gapped_kmer_from_seqlet)(
                    seqlets[i],
                    self.topn, self.min_k,
                    self.max_k, self.max_gap,
                    self.max_len, False,
                    self.onehot_track_name,
                    self.toscore_track_names_and_signs)
                   for i in range(len(seqlets))) 

        template_to_startidx, embedding_size =\
            get_template_to_startidx_and_embedding_size(
                max_len=self.max_len, min_k=self.min_k,
                max_k=self.max_k, alphabet_size=self.alphabet_size)

        sparse_agkm_embeddings_fwd = get_sparse_mat_from_agkm_embeddings(
            agkm_embeddings=advanced_gappedkmer_embeddings_fwd,
            template_to_startidx=template_to_startidx,
            embedding_size=embedding_size)
        sparse_agkm_embeddings_rev = get_sparse_mat_from_agkm_embeddings(
            agkm_embeddings=advanced_gappedkmer_embeddings_rev,
            template_to_startidx=template_to_startidx,
            embedding_size=embedding_size)

        return sparse_agkm_embeddings_fwd, sparse_agkm_embeddings_rev


def get_template_to_startidx_and_embedding_size(
        max_len, min_k, max_k, alphabet_size):
    template_to_startidx = {}
    start_idx = 0
    for a_len in range(min_k, max_len+1):
        for num_nongap in range(min_k-2, min(a_len-2, max_k)+1):
            nongap_pos_combos = itertools.combinations(range(a_len-2), num_nongap)
            for nongap_pos_combo in nongap_pos_combos:
                template = [False]*(a_len-2)
                for nongap_pos in nongap_pos_combo:
                    template[nongap_pos] = True
                template = tuple([True]+template+[True])
                template_to_startidx[template] = start_idx
                start_idx += alphabet_size**(num_nongap+2)
    return template_to_startidx, start_idx


def get_template_and_offset_from_gkmer(gkmer):
    template = []
    offset = 0
    for letternum, (gapbefore, letteridx) in enumerate(gkmer):
        template.extend([False]*gapbefore)
        template.append(True)
        offset += letteridx*(4**letternum)
    template = tuple(template)
    return template, offset


def map_agkm_embedding_to_sparsevec(gapped_kmer_to_totalseqimp,
                                    template_to_startidx):
    data = []
    cols = []
    for gkmer, totalimp in gapped_kmer_to_totalseqimp.items():
        template,offset = get_template_and_offset_from_gkmer(gkmer)
        gkmeridx = template_to_startidx[template] + offset
        data.append(totalimp)
        cols.append(gkmeridx)
    assert len(cols)==len(gapped_kmer_to_totalseqimp)
    return data, cols


def get_sparse_mat_from_agkm_embeddings(agkm_embeddings,
                                        template_to_startidx,
                                        embedding_size):
    #not sure why, but parallelization doesn't help that much here?
    # so I am setting n_jobs to 1.
    all_agkm_data_and_cols = Parallel(n_jobs=1, verbose=True)(
        delayed(map_agkm_embedding_to_sparsevec)(
            agkm_embedding,
            template_to_startidx)
        for agkm_embedding in agkm_embeddings)
    
    row_ind = []
    data = []
    col_ind = []
    for (this_row_idx,
         (single_agkm_data, single_agkm_cols)) in enumerate(all_agkm_data_and_cols):
        data.extend(single_agkm_data)
        col_ind.extend(single_agkm_cols)
        row_ind.extend([this_row_idx for x in single_agkm_data])

    print("Constructing csr matrix...")
    start = time.time() 
    csr_mat = scipy.sparse.csr_matrix(
        (data, (np.array(row_ind).astype("int64"),
                np.array(col_ind).astype("int64"))),
        shape=(len(agkm_embeddings), embedding_size))
    print("csr matrix made in",time.time()-start,"s")
    return csr_mat

