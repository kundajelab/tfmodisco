from __future__ import division, print_function, absolute_import
from ..affinitymat.core import MeanNormalizer
from .core import (AbstractSeqletsToOnedEmbedder,
                   AbstractSeqletsToOnedEmbedderFactory)
from .. import core as modiscocore
import itertools
import numpy as np
import sys
from joblib import Parallel, delayed
import scipy
import time


def fast_recursively_get_gappedkmersandimp(posbaseimptuples, max_k,
                                           max_gap, max_len):
    #preceding_posbaseimptuples: [(0-based-position, base, imp)...]
    #A gapped kmer representation: [(gapbefore, base)...]
    #Gappedkmersandimp: [
    #   (gapped_kmer_representation, totalimp)] <-- smallest first
    #startposandgappedkmersandimp: [ (startpos, gappedkmersandimp) ]
    #endpos_and_startposandgappedkmersandimp: [
    #   (endpos, startpos_and_gappedkmersandimp) ] <- earliest end first
    if (len(posbaseimptuples)==0):
        return []
    else:
        lastbasepos, lastbase, lastbaseimp = posbaseimptuples[-1]
        endpos_and_startposandgappedkmersandimp =\
            fast_recursively_get_gappedkmersandimp(
                posbaseimptuples[:-1], max_k=max_k,
                max_gap=max_gap, max_len=max_len)
        
        #fill out startposandgappedkmersandimp for this ending position
        startposandgappedkmersandimp_endingatthispos = []
        
        #maintain the property of 'latest start first';
        # lastbasepos is the start for the kmer of k=1
        startposandgappedkmersandimp_endingatthispos.append(
            (lastbasepos, [ ([(0, lastbase)], lastbaseimp) ] ) )
        #iterate in order of latest end first, as this will
        # allow us to 'break' early when
        # we get to endpositions that would violate the 'max_gap' criterion
        for (endpos, startposandgappedkmersandimp)\
                in endpos_and_startposandgappedkmersandimp[::-1]:     
            if ( (lastbasepos-endpos)+1 <= max_gap ):
                #iterate through startposandgappedkmersandimp in order.
                # This will go through latest start first. As a result, we
                # will be able to 'break' early when we encounter a startpos
                # that would violate the max_len criterion.
                for startpos, gappedkmersandimp in\
                        startposandgappedkmersandimp:
                    gappedkmersandimp_startingatthispos = []
                    if ( (lastbasepos-startpos)+1 <= max_len):
                        #iterate through gappedkmersandimp in forward order.
                        # This iterates through in order of smallest
                        # gappedkmer_rep first. As a result, we can break out
                        # of the loop early when we encounter a
                        # len(gappedkmer_rep) that would violate max_k
                        for (gappedkmer_rep, totalimp) in gappedkmersandimp:
                            if (len(gappedkmer_rep) < max_k):
                                #because we iterate through gappedkmersandimp
                                # in order of smallest gappedkmer_rep first,
                                # gappedkmersandimp_startingatthispos will
                                # also maintain that property.
                                gappedkmersandimp_startingatthispos.append(
                                    (gappedkmer_rep
                                     +[(lastbasepos-endpos, lastbase)],
                                     totalimp+lastbaseimp) )
                            else:
                                break
                    if len(gappedkmersandimp_startingatthispos) > 0:
                        #would need to sort this later to make sure property of
                        # being sorted in descending order of startpos is
                        # preserved
                        startposandgappedkmersandimp_endingatthispos.append(
                            (startpos, gappedkmersandimp_startingatthispos) )
            else:
                break #can stop iterating through
                      # endpos_and_startposandgappedkmersandimp
        
        endpos_and_startposandgappedkmersandimp.append(
            (lastbasepos+1, startposandgappedkmersandimp_endingatthispos  ) )
        
        return endpos_and_startposandgappedkmersandimp


def unravel_fast_recursively_get_gappedkmersandimp(posbaseimptuples, **kwargs):
    endpos_and_startposandgappedkmersandimp =\
        fast_recursively_get_gappedkmersandimp(
            posbaseimptuples=posbaseimptuples, **kwargs)
    return [(tuple(x[0]), x[1]) for endpos,startposandgappedkmersandimp
            in endpos_and_startposandgappedkmersandimp
            for startpos,gappedkmersandimp in startposandgappedkmersandimp
            for x in gappedkmersandimp]


def prepare_gapped_kmer_from_contribs(contrib_scores, topn, min_k,
                                      max_k, max_gap, max_len,
                                      max_entries):
    #get the top n positiosn
    per_pos_imp = np.sum(contrib_scores, axis=-1)
    per_pos_bases = np.argmax(contrib_scores, axis=-1)
    #get the top n positions
    topn_pos = np.argsort(-per_pos_imp)[:topn]
    #prepare 'posbaseimp tuples':
    posbaseimptuples = sorted([ (pos, per_pos_bases[pos], per_pos_imp[pos])
                                 for pos in topn_pos ], key=lambda x:x[0])
    #gappedkmersandimp = unravel_recursively_get_gappedkmersandimp(
    #    posbaseimptuples=posbaseimptuples, max_k=max_k,
    #    max_gap=max_gap, max_len=max_len)
    gappedkmersandimp = unravel_fast_recursively_get_gappedkmersandimp(
    #gappedkmersandimp = unravel_recursively_get_gappedkmersandimp(
            posbaseimptuples=posbaseimptuples, max_k=max_k,
            max_gap=max_gap, max_len=max_len)
    #print(sorted(gappedkmersandimp)[-10:])
    #print(sorted(gappedkmersandimp_2)[-10:])
    #assert tuple(sorted(gappedkmersandimp))==tuple(sorted(gappedkmersandimp_2))
    
    #condense this by total imp on gapped gkmers across sequence
    gapped_kmer_to_totalseqimp = {}
    for gapped_kmer_rep, gapped_kmer_imp in gappedkmersandimp:
        assert gapped_kmer_rep[0][0]==0 #no superfluous pre-padding
        if (len(gapped_kmer_rep) >= min_k):
            gapped_kmer_to_totalseqimp[gapped_kmer_rep] = (
                gapped_kmer_to_totalseqimp.get(gapped_kmer_rep, 0)
                + gapped_kmer_imp/len(gapped_kmer_rep)
                )
    #only retain the top max_entries entries
    return sorted(gapped_kmer_to_totalseqimp.items(),
                  key=lambda x: -abs(x[1]))[:max_entries]


def prepare_gapped_kmer_from_seqlet(seqlet, topn, min_k,
                                    max_k, max_gap, max_len, max_entries,
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
            max_k=max_k, max_gap=max_gap, max_len=max_len,
            max_entries=max_entries) 


def prepare_gapped_kmer_from_seqlet_and_make_sparse_vec_dat(
    seqlet, topn, min_k, max_k, max_gap, max_len, max_entries,
    take_fwd, onehot_track_name,
    toscore_track_names_and_signs, template_to_startidx):
    
    gapped_kmer_to_totalseqimp = prepare_gapped_kmer_from_seqlet(
        seqlet=seqlet, topn=topn, min_k=min_k,
        max_k=max_k, max_gap=max_gap, max_len=max_len,
        max_entries=max_entries,
        take_fwd=take_fwd, onehot_track_name=onehot_track_name,
        toscore_track_names_and_signs=toscore_track_names_and_signs)

    return gapped_kmer_to_totalseqimp
    #return list(gapped_kmer_to_totalseqimp.items())
    #return map_agkm_embedding_to_sparsevec(
    #         gapped_kmer_to_totalseqimp=gapped_kmer_to_totalseqimp,
    #         template_to_startidx=template_to_startidx)


class AdvancedGappedKmerEmbedderFactory(object):

    def __init__(self, topn=20, min_k=4, max_k=6, max_gap=15, max_len=15, 
                       max_entries=500,
                       alphabet_size=4, n_jobs=10):
        self.topn = topn
        self.min_k = min_k
        self.max_k = max_k
        self.max_gap = max_gap
        self.max_len = max_len
        self.max_entries = max_entries
        self.alphabet_size = alphabet_size
        self.n_jobs = n_jobs

    def get_jsonable_config(self):
        return OrderedDict([
                ('topn', self.topn),
                ('min_k', self.min_k),
                ('max_k', self.max_k),
                ('max_gap', self.max_gap),
                ('max_len', self.max_len),
                ('max_entries', self.max_entries),
                ('alphabet_size', self.alphabet_size),
                ('n_jobs', self.n_jobs)
                ])

    def __call__(self, onehot_track_name, toscore_track_names_and_signs):
        return AdvancedGappedKmerEmbedder(
                topn=self.topn, min_k=self.min_k, max_k=self.max_k,
                max_gap=self.max_gap, max_len=self.max_len,
                max_entries=self.max_entries,
                alphabet_size=self.alphabet_size,
                n_jobs=self.n_jobs,
                onehot_track_name=onehot_track_name,
                toscore_track_names_and_signs=toscore_track_names_and_signs) 


class AdvancedGappedKmerEmbedder(AbstractSeqletsToOnedEmbedder):
    
    def __init__(self, topn, min_k, max_k, max_gap, max_len, max_entries,
                       alphabet_size,
                       n_jobs,
                       onehot_track_name,
                       toscore_track_names_and_signs):
        self.topn = topn
        self.min_k = min_k
        self.max_k = max_k
        self.max_gap = max_gap
        self.max_len = max_len
        self.max_entries = max_entries
        self.alphabet_size = alphabet_size
        self.n_jobs = n_jobs
        self.onehot_track_name = onehot_track_name
        self.toscore_track_names_and_signs = toscore_track_names_and_signs

    def __call__(self, seqlets):

        template_to_startidx, embedding_size =\
            get_template_to_startidx_and_embedding_size(
                max_len=self.max_len, min_k=self.min_k,
                max_k=self.max_k, alphabet_size=self.alphabet_size)

        #sparse_agkm_embeddings_fwd_dataandcols = (
        #    Parallel(n_jobs=self.n_jobs, verbose=True)(
        #      delayed(prepare_gapped_kmer_from_seqlet_and_make_sparse_vec_dat)(
        #          seqlets[i],
        #          self.topn, self.min_k,
        #          self.max_k, self.max_gap,
        #          self.max_len, True,
        #          self.onehot_track_name,
        #          self.toscore_track_names_and_signs,
        #          template_to_startidx)
        #         for i in range(len(seqlets)))
        #) 

        #sparse_agkm_embeddings_rev_dataandcols = (
        #    Parallel(n_jobs=self.n_jobs, verbose=True)(
        #      delayed(prepare_gapped_kmer_from_seqlet_and_make_sparse_vec_dat)(
        #          seqlets[i],
        #          self.topn, self.min_k,
        #          self.max_k, self.max_gap,
        #          self.max_len, False, #'False' determines doing rc
        #          self.onehot_track_name,
        #          self.toscore_track_names_and_signs,
        #          template_to_startidx)
        #         for i in range(len(seqlets)))
        #) 

        advanced_gappedkmer_embeddings_fwd =\
            Parallel(n_jobs=self.n_jobs, verbose=True)(
                delayed(prepare_gapped_kmer_from_seqlet)(
                    seqlets[i],
                    self.topn, self.min_k,
                    self.max_k, self.max_gap,
                    self.max_len,
                    self.max_entries,
                    True,
                    self.onehot_track_name,
                    self.toscore_track_names_and_signs)
                   for i in range(len(seqlets)))

        #from matplotlib import pyplot as plt
        #plt.hist([len(x) for x in advanced_gappedkmer_embeddings_fwd], bins=20)
        #plt.show()
        #assert False
        advanced_gappedkmer_embeddings_rev =\
            Parallel(n_jobs=self.n_jobs, verbose=True)(
                delayed(prepare_gapped_kmer_from_seqlet)(
                    seqlets[i],
                    self.topn, self.min_k,
                    self.max_k, self.max_gap,
                    self.max_len,
                    self.max_entries,
                    False,
                    self.onehot_track_name,
                    self.toscore_track_names_and_signs)
                   for i in range(len(seqlets))) 

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
    for gkmer, totalimp in gapped_kmer_to_totalseqimp:
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
    for (this_row_idx, (single_agkm_data, single_agkm_cols)) in enumerate(
                                                       all_agkm_data_and_cols):
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

