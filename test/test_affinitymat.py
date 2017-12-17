from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from modisco.affinitymat import (MeanNormalizer, MagnitudeNormalizer, 
                                      MaxCrossCorrAffinityMatrixFromSeqlets,
                                      PatternComparisonSettings,
                                      GappedKmerEmbedder) 
from modisco import core
from nose.tools import raises


class TestNormalizers(unittest.TestCase):

    def test_normalizer(self): 
        rand_array = (np.random.random((100,10))+5.0)*10.0
        normalizer = MeanNormalizer().chain(MagnitudeNormalizer())
        normalized = normalizer(rand_array)
        np.testing.assert_almost_equal(np.mean(normalized), 0.0)
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)


class TestGappedKmerEmbedder(unittest.TestCase):

    alphabet_to_idx = {'A':0, 'C':1, 'G':2, 'T':3} 
    def seq_to_onehot(self, seq):
        to_return = np.zeros((len(seq), max(self.alphabet_to_idx.values())+1))
        for idx,letter in enumerate(seq):
            to_return[idx,self.alphabet_to_idx[letter]] = 1
        return to_return

    def test_gapped_kmer_embedder(self):
        alphabet_size=4
        kmer_len=3
        num_gaps=1

        gkmer_embedder = GappedKmerEmbedder(
            alphabet_size=alphabet_size,
            kmer_len=kmer_len, num_gaps=num_gaps,
            onehot_track_name=None, toscore_track_names_and_signs=None,
            normalizer=None)

        out = gkmer_embedder.gapped_kmer_embedding_func(
            onehot=np.array([self.seq_to_onehot("ACGT")]),
            to_embed=np.random.random((1,4,4)),
            batch_size=50, progress_update=None)
        self.assertListEqual(list(np.nonzero(out.squeeze())[0]),[1,6,18,23])


class TestMaxCrossCorr(unittest.TestCase):

    def test_max_cross_corr(self):

        seqlets = []
        #make 100 seqlets
        for i in range(100):
            arr = np.random.random((100,4))
            snippet = core.Snippet(fwd=arr, rev=arr[::-1,::-1],
                                   has_pos_axis=True) 
            seqlet = core.Seqlet(coor=core.SeqletCoordinates(
                           example_idx=0, start=0, end=100, is_revcomp=False)
                           ).add_snippet(data_track_name="scores",
                                         snippet=snippet)
            seqlets.append(seqlet)
        affinitymat = MaxCrossCorrAffinityMatrixFromSeqlets(
            pattern_comparison_settings=PatternComparisonSettings( 
                track_names=["scores"],
                track_transformer=MeanNormalizer().chain(MagnitudeNormalizer()),
                min_overlap=0.3))(seqlets)
         

