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
                                 PatternCrossCorrSettings) 
from modisco import core
from nose.tools import raises


class TestNormalizers(unittest.TestCase):

    def test_normalizer(self): 
        rand_array = (np.random.random((100,10))+5.0)*10.0
        normalizer = MeanNormalizer().chain(MagnitudeNormalizer())
        normalized = normalizer(rand_array)
        np.testing.assert_almost_equal(np.mean(normalized), 0.0)
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)

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
                   pattern_crosscorr_settings=PatternCrossCorrSettings( 
                    track_names=["scores"],
                    normalizer=MeanNormalizer().chain(MagnitudeNormalizer()),
                    min_overlap=0.3))(seqlets)
         

