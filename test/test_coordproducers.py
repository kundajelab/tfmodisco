from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from modisco import core
import time
from nose.tools import raises
from modisco import coordproducers


class TestFixedWindowAroundChunks(unittest.TestCase):


    def get_fwac(self, max_peaks_per_seq,sliding, suppress): 
        fwac = coordproducers.FixedWindowAroundChunks(
                sliding=sliding,
                flank=1,
                suppress=suppress,
                min_ratio=0.5,
                max_seqlets_per_seq=max_peaks_per_seq)
        return fwac


    def test_fwac_suppress_1_min_ratio(self):
        fwac = self.get_fwac(max_peaks_per_seq=3, sliding=3, suppress=1) 
        score_track=np.array([
            [-10,-10,-0.6,3,4,5,4,3,-0.5,-10,-10],
        ]).astype("float") 
        coords = fwac.get_coords(score_track=score_track)

        self.assertEqual(coords[0].example_idx,0)
        self.assertEqual(coords[0].start,3)
        self.assertEqual(coords[0].end,8)
        self.assertEqual(coords[0].is_revcomp,False)

        self.assertEqual(coords[1].example_idx,0)
        self.assertEqual(coords[1].start,5)
        self.assertEqual(coords[1].end,10)
        self.assertEqual(coords[1].is_revcomp,False)

        self.assertEqual(len(coords),2)


    def test_fwac_suppress_1(self):
        fwac = self.get_fwac(max_peaks_per_seq=3, sliding=3, suppress=1) 
        score_track=np.array([
            [0,1,2,3,4,5,4,3.1,2,1,0],
        ]).astype("float") 
        coords = fwac.get_coords(score_track=score_track)

        self.assertEqual(coords[0].example_idx,0)
        self.assertEqual(coords[0].start,3)
        self.assertEqual(coords[0].end,8)
        self.assertEqual(coords[0].is_revcomp,False)

        self.assertEqual(coords[1].example_idx,0)
        self.assertEqual(coords[1].start,5)
        self.assertEqual(coords[1].end,10)
        self.assertEqual(coords[1].is_revcomp,False)

        self.assertEqual(coords[2].example_idx,0)
        self.assertEqual(coords[2].start,1)
        self.assertEqual(coords[2].end,6)
        self.assertEqual(coords[2].is_revcomp,False)

        self.assertEqual(len(coords),3)


    def test_fwac_suppress_0(self):
        fwac = self.get_fwac(max_peaks_per_seq=2, sliding=3, suppress=0) 
        score_track=np.array([
            [0,1,2,3,4.1,5,4,3,2,1,0],
        ]).astype("float") 
        coords = fwac.get_coords(score_track=score_track)

        self.assertEqual(coords[0].example_idx,0)
        self.assertEqual(coords[0].start,3)
        self.assertEqual(coords[0].end,8)
        self.assertEqual(coords[0].is_revcomp,False)

        self.assertEqual(coords[1].example_idx,0)
        self.assertEqual(coords[1].start,2)
        self.assertEqual(coords[1].end,7)
        self.assertEqual(coords[1].is_revcomp,False)

        self.assertEqual(len(coords),2)


    def test_fwac_basic(self):
        fwac = self.get_fwac(max_peaks_per_seq=1,
                             sliding=6, suppress=4) 
        score_track=np.array([
            [2,3,4,4,3,2,2,1,1,0],
            [1,2,3,4,4,3,2,1,1,0],
            [0,1,2,3,4,4,3,2,2,1],
            [0,1,1,3,3,4,4,2,2,1],
        ]).astype("float") 

        coords = fwac.get_coords(score_track=score_track)

        self.assertEqual(coords[0].example_idx,1)
        self.assertEqual(coords[0].start,0)
        self.assertEqual(coords[0].end,8)
        self.assertEqual(coords[0].is_revcomp,False)

        self.assertEqual(coords[1].example_idx,2)
        self.assertEqual(coords[1].start,1)
        self.assertEqual(coords[1].end,9)
        self.assertEqual(coords[1].is_revcomp,False)

        self.assertEqual(coords[2].example_idx,3)
        self.assertEqual(coords[2].start,2)
        self.assertEqual(coords[2].end,10)
        self.assertEqual(coords[2].is_revcomp,False)

