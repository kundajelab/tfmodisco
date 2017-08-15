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

    def get_fwac(self, max_peaks_per_seq): 
        fwac = coordproducers.FixedWindowAroundChunks(
                sliding=6,
                flank=1,
                suppress=4,
                min_ratio=0.5,
                max_peaks_per_seq=max_peaks_per_seq)
        return fwac

    def test_fixed_window_around_chunks(self):
        fwac = self.get_fwac(max_peaks_per_seq=1) 
        score_track=np.array([
            [0,1,2,3,4,4,3,2,2,1],
            [2,3,4,4,3,2,2,1,1,0]
        ]) 
        coords = fwac.get_coords(score_track=score_track)
        print(coords[0]+", "+coords[1]) 
