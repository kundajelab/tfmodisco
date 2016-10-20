from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import modisco
import modisco.util
import time


class TestScanRegions(unittest.TestCase):

    def setUp(self):
        pass
        
    def test_scan_regions_with_filters(self): 
        regions_to_scan = np.array([[
            [[0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.2, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0]]
        ],[
            [[0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.4, 0.5, 0.0, 0.0, 0.0],
             [0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]]
        ]])
        filters = np.array([[
            [1.0, 0.0, 1.0, 0.0],
            [2.0, 3.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 0.0]
        ],[
            [0.0, 0.0, 0.0, 0.0],
            [5.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 3.0, 2.0],
            [0.0, 1.0, 0.0, 1.0]
        ]])

        scanning_results = np.array(modisco.util.scan_regions_with_filters(
            filters=filters,
            regions_to_scan=regions_to_scan))

        print(scanning_results)
        #fwd scan: [0.5, 1.1, 1.9, 3.7, 1.0, 0.0]
        #rev scan: [0.2, 0.3, 0.6, 3.3, 2.9, 0.2]
        correct_answer = np.array([[[
                        [0.5, 1.1, 1.9, 3.7, 2.9, 0.2],
                        [0,   0,   0,   0,   1,   1]
                    ],[
                        [0.5, 1.1, 1.9, 3.7, 2.9, 0.2],
                        [1,   1,   1,   1,   0,   0]
                    ],
                  ],[
                        [[0.2, 2.9, 3.7, 1.9, 1.1, 0.5],
                         [0,   0,   1,   1,   1,   1]],
                        [[0.2, 2.9, 3.7, 1.9, 1.1, 0.5],
                         [1,   1,   0,   0,   0,   0]]
                  ]])
        np.testing.assert_allclose(scanning_results, correct_answer)
