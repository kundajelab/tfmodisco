from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
from modisco.affinitymat import MeanNormalizer, MagnitudeNormalizer  
from nose.tools import raises


class TestNormalizers(unittest.TestCase):

    def test_normalizer(self): 
        rand_array = (np.random.random((100,10))+5.0)*10.0
        normalizer = MeanNormalizer().chain(MagnitudeNormalizer())
        normalized = normalizer.normalize(rand_array)
        np.testing.assert_almost_equal(np.mean(normalized), 0.0)
        np.testing.assert_almost_equal(np.linalg.norm(normalized), 1.0)



