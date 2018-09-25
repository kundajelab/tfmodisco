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
import pytest
from modisco.sliding_similarities import sliding_continousjaccard, sliding_dotproduct, sliding_similarity


class TestMetrics(unittest.TestCase):

    def test_continousjaccard(self):
        qa = np.random.randn(10, 4)
        ta = np.random.randn(20, 100, 4)

        match, size = sliding_continousjaccard(qa, ta)
        assert match.shape == (20, 91)
        assert size.shape == (20, 91)

    def test_dotproduct(self):
        qa = np.random.randn(10, 4)
        ta = np.random.randn(20, 100, 4)

        res = sliding_dotproduct(qa, ta)

        assert res.shape == (20, 91)

    def test_sliding_similarity(self):
        qa = np.random.randn(10, 4)
        ta = np.random.randn(20, 100, 4)

        match, size = sliding_similarity(qa, ta, metric='continousjaccard', n_jobs=2)
        match2, size2 = sliding_continousjaccard(qa, ta)
        assert np.allclose(match, match2)
        assert np.allclose(size, size2)

        match = sliding_similarity(qa, ta, metric='dotproduct', n_jobs=2)
        match2 = sliding_dotproduct(qa, ta)
        assert np.allclose(match, match2)
