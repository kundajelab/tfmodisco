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


class TestJaccardify(unittest.TestCase):

    def setUp(self):
        pass
        
    def test_jaccardify(self): 
        rand_mat = np.random.random((100,100)) 
        answer = modisco.util.jaccardifyDistMat(rand_mat) 
        t1 = time.time()
        parallel_answer = modisco.util.gpu_jaccardify(rand_mat, func_params_size=1000)
        t2 = time.time()
        print("Time taken in parallel jaccardify",t2-t1)
        np.testing.assert_allclose(answer, parallel_answer)
