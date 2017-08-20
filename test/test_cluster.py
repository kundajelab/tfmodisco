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
from modisco.cluster.core import PhenographCluster


class TestPhenographCluster(unittest.TestCase):


    def test_phenograph_cluster_func(self):
        affinity_mat = np.random.random((500,500))
        clusters = PhenographCluster().cluster(affinity_mat=affinity_mat) 

