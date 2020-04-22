from __future__ import division, print_function, absolute_import
from sklearn.neighbors import NearestNeighbors
import numpy as np

class AbstractNearestNeighborsComputer(object):

    def __call__(self, affinity_mat):
        raise NotImplementedError()


class ScikitNearestNeighbors(AbstractNearestNeighborsComputer):

    def __init__(self, nn_n_jobs):
        self.nn_n_jobs = nn_n_jobs
        self.nn_object = NearestNeighbors(
            algorithm="brute", metric="precomputed",
            n_jobs=self.nn_n_jobs)

    def __call__(self, n_neighbors, affinity_mat):
        return self.nn_object.fit(np.max(affinity_mat)-affinity_mat).\
                kneighbors(X=np.max(affinity_mat) - affinity_mat, 
                  n_neighbors=min(n_neighbors+1, len(affinity_mat)),
                  return_distance=False)
