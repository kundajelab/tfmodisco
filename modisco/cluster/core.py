import sklearn
from .bruteforce_nn import knnsearch
import .phenograph_code as pheno


class AbstractClusterer(object):

    def cluster(self, affinit_mat):
        raise NotImplementedError()


class PhenographCluster(AbstractClusterer):
    
    def cluster(self, affinitymat):
        jaccardified_affinitymat = knn_jaccard_dist(affinitymat) 


