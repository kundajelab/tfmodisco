import sklearn
from . import phenograph as ph


class ClusterResults(object):

    def __init__(self, cluster_indices):
        self.cluster_indices = cluster_indices 


class PhenographClusterResults(ClusterResults):

    def __init__(self, cluster_indices, hierarchy, Q):
        super(PhenographClusterResults, self).__init__(
         cluster_indices=cluster_indices)
        self.hierarchy = hierarchy
        self.Q = Q


class AbstractClusterer(object):

    def cluster(self, affinity_mat):
        raise NotImplementedError()


class PhenographCluster(AbstractClusterer):

    def __init__(self, k=30, min_cluster_size=10, jaccard=True,
                       primary_metric='euclidean',
                       n_jobs=-1, q_tol=1e-3, louvain_time_limit=2000,
                       nn_method='kdtree'):
        self.k = k
        self.min_cluster_size = min_cluster_size
        self.jaccard = jaccard
        self.primary_metric = primary_metric
        self.n_jobs = n_jobs
        self.q_tol = q_tol
        self.louvain_time_limit = louvain_time_limit
        self.nn_method = nn_method
    
    def cluster(self, affinity_mat):
        communities, graph, Q, hierarchy = ph.cluster.cluster(
            data=affinity_mat,
            k=self.k, min_cluster_size=self.min_cluster_size,
            jaccard=self.jaccard, primary_metric=self.primary_metric,
            n_jobs=self.n_jobs, q_tol=self.q_tol,
            louvain_time_limit=self.louvain_time_limit,
            nn_method=self.nn_method)
        return PhenographClusterResults(
                cluster_indices=communities,
                hierarchy=hierarchy,
                Q=Q)
        
 
