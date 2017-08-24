from __future__ import division, print_function, absolute_import
import sklearn
from . import matplotlibhelpers as mplh
import numpy as np

def get_tsne_embedding(affinity_mat, perplexity, verbose=0, random_state=None):
    from sklearn import manifold
    tsne = sklearn.manifold.TSNE(metric='precomputed', perplexity=perplexity,
                                 verbose=verbose, random_state=random_state)
    dist_mat = np.max(affinity_mat)-affinity_mat
    embedding = tsne.fit_transform(dist_mat)
    return embedding

def color_tsne_embedding_by_clustering(embedding, clusters,
                                       *args, **kwargs):
    mplh.scatter_plot(xycoords=embedding, clusters=clusters,
                      *args, **kwargs)
    
