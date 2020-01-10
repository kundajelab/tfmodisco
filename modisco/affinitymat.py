from __future__ import division, print_function
from collections import namedtuple
import numpy as np
from joblib import Parallel, delayed
import sklearn
from sklearn.neighbors import NearestNeighbors
import time
import scipy
from scipy.sparse import csr_matrix
import leidenalg
from . import util as modiscoutil
from tqdm.notebook import trange, tqdm
import sys


#Seqlet data for imputation
_SeqlDatForImput = namedtuple("_SeqlDatForImput",
                              ["corelen", "flanklen", "onehot", "hyp"])


#fake constructor for tuple
def SeqlDatForImput(corelen, onehot, hyp):
    assert (len(onehot)-corelen)%2 == 0
    assert onehot.shape==hyp.shape
    flanklen = int((len(onehot)-corelen)/2)
    return _SeqlDatForImput(corelen=corelen, flanklen=flanklen,
                            onehot=onehot, hyp=hyp)


def compute_sim_on_pairs(oneseql_corelen, oneseql_onehot, oneseql_hyp,
                         seqlset_corelen, seqlset_onehot, seqlset_hyp,
                         min_overlap_frac, pair_sim_metric):

    assert oneseql_onehot.shape==oneseql_hyp.shape
    assert len(oneseql_onehot.shape)==2
    assert seqlset_onehot.shape==seqlset_hyp.shape
    assert len(seqlset_onehot.shape)==3

    assert (oneseql_onehot.shape[0]-oneseql_corelen)%2==0
    assert (seqlset_onehot.shape[1]-seqlset_corelen)%2==0
    oneseql_flanklen = int((oneseql_onehot.shape[0]-oneseql_corelen)/2)
    seqlset_flanklen = int((seqlset_onehot.shape[1]-oneseql_corelen)/2)

    min_overlap = int(np.ceil(min(oneseql_corelen, seqlset_corelen)
                              *min_overlap_frac))

    oneseql_actual = oneseql_onehot*oneseql_hyp
    seqlset_actual = seqlset_onehot*seqlset_hyp
    
    #iterate over all possible offsets of oneseql relative to seqlset
    startoffset = -(oneseql_corelen-min_overlap)
    endoffset = (seqlset_corelen-min_overlap)
    possible_offsets = np.array(range(startoffset, endoffset+1))
    #init the array that will store the similarity results
    sim_results = np.zeros((seqlset_onehot.shape[0], len(possible_offsets)))
    for offsetidx,offset in enumerate(possible_offsets):
        #compute the padding needed for the offset seqlets to be comparable
        oneseql_leftpad = max(offset, 0)  
        oneseql_rightpad = max(seqlset_corelen-(oneseql_corelen+offset),0) 
        #based on the padding, figure out how we would need to slice into
        # the available numpy arrays
        oneseql_slicestart = oneseql_flanklen-oneseql_leftpad
        oneseql_sliceend = oneseql_flanklen+oneseql_corelen+oneseql_rightpad

        #do the same for seqlset
        seqlset_leftpad = max(-offset, 0) 
        seqlset_rightpad = max((oneseql_corelen+offset)-seqlset_corelen, 0)
        seqlset_slicestart = seqlset_flanklen-seqlset_leftpad
        seqlset_slicesend = seqlset_flanklen+seqlset_corelen+seqlset_rightpad

        #slice to get the underlying data
        oneseqlactual_slice = (oneseql_actual[oneseql_slicestart:
                                              oneseql_sliceend])[None,:,:]
        oneseqlonehot_slice = oneseql_onehot[oneseql_slicestart:
                                             oneseql_sliceend] 
        oneseqlhyp_slice = oneseql_hyp[oneseql_slicestart:oneseql_sliceend]

        seqlsetactual_slice = seqlset_actual[:,seqlset_slicestart:
                                               seqlset_slicesend]
        seqlsetonehot_slice = seqlset_onehot[:,seqlset_slicestart:
                                               seqlset_slicesend]
        seqlsethyp_slice = seqlset_hyp[:,seqlset_slicestart:seqlset_slicesend]

        oneseql_imputed = oneseqlhyp_slice[None,:,:]*seqlsetonehot_slice
        seqlset_imputed = seqlsethyp_slice*oneseqlonehot_slice[None,:,:]
       
        sim_results[:,offsetidx] = (
            0.5*pair_sim_metric(oneseqlactual_slice, seqlset_imputed)
          + 0.5*pair_sim_metric(oneseql_imputed, seqlsetactual_slice)) 
    argmax = np.argmax(sim_results, axis=-1)
    return sim_results[np.arange(len(argmax)),argmax], possible_offsets[argmax]


class SequenceAffmatComputer_Impute(object):

    def __init__(self, metric, n_jobs, min_overlap_frac):
        self.min_overlap_frac = min_overlap_frac 
        self.pair_sim_metric = metric
        self.n_jobs = n_jobs

    def __call__(self, seqlets, onehot_trackname, hyp_trackname):

        hasrev = seqlets[0][onehot_trackname].hasrev

        seqlet_corelengths = [len(x) for x in seqlets]
        #for now, will just deal with case where all seqlets are of equal len
        assert len(set(seqlet_corelengths))==1; the_corelen=seqlet_corelengths[0]

        #max_seqlet_len will return the length of the longest seqlet core
        max_seqlet_len = max(seqlet_corelengths)
        #for each seqlet, figure out the maximum size of the flank needed
        # on each size. This is determined by the length of the longest
        # seqlet that each seqlet could be compared against
        flank_sizes = [max_seqlet_len-int(corelen*self.min_overlap_frac)
                       for corelen in seqlet_corelengths] 

        allfwd_onehot = (np.array( #I do the >0 at end to binarize
                            [seqlet[onehot_trackname].get_core_with_flank(
                             left=flank, right=flank, is_revcomp=False)
                             for seqlet,flank in zip(seqlets,flank_sizes)])>0) 
        allfwd_hyp = np.array(
                            [seqlet[hyp_trackname].get_core_with_flank(
                             left=flank, right=flank, is_revcomp=False)
                             for seqlet,flank in zip(seqlets,flank_sizes)]) 
        if (hasrev):
            allrev_onehot = allfwd_onehot[:,::-1,::-1]
            allrev_hyp = allfwd_hyp[:,::-1,::-1]

        assert allfwd_onehot.shape==allfwd_hyp.shape
        assert all([len(single_onehot)==(len(seqlet)+2*flanksize)
                    for (seqlet, single_onehot, flanksize) in
                    zip(seqlets, allfwd_onehot, flank_sizes)])

        indices = [(i,j) for i in range(len(seqlets))
                         for j in range(len(seqlets))]
        fwdresults = Parallel(n_jobs=self.n_jobs, verbose=True)(
                                delayed(compute_sim_on_pairs)(
                                    oneseql_corelen=the_corelen,
                                    oneseql_onehot=allfwd_onehot[i],
                                    oneseql_hyp=allfwd_hyp[i],
                                    seqlset_corelen=the_corelen,
                                    seqlset_onehot=allfwd_onehot,
                                    seqlset_hyp=allfwd_hyp,
                                    min_overlap_frac=self.min_overlap_frac,
                                    pair_sim_metric=self.pair_sim_metric)
                                for i in range(len(seqlets)))
        affmat = np.array([x[0] for x in fwdresults])
        assert np.max(np.abs(affmat.T-affmat)==0)
        offsets = np.array([x[1] for x in fwdresults])
        del fwdresults
        import gc
        gc.collect()

        if (hasrev):
            revresults = Parallel(n_jobs=self.n_jobs, verbose=True)(
                                delayed(compute_sim_on_pairs)(
                                    oneseql_corelen=the_corelen,
                                    oneseql_onehot=allrev_onehot[i],
                                    oneseql_hyp=allrev_hyp[i],
                                    seqlset_corelen=the_corelen,
                                    seqlset_onehot=allfwd_onehot,
                                    seqlset_hyp=allfwd_hyp,
                                    min_overlap_frac=self.min_overlap_frac,
                                    pair_sim_metric=self.pair_sim_metric)
                                for i in range(len(seqlets)))
            revaffmat = np.array([x[0] for x in revresults])
            revoffsets = np.array([x[1] for x in revresults])
            assert np.max(np.abs(revaffmat.T-revaffmat)==0)
            isfwdmat = affmat > revaffmat
            affmat = isfwdmat*affmat + (isfwdmat==False)*revaffmat
            offsets = isfwdmat*offsets + (isfwdmat==False)*revoffsets
            del revresults
            import gc
            gc.collect()

        return affmat, offsets, isfwdmat


def tsne_density_adaptation(dist_mat, perplexity,
                            max_neighbors=np.inf, min_prob=1e-4, verbose=True):
    n_samples = dist_mat.shape[0]
    #copied from https://github.com/scikit-learn/scikit-learn/blob/45dc891c96eebdb3b81bf14c2737d8f6540fabfe/sklearn/manifold/t_sne.py

    # Compute the number of nearest neighbors to find.
    # LvdM uses 3 * perplexity as the number of neighbors.
    #But i will have it be custom
    k = min(n_samples - 1, max_neighbors+1)
    # In the event that we have very small # of points
    # set the neighbors to n - 1.
    #k = min(n_samples - 1, int(3. * perplexity + 1))

    if verbose:
        print("[t-SNE] Computing {} nearest neighbors...".format(k))

    # Find the nearest neighbors for every point
    knn = NearestNeighbors(algorithm='brute', n_neighbors=k,
                           metric='precomputed')
    t0 = time.time()
    knn.fit(dist_mat)
    duration = time.time() - t0
    if verbose:
        print("[t-SNE] Indexed {} samples in {:.3f}s...".format(
            n_samples, duration))

    t0 = time.time()
    distances_nn, neighbors_nn = knn.kneighbors(
        None, n_neighbors=k)
    duration = time.time() - t0
    if verbose:
        print("[t-SNE] Computed neighbors for {} samples in {:.3f}s..."
              .format(n_samples, duration))

    # Free the memory
    del knn

    t0 = time.time()
    # Compute conditional probabilities such that they approximately match
    # the desired perplexity
    n_samples, k = neighbors_nn.shape
    distances = distances_nn.astype(np.float32, copy=False)
    neighbors = neighbors_nn.astype(np.int64, copy=False)
    conditional_P = sklearn.manifold._utils._binary_search_perplexity(
                distances, perplexity, verbose)
    #for some reason, likely a sklearn bug, a few of
    #the rows don't sum to 1...for now, fix by making them sum to 1
    #print(np.sum(np.sum(conditional_P, axis=1) > 1.1))
    #print(np.sum(np.sum(conditional_P, axis=1) < 0.9))
    assert np.all(np.isfinite(conditional_P)), \
        "All probabilities should be finite"

    P = csr_matrix((conditional_P.ravel(), neighbors.ravel(),
                    range(0, n_samples * k + 1, k)),
                   shape=(n_samples, n_samples))
    P = np.array(P.todense())
    P = P/np.sum(P,axis=1)[:,None]
    P = P*(P > min_prob) #getting rid of small probs for speed
    P = P/np.sum(P,axis=1)[:,None]

    #Symmetrize by multiplication with transpose
    P = P*P.T
    return P


class LeidenClustering(object):
    def __init__(self, partitiontype, n_iterations):
        self.partitiontype = partitiontype
        self.n_iterations = n_iterations

    def __call__(self, the_graph, seed):
        return leidenalg.find_partition(
                the_graph, self.partitiontype,
                weights=np.array(the_graph.es['weight']).astype(np.float64),
                n_iterations=self.n_iterations,
                seed=seed)


def average_over_different_seeds(
        affmat, clustering_procedure, nseeds, top_frac_to_keep=1.0):
    the_graph = modiscoutil.get_igraph_from_adjacency(adjacency=affmat)
    clusterings = []
    qualities = [] 
    for seed in tqdm(range(nseeds)):
        partition = clustering_procedure(the_graph=the_graph, seed=seed*100)
        clusterings.append(np.array(partition.membership))
        qualities.append(partition.quality())
    if (top_frac_to_keep < 1.0):
        from matplotlib import pyplot as plt
        plt.hist(qualities, bins=20)
        plt.show()
        top_frac_quality = sorted(qualities, reverse=True)[
                            int(len(qualities)*top_frac_to_keep)-1]
        clusterings = [clustering for clustering,quality
                       in zip(clusterings, qualities) if quality>=top_frac_quality]
    assert len(clusterings)==int(nseeds*top_frac_to_keep), len(clusterings)
    averaged_affmat = np.zeros((len(clusterings[0]), len(clusterings[0])))
    for clustering in clusterings:
        averaged_affmat += clustering[:,None]==clustering[None,:]
    averaged_affmat = averaged_affmat/len(clusterings)
    return averaged_affmat

    
def take_best_over_different_seeds(affmat, clustering_procedure, nseeds):
    the_graph = modiscoutil.get_igraph_from_adjacency(adjacency=affmat)
    best_clustering = None
    best_quality = None
    for seed in tqdm(range(nseeds)):
        partition = clustering_procedure(
                            the_graph=the_graph, seed=seed*100)
        quality = partition.quality()
        if ((best_quality is None) or (quality > best_quality)):
            best_quality = quality
            best_clustering = np.array(partition.membership)
            print("Quality:",best_quality)
            sys.stdout.flush()
    return best_clustering
