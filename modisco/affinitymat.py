from __future__ import division, print_function
from collections import namedtuple
import numpy as np
from joblib import Parallel, delayed
import sklearn
import sklearn.manifold
from sklearn.neighbors import NearestNeighbors
import time
import scipy
from scipy.sparse import csr_matrix
import leidenalg
from . import util as modiscoutil
from tqdm.notebook import trange, tqdm
import sys
import time


#Seqlet data for imputation
_SeqlDatForImput = namedtuple("_SeqlDatForImput",
                              ["corelen", "flanklen", "onehot", "hyp"])


def average_diffusion_distances(affmat, ts, k=None):
    evals, evecs = get_dmap_evecs_evals(affmat=affmat, k=k) 
    return np.mean([get_dists_given_dmap(
                     dmap=get_dmap_for_t(evecs=evecs, evals=evals, t=t))
                    for t in ts], axis=0) 


def get_diffusion_distances(affmat,ts,k=None):
    dmap = get_concat_dmap_coords(affmat=affmat, ts=ts, k=k)
    print(dmap.shape)
    start = time.time()
    print("Computing dists",flush=True)
    dists = get_dists_given_dmap(dmap=dmap)
    print("Computed dists",time.time()-start,flush=True)
    return dists


def get_dists_given_dmap(dmap): 
    return scipy.spatial.distance.squareform(
                scipy.spatial.distance.pdist(dmap, metric='euclidean'))


def get_concat_dmap_coords(affmat, ts, k):
    evecs, evals = get_dmap_evecs_evals(affmat=affmat, k=k) 
    return np.concatenate([get_dmap_for_t(evecs=evecs, evals=evals, t=t)
                           for t in ts], axis=1) 


def get_dmap_for_t(evecs, evals, t):
    return evecs@np.diag(np.power(evals,t)) 


def get_dmap_evecs_evals(affmat, k):
    if (k==None):
        k=len(affmat)
    assert np.abs(np.max(np.sum(affmat, axis=-1))-1.0) < 1e-7,\
                np.max(np.sum(affmat, axis=-1))
    assert np.abs(np.min(np.sum(affmat, axis=-1))-1.0) < 1e-7,\
                np.min(np.sum(affmat, axis=-1))
    start = time.time()
    print("Doing eigendecomposition",flush=True)
    evals, evecs = np.linalg.eig(affmat)
    print("Did eigendecomposition in",time.time()-start,flush=True)
    #based on https://github.com/DiffusionMapsAcademics/pyDiffMap/blob/d36e632089d564a4fc169d29f81c4783ddd39fd3/src/pydiffmap/diffusion_map.py#L154
    #discard the "all equal" eigenvector
    ix = np.argsort(evals)[::-1][1:1+k]
    evals = evals[ix]
    evecs = evecs[:,ix]
    return evecs, evals


def nearest_neighb_affmat_expo_decay(affmat, n_neighb, beta):
    #argsort each row, take n_neighb, set prob accordingly
    argsortrows = np.argsort(affmat,axis=-1)
    new_affmat = np.zeros_like(affmat)
    for idx,row in enumerate(argsortrows):
        new_affmat[idx, row[::-1][:n_neighb]] = np.exp(
                                                 -beta*np.arange(n_neighb))
    new_affmat = new_affmat/np.sum(new_affmat, axis=-1)[:,None]
    return new_affmat
    

def nearest_neighb_affmat(affmat, n_neighbs):
    #argsort each row, take n_neighb, set prob accordingly
    argsortrows = np.argsort(affmat,axis=-1)
    new_affmat = np.zeros_like(affmat)
    for idx,row in enumerate(argsortrows):
        for n_neighb in n_neighbs:
            nearest_neighbs = row[::-1][:n_neighb]
            new_affmat[idx,nearest_neighbs] += 1.0
    new_affmat = new_affmat/np.sum(new_affmat, axis=-1)[:,None]
    return new_affmat


#fake constructor for tuple
def SeqlDatForImput(corelen, onehot, hyp):
    assert (len(onehot)-corelen)%2 == 0
    assert onehot.shape==hyp.shape
    flanklen = int((len(onehot)-corelen)/2)
    return _SeqlDatForImput(corelen=corelen, flanklen=flanklen,
                            onehot=onehot, hyp=hyp)


#will return the offset w.r.t. seql1
def asymmetric_compute_sim_on_pair(seql1_corelen, seql1_hyp, seql1_onehot,
                                   seql2_corelen, seql2_hyp,
                                   min_overlap_frac, pair_sim_metric):
    sim_results, possible_offsets = asymmetric_compute_sim_on_pairs(
                                        oneseql_corelen=seql1_corelen,
                                        oneseql_hyp=seql1_hyp,
                                        oneseql_onehot=seql1_onehot, 
                                        seqlset_corelen=seql2_corelen,
                                        seqlset_hyp=np.array([seql2_hyp]),
                                        min_overlap_frac=min_overlap_frac,
                                        pair_sim_metric=pair_sim_metric)
    return sim_results[0], possible_offsets


#offsets returned will be w.r.t. oneseql_corelen
def asymmetric_compute_sim_on_pairs(
            oneseql_corelen, oneseql_hyp, oneseql_onehot,
            seqlset_corelen, seqlset_hyp,
            min_overlap_frac, pair_sim_metric):

    assert oneseql_onehot.shape==oneseql_hyp.shape
    assert len(oneseql_hyp.shape)==2
    assert len(seqlset_hyp.shape)==3

    assert (oneseql_hyp.shape[0]-oneseql_corelen)%2==0
    assert (seqlset_hyp.shape[1]-seqlset_corelen)%2==0
    oneseql_flanklen = int((oneseql_hyp.shape[0]-oneseql_corelen)/2)
    seqlset_flanklen = int((seqlset_hyp.shape[1]-seqlset_corelen)/2)

    min_overlap = int(np.ceil(min(oneseql_corelen, seqlset_corelen)
                              *min_overlap_frac))

    oneseql_actual = oneseql_onehot*oneseql_hyp

    #iterate over all possible offsets of seqlset relative to oneseql
    startoffset = -(seqlset_corelen-min_overlap)
    endoffset = (oneseql_corelen-min_overlap)
    possible_offsets = np.array(range(startoffset, endoffset+1))
    #init the array that will store the similarity results
    sim_results = np.zeros((seqlset_hyp.shape[0], len(possible_offsets)))
    for offsetidx,offset in enumerate(possible_offsets):
        #compute the padding needed for the offset seqlets to be comparable
        seqlset_leftpad = max(offset, 0)  
        seqlset_rightpad = max(oneseql_corelen-(seqlset_corelen+offset),0) 
        #based on the padding, figure out how we would need to slice into
        # the available numpy arrays
        #Check that sufficient padding is actually available
        assert seqlset_leftpad <= seqlset_flanklen
        assert seqlset_rightpad <= seqlset_flanklen
        seqlset_slicestart = seqlset_flanklen-seqlset_leftpad
        seqlset_sliceend = seqlset_flanklen+seqlset_corelen+seqlset_rightpad

        #do the same for oneseql
        oneseql_leftpad = max(-offset, 0) 
        oneseql_rightpad = max((seqlset_corelen+offset)-oneseql_corelen, 0)
        assert oneseql_leftpad <= oneseql_flanklen
        assert oneseql_rightpad <= oneseql_flanklen
        oneseql_slicestart = oneseql_flanklen-oneseql_leftpad
        oneseql_slicesend = oneseql_flanklen+oneseql_corelen+oneseql_rightpad

        #slice to get the underlying data
        seqlsethyp_slice = seqlset_hyp[:,seqlset_slicestart:seqlset_sliceend]
        oneseqlactual_slice = oneseql_actual[oneseql_slicestart:
                                             oneseql_slicesend]
        oneseqlonehot_slice = oneseql_onehot[oneseql_slicestart:
                                             oneseql_slicesend]
        seqlset_imputed = seqlsethyp_slice*oneseqlonehot_slice[None,:,:]
        sim = pair_sim_metric(seqlset_imputed, oneseqlactual_slice[None,:,:])
        sim_results[:,offsetidx] = sim
        #print(offset)
        #from modisco.visualization import viz_sequence
        #viz_sequence.plot_weights(oneseqlX_imputed) 
        #viz_sequence.plot_weights(seqlsetXactual_slice) 
        #print(sim)
    return sim_results, possible_offsets


class SequenceAffmatComputer_Impute(object):

    def __init__(self, pair_sim_metric, n_jobs, min_overlap_frac):
        self.min_overlap_frac = min_overlap_frac 
        self.pair_sim_metric = pair_sim_metric
        self.n_jobs = n_jobs

    #if other_seqlets is None, will compute similarity of seqlets to other
    # seqlets.
    def __call__(self, seqlets, onehot_trackname, hyp_trackname,
                       other_seqlets=None, verbose=True):

        hasrev = seqlets[0][onehot_trackname].hasrev

        seqlet_corelengths = [len(x) for x in seqlets]
        other_seqlet_corelengths = (
            seqlet_corelengths if other_seqlets is None else 
            [len(x) for x in other_seqlets])

        #for now, will just deal with case where all seqlets are of equal len
        assert len(set(seqlet_corelengths))==1;
        the_corelen=seqlet_corelengths[0]
        assert len(set(other_seqlet_corelengths))==1;
        other_corelen=other_seqlet_corelengths[0]

        #max_seqlet_len will return the length of the longest seqlet core
        max_seqlet_len = max(seqlet_corelengths)
        max_other_seqlet_len = max(other_seqlet_corelengths)

        #for each seqlet, figure out the maximum size of the flank needed
        # on each size. This is determined by the length of the longest
        # seqlet that each seqlet could be compared against
        flank_sizes = [max_other_seqlet_len-int(corelen*self.min_overlap_frac)
                       for corelen in seqlet_corelengths] 
        other_flank_sizes = [max_seqlet_len-int(corelen*self.min_overlap_frac)
                             for corelen in other_seqlet_corelengths]

        allfwd_onehot = np.array(
                            [seqlet[onehot_trackname].get_core_with_flank(
                             left=flank, right=flank, is_revcomp=False)
                             for seqlet,flank in zip(seqlets,flank_sizes)])
        allfwd_hyp = np.array(
                            [seqlet[hyp_trackname].get_core_with_flank(
                             left=flank, right=flank, is_revcomp=False)
                             for seqlet,flank in zip(seqlets,flank_sizes)]) 
        if (hasrev):
            allrev_onehot = allfwd_onehot[:,::-1,::-1]
            allrev_hyp = allfwd_hyp[:,::-1,::-1]

        if (other_seqlets is None):
            other_allfwd_onehot = allfwd_onehot
            other_allfwd_hyp = allfwd_hyp
            other_allrev_onehot = allrev_onehot
            other_allrev_hyp = allrev_hyp
        else:
            other_allfwd_onehot = (np.array( #I do the >0 at end to binarize
                [seqlet[onehot_trackname].get_core_with_flank(
                 left=flank, right=flank, is_revcomp=False)
                 for seqlet,flank in zip(other_seqlets,other_flank_sizes)])>0) 
            other_allfwd_hyp = np.array(
                    [seqlet[hyp_trackname].get_core_with_flank(
                     left=flank, right=flank, is_revcomp=False)
                     for seqlet,flank in zip(other_seqlets,other_flank_sizes)]) 
            if (hasrev):
                other_allrev_onehot = other_allfwd_onehot[:,::-1,::-1]
                other_allrev_hyp = other_allfwd_hyp[:,::-1,::-1]
            
        assert allfwd_onehot.shape==allfwd_hyp.shape
        assert other_allfwd_onehot.shape==other_allfwd_hyp.shape
        assert all([len(single_onehot)==(len(seqlet)+2*flanksize)
                    for (seqlet, single_onehot, flanksize) in
                    zip(seqlets, allfwd_onehot, flank_sizes)])
        if (other_seqlets is not None):
            assert all([len(single_onehot)==(len(seqlet)+2*flanksize)
                for (seqlet, single_onehot, flanksize) in
                zip(other_seqlets, other_allfwd_onehot, other_flank_sizes)])

        indices = [(i,j) for i in range(len(seqlets))
                         for j in range(len(seqlets))]
        asym_fwdresults = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                                delayed(asymmetric_compute_sim_on_pairs)(
                                    oneseql_corelen=the_corelen,
                                    oneseql_hyp=allfwd_hyp[i],
                                    oneseql_onehot=allfwd_onehot[i],
                                    seqlset_corelen=other_corelen,
                                    seqlset_hyp=other_allfwd_hyp,
                                    min_overlap_frac=self.min_overlap_frac,
                                    pair_sim_metric=self.pair_sim_metric)
                                for i in range(len(seqlets)))
        affmat = np.zeros((len(allfwd_onehot),len(other_allfwd_onehot))) 
        offsets = np.zeros((len(allfwd_onehot),len(other_allfwd_onehot))) 
        for i in range(affmat.shape[0]):
            for j in range(affmat.shape[1]):
                #if other_seqlets is None, execute procedures for symmetrizing
                # the affmat
                if (other_seqlets is None):
                    reoriented_complementary_sims =\
                        asym_fwdresults[j][0][i][::-1]
                    combined_asym_fwdresults = 0.5*(
                        asym_fwdresults[i][0][j] + reoriented_complementary_sims)
                    del reoriented_complementary_sims #defensive programming
                else:
                    combined_asym_fwdresults = asym_fwdresults[i][0][j]
                argmax_pos = np.argmax(combined_asym_fwdresults) 
                affmat[i][j] = combined_asym_fwdresults[argmax_pos]
                offsets[i][j] = asym_fwdresults[i][1][argmax_pos] 
        del asym_fwdresults
        del (argmax_pos, combined_asym_fwdresults) #deleting for defensive programming

        import gc
        gc.collect()

        if (hasrev):
            asym_revresults = Parallel(n_jobs=self.n_jobs, verbose=verbose)(
                                delayed(asymmetric_compute_sim_on_pairs)(
                                    oneseql_corelen=the_corelen,
                                    oneseql_hyp=allfwd_hyp[i],
                                    oneseql_onehot=allfwd_onehot[i],
                                    seqlset_corelen=other_corelen,
                                    seqlset_hyp=other_allrev_hyp,
                                    min_overlap_frac=self.min_overlap_frac,
                                    pair_sim_metric=self.pair_sim_metric)
                                for i in range(len(seqlets)))
            revaffmat = np.zeros((len(allrev_onehot),len(other_allrev_onehot))) 
            revoffsets = np.zeros(
                          (len(allrev_onehot),len(other_allrev_onehot))) 
            for i in range(revaffmat.shape[0]):
                for j in range(revaffmat.shape[1]):
                    #only do the symmetrization if other_seqlets was None
                    if (other_seqlets is None):
                        #no changing of coordinates needed, unlike in the
                        # fwd case
                        reoriented_complementary_sims = (
                            asym_revresults[j][0][i])
                        combined_asym_revresults = 0.5*(
                            asym_revresults[i][0][j]
                            + reoriented_complementary_sims)
                        del reoriented_complementary_sims #defensive programming
                    else:
                        combined_asym_revresults = asym_revresults[i][0][j]
                    argmax_pos = np.argmax(combined_asym_revresults) 
                    revaffmat[i][j] = combined_asym_revresults[argmax_pos]
                    revoffsets[i][j] = asym_revresults[i][1][argmax_pos] 
            del (argmax_pos, combined_asym_revresults)

            isfwdmat = affmat > revaffmat
            affmat = isfwdmat*affmat + (isfwdmat==False)*revaffmat
            offsets = isfwdmat*offsets + (isfwdmat==False)*revoffsets

            del asym_revresults
            import gc
            gc.collect()

        offsets = offsets.astype("int64")
        return affmat, offsets, isfwdmat


def tsne_density_adaptation(dist_mat, perplexity,
                            max_neighbors=np.inf, verbose=True):
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
    try:
        conditional_P = sklearn.manifold._utils._binary_search_perplexity(
                    distances, perplexity, verbose)
    except:
        #API change
        conditional_P = sklearn.manifold._utils._binary_search_perplexity(
                    distances, np.ones_like(distances).astype("int64"),
                    perplexity, verbose)
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
