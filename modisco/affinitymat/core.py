from __future__ import division, print_function, absolute_import
import numpy as np
from .. import util as modiscoutil
from .. import core as modiscocore
from . import transformers
import sys
import time
import itertools
import scipy.stats
from scipy.sparse import coo_matrix
import gc
import sklearn
from joblib import Parallel, delayed
from tqdm import tqdm


def print_memory_use():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("MEMORY",process.memory_info().rss/1000000000)


class AbstractTrackTransformer(object):

    def __call__(self, inp):
        """
            inp: 2d array
        """
        raise NotImplementedError() 

    def chain(self, other_normalizer):        
        return AdhocTrackTransformer(
                func=(lambda x: other_normalizer(
                                self(x))))


class AdhocTrackTransformer(AbstractTrackTransformer):
    def __init__(self, func):
        self.func = func

    def __call__(self, inp):
        return self.func(inp)


class MeanNormalizer(AbstractTrackTransformer):

    def __call__(self, inp):
        return inp - np.mean(inp)


class MagnitudeNormalizer(AbstractTrackTransformer):

    def __call__(self, inp):
        return (inp / (np.linalg.norm(inp.ravel())+0.0000001))


class AttenuateOutliers(AbstractTrackTransformer):

    def __init__(self, fold_above_mean_threshold):
        self.fold_above_mean_threshold = fold_above_mean_threshold

    def __call__(self, inp):
        return np.maximum(np.abs(inp)/np.mean(np.abs(inp)),
                          self.fold_above_mean_threshold)*np.sign(inp)


class SquareMagnitude(AbstractTrackTransformer):

    def __call__(self, inp):
        return np.square(inp)*np.sign(inp) 


class L1Normalizer(AbstractTrackTransformer):

    def __call__(self, inp):
        abs_sum = np.sum(np.abs(inp))
        if (abs_sum==0):
            return inp
        else:
            return (inp/abs_sum)


class PatternComparisonSettings(object):
    def __init__(self, track_names, track_transformer, min_overlap):
        assert hasattr(track_names, '__iter__')
        self.track_names = track_names
        self.track_transformer = track_transformer
        self.min_overlap = min_overlap


class AbstractAffinityMatrixFromSeqlets(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class AbstractSparseAffmatFromFwdAndRevOneDVecs(object):
    def __call__(self, fwd_vecs, rev_vecs):
        raise NotImplementedError()


class AbstractAffinityMatrixFromOneD(object):

    def __call__(self, vecs1, vecs2):
        raise NotImplementedError()


def magnitude_norm_sparsemat(sparse_mat):
    return sklearn.preprocessing.normalize(sparse_mat, norm='l2', axis=1)
    #return sparse_mat.divide(sparse_mat.multiply(sparse_mat).sum(axis=-1))


def sparse_cosine_similarity(sparse_mat_1, sparse_mat_2):
    normed_sparse_mat_1 = magnitude_norm_sparsemat(sparse_mat=sparse_mat_1)
    normed_sparse_mat_2 = magnitude_norm_sparsemat(sparse_mat=sparse_mat_2)
    return normed_sparse_mat_1.dot(normed_sparse_mat_2.transpose())


#take the dot product of fwd_vecs with
# fwd_vecs and rev_vecs, take max over the fwd and rev sim, then return
# the top k
def top_k_fwdandrev_dot_prod(fwd_vecs2, fwd_vecs, rev_vecs,
                             slice_start, slice_end, k, initclusters):
    if (initclusters is not None):
        #'initclusters' is only relevant when fwd_vecs2==fwd_vecs
        assert len(initclusters)==fwd_vecs.shape[0]
        assert len(initclusters)==fwd_vecs2.shape[0]
    fwd_vecs2_slice = fwd_vecs2[slice_start:slice_end]
    initclusters_slice = (None if initclusters is None
                          else initclusters[slice_start:slice_end])
    k = min(k, fwd_vecs.shape[0])
    if (scipy.sparse.issparse(fwd_vecs2_slice)):
        fwd_dot = np.array(fwd_vecs2_slice.dot(fwd_vecs.transpose()).todense())
    else:
        fwd_dot = np.matmul(fwd_vecs2_slice,fwd_vecs.T) 
    if (rev_vecs is not None):
        if (scipy.sparse.issparse(fwd_vecs2_slice)):
            rev_dot = np.array(fwd_vecs2_slice.dot(rev_vecs.transpose())
                                              .todense())
        else:
            rev_dot = np.matmul(fwd_vecs2_slice, rev_vecs.T)
        dotprod = np.maximum(fwd_dot, rev_dot)
    else:
        dotprod = fwd_dot

    #dotprod has shape batchsize X num_seqlets in fwd_vecs
    dotprod_argsort = np.argsort(-dotprod, axis=-1) 
    sorted_topk_indices = [] 
    sorted_topk_sims = []
    for row_idx,argsort_row in enumerate(dotprod_argsort): 
        combined_neighbor_row = [] 
        neighbor_row_topnn = argsort_row[:k] 
        #combined_neighbor_row ends up being the union of the standard nearest  
        # neighbors plus the nearest neighbors if focusing on the initclusters 
        combined_neighbor_row.extend(neighbor_row_topnn) 
        if (initclusters_slice is not None): 
            neighbor_set_topnn = set(neighbor_row_topnn) 
            initcluster_for_this_row = initclusters_slice[row_idx]
            combined_neighbor_row.extend([ 
                y for y in ([x for x in argsort_row 
                    if initclusters[x]==initcluster_for_this_row][:k]) 
                if y not in neighbor_set_topnn]) 
        sorted_topk_indices.append(
            np.array(combined_neighbor_row).astype("int")) 
        sorted_topk_sims.append(dotprod[row_idx][combined_neighbor_row])
    ##get the top k indices
    #top_k_indices = np.argpartition(dotprod, -k, axis=1)[:,-k:]
    #sims = np.take_along_axis(arr=dotprod, indices=top_k_indices, axis=1)

    ##sort by similarity
    #sims_argsort_result = np.argsort(sims, axis=-1)
    #sorted_topk_sims = np.take_along_axis(arr=sims,
    #                                      indices=sims_argsort_result, axis=1)
    #sorted_topk_indices = np.take_along_axis(arr=top_k_indices,
    #                                       indices=sims_argsort_result, axis=1)
    return (sorted_topk_indices, sorted_topk_sims)


class SparseNumpyCosineSimFromFwdAndRevOneDVecs(
        AbstractSparseAffmatFromFwdAndRevOneDVecs):

    def __init__(self, n_neighbors, verbose, memory_cap_gb=1.0):
        self.n_neighbors = n_neighbors   
        self.verbose = verbose
        self.memory_cap_gb = memory_cap_gb

    def __call__(self, fwd_vecs, rev_vecs, initclusters, fwd_vecs2=None):
        #fwd_vecs2 is used when you don't just want to compute self-similarities

        #normalize the vectors 
        fwd_vecs = magnitude_norm_sparsemat(sparse_mat=fwd_vecs)
        if (rev_vecs is not None):
            rev_vecs = magnitude_norm_sparsemat(sparse_mat=rev_vecs)
        else:
            rev_vecs = None

        if (fwd_vecs2 is None):
            fwd_vecs2 = fwd_vecs
        else:
            fwd_vecs2 = magnitude_norm_sparsemat(sparse_mat=fwd_vecs2)

        #fwd_sims = fwd_vecs.dot(fwd_vecs.transpose())
        #rev_sims = fwd_vecs.dot(rev_vecs.transpose())

        #assuming float64 for the affinity matrix, figure out the batch size
        # to use given the memory cap
        memory_cap_gb = (self.memory_cap_gb if rev_vecs
                         is None else self.memory_cap_gb/2.0)
        batch_size = int(memory_cap_gb*(2**30)/(fwd_vecs.shape[0]*8))
        batch_size = min(max(1,batch_size),fwd_vecs2.shape[0])
        if (self.verbose):
            print("Batching in slices of size",batch_size)
            sys.stdout.flush()

        neighbors, sims = [], []
        for i in tqdm(range(0,fwd_vecs2.shape[0],batch_size)):
            neighbors_batch, sims_batch = top_k_fwdandrev_dot_prod(
                                         fwd_vecs2=fwd_vecs2,
                                         fwd_vecs=fwd_vecs,
                                         rev_vecs=rev_vecs,
                                         slice_start=i,
                                         slice_end=(i+batch_size),
                                         k=self.n_neighbors+1,
                                         initclusters=initclusters)
            neighbors.extend(neighbors_batch)
            sims.extend(sims_batch)

        return sims, neighbors


class NumpyCosineSimilarity(AbstractAffinityMatrixFromOneD):

    def __init__(self, verbose, rows_per_batch=500):
        self.verbose = verbose
        self.rows_per_batch = rows_per_batch

    def __call__(self, vecs1, vecs2):

        start_time = time.time()
        if (scipy.sparse.issparse(vecs1)):
            vecs1 = magnitude_norm_sparsemat(sparse_mat=vecs1)
            vecs2 = magnitude_norm_sparsemat(sparse_mat=vecs2)

            if (self.verbose):
                print("Batching in slices of size",self.rows_per_batch)
                sys.stdout.flush()

            transpose_vecs2 = vecs2.transpose()
            to_return = np.zeros((vecs1.shape[0], vecs2.shape[0]))
            for i in tqdm(range(0, vecs1.shape[0], self.rows_per_batch)):
                to_return[i:min(i+self.rows_per_batch, vecs1.shape[0])] =\
                    np.array(vecs1[i:i+self.rows_per_batch]
                              .dot(transpose_vecs2).todense())
            #to_return = vecs1.dot(vecs2.transpose())
            #cast to dense for now
            #to_return = np.array(to_return.todense())
        else:
            normed_vecs1 = np.nan_to_num(
                            vecs1/np.linalg.norm(vecs1, axis=1)[:,None],
                            copy=False)
            normed_vecs2 = np.nan_to_num(
                            vecs2/np.linalg.norm(vecs2, axis=1)[:,None],
                            copy=False)
            if (self.verbose):
                print("Normalization computed in",
                      round(time.time()-start_time,2),"s")
                sys.stdout.flush()
            #do the multiplication on the CPU
            to_return = np.dot(normed_vecs1,normed_vecs2.T)
            end_time = time.time()
        
            if (self.verbose):
                print("Cosine similarity mat computed in",
                      round(end_time-start_time,2),"s")
                sys.stdout.flush()

        return to_return


def contin_jaccard_vec_mat_sim(a_row, mat):
    union = np.sum(np.maximum(np.abs(a_row[None,:]),
                              np.abs(mat)),axis=1)
    intersection = np.sum(np.minimum(np.abs(a_row[None,:]),
                                     np.abs(mat))
                          *np.sign(a_row[None,:])
                          *np.sign(mat), axis=1)
    union = np.maximum(union, 1e-7) #avoid div by 0
    return intersection.astype("float")/union


class ContinJaccardSimilarity(AbstractAffinityMatrixFromOneD):

    def __init__(self, verbose=True, n_cores=1, make_positive=False):
        self.verbose = verbose
        self.n_cores = n_cores
        self.make_positive = make_positive

    def __call__(self, vecs1, vecs2):

        #trying to avoid div by 0 in the normalization
        start_time = time.time()
        normed_vecs1 = vecs1/np.maximum(
            np.sum(np.abs(vecs1), axis=1)[:,None], 1e-7)
        normed_vecs2 = vecs2/np.maximum(
            np.sum(np.abs(vecs2), axis=1)[:,None], 1e-7) 
        if (self.verbose):
            print("Normalization computed in",
                  round(time.time()-start_time,2),"s")
            sys.stdout.flush()

        similarity_rows = []

        job_arguments = []
        for idx in range(0,len(normed_vecs1)):
            job_arguments.append(normed_vecs1[idx])
        to_concat = (Parallel(n_jobs=self.n_cores)
                       (delayed(contin_jaccard_vec_mat_sim)(
                            job_arg, normed_vecs2)
                        for job_arg in job_arguments))
        to_return = np.array(to_concat)
        end_time = time.time()
    
        if (self.verbose):
            print("Contin jaccard similarity mat computed in",
                  round(end_time-start_time,2),"s")
            sys.stdout.flush()

        if (self.make_positive):
            to_return = to_return + 1.0

        return to_return


class AffmatFromSeqletEmbeddings(AbstractAffinityMatrixFromSeqlets):

    def __init__(self, seqlets_to_1d_embedder,
                       affinity_mat_from_1d, verbose):
        self.seqlets_to_1d_embedder = seqlets_to_1d_embedder
        self.affinity_mat_from_1d = affinity_mat_from_1d 
        self.verbose = verbose

    def __call__(self, seqlets):

        cp1_time = time.time()
        if (self.verbose):
            print("Beginning embedding computation")
            sys.stdout.flush()

        embedding_fwd, embedding_rev = self.seqlets_to_1d_embedder(seqlets)

        cp2_time = time.time()
        if (self.verbose):
            print("Finished embedding computation in",
                  round(cp2_time-cp1_time,2),"s")
            sys.stdout.flush()

        if (self.verbose):
            print("Starting affinity matrix computations")
            sys.stdout.flush()

        affinity_mat_fwd = self.affinity_mat_from_1d(
                            vecs1=embedding_fwd, vecs2=embedding_fwd)  
        affinity_mat_rev = (self.affinity_mat_from_1d(
                             vecs1=embedding_fwd, vecs2=embedding_rev)
                            if (embedding_rev is not None) else None)
        #check for enforced symmetry
        assert np.max(np.abs(affinity_mat_fwd.T - affinity_mat_fwd))<1e-3,\
                np.max(np.abs(affinity_mat_fwd.T - affinity_mat_fwd))
        #This assert need not hold anymore with filter embeddings, which aren't
        # guaranteed revcomp symmetry...
        #if (affinity_mat_rev is not None):
        #    assert np.max(np.abs(affinity_mat_rev.T - affinity_mat_rev))<1e-3,\
        #            np.max(np.abs(affinity_mat_rev.T - affinity_mat_rev))

        cp3_time = time.time()

        if (self.verbose):
            print("Finished affinity matrix computations in",
                  round(cp3_time-cp2_time,2),"s")
            sys.stdout.flush()

        return (np.maximum(affinity_mat_fwd, affinity_mat_rev) 
                if (affinity_mat_rev is not None)
                else np.array(affinity_mat_fwd))


class SparseAffmatFromFwdAndRevSeqletEmbeddings(
        AbstractAffinityMatrixFromSeqlets):

    def __init__(self, seqlets_to_1d_embedder,
                       sparse_affmat_from_fwdnrev1dvecs, verbose):
        self.seqlets_to_1d_embedder = seqlets_to_1d_embedder
        self.sparse_affmat_from_fwdnrev1dvecs =\
            sparse_affmat_from_fwdnrev1dvecs
        self.verbose = verbose

    def __call__(self, seqlets, initclusters):

        cp1_time = time.time()
        if (self.verbose):
            print("Beginning embedding computation")
            print_memory_use()
            sys.stdout.flush()

        embedding_fwd, embedding_rev = self.seqlets_to_1d_embedder(seqlets)
        gc.collect()

        cp2_time = time.time()
        if (self.verbose):
            print("Finished embedding computation in",
                  round(cp2_time-cp1_time,2),"s")
            print_memory_use()
            sys.stdout.flush()

        if (self.verbose):
            print("Starting affinity matrix computations")
            print_memory_use()
            sys.stdout.flush()

        sparse_affmat, neighbors = self.sparse_affmat_from_fwdnrev1dvecs(
                                        fwd_vecs=embedding_fwd,
                                        rev_vecs=embedding_rev,
                                        initclusters=initclusters)

        cp3_time = time.time()

        if (self.verbose):
            print("Finished affinity matrix computations in",
                  round(cp3_time-cp2_time,2),"s")
            print_memory_use()
            sys.stdout.flush()
        return sparse_affmat, neighbors


class MaxCrossMetricAffinityMatrixFromSeqlets(
        AbstractAffinityMatrixFromSeqlets):

    def __init__(self, pattern_comparison_settings,
                       cross_metric):
        self.pattern_comparison_settings = pattern_comparison_settings
        self.cross_metric = cross_metric

    def __call__(self, seqlets):
        (all_fwd_data, all_rev_data) =\
            modiscocore.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=self.pattern_comparison_settings.track_names,
                track_transformer=
                    self.pattern_comparison_settings.track_transformer)
        #apply the cross metric
        cross_metrics_fwd = self.cross_metric(
                     filters=all_fwd_data,
                     things_to_scan=all_fwd_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap) 
        if (all_rev_data is not None):
            cross_metrics_rev = self.cross_metric(
                     filters=all_rev_data,
                     things_to_scan=all_fwd_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap) 
        else:
            cross_metrics_rev = None
        cross_metrics = (np.maximum(cross_metrics_fwd, cross_metrics_rev)
            if (cross_metrics_rev is not None) else
            np.array(cross_metrics_fwd))
        return cross_metrics


class MaxCrossCorrAffinityMatrixFromSeqlets(
        MaxCrossMetricAffinityMatrixFromSeqlets):

    def __init__(self, pattern_comparison_settings, **kwargs):
        super(MaxCrossCorrAffinityMatrixFromSeqlets, self).__init__(
            pattern_comparison_settings=pattern_comparison_settings,
            cross_metric=CrossCorrMetricGPU(**kwargs))


class TwoTierAffinityMatrixFromSeqlets(AbstractAffinityMatrixFromSeqlets):

    def __init__(self, fast_affmat_from_seqlets,
                       nearest_neighbors_object,
                       n_neighbors,
                       affmat_from_seqlets_with_nn_pairs):
        self.fast_affmat_from_seqlets = fast_affmat_from_seqlets
        self.nearest_neighbors_object = nearest_neighbors_object
        self.n_neighbors = n_neighbors
        self.affmat_from_seqlets_with_nn_pairs =\
            affmat_from_seqlets_with_nn_pairs

    def __call__(self, seqlets):
        fast_affmat = self.fast_affmat_from_seqlets(seqlets) 
        neighbors = self.nearest_neighbors_object.fit(-fast_affmat)\
                        .kneighbors(X=-fast_affmat,
                                    n_neighbors=self.n_neighbors,
                                    return_distance=False) 
        final_affmat = self.affmat_from_seqlets_with_nn_pairs(
                         seqlet_neighbors=neighbors,
                         seqlets=seqlets)


class AffmatFromSeqletsWithNNpairs(object):

    def __init__(self, pattern_comparison_settings,
                       sim_metric_on_nn_pairs):
        self.pattern_comparison_settings = pattern_comparison_settings 
        self.sim_metric_on_nn_pairs = sim_metric_on_nn_pairs

    def __call__(self, seqlets, filter_seqlets=None,
                       seqlet_neighbors=None, return_sparse=False):
        (all_fwd_data, all_rev_data) =\
            modiscocore.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=self.pattern_comparison_settings.track_names,
                track_transformer=
                    self.pattern_comparison_settings.track_transformer)

        if (filter_seqlets is None):
            filter_seqlets = seqlets
        (filters_all_fwd_data, filters_all_rev_data) =\
            modiscocore.get_2d_data_from_patterns(
                patterns=filter_seqlets,
                track_names=self.pattern_comparison_settings.track_names,
                track_transformer=
                    self.pattern_comparison_settings.track_transformer)

        if (seqlet_neighbors is None):
            seqlet_neighbors = [list(range(len(filter_seqlets)))
                                for x in seqlets] 

        #apply the cross metric
        affmat_fwd = self.sim_metric_on_nn_pairs(
                     neighbors_of_things_to_scan=seqlet_neighbors,
                     filters=filters_all_fwd_data,
                     things_to_scan=all_fwd_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap,
                     return_sparse=return_sparse) 
        if (filters_all_rev_data is not None):
            affmat_rev = self.sim_metric_on_nn_pairs(
                 neighbors_of_things_to_scan=seqlet_neighbors,
                 filters=filters_all_rev_data,
                 things_to_scan=all_fwd_data,
                 min_overlap=self.pattern_comparison_settings.min_overlap,
                 return_sparse=return_sparse) 
        else:
            affmat_rev = None
        if (return_sparse==False):
            if (len(affmat_fwd.shape)==3):
                #dims are N x N x 2, where first entry of last idx is sim,
                # and the second entry is the alignment.
                if (affmat_rev is None):
                    affmat = affmat_fwd
                else:
                    #will return something that's N x N x 3, where the third
                    # entry in last dim is is_fwd 
                    is_fwd = (affmat_fwd[:,:,0] > affmat_rev[:,:,0])*1.0
                    affmat = np.zeros((affmat_fwd.shape[0],
                                       affmat_fwd.shape[1],3))
                    affmat[:,:,0:2] = (affmat_fwd*is_fwd[:,:,None]
                                       + affmat_rev*(1-is_fwd[:,:,None]))
                    affmat[:,:,2] = is_fwd 

            else:
                affmat = (np.maximum(affmat_fwd, affmat_rev) if
                          (affmat_rev is not None) else np.array(affmat_fwd))
        else:
            if (len(affmat_fwd[0].shape)==2):
                #dims are N x neighbs x 2, where first entry of last idx is sim,
                # and the second entry is the alignment.
                if (affmat_rev is None):
                    affmat = affmat_fwd
                else:
                    affmat = [] 
                    for fwd, rev in zip(affmat_fwd, affmat_rev):
                        is_fwd = (fwd[:,0] > rev[:,0])*1.0 
                        new_row = np.zeros((fwd.shape[0],3))
                        new_row[:,0:2] = (fwd*is_fwd[:,None] +
                                          rev*(1-is_fwd[:,None]))
                        new_row[:,2] = is_fwd 
                        affmat.append(new_row)
            else:
                affmat = ([np.maximum(x,y) for (x,y)
                           in zip(affmat_fwd, affmat_rev)]
                          if affmat_rev is not None else affmat_fwd)
                 
        return affmat  


class AbstractSimMetricOnNNpairs(object):

    def __call__(self, neighbors_of_things_to_scan,
                       filters, things_to_scan, min_overlap):
        raise NotImplementedError()


class ParallelCpuCrossMetricOnNNpairs(AbstractSimMetricOnNNpairs):

    def __init__(self, n_cores, cross_metric_single_region, verbose=True):
        #cross_metric_single_region is, for example, an instance of
        # CrossContinJaccardSingleRegion or
        # CrossContinJaccardSingleRegionWithArgmax
        self.n_cores = n_cores
        self.cross_metric_single_region = cross_metric_single_region
        self.verbose = verbose

    #min_overlap is w.r.t. the length of 'filters'
    def __call__(self, filters, things_to_scan, min_overlap,
                       neighbors_of_things_to_scan=None,
                       return_sparse=False):
        if (neighbors_of_things_to_scan is None):
            neighbors_of_things_to_scan = [list(range(len(filters)))
                                           for x in things_to_scan] 
        assert (len(neighbors_of_things_to_scan) == things_to_scan.shape[0]),\
               (len(neighbors_of_things_to_scan), things_to_scan.shape[0]) 
        assert np.max([np.max(x) for x in neighbors_of_things_to_scan])\
                < filters.shape[0]
        assert len(things_to_scan.shape)==3
        assert len(filters.shape)==3

        filter_length = filters.shape[1]
        padding_amount = int((filter_length)*(1-min_overlap))
        things_to_scan = np.pad(array=things_to_scan,
                              pad_width=((0,0),
                                         (padding_amount, padding_amount),
                                         (0,0)),
                              mode="constant")

        #if the metric has returns_pos==False, it means that the metric
        # only returns the best similarity and not the alignment that
        # gives rise to that similarity 
        if (self.cross_metric_single_region.returns_pos==False):
            if (return_sparse==False):
                to_return = np.zeros((things_to_scan.shape[0],
                                      filters.shape[0]))
        else:
            if (return_sparse==False):
                #each return value will contain both the
                # position of the alignment
                # as well as the similarity at that position; hence the
                # length of the third dimension is 2.
                # The similarity comes first, then the position
                to_return = np.zeros((things_to_scan.shape[0],
                                      filters.shape[0], 2))

        start = time.time()
        if (self.verbose):
            print("Launching nearest neighbors affmat calculation job")
            print_memory_use()
            sys.stdout.flush()

        results = Parallel(n_jobs=self.n_cores, backend="threading")(
                    (delayed(self.cross_metric_single_region)(
                        filters[neighbors_of_things_to_scan[i]],
                        things_to_scan[i])
                        for i in range(len(things_to_scan))))

        assert len(results)==len(neighbors_of_things_to_scan)
        if (self.cross_metric_single_region.returns_pos==False):
            assert all([len(x)==len(y) for x,y in
                        zip(results, neighbors_of_things_to_scan)])
        else:
            assert all([len(x[0])==len(y) for x,y in
                        zip(results, neighbors_of_things_to_scan)])

        if (self.verbose):
            print("Parallel runs completed")
            print_memory_use()
            sys.stdout.flush()

        if (return_sparse==True):
            to_return = []
            if (self.cross_metric_single_region.returns_pos==False):
                assert len(results[0].shape)==1  
                to_return = results
            else:
                assert len(results[0].shape)==2 
                for result in results:
                    #adjust the "position" to remove the effect of the padding
                    result[1] -= padding_amount 
                    to_return.append(np.transpose(result, (1,0)))
        else: 
            for (thing_to_scan_idx, (result, thing_to_scan_neighbor_indices))\
                 in enumerate(zip(results, neighbors_of_things_to_scan)):
                #adjust the "position" to remove the effect of the padding
                if (self.cross_metric_single_region.returns_pos==True):
                    result[1] -= padding_amount 
                    to_return[thing_to_scan_idx,
                              thing_to_scan_neighbor_indices] =\
                        np.transpose(result,(1,0))
                else:
                    to_return[thing_to_scan_idx,
                              thing_to_scan_neighbor_indices] = result

        gc.collect()

        end = time.time()
        if (self.verbose):
            print("Job completed in:",round(end-start,2),"s")
            print_memory_use()
            sys.stdout.flush()

        return to_return


class CrossContinJaccardSingleRegionWithArgmax(object):

    def __init__(self):
        self.returns_pos = True

    def __call__(self, filters, thing_to_scan):
        assert len(thing_to_scan.shape)==2
        assert len(filters.shape)==3
        len_output = 1+thing_to_scan.shape[0]-filters.shape[1] 
        full_crossmetric = np.zeros((filters.shape[0],len_output))
    
        for idx in range(len_output):
            snapshot = thing_to_scan[idx:idx+filters.shape[1],:]
            full_crossmetric[:,idx] =\
                (np.sum(np.minimum(np.abs(snapshot[None,:,:]),
                                   np.abs(filters[:,:,:]))*
                        (np.sign(snapshot[None,:,:])
                         *np.sign(filters[:,:,:])),axis=(1,2))/
                 np.sum(np.maximum(np.abs(snapshot[None,:,:]),
                                   np.abs(filters[:,:,:])),axis=(1,2)))
        argmax_positions = np.argmax(full_crossmetric, axis=1)
        return np.array([full_crossmetric[np.arange(len(argmax_positions)),
                                          argmax_positions],
                         argmax_positions])


class CrossContinJaccardSingleRegion(CrossContinJaccardSingleRegionWithArgmax):

    def __init__(self):
        self.returns_pos = False

    def __call__(self, filters, thing_to_scan):
        max_vals, argmax_pos =\
            super(CrossContinJaccardSingleRegion, self).__call__(
                                                       filters, thing_to_scan)
        return max_vals


class AbstractCrossMetric(object):

    def __call__(self, filters, things_to_scan, min_overlap):
        raise NotImplementedError()


class CrossCorrMetricGPU(AbstractCrossMetric):

    def __init__(self, batch_size=50, func_params_size=1000000,
                       progress_update=1000):
        self.batch_size = batch_size
        self.func_params_size = func_params_size
        self.progress_update = progress_update

    def __call__(self, filters, things_to_scan, min_overlap):
        from .. import backend as B
        return B.max_cross_corrs(
                filters=filters,
                things_to_scan=things_to_scan,
                min_overlap=min_overlap,
                batch_size=self.batch_size,
                func_params_size=self.func_params_size,
                progress_update=self.progress_update)


class CrossContinJaccardOneCoreCPU(AbstractCrossMetric):

    def __init__(self, verbose=True):
        self.verbose = verbose

    def __call__(self, filters, things_to_scan, min_overlap):
        assert len(filters.shape)==3,"Did you pass in filters of unequal len?"
        assert len(things_to_scan.shape)==3
        assert filters.shape[-1] == things_to_scan.shape[-1]

        filter_length = filters.shape[1]
        padding_amount = int((filter_length)*(1-min_overlap))
        padded_input = np.array([np.pad(array=x,
                              pad_width=((padding_amount, padding_amount),
                                         (0,0)),
                              mode="constant") for x in things_to_scan])

        len_output = 1+padded_input.shape[1]-filters.shape[1]
        full_crossmetric = np.zeros((filters.shape[0], padded_input.shape[0],
                                       len_output))
        for idx in range(len_output):
            if (self.verbose):
                print("On offset",idx,"of",len_output-1)
                sys.stdout.flush()
            snapshot = padded_input[:,idx:idx+filters.shape[1],:]
            full_crossmetric[:,:,idx] =\
                (np.sum(np.minimum(np.abs(snapshot[None,:,:,:]),
                                  np.abs(filters[:,None,:,:]))*
                       (np.sign(snapshot[None,:,:,:])
                        *np.sign(filters[:,None,:,:])),axis=(2,3))/
                 np.sum(np.maximum(np.abs(snapshot[None,:,:,:]),
                                   np.abs(filters[:,None,:,:])),axis=(2,3)))
        return np.max(full_crossmetric, axis=-1)


def jaccard_sim_func(filters, snapshot):
    return (np.sum(np.minimum(np.abs(snapshot[None,:,:,:]),
                              np.abs(filters[:,None,:,:]))*
                   (np.sign(snapshot[None,:,:,:])
                    *np.sign(filters[:,None,:,:])),axis=(2,3))/
             np.sum(np.maximum(np.abs(snapshot[None,:,:,:]),
                               np.abs(filters[:,None,:,:])),axis=(2,3)))


class CrossContinJaccardMultiCoreCPU(AbstractCrossMetric):

    def __init__(self, n_cores, verbose=True):
        self.n_cores = n_cores
        self.verbose = verbose

    def __call__(self, filters, things_to_scan, min_overlap):

        from joblib import Parallel, delayed

        assert len(filters.shape)==3,"Did you pass in filters of unequal len?"
        assert len(things_to_scan.shape)==3
        assert filters.shape[-1] == things_to_scan.shape[-1]

        filter_length = filters.shape[1]
        padding_amount = int((filter_length)*(1-min_overlap))
        padded_input = np.array([np.pad(array=x,
                              pad_width=((padding_amount, padding_amount),
                                         (0,0)),
                              mode="constant") for x in things_to_scan])

        len_output = 1+padded_input.shape[1]-filters.shape[1]
        full_crosscontinjaccards =\
            np.zeros((filters.shape[0], padded_input.shape[0], len_output))

        start = time.time()
        if len(filters) >= 2000: 
            for idx in range(len_output):
                if (self.verbose):
                    print("On offset",idx,"of",len_output-1)
                    sys.stdout.flush()
                snapshot = padded_input[:,idx:idx+filters.shape[1],:]
                assert snapshot.shape[1]==filters.shape[1],\
                    str(snapshape.shape)+" "+filters.shape
                subsnap_size = int(np.ceil(snapshot.shape[0]/self.n_cores))
                sys.stdout.flush()
                subsnaps = [snapshot[(i*subsnap_size):(min((i+1)*subsnap_size,
                                                         snapshot.shape[0]))]
                            for i in range(self.n_cores)]
                full_crosscontinjaccards[:,:,idx] =\
                    np.concatenate(
                     Parallel(n_jobs=self.n_cores)(delayed(jaccard_sim_func)
                              (filters, subsnap) for subsnap in subsnaps),axis=1)
        else:
            #parallelize by index
            job_arguments = []
            for idx in range(0,len_output):
                snapshot = padded_input[:,idx:idx+filters.shape[1],:]
                assert snapshot.shape[1]==filters.shape[1],\
                    str(snapshot.shape)+" "+filters.shape
                job_arguments.append((filters, snapshot))

            to_concat = (Parallel(n_jobs=self.n_cores)
                           (delayed(jaccard_sim_func)(job_args[0], job_args[1])
                            for job_args in job_arguments))
            full_crosscontinjaccards[:,:,:] =\
                    np.concatenate([x[:,:,None] for x in to_concat],axis=2)

        end = time.time()
        if (self.verbose):
            print("Cross contin jaccard time taken:",round(end-start,2),"s")

        return np.max(full_crosscontinjaccards, axis=-1)


class FilterSparseRows(object):

    def __init__(self, affmat_transformer,
                       min_rows_before_applying_filtering,
                       min_edges_per_row, verbose=True):
        self.affmat_transformer = affmat_transformer
        self.min_rows_before_applying_filtering =\
             min_rows_before_applying_filtering
        self.min_edges_per_row = min_edges_per_row
        self.verbose = verbose

    def __call__(self, affinity_mat):
        if (len(affinity_mat) < self.min_rows_before_applying_filtering):
            if (self.verbose):
                print("Fewer than "
                 +str(self.min_rows_before_applying_filtering)+" rows so"
                 +" not applying filtering")
                sys.stdout.flush()
            return (np.ones(len(affinity_mat)) > 0.0) #keep all rows

        affinity_mat = self.affmat_transformer(affinity_mat) 
        per_node_neighbours = np.sum(affinity_mat > 0, axis=1) 
        passing_nodes = per_node_neighbours >= self.min_edges_per_row
        if (self.verbose):
            print(str(np.sum(passing_nodes))+" passing out of "
                  +str(len(passing_nodes)))
            sys.stdout.flush() 
        return passing_nodes


class FilterMaskFromCorrelation(object):

    def __init__(self, correlation_threshold, verbose=True):
        self.correlation_threshold = correlation_threshold
        self.verbose = verbose

    def __call__(self, main_affmat, other_affmat):
        correlations = []
        neg_log_pvals = []
        for main_affmat_row, other_affmat_row\
            in zip(main_affmat, other_affmat):
            #compare correlation on the nonzero rows
            to_compare_mask = np.abs(main_affmat_row) > 0
            corr = scipy.stats.spearmanr(
                    main_affmat_row[to_compare_mask],
                    other_affmat_row[to_compare_mask])
            correlations.append(corr.correlation)
            neg_log_pvals.append(-np.log(corr.pvalue)) 
        correlations = np.array(correlations)
        neg_log_pvals = np.array(neg_log_pvals)
        mask_to_return = (correlations > self.correlation_threshold)
        if (self.verbose):
            print("Filtered down to "+str(np.sum(mask_to_return))
                  +" of "+str(len(mask_to_return)))
            sys.stdout.flush()
        return mask_to_return


