from __future__ import division, print_function, absolute_import
from .. import backend as B
import numpy as np
from .. import util as modiscoutil
from .. import core as modiscocore
from . import transformers
import sys
import time
import itertools
from sklearn.neighbors import NearestNeighbors 
from joblib import Parallel, delayed


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


class AbstractSeqletsToOnedEmbedder(object):

    def __call__(seqlets):
        raise NotImplementedError()


class GappedKmerEmbedder(AbstractSeqletsToOnedEmbedder):
    
    def __init__(self, alphabet_size,
                       kmer_len,
                       num_gaps,
                       num_mismatches,
                       toscore_track_names_and_signs,
                       normalizer,
                       num_filters_to_retain=None,
                       onehot_track_name=None,
                       batch_size=50,
                       progress_update=None):
        self.alphabet_size = alphabet_size
        self.kmer_len = kmer_len
        self.num_gaps = num_gaps
        self.num_mismatches = num_mismatches
        self.num_filters_to_retain = num_filters_to_retain
        self.filters, self.biases = self.prepare_gapped_kmer_filters()
        self.onehot_track_name = onehot_track_name
        self.toscore_track_names_and_signs = toscore_track_names_and_signs
        assert len(toscore_track_names_and_signs) >= 0,\
            "toscore_track_names_and_signs length is 0"
        self.normalizer = normalizer
        self.batch_size = batch_size
        self.progress_update = progress_update
        self.require_onehot_match = (True if self.onehot_track_name
                                     is not None else False)
        self.gapped_kmer_embedding_func =\
            B.get_gapped_kmer_embedding_func(
                    filters=self.filters,
                    biases=self.biases,
                    require_onehot_match=self.require_onehot_match)

    def prepare_gapped_kmer_filters(self):
        nonzero_position_combos = list(itertools.combinations(
                            iterable=range(self.kmer_len),
                            r=(self.kmer_len-self.num_gaps)))
        letter_permutations = list(itertools.product(
                                *[list(range(self.alphabet_size)) for x in
                                  range(self.kmer_len-self.num_gaps)]))
        filters = []
        biases = []
        unique_nonzero_positions = set()
        for nonzero_positions in nonzero_position_combos:
            string_representation = [" " for x in range(self.kmer_len)]
            for nonzero_position in nonzero_positions:
                string_representation[nonzero_position] = "X"
            nonzero_positions_string =\
                ("".join(string_representation)).lstrip().rstrip()
            if (nonzero_positions_string not in unique_nonzero_positions):
                unique_nonzero_positions.add(nonzero_positions_string) 
                for letter_permutation in letter_permutations:
                    assert len(nonzero_positions)==len(letter_permutation)
                    the_filter = np.zeros((self.kmer_len, self.alphabet_size)) 
                    for nonzero_position, letter\
                        in zip(nonzero_positions, letter_permutation):
                        the_filter[nonzero_position, letter] = 1 
                    filters.append(the_filter)
                    biases.append(-(len(nonzero_positions)-1
                                    -self.num_mismatches))
        return np.array(filters), np.array(biases)

    def __call__(self, seqlets):
        print("Computing embeddings")
        sys.stdout.flush()
        if (self.require_onehot_match):
            onehot_track_fwd, onehot_track_rev =\
                modiscocore.get_2d_data_from_patterns(
                    patterns=seqlets,
                    track_names=[self.onehot_track_name],
                    track_transformer=None)

        data_to_embed_fwd = np.zeros((len(seqlets),
                                     len(seqlets[0]), self.alphabet_size))\
                                     .astype("float32")
        data_to_embed_rev = np.zeros((len(seqlets),
                                     len(seqlets[0]), self.alphabet_size))\
                                     .astype("float32")
        for (track_name, sign) in self.toscore_track_names_and_signs:
            fwd_data, rev_data = modiscocore.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=[track_name], track_transformer=None)  
            data_to_embed_fwd += fwd_data*sign
            data_to_embed_rev += rev_data*sign
        data_to_embed_fwd = np.array([self.normalizer(x) for x in
                                      data_to_embed_fwd])
        data_to_embed_rev = np.array([self.normalizer(x) for x in
                                      data_to_embed_rev])
        common_args = {'batch_size': self.batch_size,
                       'progress_update': self.progress_update}
        if (self.require_onehot_match):
            embedding_fwd = self.gapped_kmer_embedding_func(
                                  onehot=onehot_track_fwd,
                                  to_embed=data_to_embed_fwd,
                                  **common_args)
            embedding_rev = self.gapped_kmer_embedding_func(
                                  onehot=onehot_track_rev,
                                  to_embed=data_to_embed_rev,
                                  **common_args)
        else:
            embedding_fwd = self.gapped_kmer_embedding_func(
                                  to_embed=data_to_embed_fwd,
                                  **common_args)
            embedding_rev = self.gapped_kmer_embedding_func(
                                  to_embed=data_to_embed_rev,
                                  **common_args)
        if (self.num_filters_to_retain is not None):
            all_embeddings = np.concatenate(
                             [embedding_fwd, embedding_rev], axis=0)
            embeddings_denominators =\
                np.sum(np.abs(all_embeddings) > 0, axis=0).astype("float")
            embeddings_denominators += 10.0
            embeddings_mean_impact =\
                (np.sum(np.abs(all_embeddings), axis=0)/
                 embeddings_denominators)
            top_embedding_indices = [
                x[0] for x in sorted(enumerate(embeddings_mean_impact),
                key=lambda x: -x[1])][:self.num_filters_to_retain]
            embedding_fwd = embedding_fwd[:,top_embedding_indices]
            embedding_rev = embedding_rev[:,top_embedding_indices]
        return embedding_fwd, embedding_rev


class AbstractAffinityMatrixFromOneD(object):

    def __call__(self, vecs1, vecs2):
        raise NotImplementedError()


class NumpyCosineSimilarity(AbstractAffinityMatrixFromOneD):

    def __call__(self, vecs1, vecs2):
        print("Computing similarity")
        sys.stdout.flush()
        normed_vecs1 = vecs1/np.linalg.norm(vecs1, axis=1)[:,None] 
        normed_vecs2 = vecs2/np.linalg.norm(vecs2, axis=1)[:,None] 
        return np.dot(normed_vecs1,normed_vecs2.T)


class AffmatFromEmbeddings(AbstractAffinityMatrixFromSeqlets):

    def __init__(self, seqlets_to_1d_embedder, affinity_mat_from_1d):
        self.seqlets_to_1d_embedder = seqlets_to_1d_embedder
        self.affinity_mat_from_1d = affinity_mat_from_1d 

    def __call__(self, seqlets):
        embedding_fwd, embedding_rev = self.seqlets_to_1d_embedder(seqlets)
        affinity_mat_fwd = self.affinity_mat_from_1d(
                            vecs1=embedding_fwd, vecs2=embedding_fwd)  
        affinity_mat_rev = self.affinity_mat_from_1d(
                            vecs1=embedding_fwd, vecs2=embedding_rev)
        return np.maximum(affinity_mat_fwd, affinity_mat_rev) 


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
        cross_metrics_rev = self.cross_metric(
                     filters=all_rev_data,
                     things_to_scan=all_fwd_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap) 
        cross_metrics = np.maximum(cross_metrics_fwd, cross_metrics_rev)
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
        neighbors = self.nearest_neighbors_object.fit(fast_affmat)\
                        .kneighbors(X=fast_affmat,
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

    def __call__(self, seqlet_neighbors, seqlets):
        (all_fwd_data, all_rev_data) =\
            modiscocore.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=self.pattern_comparison_settings.track_names,
                track_transformer=
                    self.pattern_comparison_settings.track_transformer)
        #apply the cross metric
        affmat_fwd = self.sim_metric_on_nn_pairs(
                     neighbors_of_things_to_scan=seqlet_neighbors,
                     filters=all_fwd_data,
                     things_to_scan=all_fwd_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap) 
        affmat_rev = self.sim_metric_on_nn_pairs(
                     neighbors_of_things_to_scan=seqlet_neighbors,
                     filters=all_rev_data,
                     things_to_scan=all_fwd_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap) 
        affmat = np.maximum(affmat_fwd, affmat_rev)
        return affmat
        


class AbstractSimMetricOnNNpairs(object):

    def __call__(self, neighbors_of_things_to_scan,
                       filters, things_to_scan, min_overlap):
        raise NotImplementedError()


class ParallelCpuCrossMetricOnNNpairs(AbstractSimMetricOnNNpairs):

    def __init__(self, n_cores, cross_metric_single_region, verbose=True):
        self.n_cores = n_cores
        self.cross_metric_single_region = cross_metric_single_region
        self.verbose = verbose

    def __call__(self, neighbors_of_things_to_scan,
                       filters, things_to_scan, min_overlap):
        assert neighbors_of_things_to_scan.shape[0]==things_to_scan.shape[0]
        assert np.max(neighbors_of_things_to_scan) < filters.shape[0]
        assert len(things_to_scan.shape)==3
        assert len(filters.shape)==3
       
        filter_length = filters.shape[1]
        padding_amount = int((filter_length)*(1-min_overlap))
        things_to_scan = np.pad(array=things_to_scan,
                              pad_width=((0,0),
                                         (padding_amount, padding_amount),
                                         (0,0)),
                              mode="constant")
 
        to_return = np.zeros((things_to_scan.shape[0], filters.shape[0]))
        job_arguments = []

        for neighbors_of_thing_to_scan, thing_to_scan\
            in zip(neighbors_of_things_to_scan, things_to_scan): 
            args = (filters[neighbors_of_thing_to_scan], thing_to_scan) 
            job_arguments.append(args)
        
         
        start = time.time()
        if (self.verbose):
            print("launching nearest neighbors affmat calculation job")
            sys.stdout.flush()

        results = (Parallel(n_jobs=self.n_cores)                          
                   (delayed(self.cross_metric_single_region)
                           (job_args[0], job_args[1])
                    for job_args in job_arguments))

        for (thing_to_scan_idx, (result, thing_to_scan_neighbor_indices))\
             in enumerate(zip(results, neighbors_of_things_to_scan)):
            to_return[thing_to_scan_idx,
                      thing_to_scan_neighbor_indices] = result

        end = time.time()
        if (self.verbose):
            print("Job completed in:",round(end-start,2),"s")
            sys.stdout.flush()

        return to_return


class AbstractCrossMetricSingleRegion(object):

    def __call__(self, filters, thing_to_scan):
        raise NotImplementedError()


class CrossContinJaccardSingleRegion(AbstractCrossMetricSingleRegion):

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
        return np.max(full_crossmetric, axis=-1) 


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


class CrossContinJaccardMultiCoreCPU2(AbstractCrossMetric):

    def __init__(self, n_cores, verbose=True):
        self.n_cores = n_cores
        self.verbose = verbose

    def __call__(self, filters, things_to_scan, min_overlap):

        from joblib import Parallel, delayed
        if (self.verbose):
            print("Begin cross contin jaccard")

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


class CrossContinJaccardGPU(AbstractCrossMetric):

    def __init__(self, verbose=True, batch_size=100):
        self.verbose = verbose
        self.batch_size = batch_size

    def __call__(self, filters, things_to_scan, min_overlap):
        assert len(filters.shape)==3,"Did you pass in filters of unequal len?"
        assert len(things_to_scan.shape)==3
        assert filters.shape[-1] == things_to_scan.shape[-1]
        jaccard_sim_func = B.get_jaccard_sim_func(filters)

        filter_length = filters.shape[1]
        padding_amount = int((filter_length)*(1-min_overlap))
        padded_input = np.array([np.pad(array=x,
                              pad_width=((padding_amount, padding_amount),
                                         (0,0)),
                              mode="constant") for x in things_to_scan])

        len_output = 1+padded_input.shape[1]-filters.shape[1]
        full_crosscontinjaccard =\
            np.zeros((filters.shape[0], padded_input.shape[0], len_output))

        for idx in range(len_output):
            if (self.verbose):
                print("On offset",idx,"of",len_output-1)
                sys.stdout.flush()
            snapshot = padded_input[:,idx:idx+filters.shape[1],:]
            batch_start = 0
            while (batch_start < snapshot.shape[0]):
                batch_end = min(batch_start+self.batch_size, snapshot.shape[0])
                batch = snapshot[batch_start:batch_end]
                sys.stdout.flush()
                full_crosscontinjaccard[:,batch_start:batch_end,idx] =\
                    jaccard_sim_func(batch) 
                sys.stdout.flush()
                batch_start += self.batch_size
        return np.max(full_crosscontinjaccard, axis=-1) 


class AbstractGetFilteredRowsMask(object):

    def __call__(self, affinity_mat):
        raise NotImplementedError()


class FilterSparseRows(AbstractGetFilteredRowsMask):

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
