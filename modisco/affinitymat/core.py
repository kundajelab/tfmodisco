from __future__ import division, print_function, absolute_import
from .. import backend as B
import numpy as np
from .. import core as modiscocore
from . import transformers
import sys


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


class PatternCrossCorrSettings(object):
    def __init__(self, track_names, track_transformer, min_overlap):
        assert hasattr(track_names, '__iter__')
        self.track_names = track_names
        self.track_transformer = track_transformer
        self.min_overlap = min_overlap


class AbstractAffinityMatrixFromSeqlets(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class MaxCrossCorrAffinityMatrixFromSeqlets(AbstractAffinityMatrixFromSeqlets):

    def __init__(self, pattern_crosscorr_settings,
                       batch_size=50,
                       func_params_size=1000000,
                       progress_update=1000):
        self.pattern_crosscorr_settings = pattern_crosscorr_settings
        self.batch_size = batch_size
        self.func_params_size = func_params_size
        self.progress_update = progress_update

    def __call__(self, seqlets):
        (all_fwd_data, all_rev_data) =\
            modiscocore.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=self.pattern_crosscorr_settings.track_names,
                track_transformer=
                    self.pattern_crosscorr_settings.track_transformer)

        #do cross correlations
        cross_corrs_fwd = B.max_cross_corrs(
                     filters=all_fwd_data,
                     things_to_scan=all_fwd_data,
                     min_overlap=self.pattern_crosscorr_settings.min_overlap,
                     batch_size=self.batch_size,
                     func_params_size=self.func_params_size,
                     progress_update=self.progress_update) 
        cross_corrs_rev = B.max_cross_corrs(
                     filters=all_rev_data,
                     things_to_scan=all_fwd_data,
                     min_overlap=self.pattern_crosscorr_settings.min_overlap,
                     batch_size=self.batch_size,
                     func_params_size=self.func_params_size,
                     progress_update=self.progress_update) 
        cross_corrs = np.maximum(cross_corrs_fwd, cross_corrs_rev)
        return cross_corrs


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
        passing_nodes = per_node_neighbours > self.min_edges_per_row
        if (self.verbose):
            print(str(np.sum(passing_nodes))+" passing out of "
                  +str(len(passing_nodes)))
            sys.stdout.flush() 
        return passing_nodes
