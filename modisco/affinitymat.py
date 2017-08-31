from . import backend as B
import numpy as np
from . import core

class AbstractNormalizer(object):

    def __call__(self, inp):
        """
            inp: 2d array
        """
        raise NotImplementedError() 

    def chain(self, other_normalizer):        
        return AdhocNormalizer(
                func=(lambda x: other_normalizer(
                                self(x))))


class AdhocNormalizer(AbstractNormalizer):
    def __init__(self, func):
        self.func = func

    def __call__(self, inp):
        return self.func(inp)


class MeanNormalizer(AbstractNormalizer):

    def __call__(self, inp):
        return inp - np.mean(inp)


class MagnitudeNormalizer(AbstractNormalizer):

    def __call__(self, inp):
        return (inp / (np.linalg.norm(inp.ravel())+0.0000001))


class PatternCrossCorrSettings(object):
    def __init__(self, track_names, normalizer, min_overlap):
        assert hasattr(track_names, '__iter__')
        self.track_names = track_names
        self.normalizer = normalizer
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
            core.get_2d_data_from_seqlets(
                seqlets=seqlets,
                track_names=self.pattern_crosscorr_settings.track_names,
                normalizer=self.pattern_crosscorr_settings.normalizer)

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
