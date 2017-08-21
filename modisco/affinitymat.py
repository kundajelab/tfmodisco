from . import backend as B
import numpy as np

class AbstractNormalizer(object):

    def normalize(self, inp):
        """
            inp: 2d array
        """
        raise NotImplementedError() 

    def chain(self, other_normalizer):        
        return AdhocNormalizer(
                func=(lambda x: other_normalizer.normalize(
                                self.normalize(x))))


class AdhocNormalizer(AbstractNormalizer):
    def __init__(self, func):
        self.func = func

    def normalize(self, inp):
        return self.func(inp)


class MeanNormalizer(AbstractNormalizer):

    def normalize(self, inp):
        return inp - np.mean(inp)


class MagnitudeNormalizer(AbstractNormalizer):

    def normalize(self, inp):
        return (inp / (np.linalg.norm(inp.ravel())+0.0000001))


class MaxCrossCorrAffinityMatrixFromSeqlets(object):

    def __init__(self, track_names, normalizer,
                       min_overlap, batch_size=50,
                       func_params_size=1000000,
                       progress_update=1000):
        assert hasattr(track_names, '__iter__')
        assert hasattr(normalizers, '__iter__')
        self.track_names = track_names
        self.normalizer = normalizer
        self.min_overlap = min_overlap
        self.batch_size = batch_size
        self.func_params_size = func_params_size
        self.progress_update = progress_update

    def get_affinity_matrix(self, seqlets):
        (all_fwd_data, all_rev_data) =\
            get_2d_data_from_seqlets(seqlets=seqlets,
                                     track_names=self.track_names,
                                     normalizer=self.normalizer) 
        #do cross correlations
        cross_corrs_fwd = B.max_cross_corrs(
                             filters=all_fwd_data,
                             things_to_scan=all_fwd_data,
                             min_overlap=self.min_overlap,
                             batch_size=self.batch_size,
                             func_params_size=self.func_params_size,
                             progress_update=self.progress_update) 
        cross_corrs_rev = B.max_cross_corrs(
                             filters=all_rev_data,
                             things_to_scan=all_fwd_data,
                             min_overlap=self.min_overlap,
                             batch_size=self.batch_size,
                             func_params_size=self.func_params_size,
                             progress_update=self.progress_update) 
        cross_corrs = np.maximum(cross_corrs_fwd, cross_corrs_rev)
        return cross_corrs


def get_2d_data_from_seqlets(seqlets, track_names, normalizer):
    all_fwd_data = []
    all_rev_data = []
    for seqlet in seqlets:
        snippets = [seqlet[track_name]
                     for track_name in track_names] 
        fwd = np.concatenate([normalizer.normalize(
                 np.reshape(snippet.fwd, (len(snippet.fwd), -1)))
                for snippet in snippets], axis=1)
        rev = np.concatenate([normalizer.normalize(
                np.reshape(snippet.rev, (len(snippet.rev), -1)))
                for snippet in snippets], axis=1)
        all_fwd_data.append(fwd)
        all_rev_data.append(rev)
    return (np.concatenate(all_fwd_data, axis=0),
            np.concatenate(all_rev_data, axis=0))
