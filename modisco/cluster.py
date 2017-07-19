

class AbstractNormalizer(self):

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
        assert len(inp.shape)==2
        return (inp - np.mean(inp, axis=1)[:,None])


class MagnitudeNormalizer(AbstractNormalizer):

    def normalize(self, inp):
        assert len(inp.shape)==2
        return (inp / (np.linalg.norm(inp,axis=1)+0.0000001))


class DistanceMatrixFromSeqlets(object):

    def __init__(self, track_names, normalizer):
        assert hasattr(track_names, '__iter__')
        assert hasattr(normalizers, '__iter__')
        self.track_names = track_names
        self.normalizer = normalizer

    def get_distance_matrix(self, seqlets):
        (all_fwd_data, all_rev_data) =\
            get_2d_data_from_seqlets(seqlets=seqlets,
                                     track_names=self.track_names,
                                     normalizer=self.normalizer) 


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
