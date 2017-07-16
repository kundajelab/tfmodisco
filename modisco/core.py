from collections import OrderedDict


FwdAndRevSnippet = namedtuple('FwdAndRevSnippet', ['fwd', 'rev'])


class DataTrack(object):

    """
    First dimension of fwd_tracks and rev_tracks should be the example,
    second dimension should be the length
    """
    def __init__(name, fwd_tracks, rev_tracks):
        self.name = name
        assert len(fwd_tracks)==len(rev_tracks)
        assert len(fwd_tracks[0]==len(rev_tracks[0]))
        self.fwd_tracks = fwd_tracks
        self.rev_tracks = rev_tracks

    def __len__(self):
        return len(self.fwd_tracks)

    def track_length(self):
        return len(self.fwd_tracks[0])

    def get_fwd_and_rev_snippets(self, example_idx, start_idx, end_idx):
        return FwdAndRevSnippet(
                fwd=self.fwd_tracks[example_idx, start_idx:end_idx],
                rev=self.rev_tracks[example_idx, start_idx:end_idx])


class TrackSet(object):

    def __init__(self):
        self.track_name_to_data_track = OrderedDict()

    def add_track(self, data_track):
        assert type(data_track).__name__=="DataTrack"
        self.track_name_to_data_track[data_track.name] = data_track

    def create_seqlet(self, track_names, example_idx, start_idx, end_idx):
        pass

    def augment_seqlet(self, seqlet, track_names):
        pass


class Seqlet(object):

    def __init__(self, example_idx, start_idx, end_idx):
        self.example_idx = example_idx
        self.start_idx = start_idx
        self.end_idx = end_idx 
        self.track_name_to_snippet = OrderedDict()

    def add_snippet(self, data_track):
        self.track_name_to_snippet[data_track.name] =\
            data_track.get_fwd_and_rev_snippets(
                example_idx=self.example_idx,
                start_idx=self.start_idx,
                end_idx=self.end_idx)


#need to think more about design of this
class AggregatedSeqlet(object):

    def __init__(self, seqlets_and_alignments):
        #do stuff
        pass
