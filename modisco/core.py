from collections import OrderedDict
import numpy as np


class Snippet(self):

    def __init__(self, fwd, rev):
        assert len(fwd)==len(rev)
        self.fwd = fwd
        self.rev = rev

    def __len__(self):
        return len(self.fwd)

    def reverse(self):
        return Snippet(fwd=self.rev, rev=self.fwd)


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

    def get_snippet(self, example_idx, start_idx, end_idx, reverse):
        snippet = Snippet(
                fwd=self.fwd_tracks[example_idx, start_idx:end_idx],
                rev=self.rev_tracks[example_idx, start_idx:end_idx])
        if (reverse):
            snippet = snippet.reverse
        return snippet


class TrackSet(object):

    def __init__(self):
        self.track_name_to_data_track = OrderedDict()

    def add_track(self, data_track):
        assert type(data_track).__name__=="DataTrack"
        self.track_name_to_data_track[data_track.name] = data_track

    def create_seqlet(self, track_names, example_idx, start_idx, end_idx):
        seqlet = Seqlet(example_idx=example_idx,
                        start_idx=start_idx, end_idx=end_idx)
        self.augment_seqlet(seqlet=seqlet, track_names=track_names) 

    def augment_seqlet(self, seqlet, track_names):
        for track_name in track_names:
            seqlet.add_snippet_from_data_track(
                data_track=self.track_name_to_data_track[track_name])


class Seqlet(object):

    def __init__(self, example_idx, start_idx, end_idx, reverse):
        self.example_idx = example_idx
        self.start_idx = start_idx
        self.end_idx = end_idx 
        self.reverse = reverse
        self.track_name_to_snippet = OrderedDict()

    def add_snippet_from_data_track(self, data_track): 
        snippet =  data_track.get_snippet(example_idx=self.example_idx,
                                          start_idx=self.start_idx,
                                          end_idx=self.end_idx,
                                          reverse=self.reverse)
        self.add_snippet(data_track_name=data_track.name, snippet=snippet)
        
    def add_snippet(self, data_track_name, snippet): 
        self.track_name_to_snippet[data_track_name] = snippet 

    def __getitem__(self, key):
        return self.track_name_to_snippet[key]


class SeqletAndAlignment(object):

    def __init__(self, seqlet, alnmt):
        self.seqlet = seqlet
        self.alnmt = alnmt


#need to think more about design of this
class AggregatedSeqlet(object):

    def __init__(self, seqlets_and_alnmts):
        self.seqlets_and_alnmts = seqlets_and_alnmts
        self.length = max([x.alnmt + len(x.seqlet)
                           for x in self.seqlets_and_alnmts]) 
        self._compute_aggregation() 

    def _compute_aggregation(self):
        self._initialize_track_name_to_aggregation()
        self.per_position_counts = np.zeros((self.length,))
        for seqlet_and_alnmt in self.seqlets_and_alnmts:
            self._add_seqlet_with_valid_alnmt(seqlet_and_alnmt)

    def _initialize_track_name_to_aggregation(self): 
        sample_seqlet = self.seqlets_and_alnmts[0] 
        self.track_name_to_agg = OrderedDict() 
        self.track_name_to_agg_reverse = OrderedDict() 
        for track_name in self.track_name_to_snippet:
            track_shape = tuple([self.length]
                           +list(sample_seqlet[track_name].shape[1:]))
            self.track_name_to_agg[track_name] =\
                np.zeros(track_shape).astype("float") 
            self.track_name_to_agg_reverse[track_name] =\
                np.zeros(track_shape).astype("float") 

    def _pad_before(self, num_zeros):
        assert num_zeros > 0
        self.length += num_zeros
        for seqlet_and_alnmt in self.seqlets_and_alnmts:
            seqlet_and_alnmt.alnmt += num_zeros
        self.per_position_counts =\
            np.concatenate([np.zeros((num_zeros,)),
                            self.per_position_counts],axis=0) 
        for track_name in self.track_name_to_snippet:
            track = self.track_name_to_agg[track_name]
            rev_track = self.track_name_to_agg_reverse[track_name]
            padding_shape = tuple([num_zeros]+list(track.shape[1:])) 
            extended_track = np.concatenate(
                [np.zeros(padding_shape), track], axis=0)
            extended_rev_track = np.concatenate(
                [rev_track, np.zeros(padding_shape)], axis=0)
            self.track_name_to_agg[track_name] = extended_track
            self.track_name_to_agg_reverse[track_name] = extended_rev_track

    def _pad_after(self, num_zeros):
        assert num_zeros > 0
        self.length += num_zeros 
        self.per_position_counts =\
            np.concatenate([self.per_position_counts,
                            np.zeros((num_zeros,))],axis=0) 
        for track_name in self.track_name_to_snippet:
            track = self.track_name_to_agg[track_name]
            rev_track = self.track_name_to_agg_reverse[track_name]
            padding_shape = tuple([num_zeros]+list(track.shape[1:])) 
            extended_track = np.concatenate(
                [track, np.zeros(padding_shape)], axis=0)
            extended_rev_track = np.concatenate(
                [np.zeros(padding_shape),rev_track], axis=0)
            self.track_name_to_agg[track_name] = extended_track
            self.track_name_to_agg_reverse[track_name] = extended_rev_track

    def add_seqlet(self, seqlet_and_alnmt):
        self.seqlets_and_alnmts.append(seqlet_and_alnmt)
        alnmt = seqlet_and_alnmt.alnmt
        if alnmt < 0:
           self._pad_before(num_zeros=abs(alnmt)) 
        end_coor_of_seqlet = alnmt + len(seqlet_and_alnmt.seqlet)
        if (end_coor_of_seqlet > self.length):
            self._pad_after(num_zeros=(end_coor_of_seqlet - self.length))
        self._add_seqlet_with_valid_alnmt(self, seqlet_and_alnmt)

    def _add_seqlet_with_valid_alnmt(self, seqlet_and_alnmt):
        alnmt = seqlet_and_alnmt.alnmt
        seqlet = seqlet_and_alnmt.seqlet
        slice_obj = slice(alnmt, alnmt+len(seqlet))
        rev_slice_obj = slice(self.length-(alnmt+len(seqlet)),
                              self.length-alnmt)
        self.per_position_counts[slice_obj] += 1.0 
        for track_name in self.track_name_to_agg:
            self.track_name_to_agg[track_name][slice_obj] +=\
                seqlet[track_name].fwd 
            self.track_name_to_agg_reverse[track_name][rev_slice_obj] +=\
                seqlet[track_name].rev

    def __len__(self):
        return self.length



