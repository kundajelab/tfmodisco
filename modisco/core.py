from collections import OrderedDict
from collections import namedtuple
import numpy as np


class Snippet(object):

    def __init__(self, fwd, rev, has_pos_axis):
        assert len(fwd)==len(rev)
        self.fwd = fwd
        self.rev = rev
        self.has_pos_axis = has_pos_axis

    def __len__(self):
        return len(self.fwd)

    def revcomp(self):
        return Snippet(fwd=self.rev, rev=self.fwd,
                       has_pos_axis=self.has_pos_axis)


class DataTrack(object):

    """
    First dimension of fwd_tracks and rev_tracks should be the example,
    second dimension should be the position (if applicable)
    """
    def __init__(self, name, fwd_tracks, rev_tracks, has_pos_axis):
        self.name = name
        assert len(fwd_tracks)==len(rev_tracks)
        assert len(fwd_tracks[0]==len(rev_tracks[0]))
        self.fwd_tracks = fwd_tracks
        self.rev_tracks = rev_tracks
        self.has_pos_axis = has_pos_axis

    def __len__(self):
        return len(self.fwd_tracks)

    def track_length(self):
        return len(self.fwd_tracks[0])

    def get_snippet(self, coor):
        if (self.has_pos_axis==False):
            snippet = Snippet(
                    fwd=self.fwd_tracks[coor.example_idx],
                    rev=self.rev_tracks[coor.example_idx],
                    has_pos_axis=self.has_pos_axis)
        else:
            snippet = Snippet(
                    fwd=self.fwd_tracks[coor.example_idx, coor.start:coor.end],
                    rev=self.rev_tracks[
                         coor.example_idx,
                         (self.track_length()-coor.end):
                         (self.track_length()-coor.start)],
                    has_pos_axis=self.has_pos_axis)
        if (coor.revcomp):
            snippet = snippet.revcomp()
        return snippet


class TrackSet(object):

    def __init__(self, data_tracks=[]):
        self.track_name_to_data_track = OrderedDict()
        for data_track in data_tracks:
            self.add_track(data_track)

    def add_track(self, data_track):
        assert type(data_track).__name__=="DataTrack"
        if len(self.track_name_to_data_track)==0:
            self.num_items = len(data_track) 
        else:
            assert len(data_track)==self.num_items,\
                    ("first track had "+str(self.num_items)+" but "
                     "data track has "+str(len(data_track))+" items")
        self.track_name_to_data_track[data_track.name] = data_track
        return self

    def create_seqlets(self, coords, track_names=None):
        if (track_names is None):
            track_names=self.track_name_to_data_track.keys()
        seqlets = []
        for coor in coords:
            seqlet = Seqlet(coor=coor)
            self.augment_seqlet(seqlet=seqlet, track_names=track_names) 
            seqlets.append(seqlet)
        return seqlets

    def augment_seqlet(self, seqlet, track_names):
        for track_name in track_names:
            seqlet.add_snippet_from_data_track(
                data_track=self.track_name_to_data_track[track_name])
        return seqlet


SeqletCoordinates = namedtuple("SeqletCoords",
                               ['example_idx', 'start', 'end', 'revcomp'])


class SeqletCoordinates(object):

    def __init__(self, example_idx, start, end, revcomp):
        self.example_idx = example_idx
        self.start = start
        self.end = end
        self.revcomp = revcomp

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return ("example:"+str(self.example_idx)
                +",loc:"+str(self.start)+",end:"+str(self.end)
                +",rc:"+str(self.revcomp))


class Seqlet(object):

    def __init__(self, coor):
        self.coor = coor
        self.track_name_to_snippet = OrderedDict()

    def add_snippet_from_data_track(self, data_track): 
        snippet =  data_track.get_snippet(coor=self.coor)
        self.add_snippet(data_track_name=data_track.name, snippet=snippet)
        
    def add_snippet(self, data_track_name, snippet):
        if (snippet.has_pos_axis):
            assert len(snippet)==len(self),\
                   ("tried to add snippet with pos axis of len "
                    +str(len(snippet))+" but snippet coords have "
                    +"len "+str(self.coor))
        self.track_name_to_snippet[data_track_name] = snippet 

    def __len__(self):
        return len(self.coor)

    def __getitem__(self, key):
        return self.track_name_to_snippet[key]


class SeqletAndAlignment(object):

    def __init__(self, seqlet, alnmt):
        self.seqlet = seqlet
        #alnmt is the position of the beginning of seqlet
        #in the aggregated seqlet
        self.alnmt = alnmt 


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
        self.track_name_to_agg_revcomp = OrderedDict() 
        for track_name in self.track_name_to_snippet:
            track_shape = tuple([self.length]
                           +list(sample_seqlet[track_name].shape[1:]))
            self.track_name_to_agg[track_name] =\
                np.zeros(track_shape).astype("float") 
            self.track_name_to_agg_revcomp[track_name] =\
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
            if (track.has_pos_axis):
                rev_track = self.track_name_to_agg_revcomp[track_name]
                padding_shape = tuple([num_zeros]+list(track.shape[1:])) 
                extended_track = np.concatenate(
                    [np.zeros(padding_shape), track], axis=0)
                extended_rev_track = np.concatenate(
                    [rev_track, np.zeros(padding_shape)], axis=0)
                self.track_name_to_agg[track_name] = extended_track
                self.track_name_to_agg_revcomp[track_name] = extended_rev_track

    def _pad_after(self, num_zeros):
        assert num_zeros > 0
        self.length += num_zeros 
        self.per_position_counts =\
            np.concatenate([self.per_position_counts,
                            np.zeros((num_zeros,))],axis=0) 
        for track_name in self.track_name_to_snippet:
            track = self.track_name_to_agg[track_name]
            if (track.has_pos_axis):
                rev_track = self.track_name_to_agg_revcomp[track_name]
                padding_shape = tuple([num_zeros]+list(track.shape[1:])) 
                extended_track = np.concatenate(
                    [track, np.zeros(padding_shape)], axis=0)
                extended_rev_track = np.concatenate(
                    [np.zeros(padding_shape),rev_track], axis=0)
                self.track_name_to_agg[track_name] = extended_track
                self.track_name_to_agg_revcomp[track_name] = extended_rev_track

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
            self.track_name_to_agg_revcomp[track_name][rev_slice_obj] +=\
                seqlet[track_name].rev

    def __len__(self):
        return self.length

