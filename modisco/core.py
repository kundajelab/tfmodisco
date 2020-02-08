from __future__ import division, print_function, absolute_import
from collections import OrderedDict
from collections import namedtuple
from collections import defaultdict
import numpy as np
import itertools
import sys


class Coordinate(object):

    def __init__(self, example_idx, start, end, is_revcomp):
        self.example_idx = example_idx
        self.start = start
        self.end = end
        self.is_revcomp = is_revcomp

    def __len__(self):
        return self.end-self.start

    def get_revcomp(self):
        return Coordinate(
                example_idx=self.example_idx,
                start=self.start, end=self.end,
                is_revcomp=(self.is_revcomp==False))


class GenericSeqData(object):
    def __init__(self, fwd, rev):
        self.fwd = fwd
        self.rev = rev 

    @property
    def hasrev(self):
        return self.rev is not None

    def get_revcomp(self):
        if (self.rev is None):
            raise RuntimeError("Trying to reverse-complement something that"
                               +" has no rc data")
        return GenericSeqData(fwd=self.rev, rev=self.fwd)


class SeqletData(object):

    def __init__(self, left_flank, right_flank, fwd, rev):
        self.left_flank = left_flank
        self.right_flank = right_flank
        assert fwd.shape==rev.shape
        assert len(fwd) > (left_flank+right_flank) #needs to be some core
        self.corelength = len(fwd)-(left_flank+right_flank)
        self._fwd = fwd
        self._rev = rev

    @property
    def hasrev(self):
        return self._rev is not None

    @property 
    def corefwd(self):
        return self.get_core_with_flank(left=0, right=0, is_revcomp=False)

    @property 
    def corerev(self):
        return self.get_core_with_flank(left=0, right=0, is_revcomp=True)

    def get_core_with_flank(self, left, right, is_revcomp):
        if (left > self.left_flank):
            raise RuntimeError("Left flank requested was",left,
                               "but available flank was only",self.left_flank)
        if (right > self.right_flank):
            raise RuntimeError("Right flank requested was",right,
                               "but available flank was only",self.right_flank)
        if (is_revcomp==False):
            return self._fwd[(self.left_flank-left):
                             (self.left_flank+self.corelength+right)] 
        else:
            return self._rev[(self.right_flank-left):
                             (self.right_flank+self.corelength+right)] 

    def get_revcomp(self):
        if (self._rev is None):
            raise RuntimeError("Trying to reverse-complement something that"
                               +" has no rc data")
        return SeqletData(left_flank=self.right_flank,
                           right_flank=self.left_flank,
                           fwd=self._rev, rev=self._fwd)


class DataTrackSet(object):

    def __init__(self, data_tracks):
        self.data_tracks = data_tracks

    def create_seqlets(self, coords, flanks):
        seqlets = []
        if (hasattr(flanks, '__iter__')==False):
            flanks = [flanks for x in coords]
        for coord,flank in zip(coords, flanks):
            seqlets.append(self.create_seqlet(coord=coord, flank=flank))
        return seqlets

    def create_seqlet(self, coord, flank):
        seqlet = Seqlet(coor=coord) 
        for data_track in self.data_tracks:
            seqlet.set_seqlet_data(data_track=data_track,
                                   left_flank=flank, right_flank=flank)
        return seqlet


class DataTrack(object):

    def __init__(self, name, fwd_tracks, rev_tracks):
        self.name = name
        assert (rev_tracks is None) or (len(fwd_tracks)==len(rev_tracks))
        if (rev_tracks is not None):
            for fwd,rev in zip(fwd_tracks, rev_tracks):
                assert len(fwd)==len(rev)
        self.fwd_tracks = fwd_tracks
        self.rev_tracks = rev_tracks

    def __len__(self):
        return len(self.fwd_tracks)
    
    def get_seqlet_data(self, coor, left_flank, right_flank):
        assert coor.start-left_flank >= 0, (coor.start,left_flank)
        assert len(self.fwd_tracks[coor.example_idx]) >=\
                coor.end+right_flank, (len(self.fwd_tracks[coor.example_idx]),
                                       coor.end,right_flank)
        seqlet_data = SeqletData(
            left_flank=left_flank,
            right_flank=right_flank,
            fwd=self.fwd_tracks[coor.example_idx][coor.start-left_flank:
                                                  coor.end+right_flank],
            rev=(self.rev_tracks[coor.example_idx][
             (len(self.rev_tracks[coor.example_idx])-(coor.end+right_flank)):
             (len(self.rev_tracks[coor.example_idx])-(coor.start-left_flank))]
                 if self.rev_tracks is not None else None))

        if coor.is_revcomp:
            seqlet_data = seqlet_data.get_revcomp()
        return seqlet_data


class AggregatedSeqlet(object):

    def __init__(self, seqlets, offsets):
        track_names = list(seqlets[0].trackname_to_seqletdata.keys())
        left_span = -min(offsets) 
        assert left_span >= 0
        right_span = max(offset+len(seqlet) for (seqlet,offset)
                         in zip(seqlets,offsets))
        total_len = right_span+left_span

        position_counts = np.zeros((total_len,))

        trackname_to_tracksumfwd = OrderedDict() 
        if (seqlets[0].hasrev):
            trackname_to_tracksumrev = OrderedDict() 
        else:
            trackname_to_tracksumrev = None

        for trackname in track_names:
            trackname_to_tracksumfwd[trackname] =\
                np.zeros((total_len, seqlets[0][trackname].corefwd.shape[1]))
            if (seqlets[0].hasrev):
                trackname_to_tracksumrev[trackname] =\
                    np.zeros((total_len,
                              seqlets[0][trackname].corerev.shape[1]))

        for seqlet,offset in zip(seqlets, offsets):
            position_counts[left_span+offset:
                            left_span+offset+len(seqlet)] += 1.0 
            for trackname in track_names:
                trackname_to_tracksumfwd[trackname][left_span+offset:
                    left_span+offset+len(seqlet)] += seqlet[trackname].corefwd
                if (seqlets[0].hasrev):
                    trackname_to_tracksumrev[trackname][
                     total_len-(left_span+offset+len(seqlet)):
                     total_len-(left_span+offset)] += seqlet[trackname].corerev

        assert np.min(position_counts) >= 0 

        trackname_to_genericseqdata = OrderedDict() 
        for trackname in track_names:
            #fwdnorm = (trackname_to_tracksumfwd[trackname]/
            #           position_counts[:,None])
            fwdnorm = trackname_to_tracksumfwd[trackname]/len(seqlets)
            #revnorm = (trackname_to_tracksumrev[trackname]/
            #           position_counts[::-1,None])
            revnorm = trackname_to_tracksumrev[trackname]/len(seqlets)
            trackname_to_genericseqdata[trackname] = GenericSeqData(
                                                      fwd=fwdnorm, rev=revnorm)
        self.seqlets = seqlets
        self.offsets = np.array(offsets)+left_span
        self.position_counts = position_counts
        self.trackname_to_tracksumfwd = trackname_to_tracksumfwd
        self.trackname_to_tracksumrev = trackname_to_tracksumrev
        self.trackname_to_genericseqdata = trackname_to_genericseqdata

    def __getitem__(self, key):
        return self.trackname_to_genericseqdata[key]
         

class Seqlet(object):

    def __init__(self, coor):
        self.coor = coor
        self.trackname_to_seqletdata = OrderedDict()

    def __len__(self):
        return len(self.coor)

    @property
    def hasrev(self):
        assert len(self.trackname_to_seqletdata.keys()) > 0
        return self.trackname_to_seqletdata[
                list(self.trackname_to_seqletdata.keys())[0]].hasrev

    def set_seqlet_data(self, data_track, left_flank, right_flank):
        self[data_track.name] = data_track.get_seqlet_data(
                                    coor=self.coor,
                                    left_flank=left_flank,
                                    right_flank=right_flank) 

    def __setitem__(self, key, value):
        self.trackname_to_seqletdata[key] = value

    def __getitem__(self, key):
        return self.trackname_to_seqletdata[key]

    def get_revcomp(self):
        new_seqlet = Seqlet(coor=self.coor.get_revcomp())
        for trackname in self.trackname_to_seqletdata:
            new_seqlet[trackname] = (self.trackname_to_seqletdata[trackname]
                                         .get_revcomp())
        return new_seqlet
