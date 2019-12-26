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

    def get_revcomp(self):
        return SeqletCoordinates(
                example_idx=self.example_idx,
                start=self.start, end=self.end,
                is_revcomp=(self.is_revcomp==False))


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
            return self._rev[(self.right_flank-right):
                             (self.right_flank+self.corelength+left)] 

    def get_revcomp(self):
        return SeqletTrack(left_flank=self.right_flank,
                           right_flank=self.left_flank,
                           fwd=self._rev, rev=self._fwd)


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
            rev=self.rev_tracks[coor.example_idx][
             (len(self.rev_tracks[coor.example_idx])-(coor.end+right_flank)):
             (len(self.rev_tracks[coor.example_idx])-(coor.start-left_flank))])

        if coor.is_revcomp:
            seqlet_data = seqlet_data.get_revcomp()
        return seqlet_data


class Seqlet(object):

    def __init__(self, coor):
        self.coor = coor
        self.trackname_to_seqletdata = OrderedDict()

    def set_seqlet_data(self, data_track, left_flank, right_flank):
        self[data_track.name] = data_track.get_seqlet_data(
                                    coor=self.coor,
                                    left_flank=left_flank,
                                    right_flank=right_flank) 

    def __setitem__(self, key, value):
        self.trackname_to_seqletdata[key] = value

    def __getitem__(self, key):
        return self.trackname_to_seqletdata[key]
