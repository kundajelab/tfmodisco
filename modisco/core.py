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


#returns offset and isfwd of seql2 w.r.t. seql1
def get_best_alignment_for_pair(seql1_corelen, seql1_hyp, seql1_onehot,
                                seql2_corelen, seql2_hyp_fwd, seql2_hyp_rev,
                                min_overlap_frac, pair_sim_metric):
    from . import affinitymat
    #compute for fwd
    fwd_sim_results, possible_fwd_offsets =\
     affinitymat.asymmetric_compute_sim_on_pair(
        seql1_corelen=seql1_corelen,
        seql1_hyp=seql1_hyp,
        seql1_onehot=seql1_onehot,
        seql2_corelen=seql2_corelen,
        seql2_hyp=seql2_hyp_fwd,
        min_overlap_frac=min_overlap_frac,
        pair_sim_metric=pair_sim_metric) 
    #compute for rev
    rev_sim_results, possible_rev_offsets =\
     affinitymat.asymmetric_compute_sim_on_pair(
        seql1_corelen=seql1_corelen,
        seql1_hyp=seql1_hyp,
        seql1_onehot=seql1_onehot,
        seql2_corelen=seql2_corelen,
        seql2_hyp=seql2_hyp_rev,
        min_overlap_frac=min_overlap_frac,
        pair_sim_metric=pair_sim_metric)
    assert np.max(np.abs(possible_fwd_offsets-possible_rev_offsets))==0 #should be same
    isfwd = np.max(rev_sim_results) < np.max(fwd_sim_results)
    if (isfwd):
        offset = possible_fwd_offsets[np.argmax(fwd_sim_results)]
    else:
        offset = possible_rev_offsets[np.argmax(rev_sim_results)]
    return offset, isfwd


def create_aggregated_seqlet(
        sorted_seqlets, onehot_trackname, hyp_trackname, flanklen,
        min_overlap_frac, pair_sim_metric):
    #assert that there are > 1 seqlets 
    assert len(sorted_seqlets) > 1
    #assert that the seqlets are of equal length
    assert len(set([len(x) for x in sorted_seqlets]))==1
    #fix core length to that of the first seqlet
    corelen = len(sorted_seqlets[0]) 
    #initialize agg tracks to be core + flank of first seqlet
    aggfwd_onehot = np.array(
                     sorted_seqlets[0][onehot_trackname].get_core_with_flank(
                        left=flanklen, right=flanklen,
                        is_revcomp=False))
    aggfwd_hyp = np.array(
                  sorted_seqlets[0][hyp_trackname].get_core_with_flank(
                        left=flanklen, right=flanklen,
                        is_revcomp=False))  
    #initialize perposcount to be ones
    perposcount = np.ones(len(aggfwd_onehot))
    #iterate through the seqlets after the first seqlet:
    for unoriented_seqlet in sorted_seqlets[1:]:
        seql2_hyp_fwd = unoriented_seqlet[hyp_trackname].get_core_with_flank(
                                                left=flanklen,
                                                right=flanklen,
                                                is_revcomp=False)
        seql2_hyp_rev = unoriented_seqlet[hyp_trackname].get_core_with_flank(
                                                left=flanklen,
                                                right=flanklen,
                                                is_revcomp=True)
        #set normalized to be agg/perposcount 
        normfwd_onehot = aggfwd_onehot/perposcount[:,None]
        assert np.max(np.abs(np.sum(normfwd_onehot, axis=-1)-1.0)) < 1e-5
        normfwd_hyp = aggfwd_hyp/perposcount[:,None]
        #compute offset of current seqlet relative to normalized agg.
        offset,isfwd = get_best_alignment_for_pair(
                    seql1_corelen=corelen, seql1_hyp=normfwd_hyp,
                    seql1_onehot=normfwd_onehot,
                    seql2_corelen=corelen,
                    seql2_hyp_fwd=seql2_hyp_fwd,
                    seql2_hyp_rev=seql2_hyp_rev,
                    min_overlap_frac=min_overlap_frac,
                    pair_sim_metric=pair_sim_metric) 
        #reorient the seqlet accordingly
        if (isfwd):
            reoriented_seqlet = unoriented_seqlet
        else:
            reoriented_seqlet = unoriented_seqlet.get_revcomp() 
        reoriented_hyp = reoriented_seqlet[hyp_trackname].get_core_with_flank(
                                                left=flanklen,
                                                right=flanklen,
                                                is_revcomp=False)
        reoriented_onehot = (reoriented_seqlet[onehot_trackname]
                                               .get_core_with_flank(
                                                left=flanklen,
                                                right=flanklen,
                                                is_revcomp=False))
        assert len(reoriented_hyp)==(2*flanklen + corelen)
        assert reoriented_onehot.shape==reoriented_hyp.shape
        # update agg, but only the parts that overlap current seqlet.
        startidx_in_reoriented = max(-offset, 0) 
        endidx_in_reoriented = (2*flanklen + corelen) - max(offset, 0) 
        startidx_in_updateslice = max(offset, 0) 
        endidx_in_updateslice = (2*flanklen + corelen) - max(-offset, 0)
        assert ((endidx_in_updateslice-startidx_in_updateslice)
                ==(endidx_in_reoriented-startidx_in_reoriented))
        aggfwd_hyp[startidx_in_updateslice:endidx_in_updateslice] +=\
            reoriented_hyp[startidx_in_reoriented:endidx_in_reoriented] 
        aggfwd_onehot[startidx_in_updateslice:endidx_in_updateslice] +=\
            reoriented_onehot[startidx_in_reoriented:endidx_in_reoriented] 
        perposcount[startidx_in_updateslice:endidx_in_updateslice] += 1

    normfwd_onehot = aggfwd_onehot/perposcount[:,None]
    assert np.max(np.abs(np.sum(normfwd_onehot, axis=-1)-1.0)) < 1e-5
    normfwd_hyp = aggfwd_hyp/perposcount[:,None]
    #Once the aggregrate is found...
    #iterate through seqlets to find offsets/orientation
    offsets = []
    oriented_seqlets = []
    for unoriented_seqlet in sorted_seqlets:
        #Recalculate offsets
        seql2_hyp_fwd = unoriented_seqlet[hyp_trackname].get_core_with_flank(
                                                left=flanklen,
                                                right=flanklen,
                                                is_revcomp=False)
        seql2_hyp_rev = unoriented_seqlet[hyp_trackname].get_core_with_flank(
                                                left=flanklen,
                                                right=flanklen,
                                                is_revcomp=True)
        #get alignment
        offset,isfwd = get_best_alignment_for_pair(
                    seql1_corelen=corelen, seql1_hyp=normfwd_hyp,
                    seql1_onehot=normfwd_onehot,
                    seql2_corelen=corelen,
                    seql2_hyp_fwd=seql2_hyp_fwd,
                    seql2_hyp_rev=seql2_hyp_rev,
                    min_overlap_frac=min_overlap_frac,
                    pair_sim_metric=pair_sim_metric) 
        #orient the seqlet
        if (isfwd):
            reoriented_seqlet = unoriented_seqlet
        else:
            reoriented_seqlet = unoriented_seqlet.get_revcomp() 
        #append to offsets and oriented_seqlets
        oriented_seqlets.append(reoriented_seqlet)
        offsets.append(offset)

    # Use seqlets+offsets to return AggregatedSeqlet
    assert len(offsets)==len(oriented_seqlets)
    return AggregatedSeqlet(seqlets=oriented_seqlets, offsets=offsets)         


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
            if (trackname_to_tracksumrev is not None):
                trackname_to_tracksumrev[trackname] =\
                    np.zeros((total_len,
                              seqlets[0][trackname].corerev.shape[1]))

        for seqlet,offset in zip(seqlets, offsets):
            position_counts[left_span+offset:
                            left_span+offset+len(seqlet)] += 1.0 
            for trackname in track_names:
                trackname_to_tracksumfwd[trackname][left_span+offset:
                    left_span+offset+len(seqlet)] += seqlet[trackname].corefwd
                if (trackname_to_tracksumrev is not None):
                    trackname_to_tracksumrev[trackname][
                     total_len-(left_span+offset+len(seqlet)):
                     total_len-(left_span+offset)] += seqlet[trackname].corerev

        assert np.min(position_counts) >= 0 

        self.seqlets = seqlets
        self.offsets = np.array(offsets)+left_span
        self.position_counts = position_counts
        self._trackname_to_tracksumfwd = trackname_to_tracksumfwd
        self._trackname_to_tracksumrev = trackname_to_tracksumrev
        self._normalize_tracksums()

    def _normalize_tracksums(self):
        trackname_to_genericseqdata = OrderedDict() 
        for trackname in self._trackname_to_tracksumfwd:
            #fwdnorm = (self._trackname_to_tracksumfwd[trackname]/
            #           self.position_counts[:,None])
            fwdnorm = (self._trackname_to_tracksumfwd[trackname]/
                       len(self.seqlets))
            #revnorm = (self._trackname_to_tracksumrev[trackname]/
            #           self.position_counts[::-1,None])
            if (self._trackname_to_tracksumrev is not None):
                revnorm = (self._trackname_to_tracksumrev[trackname]/
                           len(self.seqlets))
            else:
                revnorm = None 
            trackname_to_genericseqdata[trackname] = GenericSeqData(
                                                      fwd=fwdnorm, rev=revnorm)
        self._trackname_to_genericseqdata = trackname_to_genericseqdata

    def _increment_offsets(self, inc_val):
        new_offsets = []
        for offset in self.offsets:
            new_offsets = offset + inc_val
        self.offsets = new_offsets

    def _assert_common_set_of_track_names(self):
        assert (len(set(self.trackname_to_tracksumfwd.keys()))==        
                len(set(self.trackname_to_tracksumrev.keys())))
        assert (len(set(self.trackname_to_tracksumfwd.keys()))==        
                len(set(self.trackname_to_genericseqdata.keys())))

    ##pads tracks with zeros; pad_val is the number of zeros to pad with
    #def _pad_tracks(self, pad_val, pad_before):
    #    self._assert_common_set_of_track_names()
    #    for track_name in self._trackname_to_genericseqdata:
    #        fwd_track = self._trackname_to_tracksumfwd[track_name]
    #        padding_shape = tuple([num_zeros]+list(fwd_track.shape[1:]))
    #        if (pad_before):
    #            new_fwd_track = np.concatenate(
    #                         [np.zeros(padding_shape), fwd_track], axis=0)
    #        else:
    #            new_fwd_track = np.concatenate(
    #                         [fwd_track, np.zeros(padding_shape)], axis=0)
    #        self._trackname_to_tracksumfwd[track_name] = new_fwd_track
    #        if (self._trackname_to_tracksumrev is not None):
    #            rev_track = self._trackname_to_tracksumrev[track_name] 
    #            if (pad_before):
    #                new_rev_track = np.concatenate(
    #                              [rev_track, np.zeros(padding_shape)], axis=0)
    #            else:
    #                new_rev_track = np.concatenate(
    #                              [np.zeros(padding_shape),rev_track], axis=0)
    #            self._trackname_to_tracksumrev[track_name] = new_rev_track
    #        else: 
    #            rev_track = None
    #    self._normalize_tracksums()
    #            
    #def _pad_before(self, pad_val):
    #    assert pad_val > 0
    #    self._increment_offsets(inc_val=pad_val)
    #    #left pad per_position_counts
    #    self.per_position_counts = np.concatenate([np.zeros((pad_val,)),
    #                                          self.per_position_counts],axis=0) 
    #    self._pad_tracks(pad_val=pad_val, pad_before=True)
    #            
    #def _pad_after(self, pad_val):
    #    assert pad_val > 0
    #    #right pad per_position_counts
    #    self.per_position_counts = np.concatenate([self.per_position_counts,
    #                                              np.zeros((pad_val,))],axis=0) 
    #    self._pad_tracks(pad_val=pad_val, pad_before=False)

    def __getitem__(self, key):
        return self._trackname_to_genericseqdata[key]
         

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
