from __future__ import division, print_function, absolute_import
from collections import OrderedDict
from collections import namedtuple
import numpy as np
import scipy


class Snippet(object):

    def __init__(self, fwd, rev, has_pos_axis):
        assert len(fwd)==len(rev),str(len(fwd))+" "+str(len(rev))
        self.fwd = fwd
        self.rev = rev
        self.has_pos_axis = has_pos_axis

    def __len__(self):
        return len(self.fwd)

    def revcomp(self):
        return Snippet(fwd=np.copy(self.rev), rev=np.copy(self.fwd),
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

    @property
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
                         (self.track_length-coor.end):
                         (self.track_length-coor.start)],
                    has_pos_axis=self.has_pos_axis)
        if (coor.is_revcomp):
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


class SeqletCoordinates(object):

    def __init__(self, example_idx, start, end, is_revcomp):
        self.example_idx = example_idx
        self.start = start
        self.end = end
        self.is_revcomp = is_revcomp

    def revcomp(self):
        return SeqletCoordinates(
                example_idx=self.example_idx,
                start=self.start, end=self.end,
                is_revcomp=(self.is_revcomp==False))

    def __len__(self):
        return self.end - self.start

    def __str__(self):
        return ("example:"+str(self.example_idx)
                +",loc:"+str(self.start)+",end:"+str(self.end)
                +",rc:"+str(self.is_revcomp))


class Pattern(object):

    def __init__(self):
        self.track_name_to_snippet = OrderedDict()

    def __getitem__(self, key):
        return self.track_name_to_snippet[key]

    def __len__(self):
        raise NotImplementedError()

    def revcomp(self):
        raise NotImplementedError()


class Seqlet(Pattern):

    def __init__(self, coor):
        self.coor = coor
        super(Seqlet, self).__init__()

    def add_snippet_from_data_track(self, data_track): 
        snippet = data_track.get_snippet(coor=self.coor)
        return self.add_snippet(data_track_name=data_track.name,
                                snippet=snippet)
        
    def add_snippet(self, data_track_name, snippet):
        if (snippet.has_pos_axis):
            assert len(snippet)==len(self),\
                   ("tried to add snippet with pos axis of len "
                    +str(len(snippet))+" but snippet coords have "
                    +"len "+str(self.coor))
        self.track_name_to_snippet[data_track_name] = snippet 
        return self

    def revcomp(self):
        seqlet = Seqlet(coor=self.coor.revcomp())
        for track_name in self.track_name_to_snippet:
            seqlet.add_snippet(
                data_track_name=track_name,
                snippet=self.track_name_to_snippet[track_name].revcomp()) 
        return seqlet

    def __len__(self):
        return len(self.coor)

    @property
    def exidx_start_end_string(self):
        return (str(self.coor.example_idx)+"_"
                +str(self.coor.start)+"_"+str(self.coor.end))
 
        
class SeqletAndAlignment(object):

    def __init__(self, seqlet, alnmt):
        self.seqlet = seqlet
        #alnmt is the position of the beginning of seqlet
        #in the aggregated seqlet
        self.alnmt = alnmt 


class AbstractPatternAligner(object):

    def __init__(self, track_names, normalizer):
        self.track_names = track_names
        self.normalizer = normalizer

    def __call__(self, parent_pattern, child_pattern):
        #return an alignment
        raise NotImplementedError()     


class CrossCorrelationPatternAligner(AbstractPatternAligner):

    def __init__(self, pattern_crosscorr_settings):
        self.pattern_crosscorr_settings = pattern_crosscorr_settings

    def __call__(self, parent_pattern, child_pattern):
        fwd_data_parent, rev_data_parent = get_2d_data_from_pattern(
            pattern=parent_pattern,
            track_names=self.pattern_crosscorr_settings.track_names,
            normalizer=self.pattern_crosscorr_settings.normalizer) 
        fwd_data_child, rev_data_child = get_2d_data_from_pattern(
            pattern=child_pattern,
            track_names=self.pattern_crosscorr_settings.track_names,
            normalizer=self.pattern_crosscorr_settings.normalizer) 
        #find optimal alignments of fwd_data_child and rev_data_child
        #with fwd_data_parent.
        best_crosscorr, best_crosscorr_argmax =\
            get_best_alignment_crosscorr(
                parent_matrix=fwd_data_parent,
                child_matrix=fwd_data_child,
                min_overlap=self.pattern_crosscorr_settings.min_overlap)  
        best_crosscorr_rev, best_crosscorr_argmax_rev =\
            get_best_alignment_crosscorr(
                parent_matrix=fwd_data_parent,
                child_matrix=rev_data_child,
                min_overlap=self.pattern_crosscorr_settings.min_overlap) 
        if (best_crosscorr_rev > best_crosscorr):
            return (best_crosscorr_argmax_rev, True)
        else:
            return (best_crosscorr_argmax, False)


#implements the array interface but also tracks the
#unique seqlets for quick membership testing
class SeqletsAndAlignments(object):

    def __init__(self):
        self.arr = []
        self.unique_seqlets = {} 

    def __len__(self):
        return len(self.arr)

    def __iter__(self):
        return self.arr.__iter__()

    def __getitem__(self, idx):
        return self.arr[idx]

    def __contains__(self, seqlet):
        return (seqlet.exidx_start_end_string in self.unique_seqlets)

    def append(self, seqlet_and_alnmt):
        seqlet = seqlet_and_alnmt.seqlet
        if (seqlet.exidx_start_end_string in self.unique_seqlets):
            raise RuntimeError("Seqlet "
             +seqlet.exidx_start_end_string
             +" is already in SeqletsAndAlignments array")
        self.arr.append(seqlet_and_alnmt)
        self.unique_seqlets[seqlet.exidx_start_end_string] = seqlet


class AggregatedSeqlet(Pattern):

    def __init__(self, seqlets_and_alnmts_arr):
        super(AggregatedSeqlet, self).__init__()
        self._seqlets_and_alnmts = SeqletsAndAlignments()
        if (len(seqlets_and_alnmts_arr)>0):
            self._set_length(seqlets_and_alnmts_arr)
            self._compute_aggregation(seqlets_and_alnmts_arr) 

    def trim_to_positions_with_frac_support_of_peak(self, frac):
        per_position_center_counts =\
            self.get_per_position_seqlet_center_counts()
        max_support = max(per_position_center_counts)
        left_idx = 0
        while per_position_center_counts[left_idx] < frac*max_support:
            left_idx += 1
        right_idx = len(per_position_center_counts)
        while per_position_center_counts[right_idx-1] < frac*max_support:
            right_idx -= 1

        retained_seqlets_and_alnmts = []
        for seqlet_and_alnmt in self.seqlets_and_alnmts:
            seqlet_center = (
                seqlet_and_alnmt.alnmt+0.5*len(seqlet_and_alnmt.seqlet))
            #if the seqlet will fit within the trimmed pattern
            if ((seqlet_center >= left_idx) and
                (seqlet_center <= right_idx)):
                retained_seqlets_and_alnmts.append(seqlet_and_alnmt)
        new_start_idx = min([x.alnmt for x in retained_seqlets_and_alnmts])
        new_seqlets_and_alnmnts = [SeqletAndAlignment(seqlet=x.seqlet,
                                    alnmt=x.alnmt-new_start_idx) for x in
                                    retained_seqlets_and_alnmts] 
        return AggregatedSeqlet(seqlets_and_alnmts_arr=new_seqlets_and_alnmnts) 

    def get_per_position_seqlet_center_counts(self):
        per_position_center_counts = np.zeros(len(self.per_position_counts))
        for seqlet_and_alnmt in self._seqlets_and_alnmts:
            center = seqlet_and_alnmt.alnmt + len(seqlet_and_alnmt.seqlet)*0.5 
            per_position_center_counts[center] += 1
        return per_position_center_counts

    def plot_counts(self, counts, figsize=(20,2)):
        from matplotlib import pyplot as plt
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        self.plot_counts_given_ax(ax=ax, counts=counts)
        plt.show()

    def plot_counts_given_ax(self, ax, counts):
        ax.plot(counts)
        ax.set_ylim((0,max(counts)*1.1))
        ax.set_xlim((0, len(self)))

    def _set_length(self, seqlets_and_alnmts_arr):
        self.length = max([x.alnmt + len(x.seqlet)
                       for x in seqlets_and_alnmts_arr])  

    @property
    def seqlets_and_alnmts(self):
        return self._seqlets_and_alnmts

    @seqlets_and_alnmts.setter
    def seqlets_and_alnmts(self, val):
        assert type(val).__name__ == "SeqletsAndAlignments"
        self._seqlets_and_alnmts = val

    @property
    def num_seqlets(self):
        return len(self.seqlets_and_alnmts)

    @staticmethod 
    def from_seqlet(seqlet):
        return AggregatedSeqlet(seqlets_and_alnmts_arr=
                                [SeqletAndAlignment(seqlet,0)])

    def _compute_aggregation(self,seqlets_and_alnmts_arr):
        self._initialize_track_name_to_aggregation(
              sample_seqlet=seqlets_and_alnmts_arr[0].seqlet)
        self.per_position_counts = np.zeros((self.length,))
        for seqlet_and_alnmt in seqlets_and_alnmts_arr:
            self._add_pattern_with_valid_alnmt(
                    pattern=seqlet_and_alnmt.seqlet,
                    alnmt=seqlet_and_alnmt.alnmt)

    def _initialize_track_name_to_aggregation(self, sample_seqlet): 
        self._track_name_to_agg = OrderedDict() 
        self._track_name_to_agg_revcomp = OrderedDict() 
        for track_name in sample_seqlet.track_name_to_snippet:
            track_shape = tuple([self.length]
                           +list(sample_seqlet[track_name].fwd.shape[1:]))
            self._track_name_to_agg[track_name] =\
                np.zeros(track_shape).astype("float") 
            self._track_name_to_agg_revcomp[track_name] =\
                np.zeros(track_shape).astype("float") 
            self.track_name_to_snippet[track_name] = Snippet(
                fwd=self._track_name_to_agg[track_name],
                rev=self._track_name_to_agg_revcomp[track_name],
                has_pos_axis=sample_seqlet[track_name].has_pos_axis) 

    def _pad_before(self, num_zeros):
        assert num_zeros > 0
        self.length += num_zeros
        for seqlet_and_alnmt in self.seqlets_and_alnmts:
            seqlet_and_alnmt.alnmt += num_zeros
        self.per_position_counts =\
            np.concatenate([np.zeros((num_zeros,)),
                            self.per_position_counts],axis=0) 
        for track_name in self.track_name_to_snippet:
            track = self._track_name_to_agg[track_name]
            if (self.track_name_to_snippet[track_name].has_pos_axis):
                rev_track = self._track_name_to_agg_revcomp[track_name]
                padding_shape = tuple([num_zeros]+list(track.shape[1:])) 
                extended_track = np.concatenate(
                    [np.zeros(padding_shape), track], axis=0)
                extended_rev_track = np.concatenate(
                    [rev_track, np.zeros(padding_shape)], axis=0)
                self._track_name_to_agg[track_name] = extended_track
                self._track_name_to_agg_revcomp[track_name] =\
                    extended_rev_track

    def _pad_after(self, num_zeros):
        assert num_zeros > 0
        self.length += num_zeros 
        self.per_position_counts =\
            np.concatenate([self.per_position_counts,
                            np.zeros((num_zeros,))],axis=0) 
        for track_name in self.track_name_to_snippet:
            track = self._track_name_to_agg[track_name]
            if (self.track_name_to_snippet[track_name].has_pos_axis):
                rev_track = self._track_name_to_agg_revcomp[track_name]
                padding_shape = tuple([num_zeros]+list(track.shape[1:])) 
                extended_track = np.concatenate(
                    [track, np.zeros(padding_shape)], axis=0)
                extended_rev_track = np.concatenate(
                    [np.zeros(padding_shape),rev_track], axis=0)
                self._track_name_to_agg[track_name] = extended_track
                self._track_name_to_agg_revcomp[track_name] =\
                    extended_rev_track

    def merge_aggregated_seqlet(self, agg_seqlet, aligner):
        #only merge those seqlets in agg_seqlet that are not already
        #in the current seqlet
        for seqlet_and_alnmt in agg_seqlet.seqlets_and_alnmts:
            if seqlet_and_alnmt.seqlet not in self.seqlets_and_alnmts: 
                self.add_pattern(pattern=seqlet_and_alnmt.seqlet,
                                 aligner=aligner) 
        
    def add_pattern(self, pattern, aligner):

        (alnmt, revcomp_match) = aligner(parent_pattern=self,
                                         child_pattern=pattern)
        if (revcomp_match):
            pattern = pattern.revcomp()
        if alnmt < 0:
           self._pad_before(num_zeros=abs(alnmt)) 
           alnmt=0
        end_coor_of_pattern = (alnmt + len(pattern))
        if (end_coor_of_pattern > self.length):
            self._pad_after(num_zeros=(end_coor_of_pattern - self.length))
        self._add_pattern_with_valid_alnmt(pattern=pattern, alnmt=alnmt)

    def _add_pattern_with_valid_alnmt(self, pattern, alnmt):
        assert alnmt >= 0
        assert alnmt + len(pattern) <= self.length

        slice_obj = slice(alnmt, alnmt+len(pattern))
        rev_slice_obj = slice(self.length-(alnmt+len(pattern)),
                              self.length-alnmt)

        self.seqlets_and_alnmts.append(
             SeqletAndAlignment(seqlet=pattern, alnmt=alnmt))
        self.per_position_counts[slice_obj] += 1.0 

        for track_name in self._track_name_to_agg:
            if (self.track_name_to_snippet[track_name].has_pos_axis==False):
                self._track_name_to_agg[track_name] +=\
                    pattern[track_name].fwd
                self._track_name_to_agg_revcomp[track_name] +=\
                    pattern[track_name].rev
            else:
                self._track_name_to_agg[track_name][slice_obj] +=\
                    pattern[track_name].fwd 
                self._track_name_to_agg_revcomp[track_name][rev_slice_obj]\
                     += pattern[track_name].rev
            self.track_name_to_snippet[track_name] =\
             Snippet(
              fwd=(self._track_name_to_agg[track_name]
                   /self.per_position_counts[:,None]),
              rev=(self._track_name_to_agg_revcomp[track_name]
                   /self.per_position_counts[::-1,None]),
              has_pos_axis=self.track_name_to_snippet[track_name].has_pos_axis) 

    def __len__(self):
        return self.length

    def revcomp(self):
        rev_agg_seqlet = AggregatedSeqlet(seqlets_and_alnmts_arr=[])
        rev_agg_seqlet.per_position_counts = self.per_position_counts[::-1]
        rev_agg_seqlet._track_name_to_agg = OrderedDict(
         [(x, np.copy(self._track_name_to_agg_revcomp[x]))
           for x in self._track_name_to_agg])
        rev_agg_seqlet._track_name_to_agg_revcomp = OrderedDict(
         [(x, np.copy(self._track_name_to_agg[x]))
           for x in self._track_name_to_agg_revcomp])
        rev_agg_seqlet.track_name_to_snippet = OrderedDict([
         (x, Snippet(
             fwd=np.copy(self.track_name_to_snippet[x].rev),
             rev=np.copy(self.track_name_to_snippet[x].fwd),
             has_pos_axis=self.track_name_to_snippet[x].has_pos_axis)) 
         ]) 
        rev_seqlets_and_alignments_arr = [
            SeqletAndAlignment(seqlet=x.seqlet.revcomp(),
                               alnmt=self.length-(x.alnmt+len(x.seqlet)))
            for x in self.seqlets_and_alnmts] 
        rev_agg_seqlet._set_length(rev_seqlets_and_alignments_arr)
        for seqlet_and_alnmt in rev_seqlets_and_alignments_arr:
            rev_agg_seqlet.seqlets_and_alnmts.append(seqlet_and_alnmt)
        return rev_agg_seqlet 

    def get_seqlet_centers(self):
        return [x.alnmt + int(0.5*(len(x.seqlet)))
                for x in self.seqlets_and_alnmts] 

    def viz_seqlet_centers(self):
        from matplotlib import pyplot as plt
        plt.hist(self.get_seqlet_centers(), bins=len(self))


def get_2d_data_from_patterns(patterns, track_names, normalizer):
    all_fwd_data = []
    all_rev_data = []
    for pattern in patterns:
        fwd_data, rev_data = get_2d_data_from_pattern(
            pattern=pattern, track_names=track_names,
            normalizer=normalizer) 
        all_fwd_data.append(fwd_data)
        all_rev_data.append(rev_data)
    return (np.array(all_fwd_data),
            np.array(all_rev_data))


def get_2d_data_from_pattern(pattern, track_names, normalizer): 
    snippets = [pattern[track_name]
                 for track_name in track_names] 
    fwd_data = np.concatenate([normalizer(
             np.reshape(snippet.fwd, (len(snippet.fwd), -1)))
            for snippet in snippets], axis=1)
    rev_data = np.concatenate([normalizer(
            np.reshape(snippet.rev, (len(snippet.rev), -1)))
            for snippet in snippets], axis=1)
    return fwd_data, rev_data


def get_best_alignment_crosscorr(parent_matrix, child_matrix, min_overlap):
    assert len(np.shape(parent_matrix))==2
    assert len(np.shape(child_matrix))==2
    assert np.shape(parent_matrix)[1] == np.shape(child_matrix)[1]

    padding_amt = int(np.ceil(np.shape(child_matrix)[0]*min_overlap))
    #pad the parent matrix as necessary
    parent_matrix = np.pad(array=parent_matrix,
                           pad_width=[(padding_amt, padding_amt),(0,0)],
                           mode='constant')
    correlations = scipy.signal.correlate2d(
        in1=parent_matrix, in2=child_matrix, mode='valid')
    best_crosscorr_argmax = np.argmax(correlations)-padding_amt
    best_crosscorr = np.max(correlations)
    #subtract the padding 
    best_crosscorr = best_crosscorr - padding_amt
    return (best_crosscorr, best_crosscorr_argmax)


