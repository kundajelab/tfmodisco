from __future__ import division, print_function, absolute_import
from .core import SeqletCoordinates
from modisco import backend as B 
import numpy as np
from collections import defaultdict
import itertools


class AbstractCoordProducer(object):

    def __call__(self):
        raise NotImplementedError() 


class SeqletCoordsFWAP(SeqletCoordinates):
    """
        Coordinates for the FixedWindowAroundChunks CoordProducer 
    """
    def __init__(self, example_idx, start, end, score):
        self.score = score 
        super(SeqletCoordsFWAP, self).__init__(
            example_idx=example_idx,
            start=start, end=end,
            is_revcomp=False) 


class CoordOverlapDetector(object):

    def __init__(self, min_overlap_fraction):
        self.min_overlap_fraction = min_overlap_fraction

    def __call__(self, coord1, coord2):
        if (coord1.example_idx != coord2.example_idx):
            return False
        min_overlap = self.min_overlap_fraction*min(len(coord1), len(coord2))
        overlap_amt = (min(coord1.end, coord2.end)-
                       max(coord1.start, coord2.start))
        return (overlap_amt >= min_overlap)


class CoordComparator(object):

    def __init__(self, attribute_provider):
        self.attribute_provider = attribute_provider

    def get_larger(self, coord1, coord2):
        return (coord1 if (self.attribute_provider(coord1) >=
                           self.attribute_provider(coord2)) else coord2)

    def get_smaller(self, coord1, coord2):
        return (coord1 if (self.attribute_provider(coord1) <=
                           self.attribute_provider(coord2)) else coord2)


class ResolveOverlapsCoordProducer(AbstractCoordProducer):

    def __init__(self, coord_producer, overlap_detector, coord_comparator):
        self.coord_producer = coord_producer
        self.overlap_detector = overlap_detector
        self.coord_comparator = coord_comparator

    def __call__(self, kwargsets):
        coord_sets = [self.coord_producer(**kwargset) for
                       kwargset in kwargsets]
        example_idx_to_coords = defaultdict(list)  
        for coord in itertools.chain(*coord_sets):
            example_idx_to_coords[coord.example_idx].append(coord)
        for example_idx, coords in example_idx_to_coords.items():
            final_coords_set = set(coords)
            for i in range(len(coords)):
                coord1 = coords[i]
                for coord2 in coords[i+1:]:
                    if (coord1 not in final_coords_set):
                        break
                    if ((coord2 in final_coords_set)
                         and self.overlap_detector(coord1, coord2)):
                        final_coords_set.remove(
                         self.coord_comparator.get_smaller(coord1, coord2)) 
            example_idx_to_coords[example_idx] = list(final_coords_set)
        return list(itertools.chain(*example_idx_to_coords.values()))


class FixedWindowAroundChunks(AbstractCoordProducer):

    def __init__(self, sliding=11,
                       flank=10,
                       suppress=20,
                       max_seqlets_per_seq=5,
                       min_ratio_top_peak=0.0,
                       min_ratio_over_bg=0.0,
                       batch_size=50,
                       progress_update=5000,
                       verbose=True):
        self.sliding = sliding
        self.flank = flank
        self.suppress = suppress
        self.min_ratio_top_peak = min_ratio_top_peak
        self.min_ratio_over_bg = min_ratio_over_bg
        self.max_seqlets_per_seq = max_seqlets_per_seq
        self.batch_size = batch_size
        self.progress_update = progress_update
        self.verbose = verbose

    def __call__(self, score_track):
     
        assert len(score_track.shape)==2 
        if (self.verbose):
            print("Compiling functions") 
        window_sum_function = B.get_window_sum_function(
                                window_size=self.sliding,
                                same_size_return=False)
        argmax_func = B.get_argmax_function()

        if (self.verbose):
            print("Computing window sums") 
        summed_score_track = np.array(window_sum_function(
            inp=score_track,
            batch_size=self.batch_size,
            progress_update=
             (self.progress_update if self.verbose else None))).astype("float") 

        #As we extract seqlets, we will zero out the values at those positions
        #so that the mean of the background can be updated to exclude
        #the seqlets (which are likely to be outliers)
        zerod_out_summed_score_track = np.copy(summed_score_track)
         
        if (self.verbose):
            print("Identifying seqlet coordinates") 

        coords = []
        max_per_seq = None
        for n in range(self.max_seqlets_per_seq):
            argmax_coords = argmax_func(
                                inp=summed_score_track,
                                batch_size=self.batch_size,
                                progress_update=(self.progress_update
                                                 if self.verbose else None)) 
            unsuppressed_per_track = np.sum(summed_score_track > -np.inf,
                                            axis=1)
            bg_avg_per_track = np.sum(zerod_out_summed_score_track, axis=1)/\
                                     (unsuppressed_per_track)
            
            if (max_per_seq is None):
                max_per_seq = summed_score_track[
                               list(range(len(summed_score_track))),
                               argmax_coords]
            for example_idx,argmax in enumerate(argmax_coords):

                #suppress the chunks within +- self.suppress
                left_supp_idx = int(max(np.floor(argmax+0.5-self.suppress),0))
                right_supp_idx = int(min(np.ceil(argmax+0.5+self.suppress),
                                     len(summed_score_track[0])))

                #need to be able to expand without going off the edge
                if ((argmax >= self.flank) and
                    (argmax <= (score_track.shape[1]
                                -(self.sliding+self.flank)))): 
                    chunk_height = summed_score_track[example_idx][argmax]
                    #only include chunk that are at least a certain
                    #fraction of the max chunk
                    if ((chunk_height >=
                        max_per_seq[example_idx]*self.min_ratio_top_peak)
                        and (np.abs(chunk_height) >=
                             np.abs(bg_avg_per_track[example_idx])
                             *self.min_ratio_over_bg)):
                        coord = SeqletCoordsFWAP(
                            example_idx=example_idx,
                            start=argmax-self.flank,
                            end=argmax+self.sliding+self.flank,
                            score=chunk_height) 
                        coords.append(coord)
                    #only zero out if the region was included, so that we
                    #don't zero out sequences that do not pass the conditions
                    zerod_out_summed_score_track[
                        example_idx,
                        left_supp_idx:right_supp_idx] = 0.0
                summed_score_track[
                    example_idx, left_supp_idx:right_supp_idx] = -np.inf 
        return coords



