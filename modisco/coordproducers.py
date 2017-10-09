from __future__ import division, print_function, absolute_import
from .core import SeqletCoordinates
from modisco import backend as B 
import numpy as np


class AbstractCoordProducer(object):

    def get_coords(self):
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

    def get_coords(self, score_track):
      
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



