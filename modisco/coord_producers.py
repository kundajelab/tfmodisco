from .core import SeqletCoordinates
import .backend as B 
import numpy as np


class AbstractCoordProducer(object):

    def get_coords(self):
        raise NotImplementedError() 


class SeqletCoordsFWAP(SeqletCoordinates):
    """
        Coordinates for the FixedWindowAroundPeaks CoordProducer 
    """
    def __init__(self, example_idx, start, end, score):
        self.score = score 
        super(SeqletCoordsFWAP, self).__init__(
            example_idx=example_idx,
            start=start, end=end,
            revcomp=False) 


class FixedWindowAroundChunks(object):

    def __init__(self, sliding,
                       flank,
                       suppress,
                       min_ratio,
                       max_peaks_per_seq,
                       batch_size=50,
                       progress_update=5000,
                       verbose=True):
        self.sliding = sliding
        self.flank = flank
        self.suppress = suppress
        self.min_ratio = min_ratio
        self.max_peaks_per_seq = max_peaks_per_seq
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
        summed_score_track = window_sum_function(
            inp=score_track,
            batch_size=self.batch_size,
            progress_update=(self.progress_update if self.verbose else None)) 
         
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
            if (max_per_seq is None)
                max_per_seq = max_vals
            for example_idx,argmax in enumerate(argmax_coords):
                #need to be able to expand without going off the edge
                if ((argmax >= self.flank) and
                    (argmax <= (summed_score_track.shape[1]
                               -(self.sliding+ self.flank)))): 
                    chunk_height = summed_score_track[example_idx][argmax]
                    #only include chunk that are at least a certain
                    #fraction of the max chunk
                    if (chunk_height >=
                        max_per_seq[example_idx]*self.min_ratio):
                        coord = SeqletCoordsFWAP(
                            example_idx=example_idx,
                            start=argmax-self.flank,
                            end=argmax+self.sliding+self.flank,
                            score=chunk_height) 
                        coords.append(coord)
                #suppress the chunks within +- self.suppress
                summed_score_track[
                    example_idx,
                    (max(argmax-self.suppress,0):
                     max(argmax+self.suppress))] = -np.inf 
        return coords



