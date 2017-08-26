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
            revcomp=False) 


class FixedWindowAroundChunks(AbstractCoordProducer):

    def __init__(self, sliding,
                       flank,
                       suppress,
                       min_ratio,
                       max_seqlets_per_seq,
                       batch_size=50,
                       progress_update=5000,
                       verbose=True):
        self.sliding = sliding
        self.flank = flank
        self.suppress = suppress
        self.min_ratio = min_ratio
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
            if (max_per_seq is None):
                max_per_seq = summed_score_track[
                               list(range(len(summed_score_track))),
                               argmax_coords]
            for example_idx,argmax in enumerate(argmax_coords):
                #need to be able to expand without going off the edge
                if ((argmax >= self.flank) and
                    (argmax <= (summed_score_track.shape[1]
                                -(self.sliding+self.flank)))): 
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
                    int(max(np.floor(argmax+0.5-self.suppress),0)):
                    int(min(np.ceil(argmax+0.5+self.suppress),
                        len(summed_score_track[0])))]\
                    = -np.inf 
        return coords



