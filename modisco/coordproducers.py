from __future__ import division, print_function, absolute_import
from .core import SeqletCoordinates
from modisco import util
import numpy as np
from collections import defaultdict
import itertools
from sklearn.neighbors.kde import KernelDensity
import sys
import time


class AbstractCoordProducer(object):

    def __call__(self):
        raise NotImplementedError() 

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs["class"])
        return the_class.from_hdf5(grp)


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


class CoordProducerResults(object):

    def __init__(self, coords, tnt_results):
        self.coords = coords
        self.tnt_results = tnt_results

    @classmethod
    def from_hdf5(cls, grp):
        coord_strings = util.load_string_list(dset_name="coords",
                                              grp=grp)  
        coords = [SeqletCoordinates.from_string(x) for x in coord_strings] 
        tnt_results = AbstractThresholdingResults.from_hdf5(
                                grp["tnt_results"])
        return CoordProducerResults(coords=coords,
                                    tnt_results=tnt_results)

    def save_hdf5(self, grp):
        util.save_string_list(
            string_list=[str(x) for x in self.coords],
            dset_name="coords",
            grp=grp) 
        self.tnt_results.save_hdf5(
              grp=grp.create_group("tnt_results"))


def get_simple_window_sum_function(window_size):
    def window_sum_function(arrs):
        to_return = []
        for arr in arrs:
            current_sum = np.sum(arr[0:window_size])
            arr_running_sum = [current_sum]
            for i in range(0,(len(arr)-window_size)):
                current_sum = (current_sum +
                               arr[i+window_size] - arr[i])
                arr_running_sum.append(current_sum)
            to_return.append(np.array(arr_running_sum))
        return to_return
    return window_sum_function


class GenerateNullDist(object):

    def __call__(self, score_track):
        raise NotImplementedError() 


class FlipSignNullDist(GenerateNullDist):

    def __init__(self, num_to_samp, seed):
        self.num_to_samp = num_to_samp
        self.seed = seed
        self.rng = np.random.RandomState()

    @classmethod
    def from_hdf5(cls, grp):
        raise NotImplementedError()

    def save_hdf(cls, grp):
        raise NotImplementedError()

    def __call__(self, score_track):
        self.rng.seed(self.seed)
        null_tracks = []
        for i in range(self.num_to_samp):
            random_track = score_track[
             int(self.rng.randint(0,len(score_track)))]
            track_with_sign_flips = [
             x*(1 if self.rng.uniform() > 0.5 else -1)
             for x in random_track]
            null_tracks.append(track_with_sign_flips)
        return null_tracks


class FixedWindowAroundChunks(AbstractCoordProducer):

    def __init__(self, sliding,
                       flank,
                       thresholding_function,
                       suppress, #flanks to suppress
                       null_dist_gen=FlipSignNullDist(num_to_samp=10000,
                                                      seed=1234),
                       max_seqlets_total=None,
                       progress_update=5000,
                       verbose=True):
        self.sliding = sliding
        self.flank = flank
        self.suppress = suppress
        self.thresholding_function = thresholding_function
        self.null_dist_gen = null_dist_gen
        self.max_seqlets_total = None
        self.progress_update = progress_update
        self.verbose = verbose

    @classmethod
    def from_hdf5(cls, grp):
        sliding = grp.attrs["sliding"]
        flank = grp.attrs["flank"]
        suppress = grp.attrs["suppress"] 
        thresholding_function = AbstractThresholdingFunction.from_hdf5(
                                 grp["thresholding_function"])
        if ("max_seqlets_total" in grp.attrs):
            max_seqlets_total = grp.attrs["max_seqlets_total"]
        else:
            max_seqlets_total = None
        #TODO: load min_seqlets feature
        progress_update = grp.attrs["progress_update"]
        verbose = grp.attrs["verbose"]
        return cls(sliding=sliding, flank=flank, suppress=suppress,
                    thresholding_function=thresholding_function,
                    max_seqlets_total=max_seqlets_total,
                    progress_update=progress_update, verbose=verbose) 

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["sliding"] = self.sliding
        grp.attrs["flank"] = self.flank
        grp.attrs["suppress"] = self.suppress
        self.thresholding_function.save_hdf5(
              grp.create_group("thresholding_function"))
        #TODO: save min_seqlets feature
        if (self.max_seqlets_total is not None):
            grp.attrs["max_seqlets_total"] = self.max_seqlets_total 
        grp.attrs["progress_update"] = self.progress_update
        grp.attrs["verbose"] = self.verbose

    def __call__(self, score_track, tnt_results=None):
    
        # score_track now can be a list of arrays,
        assert all([len(x.shape)==1 for x in score_track]) 
        window_sum_function = get_simple_window_sum_function(self.sliding)

        if (self.verbose):
            print("Generating null dist")
        null_score_track = self.null_dist_gen(score_track=score_track) 

        if (self.verbose):
            print("Computing windowed sums")
            sys.stdout.flush()
        original_summed_score_track = window_sum_function(arrs=score_track) 
        null_summed_score_track = window_sum_function(arrs=null_score_track) 

        if (tnt_results is None):
            if (self.verbose):
                print("Computing threshold")
                sys.stdout.flush()
            tnt_results = self.thresholding_function(
                                    values=np.concatenate(
                                     original_summed_score_track,axis=0),
                                    null_dist=null_summed_score_track)
        neg_threshold = tnt_results.neg_threshold
        pos_threshold = tnt_results.pos_threshold

        summed_score_track = [x.copy() for x in original_summed_score_track]

        #if a position is less than the threshold, set it to -np.inf
        summed_score_track = [
            np.array([np.abs(y) if (y >= pos_threshold
                            or y <= neg_threshold)
                           else -np.inf for y in x])
            for x in summed_score_track]

        coords = []
        for example_idx,single_score_track in enumerate(summed_score_track):
            #set the stuff near the flanks to -np.inf so that we
            # don't pick it up during argmax
            single_score_track[0:self.flank] = -np.inf
            single_score_track[len(single_score_track)-(self.flank):
                               len(single_score_track)] = -np.inf
            while True:
                argmax = np.argmax(single_score_track,axis=0)
                max_val = single_score_track[argmax]

                #bail if exhausted everything that passed the threshold
                #and was not suppressed
                if (max_val == -np.inf):
                    break

                #need to be able to expand without going off the edge
                if ((argmax >= self.flank) and
                    (argmax < (len(single_score_track)-self.flank))): 

                    coord = SeqletCoordsFWAP(
                        example_idx=example_idx,
                        start=argmax-self.flank,
                        end=argmax+self.sliding+self.flank,
                        score=original_summed_score_track[example_idx][argmax]) 
                    coords.append(coord)
                else:
                    assert False,\
                     ("This shouldn't happen because I set stuff near the"
                      "border to -np.inf early on")
                #suppress the chunks within +- self.suppress
                left_supp_idx = int(max(np.floor(argmax+0.5-self.suppress),
                                                 0))
                right_supp_idx = int(min(np.ceil(argmax+0.5+self.suppress),
                                     len(single_score_track)))
                single_score_track[left_supp_idx:right_supp_idx] = -np.inf 

        if (self.verbose):
            print("Got "+str(len(coords))+" coords")
            sys.stdout.flush()

        if ((self.max_seqlets_total is not None) and
            len(coords) > self.max_seqlets_total):
            if (self.verbose):
                print("Limiting to top "+str(self.max_seqlets_total))
                sys.stdout.flush()
            coords = sorted(coords, key=lambda x: -np.abs(x.score))\
                               [:self.max_seqlets_total]
        return CoordProducerResults(
                    coords=coords,
                    tnt_results=tnt_results) 
