from __future__ import division, print_function, absolute_import
from .core import SeqletCoordinates
from modisco import backend as B 
from modisco import util
import numpy as np
from collections import defaultdict
import itertools
from sklearn.neighbors.kde import KernelDensity
import sys


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


class LaplaceThresholdingResults(object):

    def __init__(self, left_threshold, left_b, right_threshold, right_b):
        self.left_threshold = left_threshold
        self.left_b = left_b
        self.right_threshold = right_threshold
        self.right_b = right_b

    def save_hdf5(self, grp):
        grp.attrs['left_threshold'] = self.left_threshold
        grp.attrs['left_b'] = self.left_b 
        grp.attrs['right_threshold'] = self.right_threshold
        grp.attrs['right_b'] = self.right_b 


class LaplaceThreshold(object):

    def __init__(self, threshold_cdf, verbose):
        assert (threshold_cdf > 0.5 and threshold_cdf < 1.0)
        self.threshold_cdf = threshold_cdf
        self.verbose = verbose

    def __call__(self, values):

        #We assume that the null is governed by a laplace, because
        #that's what I (Av Shrikumar) have personally observed
        #But we calculate a different laplace distribution for
        # positive and negative values, in case they are
        # distributed slightly differently
        #80th percentile of things below 0 is 40th percentile of errything
        left_forty_perc = np.percentile(values[values < 0.0], 80)
        right_sixty_perc = np.percentile(values[values > 0.0], 20)
        #estimate b using the percentile
        #for x below 0:
        #cdf = 0.5*exp(x/b)
        #b = x/(log(cdf/0.5))
        left_laplace_b = left_forty_perc/(np.log(0.8))
        right_laplace_b = (-right_sixty_perc)/(np.log(0.8))
        #solve for x given the target threshold percentile
        #(assumes the target threshold percentile is > 0.5)
        left_threshold = np.log((1-self.threshold_cdf)*2)*left_laplace_b
        right_threshold = -np.log((1-self.threshold_cdf)*2)*right_laplace_b

        #plot the result
        if (self.verbose):
            left_linspace = np.linspace(np.min(values), 0, 100)
            right_linspace = np.linspace(0, np.max(values), 100)
            left_laplace_vals = (1/(2*left_laplace_b))*np.exp(
                            -np.abs(left_linspace)/left_laplace_b)
            right_laplace_vals = (1/(2*right_laplace_b))*np.exp(
                            -np.abs(right_linspace)/left_laplace_b)
            from matplotlib import pyplot as plt
            hist, _, _ = plt.hist(values, bins=100)
            plt.plot(left_linspace,
                     left_laplace_vals/(
                      np.max(left_laplace_vals))*np.max(hist))
            plt.plot(right_linspace,
                     right_laplace_vals/(
                      np.max(right_laplace_vals))*np.max(hist))
            plt.plot([left_threshold, left_threshold],
                     [0, np.max(hist)])
            plt.plot([right_threshold, right_threshold],
                     [0, np.max(hist)])
            plt.show()

        return LaplaceThresholdingResults(
                left_threshold=left_threshold,
                left_b=left_laplace_b,
                right_threshold=right_threshold,
                right_b=right_laplace_b)


class CoordProducerResults(object):

    def __init__(self, coords, thresholding_results):
        self.coords = coords
        self.thresholding_results = thresholding_results

    def save_hdf5(self, grp):
        util.save_string_list(
            string_list=[str(x) for x in self.coords],
            dset_name="coords",
            grp=grp) 
        self.thresholding_results.save_hdf5(
              grp=grp.create_group("thresholding_results"))


def get_simple_window_sum_function(window_size):
    def window_sum_function(arrs):
        to_return = []
        for arr in arrs:
            current_sum = np.sum(arr[0:window_size])
            arr_running_sum = [current_sum]
            for i in range(0,len(arr)-window_size):
                current_sum = (current_sum +
                               arr[i+window_size] - arr[i])
                arr_running_sum.append(current_sum)
            to_return.append(np.array(arr_running_sum))
        return to_return
    return window_sum_function


class FixedWindowAroundChunks(AbstractCoordProducer):

    def __init__(self, sliding=11,
                       flank=10,
                       suppress=None,
                       thresholding_function=LaplaceThreshold(
                            threshold_cdf=0.99,
                            verbose=True),
                       max_seqlets_total=20000,
                       progress_update=5000,
                       verbose=True):
        self.sliding = sliding
        self.flank = flank
        if (suppress is None):
            suppress = int(0.5*sliding) + flank
        self.suppress = suppress
        self.thresholding_function = thresholding_function
        self.max_seqlets_total = max_seqlets_total
        self.progress_update = progress_update
        self.verbose = verbose

    def __call__(self, score_track):
     
        assert len(score_track.shape)==2 
        window_sum_function = get_simple_window_sum_function(self.sliding)

        if (self.verbose):
            print("Computing windowed sums")
            sys.stdout.flush()
        original_summed_score_track = window_sum_function(arrs=score_track) 
        if (self.verbose):
            print("Computing threshold")
            sys.stdout.flush()
        thresholding_results = self.thresholding_function(
                                np.concatenate(original_summed_score_track,
                                               axis=0))
        left_threshold = thresholding_results.left_threshold
        right_threshold = thresholding_results.right_threshold
        if (self.verbose):
            print("Computed thresholds "+str(left_threshold)
                  +" and "+str(right_threshold))
            sys.stdout.flush()

        summed_score_track = [x.copy() for x in original_summed_score_track]

        #if a position is less than the threshold, set it to -np.inf
        summed_score_track = [
            np.array([np.abs(y) if (y >= right_threshold
                            or y <= left_threshold)
                           else -np.inf for y in x])
            for x in summed_score_track]

        coords = []
        for example_idx,single_score_track in enumerate(summed_score_track):
            while True:
                argmax = np.argmax(single_score_track,axis=0)
                max_val = single_score_track[argmax]
                #bail if exhausted everything that passed the threshold
                #and was not suppressed
                if (max_val == -np.inf):
                    break
                #need to be able to expand without going off the edge
                if ((argmax >= self.flank) and
                    (argmax <= (len(single_score_track)
                                -(self.sliding+self.flank)))): 
                    coord = SeqletCoordsFWAP(
                        example_idx=example_idx,
                        start=argmax-self.flank,
                        end=argmax+self.sliding+self.flank,
                        score=original_summed_score_track[example_idx][argmax]) 
                    coords.append(coord)
                #suppress the chunks within +- self.suppress
                left_supp_idx = int(max(np.floor(argmax+0.5-self.suppress),0))
                right_supp_idx = int(min(np.ceil(argmax+0.5+self.suppress),
                                     len(single_score_track)))
                single_score_track[left_supp_idx:right_supp_idx] = -np.inf 

        if (self.verbose):
            print("Got "+str(len(coords))+" coords")
            sys.stdout.flush()

        if (len(coords) > self.max_seqlets_total):
            if (self.verbose):
                print("Limiting to top "+str(self.max_seqlets_total))
                sys.stdout.flush()
            coords = sorted(coords, key=lambda x: -np.abs(x.score))\
                               [:self.max_seqlets_total]
        return CoordProducerResults(
                    coords=coords,
                    thresholding_results=thresholding_results) 
