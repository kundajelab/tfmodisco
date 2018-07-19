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

    def __init__(self, neg_threshold, neg_threshold_cdf, neg_b,
                       pos_threshold, pos_threshold_cdf, pos_b):
        self.neg_threshold = neg_threshold
        self.neg_threshold_cdf = neg_threshold_cdf
        self.neg_b = neg_b
        self.pos_threshold = pos_threshold
        self.pos_threshold_cdf = pos_threshold_cdf
        self.pos_b = pos_b

    def save_hdf5(self, grp):
        grp.attrs['neg_threshold'] = self.neg_threshold
        grp.attrs['neg_b'] = self.neg_b 
        grp.attrs['pos_threshold'] = self.pos_threshold
        grp.attrs['pos_b'] = self.pos_b 


class LaplaceThreshold(object):
    count = 0
    def __init__(self, target_fdr, verbose):
        assert (target_fdr > 0.0 and target_fdr < 1.0)
        self.target_fdr = target_fdr
        self.verbose = verbose

    def __call__(self, values):

        pos_values = np.array(sorted(values[values >= 0.0]))
        neg_values = np.array(sorted(values[values < 0.0],
                                     key=lambda x: -x))

        #We assume that the null is governed by a laplace, because
        #that's what I (Av Shrikumar) have personally observed
        #But we calculate a different laplace distribution for
        # positive and negative values, in case they are
        # distributed slightly differently
        #estimate b using the percentile
        #for x below 0:
        #cdf = 0.5*exp(x/b)
        #b = x/(log(cdf/0.5))
        neg_laplace_b = np.percentile(neg_values, 95)/(np.log(0.95))
        pos_laplace_b = (-np.percentile(pos_values, 5))/(np.log(0.95))

        #for the pos and neg, compute the expected number above a
        #particular threshold based on the total number of examples,
        #and use this to estimate the fdr
        #for pos_null_above, we estimate the total num of examples
        #as 2*len(pos_values)
        pos_fdrs = (len(pos_values)*(np.exp(-pos_values/pos_laplace_b)))/(
                    len(pos_values)-np.arange(len(pos_values)))
        pos_fdrs = np.minimum(pos_fdrs, 1.0)
        neg_fdrs = (len(neg_values)*(np.exp(neg_values/neg_laplace_b)))/(
                    len(neg_values)-np.arange(len(neg_values)))
        neg_fdrs = np.minimum(neg_fdrs, 1.0)

        pos_fdrs_passing_thresh = [x for x in zip(pos_values, pos_fdrs)
                                   if x[1] <= self.target_fdr]
        neg_fdrs_passing_thresh = [x for x in zip(neg_values, neg_fdrs)
                                   if x[1] <= self.target_fdr]
        if (len(pos_fdrs_passing_thresh) > 0):
            pos_threshold, pos_thresh_fdr = pos_fdrs_passing_thresh[0]
        else:
            pos_threshold, pos_thresh_fdr = pos_values[-1], pos_fdrs[-1]
            pos_threshold += 0.0000001
        if (len(neg_fdrs_passing_thresh) > 0):
            neg_threshold, neg_thresh_fdr = neg_fdrs_passing_thresh[0]
            neg_threshold = neg_threshold - 0.0000001
        else:
            neg_threshold, neg_thresh_fdr = neg_values[-1], neg_fdrs[-1]

        pos_threshold_cdf = 1-np.exp(-pos_threshold/pos_laplace_b) 
        neg_threshold_cdf = 1-np.exp(neg_threshold/neg_laplace_b) 
        #neg_threshold = np.log((1-self.threshold_cdf)*2)*neg_laplace_b
        #pos_threshold = -np.log((1-self.threshold_cdf)*2)*pos_laplace_b

        #plot the result
        if (self.verbose):
            print("Thresholds:",neg_threshold,"and",pos_threshold)
            print("CDFs:",neg_threshold_cdf,"and",pos_threshold_cdf)
            print("Est. FDRs:",neg_thresh_fdr,"and",pos_thresh_fdr)
            neg_linspace = np.linspace(np.min(values), 0, 100)
            pos_linspace = np.linspace(0, np.max(values), 100)
            neg_laplace_vals = (1/(2*neg_laplace_b))*np.exp(
                            -np.abs(neg_linspace)/neg_laplace_b)
            pos_laplace_vals = (1/(2*pos_laplace_b))*np.exp(
                            -np.abs(pos_linspace)/neg_laplace_b)
            from matplotlib import pyplot as plt
            plt.figure()
            hist, _, _ = plt.hist(values, bins=100)
            plt.plot(neg_linspace,
                     neg_laplace_vals/(
                      np.max(neg_laplace_vals))*np.max(hist))
            plt.plot(pos_linspace,
                     pos_laplace_vals/(
                      np.max(pos_laplace_vals))*np.max(hist))
            plt.plot([neg_threshold, neg_threshold],
                     [0, np.max(hist)])
            plt.plot([pos_threshold, pos_threshold],
                     [0, np.max(hist)])
            if plt.isinteractive():
                plt.show()
            else:
                import os, errno
                try:
                    os.makedirs("figures")
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                fname = "figures/laplace_" + str(LaplaceThreshold.count) + ".png"
                plt.savefig(fname)
                print("saving plot to " + fname)
                LaplaceThreshold.count += 1

        return LaplaceThresholdingResults(
                neg_threshold=neg_threshold,
                neg_threshold_cdf=neg_threshold_cdf,
                neg_b=neg_laplace_b,
                pos_threshold=pos_threshold,
                pos_threshold_cdf=pos_threshold_cdf,
                pos_b=pos_laplace_b)


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
                            target_fdr=0.05,
                            verbose=True),
                       max_seqlets_total=None,
                       progress_update=5000,
                       verbose=True):
        self.sliding = sliding
        self.flank = flank
        if (suppress is None):
            suppress = int(0.5*sliding) + flank
        self.suppress = suppress
        self.thresholding_function = thresholding_function
        self.max_seqlets_total = None
        self.progress_update = progress_update
        self.verbose = verbose

    def __call__(self, score_track):
     
        # score_track now can be a list of arrays, comment out the assert for now
        # assert len(score_track.shape)==2 
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
        neg_threshold = thresholding_results.neg_threshold
        pos_threshold = thresholding_results.pos_threshold

        summed_score_track = [x.copy() for x in original_summed_score_track]

        #if a position is less than the threshold, set it to -np.inf
        summed_score_track = [
            np.array([np.abs(y) if (y >= pos_threshold
                            or y <= neg_threshold)
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

        if ((self.max_seqlets_total is not None) and
            len(coords) > self.max_seqlets_total):
            if (self.verbose):
                print("Limiting to top "+str(self.max_seqlets_total))
                sys.stdout.flush()
            coords = sorted(coords, key=lambda x: -np.abs(x.score))\
                               [:self.max_seqlets_total]
        return CoordProducerResults(
                    coords=coords,
                    thresholding_results=thresholding_results) 
