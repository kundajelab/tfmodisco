from __future__ import division, print_function, absolute_import
from .core import SeqletCoordinates
from modisco import util
import numpy as np
from collections import defaultdict
import itertools
from sklearn.neighbors.kde import KernelDensity
import sys


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


class AbstractThresholdingResults(object):

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs['class'])
        return the_class.from_hdf5(grp) 


class LaplaceThresholdingResults(AbstractThresholdingResults):

    def __init__(self, neg_threshold, neg_threshold_cdf, neg_b,
                       pos_threshold, pos_threshold_cdf, pos_b, mu):
        self.neg_threshold = neg_threshold
        self.neg_threshold_cdf = neg_threshold_cdf
        self.neg_b = neg_b
        self.pos_threshold = pos_threshold
        self.pos_threshold_cdf = pos_threshold_cdf
        self.pos_b = pos_b
        self.mu = mu

    @classmethod
    def from_hdf5(cls, grp):
        mu = grp.attrs['mu'] 
        neg_threshold = grp.attrs['neg_threshold']
        neg_threshold_cdf = grp.attrs['neg_threshold_cdf']
        neg_b = grp.attrs['neg_b']
        pos_threshold = grp.attrs['pos_threshold']
        pos_threshold_cdf = grp.attrs['pos_threshold_cdf']
        pos_b = grp.attrs['pos_b']
        return cls(neg_threshold=neg_threshold,
                   neg_threshold_cdf=neg_threshold_cdf,
                   neg_b=neg_b,
                   pos_threshold=pos_threshold,
                   pos_threshold_cdf=pos_threshold_cdf,
                   pos_b=pos_b,
                   mu=mu)

    def save_hdf5(self, grp):
        grp.attrs['class'] = type(self).__name__
        grp.attrs['mu'] = self.mu
        grp.attrs['neg_threshold'] = self.neg_threshold
        grp.attrs['neg_threshold_cdf'] = self.neg_threshold_cdf
        grp.attrs['neg_b'] = self.neg_b 
        grp.attrs['pos_threshold'] = self.pos_threshold
        grp.attrs['pos_threshold_cdf'] = self.pos_threshold_cdf
        grp.attrs['pos_b'] = self.pos_b 


class AbstractThresholdingFunction(object):

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs["class"]) 
        return the_class.from_hdf5(grp) 
 

class LaplaceThreshold(AbstractThresholdingFunction):
    count = 0
    def __init__(self, target_fdr, min_seqlets, verbose):
        assert (target_fdr > 0.0 and target_fdr < 1.0)
        self.target_fdr = target_fdr
        self.verbose = verbose
        self.min_seqlets = min_seqlets

    @classmethod
    def from_hdf5(cls, grp):
        target_fdr = grp.attrs["target_fdr"]
        min_seqlets = grp.attrs["min_seqlets"]
        verbose = grp.attrs["verbose"]
        return cls(target_fdr=target_fdr,
                   min_seqlets=min_seqlets, verbose=verbose)

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["target_fdr"] = self.target_fdr
        grp.attrs["min_seqlets"] = self.min_seqlets
        grp.attrs["verbose"] = self.verbose 

    def __call__(self, values):

        # first estimate mu, using two level histogram to get to 1e-6
        hist1, bin_edges1 = np.histogram(values, bins=1000)
        peak1 = np.argmax(hist1)
        l_edge = bin_edges1[peak1]
        r_edge = bin_edges1[peak1+1]
        top_values = values[ (l_edge < values) & (values < r_edge) ]

        hist2, bin_edges2 = np.histogram(top_values, bins=1000)
        peak2 = np.argmax(hist2)
        l_edge = bin_edges2[peak2]
        r_edge = bin_edges2[peak2+1]
        mu = (l_edge + r_edge) / 2
        print("peak(mu)=", mu)

        pos_values = np.array(sorted(values[values > mu] - mu))
        neg_values = np.array(sorted(values[values < mu] - mu, key=lambda x: -x))

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

        if (self.min_seqlets is not None):
            num_pos_passing = np.sum(pos_values > pos_threshold)
            num_neg_passing = np.sum(neg_values < neg_threshold)
            if (num_pos_passing + num_neg_passing < self.min_seqlets):
                #manually adjust the threshold
                shifted_values = values - mu
                values_sorted_by_abs = sorted(np.abs(shifted_values), key=lambda x: -x)
                abs_threshold = values_sorted_by_abs[self.min_seqlets-1]
                if (self.verbose):
                    print("Manually adjusting thresholds to get desired num seqlets")
                pos_threshold = abs_threshold
                neg_threshold = -abs_threshold
        
        pos_threshold_cdf = 1-np.exp(-pos_threshold/pos_laplace_b)
        neg_threshold_cdf = 1-np.exp(neg_threshold/neg_laplace_b)
        #neg_threshold = np.log((1-self.threshold_cdf)*2)*neg_laplace_b
        #pos_threshold = -np.log((1-self.threshold_cdf)*2)*pos_laplace_b
        
        neg_threshold += mu
        pos_threshold += mu
        neg_threshold = min(neg_threshold, 0)
        pos_threshold = max(pos_threshold, 0)
        
        

        #plot the result
        if (self.verbose):
            print("Mu: %e +/- %e" % (mu, (r_edge-l_edge)/2))
            print("Lablace_b:",neg_laplace_b,"and",pos_laplace_b)
            print("Thresholds:",neg_threshold,"and",pos_threshold)
            print("#fdrs pass:",len(neg_fdrs_passing_thresh),"and", len(pos_fdrs_passing_thresh))
            print("CDFs:",neg_threshold_cdf,"and",pos_threshold_cdf)
            print("Est. FDRs:",neg_thresh_fdr,"and",pos_thresh_fdr)
            neg_linspace = np.linspace(np.min(values), mu, 100)
            pos_linspace = np.linspace(mu, np.max(values), 100)
            neg_laplace_vals = (1/(2*neg_laplace_b))*np.exp(
                            -np.abs(neg_linspace-mu)/neg_laplace_b)
            pos_laplace_vals = (1/(2*pos_laplace_b))*np.exp(
                            -np.abs(pos_linspace-mu)/pos_laplace_b)
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
                pos_b=pos_laplace_b,
                mu = mu)


class CoordProducerResults(object):

    def __init__(self, coords, thresholding_results):
        self.coords = coords
        self.thresholding_results = thresholding_results

    @classmethod
    def from_hdf5(cls, grp):
        coord_strings = util.load_string_list(dset_name="coords",
                                              grp=grp)  
        coords = [SeqletCoordinates.from_string(x) for x in coord_strings] 
        thresholding_results = AbstractThresholdingResults.from_hdf5(
                                grp["thresholding_results"])
        return CoordProducerResults(coords=coords,
                                    thresholding_results=thresholding_results)

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
            for i in range(0,(len(arr)-window_size)):
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
                            min_seqlets=500,
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
        if (self.max_seqlets_total is not None):
            grp.attrs["max_seqlets_total"] = self.max_seqlets_total 
        grp.attrs["progress_update"] = self.progress_update
        grp.attrs["verbose"] = self.verbose

    def __call__(self, score_track, thresholding_results=None):
     
        # score_track now can be a list of arrays, comment out the assert for now
        # assert len(score_track.shape)==2 
        window_sum_function = get_simple_window_sum_function(self.sliding)

        if (self.verbose):
            print("Computing windowed sums")
            sys.stdout.flush()
        original_summed_score_track = window_sum_function(arrs=score_track) 
        if (thresholding_results is None):
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
                    thresholding_results=thresholding_results) 
