from __future__ import division, print_function, absolute_import
from .core import SeqletCoordinates
from modisco import util
import numpy as np
from collections import defaultdict, Counter
import itertools
from sklearn.neighbors.kde import KernelDensity
import sys
import time
from .value_provider import (
    AbstractValTransformer, AbsPercentileValTransformer,
    SignedPercentileValTransformer, Gamma2xCdfMHalfValTransformer)
import scipy


class TransformAndThresholdResults(object):

    def __init__(self, neg_threshold,
                       transformed_neg_threshold,
                       pos_threshold,
                       transformed_pos_threshold,
                       val_transformer):
        #both 'transformed_neg_threshold' and 'transformed_pos_threshold'
        # should be positive, i.e. they should be relative to the
        # transformed distribution used to set the threshold, e.g. a
        # cdf value
        self.neg_threshold = neg_threshold
        self.transformed_neg_threshold = transformed_neg_threshold
        self.pos_threshold = pos_threshold
        self.transformed_pos_threshold = transformed_pos_threshold
        self.val_transformer = val_transformer

    def save_hdf5(self, grp):
        grp.attrs["neg_threshold"] = self.neg_threshold
        grp.attrs["transformed_neg_threshold"] = self.transformed_neg_threshold
        grp.attrs["pos_threshold"] = self.pos_threshold
        grp.attrs["transformed_pos_threshold"] = self.transformed_pos_threshold
        self.val_transformer.save_hdf5(grp.create_group("val_transformer"))

    @classmethod
    def from_hdf5(cls, grp):
        neg_threshold = grp.attrs['neg_threshold']
        transformed_neg_threshold = grp.attrs['transformed_neg_threshold']
        pos_threshold = grp.attrs['pos_threshold']
        transformed_pos_threshold = grp.attrs['transformed_pos_threshold']
        val_transformer = AbstractValTransformer.from_hdf5(
                           grp["val_transformer"]) 
        return cls(neg_threshold=neg_threshold,
                   transformed_neg_threshold=transformed_neg_threshold,
                   pos_threshold=pos_threshold,
                   transformed_pos_threshold=transformed_pos_threshold,
                   val_transformer=val_transformer) 


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
    def __init__(self, example_idx, start, end, score, score2=None):
        self.score = score 
        self.score2 = score2
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
        tnt_results = TransformAndThresholdResults.from_hdf5(
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
            cumsum = np.cumsum(arr)
            cumsum = np.array([0]+list(cumsum))
            to_return.append(cumsum[window_size:]-cumsum[:-window_size])
        return to_return
    return window_sum_function


class GenerateNullDist(object):

    def __call__(self, score_track):
        raise NotImplementedError() 


class TakeSign(GenerateNullDist):

    @classmethod
    def from_hdf5(cls, grp):
        raise NotImplementedError()

    def save_hdf(cls, grp):
        raise NotImplementedError()

    def __call__(self, score_track):
        null_tracks = [np.sign(x) for x in score_track]
        return null_tracks


class TakeAbs(GenerateNullDist):

    @classmethod
    def from_hdf5(cls, grp):
        raise NotImplementedError()

    def save_hdf(cls, grp):
        raise NotImplementedError()

    def __call__(self, score_track):
        null_tracks = [np.abs(x) for x in score_track]
        return null_tracks


class LogPercentileGammaNullDist(GenerateNullDist):

    def __init__(self, num_to_samp, random_seed=1234):
        self.num_to_samp = num_to_samp
        self.random_seed = random_seed

    @classmethod
    def from_hdf5(cls, grp):
        num_to_samp = grp.attrs["num_to_samp"]
        return cls(num_to_samp=num_to_samp)

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["num_to_samp"] = self.num_to_samp

    def __call__(self, score_track, windowsize, original_summed_score_track):

        #original_summed_score_track is supplied to avoid recomputing it 
        window_sum_function = get_simple_window_sum_function(windowsize)
        if (original_summed_score_track is not None):
            original_summed_score_track = window_sum_function(arrs=score_track) 

        rng = np.random.RandomState(self.random_seed)
        return rng.gamma(shape=windowsize, size=self.num_to_samp)


class LaplaceNullDist(GenerateNullDist):

    def __init__(self, num_to_samp, verbose=True,
                       percentiles_to_use=[5*(x+1) for x in range(19)],
                       random_seed=1234):
        self.num_to_samp = num_to_samp
        self.verbose = verbose
        self.percentiles_to_use = np.array(percentiles_to_use)
        self.random_seed = random_seed
        self.rng = np.random.RandomState()

    @classmethod
    def from_hdf5(cls, grp):
        num_to_samp = grp.attrs["num_to_samp"]
        verbose = grp.attrs["verbose"]
        percentiles_to_use = np.array(grp["percentiles_to_use"][:])
        return cls(num_to_samp=num_to_samp, verbose=verbose)

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["num_to_samp"] = self.num_to_samp
        grp.attrs["verbose"] = self.verbose 
        grp.create_dataset('percentiles_to_use',
                           data=self.percentiles_to_use)

    def __call__(self, score_track, windowsize, original_summed_score_track):

        #original_summed_score_track is supplied to avoid recomputing it 
        window_sum_function = get_simple_window_sum_function(windowsize)
        if (original_summed_score_track is not None):
            original_summed_score_track = window_sum_function(arrs=score_track) 

        values = np.concatenate(original_summed_score_track, axis=0)
       
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
        if (self.verbose):
            print("peak(mu)=", mu)

        pos_values = [x for x in values if x >= mu]
        neg_values = [x for x in values if x <= mu] 
        #for an exponential distribution:
        # cdf = 1 - exp(-lambda*x)
        # exp(-lambda*x) = 1-cdf
        # -lambda*x = log(1-cdf)
        # lambda = -log(1-cdf)/x
        # x = -log(1-cdf)/lambda
        #Take the most aggressive lambda over all percentiles
        pos_laplace_lambda = np.max(
            -np.log(1-(self.percentiles_to_use/100.0))/
            (np.percentile(a=pos_values, q=self.percentiles_to_use)-mu))
        neg_laplace_lambda = np.max(
            -np.log(1-(self.percentiles_to_use/100.0))/
            (np.abs(np.percentile(a=neg_values,
                                  q=100-self.percentiles_to_use)-mu)))

        self.rng.seed(self.random_seed)
        prob_pos = float(len(pos_values))/(len(pos_values)+len(neg_values)) 
        sampled_vals = []
        for i in range(self.num_to_samp):
            sign = 1 if (self.rng.uniform() < prob_pos) else -1
            if (sign == 1):
                sampled_cdf = self.rng.uniform()
                val = -np.log(1-sampled_cdf)/pos_laplace_lambda + mu 
            else:
                sampled_cdf = self.rng.uniform() 
                val = mu + np.log(1-sampled_cdf)/neg_laplace_lambda
            sampled_vals.append(val)
        return np.array(sampled_vals)
        

class FlipSignNullDist(GenerateNullDist):

    def __init__(self, num_seq_to_samp, shuffle_pos=False,
                       seed=1234, num_breaks=100,
                       lower_null_percentile=20,
                       upper_null_percentile=80):
        self.num_seq_to_samp = num_seq_to_samp
        self.shuffle_pos = shuffle_pos
        self.seed = seed
        self.rng = np.random.RandomState()
        self.num_breaks = num_breaks
        self.lower_null_percentile = lower_null_percentile
        self.upper_null_percentile = upper_null_percentile

    @classmethod
    def from_hdf5(cls, grp):
        raise NotImplementedError()

    def save_hdf(cls, grp):
        raise NotImplementedError()

    def __call__(self, score_track, windowsize, original_summed_score_track):
        #summed_score_track is supplied to avoid recomputing it 

        window_sum_function = get_simple_window_sum_function(windowsize)
        if (original_summed_score_track is not None):
            original_summed_score_track = window_sum_function(arrs=score_track) 

        all_orig_summed_scores = np.concatenate(
            original_summed_score_track, axis=0)
        pos_threshold = np.percentile(a=all_orig_summed_scores,
                                      q=self.upper_null_percentile)
        neg_threshold = np.percentile(a=all_orig_summed_scores,
                                      q=self.lower_null_percentile)

        #retain only the portions of the tracks that are under the
        # thresholds
        retained_track_portions = []
        num_pos_vals = 0
        num_neg_vals = 0
        for (single_score_track, single_summed_score_track)\
             in zip(score_track, original_summed_score_track):
            window_passing_track = [
                (1.0 if (x > neg_threshold and x < pos_threshold) else 0)
                for x in single_summed_score_track]
            padded_window_passing_track = [0.0]*int(windowsize-1) 
            padded_window_passing_track.extend(window_passing_track)
            padded_window_passing_track.extend([0.0]*int(windowsize-1))
            pos_in_passing_window = window_sum_function(
                                      [padded_window_passing_track])[0] 
            assert len(single_score_track)==len(pos_in_passing_window) 
            single_retained_track = []
            for (val, pos_passing) in zip(single_score_track,
                                          pos_in_passing_window):
                if (pos_passing > 0):
                    single_retained_track.append(val) 
                    num_pos_vals += (1 if val > 0 else 0)
                    num_neg_vals += (1 if val < 0 else 0)
            retained_track_portions.append(single_retained_track)

        print("Fraction of positions retained:",
              sum(len(x) for x in retained_track_portions)/
              sum(len(x) for x in score_track))
            
        prob_pos = num_pos_vals/float(num_pos_vals + num_neg_vals)
        self.rng.seed(self.seed)
        null_tracks = []
        for i in range(self.num_seq_to_samp):
            random_track = retained_track_portions[
             int(self.rng.randint(0,len(retained_track_portions)))]
            track_with_sign_flips = np.array([
             abs(x)*(1 if self.rng.uniform() < prob_pos else -1)
             for x in random_track])
            if (self.shuffle_pos):
                self.rng.shuffle(track_with_sign_flips) 
            null_tracks.append(track_with_sign_flips)
        return np.concatenate(window_sum_function(null_tracks), axis=0)


def flatten(list_of_arrs):
    to_return = []
    for arr in list_of_arrs:
        to_return.extend(arr) 
    return np.array(to_return)


def per_sequence_zscore_log_percentile_transform(score_track, seed=1234):
    transformed_all_scores = []
    for scores_row in score_track:
        median = np.median(scores_row)
        mad = scipy.stats.median_absolute_deviation(scores_row) 
        transformed_all_scores.append(
         -np.log((1-(scipy.stats.norm.cdf(
                     x=scores_row, loc=median, scale=mad))) + 1e-7 ))
    return transformed_all_scores


def per_sequence_log_percentile_transform(score_track, seed=1234):
    transformed_all_scores = []
    for scores_row in score_track:
        sorted_scores = sorted(scores_row)
        #the +1 is to avoid log(0) issues
        transformed_all_scores.append(
         -np.log(1-(np.searchsorted(a=sorted_scores, v=scores_row)/
                     len(sorted_scores))))
    return transformed_all_scores


def log_percentile_transform(max_num_to_use_for_percentile,
                             score_track, seed=1234):
    all_scores = flatten(list_of_arrs=score_track) 
    if len(all_scores) > max_num_to_use_for_percentile:
        all_scores = np.random.randomstate(seed).choice(
                        a=all_scores,
                        size=max_num_to_use_for_percentile,
                        replace=false)
    sorted_all_scores = sorted(all_scores)
    del all_scores
    transformed_all_scores = []
    for scores_row in score_track:
        #the +1 is to avoid log(0) issues
        transformed_all_scores.append(
         -np.log((1-(np.searchsorted(a=sorted_all_scores, v=scores_row)/
                     len(sorted_all_scores)))))
    return transformed_all_scores


#class PercentileBasedWindows(AbstractCoordProducer):
#    count = 0
#    def __init__(self, sliding, target_fdr, max_seqlets_total, verbose=True,
#                       plot_save_dir="figures",
#                       max_num_to_use_for_percentile=50000,
#                       seed=1234):
#        self.sliding = sliding
#        self.target_fdr = target_fdr
#        self.max_seqlets_total = max_seqlets_total
#        self.verbose = verbose
#        self.plot_save_dir = plot_save_dir
#        self.max_num_to_use_for_percentile = max_num_to_use_for_percentile
#        self.seed = seed
#
#    def __init__(self, sliding,
#                       flank,
#                       suppress, #flanks to suppress
#                       target_fdr,
#                       max_seqlets_total=None,
#                       progress_update=5000,
#                       verbose=True,
#                       plot_save_dir="figures",
#                       max_num_to_use_for_percentile=50000,
#                       seed=1234):
#        self.sliding = sliding
#        self.flank = flank
#        self.suppress = suppress
#        self.target_fdr = target_fdr
#        self.max_seqlets_total = None
#        self.progress_update = progress_update
#        self.verbose = verbose
#        self.plot_save_dir = plot_save_dir
#        self.max_num_to_use_for_percentile = max_num_to_use_for_percentile
#        self.seed = seed
#
#    def log_percentile_transform(self, score_track):
#        all_scores = flatten(list_of_arrs=score_track) 
#        if len(all_scores) > self.max_num_to_use_for_percentile:
#            all_scores = np.random.RandomState(self.seed).choice(
#                            a=all_scores,
#                            size=self.max_num_to_use_for_percentile,
#                            replace=False)
#        sorted_all_scores = sorted(all_scores)
#        del all_scores
#        transformed_all_scores = []
#        for scores_row in score_track:
#            #the +1 is to avoid log(0) issues
#            transformed_all_scores.append(
#             -np.log((np.searchsorted(a=sorted_all_scores, v=scores_row)+1)/
#                     len(sorted_all_scores)))
#        return transformed_all_scores
#
#    def __call__(self, score_track, tnt_results=None, **kwargs):
#    
#        assert all([len(x.shape)==1 for x in score_track]) 
#
#        #transform all the scores to percentiles 
#        log_percentile_scores = self.log_percentile_transform(
#                                      score_track=score_track)
#        del score_track
#
#        window_sum_function = get_simple_window_sum_function(self.sliding)
#
#        if (self.verbose):
#            print("Computing windowed sums on original")
#            sys.stdout.flush()
#        original_summed_logpercentile_track = window_sum_function(
#            arrs=log_percentile_scores) 
#
#        if (tnt_results is None):
#            all_logpercentile_windowsums = np.array(sorted(flatten(
#                original_summed_logpercentile_track)))
#
#            #Determine the window thresholds
#            # First, figure out what the cdf would look like
#            # Percentiles are uniformly distributed; -log of percentile is
#            # exponentially distributed. Sum of multiple exponential
#            # distributions is gamma distributed. Specifically, it would
#            # be a gamma distribution with shape parameter = self.sliding,
#            # and scale=1.
#           
#            #To compute the fdr, we can compare the empirical CDF to the
#            # expected CDF 
#            empirical_cdfs = (np.arange(len(all_logpercentile_windowsums))/
#                              len(all_logpercentile_windowsums))
#            val_transformer = Gamma2xCdfMHalfValTransformer(a=self.sliding) 
#            expected_cdfs = scipy.stats.gamma.cdf(
#                             x=all_logpercentile_windowsums, a=self.sliding)
#            pos_fdrs = (1-expected_cdfs)/(1-empirical_cdfs)
#            neg_fdrs = expected_cdfs/empirical_cdfs
#            
#            pos_threshold = ([x[1] for x in
#             zip(pos_fdrs, all_logpercentile_windowsums) if x[0]
#              <= self.target_fdr]+[all_logpercentile_windowsums[-1]])[0]
#            neg_threshold = ([all_logpercentile_windowsums[0]]
#             +[x[1] for x in zip(neg_fdrs, all_logpercentile_windowsums)
#             if x[0] <= self.target_fdr])[-1]
#
#            frac_passing_windows =(
#                sum(all_logpercentile_windowsums >= pos_threshold)
#                 + sum(all_logpercentile_windowsums <= neg_threshold))/float(
#                len(all_logpercentile_windowsums))
#
#            if (self.verbose):
#                print("Fraction passing threshold:",frac_passing_windows)
#                print("Thresholds from null dist were",
#                      neg_threshold," and ",pos_threshold)
#
#
#            from matplotlib import pyplot as plt
#            plt.figure()
#            hist, histbins, _ = plt.hist(all_logpercentile_windowsums,
#                                         bins=100, density=True)
#            plt.plot(histbins, scipy.stats.gamma.pdf(
#                                x=histbins, a=self.sliding)) 
#            #plt.plot(all_logpercentile_windowsums, empirical_cdfs)
#            #plt.plot(all_logpercentile_windowsums, expected_cdfs)
#            plt.plot([neg_threshold, neg_threshold], [0, max(hist)],
#                     color="red")
#            plt.plot([pos_threshold, pos_threshold], [0, max(hist)],
#                     color="red")
#            if plt.isinteractive():
#                plt.show()
#            else:
#                import os, errno
#                try:
#                    os.makedirs(self.plot_save_dir)
#                except OSError as e:
#                    if e.errno != errno.EEXIST:
#                        raise
#                fname = (self.plot_save_dir+"/scoredist_" +
#                         str(FixedWindowAroundChunks.count) + ".png")
#                plt.savefig(fname)
#                print("saving plot to " + fname)
#                FixedWindowAroundChunks.count += 1
#
#            tnt_results = TransformAndThresholdResults(
#                neg_threshold=neg_threshold,
#                transformed_neg_threshold=val_transformer(neg_threshold),
#                pos_threshold=pos_threshold,
#                transformed_pos_threshold=val_transformer(pos_threshold),
#                val_transformer=val_transformer)
#
#        neg_threshold = tnt_results.neg_threshold
#        pos_threshold = tnt_results.pos_threshold
#
#        summed_logpercentile_track = [np.array(x) for x in
#                                      original_summed_logpercentile_track]
#        #if a position is less than the threshold, set it to -np.inf
#        summed_logpercentile_track = [
#            np.array([np.abs(y) if (y > pos_threshold or y < neg_threshold)
#                      else -np.inf for y in x])
#            for x in summed_logpercentile_track]
#        #transformed_track
#        transformed_track = [
#            np.array([(val_transformer(y) if y > -np.inf else y)
#                       for y in x])
#                       for x in summed_logpercentile_track]
#        coords = []
#        for example_idx,single_score_track in enumerate(transformed_track):
#            #set the stuff near the flanks to -np.inf so that we
#            # don't pick it up during argmax
#            single_score_track[0:self.flank] = -np.inf
#            single_score_track[len(single_score_track)-(self.flank):
#                               len(single_score_track)] = -np.inf
#            while True:
#                argmax = np.argmax(single_score_track,axis=0)
#                max_val = single_score_track[argmax]
#
#                #bail if exhausted everything that passed the threshold
#                #and was not suppressed
#                if (max_val == -np.inf):
#                    break
#
#                #need to be able to expand without going off the edge
#                if ((argmax >= self.flank) and
#                    (argmax < (len(single_score_track)-self.flank))): 
#
#                    coord = SeqletCoordsFWAP(
#                        example_idx=example_idx,
#                        start=argmax-self.flank,
#                        end=argmax+self.sliding+self.flank,
#                        score=original_summed_logpercentile_track[
#                               example_idx][argmax],
#                        score2=max_val) #score2 for debugging 
#                    assert (coord.score2 <= self.target_fdr
#                            or (1-coord.score2) <= self.target_fdr)
#                    coords.append(coord)
#                else:
#                    assert False,\
#                     ("This shouldn't happen because I set stuff near the"
#                      "border to -np.inf early on")
#                #suppress the chunks within +- self.suppress
#                left_supp_idx = int(max(np.floor(argmax+0.5-self.suppress),
#                                                 0))
#                right_supp_idx = int(min(np.ceil(argmax+0.5+self.suppress),
#                                     len(single_score_track)))
#                single_score_track[left_supp_idx:right_supp_idx] = -np.inf 
#
#        if (self.verbose):
#            print("Got "+str(len(coords))+" coords")
#            sys.stdout.flush()
#
#        if ((self.max_seqlets_total is not None) and
#            len(coords) > self.max_seqlets_total):
#            if (self.verbose):
#                print("Limiting to top "+str(self.max_seqlets_total))
#                sys.stdout.flush()
#            coords = sorted(coords, key=lambda x: -np.abs(x.score))\
#                               [:self.max_seqlets_total]
#        return CoordProducerResults(
#                    coords=coords,
#                    tnt_results=tnt_results)   


class FixedWindowAroundChunks(AbstractCoordProducer):
    count = 0
    def __init__(self, sliding,
                       flank,
                       suppress, #flanks to suppress
                       target_fdr,
                       min_passing_windows_frac,
                       max_passing_windows_frac,
                       separate_pos_neg_thresholds=False,
                       max_seqlets_total=None,
                       progress_update=5000,
                       verbose=True,
                       plot_save_dir="figures"):
        self.sliding = sliding
        self.flank = flank
        self.suppress = suppress
        self.target_fdr = target_fdr
        assert max_passing_windows_frac >= min_passing_windows_frac
        self.min_passing_windows_frac = min_passing_windows_frac 
        self.max_passing_windows_frac = max_passing_windows_frac
        self.separate_pos_neg_thresholds = separate_pos_neg_thresholds
        self.max_seqlets_total = None
        self.progress_update = progress_update
        self.verbose = verbose
        self.plot_save_dir = plot_save_dir

    @classmethod
    def from_hdf5(cls, grp):
        sliding = grp.attrs["sliding"]
        flank = grp.attrs["flank"]
        suppress = grp.attrs["suppress"] 
        target_fdr = grp.attrs["target_fdr"]
        min_passing_windows_frac = grp.attrs["min_passing_windows_frac"]
        max_passing_windows_frac = grp.attrs["max_passing_windows_frac"]
        separate_pos_neg_thresholds = grp.attrs["separate_pos_neg_thresholds"]
        if ("max_seqlets_total" in grp.attrs):
            max_seqlets_total = grp.attrs["max_seqlets_total"]
        else:
            max_seqlets_total = None
        #TODO: load min_seqlets feature
        progress_update = grp.attrs["progress_update"]
        verbose = grp.attrs["verbose"]
        return cls(sliding=sliding, flank=flank, suppress=suppress,
                    target_fdr=target_fdr,
                    min_passing_windows_frac=min_passing_windows_frac,
                    max_passing_windows_frac=max_passing_windows_frac,
                    separate_pos_neg_thresholds=separate_pos_neg_thresholds,
                    max_seqlets_total=max_seqlets_total,
                    progress_update=progress_update, verbose=verbose) 

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["sliding"] = self.sliding
        grp.attrs["flank"] = self.flank
        grp.attrs["suppress"] = self.suppress
        grp.attrs["target_fdr"] = self.target_fdr
        grp.attrs["min_passing_windows_frac"] = self.min_passing_windows_frac
        grp.attrs["max_passing_windows_frac"] = self.max_passing_windows_frac
        grp.attrs["separate_pos_neg_thresholds"] =\
            self.separate_pos_neg_thresholds
        #TODO: save min_seqlets feature
        if (self.max_seqlets_total is not None):
            grp.attrs["max_seqlets_total"] = self.max_seqlets_total 
        grp.attrs["progress_update"] = self.progress_update
        grp.attrs["verbose"] = self.verbose

    def __call__(self, score_track, null_track, tnt_results=None):
    
        # score_track now can be a list of arrays,
        assert all([len(x.shape)==1 for x in score_track]) 
        window_sum_function = get_simple_window_sum_function(self.sliding)

        if (self.verbose):
            print("Computing windowed sums on original")
            sys.stdout.flush()
        original_summed_score_track = window_sum_function(arrs=score_track) 

        #Determine the window thresholds
        if (tnt_results is None):

            if (self.verbose):
                print("Generating null dist")
                sys.stdout.flush()
            if (hasattr(null_track, '__call__')):
                null_vals = null_track(
                    score_track=score_track,
                    windowsize=self.sliding,
                    original_summed_score_track=original_summed_score_track)
            else:
                null_summed_score_track = window_sum_function(arrs=null_track) 
                null_vals = list(np.concatenate(null_summed_score_track, axis=0))

            if (self.verbose):
                print("Computing threshold")
                sys.stdout.flush()
            from sklearn.isotonic import IsotonicRegression
            orig_vals = list(
                np.concatenate(original_summed_score_track, axis=0))
            pos_orig_vals = np.array(sorted([x for x in orig_vals if x >= 0]))
            neg_orig_vals = np.array(sorted([x for x in orig_vals if x < 0],
                                      key=lambda x: abs(x)))
            pos_null_vals = [x for x in null_vals if x >= 0]
            neg_null_vals = [x for x in null_vals if x < 0]
            pos_ir = IsotonicRegression().fit(
                X=np.concatenate([pos_orig_vals,pos_null_vals], axis=0),
                y=([1.0 for x in pos_orig_vals]
                   +[0.0 for x in pos_null_vals]),
                sample_weight=([1.0 for x in pos_orig_vals]+
                         [len(pos_orig_vals)/len(pos_null_vals)
                          for x in pos_null_vals]))
            pos_val_precisions = pos_ir.transform(pos_orig_vals)
            if (len(neg_orig_vals) > 0):
                neg_ir = IsotonicRegression(increasing=False).fit(
                    X=np.concatenate([neg_orig_vals,neg_null_vals], axis=0),
                    y=([1.0 for x in neg_orig_vals]
                       +[0.0 for x in neg_null_vals]),
                    sample_weight=([1.0 for x in neg_orig_vals]+
                             [len(neg_orig_vals)/len(neg_null_vals)
                              for x in neg_null_vals]))
                neg_val_precisions = neg_ir.transform(neg_orig_vals)

            pos_threshold = ([x[1] for x in
             zip(pos_val_precisions, pos_orig_vals) if x[0]
              >= (1-self.target_fdr)]+[pos_orig_vals[-1]])[0]
            if (len(neg_orig_vals) > 0):
                neg_threshold = ([x[1] for x in
                 zip(neg_val_precisions, neg_orig_vals) if x[0]
                  >= (1-self.target_fdr)]+[neg_orig_vals[-1]])[0]
            else:
                neg_threshold = -np.inf
            frac_passing_windows =(
                sum(pos_orig_vals >= pos_threshold)
                 + sum(neg_orig_vals <= neg_threshold))/float(len(orig_vals))

            if (self.verbose):
                print("Thresholds from null dist were",
                      neg_threshold," and ",pos_threshold)

            #adjust the thresholds if the fall outside the min/max
            # windows frac
            if (frac_passing_windows < self.min_passing_windows_frac):
                if (self.verbose):
                    print("Passing windows frac was",
                          frac_passing_windows,", which is below ",
                          self.min_passing_windows_frac,"; adjusting")
                if (self.separate_pos_neg_thresholds):
                    pos_threshold = np.percentile(
                        a=[x for x in orig_vals if x > 0],
                        q=100*(1-self.min_passing_windows_frac))
                    neg_threshold = np.percentile(
                        a=[x for x in orig_vals if x < 0],
                        q=100*(self.min_passing_windows_frac))
                else:
                    pos_threshold = np.percentile(
                        a=np.abs(orig_vals),
                        q=100*(1-self.min_passing_windows_frac)) 
                    neg_threshold = -pos_threshold
            if (frac_passing_windows > self.max_passing_windows_frac):
                if (self.verbose):
                    print("Passing windows frac was",
                          frac_passing_windows,", which is above ",
                          self.max_passing_windows_frac,"; adjusting")
                if (self.separate_pos_neg_thresholds):
                    pos_threshold = np.percentile(
                        a=[x for x in orig_vals if x > 0],
                        q=100*(1-self.max_passing_windows_frac))
                    neg_threshold = np.percentile(
                        a=[x for x in orig_vals if x < 0],
                        q=100*(self.max_passing_windows_frac))
                else:
                    pos_threshold = np.percentile(
                        a=np.abs(orig_vals),
                        q=100*(1-self.max_passing_windows_frac)) 
                    neg_threshold = -pos_threshold

            if (self.separate_pos_neg_thresholds):
                val_transformer = SignedPercentileValTransformer(
                    distribution=orig_vals)
            else:
                val_transformer = AbsPercentileValTransformer(
                    distribution=orig_vals)

            if (self.verbose):
                print("Final raw thresholds are",
                      neg_threshold," and ",pos_threshold)
                print("Final transformed thresholds are",
                      val_transformer(neg_threshold)," and ",
                      val_transformer(pos_threshold))

            from matplotlib import pyplot as plt

            plt.figure()
            np.random.shuffle(orig_vals)
            hist, histbins, _ = plt.hist(orig_vals[:min(len(orig_vals),
                                                        len(null_vals))],
                                         bins=100, alpha=0.5)
            np.random.shuffle(null_vals)
            _, _, _ = plt.hist(null_vals[:min(len(orig_vals),
                                              len(null_vals))],
                               bins=histbins, alpha=0.5)

            bincenters = 0.5*(histbins[1:]+histbins[:-1])
            poshistvals,posbins = zip(*[x for x in zip(hist,bincenters)
                                         if x[1] > 0])
            posbin_precisions = pos_ir.transform(posbins) 
            plt.plot([pos_threshold, pos_threshold], [0, np.max(hist)],
                     color="red")

            if (len(neg_orig_vals) > 0):
                neghistvals, negbins = zip(*[x for x in zip(hist,bincenters)
                                             if x[1] < 0])
                negbin_precisions = neg_ir.transform(negbins) 
                plt.plot(list(negbins)+list(posbins),
                     (list(np.minimum(neghistvals,
                                     neghistvals*(1-negbin_precisions)/
                                                 (negbin_precisions+1E-7)))+
                      list(np.minimum(poshistvals,
                                      poshistvals*(1-posbin_precisions)/
                                                 (posbin_precisions+1E-7)))),
                         color="purple")
                plt.plot([neg_threshold, neg_threshold], [0, np.max(hist)],
                         color="red")

            if plt.isinteractive():
                plt.show()
            else:
                import os, errno
                try:
                    os.makedirs(self.plot_save_dir)
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
                fname = (self.plot_save_dir+"/scoredist_" +
                         str(FixedWindowAroundChunks.count) + ".png")
                plt.savefig(fname)
                print("saving plot to " + fname)
                FixedWindowAroundChunks.count += 1

            tnt_results = TransformAndThresholdResults(
                neg_threshold=neg_threshold,
                transformed_neg_threshold=val_transformer(neg_threshold),
                pos_threshold=pos_threshold,
                transformed_pos_threshold=val_transformer(pos_threshold),
                val_transformer=val_transformer)

        neg_threshold = tnt_results.neg_threshold
        pos_threshold = tnt_results.pos_threshold

        summed_score_track = [np.array(x) for x in original_summed_score_track]

        #if a position is less than the threshold, set it to -np.inf
        summed_score_track = [
            np.array([np.abs(y) if (y > pos_threshold
                            or y < neg_threshold)
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
                    assert (coord.score > pos_threshold
                            or coord.score < neg_threshold)
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
