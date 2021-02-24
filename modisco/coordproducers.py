from __future__ import division, print_function, absolute_import
from .core import SeqletCoordinates
from modisco import util
import numpy as np
from collections import defaultdict, Counter, OrderedDict
import itertools
import sys
import time
from .value_provider import (
    AbstractValTransformer, AbsPercentileValTransformer,
    SignedPercentileValTransformer, PrecisionValTransformer)
import scipy
from sklearn.isotonic import IsotonicRegression


SUBSAMPLE_CAP = 1000000


#The only parts of TransformAndThresholdResults that are used in
# TfModiscoWorkflow are the transformed_pos/neg_thresholds and the
# val_transformer (used in metaclustering with multiple tasks)
#TransformAndThresholdResults are also used to be
# able to replicate the same procedure used for identifying coordinates as
# when TfMoDisco was first run; the information needed in that case would
# be specific to the type of Coordproducer used
class AbstractTransformAndThresholdResults(object):

    def __init__(self, transformed_neg_threshold, transformed_pos_threshold,
                       val_transformer):
        self.transformed_neg_threshold = transformed_neg_threshold
        self.transformed_pos_threshold = transformed_pos_threshold
        self.val_transformer = val_transformer

    @classmethod
    def from_hdf5(cls, grp):
        if "class" not in grp.attrs:
            the_class =  FWACTransformAndThresholdResults
        else:
            the_class = eval(grp.attrs["class"])
        if (the_class.__name__ != cls.__name__):
            return the_class.from_hdf5(grp) 


class BasicTransformAndThresholdResults(AbstractTransformAndThresholdResults):

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        grp.attrs["transformed_neg_threshold"] = self.transformed_neg_threshold
        grp.attrs["transformed_pos_threshold"] = self.transformed_pos_threshold
        self.val_transformer.save_hdf5(grp.create_group("val_transformer"))

    @classmethod
    def load_basic_attrs_from_hdf5(cls, grp):
        transformed_neg_threshold = grp.attrs['transformed_neg_threshold']
        transformed_pos_threshold = grp.attrs['transformed_pos_threshold']
        val_transformer = AbstractValTransformer.from_hdf5(
                            grp["val_transformer"]) 
        return (transformed_neg_threshold, transformed_pos_threshold,
                val_transformer)

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs["class"])
        (transformed_neg_threshold,
         transformed_pos_threshold,
         val_transformer) = cls.load_basic_attrs_from_hdf5(grp)
        return cls(transformed_neg_threshold=transformed_neg_threshold,
                   transformed_pos_threshold=transformed_pos_threshold,
                   val_transformer=val_transformer) 


#FWAC = FixedWindowAroundChunks; this TransformAndThresholdResults object
# is specific to the type of info needed in that case.
class FWACTransformAndThresholdResults(
        BasicTransformAndThresholdResults):

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
        self.pos_threshold = pos_threshold
        super(FWACTransformAndThresholdResults, self).__init__(
            transformed_neg_threshold=transformed_neg_threshold,
            transformed_pos_threshold=transformed_pos_threshold,
            val_transformer=val_transformer)

    def save_hdf5(self, grp):
        super(FWACTransformAndThresholdResults, self).save_hdf5(grp)
        grp.attrs["neg_threshold"] = self.neg_threshold
        grp.attrs["pos_threshold"] = self.pos_threshold

    @classmethod
    def from_hdf5(cls, grp):
        (transformed_neg_threshold, transformed_pos_threshold,
         val_transformer) = cls.load_basic_attrs_from_hdf5(grp)
        neg_threshold = grp.attrs['neg_threshold']
        pos_threshold = grp.attrs['pos_threshold']
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
    def __init__(self, example_idx, start, end, score, other_info={}):
        self.score = score 
        self.other_info = other_info
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
        tnt_results = AbstractTransformAndThresholdResults.from_hdf5(
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

    def __call__(self, score_track, window_size, original_summed_score_track):

        #original_summed_score_track is supplied to avoid recomputing it 
        if (original_summed_score_track is None):
            window_sum_function = get_simple_window_sum_function(window_size)
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


def get_null_vals(null_track, score_track, window_size,
                  original_summed_score_track):
    if (hasattr(null_track, '__call__')):
        null_vals = null_track(
            score_track=score_track,
            window_size=window_size,
            original_summed_score_track=original_summed_score_track)
    else:
        window_sum_function = get_simple_window_sum_function(window_size)
        null_summed_score_track = window_sum_function(arrs=null_track) 
        null_vals = list(np.concatenate(null_summed_score_track, axis=0))
    return null_vals


def subsample_if_large(arr):
    if (len(arr) > SUBSAMPLE_CAP):
        print("Subsampling!")
        sys.stdout.flush()
        arr = np.random.RandomState(1234).choice(a=arr, size=SUBSAMPLE_CAP,
                                                 replace=False)
    return arr


def irval_to_probpos(irval, frac_neg):
    #n(x):= pdf of null dist (negatives)
    #p(x):= pdf of positive distribution
    #f_p:= fraction of positives
    #f_n:= fraction of negatives = 1-f_p
    #o(x):= pdf of observed distribution = n(x)f_n + p(x)f_p
    #The isotonic regression produces a(x) = o(x)/[o(x) + n(x)]
    # o(x)/[o(x) + n(x)] = [n(x)f_n + o(x)f_p]/[n(x)(1+f_n) + p(x)]
    # a(x)[n(x)(1+f_n) + p(x)f_p] = n(x)f_n + p(x)f_p
    # a(x)n(x)(1+f_n) - n(x)f_n = p(x)f_p - a(x)p(x)f_p
    # n(x)[a(x)(1+f_n) - f_n] = p(x)f_p[1 - a(x)]
    # [a(x)/f_n + (a(x)-1)]/[1-a(x)] = (p(x)f_p)/(n(x)f_n) = r(x)
    #p_pos = 1 / (1 + 1/r(x))
    #      = [a(x)/f_n + (a(x)-1)]/[a(x)/f_n + (a(x)-1) + (1-a(x))]
    #      = [a(x)/f_n + a(x)-1]/[a(x)/f_n]
    #      = [a(x) + f_n(a(x)-1)]/a(x)
    #      = 1 + f_n(a(x)-1)/a(x)
    #      = 1 + f_n(1 - 1/a(x))
    #If solving for p_pos=0, we have -1/(1 - 1/a(x)) = f_n
    #As f_n --> 100%, p_pos --> 2 - 1/a(x); this assumes max(a(x)) = 0.5
    return np.minimum(np.maximum(1 + frac_neg*(
                1 - (1/np.maximum(irval,1e-7))), 0.0), 1.0)


class SavableIsotonicRegression(object):

    def __init__(self, origvals, nullvals, increasing, min_frac_neg=0.95):
        self.origvals = origvals 
        self.nullvals = nullvals
        self.increasing = increasing
        self.min_frac_neg = min_frac_neg
        self.ir = IsotonicRegression(out_of_bounds='clip',
                                     increasing=increasing).fit(
            X=np.concatenate([self.origvals, self.nullvals], axis=0),
            y=([1.0 for x in self.origvals] + [0.0 for x in self.nullvals]),
            sample_weight=([1.0 for x in self.origvals]
                           +[float(len(self.origvals))/len(self.nullvals)
                             for x in self.nullvals]))
        #Infer frac_pos based on the minimum value of the ir probs
        #See derivation in irval_to_probpos function
        min_prec_x = self.ir.X_min_ if self.increasing else self.ir.X_max_ 
        min_precision = self.ir.transform([min_prec_x])[0]
        implied_frac_neg = -1/(1-(1/max(min_precision,1e-7)))
        print("For increasing =",increasing,", the minimum IR precision was",
              min_precision,"occurring at",min_prec_x,
              "implying a frac_neg",
              "of",implied_frac_neg)
        if (implied_frac_neg > 1.0 or implied_frac_neg < self.min_frac_neg):
            implied_frac_neg = max(min(1.0,implied_frac_neg),
                                   self.min_frac_neg)
            print("To be conservative, adjusted frac neg is",implied_frac_neg)
        self.implied_frac_neg = implied_frac_neg 
         
    def transform(self, vals):
        return irval_to_probpos(self.ir.transform(vals),
                                frac_neg=self.implied_frac_neg)

    def save_hdf5(self, grp):
        grp.attrs['increasing'] = self.increasing
        grp.attrs['min_frac_neg'] = self.min_frac_neg
        grp.create_dataset('origvals', data=self.origvals)
        grp.create_dataset('nullvals', data=self.nullvals)

    @classmethod
    def from_hdf5(cls, grp):
        increasing = grp.attrs['increasing']
        min_frac_neg = grp.attrs['min_frac_neg']
        origvals = np.array(grp['origvals'])
        nullvals = np.array(grp['nullvals'])
        return cls(origvals=origvals, nullvals=nullvals,
                   increasing=increasing, min_frac_neg=min_frac_neg) 


def get_isotonic_regression_classifier(orig_vals, null_vals):
    orig_vals = subsample_if_large(orig_vals)
    null_vals = subsample_if_large(null_vals)
    pos_orig_vals = (
        np.array(sorted([x for x in orig_vals if x >= 0])))
    neg_orig_vals = (
        np.array(sorted([x for x in orig_vals if x < 0],
                         key=lambda x: abs(x))))
    pos_null_vals = [x for x in null_vals if x >= 0]
    neg_null_vals = [x for x in null_vals if x < 0]
    pos_ir = SavableIsotonicRegression(origvals=pos_orig_vals,
                nullvals=pos_null_vals, increasing=True) 

    if (len(neg_orig_vals) > 0):
        neg_ir = SavableIsotonicRegression(origvals=neg_orig_vals,
                    nullvals=neg_null_vals, increasing=False)
    else:
        neg_ir = None

    return pos_ir, neg_ir, orig_vals, null_vals


#sliding in this case would be a list of values
class VariableWindowAroundChunks(AbstractCoordProducer):
    count = 0
    def __init__(self, sliding, flank, suppress, target_fdr,
                       min_passing_windows_frac, max_passing_windows_frac,
                       separate_pos_neg_thresholds,
                       max_seqlets_total,
                       progress_update=5000,
                       verbose=True, plot_save_dir="figures"): 
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
        sliding = np.array(grp["sliding"]).astype("int")
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
        grp.create_dataset("sliding", data=np.array(self.sliding))
        grp.attrs["flank"] = self.flank
        grp.attrs["suppress"] = self.suppress
        grp.attrs["target_fdr"] = self.target_fdr
        grp.attrs["min_passing_windows_frac"] = self.min_passing_windows_frac
        grp.attrs["max_passing_windows_frac"] = self.max_passing_windows_frac
        grp.attrs["separate_pos_neg_thresholds"] =\
            self.separate_pos_neg_thresholds
        if (self.max_seqlets_total is not None):
            grp.attrs["max_seqlets_total"] = self.max_seqlets_total 
        grp.attrs["progress_update"] = self.progress_update
        grp.attrs["verbose"] = self.verbose

    def fit_pos_and_neg_irs(self, score_track, null_track):
        pos_irs = []
        neg_irs = []
        for sliding_window_size in self.sliding:
            window_sum_function = get_simple_window_sum_function(
                                        sliding_window_size)
            print("Fitting - on window size",sliding_window_size)
            if (hasattr(null_track, '__call__')):
                null_vals = null_track(
                    score_track=score_track,
                    window_size=sliding_window_size,
                    original_summed_score_track=None)
            else:
                null_summed_score_track = window_sum_function(arrs=null_track) 
                null_vals = np.concatenate(null_summed_score_track,
                                                axis=0)
            print("Computing window sums")
            sys.stdout.flush()
            window_sums_rows = window_sum_function(arrs=score_track)
            print("Done computing window sums")
            sys.stdout.flush()

            orig_vals = np.concatenate(window_sums_rows, axis=0)

            pos_ir, neg_ir, subsampled_orig_vals, subsampled_null_vals =\
                get_isotonic_regression_classifier(
                    orig_vals=np.concatenate(window_sums_rows, axis=0),
                    null_vals=null_vals)


            make_nulldist_figure(orig_vals=subsampled_orig_vals,
                                 null_vals=subsampled_null_vals,
                                 pos_ir=pos_ir, neg_ir=neg_ir,
                                 pos_threshold=None,
                                 neg_threshold=None)
            util.show_or_savefig(plot_save_dir=self.plot_save_dir,
                           filename="scoredist_window"
                                 +str(sliding_window_size)+"_"
                                 +str(VariableWindowAroundChunks.count)+".png")

            pos_irs.append(pos_ir)
            neg_irs.append(neg_ir)
        return pos_irs, neg_irs

    def __call__(self, score_track, null_track, tnt_results=None):
        if (tnt_results is None):
            pos_irs, neg_irs = self.fit_pos_and_neg_irs(
                                      score_track=score_track,
                                      null_track=null_track)
            precision_transformer = PrecisionValTransformer(
                                        sliding_window_sizes=self.sliding,
                                        pos_irs=pos_irs,
                                        neg_irs=neg_irs) 
            (precisiontransformed_score_track,
             precisiontransformed_bestwindowsizeidxs) =\
                precision_transformer.transform_score_track(
                    score_track=score_track) 

            subsampled_prec_vals = subsample_if_large(
                np.concatenate(precisiontransformed_score_track, axis=0)) 

            from matplotlib import pyplot as plt
            plt.plot(sorted(subsampled_prec_vals),
                     (np.arange(len(subsampled_prec_vals))/
                      len(subsampled_prec_vals))) 
            plt.xlabel("Tranformed IR precision value")
            plt.ylabel("CDF")
            util.show_or_savefig(plot_save_dir=self.plot_save_dir,
                           filename="final_prec_vals_cdf_dist"
                                 +str(VariableWindowAroundChunks.count)+".png")

            #Pick a threshold according the the precisiontransformed score track
            pos_threshold = (1-self.target_fdr)
            neg_threshold = -(1-self.target_fdr)

            pos_threshold, neg_threshold =\
                refine_thresholds_based_on_frac_passing(
                  vals=subsampled_prec_vals,
                  pos_threshold=pos_threshold,
                  neg_threshold=neg_threshold,
                  min_passing_windows_frac=self.min_passing_windows_frac,
                  max_passing_windows_frac=self.max_passing_windows_frac,
                  separate_pos_neg_thresholds=self.separate_pos_neg_thresholds,
                  verbose=self.verbose) 
            
            tnt_results = BasicTransformAndThresholdResults(
                transformed_neg_threshold=neg_threshold,
                transformed_pos_threshold=pos_threshold,
                val_transformer=precision_transformer) 

        else:
            precision_transformer = tnt_results.precision_transformer
            (precisiontransformed_score_track,
             precisiontransformed_bestwindowsizeidxs) =\
                precision_transformer.transform_score_track(
                    score_track=score_track) 

        #Need to remove padding because identify_coords is assumed to
        # operate on a scoretrack that has already been processed with
        # a sliding window of window_size (and assumes that partial windows
        # were not included)
        left_padding_to_remove = int((max(self.sliding)-1)/2)
        right_padding_to_remove = (max(self.sliding)-1)-left_padding_to_remove
        coords = identify_coords(
            score_track=[x[left_padding_to_remove:-right_padding_to_remove]
                         for x in precisiontransformed_score_track],
            pos_threshold=tnt_results.transformed_pos_threshold,
            neg_threshold=tnt_results.transformed_neg_threshold,
            window_size=max(self.sliding),
            flank=self.flank,
            suppress=self.suppress,
            max_seqlets_total=self.max_seqlets_total,
            verbose=self.verbose,
            other_info_tracks={'best_window_idx':
             [x[left_padding_to_remove:-right_padding_to_remove] for x in
              precisiontransformed_bestwindowsizeidxs]})

        VariableWindowAroundChunks.count += 1
        
        return CoordProducerResults(
                    coords=coords,
                    tnt_results=tnt_results) 


#identify_coords is expecting something that has already been processed
# with sliding windows of size window_size
def identify_coords(score_track, pos_threshold, neg_threshold,
                    window_size, flank, suppress,
                    max_seqlets_total, verbose, other_info_tracks={}):

    for other_info_track in other_info_tracks.values():
        assert all([x.shape==y.shape for x,y
                    in zip(other_info_track,score_track)])

    #cp_score_track = 'copy' of the score track, which can be modified as
    # coordinates are identified
    cp_score_track = [np.array(x) for x in score_track]
    #if a position is less than the threshold, set it to -np.inf
    #Note that the threshold comparisons need to be >= and not just > for
    # cases where there are lots of ties at the high end (e.g. with an IR
    # tranformation that gives a lot of values that have a precision of 1.0)
    cp_score_track = [
        np.array([np.abs(y) if (y >= pos_threshold
                        or y <= neg_threshold)
                       else -np.inf for y in x])
        for x in cp_score_track]

    coords = []
    for example_idx,single_score_track in enumerate(cp_score_track):
        #set the stuff near the flanks to -np.inf so that we
        # don't pick it up during argmax
        single_score_track[0:flank] = -np.inf
        single_score_track[len(single_score_track)-(flank):
                           len(single_score_track)] = -np.inf
        while True:
            argmax = np.argmax(single_score_track,axis=0)
            max_val = single_score_track[argmax]

            #bail if exhausted everything that passed the threshold
            #and was not suppressed
            if (max_val == -np.inf):
                break

            #need to be able to expand without going off the edge
            if ((argmax >= flank) and
                (argmax < (len(single_score_track)-flank))): 

                coord = SeqletCoordsFWAP(
                    example_idx=example_idx,
                    start=argmax-flank,
                    end=argmax+window_size+flank,
                    score=score_track[example_idx][argmax],
                    other_info = dict([
                     (track_name, track[example_idx][argmax])
                     for (track_name, track) in other_info_tracks.items()])) 
                assert (coord.score >= pos_threshold
                        or coord.score <= neg_threshold)
                coords.append(coord)
            else:
                assert False,\
                 ("This shouldn't happen because I set stuff near the"
                  "border to -np.inf early on")
            #suppress the chunks within +- suppress
            left_supp_idx = int(max(np.floor(argmax+0.5-suppress),0))
            right_supp_idx = int(min(np.ceil(argmax+0.5+suppress),
                                 len(single_score_track)))
            single_score_track[left_supp_idx:right_supp_idx] = -np.inf 

    if (verbose):
        print("Got "+str(len(coords))+" coords")
        sys.stdout.flush()

    if ((max_seqlets_total is not None) and
        len(coords) > max_seqlets_total):
        if (verbose):
            print("Limiting to top "+str(max_seqlets_total))
            sys.stdout.flush()
        coords = sorted(coords, key=lambda x: -np.abs(x.score))\
                           [:max_seqlets_total]
    
    return coords


def refine_thresholds_based_on_frac_passing(
    vals, pos_threshold, neg_threshold,
    min_passing_windows_frac, max_passing_windows_frac,
    separate_pos_neg_thresholds, verbose):

    frac_passing_windows =(
        sum(vals >= pos_threshold)
         + sum(vals <= neg_threshold))/float(len(vals))

    if (verbose):
        print("Thresholds from null dist were",
              neg_threshold," and ",pos_threshold,
              "with frac passing", frac_passing_windows)

    pos_vals = [x for x in vals if x >= 0]
    neg_vals = [x for x in vals if x < 0]
    #deal with edge case of len < 0
    pos_vals = [0] if len(pos_vals)==0 else pos_vals
    neg_vals = [0] if len(neg_vals)==0 else neg_vals

    #adjust the thresholds if the fall outside the min/max
    # windows frac
    if (frac_passing_windows < min_passing_windows_frac):
        if (verbose):
            print("Passing windows frac was",
                  frac_passing_windows,", which is below ",
                  min_passing_windows_frac,"; adjusting")
        if (separate_pos_neg_thresholds):
            pos_threshold = np.percentile(
                a=pos_vals,
                q=100*(1-min_passing_windows_frac))
            neg_threshold = np.percentile(
                a=neg_vals,
                q=100*(min_passing_windows_frac))
        else:
            pos_threshold = np.percentile(
                a=np.abs(vals),
                q=100*(1-min_passing_windows_frac)) 
            neg_threshold = -pos_threshold

    if (frac_passing_windows > max_passing_windows_frac):
        if (verbose):
            print("Passing windows frac was",
                  frac_passing_windows,", which is above ",
                  max_passing_windows_frac,"; adjusting")
        if (separate_pos_neg_thresholds):
            pos_threshold = np.percentile(
                a=pos_vals,
                q=100*(1-max_passing_windows_frac))
            neg_threshold = np.percentile(
                a=neg_vals,
                q=100*(max_passing_windows_frac))
        else:
            pos_threshold = np.percentile(
                a=np.abs(vals),
                q=100*(1-max_passing_windows_frac)) 
            neg_threshold = -pos_threshold
        if (verbose):
            print("New thresholds are",pos_threshold,"and",neg_threshold)

    return pos_threshold, neg_threshold


def make_nulldist_figure(orig_vals, null_vals, pos_ir, neg_ir,
                         pos_threshold, neg_threshold):
    from matplotlib import pyplot as plt
    fig,ax1 = plt.subplots()

    orig_vals = np.array(sorted(orig_vals))

    ax1.hist(orig_vals, bins=100, density=True, alpha=0.5) 
    ax1.hist(null_vals, bins=100, density=True, alpha=0.5) 
    ax1.set_ylabel("Probability density\n(blue=foreground, orange=null)")
    ax1.set_xlabel("Total importance in window")

    precisions = pos_ir.transform(orig_vals)
    if (neg_ir is not None):
        precisions = np.maximum(precisions, neg_ir.transform(orig_vals))

    ax2 = ax1.twinx() 
    ax2.plot(orig_vals, precisions)
    if (pos_threshold is not None):
        ax2.plot([pos_threshold, pos_threshold], [0.0, 1.0], color="red")
    if (neg_threshold is not None):
        ax2.plot([neg_threshold, neg_threshold], [0.0, 1.0], color="red")
    ax2.set_ylabel("Estimated foreground precision")
    ax2.set_ylim(0.0, 1.02)
                

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
            
            null_vals = get_null_vals(
                null_track=null_track,
                score_track=score_track,
                window_size=self.sliding,
                original_summed_score_track=original_summed_score_track)

            if (self.verbose):
                print("Computing threshold")
                sys.stdout.flush()
            orig_vals = list(
                np.concatenate(original_summed_score_track, axis=0))

            #Note that orig_vals may have been subsampled at this point
            pos_ir, neg_ir, subsampled_orig_vals, subsampled_null_vals =\
                get_isotonic_regression_classifier(
                    orig_vals=orig_vals,
                    null_vals=null_vals)
    
            subsampled_pos_orig_vals = (
                np.array(sorted([x for x in subsampled_orig_vals if x >= 0])))
            subsampled_neg_orig_vals = (
                np.array(sorted([x for x in subsampled_orig_vals if x < 0],
                         key=lambda x: abs(x))))

            subsampled_pos_val_precisions =\
                pos_ir.transform(subsampled_pos_orig_vals)
            if (len(subsampled_neg_orig_vals) > 0):
                subsampled_neg_val_precisions =\
                    neg_ir.transform(subsampled_neg_orig_vals)

            pos_threshold = ([x[1] for x in
             zip(subsampled_pos_val_precisions,
                 subsampled_pos_orig_vals) if x[0]
              >= (1-self.target_fdr)]+[subsampled_pos_orig_vals[-1]])[0]
            if (len(subsampled_neg_orig_vals) > 0):
                neg_threshold = ([x[1] for x in
                 zip(subsampled_neg_val_precisions,
                     subsampled_neg_orig_vals) if x[0]
                  >= (1-self.target_fdr)]+[subsampled_neg_orig_vals[-1]])[0]
            else:
                neg_threshold = -np.inf

            pos_threshold, neg_threshold =\
                refine_thresholds_based_on_frac_passing(
                  vals=subsampled_orig_vals,
                  pos_threshold=pos_threshold,
                  neg_threshold=neg_threshold,
                  min_passing_windows_frac=self.min_passing_windows_frac,
                  max_passing_windows_frac=self.max_passing_windows_frac,
                  separate_pos_neg_thresholds=self.separate_pos_neg_thresholds,
                  verbose=self.verbose) 

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

            make_nulldist_figure(orig_vals=subsampled_orig_vals,
                                 null_vals=subsampled_null_vals,
                                 pos_ir=pos_ir, neg_ir=neg_ir,
                                 pos_threshold=pos_threshold,
                                 neg_threshold=neg_threshold)

            util.show_or_savefig(plot_save_dir=self.plot_save_dir,
                           filename="scoredist_"
                                    +str(FixedWindowAroundChunks.count)+".png")
            FixedWindowAroundChunks.count += 1

            tnt_results = FWACTransformAndThresholdResults(
                neg_threshold=neg_threshold,
                transformed_neg_threshold=val_transformer(neg_threshold),
                pos_threshold=pos_threshold,
                transformed_pos_threshold=val_transformer(pos_threshold),
                val_transformer=val_transformer)

        coords = identify_coords(
            score_track=original_summed_score_track,
            pos_threshold=tnt_results.pos_threshold,
            neg_threshold=tnt_results.neg_threshold,
            window_size=self.sliding,
            flank=self.flank,
            suppress=self.suppress,
            max_seqlets_total=self.max_seqlets_total,
            verbose=self.verbose)

        return CoordProducerResults(
                    coords=coords,
                    tnt_results=tnt_results) 
