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


class GammaMixThresholdingResults(object):

    def __init__(self, threshold, mixture_results):
        self.threshold = threshold
        self.mixture_results = mixture_results

    def save_hdf5(self, grp):
        grp.attrs['threshold'] = self.threshold


class GammaMixThreshold(object):

    def __init__(self, init_mixprops = [[0.95, 0.05],
                                        [0.9, 0.1],
                                        [0.85, 0.15],
                                        [0.8, 0.2],
                                        [0.75, 0.25],
                                        [0.7, 0.3]],
                       gammamix_subsample_size=30000,
                       verbose=True):
        self.init_mixprops = init_mixprops
        self.verbose = verbose
        self.gammamix_subsample_size = gammamix_subsample_size

    def __call__(self, values):
        from . import gammamix 
        vals_to_score = (values if len(values) <= self.gammamix_subsample_size
                            else np.random.choice(
                                    a=values,
                                    size=self.gammamix_subsample_size,
                                    replace=False))
        vals_to_score = sorted(vals_to_score)
        best_ll = None
        best_mixture_results = None
        for init_mixprop in self.init_mixprops: 
            print("Trying with init:",init_mixprop)
            mixture_results = gammamix.gammamix_em(
                                vals_to_score,
                                mix_prop=np.array(init_mixprop),
                                verb=self.verbose) 
            print("params",mixture_results.params)
            print("Got ll:",mixture_results.ll[-1])
            if (best_ll is None or mixture_results.ll[-1] > best_ll):
                best_ll = mixture_results.ll[-1]
                best_mixture_results = mixture_results 
        print("best params",best_mixture_results.params)
        print(best_mixture_results.expected_membership[:,:50])
        threshold = [x for x in
         zip(vals_to_score, best_mixture_results.expected_membership)
         if x[1][1] > 0.5][0][0]

        if (self.verbose):
            from matplotlib import pyplot as plt
            hist_y, _, _ = plt.hist(values, bins=100)
            max_y = np.max(hist_y)
            plt.plot([threshold, threshold], [0, max_y])
            plt.show()

        return GammaMixThresholdingResults(
                threshold=threshold, mixture_results=best_mixture_results)


class MaxCurvatureThresholdingResults(object):

    def __init__(self, threshold, densities):
        self.densities = densities 
        self.threshold = threshold

    def save_hdf5(self, grp):
        grp.create_dataset("densities", data=self.densities) 
        grp.attrs['threshold'] = self.threshold


class MaxCurvatureThreshold(object):

    def __init__(self, bins, verbose, percentiles_in_bandwidth):
        self.bins = bins
        assert percentiles_in_bandwidth < 100
        self.percentiles_in_bandwidth = percentiles_in_bandwidth
        self.verbose = verbose

    def __call__(self, values):

        #determine bandwidth
        sorted_values = sorted(values)
        vals_to_avg = []
        last_val = sorted_values[0]
        num_in_bandwidth = int((0.01*len(values))
                               *self.percentiles_in_bandwidth)
        for i in range(num_in_bandwidth, len(values), num_in_bandwidth):
            vals_to_avg.append(sorted_values[i]-last_val) 
            last_val = sorted_values[i]
        #take the median of the diff between num_in_bandwidth
        self.bandwidth = np.median(np.array(vals_to_avg))
        if (self.verbose):
            print("Bandwidth calculated:",self.bandwidth)

        hist_y, hist_x = np.histogram(values, bins=self.bins*2)
        hist_x = 0.5*(hist_x[:-1]+hist_x[1:])
        global_max_x = max(zip(hist_y,hist_x), key=lambda x: x[0])[1]
        #create a symmetric reflection around global_max_x so kde does not
        #get confused
        new_values = np.array([x for x in values if x >= global_max_x])
        new_values = np.concatenate([new_values, -(new_values-global_max_x)
                                                  + global_max_x])
        kde = KernelDensity(kernel="epanechnikov",
                            bandwidth=self.bandwidth).fit(
                            [[x,0] for x in new_values])
        midpoints = np.min(values)+((np.arange(self.bins)+0.5)
                                    *(np.max(values)-np.min(values))/self.bins)
        densities = np.exp(kde.score_samples([[x,0] for x in midpoints]))

        firstd_x, firstd_y = util.angle_firstd(x_values=midpoints,
                                                y_values=densities) 
        curvature_x, curvature_y = util.angle_curvature(x_values=midpoints,
                                                y_values=densities) 
        secondd_x, secondd_y = util.angle_firstd(x_values=firstd_x,
                                           y_values=firstd_y)
        thirdd_x, thirdd_y = util.firstd(x_values=secondd_x,
                                           y_values=secondd_y)
        mean_firstd_ys_at_secondds = 0.5*(firstd_y[1:]+firstd_y[:-1])
        mean_secondd_ys_at_thirdds = 0.5*(secondd_y[1:]+secondd_y[:-1])
        #find point of maximum curvature

        #maximum_c_x = max([x for x in zip(secondd_x, secondd_y,
        #                                  mean_firstd_ys_at_secondds)
        #                   if (x[0] > global_max_x
        #                      and x[2] < 0)], key=lambda x:x[1])[0]

        maximum_c_x = ([x for x in enumerate(secondd_x)
                        if (x[1] > global_max_x and
                            x[0] < len(secondd_x)-1 and
                            secondd_y[x[0]-1] < secondd_y[x[0]] and
                            secondd_y[x[0]+1] < secondd_y[x[0]])])[0][1]

        maximum_c_x2 = max([x for x in zip(thirdd_x, thirdd_y,
                                          mean_secondd_ys_at_thirdds)
                           if (x[0] > global_max_x
                              and x[2] > 0)], key=lambda x:x[1])[0]

        if (self.verbose):
            from matplotlib import pyplot as plt
            hist_y, _, _ = plt.hist(values, bins=self.bins)
            max_y = np.max(hist_y)
            plt.plot(midpoints, densities*(max_y/np.max(densities)))
            plt.plot(firstd_x, -firstd_y*(max_y/np.max(-firstd_y))*(firstd_y<0))
            #plt.plot(secondd_x, secondd_y*(max_y/np.max(secondd_y))*(secondd_y>0))
            #plt.plot(curvature_x, curvature_y*(max_y/np.max(curvature_y))*(curvature_y>0))
            #plt.plot(thirdd_x, thirdd_y*(max_y/np.max(thirdd_y))*(thirdd_y>0))
            #plt.plot(secondd_x, (secondd_y>0)*secondd_y*(max_y/np.max(secondd_y)))
            plt.plot([maximum_c_x, maximum_c_x], [0, max_y])
            #plt.plot([maximum_c_x2, maximum_c_x2], [0, max_y])
            plt.xlim((0, maximum_c_x*5))
            plt.show()
            plt.plot(firstd_x, firstd_y)
            plt.plot(secondd_x, secondd_y*(secondd_y>0))
            plt.xlim((0, maximum_c_x*5))
            plt.show()
            plt.plot(curvature_x[curvature_x>global_max_x], curvature_y[curvature_x>global_max_x])
            plt.xlim((0, maximum_c_x*5))
            plt.show()

        return MaxCurvatureThresholdingResults(
                threshold=maximum_c_x, densities=densities)


class LaplaceThresholdingResults(object):

    def __init__(self, threshold, b):
        self.b = b
        self.threshold = threshold

    def save_hdf5(self, grp):
        grp.attrs['b'] = self.b 
        grp.attrs['threshold'] = self.threshold


class LaplaceThreshold(object):

    def __init__(self, threshold_cdf, verbose):
        assert (threshold_cdf > 0.5 and threshold_cdf < 1.0)
        self.threshold_cdf = threshold_cdf
        self.verbose = verbose

    def __call__(self, values):

        #We assume that the null is governed by a laplace, because
        #that's what I've personally observed
        #80th percentile of things below 0 is 40th percentile of errything
        forty_perc = np.percentile(values[values < 0.0], 80)
        #estimate b using the percentile
        #for x below 0:
        #cdf = 0.5*exp(x/b)
        #b = x/(log(cdf/0.5))
        laplace_b = forty_perc/(np.log(0.8))

        #solve for x given the target threshold percentile
        #(assumes the target threshold percentile is > 0.5)
        threshold = -np.log((1-self.threshold_cdf)*2)*laplace_b

        #plot the result
        if (self.verbose):
            thelinspace = np.linspace(np.min(values), np.max(values), 100)
            laplace_vals = (1/(2*laplace_b))*np.exp(
                            -np.abs(thelinspace)/laplace_b)
            from matplotlib import pyplot as plt
            hist, _, _ = plt.hist(values, bins=100)
            plt.plot(thelinspace,
                     laplace_vals/(np.max(laplace_vals))*np.max(hist))
            plt.plot([-threshold, -threshold], [0, np.max(hist)])
            plt.plot([threshold, threshold], [0, np.max(hist)])
            plt.show()

        return LaplaceThresholdingResults(
                threshold=threshold, b=laplace_b)


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


class FixedWindowAroundChunks(AbstractCoordProducer):

    def __init__(self, sliding=11,
                       flank=10,
                       suppress=None,
                       max_seqlets_per_seq=10,
                       thresholding_function=MaxCurvatureThreshold(
                            bins=100, percentiles_in_bandwidth=10,
                            verbose=True),
                       take_abs=True, 
                       min_ratio_top_peak=0.0,
                       min_ratio_over_bg=0.0,
                       apply_recentering=False,
                       max_seqlets_total=20000,
                       batch_size=50,
                       progress_update=5000,
                       verbose=True):
        self.sliding = sliding
        self.flank = flank
        if (suppress is None):
            suppress = int(0.5*sliding) + flank
        self.suppress = suppress
        self.max_seqlets_per_seq = max_seqlets_per_seq
        self.thresholding_function = thresholding_function
        self.take_abs = take_abs
        self.min_ratio_top_peak = min_ratio_top_peak
        self.min_ratio_over_bg = min_ratio_over_bg
        self.apply_recentering = apply_recentering
        self.max_seqlets_total = max_seqlets_total
        self.batch_size = batch_size
        self.progress_update = progress_update
        self.verbose = verbose

    def __call__(self, score_track):
     
        assert len(score_track.shape)==2 
        window_sum_function = B.get_window_sum_function(
                                window_size=self.sliding,
                                same_size_return=False)
        argmax_func = B.get_argmax_function()

        original_summed_score_track = np.array(window_sum_function(
            inp=score_track,
            batch_size=self.batch_size,
            progress_update=
             (self.progress_update if self.verbose else None))).astype("float") 

        thresholding_results = self.thresholding_function(
                                original_summed_score_track.ravel())
        threshold = thresholding_results.threshold
        if (self.verbose):
            print("Computed threshold "+str(threshold))
            sys.stdout.flush()

        summed_score_track = original_summed_score_track.copy()
        if (self.take_abs):
            summed_score_track = np.abs(summed_score_track)

        #As we extract seqlets, we will zero out the values at those positions
        #so that the mean of the background can be updated to exclude
        #the seqlets (which are likely to be outliers)
        zerod_out_summed_score_track = np.copy(summed_score_track)
         
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
                        score = (original_summed_score_track
                                 [example_idx][argmax])
                        coord = SeqletCoordsFWAP(
                            example_idx=example_idx,
                            start=argmax-self.flank,
                            end=argmax+self.sliding+self.flank,
                            score=score) 
                        if (self.apply_recentering):
                            half_sliding = int(0.5*self.sliding)
                            if ((argmax+half_sliding+self.flank <=
                                 original_summed_score_track.shape[1]) and
                                (argmax >= self.flank+half_sliding)):
                                arr_to_check_for_center =\
                                    original_summed_score_track[
                                        example_idx,
                                        argmax-self.flank-half_sliding:
                                         argmax+half_sliding+self.flank] 
                                adjusted_argmax =\
                                    (argmax+np.argmax(arr_to_check_for_center)
                                      -(half_sliding+self.flank))
                                if ((adjusted_argmax >= self.flank) and
                                    (adjusted_argmax <=
                                     (score_track.shape[1] 
                                      -(self.sliding+self.flank)))):
                                    coords.append(
                                        SeqletCoordsFWAP(
                                            example_idx=example_idx,
                                            start=adjusted_argmax-self.flank,
                                            end=adjusted_argmax
                                                +self.sliding+self.flank,
                                            score=original_summed_score_track
                                                       [example_idx,
                                                        adjusted_argmax]))
                        else:
                            coords.append(coord)
                    #only zero out if the region was included, so that we
                    #don't zero out sequences that do not pass the conditions
                    zerod_out_summed_score_track[
                        example_idx,
                        left_supp_idx:right_supp_idx] = 0.0
                summed_score_track[
                    example_idx, left_supp_idx:right_supp_idx] = -np.inf 

        if (self.verbose):
            print("Got "+str(len(coords))+" coords")
            sys.stdout.flush()

        coords = [x for x in coords if np.abs(x.score) >= threshold]
        if (self.verbose):
            print(str(len(coords))+" coords remaining after thresholding")
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

