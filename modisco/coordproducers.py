from __future__ import division, print_function, absolute_import
from .core import SeqletCoordinates
from modisco import backend as B 
from modisco import util
import numpy as np
from collections import defaultdict
import itertools
from sklearn.neighbors.kde import KernelDensity


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


class CoordOverlapDetector(object):

    def __init__(self, min_overlap_fraction):
        self.min_overlap_fraction = min_overlap_fraction

    def __call__(self, coord1, coord2):
        if (coord1.example_idx != coord2.example_idx):
            return False
        min_overlap = self.min_overlap_fraction*min(len(coord1), len(coord2))
        overlap_amt = (min(coord1.end, coord2.end)-
                       max(coord1.start, coord2.start))
        return (overlap_amt >= min_overlap)


class CoordComparator(object):

    def __init__(self, attribute_provider):
        self.attribute_provider = attribute_provider

    def get_larger(self, coord1, coord2):
        return (coord1 if (self.attribute_provider(coord1) >=
                           self.attribute_provider(coord2)) else coord2)

    def get_smaller(self, coord1, coord2):
        return (coord1 if (self.attribute_provider(coord1) <=
                           self.attribute_provider(coord2)) else coord2)


class ResolveOverlapsCoordProducer(AbstractCoordProducer):

    def __init__(self, coord_producer, overlap_detector, coord_comparator):
        self.coord_producer = coord_producer
        self.overlap_detector = overlap_detector
        self.coord_comparator = coord_comparator

    def __call__(self, kwargsets):
        coord_sets = [self.coord_producer(**kwargset) for
                       kwargset in kwargsets]
        example_idx_to_coords = defaultdict(list)  
        for coord in itertools.chain(*coord_sets):
            example_idx_to_coords[coord.example_idx].append(coord)
        for example_idx, coords in example_idx_to_coords.items():
            final_coords_set = set(coords)
            for i in range(len(coords)):
                coord1 = coords[i]
                for coord2 in coords[i+1:]:
                    if (coord1 not in final_coords_set):
                        break
                    if ((coord2 in final_coords_set)
                         and self.overlap_detector(coord1, coord2)):
                        final_coords_set.remove(
                         self.coord_comparator.get_smaller(coord1, coord2)) 
            example_idx_to_coords[example_idx] = list(final_coords_set)
        return list(itertools.chain(*example_idx_to_coords.values()))


class MaxCurvatureThreshold(object):

    def __init__(self, bins, bandwidth, num_to_consider, verbose):
        self.bins = bins
        self.bandwidth = bandwidth
        self.num_to_consider = num_to_consider
        self.verbose = verbose

    def __call__(self, values):
        assert np.min(values) >= 0

        hist_y, hist_x = np.histogram(values, bins=self.bins*2)
        hist_x = 0.5*(hist_x[:-1]+hist_x[1:])
        global_max_x = max(zip(hist_y,hist_x), key=lambda x: x[0])[1]
        #create a symmetric reflection around global_max_x so kde does not
        #get confused
        new_values = np.array([x for x in values if x >= global_max_x])
        new_values = np.concatenate([new_values, -(new_values-global_max_x)
                                                  + global_max_x])
        kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth).fit(
                    [[x,0] for x in new_values])
        midpoints = np.min(values)+((np.arange(self.bins)+0.5)
                                    *(np.max(values)-np.min(values))/self.bins)
        densities = np.exp(kde.score_samples([[x,0] for x in midpoints]))

        firstd_x, firstd_y = util.firstd(x_values=midpoints, y_values=densities) 
        secondd_x, secondd_y = util.firstd(x_values=firstd_x, y_values=firstd_y)
        #find point of maximum curvature
        maximum_c_x = max(zip(secondd_x, secondd_y), key=lambda x:x[1])[0]

        if (self.verbose):
            from matplotlib import pyplot as plt
            hist_y, _, _ = plt.hist(new_values, bins=self.bins)
            max_y = np.max(hist_y)
            plt.plot(midpoints, densities*(max_y/np.max(densities)))
            plt.plot([maximum_c_x, maximum_c_x], [0, max_y])
            plt.show()
            plt.plot(secondd_x, secondd_y)
            plt.show()

        return maximum_c_x


class FixedWindowAroundChunks(AbstractCoordProducer):

    def __init__(self, sliding=11,
                       flank=10,
                       suppress=20,
                       max_seqlets_per_seq=5,
                       thresholding_function=MaxCurvatureThreshold(
                            bins=200, bandwidth=0.1,
                            num_to_consider=1000000, verbose=True),
                       min_ratio_top_peak=0.0,
                       min_ratio_over_bg=0.0,
                       max_seqlets_total=10000,
                       batch_size=50,
                       progress_update=5000,
                       verbose=True):
        self.sliding = sliding
        self.flank = flank
        self.suppress = suppress
        self.max_seqlets_per_seq = max_seqlets_per_seq
        self.thresholding_function = thresholding_function
        self.min_ratio_top_peak = min_ratio_top_peak
        self.min_ratio_over_bg = min_ratio_over_bg
        self.max_seqlets_total = max_seqlets_total
        self.batch_size = batch_size
        self.progress_update = progress_update
        self.verbose = verbose

    def __call__(self, score_track):
     
        assert len(score_track.shape)==2 
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

        if (self.verbose):
            print("Got "+str(len(coords))+" coords")

        vals_to_threshold = np.array([np.abs(x.score) for x in coords])
        if (self.thresholding_function is not None):
            if (self.verbose):
                print("Computing thresholds")
            threshold = self.thresholding_function(vals_to_threshold) 
        else:
            threshold = 0.0
        if (self.verbose):
            percentile = np.sum(vals_to_threshold<threshold)/\
                                float(len(vals_to_threshold))
            print("threshold is "+str(threshold)
                  +" with percentile "+str(percentile))

        coords = [x for x in coords if x.score >= threshold]
        if (self.verbose):
            print(str(len(coords))+" coords remaining after thresholding")

        if (len(coords) > self.max_seqlets_total):
            if (self.verbose):
                print("Limiting to top "+str(self.max_seqlets_total))
            coords = sorted(coords, key=lambda x: -x.score)\
                               [:self.max_seqlets_total]
        return coords



