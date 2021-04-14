from __future__ import division, print_function, absolute_import
from . import util
import numpy as np


class PatternFilterer(object):

    #The idea is that 'patterns' gets divided into the patterns that pass and
    # the patterns that get filtered
    def __call__(self, patterns):
        raise NotImplementedError()

    def chain(self, pattern_filterer):
        def func(patterns):
            passing_patterns1, filtered_patterns1 = self(patterns)
            passing_patterns2, filtered_patterns2 =\
                pattern_filterer(passing_patterns1) 
            final_passing = passing_patterns2
            final_filtered = list(filtered_patterns1)+list(filtered_patterns2)
            #sanity check to make sure no patterns got lost
            assert len(final_filtered)+len(final_passing) == len(patterns)

            return (final_passing, final_filtered)

        return FuncPatternFilterer(function=func)


class FuncPatternFilterer(PatternFilterer):

    def __init__(self, function):
        self.function = function

    def __call__(self, patterns):
        return self.function(patterns)


class ConditionPatternFilterer(PatternFilterer):

    def _condition(self, pattern):
        raise NotImplementedError()

    def __call__(self, patterns):
        filtered_patterns = []
        passing_patterns = []
        for pattern in patterns:
            if self._condition(pattern):
                passing_patterns.append(pattern) 
            else:
                filtered_patterns.append(pattern)
        return (passing_patterns, filtered_patterns)


class MinSeqletSupportFilterer(ConditionPatternFilterer):

    #filter out patterns that don't have at least min_seqlet_support
    def __init__(self, min_seqlet_support):
        self.min_seqlet_support = min_seqlet_support

    def _condition(self, pattern):
        return len(pattern.seqlets) >= self.min_seqlet_support


class MinICinWindow(ConditionPatternFilterer):

    #filter out patterns that don't have at least min_seqlet_support
    def __init__(self, window_size, min_ic_in_window, background,
                       sequence_track_name,
                       ppm_pseudocount):
        self.window_size = window_size
        self.min_ic_in_window = min_ic_in_window
        self.background = background
        self.sequence_track_name = sequence_track_name 
        self.ppm_pseudocount = 0.001

    def _condition(self, pattern):
        ppm = pattern[self.sequence_track_name].fwd
        #compute per-position ic for the pattern
        per_position_ic = util.compute_per_position_ic(
                             ppm=ppm, background=self.background,
                             pseudocount=self.ppm_pseudocount)  
        if (len(per_position_ic) < self.window_size):
            print("WARNING: motif length is < window_size")          
            return np.sum(per_position_ic) >= self.min_ic_in_window 
        else:
            #do the sliding window sum rearrangement
            windowed_ic = np.sum(util.rolling_window(
                            a=per_position_ic, window=self.window_size), 
                            axis=-1) 
            return np.max(windowed_ic) >= self.min_ic_in_window


