from __future__ import division, print_function
from collections import namedtuple
import numpy as np
from .. import util
from .. import core


RankNormedScoreResults =\
    namedtuple("RankNormedScoreResults",
               ["pattern_idx", "percnormed_score",
                "score", "offset", "revcomp"])


#this has a lot of similarities to
# aggregator.TrimToBestWindow; reduce
# redundancy at some point
class SeqletTrimToBestWindow(object):
    
    def __init__(self, window_size, track_names):
        self.window_size = window_size 
        self.track_names = track_names

    def __call__(self, seqlets):
        trimmed_seqlets = []
        for seqlet in seqlets:
            #trimming to the central window with the largest
            # absolute scores across all the tracks specified
            # in self.track_names
            start_idx = np.argmax(util.cpu_sliding_window_sum(
                arr=np.sum(np.abs(
                    np.concatenate(
                    [seqlet[track_name].fwd
                      .reshape(len(seqlet),-1) for
                     track_name in self.track_names], axis=1)),axis=1),
                window_size=self.window_size))
            end_idx = start_idx + self.window_size
            trimmed_seqlets.append(
                seqlet.trim(
                    start_idx=start_idx,
                    end_idx=end_idx))
        return trimmed_seqlets

#instantiating cross_metric_computer:
#affmat.core.ParallelCpuCrossMetricOnNNpairs(
# n_cores=4, cross_metric_single_region=
#  affmat.core.CrossContinJaccardSingleRegionWithArgmax()
# verbose=True)
    
class PatternsToSeqletsSimComputer(object):
    
    def __init__(self, pattern_comparison_settings,
                       cross_metric_computer,
                       seqlet_trimmer=None):
        self.pattern_comparison_settings =\
            pattern_comparison_settings
        self.cross_metric_computer = cross_metric_computer 
        self.seqlet_trimmer = seqlet_trimmer
        
    def __call__(self, patterns, seqlets):
        #fix the orientation of the seqlets so they are
        # always relative to the forward strand
        seqlets = [x.revcomp() if x.coor.is_revcomp
                   else x for x in seqlets] 
                
        if (self.seqlet_trimmer is not None):
            #trim the seqlets down using seqlet_trimmer
            trimmed_seqlets = self.seqlet_trimmer(seqlets)
        else:
            trimmed_seqlets = seqlets
        #get the offset of the trimmed seqlet relative to the
        # true seqlet start.
        trimmed_offsets = [x[1].coor.start-x[0].coor.start for x in
                           zip(seqlets,trimmed_seqlets)]
        
        #gets the data from the underlying seqlets for the
        # appropriate tracks and with the appropriate normalization
        (fwd_seqlets2ddata, rev_seqlets2ddata) =\
            core.get_2d_data_from_patterns(
                patterns=trimmed_seqlets,
                track_names=self.pattern_comparison_settings.track_names,
                track_transformer=
                    self.pattern_comparison_settings.track_transformer)
        
        #gets the data from the underlying patterns for the
        # appropriate tracks and with the appropriate normalization
        fwd_patterns2ddata, rev_patterns2ddata =\
            core.get_2d_data_from_patterns(
                patterns=patterns,
                track_names=self.pattern_comparison_settings.track_names,
                track_transformer=
                    self.pattern_comparison_settings.track_transformer)
            
        #apply the cross metric 
        # min_overlap is a fraction of "filters"
        # things_to_scan is what gets padded,
        # so it should contain the longer thing
        #scan the fwd seqlets against the fwd patterns
        affmat_fwd = self.cross_metric_computer(
                      filters=fwd_seqlets2ddata,
                      things_to_scan=fwd_patterns2ddata,
                      min_overlap=self.pattern_comparison_settings.min_overlap)
        #in the third dim, the first val is the
        # similarity and the second is the pos
        # of alignment
        assert affmat_fwd.shape==(len(fwd_patterns2ddata),
                                  len(fwd_seqlets2ddata), 2)
        #scan the fwd seqlets against the rev patterns
        affmat_rev = self.cross_metric_computer(
                      filters=fwd_seqlets2ddata,
                      things_to_scan=rev_patterns2ddata,
                      min_overlap=self.pattern_comparison_settings.min_overlap)
        assert affmat_rev.shape==(len(fwd_patterns2ddata),
                                  len(fwd_seqlets2ddata), 2)
        
        #Figure out the maximum similarity between the fwd and rev orientation
        # for each seqlet
        concated_sims = np.concatenate([affmat_fwd[None,:,:,:],
                                        affmat_rev[None,:,:,:]], axis=0)
        #best_match_is_rev_mask is 1 if best match is rev, 0 otherwise
        # has dimensions n_patterns x n_seqlets
        best_match_is_rev_mask = np.argmax(concated_sims[:,:,:,0], axis=0)
        best_match_is_fwd_mask = 1-best_match_is_rev_mask
        #to_return has dimensions n_patterns x n_seqlets x 3
        # first index of last dim is similarity
        # second index of last dim is offset relative to pattern start
        # third index of last dim is "fwd" or "rev".
        to_return = np.concatenate([(
            best_match_is_fwd_mask[:,:,None]*affmat_fwd +
            best_match_is_rev_mask[:,:,None]*affmat_rev),
            best_match_is_rev_mask[:,:,None]], axis=-1)
        #adjust the offsets according to the seqlet trim
        to_return[:,:,1] = (to_return[:,:,1]
                            - np.array(trimmed_offsets)[None,:])
        return to_return


class MaxRankBasedPatternScorer(object):

    def __init__(self, pattern_scorers):
        self.pattern_scorers = pattern_scorers

    def __call__(self, seqlets):
        pattern_seqlet_scores = []
        for pattern_scorer in self.pattern_scorers:
            pattern_seqlet_scores.append(pattern_scorer(seqlets)) 
        to_return = []
        for i in range(len(seqlets)):
            best_pattern_idx = np.argmax(np.array([x[i].percnormed_score
                                          for x in pattern_seqlet_scores]))
            best_res = pattern_seqlet_scores[best_pattern_idx][i]
            to_return.append(
                RankNormedScoreResults(
                    pattern_idx=best_pattern_idx,
                    percnormed_score=best_res.percnormed_score,
                    score=best_res.score, offset=best_res.offset,
                    revcomp=best_res.revcomp)) 
        return to_return


class RankBasedPatternScorer(object):
    
    def __init__(self, aggseqlets,
                       patterns_to_seqlets_sim_computer):
        if (isinstance(aggseqlets, list)==False):
            aggseqlets = [aggseqlets]
        else:
            print("Consider using MaxRankBasedPatternScorer in conjunction"
                   " with individual pattern scorers instead")
        self.aggseqlets = aggseqlets
        self.patterns_to_seqlets_sim_computer =\
            patterns_to_seqlets_sim_computer
        self._build()
    
    def _build(self):
        self.sorted_aggseqlet_selfsimilarities = []
        for aggseqlet in self.aggseqlets:
            member_seqlets = aggseqlet.seqlets
            this_pattern_similarities = self.patterns_to_seqlets_sim_computer(
                patterns = [aggseqlet],
                seqlets = member_seqlets
            )[0,:,0]
            self.sorted_aggseqlet_selfsimilarities.append(
                np.array(sorted(this_pattern_similarities)))
        
    def __call__(self, seqlets):
        
        #compute similarities of trimmed seqlets to the pattern
        patterns_to_seqlets_sim = self.patterns_to_seqlets_sim_computer(
            seqlets=seqlets,
            patterns=self.aggseqlets
        )
        
        #get the percentile rank for each seqlet and pattern
        percnormed_patterns_to_seqlets_sim =\
            np.zeros((len(self.aggseqlets),len(seqlets)))
        
        for (pattern_idx,pattern_selfsims) in\
             enumerate(self.sorted_aggseqlet_selfsimilarities):
            percentiles = (np.searchsorted(
                pattern_selfsims,
                patterns_to_seqlets_sim[pattern_idx,:,0])
            /float(len(pattern_selfsims)))
            percnormed_patterns_to_seqlets_sim[pattern_idx] = percentiles
        
        best_pattern_indices = np.argmax(percnormed_patterns_to_seqlets_sim,
                                            axis=0)
        to_return = [
            RankNormedScoreResults(
             pattern_idx=(best_pattern_idx
                          if len(self.aggseqlets) > 0 else None),
             percnormed_score=percnormed_patterns_to_seqlets_sim[
                               best_pattern_idx, seqlet_idx],
             score=patterns_to_seqlets_sim[best_pattern_idx, seqlet_idx][0],
             offset=patterns_to_seqlets_sim[best_pattern_idx, seqlet_idx][1],
             revcomp=patterns_to_seqlets_sim[best_pattern_idx, seqlet_idx][2])
            for (seqlet_idx, best_pattern_idx)
                 in enumerate(best_pattern_indices)
        ]
        
        return to_return
