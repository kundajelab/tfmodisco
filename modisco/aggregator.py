from __future__ import division, print_function, absolute_import
import numpy as np
from . import affinitymat
from . import core
from . import util
from . import backend as B
from collections import OrderedDict
import itertools
import sys


class AbstractAggSeqletPostprocessor(object):

    def __call__(self, aggregated_seqlets):
        raise NotImplementedError() #return an array

    def chain(self, postprocessor):
        return AdhocAggSeqletPostprocessor(
                func=lambda x: postprocessor(self(x)))


class AdhocAggSeqletPostprocessor(AbstractAggSeqletPostprocessor):

    def __init__(self, func):
        self.func = func

    def __call__(self, aggregated_seqlets):
        return self.func(aggregated_seqlets)


class TrimToFracSupport(AbstractAggSeqletPostprocessor):

    def __init__(self, frac):
        self.frac = frac

    def __call__(self, aggregated_seqlets):
        return [x.trim_to_positions_with_frac_support_of_peak(
                  frac=self.frac) for x in aggregated_seqlets]


class TrimToBestWindow(AbstractAggSeqletPostprocessor):

    def __init__(self, window_size, track_names):
        self.window_size = window_size
        self.track_names = track_names

    def __call__(self, aggregated_seqlets):
        trimmed_agg_seqlets = []
        for aggregated_seqlet in aggregated_seqlets:
            start_idx = np.argmax(util.cpu_sliding_window_sum(
                arr=np.sum(np.abs(
                    np.concatenate(
                    [aggregated_seqlet[track_name].fwd
                      .reshape(len(aggregated_seqlet),-1) for
                     track_name in self.track_names], axis=1)),axis=1),
                window_size=self.window_size))
            end_idx = start_idx + self.window_size
            trimmed_agg_seqlets.append(
                aggregated_seqlet.trim_to_start_and_end_idx(
                    start_idx=start_idx, end_idx=end_idx)) 
        return trimmed_agg_seqlets


class ExpandSeqletsToFillPattern(AbstractAggSeqletPostprocessor):

    def __init__(self, track_set, flank_to_add=0,
                       track_names=None, verbose=True):
        self.track_set = track_set 
        self.track_names = track_names
        self.verbose = verbose
        self.flank_to_add = flank_to_add

    def __call__(self, aggregated_seqlets):
        new_aggregated_seqlets = []
        for aggregated_seqlet in aggregated_seqlets:
            new_seqlets_and_alnmts = []
            skipped_seqlets = 0
            for seqlet_and_alnmt in aggregated_seqlet.seqlets_and_alnmts:
                seqlet = seqlet_and_alnmt.seqlet
                alnmt = seqlet_and_alnmt.alnmt
                left_expansion = alnmt+self.flank_to_add 
                right_expansion = (len(aggregated_seqlet) -
                                    (alnmt+len(seqlet)))+self.flank_to_add
                if (seqlet.coor.is_revcomp == False):
                    start = seqlet.coor.start - left_expansion
                    end = seqlet.coor.end + right_expansion
                else:
                    start = seqlet.coor.start - right_expansion
                    end = seqlet.coor.end + left_expansion
                if (start >= 0 and end <= self.track_set.track_length):
                    seqlet = self.track_set.create_seqlet(
                        coor=core.SeqletCoordinates(
                            example_idx=seqlet.coor.example_idx,
                            start=start, end=end,
                            is_revcomp=seqlet.coor.is_revcomp),
                        track_names=self.track_names) 
                    new_seqlets_and_alnmts.append(
                     core.SeqletAndAlignment(seqlet=seqlet, alnmt=0))
                else:
                    skipped_seqlets += 1 
            if self.verbose and (skipped_seqlets > 0):
                print("Skipped "+str(skipped_seqlets)+" seqlets") 
                sys.stdout.flush()
            if (len(new_seqlets_and_alnmts) > 0):
                new_aggregated_seqlets.append(core.AggregatedSeqlet(
                    seqlets_and_alnmts_arr=new_seqlets_and_alnmts))
        return new_aggregated_seqlets 


class AbstractTwoDMatSubclusterer(object):

    def __call__(self, twod_mat):
        #return subcluster_indices, num_subclusters
        raise NotImplementedError()


class RecursiveKmeans(AbstractTwoDMatSubclusterer):

    def __init__(self, threshold, minimum_size_for_splitting, verbose=True):
        self.threshold = threshold
        self.minimum_size_for_splitting = minimum_size_for_splitting
        self.verbose = verbose

    def __call__(self, twod_mat):
        import sklearn.cluster

        if (len(twod_mat) < self.minimum_size_for_splitting):
            print("No split; cluster size is "+str(len(twod_mat)))
            return np.zeros(len(twod_mat))
            
        cluster_indices = sklearn.cluster.KMeans(n_clusters=2).\
                               fit_predict(twod_mat)

        cluster1_mean = np.mean(twod_mat[cluster_indices==0], axis=0)
        cluster2_mean = np.mean(twod_mat[cluster_indices==1], axis=0)
        dist = ((np.sum(np.minimum(np.abs(cluster1_mean),
                                   np.abs(cluster1_mean))*
                        np.sign(cluster1_mean)*
                        np.sign(cluster2_mean)))/
                 np.sum(np.maximum(np.abs(cluster1_mean),
                                   np.abs(cluster2_mean)))) 
        #dist = np.sum(cluster1_mean*cluster2_mean)/(
        #               np.linalg.norm(cluster1_mean)
        #               *np.linalg.norm(cluster2_mean))

        if (dist > self.threshold):
            print("No split; similarity is "+str(dist)+" and "
                  "cluster size is "+str(len(twod_mat)))
            return np.zeros(len(twod_mat))
        else:
            if (self.verbose):
                print("Split detected; similarity is "+str(dist)+" and "
                      "cluster size is "+str(len(twod_mat)))
                sys.stdout.flush()
            for i in range(2):
                max_cluster_idx = np.max(cluster_indices)
                mask_for_this_cluster = (cluster_indices==i)
                subcluster_indices = self(twod_mat[mask_for_this_cluster])
                subcluster_indices = np.array(
                    [i if x==0 else x+max_cluster_idx
                     for x in subcluster_indices])
                cluster_indices[mask_for_this_cluster]=subcluster_indices
            return cluster_indices 


class DetectSpuriousMerging(AbstractAggSeqletPostprocessor):

    def __init__(self, track_names, track_transformer,
                       subclusters_detector, verbose=True):
        self.verbose = verbose
        self.track_names = track_names
        self.track_transformer = track_transformer
        self.subclusters_detector = subclusters_detector

    def __call__(self, aggregated_seqlets):
        to_return = []
        for agg_seq_idx, aggregated_seqlet in enumerate(aggregated_seqlets):
            if (self.verbose):
                print("Inspecting for spurious merging")
                sys.stdout.flush()
            assert len(set(len(x.seqlet) for x in
                       aggregated_seqlet._seqlets_and_alnmts))==1,\
                ("all seqlets should be same length; use "+
                 "ExpandSeqletsToFillPattern to equalize lengths")
            fwd_seqlet_data = aggregated_seqlet.get_fwd_seqlet_data(
                                track_names=self.track_names,
                                track_transformer=self.track_transformer)
            sum_per_position = np.sum(np.abs(fwd_seqlet_data),axis=-1)
            subcluster_indices = self.subclusters_detector(sum_per_position)
            num_subclusters = np.max(subcluster_indices)+1
            if (num_subclusters > 1):
                if (self.verbose):
                    print("Got "+str(num_subclusters)+" subclusters")
                    sys.stdout.flush()
                for i in range(num_subclusters):
                    seqlets_and_alnmts_for_subcluster = [x[0] for x in
                        zip(aggregated_seqlet._seqlets_and_alnmts,
                            subcluster_indices) if x[1]==i]
                    to_return.append(core.AggregatedSeqlet(
                     seqlets_and_alnmts_arr=seqlets_and_alnmts_for_subcluster))
            else:
                to_return.append(aggregated_seqlet)
        return to_return


class ReassignSeqletsFromSmallClusters(AbstractAggSeqletPostprocessor):

    def __init__(self, seqlet_assigner,
                       min_cluster_size, postprocessor, verbose):
        self.seqlet_assigner = seqlet_assigner
        self.min_cluster_size = min_cluster_size
        self.postprocessor = postprocessor
        self.verbose = verbose

    def __call__(self, patterns):
        
        #do a final assignment
        small_patterns =\
            [x for x in patterns if x.num_seqlets < self.min_cluster_size]
        large_patterns =\
            [x for x in patterns if x.num_seqlets >= self.min_cluster_size]
        seqlets_to_assign = list(itertools.chain(
            *[[x.seqlet for x in pattern._seqlets_and_alnmts]
              for pattern in small_patterns]))
        if (len(large_patterns) > 0):
            if (len(seqlets_to_assign) > 0):
                large_patterns, new_assignments =\
                    self.seqlet_assigner(patterns=large_patterns,
                                         seqlets_to_assign=seqlets_to_assign,
                                         merge_into_existing_patterns=True)
            large_patterns = self.postprocessor(large_patterns)
            return large_patterns
        else:
            return []


class ReassignSeqletsTillConvergence(AbstractAggSeqletPostprocessor):

    def __init__(self, seqlet_assigner, percent_change_tolerance, max_rounds,
                       postprocessor, verbose):
        self.seqlet_assigner = seqlet_assigner
        self.percent_change_tolerance = percent_change_tolerance
        self.max_rounds = max_rounds
        self.postprocessor = postprocessor
        self.verbose = verbose

    def __call__(self, patterns):
        
        for a_round in range(self.max_rounds):
            if (self.verbose):
                print("On reassignment round "+str(a_round))
            all_seqlets = list(itertools.chain(
                *[[x.seqlet for x in pattern._seqlets_and_alnmts]
                  for pattern in patterns]))
            initial_assignments = list(itertools.chain(
                *[[pattern_idx for x in pattern._seqlets_and_alnmts]
                  for pattern_idx, pattern in enumerate(patterns)]))
            patterns, new_assignments =\
                self.seqlet_assigner(patterns=patterns,
                                     seqlets_to_assign=all_seqlets,
                                     merge_into_existing_patterns=False)
            patterns = self.postprocessor(patterns)
            changed_assignments = np.sum(
                1-(np.array(initial_assignments)==np.array(new_assignments)))
            percent_change = 100*changed_assignments/\
                                float(len(initial_assignments))
            if (self.verbose):
                print("Percent assignments changed: "
                      +str(round(percent_change,2)))
            if (percent_change <= self.percent_change_tolerance):
                break 

        return patterns


#reassign seqlets to best match motif
class AssignSeqletsByBestMetric(object):

    def __init__(self, pattern_comparison_settings,
                       individual_aligner_metric,
                       matrix_affinity_metric,
                       min_similarity=0.0,
                       verbose=True):
        self.pattern_comparison_settings = pattern_comparison_settings
        self.pattern_aligner = core.CrossMetricPatternAligner( 
                    pattern_comparison_settings=pattern_comparison_settings,
                    metric=individual_aligner_metric)
        self.matrix_affinity_metric = matrix_affinity_metric
        self.min_similarity = min_similarity
        self.verbose = verbose

    def __call__(self, patterns, seqlets_to_assign,
                       merge_into_existing_patterns):

        (pattern_fwd_data, pattern_rev_data) =\
            core.get_2d_data_from_patterns(
                patterns=patterns,
                track_names=self.pattern_comparison_settings.track_names,
                track_transformer=
                    self.pattern_comparison_settings.track_transformer)
        (seqlet_fwd_data, seqlet_rev_data) =\
            core.get_2d_data_from_patterns(
                patterns=seqlets_to_assign,
                track_names=self.pattern_comparison_settings.track_names,
                track_transformer=
                    self.pattern_comparison_settings.track_transformer)

        cross_metric_fwd = self.matrix_affinity_metric(
                     filters=pattern_fwd_data,
                     things_to_scan=seqlet_fwd_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap) 
        cross_metric_rev = self.matrix_affinity_metric(
                     filters=pattern_fwd_data,
                     things_to_scan=seqlet_rev_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap) 
        cross_metrics = np.maximum(cross_metric_fwd, cross_metric_rev)
        assert cross_metrics.shape == (len(patterns), len(seqlets_to_assign))
        seqlet_assignments = np.argmax(cross_metrics, axis=0) 
        seqlet_assignment_scores = np.max(cross_metrics, axis=0)

        seqlet_and_alnmnt_grps = [[] for x in patterns]
        discarded_seqlets = 0
        for seqlet_idx, (assignment, score)\
            in enumerate(zip(seqlet_assignments, seqlet_assignment_scores)):
            if (score >= self.min_similarity):
                alnmt, revcomp_match, score = self.pattern_aligner(
                    parent_pattern=patterns[assignment],
                    child_pattern=seqlets_to_assign[seqlet_idx]) 
                if (revcomp_match):
                    seqlet = seqlets_to_assign[seqlet_idx].revcomp()
                else:
                    seqlet = seqlets_to_assign[seqlet_idx]
                seqlet_and_alnmnt_grps[assignment].append(
                    core.SeqletAndAlignment(
                        seqlet=seqlets_to_assign[seqlet_idx],
                        alnmt=alnmt))
            else:
                seqlet_assignments[seqlet_idx] = -1
                discarded_seqlets += 1

        if (self.verbose):
            if discarded_seqlets > 0:
                print("Discarded "+str(discarded_seqlets)+" seqlets") 
                sys.stdout.flush()

        if (merge_into_existing_patterns):
            new_patterns = patterns
            for pattern,x in zip(patterns, seqlet_and_alnmnt_grps):
                pattern.merge_seqlets_and_alnmts(
                    seqlets_and_alnmts=x,
                    aligner=self.pattern_aligner) 
        else:
            new_patterns = [core.AggregatedSeqlet(seqlets_and_alnmts_arr=x)
                for x in seqlet_and_alnmnt_grps if len(x) > 0]

        return new_patterns, seqlet_assignments


#reassign seqlets to best match motif
class AssignSeqletsByBestCrossCorr(object):

    def __init__(self, pattern_crosscorr_settings,
                       min_similarity=0.0,
                       verbose=True, batch_size=50, progress_update=1000,
                       func_params_size=1000000):

        self.pattern_crosscorr_settings = pattern_crosscorr_settings
        self.pattern_aligner = core.CrossCorrelationPatternAligner(
                        pattern_comparison_settings=pattern_crosscorr_settings)
        self.min_similarity = min_similarity
        self.verbose = verbose
        self.batch_size = batch_size
        self.progress_update = progress_update
        self.func_params_size = func_params_size

    def __call__(self, patterns, seqlets_to_assign,
                       merge_into_existing_patterns):

        (pattern_fwd_data, pattern_rev_data) =\
            core.get_2d_data_from_patterns(
                patterns=patterns,
                track_names=self.pattern_crosscorr_settings.track_names,
                track_transformer=
                    self.pattern_crosscorr_settings.track_transformer)
        (seqlet_fwd_data, seqlet_rev_data) =\
            core.get_2d_data_from_patterns(
                patterns=seqlets_to_assign,
                track_names=self.pattern_crosscorr_settings.track_names,
                track_transformer=
                    self.pattern_crosscorr_settings.track_transformer)

        cross_corrs_fwd = B.max_cross_corrs(
                     filters=pattern_fwd_data,
                     things_to_scan=seqlet_fwd_data,
                     min_overlap=self.pattern_crosscorr_settings.min_overlap,
                     batch_size=self.batch_size,
                     func_params_size=self.func_params_size,
                     progress_update=self.progress_update) 
        cross_corrs_rev = B.max_cross_corrs(
                     filters=pattern_fwd_data,
                     things_to_scan=seqlet_rev_data,
                     min_overlap=self.pattern_crosscorr_settings.min_overlap,
                     batch_size=self.batch_size,
                     func_params_size=self.func_params_size,
                     progress_update=self.progress_update) 
        cross_corrs = np.maximum(cross_corrs_fwd, cross_corrs_rev)
        assert cross_corrs.shape == (len(patterns), len(seqlets_to_assign))
        seqlet_assignments = np.argmax(cross_corrs, axis=0) 
        seqlet_assignment_scores = np.max(cross_corrs, axis=0)

        seqlet_and_alnmnt_grps = [[] for x in patterns]
        discarded_seqlets = 0
        for seqlet_idx, (assignment, score)\
            in enumerate(zip(seqlet_assignments, seqlet_assignment_scores)):
            if (score >= self.min_similarity):
                alnmt, revcomp_match, score = self.pattern_aligner(
                    parent_pattern=patterns[assignment],
                    child_pattern=seqlets_to_assign[seqlet_idx]) 
                if (revcomp_match):
                    seqlet = seqlets_to_assign[seqlet_idx].revcomp()
                else:
                    seqlet = seqlets_to_assign[seqlet_idx]
                seqlet_and_alnmnt_grps[assignment].append(
                    core.SeqletAndAlignment(
                        seqlet=seqlets_to_assign[seqlet_idx],
                        alnmt=alnmt))
            else:
                seqlet_assignments[seqlet_idx] = -1
                discarded_seqlets += 1

        if (self.verbose):
            if discarded_seqlets > 0:
                print("Discarded "+str(discarded_seqlets)+" seqlets") 
                sys.stdout.flush()

        if (merge_into_existing_patterns):
            new_patterns = patterns
            for pattern,x in zip(patterns, seqlet_and_alnmnt_grps):
                pattern.merge_seqlets_and_alnmts(
                    seqlets_and_alnmts=x,
                    aligner=self.pattern_aligner) 
        else:
            new_patterns = [core.AggregatedSeqlet(seqlets_and_alnmts_arr=x)
                for x in seqlet_and_alnmnt_grps if len(x) > 0]

        return new_patterns, seqlet_assignments


class SeparateOnSeqletCenterPeaks(AbstractAggSeqletPostprocessor):

    def __init__(self, min_support, pattern_aligner, verbose=True):
        self.verbose = verbose
        self.min_support = min_support
        self.pattern_aligner = pattern_aligner

    def __call__(self, aggregated_seqlets):
        to_return = []
        for agg_seq_idx, aggregated_seqlet in enumerate(aggregated_seqlets):
            to_return.append(aggregated_seqlet)
            seqlet_centers =\
                aggregated_seqlet.get_per_position_seqlet_center_counts()
            #find the peak indices
            enumerated_peaks = list(enumerate([x for x in
                util.identify_peaks(seqlet_centers) if x[1]
                >= self.min_support]))
            if (len(enumerated_peaks) > 1): 
                separated_seqlets_and_alnmts =\
                    [list() for x in enumerated_peaks]
                if (self.verbose):
                    print("Found "+str(len(enumerated_peaks))+" peaks") 
                    sys.stdout.flush()
                #sort the seqlets by the peak whose center they are
                #closest to 
                seqlets_and_alnmts = aggregated_seqlet._seqlets_and_alnmts
                closest_peak_idxs = []
                for seqlet_and_alnmt in seqlets_and_alnmts:
                    seqlet_mid = seqlet_and_alnmt.alnmt +\
                                 int(0.5*len(seqlet_and_alnmt.seqlet)) 
                    closest_peak_idx =\
                     min(enumerated_peaks,
                         key=lambda x: np.abs(seqlet_mid-x[1][0]))[0] 
                    separated_seqlets_and_alnmts[closest_peak_idx]\
                        .append(seqlet_and_alnmt)

                ##now create aggregated seqlets for the set of seqlets
                ##assigned to each peak 
                proposed_new_patterns = [
                    core.AggregatedSeqlet(seqlets_and_alnmts_arr=x)
                    for x in separated_seqlets_and_alnmts]
                #to_return.extend(proposed_new_patterns)

                #having formulated the proposed new patterns, go back
                #and figure out which pattern each seqlet best aligns to
                final_separated_seqlets_and_alnmnts =\
                    [list() for x in proposed_new_patterns]
                for seqlet_and_alnmt in seqlets_and_alnmts:
                    best_pattern_idx, (alnmt, revcomp_match, score) =\
                        max([(idx, self.pattern_aligner(parent_pattern=x,
                                child_pattern=seqlet_and_alnmt.seqlet))
                            for idx,x in enumerate(proposed_new_patterns)],
                        key=lambda x: x[1][2])
                    if (revcomp_match):
                        seqlet = seqlet_and_alnmt.seqlet.revcomp() 
                    else:
                        seqlet = seqlet_and_alnmt.seqlet
                    final_separated_seqlets_and_alnmnts[best_pattern_idx]\
                        .append(core.SeqletAndAlignment(seqlet=seqlet,
                                                        alnmt=alnmt))

                #get the final patterns from the final seqlet assignment 
                final_new_patterns = [
                    core.AggregatedSeqlet(seqlets_and_alnmts_arr=x)
                    for x in final_separated_seqlets_and_alnmnts]
                to_return.extend(final_new_patterns) 

        return to_return


class AbstractSeqletsAggregator(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class HierarchicalSeqletAggregator(object):

    def __init__(self, pattern_aligner, affinity_mat_from_seqlets,
                       postprocessor=None):
        self.pattern_aligner = pattern_aligner
        self.affinity_mat_from_seqlets = affinity_mat_from_seqlets
        self.postprocessor = postprocessor

    def __call__(self, seqlets):
        affinity_mat = self.affinity_mat_from_seqlets(seqlets)
        return self.aggregate_seqlets_by_affinity_mat(
                    seqlets=seqlets, affinity_mat=affinity_mat)

    def aggregate_seqlets_by_affinity_mat(self, seqlets, affinity_mat):

        aggregated_seqlets = [core.AggregatedSeqlet.from_seqlet(x)
                              for x in seqlets]
        #get the affinity mat as a list of 3-tuples
        affinity_tuples = []
        for i in range(len(affinity_mat)-1):
            for j in range(i+1,len(affinity_mat)):
                affinity_tuples.append((affinity_mat[i,j],i,j))
        #sort to get closest first
        affinity_tuples = sorted(affinity_tuples, key=lambda x: -x[0])

        #now repeatedly merge, unless already merged.
        for (affinity,i,j) in affinity_tuples:
            aggregated_seqlet_i = aggregated_seqlets[i]
            aggregated_seqlet_j = aggregated_seqlets[j]
            #if they are not already the same aggregation object...
            if (aggregated_seqlet_i != aggregated_seqlet_j):
                if (aggregated_seqlet_i.num_seqlets <
                    aggregated_seqlet_j.num_seqlets):
                    parent_agg_seqlet = aggregated_seqlet_j 
                    child_agg_seqlet = aggregated_seqlet_i
                else:
                    parent_agg_seqlet = aggregated_seqlet_i
                    child_agg_seqlet = aggregated_seqlet_j
                parent_agg_seqlet.merge_aggregated_seqlet(
                    agg_seqlet=child_agg_seqlet,
                    aligner=self.pattern_aligner)
                for k in range(len(aggregated_seqlets)):
                    if (aggregated_seqlets[k] == child_agg_seqlet):
                        aggregated_seqlets[k] = parent_agg_seqlet

        initial_aggregated = list(set(aggregated_seqlets))
        assert len(initial_aggregated)==1,\
            str(len(initial_aggregated))+" "+str(initial_aggregated)
        
        to_return = initial_aggregated
        if (self.postprocessor is not None):
            to_return = self.postprocessor(to_return)
        
        #sort by number of seqlets in each 
        return sorted(to_return,
                      key=lambda x: -x.num_seqlets)


class SimilarPatternsCollapser(object):

    def __init__(self, pattern_aligner,
                       merging_threshold,
                       postprocessor,
                       verbose=True):
        self.pattern_aligner = pattern_aligner
        self.merging_threshold = merging_threshold
        self.verbose=verbose
        self.postprocessor = postprocessor

    def __call__(self, original_patterns):
        #make a copy of the patterns
        original_patterns = [x.copy() for x in original_patterns]
        for i in range(len(original_patterns)):
            for j in range(len(original_patterns[i:])):
                pattern1 = original_patterns[i]
                pattern2 = original_patterns[i+j]
                if (pattern1 != pattern2): #if not the same object
                    if (pattern1.num_seqlets < pattern2.num_seqlets):
                        parent_pattern, child_pattern = pattern2, pattern1
                    else:
                        parent_pattern, child_pattern = pattern1, pattern2
                    (best_crosscorr_argmax, is_revcomp, best_crosscorr) =\
                        self.pattern_aligner(parent_pattern=parent_pattern,
                                             child_pattern=child_pattern)  
                    if (best_crosscorr > self.merging_threshold): 
                        if (self.verbose):
                            print("Collapsing "+str(i)+" & "+str(j+i)
                                 +" with similarity "+str(best_crosscorr)) 
                            sys.stdout.flush()
                        parent_pattern.merge_aggregated_seqlet(
                            agg_seqlet=child_pattern,
                            aligner=self.pattern_aligner) 
                        new_parent_pattern =\
                            self.postprocessor([parent_pattern])
                        assert len(new_parent_pattern)==1
                        new_parent_pattern = new_parent_pattern[0]
                        for k in range(len(original_patterns)):
                            if (original_patterns[k]==parent_pattern or
                                original_patterns[k]==child_pattern):
                                original_patterns[k]=new_parent_pattern
                    else:
                        if (self.verbose):
                            print("Not collapsing "+str(i)+" & "+str(j+i)
                                 +" with similarity "+str(best_crosscorr)) 
                            sys.stdout.flush()

        return sorted(self.postprocessor(list(set(original_patterns))),
                      key=lambda x: -x.num_seqlets)
