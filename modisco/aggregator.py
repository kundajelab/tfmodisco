from __future__ import division, print_function, absolute_import
import numpy as np
from . import affinitymat
from . import core
from . import util
from collections import OrderedDict, defaultdict
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

    def __init__(self, min_frac, min_num, verbose):
        self.min_frac = min_frac
        self.min_num = min_num
        self.verbose = verbose

    def __call__(self, aggregated_seqlets):
        return [x.trim_to_positions_with_min_support(
                  min_frac=self.min_frac,
                  min_num=self.min_num,
                  verbose=self.verbose) for x in aggregated_seqlets]


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
                if (start >= 0 and
                    end <=
                     self.track_set.get_example_idx_len(
                           seqlet.coor.example_idx)):
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


class IsDissimilarFunc(object):

    def __init__(self, threshold, sim_func, verbose=False):
        self.threshold = threshold
        self.sim_func = sim_func
        self.verbose = verbose

    def __call__(self, inp1, inp2):
        sim = self.sim_func(inp1, inp2)
        is_dissimilar = (sim < self.threshold)
        if (self.verbose):
            print("Similarity is "+str(sim)
                  +"; is_dissimilar is "+str(is_dissimilar))  
            sys.stdout.flush()
        return is_dissimilar


def pearson_corr(x, y):
    x = x - np.mean(x)
    x = x/(np.linalg.norm(x))
    y = y - np.mean(y)
    y = y/np.linalg.norm(y)
    return np.sum(x*y)


class PearsonCorrIsDissimilarFunc(IsDissimilarFunc):

    def __init__(self, threshold, verbose):
        super(PearsonCorrIsDissimilarFunc, self).__init__(
            threshold=threshold,
            sim_func=pearson_corr,
            verbose=verbose)


class DetectSpuriousMerging(AbstractAggSeqletPostprocessor):

    def __init__(self, track_names, track_transformer,
                       affmat_from_1d, diclusterer,
                       is_dissimilar_func,
                       min_in_subcluster, verbose=True):
        self.track_names = track_names
        self.track_transformer = track_transformer
        self.affmat_from_1d = affmat_from_1d
        self.diclusterer = diclusterer
        self.is_dissimilar_func = is_dissimilar_func
        self.min_in_subcluster = min_in_subcluster
        self.verbose = verbose

    def cluster_fwd_seqlet_data(self, fwd_seqlet_data, affmat):

        if (len(fwd_seqlet_data) < self.min_in_subcluster):
            return np.zeros(len(fwd_seqlet_data)) 
        if (self.verbose):
            print("Inspecting for spurious merging")
            sys.stdout.flush()
        dicluster_results = self.diclusterer(affmat) 
        dicluster_indices = dicluster_results.cluster_indices 
        assert np.max(dicluster_indices)==1 or np.max(dicluster_indices==0)
        if (np.sum(dicluster_indices==1)==0 or
            np.sum(dicluster_indices==0)==0):
            sufficiently_dissimilar = False
        else:
            #check that the subclusters are more
            #dissimilar than sim_split_threshold 
            mat1_agg = np.mean(fwd_seqlet_data[dicluster_indices==0], axis=0)
            mat2_agg = np.mean(fwd_seqlet_data[dicluster_indices==1], axis=0)
            sufficiently_dissimilar = self.is_dissimilar_func(
                                            inp1=mat1_agg, inp2=mat2_agg)
        if (not sufficiently_dissimilar):
            return np.zeros(len(fwd_seqlet_data))
        else:
            #if sufficiently dissimilar, check for subclusters
            cluster_indices = np.array(dicluster_indices)
            for i in [0, 1]:
                mask_for_this_cluster = dicluster_indices==i  
                subcluster_indices =\
                    self.cluster_fwd_seqlet_data(
                        fwd_seqlet_data=fwd_seqlet_data[mask_for_this_cluster],
                        affmat=(affmat[mask_for_this_cluster]
                                     [:,mask_for_this_cluster]))
                subcluster_indices = np.array([
                    i if x==0 else x+1 for x in subcluster_indices])
                cluster_indices[mask_for_this_cluster] = subcluster_indices 
            return cluster_indices

    def __call__(self, aggregated_seqlets):
        to_return = []
        for agg_seq_idx, aggregated_seqlet in enumerate(aggregated_seqlets):

            assert len(set(len(x.seqlet) for x in
                       aggregated_seqlet._seqlets_and_alnmts))==1,\
                ("all seqlets should be same length; use "+
                 "ExpandSeqletsToFillPattern to equalize lengths")
            fwd_seqlet_data = aggregated_seqlet.get_fwd_seqlet_data(
                                track_names=self.track_names,
                                track_transformer=self.track_transformer)
            vecs_1d = np.array([x.ravel() for x in fwd_seqlet_data]) 
            affmat = self.affmat_from_1d(vecs1=vecs_1d, vecs2=vecs_1d)

            subcluster_indices = self.cluster_fwd_seqlet_data(
                                    fwd_seqlet_data=fwd_seqlet_data,
                                    affmat=affmat)  
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
        if (seqlet_rev_data is not None):
            cross_metric_rev = self.matrix_affinity_metric(
                     filters=pattern_fwd_data,
                     things_to_scan=seqlet_rev_data,
                     min_overlap=self.pattern_comparison_settings.min_overlap) 
        if (seqlet_rev_data is not None):
            cross_metrics = np.maximum(cross_metric_fwd, cross_metric_rev)
        else:
            cross_metrics = cross_metric_fwd
            
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


class GreedySeqletAggregator(AbstractSeqletsAggregator):

    def __init__(self, pattern_aligner,
                       seqlet_sort_metric,
                       postprocessor=None):
        self.pattern_aligner = pattern_aligner
        self.seqlet_sort_metric = seqlet_sort_metric
        self.postprocessor = postprocessor

    def __call__(self, seqlets):
        sorted_seqlets = sorted(seqlets,
                                key=self.seqlet_sort_metric) 
        aggregated_seqlet = core.AggregatedSeqlet.from_seqlet(
                                                   sorted_seqlets[0])
        if (len(sorted_seqlets) > 1):
            for seqlet in sorted_seqlets[1:]:
                aggregated_seqlet.merge_aggregated_seqlet(
                    agg_seqlet=core.AggregatedSeqlet.from_seqlet(seqlet),
                    aligner=self.pattern_aligner) 
        to_return = [aggregated_seqlet]
        if (self.postprocessor is not None):
            to_return = self.postprocessor(to_return)

        #sort by number of seqlets in each 
        return sorted(to_return,
                      key=lambda x: -x.num_seqlets)


class HierarchicalSeqletAggregator(AbstractSeqletsAggregator):

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


class AbstractMergeAlignedPatternsCondition(object):

    def __call__(self, parent_pattern, child_pattern, alnmt):
        raise NotImplementedError()

    def chain(self, other_merge_aligned_patterns_condition):
        return AdhocMergeAlignedPatternsCondition( 
                lambda parent_pattern, child_pattern, alnmt:
                 (self(parent_pattern, child_pattern, alnmt) and
                  other_merge_aligned_patterns_condition(
                    parent_pattern=parent_pattern,
                    child_pattern=child_pattern, alnmt=alnmt)))


class AdhocMergeAlignedPatternsCondition(
        AbstractMergeAlignedPatternsCondition):

    def __init__(self, func):
        self.func = func

    def __call__(self, parent_pattern, child_pattern, alnmt):
        return self.func(parent_pattern=parent_pattern,
                         child_pattern=child_pattern, alnmt=alnmt)


class PatternMergeHierarchy(object):

    def __init__(self, root_nodes):
        self.root_nodes = root_nodes

    def add_level(self, level_arr):
        self.levels.append(level_arr)


class PatternMergeHierarchyNode(object):

    def __init__(self, pattern, child_nodes=None, parent_node=None): 
        self.pattern = pattern 
        if (child_nodes is None):
            child_nodes = []
        self.child_nodes = child_nodes
        self.parent_node = parent_node


class DynamicThresholdSimilarPatternsCollapser(object):

    def __init__(self, pattern_to_seqlet_sim_computer,
                       pattern_aligner,
                       collapse_condition, dealbreaker_condition,
                       postprocessor,
                       verbose=True):
        self.pattern_to_seqlet_sim_computer = pattern_to_seqlet_sim_computer
        self.pattern_aligner = pattern_aligner
        self.collapse_condition = collapse_condition
        self.dealbreaker_condition = dealbreaker_condition
        self.postprocessor = postprocessor
        self.verbose = verbose

    def __call__(self, patterns, seqlets):

        patterns = [x.copy() for x in patterns]
        merge_hierarchy_levels = []        
        current_level_nodes = [
            PatternMergeHierarchyNode(pattern=x) for x in patterns]
        merge_hierarchy_levels.append(current_level_nodes)

        merge_occurred_last_iteration = True
        merging_iteration = 0

        #loop until no more patterns get merged
        while (merge_occurred_last_iteration):
            
            merging_iteration += 1
            if (self.verbose):
                print("On merging iteration",merging_iteration) 
                sys.stdout.flush()
            merge_occurred_last_iteration = False

            if (self.verbose):
                print("Computing pattern to seqlet similarities")
                sys.stdout.flush()
            patterns_to_seqlets_sims = self.pattern_to_seqlet_sim_computer(
                                            seqlets=seqlets,
                                            filter_seqlets=patterns)
            assert len(pattern_to_seqlets_sims)==len(seqlets)
            assert len(pattern_to_seqlets_sims[0])==len(patterns)
            if (self.verbose):
                print("Computing pattern to pattern similarities")
                sys.stdout.flush()
            patterns_to_patterns_aligner_sim =\
                np.zeros((len(patterns), len(patterns))) 
            for i,pattern1 in enumerate(patterns):
                for j,pattern2 in enumerate(patterns):
                    (alnmt, rc, aligner_sim) =\
                        self.pattern_aligner(pattern1, pattern2)
                    patterns_to_patterns_aligner_sim[i,j] = aligner_sim

            indices_to_merge = []
            merge_partners_so_far = dict([(i, set([i])) for i in
                                          range(len(patterns))])

            #merge patterns with highest similarity first
            sorted_pairs = sorted([(i,j,patterns_to_patterns_aligner_sim[i,j])
                            for i in range(len(patterns))
                            for j in range(len(patterns)) if (i < j)],
                            key=lambda x: -x[2])
            #iterate over pairs
            for (i,j,aligner_sim) in sorted_pairs:
                dist_prob = min(patterns_dist_probs[i,j],
                                patterns_dist_probs[j,i])
                if (self.collapse_condition(dist_prob=dist_prob,
                                            aligner_sim=aligner_sim)):
                    if (self.verbose):
                        print("Collapsing "+str(i)
                              +" & "+str(j)
                              +" with prob "+str(dist_prob)+" and"
                              +" sim "+str(aligner_sim)) 
                        sys.stdout.flush()

                    collapse_passed = True
                    #check compatibility for all indices that are
                    #about to be merged
                    merge_under_consideration = set(
                        list(merge_partners_so_far[i])
                        +list(merge_partners_so_far[j]))
                    for m1 in merge_under_consideration:
                        for m2 in merge_under_consideration:
                            if (m1 < m2):
                                min_dist_prob_here =\
                                    min(patterns_dist_probs[m1, m2],
                                        patterns_dist_probs[m2, m1])
                                aligner_sim_here =\
                                    patterns_to_patterns_aligner_sim[
                                        m1, m2]
                                if (self.dealbreaker_condition(
                                        dist_prob=min_dist_prob_here,
                                        aligner_sim=aligner_sim_here)):
                                    collapse_passed=False                     
                                    if (self.verbose):
                                        print("Aborting collapse as "
                                              +str(m1)
                                              +" & "+str(m2)
                                              +" have prob "
                                              +str(min_dist_prob_here)
                                              +" and"
                                              +" sim "
                                              +str(aligner_sim_here)) 
                                        sys.stdout.flush()
                                    break

                    if (collapse_passed):
                        indices_to_merge.append((i,j))
                        for an_idx in merge_under_consideration:
                            merge_partners_so_far[an_idx]=\
                                merge_under_consideration 
                else:
                    if (self.verbose):
                        pass
                        #print("Not collapsed "+str(i)+" & "+str(j)
                        #      +" with prob "+str(dist_prob)+" and"
                        #      +" sim "+str(aligner_sim)) 
                        #sys.stdout.flush()

            for i,j in indices_to_merge:
                pattern1 = patterns[i]
                pattern2 = patterns[j]
                if (pattern1 != pattern2): #if not the same object
                    if (pattern1.num_seqlets < pattern2.num_seqlets):
                        parent_pattern, child_pattern = pattern2, pattern1
                    else:
                        parent_pattern, child_pattern = pattern1, pattern2
                    new_pattern = parent_pattern.copy()
                    new_pattern.merge_aggregated_seqlet(
                        agg_seqlet=child_pattern,
                        aligner=self.pattern_aligner) 
                    new_pattern =\
                        self.postprocessor([new_pattern])
                    assert len(new_pattern)==1
                    new_pattern = new_pattern[0]
                    for k in range(len(patterns)):
                        if (patterns[k]==parent_pattern or
                            patterns[k]==child_pattern):
                            patterns[k]=new_pattern
            merge_occurred_last_iteration = (len(indices_to_merge) > 0)

            if (merge_occurred_last_iteration):
                #Once we are out of this loop, each element of 'patterns'
                #will have the new parent of the corresponding element
                #of 'old_patterns'
                old_to_new_pattern_mapping = patterns

                #sort by size and remove redundant patterns 
                patterns = sorted(patterns, key=lambda x: -x.num_seqlets)
                patterns = list(OrderedDict([(x,1) for x in patterns]).keys())

                #update the hierarchy
                next_level_nodes = [PatternMergeHierarchyNode(x)
                                    for x in patterns]
                for next_level_node in next_level_nodes:
                    #iterate over all the old patterns and their new parent
                    for old_pattern_node, corresp_new_pattern\
                        in zip(current_level_nodes,
                               old_to_new_pattern_mapping):
                        #if the node has a new parent
                        if (old_pattern_node.pattern != corresp_new_pattern):
                            if (next_level_node.pattern==corresp_new_pattern):
                                next_level_node.child_nodes.append(
                                                old_pattern_node) 
                                assert old_pattern_node.parent_node is None
                                old_pattern_node.parent_node = next_level_node

                current_level_nodes=next_level_nodes

        return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)


class DynamicDistanceSimilarPatternsCollapser(object):

    def __init__(self, pattern_to_pattern_sim_computer,
                       aff_to_dist_mat, pattern_aligner,
                       collapse_condition, dealbreaker_condition,
                       postprocessor,
                       verbose=True):
        self.pattern_to_pattern_sim_computer = pattern_to_pattern_sim_computer 
        self.aff_to_dist_mat = aff_to_dist_mat
        self.pattern_aligner = pattern_aligner
        self.collapse_condition = collapse_condition
        self.dealbreaker_condition = dealbreaker_condition
        self.postprocessor = postprocessor
        self.verbose = verbose

    def __call__(self, patterns, seqlets):

        patterns = [x.copy() for x in patterns]
        merge_hierarchy_levels = []        
        current_level_nodes = [
            PatternMergeHierarchyNode(pattern=x) for x in patterns]
        merge_hierarchy_levels.append(current_level_nodes)

        merge_occurred_last_iteration = True
        merging_iteration = 0

        #loop until no more patterns get merged
        while (merge_occurred_last_iteration):
            
            merging_iteration += 1
            if (self.verbose):
                print("On merging iteration",merging_iteration) 
                sys.stdout.flush()
            merge_occurred_last_iteration = False

            if (self.verbose):
                print("Computing pattern to seqlet distances")
                sys.stdout.flush()
            patterns_to_seqlets_dist =\
                self.aff_to_dist_mat(self.pattern_to_pattern_sim_computer(
                                        seqlets=seqlets,
                                        filter_seqlets=patterns))
            desired_perplexities = [len(pattern.seqlets_and_alnmts)
                                    for pattern in patterns]
            pattern_betas = np.array([
                util.binary_search_perplexity(
                    desired_perplexity=desired_perplexity,
                    distances=distances)[0]
                for desired_perplexity,distances
                in zip(desired_perplexities, patterns_to_seqlets_dist.T)])

            if (self.verbose):
                print("Computing pattern to pattern distances")
                sys.stdout.flush()
            patterns_to_patterns_dist =\
                self.aff_to_dist_mat(self.pattern_to_pattern_sim_computer(
                                     seqlets=patterns,
                                     filter_seqlets=patterns))
            patterns_dist_probs = np.exp(-pattern_betas[:,None]*
                                         patterns_to_patterns_dist)
            patterns_to_patterns_aligner_sim =\
                np.zeros((len(patterns), len(patterns))) 
            for i,pattern1 in enumerate(patterns):
                for j,pattern2 in enumerate(patterns):
                    (alnmt, rc, aligner_sim) =\
                        self.pattern_aligner(pattern1, pattern2)
                    patterns_to_patterns_aligner_sim[i,j] = aligner_sim

            indices_to_merge = []
            merge_partners_so_far = dict([(i, set([i])) for i in
                                          range(len(patterns))])

            #merge patterns with highest similarity first
            sorted_pairs = sorted([(i,j,patterns_to_patterns_aligner_sim[i,j])
                            for i in range(len(patterns))
                            for j in range(len(patterns)) if (i < j)],
                            key=lambda x: -x[2])
            #iterate over pairs
            for (i,j,aligner_sim) in sorted_pairs:
                dist_prob = min(patterns_dist_probs[i,j],
                                patterns_dist_probs[j,i])
                if (self.collapse_condition(dist_prob=dist_prob,
                                            aligner_sim=aligner_sim)):
                    if (self.verbose):
                        print("Collapsing "+str(i)
                              +" & "+str(j)
                              +" with prob "+str(dist_prob)+" and"
                              +" sim "+str(aligner_sim)) 
                        sys.stdout.flush()

                    collapse_passed = True
                    #check compatibility for all indices that are
                    #about to be merged
                    merge_under_consideration = set(
                        list(merge_partners_so_far[i])
                        +list(merge_partners_so_far[j]))
                    for m1 in merge_under_consideration:
                        for m2 in merge_under_consideration:
                            if (m1 < m2):
                                min_dist_prob_here =\
                                    min(patterns_dist_probs[m1, m2],
                                        patterns_dist_probs[m2, m1])
                                aligner_sim_here =\
                                    patterns_to_patterns_aligner_sim[
                                        m1, m2]
                                if (self.dealbreaker_condition(
                                        dist_prob=min_dist_prob_here,
                                        aligner_sim=aligner_sim_here)):
                                    collapse_passed=False                     
                                    if (self.verbose):
                                        print("Aborting collapse as "
                                              +str(m1)
                                              +" & "+str(m2)
                                              +" have prob "
                                              +str(min_dist_prob_here)
                                              +" and"
                                              +" sim "
                                              +str(aligner_sim_here)) 
                                        sys.stdout.flush()
                                    break

                    if (collapse_passed):
                        indices_to_merge.append((i,j))
                        for an_idx in merge_under_consideration:
                            merge_partners_so_far[an_idx]=\
                                merge_under_consideration 
                else:
                    if (self.verbose):
                        pass
                        #print("Not collapsed "+str(i)+" & "+str(j)
                        #      +" with prob "+str(dist_prob)+" and"
                        #      +" sim "+str(aligner_sim)) 
                        #sys.stdout.flush()

            for i,j in indices_to_merge:
                pattern1 = patterns[i]
                pattern2 = patterns[j]
                if (pattern1 != pattern2): #if not the same object
                    if (pattern1.num_seqlets < pattern2.num_seqlets):
                        parent_pattern, child_pattern = pattern2, pattern1
                    else:
                        parent_pattern, child_pattern = pattern1, pattern2
                    new_pattern = parent_pattern.copy()
                    new_pattern.merge_aggregated_seqlet(
                        agg_seqlet=child_pattern,
                        aligner=self.pattern_aligner) 
                    new_pattern =\
                        self.postprocessor([new_pattern])
                    assert len(new_pattern)==1
                    new_pattern = new_pattern[0]
                    for k in range(len(patterns)):
                        if (patterns[k]==parent_pattern or
                            patterns[k]==child_pattern):
                            patterns[k]=new_pattern
            merge_occurred_last_iteration = (len(indices_to_merge) > 0)

            if (merge_occurred_last_iteration):
                #Once we are out of this loop, each element of 'patterns'
                #will have the new parent of the corresponding element
                #of 'old_patterns'
                old_to_new_pattern_mapping = patterns

                #sort by size and remove redundant patterns 
                patterns = sorted(patterns, key=lambda x: -x.num_seqlets)
                patterns = list(OrderedDict([(x,1) for x in patterns]).keys())

                #update the hierarchy
                next_level_nodes = [PatternMergeHierarchyNode(x)
                                    for x in patterns]
                for next_level_node in next_level_nodes:
                    #iterate over all the old patterns and their new parent
                    for old_pattern_node, corresp_new_pattern\
                        in zip(current_level_nodes,
                               old_to_new_pattern_mapping):
                        #if the node has a new parent
                        if (old_pattern_node.pattern != corresp_new_pattern):
                            if (next_level_node.pattern==corresp_new_pattern):
                                next_level_node.child_nodes.append(
                                                old_pattern_node) 
                                assert old_pattern_node.parent_node is None
                                old_pattern_node.parent_node = next_level_node

                current_level_nodes=next_level_nodes

        return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)
    

class BasicSimilarPatternsCollapser(object):

    def __init__(self, pattern_aligner,
                       merge_aligned_patterns_condition,
                       postprocessor,
                       verbose=True):
        self.merge_aligned_patterns_condition = merge_aligned_patterns_condition
        self.pattern_aligner = pattern_aligner
        self.verbose = verbose
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
                    merge = self.merge_aligned_patterns_condition(
                                parent_pattern=parent_pattern,
                                child_pattern=(child_pattern.revcomp()
                                  if is_revcomp else child_pattern),
                                alnmt=best_crosscorr_argmax)
                    if (merge): 
                        if (self.verbose):
                            print("Collapsed "+str(i)+" & "+str(j+i)) 
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
                            print("Not collapsing "+str(i)+" & "+str(j+i)) 
                            sys.stdout.flush()

        return sorted(self.postprocessor(list(set(original_patterns))),
                      key=lambda x: -x.num_seqlets)
