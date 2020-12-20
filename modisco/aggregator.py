from __future__ import division, print_function, absolute_import
import numpy as np
from . import affinitymat
from . import core
from . import util
from collections import OrderedDict, defaultdict
import itertools
import sys
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import time


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


class AbstractTrimToBestWindow(AbstractAggSeqletPostprocessor):

    def __init__(self, window_size):
        self.window_size = window_size

    def score_positions(self, aggregated_seqlet):
        raise NotImplementError()

    def __call__(self, aggregated_seqlets):
        trimmed_agg_seqlets = []
        for aggregated_seqlet in aggregated_seqlets:
            start_idx = np.argmax(util.cpu_sliding_window_sum(
                arr=self.score_positions(aggregated_seqlet),
                window_size=self.window_size))
            end_idx = start_idx + self.window_size
            trimmed_agg_seqlets.append(
                aggregated_seqlet.trim_to_start_and_end_idx(
                    start_idx=start_idx, end_idx=end_idx)) 
        return trimmed_agg_seqlets


class TrimToBestWindowByIC(AbstractTrimToBestWindow):

    def __init__(self, window_size, onehot_track_name, bg_freq):
        super(TrimToBestWindowByIC, self).__init__(window_size=window_size)
        self.onehot_track_name = onehot_track_name
        self.bg_freq = bg_freq

    #sub up imp for each track, take l1 norm, average across seqlets
    def score_positions(self, aggregated_seqlet):
        ppm = aggregated_seqlet[self.onehot_track_name].fwd
        per_pos_ic = util.compute_per_position_ic(
            ppm=ppm, background=self.bg_freq, pseudocount=0.001)
        return per_pos_ic


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
                                         merge_into_existing_patterns=False)
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
        else:
            #Make a copy of each one
            new_patterns = [
                core.AggregatedSeqlet(pattern._seqlets_and_alnmts.copy())
                for pattern in patterns]
            
        for pattern,x in zip(new_patterns, seqlet_and_alnmnt_grps):
            pattern.merge_seqlets_and_alnmts(
                seqlets_and_alnmts=x,
                aligner=self.pattern_aligner) 

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

    def save_hdf5(self, grp):
        root_node_names = []
        for i in range(len(self.root_nodes)):
            node_name = "root_node"+str(i)
            root_node_names.append(node_name) 
            self.root_nodes[i].save_hdf5(grp.create_group(node_name))
        util.save_string_list(root_node_names,
                              dset_name="root_node_names",
                              grp=grp) 

    @classmethod
    def from_hdf5(cls, grp, track_set):
        root_node_names = util.load_string_list(dset_name="root_node_names",
                                                grp=grp) 
        root_nodes = []
        for root_node_name in root_node_names:
            root_node = PatternMergeHierarchyNode.from_hdf5(
                            grp=grp[root_node_name],
                            track_set=track_set)
            root_nodes.append(root_node)
        return cls(root_nodes=root_nodes) 


class PatternMergeHierarchyNode(object):

    def __init__(self, pattern, child_nodes=None, parent_node=None,
                       indices_merged=None, submat_crosscontam=None,
                       submat_alignersim=None): 
        self.pattern = pattern 
        if (child_nodes is None):
            child_nodes = []
        self.child_nodes = child_nodes
        self.parent_node = parent_node
        self.indices_merged = indices_merged
        self.submat_crosscontam = submat_crosscontam
        self.submat_alignersim = submat_alignersim

    def save_hdf5(self, grp):
        if (self.indices_merged is not None):
            grp.create_dataset("indices_merged",
                               data=np.array(self.indices_merged)) 
            grp.create_dataset("submat_crosscontam",
                               data=np.array(self.submat_crosscontam)) 
            grp.create_dataset("submat_alignersim",
                               data=np.array(self.submat_alignersim)) 
        self.pattern.save_hdf5(grp=grp.create_group("pattern"))
        if (self.child_nodes is not None):
            child_node_names = []
            for i in range(len(self.child_nodes)):
                child_node_name = "child_node"+str(i)
                child_node_names.append(child_node_name)
                self.child_nodes[i].save_hdf5(
                    grp.create_group(child_node_name))
            util.save_string_list(child_node_names,
                                  dset_name="child_node_names",
                                  grp=grp)

    @classmethod
    def from_hdf5(cls, grp, track_set):
        pattern = core.AggregatedSeqlet.from_hdf5(grp=grp["pattern"],
                                                  track_set=track_set)  
        if "indices_merged" in grp:
            indices_merged = tuple(grp["indices_merged"][:])
            submat_crosscontam = np.array(grp["submat_crosscontam"])
            submat_alignersim = np.array(grp["submat_alignersim"])
        else:
            (indices_merged, submat_crosscontam,
             submat_alignersim) = (None, None, None)
        if "child_node_names" in grp:
            child_node_names = util.load_string_list(
                                dset_name="child_node_names",
                                grp=grp)
            child_nodes = []
            for child_node_name in child_node_names:
                child_node = PatternMergeHierarchyNode.from_hdf5(
                               grp=grp[child_node_name],
                               track_set=track_set) 
                child_nodes.append(child_node)
               
        else:
            child_nodes = None
   
        to_return = cls(pattern=pattern,
                        child_nodes=child_nodes,
                        indices_merged=indices_merged,
                        submat_crosscontam=submat_crosscontam,
                        submat_alignersim=submat_alignersim) 

        if (child_nodes is not None):
            for child_node in child_nodes:
                child_node.parent_node = to_return

        return to_return


def compute_continjacc_vec_vs_arr(vec, arr):
    abs_vec = np.abs(vec)
    abs_arr = np.abs(arr)
    union = np.sum(np.maximum(abs_vec[None,:], abs_arr), axis=-1)
    intersection = np.sum((np.minimum(abs_vec[None,:], abs_arr)
                    *np.sign(vec[None,:])*np.sign(arr)), axis=-1)
    zeros_mask = (union==0)
    union = (union*(zeros_mask==False) + 1e-7*zeros_mask)
    return intersection/union


def compute_continjacc_arr1_vs_arr2(arr1, arr2, n_cores): 
    return np.array(
        Parallel(n_jobs=n_cores)(
            delayed(compute_continjacc_vec_vs_arr)(vec, arr2) for vec in arr1
        ))


def compute_continjacc_arr1_vs_arr2fwdandrev(arr1, arr2fwd, arr2rev, n_cores):
    sims = compute_continjacc_arr1_vs_arr2(arr1=arr1, arr2=arr2fwd,
                                               n_cores=n_cores)
    if (arr2rev is not None):
        rev_sims = compute_continjacc_arr1_vs_arr2(arr1=arr1, arr2=arr2rev,
                                                   n_cores=n_cores)
        sims = np.maximum(sims, rev_sims)
    return sims


class DynamicDistanceSimilarPatternsCollapser2(object):

    def __init__(self, pattern_comparison_settings,
                       track_set,
                       pattern_aligner,
                       collapse_condition, dealbreaker_condition,
                       postprocessor,
                       verbose=True,
                       max_seqlets_subsample=1000,
                       n_cores=1):
        self.pattern_comparison_settings = pattern_comparison_settings
        self.track_set = track_set
        self.pattern_aligner = pattern_aligner
        self.collapse_condition = collapse_condition
        self.dealbreaker_condition = dealbreaker_condition
        self.postprocessor = postprocessor
        self.verbose = verbose
        self.n_cores = n_cores
        self.max_seqlets_subsample = max_seqlets_subsample

    def subsample_pattern(self, pattern):
        seqlets_and_alnmts_list = list(pattern.seqlets_and_alnmts)
        subsample = [seqlets_and_alnmts_list[i]
                     for i in
                     np.random.RandomState(1234).choice(
                         a=np.arange(len(seqlets_and_alnmts_list)),
                         replace=False,
                         size=self.max_seqlets_subsample)]
        return core.AggregatedSeqlet(seqlets_and_alnmts_arr=subsample) 

    def __call__(self, patterns):

        patterns = [x.copy() for x in patterns]

        #Let's subsample 'patterns' to prevent runtime from being too
        # large in calculating pairwise sims. Max 1000, and also add in
        # parallelization.
        merge_hierarchy_levels = []        
        current_level_nodes = [
            PatternMergeHierarchyNode(pattern=x) for x in patterns]
        merge_hierarchy_levels.append(current_level_nodes)

        merge_occurred_last_iteration = True
        merging_iteration = 0

        #negative numbers to indicate which
        # entries need to be filled (versus entries we can infer
        # from the previous iteration of the while loop)
        pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
        pairwise_sims = np.zeros((len(patterns), len(patterns)))

        #loop until no more patterns get merged
        while (merge_occurred_last_iteration):

            start  = time.time()
            
            merging_iteration += 1
            if (self.verbose):
                print("On merging iteration",merging_iteration) 
                sys.stdout.flush()
            merge_occurred_last_iteration = False

            if (self.verbose):
                print("Numbers for each pattern pre-subsample:",
                      str([len(x.seqlets) for x in patterns]))
            subsample_patterns = [
                (x if x.num_seqlets <= self.max_seqlets_subsample
                 else self.subsample_pattern(x)) for x in patterns]
            if (self.verbose):
                print("Numbers after subsampling:",
                      str([len(x.seqlets) for x in subsample_patterns]))


            for i,(pattern1, subsample_pattern1) in enumerate(
                                            zip(patterns, subsample_patterns)):
                #from modisco.visualization import viz_sequence
                #viz_sequence.plot_weights(pattern1["task0_contrib_scores"].fwd)
                for j,(pattern2, subsample_pattern2) in enumerate(
                                            zip(patterns, subsample_patterns)):
                    #Note: I compute both i,j AND j,i because although
                    # the result is the same for the sim, it can be different
                    # for the auroc because a different motif is getting
                    # shifted over.
                    if (j==i):
                        pairwise_aurocs[i,j] = 0.5
                        pairwise_sims[i,j] = 1.0
                        continue
                    if pairwise_aurocs[i,j] >= 0: #filled in from previous iter
                        assert pairwise_aurocs[j,i] >= 0
                        continue 
                        
                    #Compute best alignment between pattern pair
                    (alnmt, rc, aligner_sim) =\
                        self.pattern_aligner(pattern1, pattern2)
                    pairwise_sims[i,j] = aligner_sim

                    #get realigned pattern2
                    pattern2_coords = [x.coor
                        for x in subsample_pattern2.seqlets]
                    if (rc):
                        pattern2_coords  = [x.revcomp()
                         for x in pattern2_coords]
                    #now apply the alignment
                    pattern2_coords = [
                        x.shift((1 if x.is_revcomp else -1)*alnmt)
                        for x in pattern2_coords] 

                    pattern2_shifted_seqlets = self.track_set.create_seqlets(
                        coords=pattern2_coords,
                        track_names=
                         self.pattern_comparison_settings.track_names) 

                    pattern1_fwdseqdata, pattern1_revseqdata =\
                      core.get_2d_data_from_patterns(
                        patterns=subsample_pattern1.seqlets,
                        track_names=
                         self.pattern_comparison_settings.track_names,
                        track_transformer=
                         self.pattern_comparison_settings.track_transformer)
                    pattern2_fwdseqdata, pattern2_revseqdata =\
                      core.get_2d_data_from_patterns(
                        patterns=pattern2_shifted_seqlets,
                        track_names=
                         self.pattern_comparison_settings.track_names,
                        track_transformer=
                         self.pattern_comparison_settings.track_transformer)

                    #Flatten, compute continjacc sim at this alignment
                    flat_pattern1_fwdseqdata = pattern1_fwdseqdata.reshape(
                        (len(pattern1_fwdseqdata), -1))
                    flat_pattern2_fwdseqdata = pattern2_fwdseqdata.reshape(
                        (len(pattern2_fwdseqdata), -1))
                    if (pattern1_revseqdata is not None):
                        flat_pattern1_revseqdata = pattern1_revseqdata.reshape(
                            (len(pattern1_revseqdata), -1))
                        flat_pattern2_revseqdata = pattern2_revseqdata.reshape(
                            (len(pattern2_fwdseqdata), -1))
                    else:
                        flat_pattern1_revseqdata = None 
                        flat_pattern2_revseqdata = None 
                        assert rc==False

                    #Do a check for all-zero scores, print warning
                    #do a check about the per-example sum
                    per_ex_sum_pattern1_zeromask = (np.sum(np.abs(
                        flat_pattern1_fwdseqdata),axis=-1))==0
                    per_ex_sum_pattern2_zeromask = (np.sum(np.abs(
                        flat_pattern2_fwdseqdata),axis=-1))==0
                    if (np.sum(per_ex_sum_pattern1_zeromask) > 0):
                        print("WARNING: Zeros present for pattern1 coords")
                        zero_seqlet_locs =\
                            np.nonzero(per_ex_sum_pattern1_zeromask) 
                        print("\n".join([str(s.coor) for s in
                                         subsample_pattern1.seqlets]))
                    if (np.sum(per_ex_sum_pattern2_zeromask) > 0):
                        print("WARNING: Zeros present for pattern2 coords")
                        zero_seqlet_locs =\
                            np.nonzero(per_ex_sum_pattern2_zeromask)
                        print("\n".join([str(coor) for coor in
                                         pattern2_coords]))

                    between_pattern_sims =\
                     compute_continjacc_arr1_vs_arr2fwdandrev(
                        arr1=flat_pattern1_fwdseqdata,
                        arr2fwd=flat_pattern2_fwdseqdata,
                        arr2rev=flat_pattern2_revseqdata,
                        n_cores=self.n_cores).ravel()

                    within_pattern1_sims =\
                     compute_continjacc_arr1_vs_arr2fwdandrev(
                        arr1=flat_pattern1_fwdseqdata,
                        arr2fwd=flat_pattern1_fwdseqdata,
                        arr2rev=flat_pattern1_revseqdata,
                        n_cores=self.n_cores).ravel()

                    auroc = roc_auc_score(
                        y_true=[0 for x in between_pattern_sims]
                               +[1 for x in within_pattern1_sims],
                        y_score=list(between_pattern_sims)
                                +list(within_pattern1_sims))

                    #The 'within pattern2 sims' may be less reliable due
                    # to the shiftover, that's why I won't compute them
                    #within_pattern2_sims =\
                    # compute_continjacc_arr1_vs_arr2fwdandrev(
                    #    arr1=flat_pattern2_fwdseqdata,
                    #    arr2fwd=flat_pattern2_fwdseqdata,
                    #    arr2rev=flat_pattern2_revseqdata,
                    #    n_cores=self.n_cores).ravel()
                    #auroc2 = roc_auc_score(
                    #    y_true=[0 for x in between_pattern_sims]
                    #           +[1 for x in within_pattern2_sims],
                    #    y_score=list(between_pattern_sims)
                    #            +list(within_pattern2_sims))

                    #The symmetrization over i,j and j,i is done later
                    pairwise_aurocs[i,j] = auroc


            patterns_to_patterns_aligner_sim = pairwise_sims
            cross_contamination = 2*(1-np.maximum(pairwise_aurocs,0.5))
            
            if (self.verbose):
                print("Cluster sizes")
                print(np.array([len(x.seqlets) for x in patterns]))
                print("Cross-contamination matrix:")
                print(np.round(cross_contamination,2))
                print("Pattern-to-pattern sim matrix:")
                print(np.round(patterns_to_patterns_aligner_sim,2))

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
                #symmetrize asymmetric crosscontam
                cross_contam = 0.5*(cross_contamination[i,j]+
                                    cross_contamination[j,i])
                if (self.collapse_condition(prob=cross_contam,
                                            aligner_sim=aligner_sim)):
                    if (self.verbose):
                        print("Collapsing "+str(i)
                              +" & "+str(j)
                              +" with crosscontam "+str(cross_contam)+" and"
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
                                cross_contam_here =\
                                    0.5*(cross_contamination[m1, m2]+
                                         cross_contamination[m2, m1])
                                aligner_sim_here =\
                                    patterns_to_patterns_aligner_sim[
                                        m1, m2]
                                if (self.dealbreaker_condition(
                                        prob=cross_contam_here,
                                        aligner_sim=aligner_sim_here)):
                                    collapse_passed=False                     
                                    if (self.verbose):
                                        print("Aborting collapse as "
                                              +str(m1)
                                              +" & "+str(m2)
                                              +" have cross-contam "
                                              +str(cross_contam_here)
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
                        #      +" with cross-contam "+str(cross_contam)+" and"
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
                        #Replace EVERY case where the parent or child
                        # pattern is present with the new pattern. This
                        # effectively does single-linkage.
                        if (patterns[k]==parent_pattern or
                            patterns[k]==child_pattern):
                            patterns[k]=new_pattern
            merge_occurred_last_iteration = (len(indices_to_merge) > 0)

            if (merge_occurred_last_iteration):
                #Once we are here, each element of 'patterns'
                #will have the new parent of the corresponding element
                #of 'old_patterns'
                old_to_new_pattern_mapping = patterns

                #sort by size and remove redundant patterns
                patterns = sorted(patterns, key=lambda x: -x.num_seqlets)
                patterns = list(OrderedDict([(x,1) for x in patterns]).keys())

                #let's figure out which indices don't require recomputation
                # and use it to repopulate pairwise_sims and pairwise_aurocs
                old_to_new_index_mappings = OrderedDict()
                for old_pattern_idx,(old_pattern_node, corresp_new_pattern)\
                    in enumerate(zip(current_level_nodes,
                                     old_to_new_pattern_mapping)):
                    #if the old pattern was NOT changed in this iteration
                    if (old_pattern_node.pattern == corresp_new_pattern):
                        new_idx = patterns.index(corresp_new_pattern) 
                        old_to_new_index_mappings[old_pattern_idx] = new_idx
                print("Unmerged patterns remapping:",old_to_new_index_mappings)
                new_pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
                new_pairwise_sims = np.zeros((len(patterns), len(patterns)))
                for old_idx_i, new_idx_i in\
                    old_to_new_index_mappings.items():
                    for old_idx_j, new_idx_j in\
                        old_to_new_index_mappings.items():
                        new_pairwise_aurocs[new_idx_i, new_idx_j] =\
                            pairwise_aurocs[old_idx_i, old_idx_j]
                        new_pairwise_sims[new_idx_i, new_idx_j] =\
                            pairwise_sims[old_idx_i, old_idx_j]
                pairwise_aurocs = new_pairwise_aurocs 
                pairwise_sims = new_pairwise_sims
                     

                #update the hierarchy
                next_level_nodes = [PatternMergeHierarchyNode(x)
                                    for x in patterns]
                for next_level_node in next_level_nodes:
                    #iterate over all the old patterns and their new parent
                    # in order to set up the child nodes correctly
                    for old_pattern_idx,(old_pattern_node, corresp_new_pattern)\
                        in enumerate(zip(current_level_nodes,
                                         old_to_new_pattern_mapping)):
 
                        #if the node has a new parent
                        if (old_pattern_node.pattern != corresp_new_pattern):
                            if (next_level_node.pattern==corresp_new_pattern):

                                
                                #corresp_new_pattern should be comprised of a 
                                # merging of all the old patterns at
                                # indices_merged_with
                                indices_merged = tuple(sorted(
                                    merge_partners_so_far[old_pattern_idx])) 
                                #get the relevant slice         
                                submat_crosscontam =\
                                 cross_contamination[indices_merged,:][:,
                                                     indices_merged]
                                submat_alignersim =\
                                 patterns_to_patterns_aligner_sim[
                                    indices_merged, :][:,indices_merged]

                                if (next_level_node.indices_merged is not None):
                                    assert (next_level_node.indices_merged
                                            ==indices_merged),\
                                     (next_level_node.indices_merged,
                                      indices_merged)
                                else:
                                    next_level_node.indices_merged =\
                                        indices_merged
                                    next_level_node.submat_crosscontam =\
                                        submat_crosscontam
                                    next_level_node.submat_alignersim =\
                                        submat_alignersim

                                next_level_node.child_nodes.append(
                                                old_pattern_node) 
                                assert old_pattern_node.parent_node is None
                                old_pattern_node.parent_node = next_level_node
                            

                current_level_nodes=next_level_nodes
                print("Time spent on merging iteration:", time.time()-start)

        return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)


#class DynamicDistanceSimilarPatternsCollapser(object):
#
#    def __init__(self, pattern_to_pattern_sim_computer,
#                       aff_to_dist_mat, pattern_aligner,
#                       collapse_condition, dealbreaker_condition,
#                       postprocessor,
#                       verbose=True):
#        self.pattern_to_pattern_sim_computer = pattern_to_pattern_sim_computer 
#        self.aff_to_dist_mat = aff_to_dist_mat
#        self.pattern_aligner = pattern_aligner
#        self.collapse_condition = collapse_condition
#        self.dealbreaker_condition = dealbreaker_condition
#        self.postprocessor = postprocessor
#        self.verbose = verbose
#
#    def __call__(self, patterns):
#
#
#        patterns = [x.copy() for x in patterns]
#        merge_hierarchy_levels = []        
#        current_level_nodes = [
#            PatternMergeHierarchyNode(pattern=x) for x in patterns]
#        merge_hierarchy_levels.append(current_level_nodes)
#
#        merge_occurred_last_iteration = True
#        merging_iteration = 0
#
#        #loop until no more patterns get merged
#        while (merge_occurred_last_iteration):
#            
#            merging_iteration += 1
#            if (self.verbose):
#                print("On merging iteration",merging_iteration) 
#                sys.stdout.flush()
#            merge_occurred_last_iteration = False
#
#            if (self.verbose):
#                print("Computing pattern to seqlet distances")
#                sys.stdout.flush()
#
#            seqlets = [x for pattern in patterns for x in pattern.seqlets]
#            orig_seqlet_membership = [pattern_idx for (pattern_idx,pattern)
#                                      in enumerate(patterns)
#                                      for x in pattern.seqlets]
#            patterns_to_seqlets_simmat =\
#                self.pattern_to_pattern_sim_computer(
#                                        seqlets=seqlets,
#                                        filter_seqlets=patterns).transpose(
#                                         (1,0))
#            cross_contamination = np.zeros((len(patterns_to_seqlets_simmat),
#                                            len(patterns_to_seqlets_simmat)))
#            for seqlet_idx in range(patterns_to_seqlets_simmat.shape[1]):
#                seqlet_orig_pattern = orig_seqlet_membership[seqlet_idx]
#                seqlet_self_sim = patterns_to_seqlets_simmat[
#                                   seqlet_orig_pattern, seqlet_idx]
#                for pattern_idx in range(patterns_to_seqlets_simmat.shape[0]):
#                    if (patterns_to_seqlets_simmat[pattern_idx, seqlet_idx] >=
#                        seqlet_self_sim):
#                        cross_contamination[seqlet_orig_pattern,
#                                            pattern_idx] += 1 
#            #normalize the cross_contamination rows by the num in the diagonal 
#            for pattern_idx in range(len(cross_contamination)):
#                assert cross_contamination[pattern_idx,pattern_idx] > 0
#                cross_contamination[pattern_idx] =\
#                 (cross_contamination[pattern_idx]/
#                  float(cross_contamination[pattern_idx,pattern_idx]))
#
#            if (self.verbose):
#                print("Computing pattern to pattern sims")
#                sys.stdout.flush()
#            patterns_to_patterns_aligner_sim =\
#                np.zeros((len(patterns), len(patterns))) 
#            for i,pattern1 in enumerate(patterns):
#                for j,pattern2 in enumerate(patterns):
#                    (alnmt, rc, aligner_sim) =\
#                        self.pattern_aligner(pattern1, pattern2)
#                    patterns_to_patterns_aligner_sim[i,j] = aligner_sim
#            if (self.verbose):
#                print("Cluster sizes")
#                print(np.array([len(x.seqlets) for x in patterns]))
#                print("Cross-contamination matrix:")
#                print(np.round(cross_contamination,2))
#                print("Pattern-to-pattern sim matrix:")
#                print(np.round(patterns_to_patterns_aligner_sim,2))
#
#            indices_to_merge = []
#            merge_partners_so_far = dict([(i, set([i])) for i in
#                                          range(len(patterns))])
#
#            #merge patterns with highest similarity first
#            sorted_pairs = sorted([(i,j,patterns_to_patterns_aligner_sim[i,j])
#                            for i in range(len(patterns))
#                            for j in range(len(patterns)) if (i < j)],
#                            key=lambda x: -x[2])
#            #iterate over pairs
#            for (i,j,aligner_sim) in sorted_pairs:
#                cross_contam = max(cross_contamination[i,j],
#                                   cross_contamination[j,i])
#                if (self.collapse_condition(prob=cross_contam,
#                                            aligner_sim=aligner_sim)):
#                    if (self.verbose):
#                        print("Collapsing "+str(i)
#                              +" & "+str(j)
#                              +" with crosscontam "+str(cross_contam)+" and"
#                              +" sim "+str(aligner_sim)) 
#                        sys.stdout.flush()
#
#                    collapse_passed = True
#                    #check compatibility for all indices that are
#                    #about to be merged
#                    merge_under_consideration = set(
#                        list(merge_partners_so_far[i])
#                        +list(merge_partners_so_far[j]))
#                    for m1 in merge_under_consideration:
#                        for m2 in merge_under_consideration:
#                            if (m1 < m2):
#                                cross_contam_here =\
#                                    max(cross_contamination[m1, m2],
#                                        cross_contamination[m2, m1])
#                                aligner_sim_here =\
#                                    patterns_to_patterns_aligner_sim[
#                                        m1, m2]
#                                if (self.dealbreaker_condition(
#                                        prob=cross_contam_here,
#                                        aligner_sim=aligner_sim_here)):
#                                    collapse_passed=False                     
#                                    if (self.verbose):
#                                        print("Aborting collapse as "
#                                              +str(m1)
#                                              +" & "+str(m2)
#                                              +" have cross-contam "
#                                              +str(cross_contam_here)
#                                              +" and"
#                                              +" sim "
#                                              +str(aligner_sim_here)) 
#                                        sys.stdout.flush()
#                                    break
#
#                    if (collapse_passed):
#                        indices_to_merge.append((i,j))
#                        for an_idx in merge_under_consideration:
#                            merge_partners_so_far[an_idx]=\
#                                merge_under_consideration 
#                else:
#                    if (self.verbose):
#                        pass
#                        #print("Not collapsed "+str(i)+" & "+str(j)
#                        #      +" with cross-contam "+str(cross_contam)+" and"
#                        #      +" sim "+str(aligner_sim)) 
#                        #sys.stdout.flush()
#
#            for i,j in indices_to_merge:
#                pattern1 = patterns[i]
#                pattern2 = patterns[j]
#                if (pattern1 != pattern2): #if not the same object
#                    if (pattern1.num_seqlets < pattern2.num_seqlets):
#                        parent_pattern, child_pattern = pattern2, pattern1
#                    else:
#                        parent_pattern, child_pattern = pattern1, pattern2
#                    new_pattern = parent_pattern.copy()
#                    new_pattern.merge_aggregated_seqlet(
#                        agg_seqlet=child_pattern,
#                        aligner=self.pattern_aligner) 
#                    new_pattern =\
#                        self.postprocessor([new_pattern])
#                    assert len(new_pattern)==1
#                    new_pattern = new_pattern[0]
#                    for k in range(len(patterns)):
#                        if (patterns[k]==parent_pattern or
#                            patterns[k]==child_pattern):
#                            patterns[k]=new_pattern
#            merge_occurred_last_iteration = (len(indices_to_merge) > 0)
#
#            if (merge_occurred_last_iteration):
#                #Once we are out of this loop, each element of 'patterns'
#                #will have the new parent of the corresponding element
#                #of 'old_patterns'
#                old_to_new_pattern_mapping = patterns
#
#                #sort by size and remove redundant patterns 
#                patterns = sorted(patterns, key=lambda x: -x.num_seqlets)
#                patterns = list(OrderedDict([(x,1) for x in patterns]).keys())
#
#                #update the hierarchy
#                next_level_nodes = [PatternMergeHierarchyNode(x)
#                                    for x in patterns]
#                for next_level_node in next_level_nodes:
#                    #iterate over all the old patterns and their new parent
#                    for old_pattern_node, corresp_new_pattern\
#                        in zip(current_level_nodes,
#                               old_to_new_pattern_mapping):
#                        #if the node has a new parent
#                        if (old_pattern_node.pattern != corresp_new_pattern):
#                            if (next_level_node.pattern==corresp_new_pattern):
#                                next_level_node.child_nodes.append(
#                                                old_pattern_node) 
#                                assert old_pattern_node.parent_node is None
#                                old_pattern_node.parent_node = next_level_node
#
#                current_level_nodes=next_level_nodes
#
#        return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)


#class OldDynamicDistanceSimilarPatternsCollapser(object):
#
#    def __init__(self, pattern_to_pattern_sim_computer,
#                       aff_to_dist_mat, pattern_aligner,
#                       collapse_condition, dealbreaker_condition,
#                       postprocessor,
#                       verbose=True):
#        self.pattern_to_pattern_sim_computer = pattern_to_pattern_sim_computer 
#        self.aff_to_dist_mat = aff_to_dist_mat
#        self.pattern_aligner = pattern_aligner
#        self.collapse_condition = collapse_condition
#        self.dealbreaker_condition = dealbreaker_condition
#        self.postprocessor = postprocessor
#        self.verbose = verbose
#
#    def __call__(self, patterns, seqlets):
#
#        patterns = [x.copy() for x in patterns]
#        merge_hierarchy_levels = []        
#        current_level_nodes = [
#            PatternMergeHierarchyNode(pattern=x) for x in patterns]
#        merge_hierarchy_levels.append(current_level_nodes)
#
#        merge_occurred_last_iteration = True
#        merging_iteration = 0
#
#        #loop until no more patterns get merged
#        while (merge_occurred_last_iteration):
#            
#            merging_iteration += 1
#            if (self.verbose):
#                print("On merging iteration",merging_iteration) 
#                sys.stdout.flush()
#            merge_occurred_last_iteration = False
#
#            if (self.verbose):
#                print("Computing pattern to seqlet distances")
#                sys.stdout.flush()
#            patterns_to_seqlets_dist =\
#                self.aff_to_dist_mat(self.pattern_to_pattern_sim_computer(
#                                        seqlets=seqlets,
#                                        filter_seqlets=patterns))
#            desired_perplexities = [len(pattern.seqlets_and_alnmts)
#                                    for pattern in patterns]
#            pattern_betas = np.array([
#                util.binary_search_perplexity(
#                    desired_perplexity=desired_perplexity,
#                    distances=distances)[0]
#                for desired_perplexity,distances
#                in zip(desired_perplexities, patterns_to_seqlets_dist.T)])
#
#            if (self.verbose):
#                print("Computing pattern to pattern distances")
#                sys.stdout.flush()
#            patterns_to_patterns_dist =\
#                self.aff_to_dist_mat(self.pattern_to_pattern_sim_computer(
#                                     seqlets=patterns,
#                                     filter_seqlets=patterns))
#            patterns_dist_probs = np.exp(-pattern_betas[:,None]*
#                                         patterns_to_patterns_dist)
#            patterns_to_patterns_aligner_sim =\
#                np.zeros((len(patterns), len(patterns))) 
#            for i,pattern1 in enumerate(patterns):
#                for j,pattern2 in enumerate(patterns):
#                    (alnmt, rc, aligner_sim) =\
#                        self.pattern_aligner(pattern1, pattern2)
#                    patterns_to_patterns_aligner_sim[i,j] = aligner_sim
#
#            indices_to_merge = []
#            merge_partners_so_far = dict([(i, set([i])) for i in
#                                          range(len(patterns))])
#
#            #merge patterns with highest similarity first
#            sorted_pairs = sorted([(i,j,patterns_to_patterns_aligner_sim[i,j])
#                            for i in range(len(patterns))
#                            for j in range(len(patterns)) if (i < j)],
#                            key=lambda x: -x[2])
#            #iterate over pairs
#            for (i,j,aligner_sim) in sorted_pairs:
#                dist_prob = min(patterns_dist_probs[i,j],
#                                patterns_dist_probs[j,i])
#                if (self.collapse_condition(dist_prob=dist_prob,
#                                            aligner_sim=aligner_sim)):
#                    if (self.verbose):
#                        print("Collapsing "+str(i)
#                              +" & "+str(j)
#                              +" with prob "+str(dist_prob)+" and"
#                              +" sim "+str(aligner_sim)) 
#                        sys.stdout.flush()
#
#                    collapse_passed = True
#                    #check compatibility for all indices that are
#                    #about to be merged
#                    merge_under_consideration = set(
#                        list(merge_partners_so_far[i])
#                        +list(merge_partners_so_far[j]))
#                    for m1 in merge_under_consideration:
#                        for m2 in merge_under_consideration:
#                            if (m1 < m2):
#                                min_dist_prob_here =\
#                                    min(patterns_dist_probs[m1, m2],
#                                        patterns_dist_probs[m2, m1])
#                                aligner_sim_here =\
#                                    patterns_to_patterns_aligner_sim[
#                                        m1, m2]
#                                if (self.dealbreaker_condition(
#                                        dist_prob=min_dist_prob_here,
#                                        aligner_sim=aligner_sim_here)):
#                                    collapse_passed=False                     
#                                    if (self.verbose):
#                                        print("Aborting collapse as "
#                                              +str(m1)
#                                              +" & "+str(m2)
#                                              +" have prob "
#                                              +str(min_dist_prob_here)
#                                              +" and"
#                                              +" sim "
#                                              +str(aligner_sim_here)) 
#                                        sys.stdout.flush()
#                                    break
#
#                    if (collapse_passed):
#                        indices_to_merge.append((i,j))
#                        for an_idx in merge_under_consideration:
#                            merge_partners_so_far[an_idx]=\
#                                merge_under_consideration 
#                else:
#                    if (self.verbose):
#                        pass
#                        #print("Not collapsed "+str(i)+" & "+str(j)
#                        #      +" with prob "+str(dist_prob)+" and"
#                        #      +" sim "+str(aligner_sim)) 
#                        #sys.stdout.flush()
#
#            for i,j in indices_to_merge:
#                pattern1 = patterns[i]
#                pattern2 = patterns[j]
#                if (pattern1 != pattern2): #if not the same object
#                    if (pattern1.num_seqlets < pattern2.num_seqlets):
#                        parent_pattern, child_pattern = pattern2, pattern1
#                    else:
#                        parent_pattern, child_pattern = pattern1, pattern2
#                    new_pattern = parent_pattern.copy()
#                    new_pattern.merge_aggregated_seqlet(
#                        agg_seqlet=child_pattern,
#                        aligner=self.pattern_aligner) 
#                    new_pattern =\
#                        self.postprocessor([new_pattern])
#                    assert len(new_pattern)==1
#                    new_pattern = new_pattern[0]
#                    for k in range(len(patterns)):
#                        if (patterns[k]==parent_pattern or
#                            patterns[k]==child_pattern):
#                            patterns[k]=new_pattern
#            merge_occurred_last_iteration = (len(indices_to_merge) > 0)
#
#            if (merge_occurred_last_iteration):
#                #Once we are out of this loop, each element of 'patterns'
#                #will have the new parent of the corresponding element
#                #of 'old_patterns'
#                old_to_new_pattern_mapping = patterns
#
#                #sort by size and remove redundant patterns 
#                patterns = sorted(patterns, key=lambda x: -x.num_seqlets)
#                patterns = list(OrderedDict([(x,1) for x in patterns]).keys())
#
#                #update the hierarchy
#                next_level_nodes = [PatternMergeHierarchyNode(x)
#                                    for x in patterns]
#                for next_level_node in next_level_nodes:
#                    #iterate over all the old patterns and their new parent
#                    for old_pattern_node, corresp_new_pattern\
#                        in zip(current_level_nodes,
#                               old_to_new_pattern_mapping):
#                        #if the node has a new parent
#                        if (old_pattern_node.pattern != corresp_new_pattern):
#                            if (next_level_node.pattern==corresp_new_pattern):
#                                next_level_node.child_nodes.append(
#                                                old_pattern_node) 
#                                assert old_pattern_node.parent_node is None
#                                old_pattern_node.parent_node = next_level_node
#
#                current_level_nodes=next_level_nodes
#
#        return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)
    

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
