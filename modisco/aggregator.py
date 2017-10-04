from __future__ import division, print_function, absolute_import
import numpy as np
from . import affinitymat
from . import core
from . import util
from collections import OrderedDict


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

    def __init__(self, window_size, track_name):
        self.window_size = window_size
        self.track_name = track_name

    def __call__(self, aggregated_seqlets):
        trimmed_agg_seqlets = []
        for aggregated_seqlet in aggregated_seqlets:
            start_idx = np.argmax(util.cpu_sliding_window_sum(
                arr=np.sum(np.abs(aggregated_seqlet[self.track_name].fwd
                                  .reshape(len(aggregated_seqlet),-1)),axis=1),
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
            new_aggregated_seqlets.append(core.AggregatedSeqlet(
                seqlets_and_alnmts_arr=new_seqlets_and_alnmts))
        return new_aggregated_seqlets 


class AbstractTwoDMatSubclusterer(object):

    def __call__(self, twod_mat):
        #return subcluster_indices, num_subclusters
        raise NotImplementedError()


class RecursiveKmeans(AbstractTwoDMatSubclusterer):

    def __init__(self, threshold, min_before_split, verbose=True):
        self.threshold = threshold
        self.min_before_split = min_before_split
        self.verbose = verbose

    def __call__(self, twod_mat):
        import sklearn.cluster

        cluster_indices = sklearn.cluster.KMeans(n_clusters=2).\
                               fit_predict(twod_mat)

        cluster1_mean = np.mean(twod_mat[cluster_indices==0], axis=0)
        cluster2_mean = np.mean(twod_mat[cluster_indices==1], axis=0)
        cosine_dist = np.sum(cluster1_mean*cluster2_mean)/(
                       np.linalg.norm(cluster1_mean)
                       *np.linalg.norm(cluster2_mean))

        if (cosine_dist > self.threshold or
             (len(twod_mat) < self.min_before_split)):
            print("No split; similarity is "+str(cosine_dist)+" and "
                  "cluster size is "+str(len(twod_mat)))
            return np.zeros(len(twod_mat))
        else:
            print("Split detected; similarity is "+str(cosine_dist)+" and "
                  "cluster size is "+str(len(twod_mat)))
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
                for i in range(num_subclusters):
                    seqlets_and_alnmts_for_subcluster = [x[0] for x in
                        zip(aggregated_seqlet._seqlets_and_alnmts,
                            subcluster_indices) if x[1]==i]
                    to_return.append(core.AggregatedSeqlet(
                     seqlets_and_alnmts_arr=seqlets_and_alnmts_for_subcluster))
            else:
                to_return.append(aggregated_seqlet)
        return to_return


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
                aggregated_seqlets[i] = parent_agg_seqlet 
                aggregated_seqlets[j] = parent_agg_seqlet

        initial_aggregated = list(set(aggregated_seqlets))
        assert len(initial_aggregated)==1,str(len(initial_aggregated))+" "+str(initial_aggregated)
        
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

    def __call__(self, name_to_pattern):
        #make a copy of the dictionary
        name_to_new_pattern = OrderedDict([(x[0], x[1].copy()) for 
                                       x in name_to_pattern.items()])
        name_to_new_name = OrderedDict(
            zip(name_to_pattern.keys(),
            [set([x]) for x in name_to_pattern.keys()]))
        for i,name1 in enumerate(name_to_pattern.keys()):
            for j,name2 in enumerate(name_to_pattern.keys()):
                pattern1 = name_to_new_pattern[name1]
                pattern2 = name_to_new_pattern[name2]
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
                            print("Collapsing "+str(name1)+" & "+str(name2)) 
                        parent_pattern.merge_aggregated_seqlet(
                            agg_seqlet=child_pattern,
                            aligner=self.pattern_aligner) 
                        name_to_new_pattern[name1] = parent_pattern
                        name_to_new_pattern[name2] = parent_pattern
                        name_to_new_name[name1].update(
                                                 name_to_new_name[name2])
                        name_to_new_name[name2] = name_to_new_name[name1]

        #convert the sets into strings, find num unique new clusters
        name_to_new_name = OrderedDict([
            (name,
             "_".join([str(x) for x in sorted(list(new_name_contents))]))
            for (name, new_name_contents) in name_to_new_name.items()]) 
        new_names_list = sorted(list(set(name_to_new_name.values())))
        new_name_to_idx = OrderedDict([(x[1], x[0]) for x in
                                       enumerate(new_names_list)])
        name_to_new_cluster_idx = OrderedDict([
            (name, new_name_to_idx[name_to_new_name[name]]) for
             name in name_to_new_name.keys()]) 
        new_cluster_idx_to_new_pattern = OrderedDict([
            (name_to_new_cluster_idx[name],
             self.postprocessor([name_to_new_pattern[name]]))
            for name in name_to_new_name.keys()
        ])
                        
        return new_cluster_idx_to_new_pattern, name_to_new_cluster_idx
