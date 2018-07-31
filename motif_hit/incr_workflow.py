from __future__ import division, print_function, absolute_import

import sys
from collections import OrderedDict

import h5py
import numpy as np

from modisco import coordproducers
from modisco import core
from modisco import metaclusterers

from modisco.tfmodisco_workflow import seqlets_to_patterns
from modisco.tfmodisco_workflow import workflow

from . import motif_hits
from . import seqlets_to_metacluster


class SubMetaclusterPriorResults(object):

    def __init__(self):
        self.metacluster_size = 0
        self.activity_pattern = []
        self.seqlets_per_pattern = [] # list of seqlets, one list per pattern
        self.seqlets_to_patterns_result = []
        self.all_pattern_names = []
        self.pattern_seqlets = []

class TfModiscoPriorResults(object):
    """
    Class to load hdf5 results from a prior tf-modisco run, specifically
    (1) identified motifs and
    (2) their associated seqlets in the same cluster
    """
    def __init__(self, file_name, track_set):
        self.file_name = file_name

        self.build(file_name, track_set)


    def build(self, file_name, track_set):
        
        self.hdf5_results = h5py.File(file_name,"r")
        
        self.metacluster_names = [
            x.decode("utf-8") for x in 
            list(self.hdf5_results["metaclustering_results"]
                 ["all_metacluster_names"][:])]

        metacluster_grp = None #temp
        self.all_activity_patterns = []
        self.metaclusters = OrderedDict()
        # recover pattern seqlets in each metacluster
        # based on visualization ipynb
        for metacluster_name in self.metacluster_names:
            print(metacluster_name)
            metacluster = SubMetaclusterPriorResults()
            self.metaclusters[metacluster_name] = metacluster

            metacluster_grp = (self.hdf5_results["metacluster_idx_to_submetacluster_results"][metacluster_name])
            metacluster.all_pattern_names = [x.decode("utf-8") for x in
                                      list(metacluster_grp["seqlets_to_patterns_result"]
                                           ["patterns"]["all_pattern_names"][:])]
            if (len(metacluster.all_pattern_names) == 0):
                print("No motifs found for this activity pattern")
            metacluster.metacluster_size = len(metacluster.all_pattern_names)

            activity_pattern = metacluster_grp["activity_pattern"][:]
            metacluster.activity_pattern = activity_pattern
            self.all_activity_patterns.append(activity_pattern)
            print("activity pattern:", activity_pattern)

            for pattern_name in metacluster.all_pattern_names:
                print(metacluster_name, pattern_name)
                pattern_grp = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
                pattern_seqlets = self.create_pattern_seqlets(pattern_grp, pattern_name)
                metacluster.pattern_seqlets += pattern_seqlets
                seqlets = self.create_seqlets(pattern_grp, track_set)
                metacluster.seqlets_per_pattern.append(seqlets)

    # read in seqlets_per_pattern
    def create_seqlets(self, pattern_grp, track_set):
        coords_strs = pattern_grp["seqlets_and_alnmts"]["seqlets"].value
        coords = []
        for coord_str in coords_strs:
            item_strs = coord_str.split(",")
            ex_id = int(item_strs[0].split(":")[1])
            start = int(item_strs[1].split(":")[1])
            end   = int(item_strs[2].split(":")[1])
            rc    = item_strs[3].split(":")[1] == "True"
            coord = core.SeqletCoordinates(ex_id, start+10, end-10, rc)
            coords.append(coord)
        seqlets = track_set.create_seqlets(coords)
        print("created seqlets_per_pattern for ")

        return seqlets

    # make a seqlet out of a given pattern
    def create_pattern_seqlets(self, pattern, pattern_name):
        from modisco import core
        task_names = ["task0", "task1", "task2"]
        print([track for track in pattern])
        contrib_scores_tracks = [
            core.DataTrack(name=key + "_contrib_scores",
                           fwd_tracks=[pattern[key + "_contrib_scores"]["fwd"]],
                           rev_tracks=[pattern[key + "_contrib_scores"]["rev"]],
                           has_pos_axis=True) for key in task_names]
        hypothetical_contribs_tracks = [
            core.DataTrack(name=key + "_hypothetical_contribs",
                           fwd_tracks=[pattern[key + "_hypothetical_contribs"]["fwd"]],
                           rev_tracks=[pattern[key + "_hypothetical_contribs"]["rev"]],
                           has_pos_axis=True) for key in task_names]
        onehot_track = core.DataTrack(name="sequence",
                                      fwd_tracks=[pattern["sequence"]["fwd"]],
                                      rev_tracks=[pattern["sequence"]["rev"]],
                                      has_pos_axis=True)
        track_set = core.TrackSet(data_tracks=contrib_scores_tracks
                                              + hypothetical_contribs_tracks + [onehot_track])

        seq_size = onehot_track.fwd_tracks[0].shape[0]
        coord = core.SeqletCoordinates(0, 10, seq_size-10, False)
        seqlets = track_set.create_seqlets([coord])
        print(seqlets)
        return seqlets

class TfModiscoIncrementalWorkflow(object):

    def __init__(self,
                 seqlets_to_patterns_factory=
                 seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(),
                 sliding_window_size=21, flank_size=10,
                 histogram_bins=100, percentiles_in_bandwidth=10, 
                 overlap_portion=0.5,
                 min_cluster_size=100,
                 target_seqlet_fdr = 0.05,
                 weak_threshold_for_counting_sign = 0.99,
                 max_seqlets_per_metacluster=20000,
                 max_seqlets_per_task=None,
                 verbose=True,
                 prior_results=None):

        self.seqlets_to_patterns_factory = seqlets_to_patterns_factory
        self.sliding_window_size = sliding_window_size
        self.flank_size = flank_size
        self.histogram_bins = histogram_bins
        self.percentiles_in_bandwidth = percentiles_in_bandwidth
        self.overlap_portion = overlap_portion
        self.min_cluster_size = min_cluster_size
        self.target_seqlet_fdr = target_seqlet_fdr
        self.weak_threshold_for_counting_sign = weak_threshold_for_counting_sign
        self.max_seqlets_per_metacluster = max_seqlets_per_metacluster
        self.max_seqlets_per_task = max_seqlets_per_task
        self.verbose = verbose
        self.prior_results = prior_results

        self.build()

    def build(self):
        
        self.overlap_resolver = core.SeqletsOverlapResolver(
            overlap_detector=core.CoordOverlapDetector(self.overlap_portion),
            seqlet_comparator=core.SeqletComparator(
                                value_provider=lambda x: x.coor.score))

        self.threshold_score_transformer_factory =\
            core.LaplaceCdfFactory(flank_to_ignore=self.flank_size)


    def get_metacluster_seqlets(self, task_names, contrib_scores,
                                hypothetical_contribs, one_hot):
        '''
        create metaclusters based on activity patterns, and assign seqlets.
        TODO: is common between workflow and incremental workflow, could merge later.
        Args:
            task_names:
            contrib_scores:
            hypothetical_contribs:
            one_hot:
        Returns:
            TfModiscoResults, excluding seqlets_to_patterns_result
        '''

        self.coord_producer = coordproducers.FixedWindowAroundChunks(
            sliding=self.sliding_window_size,
            flank=self.flank_size,
            thresholding_function=coordproducers.LaplaceThreshold(
                                    target_fdr=self.target_seqlet_fdr,
                                    verbose=self.verbose),
            max_seqlets_total=self.max_seqlets_per_task,
            verbose=self.verbose) 

        contrib_scores_tracks = [
            core.DataTrack(
                name=key+"_contrib_scores",
                fwd_tracks=contrib_scores[key],
                rev_tracks=[x[::-1, ::-1] for x in 
                            contrib_scores[key]],
                has_pos_axis=True) for key in task_names] 

        hypothetical_contribs_tracks = [
            core.DataTrack(name=key+"_hypothetical_contribs",
                           fwd_tracks=hypothetical_contribs[key],
                           rev_tracks=[x[::-1, ::-1] for x in 
                                        hypothetical_contribs[key]],
                           has_pos_axis=True)
                           for key in task_names]

        onehot_track = core.DataTrack(
                            name="sequence", fwd_tracks=one_hot,
                            rev_tracks=[x[::-1, ::-1] for x in one_hot],
                            has_pos_axis=True)

        track_set = core.TrackSet(
                        data_tracks=contrib_scores_tracks
                        +hypothetical_contribs_tracks+[onehot_track])

        per_position_contrib_scores = OrderedDict([
            (x, [np.sum(s,axis=1) for s in contrib_scores[x]]) for x in task_names])


        task_name_to_threshold_transformer = OrderedDict([
            (task_name, self.threshold_score_transformer_factory(
                name=task_name+"_label",
                track_name=task_name+"_contrib_scores"))
             for task_name in task_names]) 

        multitask_seqlet_creation_results = core.MultiTaskSeqletCreation(
            coord_producer=self.coord_producer,
            track_set=track_set,
            overlap_resolver=self.overlap_resolver)(
                task_name_to_score_track=per_position_contrib_scores,
                task_name_to_threshold_transformer=\
                    task_name_to_threshold_transformer)

        #find the weakest laplace cdf threshold used across all tasks
        laplace_threshold_cdf = min(
            [min(x.thresholding_results.pos_threshold_cdf,
                 x.thresholding_results.neg_threshold_cdf)
                 for x in multitask_seqlet_creation_results.
                      task_name_to_coord_producer_results.values()])
        print("Across all tasks, the weakest laplace threshold used"
              +" was: "+str(laplace_threshold_cdf))

        seqlets = multitask_seqlet_creation_results.final_seqlets

        print(str(len(seqlets))+" identified in total")
        if (len(seqlets) < 100):
            print("WARNING: you found relatively few seqlets."
                  +" Consider dropping target_seqlet_fdr") 

        attribute_vectors = (np.array([
                              [x[key+"_label"] for key in task_names]
                               for x in seqlets]))

        if (self.weak_threshold_for_counting_sign is None):
            weak_threshold_for_counting_sign = laplace_threshold_cdf
        else:
            weak_threshold_for_counting_sign =\
                self.weak_threshold_for_counting_sign
        if (weak_threshold_for_counting_sign > laplace_threshold_cdf):
            print("Reducing weak_threshold_for_counting_sign to"
                  +" match laplace_threshold_cdf, from "
                  +str(weak_threshold_for_counting_sign)
                  +" to "+str(laplace_threshold_cdf))
            weak_threshold_for_counting_sign = laplace_threshold_cdf

        possible_activity_patterns = None
        if self.prior_results != None:
            possible_activity_patterns = self.prior_results.all_activity_patterns
        metaclusterer = metaclusterers.SignBasedPatternClustering(
                                min_cluster_size=self.min_cluster_size,
                                threshold_for_counting_sign=
                                    laplace_threshold_cdf,
                                weak_threshold_for_counting_sign=
                                    weak_threshold_for_counting_sign)

        # new: map seqlet to metacluster
        metaclustering_results = seqlets_to_metacluster.map_seqlet_to_metacluster(metaclusterer, attribute_vectors,
                                    possible_activity_patterns = possible_activity_patterns)

        metacluster_indices = metaclustering_results.metacluster_indices
        metacluster_idx_to_activity_pattern =\
            metaclustering_results.metacluster_idx_to_activity_pattern

        num_metaclusters = max(metacluster_indices)+1
        metacluster_sizes = [np.sum(metacluster_idx==metacluster_indices)
                              for metacluster_idx in range(num_metaclusters)]
        if (self.verbose):
            print("Metacluster sizes: ",metacluster_sizes)
            print("Idx to activities: ",metacluster_idx_to_activity_pattern)
            sys.stdout.flush()

        metacluster_idx_to_submetacluster_results = OrderedDict()

        for metacluster_idx, metacluster_size in\
            sorted(enumerate(metacluster_sizes), key=lambda x: x[1]):
            print("On metacluster "+str(metacluster_idx))
            if (self.max_seqlets_per_metacluster is None
                or self.max_seqlets_per_metacluster >= metacluster_size):
                print("Metacluster size", metacluster_size)
            else:
                print("Metacluster size {0} limited to {1}".format(
                        metacluster_size, self.max_seqlets_per_metacluster))
            sys.stdout.flush()
            metacluster_activities = [
                int(x) for x in
                metacluster_idx_to_activity_pattern[metacluster_idx].split(",")]
            assert len(seqlets)==len(metacluster_indices)

            metacluster_seqlets = [
                x[0] for x in zip(seqlets, metacluster_indices)
                if x[1]==metacluster_idx][:self.max_seqlets_per_metacluster]
            relevant_task_names, relevant_task_signs =\
                zip(*[(x[0], x[1]) for x in
                    zip(task_names, metacluster_activities) if x[1] != 0])
            print('Relevant tasks: ', relevant_task_names)
            print('Relevant signs: ', relevant_task_signs)
            sys.stdout.flush()
            if (len(relevant_task_names) == 0):
                assert False, "This should not happen"
                sys.stdout.flush()


            metacluster_idx_to_submetacluster_results[metacluster_idx] =\
                workflow.SubMetaclusterResults(
                    metacluster_size=metacluster_size,
                    activity_pattern=np.array(metacluster_activities),
                    seqlets=metacluster_seqlets,
                    seqlets_to_patterns_result=[])

        return workflow.TfModiscoResults(
                 task_names=task_names,
                 multitask_seqlet_creation_results=
                    multitask_seqlet_creation_results,
                 seqlet_attribute_vectors=attribute_vectors,
                 metaclustering_results=metaclustering_results,
                 metacluster_idx_to_submetacluster_results=
                    metacluster_idx_to_submetacluster_results), track_set

    def __call__(self, task_names, contrib_scores,
                 hypothetical_contribs, one_hot):
        '''
        Takes input hypothetical scores, contrib scores, and sequence, create track_set, seqlets, attribute_vectors, and metaclusters.
        TODO: is common between workflow and incremental workflow, could merge later.
        Args:
            task_names: list of task names
            contrib_scores: contribution scores
            hypothetical_contribs: hyp scores
            one_hot: onehot encoding of the sequence
        Returns:
            TfModiscoResults, excluding seqlets_to_patterns_result
        '''

        results, track_set = self.get_metacluster_seqlets(task_names, contrib_scores,
                                                          hypothetical_contribs, one_hot)
        num_metaclusters = len(results.metacluster_idx_to_submetacluster_results)
        for metacluster_idx in range(num_metaclusters):
            sub_metacluster_results = results.metacluster_idx_to_submetacluster_results[metacluster_idx]
            metacluster_size = sub_metacluster_results.metacluster_size

            metacluster_activities = sub_metacluster_results.activity_pattern

            metacluster_seqlets = sub_metacluster_results.seqlets

            relevant_task_names, relevant_task_signs = \
                zip(*[(x[0], x[1]) for x in
                      zip(task_names, metacluster_activities) if x[1] != 0])

            seqlets_to_patterns = self.seqlets_to_patterns_factory(
                                        track_set=track_set,
                                        onehot_track_name="sequence",
                                        contrib_scores_track_names= \
                                            [key + "_contrib_scores" for key in relevant_task_names],
                                        hypothetical_contribs_track_names= \
                                            [key + "_hypothetical_contribs" for key in relevant_task_names],
                                        track_signs=relevant_task_signs,
                                        other_comparison_track_names=[])

            # new: map seqlets to patterns
            seqlets_to_patterns_result = motif_hits.map_seqlets_to_patterns(seqlets_to_patterns,
                                                                            metacluster_seqlets,
                                                                            self.prior_results,
                                                                            metacluster_idx)

            sub_metacluster_results.seqlets_to_patterns_result = \
                    seqlets_to_patterns_result

        return results

