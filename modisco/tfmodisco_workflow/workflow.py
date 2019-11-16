from __future__ import division, print_function, absolute_import
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import itertools
import time
import sys
import h5py
import json
from . import seqlets_to_patterns
from .. import core
from .. import coordproducers
from .. import metaclusterers
from .. import util
from .. import value_provider


def print_memory_use():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("MEMORY",process.memory_info().rss/1000000000)


class TfModiscoResults(object):

    #TODO: REWRITE ARGUMENTS
    def __init__(self,
                 task_names,
                 multitask_seqlet_creation_results,
                 metaclustering_results,
                 metacluster_idx_to_submetacluster_results,
                 **kwargs):
        self.task_names = task_names
        self.multitask_seqlet_creation_results =\
                multitask_seqlet_creation_results
        self.metaclustering_results = metaclustering_results
        self.metacluster_idx_to_submetacluster_results =\
            metacluster_idx_to_submetacluster_results

        self.__dict__.update(**kwargs)

    @classmethod
    def from_hdf5(cls, grp, track_set):
        task_names = util.load_string_list(dset_name="task_names",
                                           grp=grp)
        multitask_seqlet_creation_results =\
            core.MultiTaskSeqletCreationResults.from_hdf5(
                grp=grp["multitask_seqlet_creation_results"],
                track_set=track_set)
        metaclustering_results =\
            metaclusterers.MetaclusteringResults.from_hdf5(
                grp["metaclustering_results"])
        metacluster_idx_to_submetacluster_results = OrderedDict()
        metacluster_idx_to_submetacluster_results_group =\
            grp["metacluster_idx_to_submetacluster_results"]
        for metacluster_idx in metacluster_idx_to_submetacluster_results_group:
            metacluster_idx_to_submetacluster_results[metacluster_idx] =\
             SubMetaclusterResults.from_hdf5(
                grp=metacluster_idx_to_submetacluster_results_group[
                     metacluster_idx],
                track_set=track_set)

        return cls(task_names=task_names,
                   multitask_seqlet_creation_results=
                    multitask_seqlet_creation_results,
                   metaclustering_results=metaclustering_results,
                   metacluster_idx_to_submetacluster_results=
                    metacluster_idx_to_submetacluster_results)

    def save_hdf5(self, grp):
        util.save_string_list(string_list=self.task_names, 
                              dset_name="task_names", grp=grp)
        self.multitask_seqlet_creation_results.save_hdf5(
            grp.create_group("multitask_seqlet_creation_results"))
        self.metaclustering_results.save_hdf5(
            grp.create_group("metaclustering_results"))

        metacluster_idx_to_submetacluster_results_group = grp.create_group(
                                "metacluster_idx_to_submetacluster_results")
        for idx in self.metacluster_idx_to_submetacluster_results:
            self.metacluster_idx_to_submetacluster_results[idx].save_hdf5(
                grp=metacluster_idx_to_submetacluster_results_group
                    .create_group("metacluster_"+str(idx))) 



# self.coord_producer = coordproducers.FixedWindowAroundChunks(
#     sliding=self.sliding_window_size,
#     flank=self.flank_size,
#     suppress=(int(0.5*self.sliding_window_size)
#               + self.flank_size),
#     target_fdr=self.target_seqlet_fdr,
#     min_passing_windows_frac=self.min_passing_windows_frac,
#     max_passing_windows_frac=self.max_passing_windows_frac,
#     separate_pos_neg_thresholds=self.separate_pos_neg_thresholds,
#     max_seqlets_total=None,
#     verbose=self.verbose) 


def make_seqlets_to_patterns_settings(per_position_contrib_track_name,
                                      tracknames_for_finegrained_sim,
                                      tracknames_for_coarsegrained_sim,
                                      pattern_comparison_min_overlap):
    
    seqlets_sorter = (lambda arr:
                           sorted(arr,
                             key=lambda x:
                                  -np.sum(x[per_position_contrib_scores].fwd)))

    pattern_comparison_settings = pattern_comparison_settings =\
            affinitymat.core.PatternComparisonSettings(
                track_names=tracknames_for_finegrained_sim, 
                track_transformer=affinitymat.L1Normalizer(), 
                min_overlap=pattern_comparison_min_overlap)

    settings = util.enum(
        seqlets_sorter=seqlets_sorter
        pattern_comparison_settings=pattern_comparison_settings,
        tracknames_for_coarsegrained_sim=tracknames_for_coarsegrained_sim)
    return settings


class TfModiscoCoreWorkflow(object):

    def __init__(self, coord_producer, seqlets_to_patterns):
        self.coord_producer = coord_producer
        self.seqlets_to_patterns = seqlets_to_patterns

    def __call__(self, per_position_contrib_scores,
                       per_position_contrib_track_name,
                       null_per_pos_scores,
                       all_tracks,
                       tracknames_for_finegrained_sim,
                       tracknames_for_coarsegrained_sim):

        coord_producer_results = self.coord_producer(
                                    score_track=per_position_contrib_scores,
                                    null_track=null_per_pos_scores)
        track_set = core.TrackSet(
            data_tracks=all_tracks+[
                core.DataTrack(
                    name=per_position_contrib_track_name,
                    fwd_tracks=per_position_contrib_scores[:,:,None],
                    rev_tracks=per_position_contrib_scores[:,::-1,None],
                    has_pos_axis=False)])

        #create seqlets from the positive coordinates; user can flip
        # sign of per_position_contrib scores if needed
        seqlets = track_set.create_seqlets(
                    coords=coord_producer_results.pos_coords)
        print(str(len(seqlets))+" identified in total")

        settings = make_seqlets_to_patterns_settings(
                     per_position_contrib_track_name,
                     tracknames_for_finegrained_sim,
                     tracknames_for_coarsegrained_sim)

        seqlets_to_patterns_result = seqlets_to_patterns(
            seqlets=seqlets,
            settings=settings)

        return TfModiscoResults(
                 coord_producer_results=coord_producer_results,
                 seqlets_to_patterns_result=seqlets_to_patterns_result)
