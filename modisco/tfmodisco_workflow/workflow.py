from __future__ import division, print_function, absolute_import
from . import affinitymat as affmat
from . import cluster
from . import aggregator
from . import core
from . import util
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import itertools
import time
import sys
import h5py
import json


class TfModiscoResults(object):

    def __init__(self,
                 task_names,
                 multitask_seqlet_creation_results,
                 seqlet_attribute_vectors,
                 metaclustering_results,
                 metacluster_idx_to_submetacluster_results,
                 **kwargs):
        self.task_names = task_names
        self.multitask_seqlet_creation_results =\
                multitask_seqlet_creation_results
        self.seqlet_attribute_vectors = seqlet_attribute_vectors
        self.metaclustering_results = metaclustering_results
        self.metacluster_idx_to_submetacluster_results =\
            metacluster_idx_to_submetacluster_results

        self.__dict__.update(**kwargs)

    def save_hdf5(self, grp):
        util.save_string_list(string_list=self.task_names, 
                              dset_name="task_names", grp=grp)
        self.multitask_seqlet_creation_results.save_hdf(
            grp.create_group("multitask_seqlet_creation_results"))
        self.metaclustering_results.save_hdf(
            grp.create_group("metaclustering_results"))

        metacluster_idx_to_submetacluster_results_group = grp.create_group(
                                "metacluster_idx_to_submetacluster_results")
        for idx in metacluster_idx_to_submetacluster_results:
            self.metacluster_idx_to_submetacluster_results[idx].save_hdf5(
                grp=metaclusters_group.create_group("metacluster"+str(idx))) 


class SubMetaclusterResults(object):

    def __init__(self, metacluster_size, activity_pattern,
                       seqlets, seqlets_to_patterns_result):
        self.metacluster_size = metacluster_size
        self.activity_pattern = activity_pattern
        self.seqlets = seqlets
        self.seqlets_to_patterns_result

    def save_hdf5(self, grp):
        grp.attrs['size'] = metacluster_size
        grp.create_dataset('activity_pattern', data=activity_pattern)
        util.save_seqlet_coords(seqlets=self.seqlets,
                                dset_name="seqlets", grp=grp)   
        self.seqlets_to_patterns_result.save_hdf5(
            grp=grp.create_dataset('seqlets_to_patterns_result'))


def run_workflow(task_names, contrib_scores, hypothetical_contribs,
                 one_hot, coord_producer, overlap_resolver,
                 metaclusterer,
                 seqlets_to_patterns_factory,
                 verbose=True):

    contrib_scores_tracks = [
        modisco.core.DataTrack(
            name=key+"_contrib_scores",
            fwd_tracks=contrib_scores[key],
            rev_tracks=contrib_scores[key][:,::-1,::-1],
            has_pos_axis=True) for key in task_names] 

    hypothetical_contribs_tracks = [
        modisco.core.DataTrack(name=key+"_hypothetical_contribs",
                       fwd_tracks=hypothetical_contribs[key],
                       rev_tracks=hypothetical_contribs[key][:,::-1,::-1],
                       has_pos_axis=True)
                       for key in task_names]

    onehot_track = modisco.core.DataTrack(name="sequence", fwd_tracks=onehot,
                               rev_tracks=onehot[:,::-1,::-1],
                               has_pos_axis=True)

    track_set = modisco.core.TrackSet(
                    data_tracks=contrib_scores_tracks
                    +hypothetical_contribs_tracks+[onehot_track])

    per_position_contrib_scores = dict([
        (x, np.sum(contrib_scores[x],axis=2)) for x in task_names])

    task_name_to_labeler = dict([
        (task_name, modisco.core.SignedContribThresholdLabeler(
            flank_to_ignore=flank,
            name=task_name+"_label",
            track_name=task_name+"_contrib_scores"))
         for task_name in task_names]) 

    multitask_seqlet_creation_results = modisco.core.MultiTaskSeqletCreation(
        coord_producer=coord_producer,
        track_set=track_set,
        overlap_resolver=overlap_resolver)(
            task_name_to_score_track=per_position_contrib_scores,
            task_name_to_labeler=task_name_to_labeler)

    seqlets = multitask_seqlet_creation_results.final_seqlets

    attribute_vectors = (np.array([
                          [x[key+"_label"] for key in task_names]
                           for x in seqlets]))

    metaclustering_results = metaclusterer(attribute_vectors)
    metacluster_indices = metaclustering_results.clusters_mapping 
    metacluster_idx_to_activity_pattern =\
        metaclustering_results.metacluster_idx_to_activity_pattern

    num_metaclusters = max(clusters_mapping)+1
    metacluster_indices = np.array(clusters_mapping)
    metacluster_sizes = [np.sum(metacluster_idx==metacluster_indices)
                          for metacluster_idx in range(num_metaclusters)]
    if (self.verbose):
        print("Metacluster sizes: ",metacluster_sizes)
        print("Idx to activities: ",metacluster_idx_to_activity_pattern)
        sys.out.flush()

    metacluster_idx_to_submetacluster_results = {}

    for metacluster_idx, metacluster_size in\
        sorted(enumerate(metacluster_sizes), key=lambda x: x[1]):
        print("On metacluster "+str(metacluster_idx))
        print("Metacluster size", metacluster_size)
        sys.out.flush()
        metacluster_activities = [
            int(x) for x in
            metacluster_idx_to_activity_pattern[metacluster_idx].split(",")]
        assert len(seqlets)==len(metacluster_indices)
        metacluster_seqlets = [
            x[0] for x in zip(seqlets, metacluster_indices)
            if x[1]==metacluster_idx]
        relevant_task_names, relevant_task_signs =\
            zip(*[(x[0], x[1]) for x in
                zip(task_names, metacluster_activities) if x[1] != 0])
        print('Relevant tasks: ', relevant_task_names)
        print('Relevant signs: ', relevant_task_signs)
        sys.out.flush()
        if (len(relevant_task_names) == 0):
            print("No tasks found relevant; skipping")
            sys.out.flush()
            continue
        
        seqlets_to_patterns = seqlets_to_patterns_factory(
            track_set=track_set,
            onehot_track_name="sequence",
            contrib_scores_track_names =\
                [key+"_contrib_scores" for key in relevant_task_names],
            hypothetical_contribs_track_names=\
                [key+"_hypothetical_contribs" for key in relevant_task_names],
            track_signs=relevant_task_signs,
            other_comparison_track_names=[])

        seqlets_to_patterns_result = seqlets_to_patterns(metacluster_seqlets)
        metacluster_idx_to_submetacluster_results[metacluster_idx] =\
            SubMetaclusterResults(
                metacluster_size=metacluster_size,
                activity_pattern=np.array(metacluster_activities),
                seqlets=metacluster_seqlets,
                seqlets_to_patterns_result=seqlets_to_patterns_result)

    return TfModiscoResults(
             task_names=task_names,
             seqlet_creation_results=multitask_seqlet_creation_results,
             seqlet_attribute_vectors=attribute_vectors,
             metaclustering_results=metaclustering_results,
             metacluster_idx_to_submetacluster_results)

