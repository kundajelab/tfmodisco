from __future__ import division, print_function, absolute_import
from collections import defaultdict, OrderedDict, Counter
import numpy as np

from .seqlets_to_patterns import TfModiscoSeqletsToPatternsFactory
from .. import core
from .. import coordproducers
from .. import metaclusterers
from .. import util
from .. import value_provider

import line_profiler
profile = line_profiler.LineProfiler()

def prep_track_set(task_names, contrib_scores,
					hypothetical_contribs, one_hot,
					revcomp=True):

	contrib_scores_tracks = [
		core.DataTrack(
			name=key+"_contrib_scores",
			fwd_tracks=contrib_scores[key],
			rev_tracks=(([x[::-1, ::-1] for x in 
						 contrib_scores[key]])
						if revcomp else
						 None),
			has_pos_axis=True) for key in task_names] 

	hypothetical_contribs_tracks = [
		core.DataTrack(name=key+"_hypothetical_contribs",
					   fwd_tracks=hypothetical_contribs[key],
					   rev_tracks=(([x[::-1, ::-1] for x in 
									 hypothetical_contribs[key]])
								   if revcomp else
									None),
					   has_pos_axis=True)
					   for key in task_names]

	onehot_track = core.DataTrack(
						name="sequence", fwd_tracks=one_hot,
						rev_tracks=([x[::-1, ::-1] for x in one_hot]
									if revcomp else None),
						has_pos_axis=True)

	track_set = core.TrackSet(
					data_tracks=contrib_scores_tracks
					+hypothetical_contribs_tracks+[onehot_track])

	return track_set

def TfModiscoWorkflow(task_names, contrib_scores,
				 hypothetical_contribs, one_hot,
				 sliding_window_size=21, 
				 flank_size=10,
				 overlap_portion=0.5,
				 min_metacluster_size=100,
				 min_metacluster_size_frac=0.01,
				 weak_threshold_for_counting_sign=0.8,
				 max_seqlets_per_metacluster=20000,
				 target_seqlet_fdr=0.2,
				 min_passing_windows_frac=0.03,
				 max_passing_windows_frac=0.2,
				 verbose=True):

		track_set = prep_track_set(
						task_names=task_names,
						contrib_scores=contrib_scores,
						hypothetical_contribs=hypothetical_contribs,
						one_hot=one_hot)

		
		seqlet_coords = coordproducers.FixedWindowAroundChunks(
			attribution_scores=contrib_scores['task0'].sum(axis=2),
			window_size=sliding_window_size,
			flank=flank_size,
			suppress=(int(0.5*sliding_window_size) + flank_size),
			target_fdr=target_seqlet_fdr,
			min_passing_windows_frac=min_passing_windows_frac,
			max_passing_windows_frac=max_passing_windows_frac,
			max_seqlets_total=None,
			verbose=verbose) 

		seqlets = track_set.create_seqlets(coords=seqlet_coords['coords']) 
		seqlets = core.SeqletsOverlapResolver(seqlets, overlap_portion)

		multitask_seqlet_creation_results = {
			'final_seqlets': seqlets,
			'task_name_to_coord_producer_results': {
				'task0': seqlet_coords
			}
		}

		#find the weakest transformed threshold used across all tasks
		weakest_transformed_thresh = min(
			seqlet_coords['transformed_pos_threshold'], 
			abs(seqlet_coords['transformed_neg_threshold'])
		) - 0.0001
		
		if int(min_metacluster_size_frac * len(seqlets)) > min_metacluster_size:
			min_metacluster_size = int(min_metacluster_size_frac * len(seqlets))

		if weak_threshold_for_counting_sign > weakest_transformed_thresh:
			weak_threshold_for_counting_sign = weakest_transformed_thresh


		metaclustering_results = metaclusterers.sign_split_seqlets(seqlets, 
			min_cluster_size=min_metacluster_size, 
			central_window=sliding_window_size, 
			value_transformer=seqlet_coords['val_transformer'],					
			threshold=weak_threshold_for_counting_sign)
		metaclustering_results['_threshold'] = weakest_transformed_thresh

		metacluster_indices = np.array(metaclustering_results['metacluster_indices'])
		metacluster_idx_to_activity_pattern = metaclustering_results['metacluster_idx_to_activity_pattern']

		num_metaclusters = max(metacluster_indices)+1
		metacluster_sizes = [np.sum(metacluster_idx==metacluster_indices)
							  for metacluster_idx in range(num_metaclusters)]

		metacluster_idx_to_submetacluster_results = OrderedDict()

		for metacluster_idx, metacluster_size in sorted(enumerate(metacluster_sizes), key=lambda x: x[1]):			
			metacluster_activities = [int(x) for x in
				metacluster_idx_to_activity_pattern[metacluster_idx].split(",")]
			
			metacluster_seqlets = [
				x[0] for x in zip(seqlets, metacluster_indices)
				if x[1]==metacluster_idx][:max_seqlets_per_metacluster]
			
			relevant_task_names, relevant_task_signs =\
				zip(*[(x[0], x[1]) for x in
					zip(task_names, metacluster_activities) if x[1] != 0])
			
			seqlets_to_patterns_results = TfModiscoSeqletsToPatternsFactory(
			  seqlets=metacluster_seqlets,
			  track_set=track_set,
			  onehot_track_name="sequence",
			  contrib_scores_track_names = [key + "_contrib_scores" for key in relevant_task_names],
			  hypothetical_contribs_track_names=[key + "_hypothetical_contribs" for key in relevant_task_names],
			  track_signs=relevant_task_signs,
			  other_comparison_track_names=[])

			metacluster_idx_to_submetacluster_results[metacluster_idx] = {
				'metacluster_size': metacluster_size, 
				'activity_pattern': np.array(metacluster_activities), 
				'seqlets': metacluster_seqlets,
				'seqlets_to_patterns_result': seqlets_to_patterns_results
			}

		return task_names, multitask_seqlet_creation_results, metaclustering_results, metacluster_idx_to_submetacluster_results
