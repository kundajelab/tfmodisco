from collections import OrderedDict
import numpy as np

from .seqlets_to_patterns import TfModiscoSeqletsToPatternsFactory
from . import core
from . import coordproducers



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

def _sign_split_seqlets(seqlets, central_window, distribution,
	min_cluster_size, threshold):
	threshold_val = distribution[int(threshold * len(distribution))]

	attr_scores = []
	for seqlet in seqlets:
		flank = int(0.5*(len(seqlet)-central_window))
		val = np.sum(seqlet['task0_contrib_scores'].fwd[flank:-flank])
		attr_scores.append(val)

	pos_count = (attr_scores > threshold_val).sum()
	neg_count =  (attr_scores < -threshold_val).sum()
	counts = sorted([(-1, neg_count), (1, pos_count)], key=lambda x: -x[1])

	final_surviving_activity_patterns = [sign for sign, count
		in counts if count > min_cluster_size]
	
	cluster_idxs = {str(x[1][0]): x[0] for x in enumerate(counts)}
	activity_patterns = {x[0]: int(x[1][0]) for x in enumerate(counts)}

	metacluster_indices = []
	for x in attr_scores:
		val = int(np.sign(x) * (np.abs(x) > threshold_val))

		if val in final_surviving_activity_patterns:
			metacluster_indices.append(cluster_idxs[str(val)])
		else:
			metacluster_indices.append(-1)

	return {
		'metacluster_indices': metacluster_indices,
		'attribute_vectors': attr_scores,
		'pattern_to_cluster_idx': cluster_idxs,
		'metacluster_idx_to_activity_pattern': activity_patterns,
		'distribution': distribution,
		'final_surviving_activity_patterns': final_surviving_activity_patterns,
		'threshold': threshold
	}


def TfModiscoWorkflow(task_names, contrib_scores, hypothetical_contribs, 
	one_hot, sliding_window_size=21, flank_size=10, overlap_portion=0.5,
	min_metacluster_size=100,
	weak_threshold_for_counting_sign=0.8, max_seqlets_per_metacluster=20000,
	target_seqlet_fdr=0.2, min_passing_windows_frac=0.03,
	max_passing_windows_frac=0.2, n_leiden_runs=50, n_cores=10, verbose=True):

	print("starting workflow")

	track_set = prep_track_set(
					task_names=task_names,
					contrib_scores=contrib_scores,
					hypothetical_contribs=hypothetical_contribs,
					one_hot=one_hot)
	
	seqlet_coords = coordproducers.extract_seqlet_coords(
		attribution_scores=contrib_scores['task0'].sum(axis=2),
		window_size=sliding_window_size,
		flank=flank_size,
		suppress=(int(0.5*sliding_window_size) + flank_size),
		target_fdr=target_seqlet_fdr,
		min_passing_windows_frac=min_passing_windows_frac,
		max_passing_windows_frac=max_passing_windows_frac) 

	seqlets = track_set.create_seqlets(coords=seqlet_coords['coords']) 

	multitask_seqlet_creation_results = {
		'final_seqlets': seqlets,
		'task_name_to_coord_producer_results': {
			'task0': seqlet_coords
		}
	}

	#find the weakest transformed threshold used across all tasks
	weak_transformed_thresh = min(min(
		seqlet_coords['transformed_pos_threshold'], 
		abs(seqlet_coords['transformed_neg_threshold'])
	) - 0.0001, weak_threshold_for_counting_sign)

	metaclustering_results = _sign_split_seqlets(seqlets, 
		min_cluster_size=min_metacluster_size, 
		central_window=sliding_window_size, 
		distribution=seqlet_coords['distribution'],					
		threshold=weak_threshold_for_counting_sign)
	
	# not necessary
	metaclustering_results['_threshold'] = min(
		seqlet_coords['transformed_pos_threshold'], 
		abs(seqlet_coords['transformed_neg_threshold'])
	) - 0.0001

	metacluster_indices = np.array(metaclustering_results['metacluster_indices'])
	metacluster_idx_to_activity_pattern = metaclustering_results['metacluster_idx_to_activity_pattern']

	num_metaclusters = max(metacluster_indices)+1
	metacluster_sizes = [np.sum(metacluster_idx==metacluster_indices)
						  for metacluster_idx in range(num_metaclusters)]

	metacluster_idx_to_submetacluster_results = OrderedDict()

	for metacluster_idx, metacluster_size in enumerate(metacluster_sizes):
		metacluster_activities = [metacluster_idx_to_activity_pattern[metacluster_idx]]

		metacluster_seqlets = [seqlet for seqlet, idx in zip(
			seqlets, metacluster_indices) if idx == metacluster_idx]
		metacluster_seqlets = metacluster_seqlets[:max_seqlets_per_metacluster]
		
		relevant_task_signs = metacluster_idx_to_activity_pattern[metacluster_idx]

		seqlets_to_patterns_results = TfModiscoSeqletsToPatternsFactory(
			one_hot_sequence=one_hot, contrib_scores=contrib_scores['task0'], 
			hypothetical_contribs=hypothetical_contribs['task0'], 
			seqlets=metacluster_seqlets,
			track_set=track_set,
			track_signs=relevant_task_signs,
			n_leiden_runs=n_leiden_runs,
			n_cores=n_cores)

		metacluster_idx_to_submetacluster_results[metacluster_idx] = {
			'metacluster_size': len(metacluster_seqlets),
			'activity_pattern': np.array(metacluster_activities), 
			'seqlets': metacluster_seqlets,
			'seqlets_to_patterns_result': seqlets_to_patterns_results
		}

	print("done!!")

	return task_names, multitask_seqlet_creation_results, metaclustering_results, metacluster_idx_to_submetacluster_results
