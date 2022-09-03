from collections import OrderedDict
import numpy as np

from .seqlets_to_patterns import TfModiscoSeqletsToPatternsFactory
from . import core
from . import coordproducers

def _sign_split_seqlets(seqlets, central_window, min_cluster_size, threshold):
	attr_scores = []
	for seqlet in seqlets:
		flank = int(0.5*(len(seqlet)-central_window))
		val = np.sum(seqlet.snippets['task0_contrib_scores'].fwd[flank:-flank])
		attr_scores.append(val)

	pos_count = (attr_scores > threshold).sum()
	neg_count =  (attr_scores < -threshold).sum()
	counts = sorted([(-1, neg_count), (1, pos_count)], key=lambda x: -x[1])

	final_surviving_activity_patterns = [sign for sign, count
		in counts if count > min_cluster_size]
	
	cluster_idxs = {str(x[1][0]): x[0] for x in enumerate(counts)}
	activity_patterns = {x[0]: int(x[1][0]) for x in enumerate(counts)}

	metacluster_indices = []
	for x in attr_scores:
		val = int(np.sign(x) * (np.abs(x) > threshold))

		if val in final_surviving_activity_patterns:
			metacluster_indices.append(cluster_idxs[str(val)])
		else:
			metacluster_indices.append(-1)

	return {
		'metacluster_indices': metacluster_indices,
		'pattern_to_cluster_idx': cluster_idxs,
		'activity_patterns': activity_patterns,
	}


def TfModiscoWorkflow(contrib_scores, hypothetical_contribs, 
	one_hot, sliding_window_size=21, flank_size=10, overlap_portion=0.5,
	min_metacluster_size=100,
	weak_threshold_for_counting_sign=0.8, max_seqlets_per_metacluster=20000,
	target_seqlet_fdr=0.2, min_passing_windows_frac=0.03,
	max_passing_windows_frac=0.2, n_leiden_runs=50, n_cores=10, verbose=True):

	print("starting workflow")

	track_set = core.TrackSet(one_hot=one_hot, contrib_scores=contrib_scores,
		hypothetical_contribs=hypothetical_contribs)

	seqlet_coords = coordproducers.extract_seqlets(
		attribution_scores=contrib_scores.sum(axis=2),
		window_size=sliding_window_size,
		flank=flank_size,
		suppress=(int(0.5*sliding_window_size) + flank_size),
		target_fdr=target_seqlet_fdr,
		min_passing_windows_frac=min_passing_windows_frac,
		max_passing_windows_frac=max_passing_windows_frac,
		weak_threshold_for_counting_sign=weak_threshold_for_counting_sign) 

	seqlets = track_set.create_seqlets(seqlets=seqlet_coords['seqlets']) 

	multitask_seqlet_creation_results = {
		'final_seqlets': seqlets,
		'task_name_to_coord_producer_results': {
			'task0': seqlet_coords
		}
	}

	metaclustering_results = _sign_split_seqlets(seqlets, 
		min_cluster_size=min_metacluster_size, 
		central_window=sliding_window_size,					
		threshold=seqlet_coords['threshold'])

	metacluster_indices = np.array(metaclustering_results['metacluster_indices'])
	num_metaclusters = max(metacluster_indices)+1
	submetacluster_results = OrderedDict()

	for idx in range(num_metaclusters):
		metacluster_seqlets = [seqlet for seqlet, idx_ in zip(
			seqlets, metacluster_indices) if idx_ == idx]
		metacluster_seqlets = metacluster_seqlets[:max_seqlets_per_metacluster]

		seqlets_to_patterns_results = TfModiscoSeqletsToPatternsFactory(
			one_hot_sequence=one_hot, contrib_scores=contrib_scores, 
			hypothetical_contribs=hypothetical_contribs, 
			seqlets=metacluster_seqlets,
			track_set=track_set,
			track_signs=metaclustering_results['activity_patterns'][idx],
			n_leiden_runs=n_leiden_runs,
			n_cores=n_cores)

		submetacluster_results[idx] = {
			'metacluster_size': len(metacluster_seqlets),
			'seqlets': metacluster_seqlets,
			'seqlets_to_patterns_result': seqlets_to_patterns_results
		}

	print("done!!")

	return multitask_seqlet_creation_results, metaclustering_results, submetacluster_results
