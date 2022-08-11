# metaclusterers.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import numpy as np

def _seqlet_to_attr(seqlet, central_window, value_transformer):
	flank = int(0.5*(len(seqlet)-central_window))
	track_values = seqlet['task0_contrib_scores'].fwd[flank:-flank]
	return value_transformer(val=np.sum(track_values))

def sign_split_seqlets(seqlets, central_window, value_transformer,
	min_cluster_size, threshold):
	attr_scores = np.array([_seqlet_to_attr(seqlet, central_window, 
		value_transformer) for seqlet in seqlets])

	weak_thresh_attrs = np.sign(attr_scores).astype('int32')
	weak_thresh_attrs[np.abs(attr_scores) < threshold] = 0

	weak_thresh_attrs_ = weak_thresh_attrs[weak_thresh_attrs != 0]
	weak_sign_count = np.unique(weak_thresh_attrs_, return_counts=True)

	activity_patterns = np.unique(weak_thresh_attrs)

	final_surviving_activity_patterns = [val for val, count in
		zip(*weak_sign_count) if count > min_cluster_size]

	idxs = np.argsort(weak_sign_count[-1])[::-1]
	sorted_activity_patterns = weak_sign_count[0][idxs]
	
	pattern_to_cluster_idx = dict([(str(x[1]),x[0]) 
		for x in enumerate(sorted_activity_patterns)])
	metacluster_idx_to_activity_pattern = dict([(x[0],str(x[1])) 
		for x in enumerate(sorted_activity_patterns)])

	metacluster_indices = []
	for x in weak_thresh_attrs:
		if x in final_surviving_activity_patterns:
			metacluster_indices.append(pattern_to_cluster_idx[str(x)])
		else:
			metacluster_indices.append(-1)

	return {
		'metacluster_indices': metacluster_indices,
		'attribute_vectors': attr_scores,
		'pattern_to_cluster_idx': pattern_to_cluster_idx,
		'metacluster_idx_to_activity_pattern': 
			metacluster_idx_to_activity_pattern,
		'val_transformer': value_transformer,
		'final_surviving_activity_patterns': final_surviving_activity_patterns,
		'threshold': threshold
	}
