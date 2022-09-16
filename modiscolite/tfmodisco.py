# tfmodisco.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import numpy as np

import scipy
import scipy.sparse

from collections import OrderedDict
from collections import defaultdict

from . import affinitymat
from . import aggregator
from . import extract_seqlets
from . import core
from . import util
from . import cluster

def _density_adaptation(affmat_nn, seqlet_neighbors, tsne_perplexity):
	eps = 0.0000001

	rows, cols, data = [], [], []
	for row in range(len(affmat_nn)):
		for col, datum in zip(seqlet_neighbors[row], affmat_nn[row]):
			rows.append(row)
			cols.append(col)
			data.append(datum)

	affmat_nn = scipy.sparse.csr_matrix((data, (rows, cols)), 
		shape=(len(affmat_nn), len(affmat_nn)), dtype='float64')
	
	affmat_nn.data = np.maximum(np.log((1.0/(0.5*np.maximum(affmat_nn.data, eps)))-1), 0)
	affmat_nn.eliminate_zeros()

	counts_nn = scipy.sparse.csr_matrix((np.ones_like(affmat_nn.data), 
		affmat_nn.indices, affmat_nn.indptr), shape=affmat_nn.shape, dtype='float32')

	affmat_nn += affmat_nn.T
	counts_nn += counts_nn.T
	affmat_nn.data /= counts_nn.data
	del counts_nn

	betas = [util.binary_search_perplexity(tsne_perplexity, affmat_nn[i].data) for i in range(affmat_nn.shape[0])]
	normfactors = np.array([np.exp(-np.array(affmat_nn[i].data)/beta).sum()+1 for i, beta in enumerate(betas)])

	for i in range(affmat_nn.shape[0]):
		for j_idx in range(affmat_nn.indptr[i], affmat_nn.indptr[i+1]):
			j = affmat_nn.indices[j_idx]
			distance = affmat_nn.data[j_idx]

			rbf_i = np.exp(-distance / betas[i]) / normfactors[i]
			rbf_j = np.exp(-distance / betas[j]) / normfactors[j]

			affmat_nn.data[j_idx] = np.sqrt(rbf_i * rbf_j)

	affmat_diags = scipy.sparse.diags(1.0 / normfactors)
	affmat_nn += affmat_diags
	return affmat_nn

def _filter_patterns(patterns, min_seqlet_support, window_size, 
	min_ic_in_window, background, ppm_pseudocount):
	passing_patterns, filtered_patterns = [], []
	for pattern in patterns:
		if len(pattern.seqlets) < min_seqlet_support:
			filtered_patterns.append(pattern)
			continue

		ppm = pattern.sequence
		per_position_ic = util.compute_per_position_ic(
			ppm=ppm, background=background, pseudocount=ppm_pseudocount)

		if len(per_position_ic) < window_size:       
			if np.sum(per_position_ic) < min_ic_in_window:
				filtered_patterns.append(pattern)
				continue
		else:
			#do the sliding window sum rearrangement
			windowed_ic = np.sum(util.rolling_window(
				a=per_position_ic, window=window_size), axis=-1)

			if np.max(windowed_ic) < min_ic_in_window:
				filtered_patterns.append(pattern)
				continue

		passing_patterns.append(pattern)

	return passing_patterns, filtered_patterns


def _motif_from_clusters(seqlets, track_set, min_overlap,
	min_frac, min_num, flank_to_add, window_size, bg_freq, cluster_indices, 
	track_sign):

	seqlet_sort_metric = lambda x: -np.sum(np.abs(x.contrib_scores))
	num_clusters = max(cluster_indices+1)
	cluster_to_seqlets = defaultdict(list) 

	for seqlet,idx in zip(seqlets, cluster_indices):
		cluster_to_seqlets[idx].append(seqlet)

	cluster_to_motif = []

	for i in range(num_clusters):
		sorted_seqlets = sorted(cluster_to_seqlets[i], key=seqlet_sort_metric) 
		pattern = core.AggregatedSeqlet([sorted_seqlets[0]])

		if len(sorted_seqlets) > 1:
			pattern = aggregator.merge_in_seqlets_filledges(
				parent_pattern=pattern, seqlets_to_merge=sorted_seqlets[1:],
				track_set=track_set, metric=affinitymat.jaccard,
				min_overlap=min_overlap)

		pattern = aggregator.polish_pattern(pattern, min_frac=min_frac, 
			min_num=min_num, track_set=track_set, flank=flank_to_add, 
			window_size=window_size, bg_freq=bg_freq)

		if pattern is not None:
			if np.sign(np.sum(pattern.contrib_scores)) == track_sign:
				cluster_to_motif.append(pattern)

	return cluster_to_motif


def _filter_by_correlation(seqlets, seqlet_neighbors, coarse_affmat_nn, 
	fine_affmat_nn, correlation_threshold):

	correlations = []
	for fine_affmat_row, coarse_affmat_row in zip(fine_affmat_nn, coarse_affmat_nn):
		to_compare_mask = np.abs(fine_affmat_row) > 0
		corr = scipy.stats.spearmanr(fine_affmat_row[to_compare_mask],
			coarse_affmat_row[to_compare_mask])
		correlations.append(corr.correlation)

	correlations = np.array(correlations)
	filtered_rows_mask = np.array(correlations) > correlation_threshold

	filtered_seqlets = [seqlet for seqlet, mask in zip(seqlets, 
		filtered_rows_mask) if mask == True]

	#figure out a mapping from pre-filtering to the
	# post-filtering indices
	new_idx_mapping = np.cumsum(filtered_rows_mask) - 1
	retained_indices = set(np.where(filtered_rows_mask == True)[0])

	filtered_neighbors = []
	filtered_affmat_nn = []
	for old_row_idx, (old_neighbors, affmat_row) in enumerate(zip(seqlet_neighbors, fine_affmat_nn)):
		if old_row_idx in retained_indices:
			filtered_old_neighbors = [neighbor for neighbor in old_neighbors if neighbor in retained_indices]
			filtered_affmat_row = [affmatval for affmatval, neighbor in zip(affmat_row,old_neighbors) if neighbor in retained_indices]
			filtered_neighbors_row = [new_idx_mapping[neighbor] for neighbor in filtered_old_neighbors]
			filtered_neighbors.append(filtered_neighbors_row)
			filtered_affmat_nn.append(filtered_affmat_row)

	return filtered_seqlets, filtered_neighbors, filtered_affmat_nn


def _extract_seqlet_data(one_hot_sequence, contrib_scores, hypothetical_contribs, seqlets):
	X_ohe = []
	X_contrib = []
	X_hypo_contrib = []

	for seqlet in seqlets:
		idx = seqlet.example_idx
		start, end = seqlet.start, seqlet.end

		X_ohe.append(one_hot_sequence[idx][start:end])
		X_contrib.append(contrib_scores[idx][start:end])
		X_hypo_contrib.append(hypothetical_contribs[idx][start:end])

	return X_ohe, X_contrib, X_hypo_contrib

def seqlets_to_patterns(seqlets, one_hot_sequence, contrib_scores, 
	hypothetical_contribs, track_set, track_signs=None, 
	min_overlap_while_sliding=0.7, nearest_neighbors_to_compute=500, 
	affmat_correlation_threshold=0.15, tsne_perplexity=10.0, 
	n_leiden_iterations=-1, n_leiden_runs=50, frac_support_to_trim_to=0.2,
	min_num_to_trim_to=30, trim_to_window_size=20, initial_flank_to_add=5,
	prob_and_pertrack_sim_merge_thresholds=[(0.8,0.8), (0.5, 0.85), (0.2, 0.9)],
	prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],
	subcluster_perplexity=50, merging_max_seqlets_subsample=300,
	final_min_cluster_size=20,min_ic_in_window=0.6, min_ic_windowsize=6,
	ppm_pseudocount=0.001):

	X_ohe, X_contrib, X_hypo_contrib = _extract_seqlet_data(one_hot_sequence, 
		contrib_scores, hypothetical_contribs, seqlets)

	bg_freq = np.mean(X_ohe, axis=(0,1))
	del X_ohe, X_contrib, X_hypo_contrib

	other_config={
			 'onehot_track_name': "sequence",
			 'contrib_scores_track_names': ["task0_contrib_scores"],
			 'hypothetical_contribs_track_names':["task0_hypothetical_contribs"],
			 'track_signs': [track_signs], 
			 'other_comparison_track_names': []
	}

	seqlets_sorter = (lambda arr: sorted(arr, key=lambda x:
		-np.sum(np.abs(x.contrib_scores))))

	seqlets = seqlets_sorter(seqlets)

	failure = {
				'each_round_initcluster_motifs': None,
				'patterns': None,
				'remaining_patterns': None,
				'pattern_merge_hierarchy': None,
				'cluster_results': None, 
				'total_time_taken': None,
				'success': False,
				'seqlets': None,
				'affmat': None,
				'other_config': other_config
	}

	for round_idx in range(2):			
		if len(seqlets) == 0:
			return failure

		# Step 1: Generate coarse resolution
		coarse_affmat_nn, seqlet_neighbors = affinitymat.cosine_similarity_from_seqlets(
			seqlets=seqlets, n_neighbors=nearest_neighbors_to_compute, sign=track_signs)

		# Step 2: Generate fine representation
		fine_affmat_nn = affinitymat.jaccard_from_seqlets(
			seqlets=seqlets, seqlet_neighbors=seqlet_neighbors,
			min_overlap=min_overlap_while_sliding)

		if round_idx == 0:
			filtered_seqlets, seqlet_neighbors, filtered_affmat_nn = (
				_filter_by_correlation(seqlets, seqlet_neighbors, 
					coarse_affmat_nn, fine_affmat_nn, 
					affmat_correlation_threshold))
		else:
			filtered_seqlets = seqlets
			filtered_affmat_nn = fine_affmat_nn

		del coarse_affmat_nn
		del fine_affmat_nn
		del seqlets

		# Step 4: Density adaptation
		csr_density_adapted_affmat = _density_adaptation(
			filtered_affmat_nn, seqlet_neighbors, tsne_perplexity)

		del filtered_affmat_nn
		del seqlet_neighbors

		# Step 5: Clustering
		cluster_results = cluster.LeidenCluster(
			csr_density_adapted_affmat,
			n_seeds=n_leiden_runs,
			n_leiden_iterations=n_leiden_iterations)

		del csr_density_adapted_affmat

		motifs = _motif_from_clusters(filtered_seqlets, 
			track_set=track_set, 
			min_overlap=min_overlap_while_sliding, 
			min_frac=frac_support_to_trim_to, 
			min_num=min_num_to_trim_to, 
			flank_to_add=initial_flank_to_add, 
			window_size=trim_to_window_size, 
			bg_freq=bg_freq, 
			cluster_indices=cluster_results['cluster_indices'], 
			track_sign=track_signs)


		#obtain unique seqlets from adjusted motifs
		seqlets = list(dict([(y.string, y)
						 for x in motifs for y in x.seqlets]).values())

	del seqlets

	merged_patterns, pattern_merge_hierarchy = aggregator._detect_spurious_merging(
		patterns=motifs, track_set=track_set, perplexity=subcluster_perplexity, 
		min_in_subcluster=max(final_min_cluster_size, subcluster_perplexity), 
		min_overlap=min_overlap_while_sliding,
		prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
		prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
		min_frac=frac_support_to_trim_to, min_num=min_num_to_trim_to,
		flank_to_add=initial_flank_to_add,
		window_size=trim_to_window_size, bg_freq=bg_freq,
		max_seqlets_subsample=merging_max_seqlets_subsample,
		n_seeds=n_leiden_runs)

	#Now start merging patterns 
	#merged_patterns, pattern_merge_hierarchy = similar_patterns_collapser(patterns=split_patterns) 
	merged_patterns = sorted(merged_patterns, key=lambda x: -len(x.seqlets))

	final_patterns, remaining_patterns = _filter_patterns(merged_patterns, 
		min_seqlet_support=final_min_cluster_size, 
		window_size=min_ic_windowsize, min_ic_in_window=min_ic_in_window, 
		background=bg_freq, ppm_pseudocount=ppm_pseudocount)

	#apply subclustering procedure on the final patterns
	for patternidx, pattern in enumerate(final_patterns):
		pattern.compute_subpatterns(subcluster_perplexity, 
			n_seeds=n_leiden_runs, n_iterations=n_leiden_iterations)

	return {
		'success': True,
		'each_round_initcluster_motifs': None,             
		'patterns': final_patterns,
		'remaining_patterns': remaining_patterns,
		'seqlets': filtered_seqlets, 
		'cluster_results': cluster_results, 
		'merged_patterns': merged_patterns,
		'pattern_merge_hierarchy': pattern_merge_hierarchy,
		'other_config': other_config
	}

def _sign_split_seqlets(seqlets, central_window, min_cluster_size, threshold):
	attr_scores = []
	for seqlet in seqlets:
		flank = int(0.5*(len(seqlet)-central_window))
		val = np.sum(seqlet.contrib_scores[flank:-flank])
		attr_scores.append(val)

	pos_count = (attr_scores > threshold).sum()
	neg_count =  (attr_scores < -threshold).sum()
	counts = sorted([(-1, neg_count), (1, pos_count)], key=lambda x: -x[1])

	final_surviving_activity_patterns = [sign for sign, count
		in counts if count > min_cluster_size]
	
	cluster_idxs = {str(x[1][0]): x[0] for x in enumerate(counts)}
	activity_patterns = {x[0]: int(x[1][0]) for x in enumerate(counts)}

	metacluster = []
	for x in attr_scores:
		val = int(np.sign(x) * (np.abs(x) > threshold))

		if val in final_surviving_activity_patterns:
			metacluster.append(cluster_idxs[str(val)])
		else:
			metacluster.append(-1)

	return {
		'metacluster_indices': metacluster,
		'pattern_to_cluster_idx': cluster_idxs,
		'activity_patterns': activity_patterns,
	}


def TFMoDISco(one_hot, hypothetical_contribs, sliding_window_size=21, 
	flank_size=10, overlap_portion=0.5, min_metacluster_size=100,
	weak_threshold_for_counting_sign=0.8, max_seqlets_per_metacluster=20000,
	target_seqlet_fdr=0.2, min_passing_windows_frac=0.03,
	max_passing_windows_frac=0.2, n_leiden_runs=50):

	contrib_scores = np.multiply(one_hot, hypothetical_contribs)

	track_set = core.TrackSet(one_hot=one_hot, contrib_scores=contrib_scores,
		hypothetical_contribs=hypothetical_contribs)

	seqlet_coords = extract_seqlets.extract_seqlets(
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

	metacluster = np.array(metaclustering_results['metacluster_indices'])
	submetacluster_results = OrderedDict()

	for idx in range(max(metacluster)+1):
		metacluster_seqlets = [seqlet for seqlet, idx_ in zip(
			seqlets, metacluster) if idx_ == idx]
		metacluster_seqlets = metacluster_seqlets[:max_seqlets_per_metacluster]

		seqlets_to_patterns_results = seqlets_to_patterns(
			one_hot_sequence=one_hot, contrib_scores=contrib_scores, 
			hypothetical_contribs=hypothetical_contribs, 
			seqlets=metacluster_seqlets,
			track_set=track_set,
			track_signs=metaclustering_results['activity_patterns'][idx],
			n_leiden_runs=n_leiden_runs)

		submetacluster_results[idx] = {
			'metacluster_size': len(metacluster_seqlets),
			'seqlets': metacluster_seqlets,
			'seqlets_to_patterns_result': seqlets_to_patterns_results
		}

	return multitask_seqlet_creation_results, metaclustering_results, submetacluster_results
