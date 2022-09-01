from . import affinitymat
from . import aggregator
from . import core
from . import util
from . import gapped_kmer
from . import cluster
from joblib import Parallel, delayed
from collections import defaultdict
import numpy as np
import time

import scipy
import scipy.sparse

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

		ppm = pattern['sequence'].fwd
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

	seqlet_sort_metric = lambda x: -np.sum(np.abs(x["task0_contrib_scores"].fwd))
	num_clusters = max(cluster_indices+1)
	cluster_to_seqlets = defaultdict(list) 

	for seqlet,idx in zip(seqlets, cluster_indices):
		cluster_to_seqlets[idx].append(seqlet)

	cluster_to_motif = []

	for i in range(num_clusters):
		sorted_seqlets = sorted(cluster_to_seqlets[i], key=seqlet_sort_metric) 
		pattern = core.AggregatedSeqlet.from_seqlet(sorted_seqlets[0])

		if len(sorted_seqlets) > 1:
			pattern = aggregator.merge_in_seqlets_filledges(
				parent_pattern=pattern,
				seqlets_to_merge=sorted_seqlets[1:],
				track_set=track_set,
				metric=affinitymat.jaccard,
				min_overlap=min_overlap,
				track_transformer=affinitymat.L1Normalizer(),
				track_names=["task0_hypothetical_contribs", 
					"task0_contrib_scores"],
				verbose=True)

		pattern = aggregator._trim_to_frac_support([pattern], 
			min_frac=min_frac, min_num=min_num)[0]

		pattern = aggregator._expand_seqlets_to_fill_pattern([pattern], 
			track_set=track_set, left_flank_to_add=flank_to_add,
			right_flank_to_add=flank_to_add)[0]

		pattern = aggregator._trim_to_best_window_by_ic([pattern],
				window_size=window_size,
				bg_freq=bg_freq)[0]

		pattern = aggregator._expand_seqlets_to_fill_pattern([pattern], 
			track_set=track_set, left_flank_to_add=flank_to_add,
			right_flank_to_add=flank_to_add)[0]

		if np.sign(np.sum(pattern["task0_contrib_scores"].fwd)) == track_sign:
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
		idx = seqlet.coor.example_idx
		start, end = seqlet.coor.start, seqlet.coor.end

		X_ohe.append(one_hot_sequence[idx][start:end])
		X_contrib.append(contrib_scores[idx][start:end])
		X_hypo_contrib.append(hypothetical_contribs[idx][start:end])

	return X_ohe, X_contrib, X_hypo_contrib

def TfModiscoSeqletsToPatternsFactory(seqlets, one_hot_sequence, contrib_scores, hypothetical_contribs, track_set, 
					   track_signs=None,
					   n_cores=10,
					   min_overlap_while_sliding=0.7,
					   nearest_neighbors_to_compute=500,

					   affmat_correlation_threshold=0.15,
					   tsne_perplexity=10.0,

					   n_leiden_iterations=-1,
					   n_leiden_runs=50,

					   frac_support_to_trim_to=0.2,
					   min_num_to_trim_to=30,
					   trim_to_window_size=20,
					   initial_flank_to_add=5,

					   prob_and_pertrack_sim_merge_thresholds=[
					   (0.8,0.8), (0.5, 0.85), (0.2, 0.9)],

					   prob_and_pertrack_sim_dealbreaker_thresholds=[
						(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],

					   subcluster_perplexity=50,
					   merging_max_seqlets_subsample=300,
					   final_min_cluster_size=20,
					   min_ic_in_window=0.6,
					   min_ic_windowsize=6,
					   ppm_pseudocount=0.001,

					   verbose=True, 
					   seed=1234):

		X_ohe, X_contrib, X_hypo_contrib = _extract_seqlet_data(one_hot_sequence, 
			contrib_scores, hypothetical_contribs, seqlets)

		bg_freq = np.mean(X_ohe, axis=(0,1))

		other_config={
				 'onehot_track_name': "sequence",
				 'contrib_scores_track_names': ["task0_contrib_scores"],
				 'hypothetical_contribs_track_names':["task0_hypothetical_contribs"],
				 'track_signs': [track_signs], 
				 'other_comparison_track_names': []
		}

		seqlets_sorter = (lambda arr: sorted(arr, key=lambda x:
			-np.sum(np.abs(x["task0_contrib_scores"].fwd))))

		seqlets = seqlets_sorter(seqlets)
		start = time.time()

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
				track_names=["task0_hypothetical_contribs", "task0_contrib_scores"],
				transformer=affinitymat.L1Normalizer(),
				min_overlap=min_overlap_while_sliding)

			if round_idx == 0:
				filtered_seqlets, seqlet_neighbors, filtered_affmat_nn = (
					_filter_by_correlation(seqlets, seqlet_neighbors, 
						coarse_affmat_nn, fine_affmat_nn, 
						affmat_correlation_threshold))
			else:
				filtered_seqlets = seqlets
				filtered_affmat_nn = fine_affmat_nn

			# Step 4: Density adaptation
			csr_density_adapted_affmat = _density_adaptation(filtered_affmat_nn, seqlet_neighbors, tsne_perplexity)


			# Step 5: Clustering
			cluster_results = cluster.LeidenCluster(
				csr_density_adapted_affmat,
				n_seeds=n_leiden_runs,
				n_leiden_iterations=n_leiden_iterations,
				verbose=verbose)

			#adwawadwa

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
			seqlets = list(dict([(y.exidx_start_end_string, y)
							 for x in motifs for y in x.seqlets]).values())


		merged_patterns, pattern_merge_hierarchy = aggregator._detect_spurious_merging(
			patterns=motifs, track_set=track_set, perplexity=subcluster_perplexity, 
			min_in_subcluster=max(final_min_cluster_size, subcluster_perplexity), 
			min_overlap=min_overlap_while_sliding,
			prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
			prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
    		min_frac=frac_support_to_trim_to,
				min_num=min_num_to_trim_to,
				flank_to_add=initial_flank_to_add,
				window_size=trim_to_window_size,
				bg_freq=bg_freq,
				verbose=verbose,
				max_seqlets_subsample=merging_max_seqlets_subsample,
				n_cores=n_cores)

		#Now start merging patterns 
		#merged_patterns, pattern_merge_hierarchy = similar_patterns_collapser(patterns=split_patterns) 
		merged_patterns = sorted(merged_patterns, key=lambda x: -x.num_seqlets)

		final_patterns, remaining_patterns = _filter_patterns(merged_patterns, 
			min_seqlet_support=final_min_cluster_size, 
			window_size=min_ic_windowsize, min_ic_in_window=min_ic_in_window, 
			background=bg_freq, ppm_pseudocount=ppm_pseudocount)

		total_time_taken = round(time.time()-start,2)

		#apply subclustering procedure on the final patterns
		for patternidx, pattern in enumerate(final_patterns):
			pattern.compute_subclusters_and_embedding(
				verbose=verbose, perplexity=subcluster_perplexity, n_jobs=n_cores)

		return {
			'success': True,
			'each_round_initcluster_motifs': None,             
			'patterns': final_patterns,
			'remaining_patterns': remaining_patterns,
			'seqlets': filtered_seqlets, 
			'cluster_results': cluster_results, 
			'total_time_taken': total_time_taken,
			'merged_patterns': merged_patterns,
			'pattern_merge_hierarchy': pattern_merge_hierarchy,
			'other_config': other_config
		}
