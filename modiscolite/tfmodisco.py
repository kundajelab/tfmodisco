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
		affmat_nn.indices, affmat_nn.indptr), shape=affmat_nn.shape, dtype='float64')

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
	passing_patterns = []
	for pattern in patterns:
		if len(pattern.seqlets) < min_seqlet_support:
			continue

		ppm = pattern.sequence
		per_position_ic = util.compute_per_position_ic(ppm=ppm, 
			background=background, pseudocount=ppm_pseudocount)

		if len(per_position_ic) < window_size:       
			if np.sum(per_position_ic) < min_ic_in_window:
				continue
		else:
			#do the sliding window sum rearrangement
			windowed_ic = np.sum(util.rolling_window(
				a=per_position_ic, window=window_size), axis=-1)

			if np.max(windowed_ic) < min_ic_in_window:
				continue

		passing_patterns.append(pattern)

	return passing_patterns


def _patterns_from_clusters(seqlets, track_set, min_overlap,
	min_frac, min_num, flank_to_add, window_size, bg_freq, cluster_indices, 
	track_sign):

	seqlet_sort_metric = lambda x: -np.sum(np.abs(x.contrib_scores))
	num_clusters = max(cluster_indices+1)
	cluster_to_seqlets = defaultdict(list) 

	for seqlet, idx in zip(seqlets, cluster_indices):
		cluster_to_seqlets[idx].append(seqlet)

	patterns = []
	for i in range(num_clusters):
		sorted_seqlets = sorted(cluster_to_seqlets[i], key=seqlet_sort_metric) 
		pattern = core.SeqletSet([sorted_seqlets[0]])

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
				patterns.append(pattern)

	return patterns


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


def seqlets_to_patterns(seqlets, track_set, track_signs=None, 
	min_overlap_while_sliding=0.7, nearest_neighbors_to_compute=500, 
	affmat_correlation_threshold=0.15, tsne_perplexity=10.0, 
	n_leiden_iterations=-1, n_leiden_runs=50, frac_support_to_trim_to=0.2,
	min_num_to_trim_to=30, trim_to_window_size=20, initial_flank_to_add=5,
	prob_and_pertrack_sim_merge_thresholds=[(0.8,0.8), (0.5, 0.85), (0.2, 0.9)],
	prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],
	subcluster_perplexity=50, merging_max_seqlets_subsample=300,
	final_min_cluster_size=20,min_ic_in_window=0.6, min_ic_windowsize=6,
	ppm_pseudocount=0.001):

	bg_freq = np.mean([seqlet.sequence for seqlet in seqlets], axis=(0, 1)) 

	seqlets_sorter = (lambda arr: sorted(arr, key=lambda x:
		-np.sum(np.abs(x.contrib_scores))))

	seqlets = seqlets_sorter(seqlets)

	for round_idx in range(2):
		if len(seqlets) == 0:
			return None

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
		cluster_indices = cluster.LeidenCluster(
			csr_density_adapted_affmat,
			n_seeds=n_leiden_runs,
			n_leiden_iterations=n_leiden_iterations)

		del csr_density_adapted_affmat

		patterns = _patterns_from_clusters(filtered_seqlets, 
			track_set=track_set, 
			min_overlap=min_overlap_while_sliding, 
			min_frac=frac_support_to_trim_to, 
			min_num=min_num_to_trim_to, 
			flank_to_add=initial_flank_to_add, 
			window_size=trim_to_window_size, 
			bg_freq=bg_freq, 
			cluster_indices=cluster_indices, 
			track_sign=track_signs)

		#obtain unique seqlets from adjusted motifs
		seqlets = list(dict([(y.string, y)
						 for x in patterns for y in x.seqlets]).values())

	del seqlets

	merged_patterns, pattern_merge_hierarchy = aggregator._detect_spurious_merging(
		patterns=patterns, track_set=track_set, perplexity=subcluster_perplexity, 
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
	merged_patterns = sorted(merged_patterns, key=lambda x: -len(x.seqlets))

	patterns = _filter_patterns(merged_patterns, 
		min_seqlet_support=final_min_cluster_size, 
		window_size=min_ic_windowsize, min_ic_in_window=min_ic_in_window, 
		background=bg_freq, ppm_pseudocount=ppm_pseudocount)

	#apply subclustering procedure on the final patterns
	for patternidx, pattern in enumerate(patterns):
		pattern.compute_subpatterns(subcluster_perplexity, 
			n_seeds=n_leiden_runs, n_iterations=n_leiden_iterations)

	return patterns


def TFMoDISco(one_hot, hypothetical_contribs, sliding_window_size=21, 
	flank_size=10, min_metacluster_size=100,
	weak_threshold_for_counting_sign=0.8, max_seqlets_per_metacluster=20000,
	target_seqlet_fdr=0.2, min_passing_windows_frac=0.03,
	max_passing_windows_frac=0.2, n_leiden_runs=50, n_leiden_iterations=-1, 
	min_overlap_while_sliding=0.7, nearest_neighbors_to_compute=500, 
	affmat_correlation_threshold=0.15, tsne_perplexity=10.0, 
	frac_support_to_trim_to=0.2, min_num_to_trim_to=30, trim_to_window_size=20, 
	initial_flank_to_add=5,
	prob_and_pertrack_sim_merge_thresholds=[(0.8,0.8), (0.5, 0.85), (0.2, 0.9)],
	prob_and_pertrack_sim_dealbreaker_thresholds=[(0.4, 0.75), (0.2,0.8), (0.1, 0.85), (0.0,0.9)],
	subcluster_perplexity=50, merging_max_seqlets_subsample=300,
	final_min_cluster_size=20, min_ic_in_window=0.6, min_ic_windowsize=6,
	ppm_pseudocount=0.001, verbose=False):

	contrib_scores = np.multiply(one_hot, hypothetical_contribs)

	track_set = core.TrackSet(one_hot=one_hot, 
		contrib_scores=contrib_scores,
		hypothetical_contribs=hypothetical_contribs)

	seqlet_coords, threshold = extract_seqlets.extract_seqlets(
		attribution_scores=contrib_scores.sum(axis=2),
		window_size=sliding_window_size,
		flank=flank_size,
		suppress=(int(0.5*sliding_window_size) + flank_size),
		target_fdr=target_seqlet_fdr,
		min_passing_windows_frac=min_passing_windows_frac,
		max_passing_windows_frac=max_passing_windows_frac,
		weak_threshold_for_counting_sign=weak_threshold_for_counting_sign) 

	seqlets = track_set.create_seqlets(seqlet_coords) 

	pos_seqlets, neg_seqlets = [], []
	for seqlet in seqlets:
		flank = int(0.5*(len(seqlet)-sliding_window_size))
		attr = np.sum(seqlet.contrib_scores[flank:-flank])

		if attr > threshold:
			pos_seqlets.append(seqlet)
		elif attr < -threshold:
			neg_seqlets.append(seqlet)

	del seqlets

	if len(pos_seqlets) > min_metacluster_size:
		pos_seqlets = pos_seqlets[:max_seqlets_per_metacluster]
		if verbose:
			print("Using {} positive seqlets".format(len(pos_seqlets)))

		pos_patterns = seqlets_to_patterns(seqlets=pos_seqlets,
			track_set=track_set, 
			track_signs=1,
			min_overlap_while_sliding=min_overlap_while_sliding,
			nearest_neighbors_to_compute=nearest_neighbors_to_compute,
			affmat_correlation_threshold=affmat_correlation_threshold,
			tsne_perplexity=tsne_perplexity,
			n_leiden_iterations=n_leiden_iterations,
			n_leiden_runs=n_leiden_runs,
			frac_support_to_trim_to=frac_support_to_trim_to,
			min_num_to_trim_to=min_num_to_trim_to,
			trim_to_window_size=trim_to_window_size,
			initial_flank_to_add=initial_flank_to_add,
			prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
			prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
			subcluster_perplexity=subcluster_perplexity,
			merging_max_seqlets_subsample=merging_max_seqlets_subsample,
			final_min_cluster_size=final_min_cluster_size,
			min_ic_in_window=min_ic_in_window,
			min_ic_windowsize=min_ic_windowsize,
			ppm_pseudocount=ppm_pseudocount)
	else:
		pos_patterns = None

	if len(neg_seqlets) > min_metacluster_size:
		neg_seqlets = neg_seqlets[:max_seqlets_per_metacluster]
		if verbose:
			print("Extracted {} negative seqlets".format(len(neg_seqlets)))

		neg_patterns = seqlets_to_patterns(seqlets=neg_seqlets,
			track_set=track_set, 
			track_signs=-1,
			min_overlap_while_sliding=min_overlap_while_sliding,
			nearest_neighbors_to_compute=nearest_neighbors_to_compute,
			affmat_correlation_threshold=affmat_correlation_threshold,
			tsne_perplexity=tsne_perplexity,
			n_leiden_iterations=n_leiden_iterations,
			n_leiden_runs=n_leiden_runs,
			frac_support_to_trim_to=frac_support_to_trim_to,
			min_num_to_trim_to=min_num_to_trim_to,
			trim_to_window_size=trim_to_window_size,
			initial_flank_to_add=initial_flank_to_add,
			prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
			prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
			subcluster_perplexity=subcluster_perplexity,
			merging_max_seqlets_subsample=merging_max_seqlets_subsample,
			final_min_cluster_size=final_min_cluster_size,
			min_ic_in_window=min_ic_in_window,
			min_ic_windowsize=min_ic_windowsize,
			ppm_pseudocount=ppm_pseudocount)
	else:
		neg_patterns = None

	return pos_patterns, neg_patterns
