from __future__ import division, absolute_import, print_function
from .. import affinitymat
from .. import aggregator
from .. import core
from .. import util
from .. import gapped_kmer
from .. import pattern_filterer as pattern_filterer_module
from .. import cluster
from joblib import Parallel, delayed
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import time
import sys
import gc
import json
from ..util import print_memory_use



def _density_adaptation(affmat_nn, seqlet_neighbors, tsne_perplexity, n_jobs):
	eps = 0.0000001
	distmat_nn = [np.maximum(np.log((1.0/(0.5*np.maximum(x, eps)))-1), 
		0.0) for x in affmat_nn] 

	#Note: the fine-grained similarity metric isn't actually symmetric
	# because a different input will get padded with zeros depending
	# on which seqlets are specified as the filters and which seqlets
	# are specified as the 'thing to scan'. So explicit symmetrization
	# is worthwhile
	seqlet_neighbors, distmat_nn = util.symmetrize_nn_distmat(
		distmat_nn=distmat_nn, nn=seqlet_neighbors)

	#Compute beta values for the density adaptation. *store it*
	betas_and_ps = Parallel(n_jobs=n_jobs)(delayed(
		util.binary_search_perplexity)(tsne_perplexity, distances) 
			for distances in distmat_nn)
	betas = np.array([x[0] for x in betas_and_ps])

	#also compute the normalization factor needed to get probs to sum to 1
	#note: sticking to lists here because different rows of
	# sym_distmat_nn may have different lengths after adding in
	# the symmetric pairs
	densadapted_affmat_nn_unnorm = [np.exp(-np.array(distmat_row)/beta)
		for distmat_row, beta in zip(distmat_nn, betas)]
	normfactors = np.array([max(np.sum(x), 1e-8) 
		for x in densadapted_affmat_nn_unnorm])

	densadapted_affmat_nn = []
	for i in range(len(distmat_nn)):
		densadapted_row = []

		betas_i = betas[i]
		norms_i = normfactors[i]

		for j, distance in zip(seqlet_neighbors[i], distmat_nn[i]):
			betas_j = betas[j]
			norms_j = normfactors[j]
 
			rbf_i = np.exp(-distance / betas_i) / norms_i
			rbf_j = np.exp(-distance / betas_j) / norms_j

			density_adapted_row = np.sqrt(rbf_i * rbf_j)
			densadapted_row.append(density_adapted_row)

		densadapted_affmat_nn.append(densadapted_row)

	csr_density_adapted_affmat = util.coo_matrix_from_neighborsformat(
		entries=densadapted_affmat_nn, neighbors=seqlet_neighbors,
		ncols=len(densadapted_affmat_nn)).tocsr()

	return csr_density_adapted_affmat

def get_cluster_to_aggregate_motif(seqlets, seqlet_aggregator,
								   cluster_indices,
								   sign_consistency_check,
								   min_seqlets_in_motif,
								   contrib_scores_track_names,
								   track_signs):
	num_clusters = max(cluster_indices+1)
	cluster_to_seqlets = defaultdict(list) 

	for seqlet,idx in zip(seqlets, cluster_indices):
		cluster_to_seqlets[idx].append(seqlet)

	cluster_to_motif = OrderedDict()
	cluster_to_eliminated_motif = OrderedDict()
	for i in range(num_clusters):
		if len(cluster_to_seqlets[i]) >= min_seqlets_in_motif:
			motifs = seqlet_aggregator(cluster_to_seqlets[i])

			if len(motifs) > 0:
				motif = motifs[0]
				if (sign_consistency_check==False or sign_consistency_func(motif, contrib_scores_track_names, track_signs)):
					cluster_to_motif[i] = motif
				else:
					cluster_to_eliminated_motif[i] = motif

	return cluster_to_motif, cluster_to_eliminated_motif

def sign_consistency_func(motif, contrib_scores_track_names, track_signs):
	motif_track_signs = [
		np.sign(np.sum(motif[contrib_scores_track_name].fwd)) for
		contrib_scores_track_name in contrib_scores_track_names]
	return all([(x==y) for x,y in zip(motif_track_signs, track_signs)])

def _filter_by_correlation(seqlets, seqlet_neighbors, coarse_affmat_nn, 
	fine_affmat_nn, correlation_threshold):

	filtered_rows_mask = affinitymat.core.FilterMaskFromCorrelation(
		main_affmat=fine_affmat_nn, other_affmat=coarse_affmat_nn,
		correlation_threshold=correlation_threshold)

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


def TfModiscoSeqletsToPatternsFactory(seqlets, track_set, 
					   onehot_track_name="sequence",
					   contrib_scores_track_names=["task0_contrib_scores"],
					   hypothetical_contribs_track_names=["task0_hypothetical_contribs"],
					   track_signs=None,
					   n_cores=10,
					   min_overlap_while_sliding=0.7,
					   nearest_neighbors_to_compute=500,

					   affmat_correlation_threshold=0.15,
					   tsne_perplexity=10,

					   n_leiden_iterations=-1,
					   contin_runs=50,

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

					   final_flank_to_add=0,

					   verbose=True, 
					   seed=1234):

		bg_freq = np.mean(track_set.track_name_to_data_track[
			onehot_track_name].fwd_tracks, axis=(0,1))

		pattern_comparison_settings =\
			affinitymat.core.PatternComparisonSettings(
				track_names=hypothetical_contribs_track_names
							+contrib_scores_track_names,
				track_transformer=affinitymat.L1Normalizer(), 
				min_overlap=min_overlap_while_sliding)

		print("TfModiscoSeqletsToPatternsFactory: seed=%d" % seed)

		#prepare the seqlet aggregator
		expand_trim_expand1 =\
			aggregator.ExpandSeqletsToFillPattern(
				track_set=track_set,
				flank_to_add=initial_flank_to_add).chain(
			aggregator.TrimToBestWindowByIC(
				window_size=trim_to_window_size,
				onehot_track_name=onehot_track_name,
				bg_freq=bg_freq)).chain(
			aggregator.ExpandSeqletsToFillPattern(
				track_set=track_set,
				flank_to_add=initial_flank_to_add))

		postprocessor1 =\
			aggregator.TrimToFracSupport(
						min_frac=frac_support_to_trim_to,
						min_num=min_num_to_trim_to,
						verbose=verbose)\
					  .chain(expand_trim_expand1)

		seqlet_aggregator = aggregator.GreedySeqletAggregator(
			pattern_comparison_settings=pattern_comparison_settings,
			seqlet_sort_metric=lambda x: -np.sum(np.abs(x["task0_contrib_scores"].fwd)),
			track_set=track_set, #needed for seqlet expansion
			postprocessor=postprocessor1)


		#similarity settings for merging
		prob_and_sim_merge_thresholds =\
			prob_and_pertrack_sim_merge_thresholds
		prob_and_sim_dealbreaker_thresholds =\
			prob_and_pertrack_sim_dealbreaker_thresholds

		similar_patterns_collapser =\
			aggregator.DynamicDistanceSimilarPatternsCollapser2(
				pattern_comparison_settings=pattern_comparison_settings,
				track_set=track_set,
				pattern_aligner=core.CrossCorrelationPatternAligner(
					pattern_comparison_settings=
						affinitymat.core.PatternComparisonSettings(
							track_names=contrib_scores_track_names, 
							track_transformer=
								affinitymat.MeanNormalizer().chain(
								affinitymat.MagnitudeNormalizer()), 
							min_overlap=min_overlap_while_sliding)),
				collapse_condition=(lambda prob, aligner_sim:
					any([(prob >= x[0] and aligner_sim >= x[1])
						 for x in prob_and_sim_merge_thresholds])),
				dealbreaker_condition=(lambda prob, aligner_sim:
					any([(prob <= x[0] and aligner_sim <= x[1])              
						 for x in prob_and_sim_dealbreaker_thresholds])),
				postprocessor=postprocessor1,
				verbose=verbose,
				max_seqlets_subsample=merging_max_seqlets_subsample,
				n_cores=n_cores)

		pattern_filterer = pattern_filterer_module.MinSeqletSupportFilterer(
			min_seqlet_support=final_min_cluster_size).chain(
				pattern_filterer_module.MinICinWindow(
					window_size=min_ic_windowsize,
					min_ic_in_window=min_ic_in_window,
					background=bg_freq,
					sequence_track_name=onehot_track_name,
					ppm_pseudocount=ppm_pseudocount   
				) 
			)

		final_postprocessor = aggregator.ExpandSeqletsToFillPattern(
										track_set=track_set,
										flank_to_add=final_flank_to_add) 

		other_config={
				 'onehot_track_name': onehot_track_name,
				 'contrib_scores_track_names': contrib_scores_track_names,
				 'hypothetical_contribs_track_names':
					hypothetical_contribs_track_names,
				 'track_signs': track_signs, 
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

			# Step 1: Generate coarse representation
			embedding_fwd, embedding_rev = gapped_kmer.AdvancedGappedKmerEmbedder(
				seqlets=seqlets, sign=track_signs[0], n_jobs=n_cores)

			coarse_affmat_nn, seqlet_neighbors = affinitymat.core.sparse_cosine_similarity(
				fwd_vecs=embedding_fwd, rev_vecs=embedding_rev,
				n_neighbors=nearest_neighbors_to_compute)

			# Step 2: Generate fine representation
			fine_affmat_nn = affinitymat.core.AffmatFromSeqletsWithNNpairs(
				seqlets=seqlets, seqlet_neighbors=seqlet_neighbors,
				track_names=hypothetical_contribs_track_names
							+contrib_scores_track_names,
				transformer=affinitymat.L1Normalizer(), 
				min_overlap=min_overlap_while_sliding,
				return_sparse=True, n_cores=n_cores)

			# Step 3: Filter out to top NN based on fine representation 
			reorderings = np.argsort(-fine_affmat_nn, axis=-1)

			fine_affmat_nn = np.array([finesimsinrow[rowreordering]
				for (finesimsinrow, rowreordering) in zip(fine_affmat_nn, reorderings)])
			coarse_affmat_nn = np.array([coarsesimsinrow[rowreordering]
				for (coarsesimsinrow, rowreordering) in zip(coarse_affmat_nn, reorderings)])
			seqlet_neighbors = np.array([nnrow[rowreordering]
				for (nnrow, rowreordering) in zip(seqlet_neighbors, reorderings)])
			
			if round_idx == 0:
				filtered_seqlets, seqlet_neighbors, filtered_affmat_nn = (
					_filter_by_correlation(seqlets, seqlet_neighbors, 
						coarse_affmat_nn, fine_affmat_nn, 
						affmat_correlation_threshold))
			else:
				filtered_seqlets = seqlets
				filtered_affmat_nn = fine_affmat_nn

			# Step 4: Density adaptation
			csr_density_adapted_affmat = _density_adaptation(filtered_affmat_nn, seqlet_neighbors, tsne_perplexity, n_cores)

			# Step 5: Clustering
			cluster_results = cluster.LeidenCluster(
				csr_density_adapted_affmat,
				initclusters=None,
				n_jobs=n_cores, 
				affmat_transformer=None,
				numseedstotry=contin_runs,
				n_leiden_iterations=n_leiden_iterations,
				verbose=verbose)

			del csr_density_adapted_affmat

			cluster_to_motif, cluster_to_eliminated_motif =\
				get_cluster_to_aggregate_motif(
					seqlets=filtered_seqlets,
					seqlet_aggregator=seqlet_aggregator,
					cluster_indices=cluster_results['cluster_indices'],
					sign_consistency_check=True,
					min_seqlets_in_motif=0,
					contrib_scores_track_names=contrib_scores_track_names,
					track_signs=track_signs)

			#obtain unique seqlets from adjusted motifs
			seqlets = list(dict([(y.exidx_start_end_string, y)
							 for x in cluster_to_motif.values()
							 for y in x.seqlets]).values())


		subcluster_settings = {
			"pattern_comparison_settings": pattern_comparison_settings,
			"perplexity": subcluster_perplexity,
			"n_jobs": n_cores,
		}

		spurious_merge_detector = aggregator.DetectSpuriousMerging2(
			subcluster_settings=subcluster_settings, verbose=verbose,
			min_in_subcluster=max(final_min_cluster_size, subcluster_perplexity),
			similar_patterns_collapser=similar_patterns_collapser)

		split_patterns = spurious_merge_detector(cluster_to_motif.values())

		if len(split_patterns) == 0:
			return failure

		#Now start merging patterns 
		merged_patterns, pattern_merge_hierarchy = similar_patterns_collapser(patterns=split_patterns) 
		merged_patterns = sorted(merged_patterns, key=lambda x: -x.num_seqlets)

		final_patterns, remaining_patterns = pattern_filterer(merged_patterns)
		final_patterns = final_postprocessor(final_patterns)
		remaining_patterns = final_postprocessor(remaining_patterns)

		total_time_taken = round(time.time()-start,2)

		#apply subclustering procedure on the final patterns
		for patternidx, pattern in enumerate(final_patterns):
			pattern.compute_subclusters_and_embedding(
				verbose=verbose, **subcluster_settings)

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
