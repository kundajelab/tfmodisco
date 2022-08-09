from __future__ import division, absolute_import, print_function
from .. import affinitymat
from .. import nearest_neighbors
from .. import cluster
from .. import aggregator
from .. import core
from .. import util
from .. import seqlet_embedding
from .. import pattern_filterer as pattern_filterer_module
from joblib import Parallel, delayed
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import time
import sys
import gc
import json
from ..util import print_memory_use


def do_density_adaptation(new_rows_distmat_nn, new_rows_nn,
								new_rows_betas, new_rows_normfactors):
	new_rows_densadapted_affmat_nn = []
	for i in range(len(new_rows_distmat_nn)):
		densadapted_row = []

		for j,distance in zip(new_rows_nn[i], new_rows_distmat_nn[i]):
			densadapted_row.append(np.sqrt(
			  (np.exp(-distance/new_rows_betas[i])/new_rows_normfactors[i])
			 *(np.exp(-distance/new_rows_betas[j])/
			   new_rows_normfactors[j]))) 
		new_rows_densadapted_affmat_nn.append(densadapted_row)

	return new_rows_densadapted_affmat_nn

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

def TfModiscoSeqletsToPatternsFactory(seqlets, track_set, 
					   onehot_track_name,
					   contrib_scores_track_names,
					   hypothetical_contribs_track_names,
					   track_signs,
					   other_comparison_track_names=[],
					   n_cores=10,
					   min_overlap_while_sliding=0.7,
					   embedder_factory=(
						seqlet_embedding.advanced_gapped_kmer
										.AdvancedGappedKmerEmbedderFactory()),

					   nearest_neighbors_to_compute=500,

					   affmat_correlation_threshold=0.15,

					   tsne_perplexity=10,

					   n_leiden_iterations_r1=-1,
					   n_leiden_iterations_r2=-1,
					   contin_runs_r1=50,
					   contin_runs_r2=50,

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

		bg_freq = np.mean(
			track_set.track_name_to_data_track[onehot_track_name].fwd_tracks,
			axis=(0,1))

		seqlets_sorter = (lambda arr:
						  sorted(arr,
								 key=lambda x:
								  -np.sum([np.sum(np.abs(x[track_name].fwd))
									 for track_name
									 in contrib_scores_track_names])))

		pattern_comparison_settings =\
			affinitymat.core.PatternComparisonSettings(
				track_names=hypothetical_contribs_track_names
							+contrib_scores_track_names
							+other_comparison_track_names, 
				track_transformer=affinitymat.L1Normalizer(), 
				min_overlap=min_overlap_while_sliding)

		#coarse_grained 1d embedder
		seqlets_to_1d_embedder = embedder_factory(
				onehot_track_name=onehot_track_name,
				toscore_track_names_and_signs=list(
				zip(hypothetical_contribs_track_names,
					[np.sign(x) for x in track_signs])),
				n_jobs=n_cores)

		#affinity matrix from embeddings
		sparse_affmat_from_fwdnrev1dvecs =\
			affinitymat.core.SparseNumpyCosineSimFromFwdAndRevOneDVecs(
					n_neighbors=nearest_neighbors_to_compute, 
					verbose=verbose)

		coarse_affmat_computer =\
		  affinitymat.core.SparseAffmatFromFwdAndRevSeqletEmbeddings(
			seqlets_to_1d_embedder=seqlets_to_1d_embedder,
			sparse_affmat_from_fwdnrev1dvecs=sparse_affmat_from_fwdnrev1dvecs,
			verbose=verbose)

		affmat_from_seqlets_with_nn_pairs =\
			affinitymat.core.AffmatFromSeqletsWithNNpairs(
				pattern_comparison_settings=pattern_comparison_settings,
				sim_metric_on_nn_pairs=\
					affinitymat.core.ParallelCpuCrossMetricOnNNpairs(
						n_cores=n_cores,
						cross_metric_single_region=
							affinitymat.core.CrossContinJaccardSingleRegion()))

		filter_mask_from_correlation =\
			affinitymat.core.FilterMaskFromCorrelation(
				correlation_threshold=affmat_correlation_threshold,
				verbose=verbose)

		aff_to_dist_mat = affinitymat.transformers.AffToDistViaInvLogistic() 


		#prepare the clusterers for the different rounds
		# No longer a need for symmetrization because am symmetrizing by
		# taking the geometric mean elsewhere
		affmat_transformer_r1 =\
			affinitymat.transformers.AdhocAffMatTransformer(lambda x: x)

		print("TfModiscoSeqletsToPatternsFactory: seed=%d" % seed)
		clusterer_r1 = cluster.core.LeidenClusterParallel(
			n_jobs=n_cores, 
			affmat_transformer=affmat_transformer_r1,
			numseedstotry=contin_runs_r1,
			n_leiden_iterations=n_leiden_iterations_r1,
			verbose=verbose)

		#No longer a need for symmetrization because am symmetrizing by
		# taking the geometric mean elsewhere
		affmat_transformer_r2 =\
			affinitymat.transformers.AdhocAffMatTransformer(lambda x: x)

		clusterer_r2 = cluster.core.LeidenClusterParallel(
			n_jobs=n_cores, 
			affmat_transformer=affmat_transformer_r2,
			numseedstotry=contin_runs_r2,
			n_leiden_iterations=n_leiden_iterations_r2,
			verbose=verbose)
		
		clusterer_per_round = [clusterer_r1, clusterer_r2]

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
			pattern_aligner=core.CrossContinJaccardPatternAligner(
				pattern_comparison_settings=pattern_comparison_settings),
				seqlet_sort_metric=
					lambda x: -sum([np.sum(np.abs(x[track_name].fwd)) for
							   track_name in contrib_scores_track_names]),
			track_set=track_set, #needed for seqlet expansion
			postprocessor=postprocessor1)

		#prepare the similar patterns collapser
		pattern_to_seqlet_sim_computer =\
			affinitymat.core.AffmatFromSeqletsWithNNpairs(
				pattern_comparison_settings=pattern_comparison_settings,
				sim_metric_on_nn_pairs=\
					affinitymat.core.ParallelCpuCrossMetricOnNNpairs(
						n_cores=n_cores,
						cross_metric_single_region=\
							affinitymat.core.CrossContinJaccardSingleRegion(),
						verbose=False))

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
							track_names=(
								contrib_scores_track_names+
								other_comparison_track_names), 
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
				 'other_comparison_track_names': other_comparison_track_names
		}

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


		for round_idx, clusterer in enumerate(clusterer_per_round):
			import gc
			gc.collect()
			
			if len(seqlets) == 0:
				return failure

			coarse_affmat_nn, seqlet_neighbors = coarse_affmat_computer(seqlets, initclusters=None)
			gc.collect()

			nn_affmat_start = time.time() 

			fine_affmat_nn = affmat_from_seqlets_with_nn_pairs(
								seqlet_neighbors=seqlet_neighbors,
								seqlets=seqlets,
								return_sparse=True)

			#get the fine_affmat_nn reorderings
			reorderings = np.array([np.argsort(-finesimsinrow)
									for finesimsinrow in fine_affmat_nn])

			#reorder fine_affmat_nn, coarse_affmat_nn and seqlet_neighbors
			# according to reorderings
			fine_affmat_nn = [finesimsinrow[rowreordering]
								  for (finesimsinrow, rowreordering)
								  in zip(fine_affmat_nn, reorderings)]
			coarse_affmat_nn = [coarsesimsinrow[rowreordering]
								  for (coarsesimsinrow, rowreordering)
								  in zip(coarse_affmat_nn, reorderings)]
			seqlet_neighbors = [nnrow[rowreordering]
								  for (nnrow, rowreordering)
								  in zip(seqlet_neighbors, reorderings)]

			del reorderings
			gc.collect()

			#filter by correlation
			if round_idx == 0:
				#the filter_mask_from_correlation function only operates
				# on columns in which np.abs(main_affmat) > 0
				filtered_rows_mask = filter_mask_from_correlation(
										main_affmat=fine_affmat_nn,
										other_affmat=coarse_affmat_nn)
			else:
				filtered_rows_mask = np.array([True for x in seqlets])


			del coarse_affmat_nn 
			gc.collect()

			filtered_seqlets = [x[0] for x in zip(seqlets, filtered_rows_mask) if (x[1])]

			#figure out a mapping from pre-filtering to the
			# post-filtering indices
			new_idx_mapping = (
				np.cumsum(1.0*(filtered_rows_mask)).astype("int")-1)
			retained_indices = set(np.arange(len(filtered_rows_mask))[
											  filtered_rows_mask])
			del filtered_rows_mask
			filtered_neighbors = []
			filtered_affmat_nn = []
			for old_row_idx, (old_neighbors,affmat_row) in enumerate(
								zip(seqlet_neighbors, fine_affmat_nn)): 
				if old_row_idx in retained_indices:
					filtered_old_neighbors = [
						neighbor for neighbor in old_neighbors if neighbor
						in retained_indices]
					filtered_affmat_row = [
						affmatval for affmatval,neighbor
						in zip(affmat_row,old_neighbors)
						if neighbor in retained_indices]
					filtered_neighbors_row = [
						new_idx_mapping[neighbor] for neighbor
						in filtered_old_neighbors]
					filtered_neighbors.append(filtered_neighbors_row)
					filtered_affmat_nn.append(filtered_affmat_row)

			#overwrite seqlet_neighbors...should be ok if the rows are
			# not all the same length
			seqlet_neighbors = filtered_neighbors
			del (filtered_neighbors, retained_indices, new_idx_mapping)

			#apply aff_to_dist_mat one row at a time
			distmat_nn = [aff_to_dist_mat(affinity_mat=x)
						  for x in filtered_affmat_nn] 
			del filtered_affmat_nn

			#Note: the fine-grained similarity metric isn't actually symmetric
			# because a different input will get padded with zeros depending
			# on which seqlets are specified as the filters and which seqlets
			# are specified as the 'thing to scan'. So explicit symmetrization
			# is worthwhile
			sym_seqlet_neighbors, sym_distmat_nn = util.symmetrize_nn_distmat(
				distmat_nn=distmat_nn, nn=seqlet_neighbors)
			del distmat_nn
			del seqlet_neighbors

			#Compute beta values for the density adaptation. *store it*
			betas_and_ps = Parallel(n_jobs=n_cores)(
					 delayed(util.binary_search_perplexity)(
						  tsne_perplexity, distances)
					 for distances in sym_distmat_nn)
			betas = np.array([x[0] for x in betas_and_ps])

			#also compute the normalization factor needed to get probs to sum to 1
			#note: sticking to lists here because different rows of
			# sym_distmat_nn may have different lengths after adding in
			# the symmetric pairs
			densadapted_affmat_nn_unnorm = [np.exp(-np.array(distmat_row)/beta)
				for distmat_row, beta in zip(sym_distmat_nn, betas)]
			normfactors = np.array([max(np.sum(x),1e-8) for x in densadapted_affmat_nn_unnorm])

			sym_densadapted_affmat_nn = do_density_adaptation(
				new_rows_distmat_nn=sym_distmat_nn,
				new_rows_nn=sym_seqlet_neighbors,
				new_rows_betas=betas,
				new_rows_normfactors=normfactors)

			#util.verify_symmetric_nn_affmat(
			#	affmat_nn=sym_densadapted_affmat_nn,
			#	nn=sym_seqlet_neighbors)

			#Make csr matrix
			csr_density_adapted_affmat =\
				util.coo_matrix_from_neighborsformat(
					entries=sym_densadapted_affmat_nn,
					neighbors=sym_seqlet_neighbors,
					ncols=len(sym_densadapted_affmat_nn)).tocsr()

			cluster_results = clusterer(csr_density_adapted_affmat, initclusters=None)
			del csr_density_adapted_affmat

			cluster_to_motif, cluster_to_eliminated_motif =\
				get_cluster_to_aggregate_motif(
					seqlets=filtered_seqlets,
					seqlet_aggregator=seqlet_aggregator,
					cluster_indices=cluster_results.cluster_indices,
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
