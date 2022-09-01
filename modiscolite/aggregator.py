import numpy as np
from . import affinitymat
from . import core
from . import util
from collections import OrderedDict, defaultdict
import itertools
import sys
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import time

def _trim_to_frac_support(aggregated_seqlets, min_frac, min_num):
	return [x.trim_to_positions_with_min_support(min_frac=min_frac,
		min_num=min_num, verbose=True) for x in aggregated_seqlets]

def _trim_to_best_window_by_ic(aggregated_seqlets, window_size, bg_freq):
	trimmed_agg_seqlets = []
	for aggregated_seqlet in aggregated_seqlets:
		ppm = aggregated_seqlet["sequence"].fwd
		per_pos_ic = util.compute_per_position_ic(
			ppm=ppm, background=bg_freq, pseudocount=0.001)


		start_idx = np.argmax(util.cpu_sliding_window_sum(
			arr=per_pos_ic, window_size=window_size))

		end_idx = start_idx + window_size
		trimmed_agg_seqlets.append(
			aggregated_seqlet.trim_to_start_and_end_idx(
				start_idx=start_idx, end_idx=end_idx))

	return trimmed_agg_seqlets

def _expand_seqlets_to_fill_pattern(patterns, track_set, left_flank_to_add,
	right_flank_to_add):
	patterns_ = []
	for pattern in patterns:
		new_seqlets_and_alnmts = []
		for seqlet_and_alnmt in pattern._seqlets_and_alnmts:
			seqlet = seqlet_and_alnmt.seqlet
			alnmt = seqlet_and_alnmt.alnmt
			left_expansion = alnmt + left_flank_to_add 
			right_expansion = ((len(pattern) - 
							   (alnmt+len(seqlet))) + right_flank_to_add)

			if seqlet.coor.is_revcomp == False:
				start = seqlet.coor.start - left_expansion
				end = seqlet.coor.end + right_expansion
			else:
				start = seqlet.coor.start - right_expansion
				end = seqlet.coor.end + left_expansion
			
			if start >= 0 and end <= track_set.length:
				seqlet = track_set.create_seqlets(
					coords=[core.SeqletCoordinates(
						example_idx=seqlet.coor.example_idx,
						start=start, end=end,
						is_revcomp=seqlet.coor.is_revcomp)])[0]

				new_seqlets_and_alnmts.append(
				 core.SeqletAndAlignment(seqlet=seqlet, alnmt=0))

		if len(new_seqlets_and_alnmts) > 0:
			pattern_ = core.AggregatedSeqlet(seqlets_and_alnmts_arr=new_seqlets_and_alnmts)
			patterns_.append(pattern_)

	return patterns_


def _align_patterns(parent_pattern, child_pattern, metric, min_overlap, 
	track_names, track_transformer):

	fwd_data_parent, rev_data_parent = core.get_2d_data_from_patterns(
		[parent_pattern], track_names=track_names, 
		track_transformer=track_transformer)

	fwd_data_child, rev_data_child = core.get_2d_data_from_patterns(
		[child_pattern], track_names=track_names,
		track_transformer=track_transformer)

	best_crossmetric, best_crossmetric_argmax = metric(fwd_data_child, 
		fwd_data_parent, min_overlap).squeeze()

	best_crossmetric_rev, best_crossmetric_argmax_rev = metric(rev_data_child, 
		fwd_data_parent, min_overlap).squeeze()

	if best_crossmetric_rev > best_crossmetric:
		return int(best_crossmetric_argmax_rev), True, best_crossmetric_rev
	else:
		return int(best_crossmetric_argmax), False, best_crossmetric


def merge_in_seqlets_filledges(parent_pattern, seqlets_to_merge,
	track_set, track_names, metric, min_overlap, track_transformer, 
	verbose=True):

	parent_pattern = parent_pattern.copy()
	for seqlet in seqlets_to_merge:
		alnmt, revcomp_match, alnmt_score =\
			_align_patterns(parent_pattern, seqlet, metric,
				min_overlap, track_names, track_transformer)
		
		if revcomp_match:
			seqlet = seqlet.revcomp()

		preexpansion_seqletlen = len(seqlet)
		#extend seqlet according to the alignment so that it fills the
		# whole pattern
		left_expansion = max(alnmt,0)
		right_expansion = max((len(parent_pattern) - (alnmt+len(seqlet))), 0)

		if seqlet.coor.is_revcomp == False:
			start = seqlet.coor.start - left_expansion
			end = seqlet.coor.end + right_expansion
		else:
			start = seqlet.coor.start - right_expansion
			end = seqlet.coor.end + left_expansion

		example_end = track_set.length

		if start >= 0 and end <= example_end:
			seqlet = track_set.create_seqlets(
				coords=[core.SeqletCoordinates(
					example_idx=seqlet.coor.example_idx,
					start=start, end=end,
					is_revcomp=seqlet.coor.is_revcomp)])[0] 
		else:
			continue #don't try adding this seqlet

		#also expand the pattern (if needed) so that the seqlet
		# doesn't go over the edge
		parent_left_expansion = max(0, -alnmt)
		parent_right_expansion = max(0, (alnmt+preexpansion_seqletlen)
										 - len(parent_pattern))

		if (parent_left_expansion > 0) or (parent_right_expansion > 0):
			candidate_parent_pattern = _expand_seqlets_to_fill_pattern(
				patterns=[parent_pattern],
				track_set=track_set, left_flank_to_add=parent_left_expansion,
				right_flank_to_add=parent_right_expansion)

			if len(candidate_parent_pattern) > 0:
				parent_pattern = candidate_parent_pattern[0]
			else: #the flank expansion required to merge in this seqlet got
				# rid of all the other seqlets in the pattern, so we won't use
				# this seqlet
				continue

		#add the seqlet in at alignment 0, assuming it's not already
		# part of the pattern
		if seqlet not in parent_pattern.seqlets_and_alnmts:
			parent_pattern._add_pattern_with_valid_alnmt(
							pattern=seqlet, alnmt=0)

	return parent_pattern


class PatternMergeHierarchy(object):
	def __init__(self, root_nodes):
		self.root_nodes = root_nodes

	def add_level(self, level_arr):
		self.levels.append(level_arr)

	def save_hdf5(self, grp):
		root_node_names = []
		for i in range(len(self.root_nodes)):
			node_name = "root_node"+str(i)
			root_node_names.append(node_name) 
			self.root_nodes[i].save_hdf5(grp.create_group(node_name))
		util.save_string_list(root_node_names,
							  dset_name="root_node_names",
							  grp=grp) 


class PatternMergeHierarchyNode(object):

	def __init__(self, pattern, child_nodes=None, parent_node=None,
					   indices_merged=None, submat_crosscontam=None,
					   submat_alignersim=None): 
		self.pattern = pattern 
		if (child_nodes is None):
			child_nodes = []
		self.child_nodes = child_nodes
		self.parent_node = parent_node
		self.indices_merged = indices_merged
		self.submat_crosscontam = submat_crosscontam
		self.submat_alignersim = submat_alignersim

	def save_hdf5(self, grp):
		if (self.indices_merged is not None):
			grp.create_dataset("indices_merged",
							   data=np.array(self.indices_merged)) 
			grp.create_dataset("submat_crosscontam",
							   data=np.array(self.submat_crosscontam)) 
			grp.create_dataset("submat_alignersim",
							   data=np.array(self.submat_alignersim)) 
		self.pattern.save_hdf5(grp=grp.create_group("pattern"))
		if (self.child_nodes is not None):
			child_node_names = []
			for i in range(len(self.child_nodes)):
				child_node_name = "child_node"+str(i)
				child_node_names.append(child_node_name)
				self.child_nodes[i].save_hdf5(
					grp.create_group(child_node_name))
			util.save_string_list(child_node_names,
								  dset_name="child_node_names",
								  grp=grp)


def _detect_spurious_merging(patterns, track_set, perplexity,
	min_in_subcluster, min_overlap, prob_and_pertrack_sim_merge_thresholds,
	prob_and_pertrack_sim_dealbreaker_thresholds,
	min_frac, min_num, flank_to_add, window_size, bg_freq,
	verbose=True, max_seqlets_subsample=1000, n_cores=1):

	to_return = []
	for i, pattern in enumerate(patterns):
		if len(pattern.seqlets) > min_in_subcluster:
			pattern.compute_subclusters_and_embedding(
				verbose=False,
				compute_embedding=False,
				perplexity=perplexity, n_jobs=n_cores)

			subpatterns = pattern.subcluster_to_subpattern.values()
			refined_subpatterns = SimilarPatternsCollapser(patterns=subpatterns, 
				track_set=track_set, min_overlap=min_overlap, 
				prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
				prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
				min_frac=min_frac, min_num=min_num, flank_to_add=flank_to_add, window_size=window_size, 
				bg_freq=bg_freq, verbose=True, max_seqlets_subsample=1000, n_cores=1)


			to_return.extend(refined_subpatterns[0]) 
		else:
			to_return.append(pattern)
	
	return SimilarPatternsCollapser(patterns=to_return, 
				track_set=track_set, min_overlap=min_overlap, 
				prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
				prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
				min_frac=min_frac, min_num=min_num, flank_to_add=flank_to_add, window_size=window_size, 
				bg_freq=bg_freq, verbose=True, max_seqlets_subsample=1000, n_cores=1)

def SimilarPatternsCollapser(patterns, track_set,
	min_overlap, prob_and_pertrack_sim_merge_thresholds,
	prob_and_pertrack_sim_dealbreaker_thresholds,
	min_frac, min_num, flank_to_add, window_size, bg_freq,
	verbose=True, max_seqlets_subsample=1000, n_cores=1):
	patterns = [x.copy() for x in patterns]

	merge_hierarchy_levels = []        
	current_level_nodes = [
		PatternMergeHierarchyNode(pattern=x) for x in patterns]
	merge_hierarchy_levels.append(current_level_nodes)

	merge_occurred_last_iteration = True
	merging_iteration = 0

	#negative numbers to indicate which
	# entries need to be filled (versus entries we can infer
	# from the previous iteration of the while loop)
	pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
	pairwise_sims = np.zeros((len(patterns), len(patterns)))

	#loop until no more patterns get merged
	while merge_occurred_last_iteration:
		start  = time.time()
		merging_iteration += 1

		merge_occurred_last_iteration = False

		#Let's subsample 'patterns' to prevent runtime from being too
		# large in calculating pairwise sims. 
		subsample_patterns = [
			(x if x.num_seqlets <= max_seqlets_subsample
			 else util.subsample_pattern(x, num_to_subsample=max_seqlets_subsample)) for x in patterns]

		for i, (pattern1, subsample_pattern1) in enumerate(zip(patterns, subsample_patterns)):
			start = time.time()
			#from modisco.visualization import viz_sequence
			#viz_sequence.plot_weights(pattern1["task0_contrib_scores"].fwd)
			for j, (pattern2, subsample_pattern2) in enumerate(
										zip(patterns, subsample_patterns)):
				#Note: I compute both i,j AND j,i because although
				# the result is the same for the sim, it can be different
				# for the auroc because a different motif is getting
				# shifted over.
				if j == i:
					pairwise_aurocs[i, j] = 0.5
					pairwise_sims[i, j] = 1.0
					continue

				if pairwise_aurocs[i, j] >= 0: #filled in from previous iter
					continue 

				#Compute best alignment between pattern pair
				alnmt, rc, aligner_sim =\
					_align_patterns(parent_pattern=pattern1, child_pattern=pattern2, 
						metric=affinitymat.pearson_correlation, 
						min_overlap=min_overlap, 
						track_names=['task0_contrib_scores'],
						track_transformer=affinitymat.MagnitudeNormalizer()) 

				pairwise_sims[i, j] = aligner_sim

				#get realigned pattern2
				pattern2_coords = [x.coor for x in subsample_pattern2.seqlets]
				if rc: #flip strand if needed to align
					pattern2_coords  = [x.revcomp() for x in pattern2_coords]

				#now apply the alignment
				pattern2_coords = [
					x.shift((1 if x.is_revcomp else -1)*alnmt)
					for x in pattern2_coords] 

				pattern2_shifted_seqlets = track_set.create_seqlets(
					coords=pattern2_coords,
					track_names=['task0_hypothetical_contribs', 
						'task0_contrib_scores'] ) 

				pattern1_fwdseqdata, _ =\
				  core.get_2d_data_from_patterns(
					patterns=subsample_pattern1.seqlets,
					track_names=['task0_hypothetical_contribs', 
						'task0_contrib_scores'] ,
					track_transformer=affinitymat.L1Normalizer())

				pattern2_fwdseqdata, _ =\
				  core.get_2d_data_from_patterns(
					patterns=pattern2_shifted_seqlets,
					track_names=['task0_hypothetical_contribs', 
						'task0_contrib_scores'] ,
					track_transformer=affinitymat.L1Normalizer())

				#Flatten, compute continjacc sim at this alignment
				flat_pattern1_fwdseqdata = pattern1_fwdseqdata.reshape(
					(len(pattern1_fwdseqdata), -1))
				flat_pattern2_fwdseqdata = pattern2_fwdseqdata.reshape(
					(len(pattern2_fwdseqdata), -1))
			
				between_pattern_sims = affinitymat.jaccard(
					flat_pattern1_fwdseqdata[:, :, None], 
					flat_pattern2_fwdseqdata[:, :, None])[:, :, 0].flatten()

				within_pattern1_sims = affinitymat.jaccard(
					flat_pattern1_fwdseqdata[:, :, None], 
					flat_pattern1_fwdseqdata[:, :, None])[:, :, 0].flatten()

				auroc = roc_auc_score(
					y_true=[0 for x in between_pattern_sims]
						   +[1 for x in within_pattern1_sims],
					y_score=list(between_pattern_sims)
							+list(within_pattern1_sims))

				#The symmetrization over i,j and j,i is done later
				pairwise_aurocs[i,j] = auroc


		#pairwise_sims is not symmetric; differ based on which pattern is
		# padded with zeros.
		patterns_to_patterns_aligner_sim =\
			0.5*(pairwise_sims + pairwise_sims.T)
		cross_contamination = 2*(1-np.maximum(pairwise_aurocs,0.5))

		indices_to_merge = []
		merge_partners_so_far = dict([(i, set([i])) for i in
									  range(len(patterns))])

		#merge patterns with highest similarity first
		sorted_pairs = sorted([(i,j,patterns_to_patterns_aligner_sim[i,j])
						for i in range(len(patterns))
						for j in range(len(patterns)) if (i < j)],
						key=lambda x: -x[2])

		#iterate over pairs
		for i, j, aligner_sim in sorted_pairs:
			#symmetrize asymmetric crosscontam
			# take min rather than avg to avoid aggressive merging
			cross_contam = min(cross_contamination[i,j],
								cross_contamination[j,i])

			collapse = any([(cross_contam >= x[0] and aligner_sim >= x[1])
				for x in prob_and_pertrack_sim_merge_thresholds])

			if collapse:
				collapse_passed = True
				#check compatibility for all indices that are
				#about to be merged
				merge_under_consideration = set(
					list(merge_partners_so_far[i])
					+list(merge_partners_so_far[j]))

				for m1 in merge_under_consideration:
					for m2 in merge_under_consideration:
						if (m1 < m2):
							cross_contam_here =\
								0.5*(cross_contamination[m1, m2]+
									 cross_contamination[m2, m1])
							aligner_sim_here =\
								patterns_to_patterns_aligner_sim[
									m1, m2]

							dealbreaker = any([(cross_contam_here <= x[0] and aligner_sim_here <= x[1])              
								for x in prob_and_pertrack_sim_dealbreaker_thresholds])

							if dealbreaker:
								collapse_passed = False
								break

				if collapse_passed:
					indices_to_merge.append((i,j))
					for an_idx in merge_under_consideration:
						merge_partners_so_far[an_idx]=\
							merge_under_consideration 

		for i,j in indices_to_merge:
			pattern1 = patterns[i]
			pattern2 = patterns[j]

			if pattern1 != pattern2: #if not the same object
				if pattern1.num_seqlets < pattern2.num_seqlets:
					parent_pattern, child_pattern = pattern2, pattern1
				else:
					parent_pattern, child_pattern = pattern1, pattern2

				new_pattern = merge_in_seqlets_filledges(
					parent_pattern=parent_pattern,
					seqlets_to_merge=child_pattern.seqlets,
					track_names=['task0_contrib_scores'],
					metric=affinitymat.pearson_correlation,
					min_overlap=min_overlap,
					track_transformer=affinitymat.MagnitudeNormalizer(),
					track_set=track_set,
					verbose=verbose)

				new_pattern = _trim_to_frac_support([new_pattern], 
					min_frac=min_frac, min_num=min_num)[0]

				new_pattern = _expand_seqlets_to_fill_pattern([new_pattern], 
					track_set=track_set, 
					left_flank_to_add=flank_to_add,
					right_flank_to_add=flank_to_add)[0]

				new_pattern = _trim_to_best_window_by_ic([new_pattern],
						window_size=window_size,
						bg_freq=bg_freq)[0]

				new_pattern = _expand_seqlets_to_fill_pattern([new_pattern], 
					track_set=track_set, 
					left_flank_to_add=flank_to_add,
					right_flank_to_add=flank_to_add)[0]

				for k in range(len(patterns)):
					#Replace EVERY case where the parent or child
					# pattern is present with the new pattern. This
					# effectively does single-linkage.
					if (patterns[k]==parent_pattern or
						patterns[k]==child_pattern):
						patterns[k]=new_pattern

		merge_occurred_last_iteration = (len(indices_to_merge) > 0)

		if merge_occurred_last_iteration:
			#Once we are here, each element of 'patterns'
			#will have the new parent of the corresponding element
			#of 'old_patterns'
			old_to_new_pattern_mapping = patterns

			#sort by size and remove redundant patterns
			patterns = sorted(patterns, key=lambda x: -x.num_seqlets)
			patterns = list(OrderedDict([(x,1) for x in patterns]).keys())

			#let's figure out which indices don't require recomputation
			# and use it to repopulate pairwise_sims and pairwise_aurocs
			old_to_new_index_mappings = OrderedDict()
			for old_pattern_idx,(old_pattern_node, corresp_new_pattern)\
				in enumerate(zip(current_level_nodes,
								 old_to_new_pattern_mapping)):
				#if the old pattern was NOT changed in this iteration
				if old_pattern_node.pattern == corresp_new_pattern:
					new_idx = patterns.index(corresp_new_pattern) 
					old_to_new_index_mappings[old_pattern_idx] = new_idx

			new_pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
			new_pairwise_sims = np.zeros((len(patterns), len(patterns)))
			for old_idx_i, new_idx_i in\
				old_to_new_index_mappings.items():
				for old_idx_j, new_idx_j in\
					old_to_new_index_mappings.items():
					new_pairwise_aurocs[new_idx_i, new_idx_j] =\
						pairwise_aurocs[old_idx_i, old_idx_j]
					new_pairwise_sims[new_idx_i, new_idx_j] =\
						pairwise_sims[old_idx_i, old_idx_j]
			pairwise_aurocs = new_pairwise_aurocs 
			pairwise_sims = new_pairwise_sims
				 

			#update the hierarchy
			#the current 'top level' will consist of all the current
			# nodes that didn't get a new parent, plus any new parents
			# created                
			next_level_nodes = []
			for frontier_pattern in patterns:
				#either this pattern is in old_pattern_nodes, in which
				# case take the old_pattern_node entry, or it's a completely
				# new pattern in which case make a node for it
				old_pattern_node_found = False
				for old_pattern_node in current_level_nodes:
					if (old_pattern_node.pattern==frontier_pattern):
						#sanity check..there should be only one node
						# per pattern
						assert old_pattern_node_found==False
						next_level_nodes.append(old_pattern_node)
						old_pattern_node_found = True 
				if (old_pattern_node_found==False):
				   next_level_nodes.append(
					PatternMergeHierarchyNode(frontier_pattern)) 

			for next_level_node in next_level_nodes:
				#iterate over all the old patterns and their new parent
				# in order to set up the child nodes correctly
				for old_pattern_idx,(old_pattern_node, corresp_new_pattern)\
					in enumerate(zip(current_level_nodes,
									 old_to_new_pattern_mapping)):

					#if the node has a new parent
					if (old_pattern_node.pattern != corresp_new_pattern):
						if (next_level_node.pattern==corresp_new_pattern):

							
							#corresp_new_pattern should be comprised of a 
							# merging of all the old patterns at
							# indices_merged_with
							indices_merged = tuple(sorted(
								merge_partners_so_far[old_pattern_idx])) 
							#get the relevant slice         
							submat_crosscontam =\
							 cross_contamination[indices_merged,:][:,
												 indices_merged]
							submat_alignersim =\
							 patterns_to_patterns_aligner_sim[
								indices_merged, :][:,indices_merged]

							if (next_level_node.indices_merged is not None):
								assert (next_level_node.indices_merged
										==indices_merged),\
								 (next_level_node.indices_merged,
								  indices_merged)
							else:
								next_level_node.indices_merged =\
									indices_merged
								next_level_node.submat_crosscontam =\
									submat_crosscontam
								next_level_node.submat_alignersim =\
									submat_alignersim

							next_level_node.child_nodes.append(
											old_pattern_node) 
							assert old_pattern_node.parent_node is None
							old_pattern_node.parent_node = next_level_node
						

			current_level_nodes=next_level_nodes

	return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)
	