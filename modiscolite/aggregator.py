# aggregator.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import numpy as np

from . import affinitymat
from . import core
from . import util

from collections import OrderedDict
from sklearn.metrics import roc_auc_score


def polish_pattern(pattern, min_frac, min_num, track_set, flank, window_size, bg_freq):
	pattern = pattern.trim_to_support(min_frac=min_frac, min_num=min_num)

	pattern = _expand_seqlets_to_fill_pattern(pattern, 
		track_set=track_set, left_flank_to_add=flank,
		right_flank_to_add=flank)

	if pattern is None:
		return None

	# Trim by IC
	ppm = pattern.sequence
	per_pos_ic = util.compute_per_position_ic(
		ppm=ppm, background=bg_freq, pseudocount=0.001)

	start_idx = np.argmax(util.cpu_sliding_window_sum(
		arr=per_pos_ic, window_size=window_size))

	end_idx = start_idx + window_size
	pattern = pattern.trim_to_idx(start_idx=start_idx, end_idx=end_idx)

	pattern = _expand_seqlets_to_fill_pattern(pattern, 
		track_set=track_set, left_flank_to_add=flank,
		right_flank_to_add=flank)

	return pattern

def _expand_seqlets_to_fill_pattern(pattern, track_set, left_flank_to_add,
	right_flank_to_add):

	new_seqlets = []
	for seqlet in pattern.seqlets:
		left_expansion = left_flank_to_add 
		right_expansion = len(pattern) - len(seqlet) + right_flank_to_add

		if seqlet.is_revcomp == False:
			start = seqlet.start - left_expansion
			end = seqlet.end + right_expansion
		else:
			start = seqlet.start - right_expansion
			end = seqlet.end + left_expansion
		
		if start >= 0 and end <= track_set.length:
			seqlet = track_set.create_seqlets(
				seqlets=[core.Seqlet(example_idx=seqlet.example_idx,
					start=start, end=end, is_revcomp=seqlet.is_revcomp)])[0]

			new_seqlets.append(seqlet)

	if len(new_seqlets) > 0:
		return core.SeqletSet(seqlets=new_seqlets)
	else:
		return None


def _align_patterns(parent_pattern, child_pattern, metric, min_overlap, 
	transformer, include_hypothetical):

	fwd_data_parent, rev_data_parent = util.get_2d_data_from_patterns(
		[parent_pattern], transformer=transformer,
		include_hypothetical=include_hypothetical)

	fwd_data_child, rev_data_child = util.get_2d_data_from_patterns(
		[child_pattern], transformer=transformer,
		include_hypothetical=include_hypothetical)

	best_crossmetric, best_crossmetric_argmax = metric(fwd_data_child, 
		fwd_data_parent, min_overlap).squeeze()

	best_crossmetric_rev, best_crossmetric_argmax_rev = metric(rev_data_child, 
		fwd_data_parent, min_overlap).squeeze()

	if best_crossmetric_rev > best_crossmetric:
		return int(best_crossmetric_argmax_rev), True, best_crossmetric_rev
	else:
		return int(best_crossmetric_argmax), False, best_crossmetric


def merge_in_seqlets_filledges(parent_pattern, seqlets_to_merge,
	track_set, metric, min_overlap, transformer='l1', 
	include_hypothetical=True):

	parent_pattern = parent_pattern.copy()

	for seqlet in seqlets_to_merge:
		alnmt, revcomp_match, alnmt_score = _align_patterns(parent_pattern, 
			seqlet, metric, min_overlap, transformer, include_hypothetical)
		
		if revcomp_match:
			seqlet = seqlet.revcomp()

		preexpansion_seqletlen = len(seqlet)

		left_expansion = max(alnmt,0)
		right_expansion = max((len(parent_pattern) - (alnmt+len(seqlet))), 0)

		if seqlet.is_revcomp == False:
			start = seqlet.start - left_expansion
			end = seqlet.end + right_expansion
		else:
			start = seqlet.start - right_expansion
			end = seqlet.end + left_expansion

		if start >= 0 and end <= track_set.length:
			seqlet = track_set.create_seqlets(
				seqlets=[core.Seqlet(example_idx=seqlet.example_idx,
					start=start, end=end, is_revcomp=seqlet.is_revcomp)])[0] 
		else:
			continue #don't try adding this seqlet

		#also expand the pattern (if needed) so that the seqlet
		# doesn't go over the edge
		parent_left_expansion = max(0, -alnmt)
		parent_right_expansion = max(0, (alnmt+preexpansion_seqletlen)
										 - len(parent_pattern))

		if (parent_left_expansion > 0) or (parent_right_expansion > 0):
			candidate_parent_pattern = _expand_seqlets_to_fill_pattern(
				parent_pattern, track_set=track_set, 
				left_flank_to_add=parent_left_expansion,
				right_flank_to_add=parent_right_expansion)

			if candidate_parent_pattern is not None:
				parent_pattern = candidate_parent_pattern
			else:
				continue

		#add the seqlet in at alignment 0, assuming it's not already
		# part of the pattern
		if seqlet.string not in parent_pattern.unique_seqlets:
			parent_pattern._add_seqlet(seqlet=seqlet)

	return parent_pattern


class PatternMergeHierarchy(object):
	def __init__(self, root_nodes):
		self.root_nodes = root_nodes

	def add_level(self, level_arr):
		self.levels.append(level_arr)

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


def _detect_spurious_merging(patterns, track_set, perplexity,
	min_in_subcluster, min_overlap, prob_and_pertrack_sim_merge_thresholds,
	prob_and_pertrack_sim_dealbreaker_thresholds,
	min_frac, min_num, flank_to_add, window_size, bg_freq,
	n_seeds, max_seqlets_subsample=1000):

	to_return = []
	for i, pattern in enumerate(patterns):
		if len(pattern.seqlets) > min_in_subcluster:
			pattern.compute_subpatterns(perplexity=perplexity, n_seeds=n_seeds)

			subpatterns = pattern.subcluster_to_subpattern.values()
			refined_subpatterns = SimilarPatternsCollapser(patterns=subpatterns, 
				track_set=track_set, min_overlap=min_overlap, 
				prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
				prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
				min_frac=min_frac, min_num=min_num, flank_to_add=flank_to_add, window_size=window_size, 
				bg_freq=bg_freq, max_seqlets_subsample=1000)

			to_return.extend(refined_subpatterns[0]) 
		else:
			to_return.append(pattern)
	
	return SimilarPatternsCollapser(patterns=to_return, 
				track_set=track_set, min_overlap=min_overlap, 
				prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
				prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
				min_frac=min_frac, min_num=min_num, flank_to_add=flank_to_add, window_size=window_size, 
				bg_freq=bg_freq, max_seqlets_subsample=1000)

def SimilarPatternsCollapser(patterns, track_set,
	min_overlap, prob_and_pertrack_sim_merge_thresholds,
	prob_and_pertrack_sim_dealbreaker_thresholds,
	min_frac, min_num, flank_to_add, window_size, bg_freq,
	max_seqlets_subsample=1000):
	patterns = [x.copy() for x in patterns]

	merge_hierarchy_levels = []        
	current_level_nodes = [
		PatternMergeHierarchyNode(pattern=x) for x in patterns]
	merge_hierarchy_levels.append(current_level_nodes)

	merge_occurred_last_iteration = True

	#negative numbers to indicate which
	# entries need to be filled (versus entries we can infer
	# from the previous iteration of the while loop)
	pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
	pairwise_sims = np.zeros((len(patterns), len(patterns)))

	#loop until no more patterns get merged
	while merge_occurred_last_iteration:
		merge_occurred_last_iteration = False

		#Let's subsample 'patterns' to prevent runtime from being too
		# large in calculating pairwise sims. 
		subsample_patterns = []
		for pattern in patterns:
			if len(pattern.seqlets) > max_seqlets_subsample:
				subsample = np.random.RandomState(1234).choice(
					a=pattern.seqlets, replace=False, size=max_seqlets_subsample)
				pattern = core.SeqletSet(seqlets=subsample)

			subsample_patterns.append(pattern)

		n = len(patterns)
		for i in range(n):
			for j in range(n):
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
					_align_patterns(parent_pattern=patterns[i], 
						child_pattern=patterns[j], 
						metric=affinitymat.pearson_correlation, 
						min_overlap=min_overlap, 
						include_hypothetical=False,
						transformer='magnitude') 

				pairwise_sims[i, j] = aligner_sim

				#get realigned pattern2
				pattern2_coords = subsample_patterns[j].seqlets
				if rc: #flip strand if needed to align
					pattern2_coords  = [x.revcomp() for x in pattern2_coords]

				#now apply the alignment
				pattern2_coords = [x.shift((1 if x.is_revcomp else -1)*alnmt)
					for x in pattern2_coords] 

				# Filter out bad seqlets
				pattern2_coords = [seqlet for seqlet in pattern2_coords 
					if seqlet.start >= 0 and seqlet.end < track_set.length]

				if len(pattern2_coords) == 0:
					pairwise_sims[i, j] = 0.0
					pairwise_aurocs[i, j] = 0.5
					continue

				pattern2_shifted_seqlets = track_set.create_seqlets(
					seqlets=pattern2_coords)

				pattern1_fwdseqdata, _ =\
				  util.get_2d_data_from_patterns(subsample_patterns[i].seqlets)

				pattern2_fwdseqdata, _ =\
				  util.get_2d_data_from_patterns(pattern2_shifted_seqlets)

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
		patterns_to_patterns_aligner_sim = 0.5*(pairwise_sims+pairwise_sims.T)
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

		for i, j in indices_to_merge:
			pattern1 = patterns[i]
			pattern2 = patterns[j]

			if pattern1 != pattern2: #if not the same object
				if len(pattern1.seqlets) < len(pattern2.seqlets):
					parent_pattern, child_pattern = pattern2, pattern1
				else:
					parent_pattern, child_pattern = pattern1, pattern2

				new_pattern = merge_in_seqlets_filledges(
					parent_pattern=parent_pattern,
					seqlets_to_merge=child_pattern.seqlets,
					include_hypothetical=False,
					metric=affinitymat.pearson_correlation,
					min_overlap=min_overlap,
					transformer='magnitude',
					track_set=track_set)

				new_pattern = polish_pattern(new_pattern, min_frac=min_frac, 
					min_num=min_num, track_set=track_set, flank=flank_to_add, 
					window_size=window_size, bg_freq=bg_freq)

				if new_pattern is not None:
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
			patterns = sorted(patterns, key=lambda x: -len(x.seqlets))
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

							old_pattern_node.parent_node = next_level_node

			current_level_nodes = next_level_nodes

	return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)
	