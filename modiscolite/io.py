# io.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>

import h5py
import hdf5plugin

import numpy as np

def convert(old_filename, filename):
	old_grp = h5py.File(old_filename, "r")['metacluster_idx_to_submetacluster_results']
	new_grp = h5py.File(filename, "w")

	if 'metacluster_0' in old_grp.keys():
		pos_patterns_grp = new_grp.create_group("pos_patterns")
		old_patterns_grp = old_grp['metacluster_0']['seqlets_to_patterns_result']

		if 'patterns' in old_patterns_grp:
			old_patterns_grp = old_patterns_grp['patterns'] 

			for pattern in old_patterns_grp['all_pattern_names'][:]:
				old_pattern = old_patterns_grp[pattern]

				sequence = old_pattern['sequence']['fwd'][:]
				contrib_scores = old_pattern['task0_contrib_scores']['fwd'][:]
				hypothetical_contribs = old_pattern['task0_hypothetical_contribs']['fwd'][:]


				pattern_grp = pos_patterns_grp.create_group(pattern)
				pattern_grp.create_dataset("sequence", data=sequence)
				pattern_grp.create_dataset("contrib_scores", data=contrib_scores)
				pattern_grp.create_dataset("hypothetical_contribs", data=hypothetical_contribs)
				
				seqlet_grp = pattern_grp.create_group("seqlets")
				seqlet_grp.create_dataset("n_seqlets", 
					data=np.array([len(old_pattern['seqlets_and_alnmts']['seqlets'])]))

				if 'subcluster_to_subpattern' in old_pattern.keys():
					old_subpatterns_grp = old_pattern['subcluster_to_subpattern']
					for subpattern in old_subpatterns_grp['subcluster_names'][:]:
						old_subpattern = old_subpatterns_grp[subpattern]

						sequence = old_subpattern['sequence']['fwd'][:]
						contrib_scores = old_subpattern['task0_contrib_scores']['fwd'][:]
						hypothetical_contribs = old_subpattern['task0_hypothetical_contribs']['fwd'][:]

						subpattern_grp = pattern_grp.create_group(subpattern)
						subpattern_grp.create_dataset("sequence", data=sequence)
						subpattern_grp.create_dataset("contrib_scores", data=contrib_scores)
						subpattern_grp.create_dataset("hypothetical_contribs", data=hypothetical_contribs)
						
						seqlet_grp = subpattern_grp.create_group("seqlets")
						seqlet_grp.create_dataset("n_seqlets", 
							data=np.array([len(old_subpattern['seqlets_and_alnmts']['seqlets'])]))

	if 'metacluster_1' in old_grp.keys():
		neg_patterns_grp = new_grp.create_group("neg_patterns")
		old_patterns_grp = old_grp['metacluster_1']['seqlets_to_patterns_result']

		if 'patterns' in old_patterns_grp:
			old_patterns_grp = old_patterns_grp['patterns']

			for pattern in old_patterns_grp['all_pattern_names'][:]:
				old_pattern = old_patterns_grp[pattern]

				sequence = old_pattern['sequence']['fwd'][:]
				contrib_scores = old_pattern['task0_contrib_scores']['fwd'][:]
				hypothetical_contribs = old_pattern['task0_hypothetical_contribs']['fwd'][:]

				pattern_grp = neg_patterns_grp.create_group(pattern)
				pattern_grp.create_dataset("sequence", data=sequence)
				pattern_grp.create_dataset("contrib_scores", data=contrib_scores)
				pattern_grp.create_dataset("hypothetical_contribs", data=hypothetical_contribs)

				seqlet_grp = pattern_grp.create_group("seqlets")
				seqlet_grp.create_dataset("n_seqlets", 
					data=np.array([len(old_pattern['seqlets_and_alnmts']['seqlets'])]))

				if 'subcluster_to_subpattern' in old_pattern.keys():
					old_subpatterns_grp = old_pattern['subcluster_to_subpattern']
					for subpattern in old_subpatterns_grp['subcluster_names'][:]:
						old_subpattern = old_subpatterns_grp[subpattern]
						sequence = old_subpattern['sequence']['fwd'][:]
						contrib_scores = old_subpattern['task0_contrib_scores']['fwd'][:]
						hypothetical_contribs = old_subpattern['task0_hypothetical_contribs']['fwd'][:]

						subpattern_grp = pattern_grp.create_group(subpattern)
						subpattern_grp.create_dataset("sequence", data=sequence)
						subpattern_grp.create_dataset("contrib_scores", data=contrib_scores)
						subpattern_grp.create_dataset("hypothetical_contribs", data=hypothetical_contribs)

						seqlet_grp = subpattern_grp.create_group("seqlets")
						seqlet_grp.create_dataset("n_seqlets", 
							data=np.array([len(old_subpattern['seqlets_and_alnmts']['seqlets'])]))


def save_pattern(pattern, grp):
	grp.create_dataset("sequence", data=pattern.sequence)
	grp.create_dataset("contrib_scores", data=pattern.contrib_scores)
	grp.create_dataset("hypothetical_contribs", data=pattern.hypothetical_contribs)

	seqlet_grp = grp.create_group("seqlets")
	seqlet_grp.create_dataset("n_seqlets", data=np.array([len(pattern.seqlets)]))
	seqlet_grp.create_dataset("start", 
		data=np.array([seqlet.start for seqlet in pattern.seqlets]))
	seqlet_grp.create_dataset("end", 
		data=np.array([seqlet.end for seqlet in pattern.seqlets]))
	seqlet_grp.create_dataset("example_idx", 
		data=np.array([seqlet.example_idx for seqlet in pattern.seqlets]))
	seqlet_grp.create_dataset("is_revcomp",
		data=np.array([seqlet.is_revcomp for seqlet in pattern.seqlets]))

	seqlet_grp.create_dataset("sequence",
		data=np.array([seqlet.sequence for seqlet in pattern.seqlets]))
	seqlet_grp.create_dataset("contrib_scores",
		data=np.array([seqlet.contrib_scores for seqlet in pattern.seqlets]))
	seqlet_grp.create_dataset("hypothetical_contribs",
		data=np.array([seqlet.hypothetical_contribs for seqlet in pattern.seqlets]))

	if pattern.subclusters is not None:
		for subcluster, subpattern in pattern.subcluster_to_subpattern.items():
			subpattern_grp = grp.create_group("subpattern_"+str(subcluster)) 
			save_pattern(subpattern, subpattern_grp)


def save_hdf5(filename, pos_patterns, neg_patterns):
	"""Save the results of tf-modisco to a h5 file.

	This function will save the SeqletSets and their associated seqlets in
	a minimal new minimal format or in the older format originally used by
	TF-MoDISco. Regardless of format, only information used in the SeqletSets
	are saved.


	Parameters
	----------
	filename: str
		The name of the h5 file to save to.

	pos_patterns: list or None
		A list of SeqletSet objects or None.

	neg_patterns: list or None
		A list of SeqletSet objects or None.
	"""

	grp = h5py.File(filename, 'w')
	
	if pos_patterns is not None:
		pos_group = grp.create_group("pos_patterns")
		for idx, pattern in enumerate(pos_patterns):
			pos_pattern = pos_group.create_group("pattern_"+str(idx))
			save_pattern(pattern, pos_pattern)

	if neg_patterns is not None:
		neg_group = grp.create_group("neg_patterns")
		for idx, pattern in enumerate(neg_patterns):
			neg_pattern = neg_group.create_group("pattern_"+str(idx))
			save_pattern(pattern, neg_pattern)
