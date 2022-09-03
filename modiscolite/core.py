import numpy as np

from . import util
from collections import OrderedDict

class Snippet(object):
	def __init__(self, fwd, rev):
		self.fwd = fwd
		self.rev = rev

	def trim(self, start_idx, end_idx):
		new_fwd = self.fwd[start_idx:end_idx]
		new_rev = (self.rev[len(self)-end_idx:len(self)-start_idx]
				   if self.rev is not None else None)
		return Snippet(fwd=new_fwd, rev=new_rev)

	def save_hdf5(self, grp):
		grp.create_dataset("fwd", data=self.fwd)  
		grp.create_dataset("rev", data=self.rev)
		grp.attrs["has_pos_axis"] = True

	def __len__(self):
		return len(self.fwd)


class TrackSet(object):
	def __init__(self, one_hot, contrib_scores, hypothetical_contribs):
		self.one_hot = one_hot
		self.contrib_scores = contrib_scores
		self.hypothetical_contribs = hypothetical_contribs
		self.length = len(one_hot[0])

	def create_seqlets(self, seqlets):
		tracks = [self.one_hot, self.contrib_scores, self.hypothetical_contribs]
		names = ['sequence', 'task0_contrib_scores', 'task0_hypothetical_contribs']

		for seqlet in seqlets:
			idx = seqlet.example_idx

			for track, track_name in zip(tracks, names):
				fwd = track[idx][seqlet.start:seqlet.end]
				rev = fwd[::-1, ::-1]

				if seqlet.is_revcomp:
					snippet = Snippet(fwd=rev, rev=fwd)
				else:
					snippet = Snippet(fwd=fwd, rev=rev)

				seqlet.snippets[track_name] = snippet

		return seqlets


class Seqlet(object):
	def __init__(self, example_idx, start, end, is_revcomp):
		self.example_idx = example_idx
		self.start = start
		self.end = end
		self.is_revcomp = is_revcomp
		self.snippets = OrderedDict()
		super(Seqlet, self).__init__()

	def __str__(self):
		return ("example:"+str(self.example_idx)
				+",start:"+str(self.start)+",end:"+str(self.end)
				+",rc:"+str(self.is_revcomp))

	def __len__(self):
		return self.end - self.start

	@property
	def exidx_start_end_string(self):
		return (str(self.example_idx)+"_"
				+str(self.start)+"_"+str(self.end))

	def revcomp(self):
		new_seqlet = Seqlet(
				example_idx=self.example_idx,
				start=self.start, end=self.end,
				is_revcomp=(self.is_revcomp==False))

		for name, snippet in self.snippets.items():
			new_seqlet.snippets[name] = Snippet(fwd=np.copy(snippet.rev), rev=np.copy(snippet.fwd))

		return new_seqlet

	def shift(self, shift_amt):
		return Seqlet(
				example_idx=self.example_idx,
				start=self.start+shift_amt, end=self.end+shift_amt,
				is_revcomp=self.is_revcomp)

	def trim(self, start_idx, end_idx):
		if self.is_revcomp == False:
			new_start = self.start + start_idx 
			new_end = self.start + end_idx
		else:
			new_start = self.end - end_idx
			new_end = self.end - start_idx

		new_seqlet = Seqlet(example_idx=self.example_idx,
			start=new_start, end=new_end, is_revcomp=self.is_revcomp)

		for track_name, snippet in self.snippets.items():
			new_seqlet.snippets[track_name] = snippet.trim(start_idx, end_idx)
		
		return new_seqlet


class AggregatedSeqlet():
	def __init__(self, seqlets_and_alnmts_arr):
		self.seqlets_and_alnmts = []
		self.unique_seqlets = {}
		self.snippets = OrderedDict()

		the_sum = sum([alnmt for _, alnmt in seqlets_and_alnmts_arr])
		if the_sum != 0:
			print(the_sum, "DAWWAD?")
			awdwadwwaawd

		start_idx = min([alnmt for _, alnmt in seqlets_and_alnmts_arr])
		seqlets_and_alnmts_arr = [(seqlet, alnmt-start_idx) for seqlet, alnmt in seqlets_and_alnmts_arr]

		self.length = max([alnmt + len(seqlet) for seqlet, alnmt in seqlets_and_alnmts_arr])  
		self._track_name_to_agg = OrderedDict() 

		sample_seqlet = seqlets_and_alnmts_arr[0][0]
		for track_name in sample_seqlet.snippets:
			track_shape = self.length, 4
			self._track_name_to_agg[track_name] = np.zeros(track_shape, dtype='float') 

			self.snippets[track_name] = Snippet(
				fwd=self._track_name_to_agg[track_name],
				rev=self._track_name_to_agg[track_name][::-1, ::-1] 
			) 

		self.per_position_counts = np.zeros((self.length,))

		for seqlet, alnmt in seqlets_and_alnmts_arr:
			if seqlet.exidx_start_end_string not in self.unique_seqlets: 
				self._add_pattern_with_valid_alnmt(pattern=seqlet, alnmt=alnmt)
		
		self.subclusters = None
		self.subcluster_to_subpattern = None

	def save_hdf5(self, grp):
		for track_name,snippet in self.snippets.items():
			snippet.save_hdf5(grp.create_group(track_name))

		the_sum = sum([alnmt for _, alnmt in self.seqlets_and_alnmts])
		if the_sum != 0:
			print(the_sum, "DAWWAD?2")
			awdwadwwaawd

		seqlets_and_alnmts_grp = grp.create_group("seqlets_and_alnmts")
		util.save_seqlet_coords(seqlets=[seqlet for seqlet, _ in self.seqlets_and_alnmts],
								dset_name="seqlets", grp=seqlets_and_alnmts_grp) 
		seqlets_and_alnmts_grp.create_dataset("alnmts",
						   data=np.array([alnmt for _, alnmt in self.seqlets_and_alnmts]))

		if self.subclusters is not None:
			grp.create_dataset("subclusters", data=self.subclusters)

			subcluster_to_subpattern_grp =\
				grp.create_group("subcluster_to_subpattern")
			util.save_string_list(
				["subcluster_"+str(x) for x in self.subcluster_to_subpattern.keys()],
				dset_name="subcluster_names", grp=subcluster_to_subpattern_grp)
			for subcluster,subpattern in self.subcluster_to_subpattern.items():
				subpattern_grp = subcluster_to_subpattern_grp.create_group(
								"subcluster_"+str(subcluster)) 
				subpattern.save_hdf5(subpattern_grp)

	def compute_subclusters_and_embedding(self, perplexity, n_jobs, 
		verbose=True, compute_embedding=True):

		from . import affinitymat
		from . import cluster

		the_sum = sum([alnmt for _, alnmt in self.seqlets_and_alnmts])
		if the_sum != 0:
			print(the_sum, "DAWWAD?3")
			awdwadwwaawd

		#this method assumes all the seqlets have been expanded so they
		# all start at 0
		fwd_seqlet_data, _ = get_2d_data_from_patterns(
			patterns=self.seqlets,
			track_names=["task0_hypothetical_contribs", "task0_contrib_scores"],
			track_transformer=affinitymat.L1Normalizer())
		fwd_seqlet_data_vectors = fwd_seqlet_data.reshape(len(fwd_seqlet_data), -1)
	
		n_neighb = min(int(perplexity*3 + 2), len(fwd_seqlet_data_vectors))

		affmat_nn = []
		seqlet_neighbors = []
		for x in fwd_seqlet_data_vectors:
			affmat = affinitymat.jaccard(X=x[None, :, None], 
				Y=fwd_seqlet_data_vectors[:, :, None])[:, 0, 0]
			
			neighbors = np.argsort(-affmat)[:n_neighb]

			affmat_nn.append(affmat[neighbors])
			seqlet_neighbors.append(neighbors)

		affmat_nn = np.array(affmat_nn)
		distmat_nn = np.log((1.0/(0.5*np.maximum(affmat_nn, 0.0000001)))-1)
		distmat_nn = np.maximum(distmat_nn, 0.0) #eliminate tiny neg floats

		distmat_sp = util.coo_matrix_from_neighborsformat(
			entries=distmat_nn, neighbors=seqlet_neighbors,
			ncols=len(distmat_nn)).tocsr()
		distmat_sp.sort_indices()

		#do density adaptation
		density_adapted_affmat_transformer =\
			affinitymat.NNTsneConditionalProbs(
				perplexity=perplexity)
		sp_density_adapted_affmat = density_adapted_affmat_transformer(
										affmat_nn, seqlet_neighbors)

		sp_density_adapted_affmat += sp_density_adapted_affmat.T
		sp_density_adapted_affmat /= np.sum(sp_density_adapted_affmat.data)

		#Do Leiden clustering
		cluster_results = cluster.LeidenCluster(sp_density_adapted_affmat,
				n_seeds=50,
				n_leiden_iterations=-1,
				verbose=verbose)

		self.subclusters = cluster_results['cluster_indices']

		#this method assumes all the seqlets have been expanded so they
		# all start at 0
		subcluster_to_seqletsandalignments = OrderedDict()
		for seqlet, subcluster in zip(self.seqlets, self.subclusters):
			if (subcluster not in subcluster_to_seqletsandalignments):
				subcluster_to_seqletsandalignments[subcluster] = []
			
			subcluster_to_seqletsandalignments[subcluster].append((seqlet, 0))

		subcluster_to_subpattern = OrderedDict([
			(subcluster, AggregatedSeqlet(seqletsandalignments))
			for subcluster,seqletsandalignments in
			subcluster_to_seqletsandalignments.items()])

		#resort subcluster_to_subpattern so that the subclusters with the
		# most seqlets come first
		self.subcluster_to_subpattern = OrderedDict(
			sorted(subcluster_to_subpattern.items(),
				   key=lambda x: -len(x[1].seqlets)))

	def copy(self):
		return AggregatedSeqlet(seqlets_and_alnmts_arr=[(seqlet, alnmt) for seqlet, alnmt in self.seqlets_and_alnmts])

	def trim_to_support(self,
			min_frac, min_num, verbose=True):
		max_support = max(self.per_position_counts)
		num = min(min_num, max_support*min_frac)
		
		left_idx = 0
		while self.per_position_counts[left_idx] < num:
			left_idx += 1

		right_idx = len(self.per_position_counts)
		while self.per_position_counts[right_idx-1] < num:
			right_idx -= 1
		
		return self.trim_to_idx(start_idx=left_idx, end_idx=right_idx,
			no_skip=False) 


	def trim_to_idx(self, start_idx, end_idx, no_skip=True):
		new_seqlets_and_alnmnts = [] 
		for seqlet, alnmt in self.seqlets_and_alnmts:
			if alnmt < end_idx and (alnmt + len(seqlet)) > start_idx:
				if alnmt > start_idx:
					seqlet_start_idx_trim = 0 
					new_alnmt = alnmt - start_idx
				else:
					seqlet_start_idx_trim = start_idx - alnmt 
					new_alnmt = 0

				if (alnmt+len(seqlet)) < end_idx:
					seqlet_end_idx_trim = len(seqlet)
				else:
					seqlet_end_idx_trim = end_idx - alnmt

				new_seqlet = seqlet.trim(start_idx=seqlet_start_idx_trim,
					end_idx=seqlet_end_idx_trim)
				
				new_seqlets_and_alnmnts.append((new_seqlet, new_alnmt))

		return AggregatedSeqlet(seqlets_and_alnmts_arr=new_seqlets_and_alnmnts)


	@property
	def seqlets(self):
		return [seqlet for seqlet, _ in self.seqlets_and_alnmts]

	@property
	def num_seqlets(self):
		return len(self.seqlets_and_alnmts)

	def _add_pattern_with_valid_alnmt(self, pattern, alnmt):
		slice_obj = slice(alnmt, alnmt+len(pattern))

		self.seqlets_and_alnmts.append((pattern, alnmt))
		self.unique_seqlets[pattern.exidx_start_end_string] = pattern
		self.per_position_counts[slice_obj] += 1.0 

		for track_name in self._track_name_to_agg:
			self._track_name_to_agg[track_name][slice_obj] += pattern.snippets[track_name].fwd 

			ppc = self.per_position_counts[:, None]
			track = self._track_name_to_agg[track_name] / (ppc + 1E-7*(ppc == 0))

			self.snippets[track_name] = Snippet(
				fwd=track, rev=track[::-1, ::-1]
			)

	def __len__(self):
		return self.length


def get_2d_data_from_patterns(patterns, track_names, track_transformer):
	all_fwd_data, all_rev_data = [], []

	for pattern in patterns:
		snippets = [pattern.snippets[track_name] for track_name in track_names] 

		fwd_data = np.concatenate([track_transformer(
				 np.reshape(snippet.fwd, (len(snippet.fwd), -1)))
				for snippet in snippets], axis=1)

		rev_data = np.concatenate([track_transformer(
				np.reshape(snippet.rev, (len(snippet.rev), -1)))
				for snippet in snippets], axis=1)

		all_fwd_data.append(fwd_data)
		all_rev_data.append(rev_data)
	
	return np.array(all_fwd_data), np.array(all_rev_data)
