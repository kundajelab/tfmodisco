import numpy as np

from . import util
from collections import OrderedDict

class Snippet(object):
	def __init__(self, fwd, rev, has_pos_axis):
		self.fwd = fwd
		self.rev = rev
		self.has_pos_axis = has_pos_axis

	def trim(self, start_idx, end_idx):
		new_fwd = self.fwd[start_idx:end_idx]
		new_rev = (self.rev[len(self)-end_idx:len(self)-start_idx]
				   if self.rev is not None else None)
		return Snippet(fwd=new_fwd, rev=new_rev,
					   has_pos_axis=self.has_pos_axis)

	def save_hdf5(self, grp):
		grp.create_dataset("fwd", data=self.fwd)  
		if (self.rev is not None):
			grp.create_dataset("rev", data=self.rev)
		grp.attrs["has_pos_axis"] = self.has_pos_axis

	def __len__(self):
		return len(self.fwd)

	def revcomp(self):
		return Snippet(fwd=np.copy(self.rev), rev=np.copy(self.fwd),
					   has_pos_axis=self.has_pos_axis)


class DataTrack(object):
	def __init__(self, name, fwd_tracks, rev_tracks, has_pos_axis):
		self.name = name
		self.fwd_tracks = fwd_tracks
		self.rev_tracks = rev_tracks
		self.has_pos_axis = has_pos_axis

	def __len__(self):
		return len(self.fwd_tracks)

	def get_snippet(self, coor):
		right_pad_needed = max((
			 coor.end - len(self.fwd_tracks[coor.example_idx])),0)
		left_pad_needed = max(-coor.start, 0)

		fwd = self.fwd_tracks[coor.example_idx][max(coor.start,0):coor.end]
		rev = (self.rev_tracks[
					coor.example_idx][
					max(len(self.rev_tracks[coor.example_idx])-coor.end,0):
					(len(self.rev_tracks[coor.example_idx])-coor.start)]
					if self.rev_tracks is not None else None)

		if (left_pad_needed > 0 or right_pad_needed > 0):
			print("Applying left/right pad of",left_pad_needed,"and",
				  right_pad_needed,"for",
				  (coor.example_idx, coor.start, coor.end),
				  "with total sequence length",
				  len(self.fwd_tracks[coor.example_idx]))
			fwd = np.pad(array=fwd,
						 pad_width=((left_pad_needed, right_pad_needed),
									(0,0)),
						 mode="constant")
			if (self.rev_tracks is not None):
				rev = np.pad(array=rev,
							 pad_width=(
							  (right_pad_needed, left_pad_needed),
							  (0,0)),
							 mode="constant")
		snippet = Snippet(
				fwd=fwd,
				rev=rev,
				has_pos_axis=self.has_pos_axis)
		if (coor.is_revcomp):
			snippet = snippet.revcomp()
		return snippet


class TrackSet(object):
	def __init__(self, data_tracks=[]):
		self.track_name_to_data_track = OrderedDict()
		for data_track in data_tracks:
			self.add_track(data_track)

	def get_example_idx_len(self, example_idx):
		return len(self.track_name_to_data_track[
					list(self.track_name_to_data_track.keys())[0]]
					.fwd_tracks[example_idx])

	@property
	def num_examples(self):
		return len(self.track_name_to_data_track[
					list(self.track_name_to_data_track.keys())[0]].fwd_tracks)

	def add_track(self, data_track):
		self.num_items = len(data_track) 
		self.track_name_to_data_track[data_track.name] = data_track
		return self

	def create_seqlets(self, coords, track_names=None):
		seqlets = []
		for coor in coords:
			if track_names is None:
				track_names=self.track_name_to_data_track.keys()

			seqlet = Seqlet(coor=coor)
			for track_name in track_names:
				seqlet.add_snippet_from_data_track(
					data_track=self.track_name_to_data_track[track_name])

			seqlets.append(seqlet)

		return seqlets

			
class SeqletCoordinates(object):
	def __init__(self, example_idx, start, end, is_revcomp, score=None):
		self.example_idx = example_idx
		self.start = start
		self.end = end
		self.is_revcomp = is_revcomp
		self.score = score

	def revcomp(self):
		return SeqletCoordinates(
				example_idx=self.example_idx,
				start=self.start, end=self.end,
				is_revcomp=(self.is_revcomp==False))

	def shift(self, shift_amt):
		return SeqletCoordinates(
				example_idx=self.example_idx,
				start=self.start+shift_amt, end=self.end+shift_amt,
				is_revcomp=self.is_revcomp)

	def __len__(self):
		return self.end - self.start

	def __str__(self):
		return ("example:"+str(self.example_idx)
				+",start:"+str(self.start)+",end:"+str(self.end)
				+",rc:"+str(self.is_revcomp))

class Seqlet(object):
	def __init__(self, coor=None):
		self.coor = coor
		self.track_name_to_snippet = OrderedDict()
		self.attribute_name_to_attribute = OrderedDict()
		super(Seqlet, self).__init__()

	def __getitem__(self, key):
		if (key in self.track_name_to_snippet):
			return self.track_name_to_snippet[key]
		elif (key in self.attribute_name_to_attribute):
			return self.attribute_name_to_attribute[key]

	def __setitem__(self, key, value):
		self.attribute_name_to_attribute[key] = value

	def set_attribute(self, attribute_provider):
		self[attribute_provider.name] = attribute_provider(self)

	def add_snippet_from_data_track(self, data_track): 
		snippet = data_track.get_snippet(coor=self.coor)
		return self.add_snippet(data_track_name=data_track.name,
								snippet=snippet)

	def add_snippet(self, data_track_name, snippet):
		self.track_name_to_snippet[data_track_name] = snippet 
		return self

	def add_attribute(self, attribute_name, attribute):
		self.attribute_name_to_attribute[attribute_name] = attribute

	def revcomp(self):
		seqlet = Seqlet(coor=self.coor.revcomp())
		for track_name in self.track_name_to_snippet:
			seqlet.add_snippet(
				data_track_name=track_name,
				snippet=self.track_name_to_snippet[track_name].revcomp()) 
		for attribute_name in self.attribute_name_to_attribute:
			seqlet.add_attribute(
				attribute_name=attribute_name,
				attribute=self.attribute_name_to_attribute[attribute_name])
		return seqlet

	def trim(self, start_idx, end_idx):
		if (self.coor.is_revcomp == False):
			new_coor_start = self.coor.start+start_idx 
			new_coor_end = self.coor.start+end_idx
		else:
			new_coor_start = self.coor.start + (len(self)-end_idx)
			new_coor_end = self.coor.end-start_idx
		new_coor = SeqletCoordinates(
					start=new_coor_start,
					end=new_coor_end,
					example_idx=self.coor.example_idx,
					is_revcomp=self.coor.is_revcomp) 
		new_seqlet = Seqlet(coor=new_coor)  
		for data_track_name in self.track_name_to_snippet:
			new_seqlet.add_snippet(
				data_track_name=data_track_name,
				snippet=self[data_track_name].trim(start_idx, end_idx))
		return new_seqlet

	def __len__(self):
		return len(self.coor)

	@property
	def exidx_start_end_string(self):
		return (str(self.coor.example_idx)+"_"
				+str(self.coor.start)+"_"+str(self.coor.end))


#Using an object rather than namedtuple because alnmt is mutable
class SeqletAndAlignment(object):
	def __init__(self, seqlet, alnmt):
		self.seqlet = seqlet
		self.alnmt = alnmt

#implements the array interface but also tracks the
#unique seqlets for quick membership testing

class SeqletsAndAlignments(object):

	def __init__(self):
		self.arr = []
		self.unique_seqlets = {} 

	@classmethod
	def create(cls, seqlets_and_alnmts):
		obj = cls() 
		for seqlet_and_alnmt in seqlets_and_alnmts:
			obj.append(seqlet_and_alnmt)
		return obj

	def __len__(self):
		return len(self.arr)

	def __iter__(self):
		return self.arr.__iter__()

	def __getitem__(self, idx):
		return self.arr[idx]

	def __contains__(self, seqlet):
		return (seqlet.exidx_start_end_string in self.unique_seqlets)

	def append(self, seqlet_and_alnmt):
		seqlet = seqlet_and_alnmt.seqlet
		if (seqlet.exidx_start_end_string in self.unique_seqlets):
			raise RuntimeError("Seqlet "
			 +seqlet.exidx_start_end_string
			 +" is already in SeqletsAndAlignments array")
		self.arr.append(seqlet_and_alnmt)
		self.unique_seqlets[seqlet.exidx_start_end_string] = seqlet

	def get_seqlets(self):
		return [x.seqlet for x in self.arr]

	def save_hdf5(self, grp):
		util.save_seqlet_coords(seqlets=self.get_seqlets(),
								dset_name="seqlets", grp=grp) 
		grp.create_dataset("alnmts",
						   data=np.array([x.alnmt for x in self.arr]))

	def copy(self):
		the_copy = SeqletsAndAlignments()
		for seqlet_and_alnmt in self:
			the_copy.append(seqlet_and_alnmt)
		return the_copy


class AggregatedSeqlet(Seqlet):
	def __init__(self, seqlets_and_alnmts_arr):
		super(AggregatedSeqlet, self).__init__()

		self._seqlets_and_alnmts = SeqletsAndAlignments()
		if (len(seqlets_and_alnmts_arr)>0):
			#make sure the start is 0
			start_idx = min([x.alnmt for x in seqlets_and_alnmts_arr])
			seqlets_and_alnmts_arr = [SeqletAndAlignment(seqlet=x.seqlet,
				alnmt=x.alnmt-start_idx) for x in seqlets_and_alnmts_arr] 
			self._set_length(seqlets_and_alnmts_arr)
			self._compute_aggregation(seqlets_and_alnmts_arr)
		
		self.subclusters = None
		self.subcluster_to_subpattern = None

	def save_hdf5(self, grp):
		for track_name,snippet in self.track_name_to_snippet.items():
			snippet.save_hdf5(grp.create_group(track_name))
		self._seqlets_and_alnmts.save_hdf5(
			 grp.create_group("seqlets_and_alnmts"))
		if (self.subclusters is not None):
			grp.create_dataset("subclusters", data=self.subclusters)
			#save subcluster_to_subpattern
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

		#this method assumes all the seqlets have been expanded so they
		# all start at 0
		fwd_seqlet_data, _ = get_2d_data_from_patterns(
			patterns=self.seqlets,
			track_names=["task0_hypothetical_contribs", "task0_contrib_scores"],
			track_transformer=affinitymat.L1Normalizer())
		fwd_seqlet_data_vectors = fwd_seqlet_data.reshape(len(fwd_seqlet_data), -1)
	
		n_neighb = min(int(perplexity*3 + 2), len(fwd_seqlet_data_vectors))

		affmat = affinitymat.jaccard(
			X=fwd_seqlet_data_vectors[:, :, None], Y=fwd_seqlet_data_vectors[:, :, None])[:, :, 0]

		seqlet_neighbors = np.argsort(-affmat, axis=1)[:, :n_neighb]
		affmat_nn = np.array([x[neighbors] for x, neighbors in zip(affmat, seqlet_neighbors)])

		aff_to_dist_mat = affinitymat.AffToDistViaInvLogistic() 

		#Got the nearest-neighbor distances, now need to put in sparse matrix
		# format
		distmat_nn = aff_to_dist_mat(affinity_mat=affmat_nn) 
		distmat_sp = util.coo_matrix_from_neighborsformat(
			entries=distmat_nn, neighbors=seqlet_neighbors,
			ncols=len(distmat_nn))
		#convert to csr and sort by indices to (try to) get rid of efficiency warning
		distmat_sp = distmat_sp.tocsr()
		distmat_sp.sort_indices()

		#do density adaptation
		density_adapted_affmat_transformer =\
			affinitymat.NNTsneConditionalProbs(
				perplexity=perplexity,
				aff_to_dist_mat=aff_to_dist_mat)
		sp_density_adapted_affmat = density_adapted_affmat_transformer(
										affmat_nn, seqlet_neighbors)

		#Do Leiden clustering
		cluster_results = cluster.LeidenCluster(sp_density_adapted_affmat,
			initclusters=None, n_jobs=n_jobs,
				affmat_transformer=
					affinitymat.SymmetrizeByAddition(
												   probability_normalize=True),
				numseedstotry=50,
				n_leiden_iterations=-1,
				verbose=verbose)

		self.subclusters = cluster_results['cluster_indices']

		#this method assumes all the seqlets have been expanded so they
		# all start at 0
		subcluster_to_seqletsandalignments = OrderedDict()
		for seqlet, subcluster in zip(self.seqlets, self.subclusters):
			if (subcluster not in subcluster_to_seqletsandalignments):
				subcluster_to_seqletsandalignments[subcluster] = []
			subcluster_to_seqletsandalignments[subcluster].append(
				SeqletAndAlignment(seqlet=seqlet, alnmt=0) )
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
		return AggregatedSeqlet(seqlets_and_alnmts_arr=
								self._seqlets_and_alnmts.copy())


	def trim_to_positions_with_min_support(self,
			min_frac, min_num, verbose=True):
		max_support = max(self.per_position_counts)
		num = min(min_num, max_support*min_frac)
		left_idx = 0
		while self.per_position_counts[left_idx] < num:
			left_idx += 1
		right_idx = len(self.per_position_counts)
		while self.per_position_counts[right_idx-1] < num:
			right_idx -= 1
		return self.trim_to_start_and_end_idx(start_idx=left_idx,
											  end_idx=right_idx,
											  no_skip=False) 


	def trim_to_start_and_end_idx(self, start_idx, end_idx, no_skip=True):
		new_seqlets_and_alnmnts = [] 
		skipped = 0
		for seqlet_and_alnmt in self._seqlets_and_alnmts:
			#if the seqlet overlaps with the target region
			if (seqlet_and_alnmt.alnmt < end_idx and
				((seqlet_and_alnmt.alnmt + len(seqlet_and_alnmt.seqlet))
				  > start_idx)):
				if seqlet_and_alnmt.alnmt > start_idx:
					seqlet_start_idx_trim = 0 
					new_alnmt = seqlet_and_alnmt.alnmt-start_idx
				else:
					seqlet_start_idx_trim = start_idx - seqlet_and_alnmt.alnmt 
					new_alnmt = 0
				if (seqlet_and_alnmt.alnmt+len(seqlet_and_alnmt.seqlet)
					< end_idx):
					seqlet_end_idx_trim = len(seqlet_and_alnmt.seqlet)
				else:
					seqlet_end_idx_trim = end_idx - seqlet_and_alnmt.alnmt
				new_seqlet = seqlet_and_alnmt.seqlet.trim(
								start_idx=seqlet_start_idx_trim,
								end_idx=seqlet_end_idx_trim)
				new_seqlets_and_alnmnts.append(
					SeqletAndAlignment(seqlet=new_seqlet,
									   alnmt=new_alnmt)) 
			else:
				skipped += 1


		return AggregatedSeqlet(seqlets_and_alnmts_arr=new_seqlets_and_alnmnts)

	def _set_length(self, seqlets_and_alnmts_arr):
		self.length = max([x.alnmt + len(x.seqlet)
					   for x in seqlets_and_alnmts_arr])  

	@property
	def seqlets_and_alnmts(self):
		return self._seqlets_and_alnmts

	@property
	def seqlets(self):
		return self._seqlets_and_alnmts.get_seqlets()

	@seqlets_and_alnmts.setter
	def seqlets_and_alnmts(self, val):
		assert type(val).__name__ == "SeqletsAndAlignments"
		self._seqlets_and_alnmts = val

	@property
	def num_seqlets(self):
		return len(self.seqlets_and_alnmts)

	@staticmethod 
	def from_seqlet(seqlet):
		return AggregatedSeqlet(seqlets_and_alnmts_arr=
								[SeqletAndAlignment(seqlet,0)])

	def _compute_aggregation(self,seqlets_and_alnmts_arr):
		self._initialize_track_name_to_aggregation(
			  sample_seqlet=seqlets_and_alnmts_arr[0].seqlet)
		self.per_position_counts = np.zeros((self.length,))
		duplicates = 0
		for seqlet_and_alnmt in seqlets_and_alnmts_arr:
			if (seqlet_and_alnmt.seqlet not in self.seqlets_and_alnmts): 
				self._add_pattern_with_valid_alnmt(
						pattern=seqlet_and_alnmt.seqlet,
						alnmt=seqlet_and_alnmt.alnmt)
			else:
			   duplicates += 1 


	def _initialize_track_name_to_aggregation(self, sample_seqlet): 
		self._track_name_to_agg = OrderedDict() 
		self._track_name_to_agg_revcomp = OrderedDict() 
		for track_name in sample_seqlet.track_name_to_snippet:
			track_shape = tuple([self.length]
						   +list(sample_seqlet[track_name].fwd.shape[1:]))
			self._track_name_to_agg[track_name] =\
				np.zeros(track_shape).astype("float") 
			if (sample_seqlet[track_name].rev is not None):
				self._track_name_to_agg_revcomp[track_name] =\
					np.zeros(track_shape).astype("float") 
			else:
				self._track_name_to_agg_revcomp[track_name] = None
			self.track_name_to_snippet[track_name] = Snippet(
				fwd=self._track_name_to_agg[track_name],
				rev=self._track_name_to_agg_revcomp[track_name],
				has_pos_axis=sample_seqlet[track_name].has_pos_axis) 

	def _add_pattern_with_valid_alnmt(self, pattern, alnmt):
		slice_obj = slice(alnmt, alnmt+len(pattern))
		rev_slice_obj = slice(self.length-(alnmt+len(pattern)),
							  self.length-alnmt)

		self.seqlets_and_alnmts.append(
			 SeqletAndAlignment(seqlet=pattern, alnmt=alnmt))
		self.per_position_counts[slice_obj] += 1.0 

		for track_name in self._track_name_to_agg:
			if (self.track_name_to_snippet[track_name].has_pos_axis==False):
				self._track_name_to_agg[track_name] +=\
					pattern[track_name].fwd
				if (self._track_name_to_agg_revcomp[track_name] is not None):
					self._track_name_to_agg_revcomp[track_name] +=\
						pattern[track_name].rev
			else:
				self._track_name_to_agg[track_name][slice_obj] +=\
					pattern[track_name].fwd 
				if (self._track_name_to_agg_revcomp[track_name] is not None):
					self._track_name_to_agg_revcomp[track_name]\
						 [rev_slice_obj] += pattern[track_name].rev
			self.track_name_to_snippet[track_name] =\
			 Snippet(
			  fwd=(self._track_name_to_agg[track_name]
				   /(self.per_position_counts[:,None]
					 + 1E-7*(self.per_position_counts[:,None]==0))),
			  rev=((self._track_name_to_agg_revcomp[track_name]
				   /(self.per_position_counts[::-1,None]
					 + 1E-7*(self.per_position_counts[::-1,None]==0)))
				   if (self._track_name_to_agg_revcomp[track_name]
					   is not None) else None),
			  has_pos_axis=
			   self.track_name_to_snippet[track_name].has_pos_axis) 

	def __len__(self):
		return self.length


def get_2d_data_from_patterns(patterns, track_names, track_transformer):
	all_fwd_data, all_rev_data = [], []

	for pattern in patterns:
		snippets = [pattern[track_name] for track_name in track_names] 

		fwd_data = np.concatenate([track_transformer(
				 np.reshape(snippet.fwd, (len(snippet.fwd), -1)))
				for snippet in snippets], axis=1)

		rev_data = np.concatenate([track_transformer(
				np.reshape(snippet.rev, (len(snippet.rev), -1)))
				for snippet in snippets], axis=1)

		all_fwd_data.append(fwd_data)
		all_rev_data.append(rev_data)
	
	return np.array(all_fwd_data), np.array(all_rev_data)
