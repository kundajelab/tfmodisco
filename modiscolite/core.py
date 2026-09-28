# core.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>
# adapted from code written by Avanti Shrikumar 

import pickle

import numpy as np
import scipy.sparse

from . import affinitymat
from . import cluster
from . import util

from collections import OrderedDict

class TrackSet(object):
	def __init__(self, one_hot, contrib_scores, hypothetical_contribs):
		self.one_hot = one_hot
		self.contrib_scores = contrib_scores
		self.hypothetical_contribs = hypothetical_contribs
		self.length = len(one_hot[0])

	def create_seqlets(self, seqlets):
		for seqlet in seqlets:
			idx = seqlet.example_idx
			s, e = seqlet.start, seqlet.end

			if seqlet.is_revcomp:
				seqlet.sequence = self.one_hot[idx][s:e][::-1, ::-1]
				seqlet.contrib_scores = self.contrib_scores[idx][s:e][::-1, ::-1]
				seqlet.hypothetical_contribs = self.hypothetical_contribs[idx][s:e][::-1, ::-1]
			else:
				seqlet.sequence = self.one_hot[idx][s:e]
				seqlet.contrib_scores = self.contrib_scores[idx][s:e]
				seqlet.hypothetical_contribs = self.hypothetical_contribs[idx][s:e]				

		return seqlets

	def _add_seqlet_sums(self, seqlets, length, sums):
		"""Add up the three tracks of `seqlets` into `sums`, the zeroed
		running sums of a SeqletSet of that length, as SeqletSet.__init__
		adds them up: in seqlet order, from the seqlets' coordinates rather
		than their arrays (util._seqlet_sums). The seqlets must be this
		track set's: each one's arrays the views create_seqlets makes at its
		coordinates, and each `length` long. Returns False, with `sums`
		zero again, when the kernel does not apply: tracks that are not
		float32 (N, L, 4) arrays of one shape, a coordinate outside them, or
		a NaN among the values."""

		tracks = (self.one_hot, self.contrib_scores, self.hypothetical_contribs)
		for track in tracks:
			if not (isinstance(track, np.ndarray) and track.dtype == np.float32
					and track.ndim == 3 and track.shape == tracks[0].shape
					and track.shape[2] == 4):
				return False

		if len(seqlets) == 0 or sums[0].shape != (length, 4):
			return False

		example_idx = np.array([seqlet.example_idx for seqlet in seqlets],
			dtype=np.int64)
		starts = np.array([seqlet.start for seqlet in seqlets], dtype=np.int64)
		revcomp = np.array([bool(seqlet.is_revcomp) for seqlet in seqlets],
			dtype=np.uint8)

		n_examples, track_length, _ = tracks[0].shape
		if (example_idx.min() < 0 or example_idx.max() >= n_examples
				or starts.min() < 0 or starts.max() + length > track_length):
			return False

		if util._seqlet_sums(*tracks, example_idx, starts, revcomp, *sums):
			return True

		for total in sums:
			total[...] = 0.0
		return False


class Seqlet(object):
	def __init__(self, example_idx, start, end, is_revcomp):
		self.example_idx = example_idx
		self.start = start
		self.end = end
		self.is_revcomp = is_revcomp

		self.sequence = None
		self.contrib_scores = None
		self.hypothetical_contribs = None

		super(Seqlet, self).__init__()

	def __str__(self):
		return ("example:"+str(self.example_idx)
				+",start:"+str(self.start)+",end:"+str(self.end)
				+",rc:"+str(self.is_revcomp))

	def __len__(self):
		return self.end - self.start

	@property
	def string(self):
		return str(self.example_idx)+"_"+str(self.start)+"_"+str(self.end)

	def revcomp(self):
		new_seqlet = Seqlet(
				example_idx=self.example_idx,
				start=self.start, end=self.end,
				is_revcomp=(self.is_revcomp==False))

		new_seqlet.sequence = self.sequence[::-1, ::-1]
		new_seqlet.contrib_scores = self.contrib_scores[::-1, ::-1]
		new_seqlet.hypothetical_contribs = self.hypothetical_contribs[::-1, ::-1]
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

		s, e = start_idx, end_idx
		new_seqlet.sequence = self.sequence[s:e]
		new_seqlet.contrib_scores = self.contrib_scores[s:e]
		new_seqlet.hypothetical_contribs = self.hypothetical_contribs[s:e]
		return new_seqlet


class SeqletSet():
	def __init__(self, seqlets, track_set=None):
		# track_set: the TrackSet whose create_seqlets made `seqlets`, when
		# the caller knows it, so that the sums can be added up from the
		# seqlets' coordinates (TrackSet._add_seqlet_sums).
		self.seqlets = []
		self.unique_seqlets = {}
		lengths = [len(seqlet) for seqlet in seqlets]
		self.length = max(lengths)

		self._sequence_sum = np.zeros((self.length, 4), dtype='float')
		self._contrib_sum = np.zeros((self.length, 4), dtype='float')
		self._hypothetical_sum = np.zeros((self.length, 4), dtype='float')

		self.per_position_counts = np.zeros((self.length,))

		# The averages `sequence`, `contrib_scores` and
		# `hypothetical_contribs` are computed from the sums when read, and
		# kept until the next _add_seqlet. None means not yet computed.
		self._ppc = None
		self._sequence = None
		self._contrib_scores = None
		self._hypothetical_contribs = None

		if min(lengths) == self.length:
			# All seqlets are self.length long: keep the first seqlet of
			# each string, in order, and build the counts and sums in one
			# batch. This gives the same bits as calling _add_seqlet on each
			# kept seqlet: numpy reduces along the stacked (outer) axis one
			# seqlet at a time, starting from `initial`, which is the order
			# of the running sums.
			for seqlet in seqlets:
				key = seqlet.string
				if key not in self.unique_seqlets:
					self.unique_seqlets[key] = seqlet
					self.seqlets.append(seqlet)

			if self.seqlets:
				self.per_position_counts += len(self.seqlets)
				if track_set is None or not track_set._add_seqlet_sums(
						self.seqlets, self.length, self._sums()):
					for total, track in ((self._sequence_sum, 'sequence'),
							(self._contrib_sum, 'contrib_scores'),
							(self._hypothetical_sum, 'hypothetical_contribs')):
						np.add.reduce(np.array([getattr(seqlet, track)
							for seqlet in self.seqlets]), axis=0,
							dtype=np.float64, initial=0.0, out=total)
		else:
			for seqlet in seqlets:
				if seqlet.string not in self.unique_seqlets:
					self._add_seqlet(seqlet=seqlet)

		self.subclusters = None
		self.subcluster_to_subpattern = None

	def compute_subpatterns(self, perplexity, n_seeds, n_iterations=-1):
		#this method assumes all the seqlets have been expanded so they
		# all start at 0
		X = util.get_2d_data_from_patterns(self.seqlets)[0]
		X = X.reshape(len(X), -1)
	
		n = len(X)
		n_neighb = min(int(perplexity*3 + 2), len(X))

		affmat_nn, seqlet_neighbors = affinitymat.pairwise_jaccard(X, n_neighb)

		# Upstream built a sparse distance matrix here that nothing read;
		# NNTsneConditionalProbs computes the same distances itself.

		#do density adaptation
		sp_density_adapted_affmat = affinitymat.NNTsneConditionalProbs(
				perplexity=perplexity)(affmat_nn, seqlet_neighbors)

		sp_density_adapted_affmat += sp_density_adapted_affmat.T
		sp_density_adapted_affmat /= np.sum(sp_density_adapted_affmat.data)

		#Do Leiden clustering
		self._set_subclusters(cluster.LeidenCluster(sp_density_adapted_affmat,
			n_seeds=n_seeds, n_leiden_iterations=n_iterations))

	def _compile_subpattern_kernels(self):
		"""Compile the numba kernel compute_subpatterns calls, for the
		argument types it passes, by calling it on this pattern's first two
		seqlets. Called before forking workers that subcluster patterns, so
		that they inherit the compiled code instead of each compiling it."""

		X = util.get_2d_data_from_patterns(self.seqlets[:2])[0]
		X = X.reshape(len(X), -1)
		try:
			affinitymat.pairwise_jaccard(X, len(X))
		except Exception:
			pass

	def _set_subclusters(self, subclusters):
		"""Store the subcluster of each seqlet (the Leiden membership) and
		build the subpatterns from it: the second half of
		compute_subpatterns, also used to attach a membership computed in a
		forked worker."""

		self.subclusters = subclusters

		#this method assumes all the seqlets have been expanded so they
		# all start at 0
		subcluster_to_seqletsandalignments = OrderedDict()
		for seqlet, subcluster in zip(self.seqlets, self.subclusters):
			if (subcluster not in subcluster_to_seqletsandalignments):
				subcluster_to_seqletsandalignments[subcluster] = []
			
			subcluster_to_seqletsandalignments[subcluster].append(seqlet)

		subcluster_to_subpattern = OrderedDict([
			(subcluster, SeqletSet(seqletsandalignments))
			for subcluster,seqletsandalignments in
			subcluster_to_seqletsandalignments.items()])

		#resort subcluster_to_subpattern so that the subclusters with the
		# most seqlets come first
		self.subcluster_to_subpattern = OrderedDict(
			sorted(subcluster_to_subpattern.items(),
				   key=lambda x: -len(x[1].seqlets)))

	@classmethod
	def _assemble(cls, seqlets, unique_seqlets, length, per_position_counts,
		sums):
		"""A SeqletSet holding this state, made without __init__'s pass over
		the seqlets. The caller guarantees that it is what
		SeqletSet(seqlets) would hold: `seqlets` and `unique_seqlets` as
		__init__ would keep them, and `per_position_counts` and `sums` (the
		sequence, contrib_scores and hypothetical_contribs running sums) as
		it would add them up from those seqlets, value for value."""

		self = cls.__new__(cls)
		self.seqlets = seqlets
		self.unique_seqlets = unique_seqlets
		self.length = length
		self._sequence_sum, self._contrib_sum, self._hypothetical_sum = sums
		self.per_position_counts = per_position_counts
		self._ppc = None
		self._sequence = None
		self._contrib_scores = None
		self._hypothetical_contribs = None
		self.subclusters = None
		self.subcluster_to_subpattern = None
		return self

	def _sums(self):
		return self._sequence_sum, self._contrib_sum, self._hypothetical_sum

	def _spans_length(self):
		"""Whether every seqlet, and each of its three tracks, is exactly as
		long as this set: then every running sum has added every seqlet, in
		order, at every position."""

		n = self.length
		for seqlet in self.seqlets:
			if (len(seqlet) != n or seqlet.sequence.shape[0] != n
					or seqlet.contrib_scores.shape[0] != n
					or seqlet.hypothetical_contribs.shape[0] != n):
				return False
		return True

	def copy(self):
		# SeqletSet(seqlets=self.seqlets) keeps every seqlet when their
		# strings are distinct, with this set's length, and adds up the same
		# counts and sums in the same order, so this set's are copied.
		if len(self.unique_seqlets) == len(self.seqlets):
			return SeqletSet._assemble(list(self.seqlets),
				dict(self.unique_seqlets), self.length,
				self.per_position_counts.copy(),
				tuple(total.copy() for total in self._sums()))

		return SeqletSet(seqlets=[seqlet for seqlet in self.seqlets])

	def trim_to_support(self, min_frac, min_num):
		max_support = max(self.per_position_counts)
		num = min(min_num, max_support*min_frac)
		
		left_idx = 0
		while self.per_position_counts[left_idx] < num:
			left_idx += 1

		right_idx = len(self.per_position_counts)
		while self.per_position_counts[right_idx-1] < num:
			right_idx -= 1
		
		return self.trim_to_idx(start_idx=left_idx, end_idx=right_idx) 

	def trim_to_idx(self, start_idx, end_idx):
		new_seqlets = []
		for seqlet in self.seqlets:
				new_seqlet = seqlet.trim(start_idx=start_idx, end_idx=end_idx)
				new_seqlets.append(new_seqlet)

		# When every seqlet spans this set and the window lies inside it,
		# each trimmed seqlet is its seqlet's rows start_idx:end_idx; if their
		# strings are still distinct, SeqletSet(new_seqlets) keeps all of
		# them and adds up at each position, in the same order, the values
		# this set added up there, so its counts and sums are this set's over
		# the window.
		if 0 <= start_idx < end_idx <= self.length:
			unique = {seqlet.string: seqlet for seqlet in new_seqlets}
			if len(unique) == len(new_seqlets) and self._spans_length():
				length = end_idx - start_idx
				counts = np.zeros((length,))
				counts += len(new_seqlets)
				return SeqletSet._assemble(new_seqlets, unique, length, counts,
					tuple(total[start_idx:end_idx].copy()
						for total in self._sums()))

		return SeqletSet(seqlets=new_seqlets)

	def _add_seqlet(self, seqlet):
		n = len(seqlet)

		self.seqlets.append(seqlet)
		self.unique_seqlets[seqlet.string] = seqlet
		self.per_position_counts[:n] += 1.0

		self._sequence_sum[:n] += seqlet.sequence
		self._contrib_sum[:n] += seqlet.contrib_scores
		self._hypothetical_sum[:n] += seqlet.hypothetical_contribs

		# Recomputed from the new sums and counts when next read.
		self._ppc = None
		self._sequence = None
		self._contrib_scores = None
		self._hypothetical_contribs = None

	def _counts(self):
		"""The per-position counts as the divisor of the averages: a column,
		with zero counts replaced by 1E-7."""

		if self._ppc is None:
			ppc = self.per_position_counts[:, None]
			self._ppc = ppc + 1E-7 * (ppc == 0)
		return self._ppc

	# Each average is a new array, made when it is first read after a
	# change, so an array read earlier keeps the values it had. Assigning
	# one stores the value until the next _add_seqlet.
	@property
	def sequence(self):
		if self._sequence is None:
			self._sequence = self._sequence_sum / self._counts()
		return self._sequence

	@sequence.setter
	def sequence(self, value):
		self._sequence = value

	@property
	def contrib_scores(self):
		if self._contrib_scores is None:
			self._contrib_scores = self._contrib_sum / self._counts()
		return self._contrib_scores

	@contrib_scores.setter
	def contrib_scores(self, value):
		self._contrib_scores = value

	@property
	def hypothetical_contribs(self):
		if self._hypothetical_contribs is None:
			self._hypothetical_contribs = (self._hypothetical_sum /
				self._counts())
		return self._hypothetical_contribs

	@hypothetical_contribs.setter
	def hypothetical_contribs(self, value):
		self._hypothetical_contribs = value

	def __len__(self):
		return self.length

	def save_seqlets(self, filename):
		bases = np.array(['A', 'C', 'G', 'T'])

		with open(filename, "w") as outfile:
			for seqlet in self.seqlets:
				sequence = "".join(bases[np.argmax(seqlet.sequence, axis=-1)])
				example_index = seqlet.example_idx
				start, end = seqlet.start, seqlet.end
				outfile.write(">example%d:%d-%d\n" % (example_index, start, end))
				outfile.write(sequence + "\n")
