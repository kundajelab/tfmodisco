# io.py
# Authors: Jacob Schreiber <jmschreiber91@gmail.com>, Ivy Raine <ivy.ember.raine@gmail.com>

from collections import OrderedDict
import os
import textwrap

import h5py
import hdf5plugin

from typing import List, Literal, Union

import numpy as np
import scipy

from . import util
from . import meme_writer
from . import bed_writer
from . import fasta_writer

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

				n_seqlets = len(old_pattern['seqlets_and_alnmts']['seqlets'])
				seqlet_grp.create_dataset("n_seqlets", data=np.array([n_seqlets]))

				starts = np.zeros(n_seqlets, dtype=int)
				ends = np.zeros(n_seqlets, dtype=int)
				example_idxs = np.zeros(n_seqlets, dtype=int)
				is_revcomps = np.zeros(n_seqlets, dtype=bool)
				for i in range(n_seqlets):
					x = old_pattern['seqlets_and_alnmts']['seqlets'][i]
					
					idx = int(x.decode('utf8').split(',')[0].split(':')[1])
					start = int(x.decode('utf8').split(',')[1].split(':')[1])
					end = int(x.decode('utf8').split(',')[2].split(':')[1])
					rc = x.decode('utf8').split(',')[3].split(':')[1] == 'True'

					starts[i] = start
					ends[i] = end
					example_idxs[i] = idx
					is_revcomps[i] = rc

				seqlet_grp.create_dataset("start", data=starts)
				seqlet_grp.create_dataset("end", data=ends)
				seqlet_grp.create_dataset("example_idx", data=example_idxs)
				seqlet_grp.create_dataset("is_revcomp", data=is_revcomps)

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

				n_seqlets = len(old_pattern['seqlets_and_alnmts']['seqlets'])
				seqlet_grp.create_dataset("n_seqlets", data=np.array([n_seqlets]))

				starts = np.zeros(n_seqlets, dtype=int)
				ends = np.zeros(n_seqlets, dtype=int)
				example_idxs = np.zeros(n_seqlets, dtype=int)
				is_revcomps = np.zeros(n_seqlets, dtype=bool)
				for i in range(n_seqlets):
					x = old_pattern['seqlets_and_alnmts']['seqlets'][i]
					
					idx = int(x.decode('utf8').split(',')[0].split(':')[1])
					start = int(x.decode('utf8').split(',')[1].split(':')[1])
					end = int(x.decode('utf8').split(',')[2].split(':')[1])
					rc = x.decode('utf8').split(',')[3].split(':')[1] == 'True'

					starts[i] = start
					ends[i] = end
					example_idxs[i] = idx
					is_revcomps[i] = rc

				seqlet_grp.create_dataset("start", data=starts)
				seqlet_grp.create_dataset("end", data=ends)
				seqlet_grp.create_dataset("example_idx", data=example_idxs)
				seqlet_grp.create_dataset("is_revcomp", data=is_revcomps)
				
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


def save_hdf5(filename: os.PathLike, pos_patterns, neg_patterns, window_size: int):
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
	
	grp.attrs['window_size'] = window_size
	
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


def write_meme_from_h5(filename: os.PathLike, datatype: util.MemeDataType, output_filename: Union[os.PathLike, None], is_quiet: bool) -> None:
	"""Write a MEME file from an h5 file output from TF-MoDISco. Based on the given datatype.

	Parameters
	----------
	filename: str
		The name of the h5 file to read.
	datatype: MemeDataType
		The datatype to use for the MEME file.
	output_filename: str
		The name of the MEME file to write.
	"""

	alphabet = 'ACGT'
	writer = meme_writer.MEMEWriter(
		memesuite_version='5',
		alphabet=alphabet,
		background_frequencies='A 0.25 C 0.25 G 0.25 T 0.25'
	)


	with h5py.File(filename, 'r') as grp:
		for (name, datasets) in grp['pos_patterns'].items():

			probability_matrix = None
			if datatype == util.MemeDataType.PFM:
				probability_matrix = datasets['sequence'][:] / np.sum(datasets['sequence'][:], axis=1, keepdims=True)
			elif datatype == util.MemeDataType.CWM:
				probability_matrix = datasets['contrib_scores'][:]
			elif datatype == util.MemeDataType.hCWM:
				probability_matrix = datasets['hypothetical_contribs'][:]
			elif datatype == util.MemeDataType.CWM_PFM:
				# Softmax version of CWM.
				probability_matrix = scipy.special.softmax(datasets['contrib_scores'][:], axis=1)
			elif datatype == util.MemeDataType.hCWM_PFM:
				# Softmax version of hCWM.
				probability_matrix = scipy.special.softmax(datasets['hypothetical_contribs'][:], axis=1)
			else:
				raise ValueError("Unknown datatype: {}".format(datatype))

			motif = meme_writer.MEMEWriterMotif(
						name=name,
						probability_matrix=probability_matrix,
						source_sites=1,
						alphabet=alphabet,
						alphabet_length=4)

			writer.add_motif(motif)
	
	if output_filename is not None:
		writer.write(output_filename)
	if not is_quiet:
		print(writer.get_output())


def write_bed_from_h5(modisco_results_filepath: os.PathLike, peaks_filepath: os.PathLike, output_filepath: os.PathLike, valid_chroms: Union[List[str], Literal['*']], window_size: Union[None, int], is_quiet: bool) -> None:
	"""Write a MEME file from an h5 file output from TF-MoDISco. Based on the given datatype.

	Parameters
	----------
	modisco_results_filepath: str
		The name of the h5 file to read.
	peaks_filepath: str
		The name of the peaks file to read.
	output_filepath: str
		The name of the BED file to write.
	valid_chroms: list
		A list of valid chromosomes to filter the peaks file by.
		Example: ['chr1', 'chr2', 'chrX'] || ['1', '2', 'X']
	window_size: int or None
		The window size to use for the BED file. If None, the window size will
		be read from the h5 file.
	"""

	# Store the entire peaks file in memory.
	peak_rows_filtered = None
	with open(peaks_filepath, 'r') as peaks_file:
		peak_rows = peaks_file.read().splitlines()
		# Filter here because each seqlet's `example_idx` is based on a list of
		# just the target chrom(s).
		peak_rows_filtered = util.filter_bed_rows_by_chrom(peak_rows,
							 valid_chroms) if valid_chroms != '*' else peak_rows

	with h5py.File(modisco_results_filepath, 'r') as grp:

		writer = bed_writer.BEDWriter()
		if window_size is None:
			if 'window_size' not in grp.attrs:
				print(textwrap.dedent("""\
					window_size must be specified either in the h5 file or as
					an argument. Older versions of modisco does not store
					`window_size` in the h5 file."""))
				exit(1)
			window_size = int(grp.attrs['window_size'])

		for contribution_dir in ['pos', 'neg']:

			patterns_category = f'{contribution_dir}_patterns'
			if patterns_category not in grp:
				continue

			for (pattern_name, datasets) in grp[patterns_category].items():

				track = bed_writer.BEDTrack(
					track_line=bed_writer.BEDTrackLine(
						arguments=OrderedDict([
							('name', pattern_name),
							('description', f"TF-MoDISco pattern '{pattern_name}' on the positive strand.")
						])
					)
				)

				assert datasets['seqlets']['start'].shape[0] == datasets['seqlets']['end'].shape[0]

				# Process each seqlet within the pattern.
				for idx in range(datasets['seqlets']['start'].shape[0]):
					seqlet_name = f'{pattern_name}.{idx}'

					row_num = datasets['seqlets']['example_idx'][idx]
					peak_row = peak_rows_filtered[row_num].split('\t')
					chrom = peak_row[0]
					score = peak_row[4]
					
					# Seqlet starts and ends are offsets relative to the given
					# window, and the window's centers aligned with the peak's
					# center.

					# Calculate the start and ends.
					absolute_peak_center = (int(peak_row[1]) + int(peak_row[2])) // 2

					window_center_offset = window_size // 2

					seqlet_start_offset = datasets['seqlets']['start'][idx] + 1
					seqlet_end_offset = datasets['seqlets']['end'][idx]
					absolute_seqlet_start = absolute_peak_center - window_center_offset + seqlet_start_offset
					absolute_seqlet_end = absolute_peak_center - window_center_offset + seqlet_end_offset

					strand_char = '-' if bool(datasets['seqlets']['is_revcomp'][idx]) is True else '+'

					track.add_row(
						bed_writer.BEDRow(
							chrom=chrom,
							chrom_start=absolute_seqlet_start,
							chrom_end=absolute_seqlet_end,
							name=seqlet_name,
							score=score,
							strand=strand_char
					))
				
				writer.add_track(track)
		
		if output_filepath is not None:
			writer.write(output_filepath)
		if not is_quiet:
			print(writer.get_output())


def write_fasta_from_h5(modisco_results_filepath: os.PathLike, peaks_filepath: os.PathLike, sequences_file: os.PathLike, output_filepath: Union[os.PathLike, None], valid_chroms: Union[List[str], Literal['*']], window_size: Union[None, int], is_quiet: bool) -> None:
	"""Write a FASTA file from an h5 file output from TF-MoDISco. 

	The results will look like:
	><chrom>:<start>-<end> <strand> <pattern_name>.<seqlet_id>
	FASTA sequence extracted from a fast file

	Parameters
	----------
	modisco_results_filepath: str
		The name of the h5 file to read.
	peaks_filepath: str
		The name of the peaks file to read.
	sequences_file: str
		The name of the sequences file to read.
	output_filepath: str
		The name of the FASTA file to write.
	valid_chroms: List[str]
		The list of valid chromosomes to consider.
		Example: ['chr1', 'chr2', 'chrX'] || ['1', '2', 'X']
	window_size: Union[None, int]
		The window size to use when extracting sequences. If None, then the
		window size will be read from the h5 file.
	"""

	# Note: Make sure this alphabet's order matches the order of the nucleotide tracks.
	alphabet = ['A', 'C', 'G', 'T']

	peak_rows_filtered = None
	with open(peaks_filepath, 'r') as peaks_file:
		peak_rows = peaks_file.read().splitlines()
		# Filter here because each seqlet's `example_idx` is based on a list of
		# just the target chrom(s).
		peak_rows_filtered = util.filter_bed_rows_by_chrom(peak_rows,
							 valid_chroms) if valid_chroms != '*' else peak_rows

	sequences = np.load(sequences_file)

	if 'arr_0' not in sequences:
		print(textwrap.dedent("""\
			The sequences file is incompatible as it does not
			contain an 'arr_0' key. This is likely because the sequences file
			was generated with an older version of modisco."""))
		exit(1)

	sequences = sequences['arr_0']

	if sequences.shape[0] != len(peak_rows_filtered):
		print(textwrap.dedent(f"""\
			The number of rows in the sequences file ({sequences.shape[0]})
			does not match the number of peaks in the peaks file ({len(peak_rows_filtered)}),
			filtered by the set of user-provided chroms. Verify that the user-provided set
			of chroms matches the set used in the interpretation step."""))
		exit(1)


	with h5py.File(modisco_results_filepath, 'r') as grp:

		writer = fasta_writer.FASTAWriter()
		if window_size is None:
			if 'window_size' not in grp.attrs:
				raise ValueError("window_size must be specified either in the h5 file or as an argument. Older versions of modisco does not store `window_size` in the h5 file.")
			window_size = int(grp.attrs['window_size'])

		for contribution_dir in ['pos', 'neg']:

			patterns_category = f'{contribution_dir}_patterns'
			if patterns_category not in grp:
				continue

			for (pattern_name, datasets) in grp[patterns_category].items():

				assert datasets['seqlets']['start'].shape[0] == datasets['seqlets']['end'].shape[0]

				# Process each seqlet within the pattern.
				for idx in range(datasets['seqlets']['start'].shape[0]):

					row_num = datasets['seqlets']['example_idx'][idx]
					peak_row = peak_rows_filtered[row_num].split('\t')
					chrom = peak_row[0]

					seqlet_start_offset = datasets['seqlets']['start'][idx] + 1
					seqlet_end_offset = datasets['seqlets']['end'][idx]

					nucleotide_tracks = sequences[row_num]
					assert nucleotide_tracks.shape[0] == 4
					sequence = []
					for pos in range(seqlet_start_offset, seqlet_end_offset):
						bp_track = nucleotide_tracks[:, pos]
						hit = np.argmax(bp_track)
						sequence.append(alphabet[hit])
					sequence_str = ''.join(sequence)

					# Calculate the start and ends.
					absolute_peak_center = (int(peak_row[1]) + int(peak_row[2])) // 2
					window_center_offset = window_size // 2
					absolute_seqlet_start = absolute_peak_center - window_center_offset + seqlet_start_offset
					absolute_seqlet_end = absolute_peak_center - window_center_offset + seqlet_end_offset

					strand_char = '-' if bool(datasets['seqlets']['is_revcomp'][idx]) is True else '+'

					writer.add_pair(
						fasta_writer.FASTAEntry(
							header=f'{chrom}:{absolute_seqlet_start}-{absolute_seqlet_end} dir={strand_char} {pattern_name}.{idx}',
							sequence=sequence_str
						)
					)
		
		if output_filepath is not None:
			writer.write(output_filepath)
		if not is_quiet:
			print(writer.get_output())


def convert_new_to_old(new_format_filename, old_format_filename):
	'''This function does the opposite of the convert function.
	Given the filepath to a tfmodisco-lite-formatted file (arg 1),
	it writes the same information in the original tfmodisco format
	to file (arg 2).
	
	This function assumes that metacluster0 should be the positive
	patterns (patterns formed from positive importance scores, and 
	that metacluster1 is the negative patterns. It also assumes that
	there is only 1 task (the standard use-case).
	
	This function does not fill in all of the information for the
	original modisco format, in part because some of that information
	is no longer in the new modisco format. But the info converted is
	sufficient to run motif hit calling using the original modisco algorithm.
	'''

	old_f = h5py.File(old_format_filename, "w")
	new_f = h5py.File(new_format_filename, "r")

	old_f.create_dataset("task_names", data=["task0"])

	old_fmt_grp = old_f.create_group('metacluster_idx_to_submetacluster_results')
	
	patterns_group_name_to_metacluster_name = {"pos_patterns" : 'metacluster_0', "neg_patterns" : 'metacluster_1'}
	
	for patterns_group_name in ['pos_patterns', 'neg_patterns']:
		if patterns_group_name in new_f.keys():
			metacluster_name = patterns_group_name_to_metacluster_name[patterns_group_name]
			metacluster_seqlet_strings = []
			
			# new format
			new_patterns_grp = new_f[patterns_group_name]
			
			# old format
			old_metacluster_grp = old_fmt_grp.create_group(metacluster_name)
			old_patterns_grp = old_metacluster_grp.create_group('seqlets_to_patterns_result')
			
			# these needed to avoid error / silent failure
			old_patterns_grp.attrs["success"] = True
			old_patterns_grp.attrs["total_time_taken"] = 1.0
			
			# if there are any patterns for this hit (should always be???)...
			if len(new_patterns_grp.keys()) > 0:
				pattern_names = list(new_patterns_grp.keys())
				# needed because otherwise order is (0, 1, 11, 12, ...)
				pattern_names = sorted(pattern_names, key = lambda name : int(name.split("_")[1]))
				
				old_patterns_subgrp = old_patterns_grp.create_group("patterns")
				
				# for each modisco hit...
				for pattern in pattern_names:
					pattern_grp = new_patterns_grp[pattern]
					
					# new format
					sequence = pattern_grp["sequence"]
					contrib_scores = pattern_grp["contrib_scores"]
					hypothetical_contribs = pattern_grp["hypothetical_contribs"]
					
					# old format
					old_pattern_grp = old_patterns_subgrp.create_group(pattern)
					old_pattern_grp.create_dataset("sequence/fwd", data=sequence)
					old_pattern_grp.create_dataset("task0_contrib_scores/fwd", data=contrib_scores)
					old_pattern_grp.create_dataset("task0_hypothetical_contribs/fwd", data=hypothetical_contribs)
					
					seqlets_grp = pattern_grp['seqlets']
					
					# in the old format, seqlets were stored as a list of strings
					seqlet_strings = []
					for i in range(len(seqlets_grp['example_idx'])):
						# new format: separate arrays for each seqlet attribute,
						# where each array is len(# seqlets)
						example_idx = str(seqlets_grp['example_idx'][i])
						start = str(seqlets_grp['start'][i])
						end = str(seqlets_grp['end'][i])
						rc = str(seqlets_grp['is_revcomp'][i])
						seqlet_str = "example:" + example_idx + ",start:" + start + ",end:" + end + ",rc:" + rc
						seqlet_strings.append(seqlet_str)
					metacluster_seqlet_strings.extend(seqlet_strings)
					
					# old format
					old_seq_align_grp = old_pattern_grp.create_group("seqlets_and_alnmts")
					old_seq_align_grp.create_dataset('seqlets', data=seqlet_strings)
					
					# dummy data to avoid error: alignments are all 0s
					# (must be len(# seqlets), and must be ints)
					old_seq_align_grp.create_dataset('alnmts', data=np.zeros((len(seqlet_strings),)), dtype="i")
					
					# repeat the process for each pattern/cluster (above) for each subpattern/subcluster (below)
					
					subcluster_names = [k for k in pattern_grp.keys() if k.startswith("subcluster_")]
					if len(subcluster_names) > 0:
						old_subpatterns_grp = old_pattern_grp.create_group("subcluster_to_subpattern")
						
						for subpattern in subcluster_names:
							subpattern_grp = pattern_grp[subpattern]
							sequence = subpattern_grp["sequence"]
							contrib_scores = subpattern_grp["contrib_scores"]
							hypothetical_contribs = subpattern_grp["hypothetical_contribs"]

							old_subpattern_grp = old_subpatterns_grp.create_group(subpattern)
							old_subpattern_grp.create_dataset("sequence/fwd", data=sequence)
							old_subpattern_grp.create_dataset("task0_contrib_scores/fwd", data=contrib_scores)
							old_subpattern_grp.create_dataset("task0_hypothetical_contribs/fwd", data=hypothetical_contribs)

							seqlets_grp = subpattern_grp['seqlets']

							seqlet_strings = []
							for i in range(len(seqlets_grp['example_idx'])):
								example_idx = str(seqlets_grp['example_idx'][i])
								start = str(seqlets_grp['start'][i])
								end = str(seqlets_grp['end'][i])
								rc = str(seqlets_grp['is_revcomp'][i])
								seqlet_str = "example:" + example_idx + ",start:" + start + ",end:" + end + ",rc:" + rc
								seqlet_strings.append(seqlet_str)

							old_seq_align_grp = old_subpattern_grp.create_group("seqlets_and_alnmts")
							old_seq_align_grp.create_dataset('seqlets', data=seqlet_strings)
							old_seq_align_grp.create_dataset('alnmts', data=np.zeros((len(seqlet_strings),)), dtype="i")

						old_subpatterns_grp.create_dataset("subcluster_names", data=subcluster_names)
					
				old_patterns_subgrp.create_dataset("all_pattern_names", data=pattern_names)

			# required to avoid error: a collection of seqlets for the entire metacluster
			old_metacluster_grp.create_dataset("seqlets", data=metacluster_seqlet_strings)

	old_f.close()
	new_f.close()
