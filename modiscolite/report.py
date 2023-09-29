import os
from pathlib import Path
import pickle
import types
from typing import List, Union
import h5py
import pandas
import tempfile
import shutil

import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

import logomaker

pd.options.display.max_colwidth = 500

def read_meme(filename):
	motifs = {}

	with open(filename, "r") as infile:
		motif, width, i = None, None, 0

		for line in infile:
			if motif is None:
				if line[:5] == 'MOTIF':
					motif = line.split()[1]
				else:
					continue

			elif width is None:
				if line[:6] == 'letter':
					width = int(line.split()[5])
					pwm = np.zeros((width, 4))

			elif i < width:
				pwm[i] = list(map(float, line.split()))
				i += 1

			else:
				motifs[motif] = pwm
				motif, width, i = None, None, 0

	return motifs


def compute_per_position_ic(ppm, background, pseudocount):
    alphabet_len = len(background)
    ic = ((np.log((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/np.log(2))
          *ppm - (np.log(background)*background/np.log(2))[None,:])
    return np.sum(ic,axis=1)


def write_meme_file(ppm, bg, fname):
	f = open(fname, 'w')
	f.write('MEME version 4\n\n')
	f.write('ALPHABET= ACGT\n\n')
	f.write('strands: + -\n\n')
	f.write('Background letter frequencies (from unknown source):\n')
	f.write('A %.3f C %.3f G %.3f T %.3f\n\n' % tuple(list(bg)))
	f.write('MOTIF 1 TEMP\n\n')
	f.write('letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n' % ppm.shape[0])
	for s in ppm:
		f.write('%.5f %.5f %.5f %.5f\n' % tuple(s))
	f.close()


def fetch_tomtom_matches(ppm, cwm, is_writing_tomtom_matrix, output_dir,
	pattern_name, motifs_db, background=[0.25, 0.25, 0.25, 0.25],
	tomtom_exec_path='tomtom', trim_threshold=0.3, trim_min_length=3):

	"""Fetches top matches from a motifs database using TomTom.
	Args:
		ppm: position probability matrix- numpy matrix of dimension (N,4)
		cwm: contribution weight matrix- numpy matrix of dimension (N,4)
		is_writing_tomtom_matrix: if True, write the tomtom matrix to a file
		output_dir: directory for writing the TOMTOM file
		pattern_name: the name of the pattern, to be used for writing to file
		background: list with ACGT background probabilities
		tomtom_exec_path: path to TomTom executable
		motifs_db: path to motifs database in meme format
		n: number of top matches to return, ordered by p-value
		temp_dir: directory for storing temp files
		trim_threshold: the ppm is trimmed from left till first position for which
			probability for any base pair >= trim_threshold. Similarly from right.
	Returns:
		list: a list of up to n results returned by tomtom, each entry is a
			dictionary with keys 'Target ID', 'p-value', 'E-value', 'q-value'
	"""

	_, fname = tempfile.mkstemp()
	_, tomtom_fname = tempfile.mkstemp()

	score = np.sum(np.abs(cwm), axis=1)
	trim_thresh = np.max(score) * trim_threshold  # Cut off anything less than 30% of max score
	pass_inds = np.where(score >= trim_thresh)[0]
	trimmed = ppm[np.min(pass_inds): np.max(pass_inds) + 1]

	# can be None of no base has prob>t
	if trimmed is None:
		return []

	# trim and prepare meme file
	write_meme_file(trimmed, background, fname)

	if not shutil.which(tomtom_exec_path):
		raise ValueError(f'`tomtom` executable could not be called globally or locally. Please install it and try again. You may install it using conda with `conda install -c bioconda meme`')

	# run tomtom
	cmd = '%s -no-ssc -oc . --verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 %s %s > %s' % (tomtom_exec_path, fname, motifs_db, tomtom_fname)
	os.system(cmd)
	tomtom_results = pandas.read_csv(tomtom_fname, sep="\t", usecols=(1, 5))

	os.system('rm ' + fname)
	if is_writing_tomtom_matrix:
		output_subdir = os.path.join(output_dir, "tomtom")
		os.makedirs(output_subdir, exist_ok=True)
		output_filepath = os.path.join(output_subdir, f"{pattern_name}.tomtom.tsv")
		os.system(f'mv {tomtom_fname} {output_filepath}')
	else:
		os.system('rm ' + tomtom_fname)
	return tomtom_results


def generate_tomtom_dataframe(modisco_h5py: os.PathLike,
		output_dir: os.PathLike, meme_motif_db: Union[os.PathLike, None],
		is_writing_tomtom_matrix: bool, pattern_groups: List[str], 
		top_n_matches=3, tomtom_exec: str="tomtom", trim_threshold=0.3,
		trim_min_length=3):

	tomtom_results = {}

	for i in range(top_n_matches):
		tomtom_results[f'match{i}'] = []
		tomtom_results[f'qval{i}'] = []

	with h5py.File(modisco_h5py, 'r') as modisco_results:
		for contribution_dir_name in pattern_groups:
			if contribution_dir_name not in modisco_results.keys():
				continue

			metacluster = modisco_results[contribution_dir_name]
			key = lambda x: int(x[0].split("_")[-1])

			for idx, (_, pattern) in enumerate(sorted(metacluster.items(), key=key)):
   				# Rest of your code goes here

				ppm = np.array(pattern['sequence'][:])
				cwm = np.array(pattern["contrib_scores"][:])

				pattern_name = f'{contribution_dir_name}.pattern_{idx}'

				r = fetch_tomtom_matches(ppm, cwm,
			     	is_writing_tomtom_matrix=is_writing_tomtom_matrix,
					output_dir=output_dir, pattern_name=pattern_name,
					motifs_db=meme_motif_db, tomtom_exec_path=tomtom_exec,
					trim_threshold=trim_threshold,
					trim_min_length=trim_min_length)

				i = -1
				for i, (target, qval) in r.iloc[:top_n_matches].iterrows():
					tomtom_results[f'match{i}'].append(target)
					tomtom_results[f'qval{i}'].append(qval)

				for j in range(i+1, top_n_matches):
					tomtom_results[f'match{j}'].append(None)
					tomtom_results[f'qval{j}'].append(None)			

	return pandas.DataFrame(tomtom_results)


def path_to_image_html(path):
	return '<img src="'+ path + '" width="240" >'

def _plot_weights(array, path, figsize=(10,3)):
	"""Plot weights as a sequence logo and save to file."""
	fig = plt.figure(figsize=figsize)
	ax = fig.add_subplot(111) 

	df = pandas.DataFrame(array, columns=['A', 'C', 'G', 'T'])
	df.index.name = 'pos'

	crp_logo = logomaker.Logo(df, ax=ax)
	crp_logo.style_spines(visible=False)
	plt.ylim(min(df.sum(axis=1).min(), 0), df.sum(axis=1).max())

	plt.savefig(path)
	plt.close()
	
def make_logo(match, logo_dir, motifs):
	if match == 'NA':
		return

	background = np.array([0.25, 0.25, 0.25, 0.25])
	ppm = motifs[match]
	ic = compute_per_position_ic(ppm, background, 0.001)

	_plot_weights(ppm*ic[:, None], path='{}/{}.png'.format(logo_dir, match))
		

def create_modisco_logos(modisco_h5py: os.PathLike, modisco_logo_dir, trim_threshold, pattern_groups: List[str]):
	"""Open a modisco results file and create and write logos to file for each pattern."""
	modisco_results = h5py.File(modisco_h5py, 'r')

	tags = []

	for name in pattern_groups:
		if name not in modisco_results.keys():
			continue

		metacluster = modisco_results[name]
		key = lambda x: int(x[0].split("_")[-1])
		for pattern_name, pattern in sorted(metacluster.items(), key=key):
			tag = '{}.{}'.format(name, pattern_name)
			tags.append(tag)

			cwm_fwd = np.array(pattern['contrib_scores'][:])
			cwm_rev = cwm_fwd[::-1, ::-1]

			score_fwd = np.sum(np.abs(cwm_fwd), axis=1)
			score_rev = np.sum(np.abs(cwm_rev), axis=1)

			trim_thresh_fwd = np.max(score_fwd) * trim_threshold
			trim_thresh_rev = np.max(score_rev) * trim_threshold

			pass_inds_fwd = np.where(score_fwd >= trim_thresh_fwd)[0]
			pass_inds_rev = np.where(score_rev >= trim_thresh_rev)[0]

			start_fwd, end_fwd = max(np.min(pass_inds_fwd) - 4, 0), min(np.max(pass_inds_fwd) + 4 + 1, len(score_fwd) + 1)
			start_rev, end_rev = max(np.min(pass_inds_rev) - 4, 0), min(np.max(pass_inds_rev) + 4 + 1, len(score_rev) + 1)

			trimmed_cwm_fwd = cwm_fwd[start_fwd:end_fwd]
			trimmed_cwm_rev = cwm_rev[start_rev:end_rev]

			_plot_weights(trimmed_cwm_fwd, path='{}/{}.cwm.fwd.png'.format(modisco_logo_dir, tag))
			_plot_weights(trimmed_cwm_rev, path='{}/{}.cwm.rev.png'.format(modisco_logo_dir, tag))

	modisco_results.close()
	return tags

def report_motifs(modisco_h5py: Path, output_dir: os.PathLike, img_path_suffix: os.PathLike, 
	meme_motif_db: Union[os.PathLike, None], is_writing_tomtom_matrix: bool, top_n_matches=3,
	trim_threshold=0.3, trim_min_length=3):

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	modisco_logo_dir = os.path.join(output_dir, 'trimmed_logos')
	if not os.path.isdir(modisco_logo_dir):
		os.mkdir(modisco_logo_dir)

	pattern_groups = ['pos_patterns', 'neg_patterns']

	create_modisco_logos(modisco_h5py, modisco_logo_dir, trim_threshold, pattern_groups)

	results = {'pattern': [], 'num_seqlets': [], 'modisco_cwm_fwd': [], 'modisco_cwm_rev': []}

	with h5py.File(modisco_h5py, 'r') as modisco_results:
		for name in pattern_groups:
			if name not in modisco_results.keys():
				continue

			metacluster = modisco_results[name]
			key = lambda x: int(x[0].split("_")[-1])
			for pattern_name, pattern in sorted(metacluster.items(), key=key):
				num_seqlets = pattern['seqlets']['n_seqlets'][:][0]
				pattern_tag = f'{name}.{pattern_name}'

				results['pattern'].append(pattern_tag)
				results['num_seqlets'].append(num_seqlets)
				results['modisco_cwm_fwd'].append(os.path.join(img_path_suffix, 'trimmed_logos', f'{pattern_tag}.cwm.fwd.png'))
				results['modisco_cwm_rev'].append(os.path.join(img_path_suffix, 'trimmed_logos', f'{pattern_tag}.cwm.rev.png'))

	patterns_df = pd.DataFrame(results)
	reordered_columns = ['pattern', 'num_seqlets', 'modisco_cwm_fwd', 'modisco_cwm_rev']

	# If the optional meme_motif_db is not provided, then we won't generate TOMTOM comparison.
	if meme_motif_db is not None:
		motifs = read_meme(meme_motif_db)

		tomtom_df = generate_tomtom_dataframe(modisco_h5py, output_dir, meme_motif_db,
			is_writing_tomtom_matrix,
			top_n_matches=top_n_matches, tomtom_exec='tomtom', 
			pattern_groups=pattern_groups, trim_threshold=trim_threshold,
			trim_min_length=trim_min_length)
		patterns_df = pandas.concat([patterns_df, tomtom_df], axis=1)

		for i in range(top_n_matches):
			name = f'match{i}'
			logos = []

			for _, row in patterns_df.iterrows():
				if name in patterns_df.columns:
					if pandas.isnull(row[name]):
						logos.append("NA")
					else:
						make_logo(row[name], output_dir, motifs)
						logos.append(f'{img_path_suffix}{row[name]}.png')
				else:
					break

			patterns_df[f"{name}_logo"] = logos
			reordered_columns.extend([name, f'qval{i}', f'{name}_logo'])

	patterns_df = patterns_df[reordered_columns]
	patterns_df.to_html(open(os.path.join(output_dir, 'motifs.html'), 'w'),
		escape=False, formatters=dict(modisco_cwm_fwd=path_to_image_html,
			modisco_cwm_rev=path_to_image_html, match0_logo=path_to_image_html,
			match1_logo=path_to_image_html, match2_logo=path_to_image_html), 
		index=False)
