#!/usr/bin/env python
from __future__ import print_function, division


import os
import matplotlib as mpl
mpl.use('Agg')

import numpy as np
import modisco
import theano
import sys

import logging
logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')

logging.info(" ".join(sys.argv))

logging.debug("Theano version:" + str(theano.__version__))
logging.debug(sys.version)


# ### Functions for one-hot encoding sequences

import gzip

def one_hot_encode_along_channel_axis(sequence):
    #theano dim ordering, uses row axis for one-hot
    to_return = np.zeros((len(sequence),4), dtype=np.int8)
    seq_to_one_hot_fill_in_array(zeros_array=to_return,
                                 sequence=sequence, one_hot_axis=1)
    return to_return

def seq_to_one_hot_fill_in_array(zeros_array, sequence, one_hot_axis):
    assert one_hot_axis==0 or one_hot_axis==1
    if (one_hot_axis==0):
        assert zeros_array.shape[1] == len(sequence)
    elif (one_hot_axis==1): 
        assert zeros_array.shape[0] == len(sequence)
    #will mutate zeros_array
    for (i,char) in enumerate(sequence):
        if (char=="A" or char=="a"):
            char_idx = 0
        elif (char=="C" or char=="c"):
            char_idx = 1
        elif (char=="G" or char=="g"):
            char_idx = 2
        elif (char=="T" or char=="t"):
            char_idx = 3
        elif (char=="N" or char=="n"):
            continue #leave that pos as all 0's
        else:
            raise RuntimeError("Unsupported character: "+str(char))
        if (one_hot_axis==0):
            zeros_array[char_idx,i] = 1
        elif (one_hot_axis==1):
            zeros_array[i,char_idx] = 1

from merge_overlaps import MergeOverlaps
from merge_overlaps import merge_overlaps

#do_test()
#quit()

import sys
if len(sys.argv) != 6:
    print("Syntax: ", sys.argv[0] , " <score prefix> <sequence fa file> <sequence tsv> <number of tasks> <results.hdf5> ")
    quit()

score_prefix = sys.argv[1]                 # "./scores/hyp_scores_task_"
input_name   = sys.argv[2]                 # subset.fa, sequences 
input_tsv    = sys.argv[3]                 # subset.tsv
num_tasks    = int(sys.argv[4])            #
results_fn   = sys.argv[5]

logging.debug("method file prefix is %s, input seq file is %s, input tsv is %s, number of tasks is %d", 
              score_prefix, input_name, input_tsv, num_tasks)

#https://www.biostars.org/p/710/
from itertools import groupby
def fasta_iter(fasta_name):
    """
        given a fasta file, yield tuples of (header, sequence)
    """
    fh = open(fasta_name) # file handle
    # ditch the boolean (x[0]) and just keep the header or sequence since they alternate
    fa_iter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in fa_iter:
        header = header.next()[1:].strip() # drop the ">" from the header
        seq = "".join(s.strip() for s in fa_iter.next()) # join all sequence lines to one
        yield header, seq

fasta_sequences = []
fasta = fasta_iter(input_name)

for header, seq in fasta:   
    fasta_sequences.append(seq)
logging.debug("lenth of sequences = %d", len(fasta_sequences))

#onehot_data = [one_hot_encode_along_channel_axis(seq) for seq in fasta_sequences]
#logging.debug("shape of onehot" + str(onehot_data[0].shape))

# ## Prepare the data for input into TF-MoDISCo
# 
# You need a numpy array of importance scores and hypothetical importance scores for every task.

from collections import OrderedDict

task_to_scores = OrderedDict()
task_to_hyp_scores = OrderedDict()

# locations of deeplift scores
scores_loc = []
task_names = []
for i in range(num_tasks):
    loc_i = score_prefix + str(i) + ".npy"
    scores_loc.append(loc_i)
    task_names.append("task" + str(i))

# scores & their one-hot encodings

merged_seq_list        = []
merged_onehot_list     = []
for t in range(num_tasks):
    merged_hyp_scores_list     = []
    merged_contrib_scores_list = []

    task = task_names[t]
    hyp_scores_all = np.load(scores_loc[t])
    merge_overlaps(input_tsv, hyp_scores_all, merged_hyp_scores_list, fasta_sequences,
                   merged_seq_list = merged_seq_list if t==0 else None)

    for i in range(len(merged_hyp_scores_list)):
        onehot_seq = one_hot_encode_along_channel_axis(merged_seq_list[i])
        contrib_scores = merged_hyp_scores_list[i] * onehot_seq
        merged_contrib_scores_list.append(contrib_scores)
        if t == 0:
            merged_onehot_list.append(onehot_seq)

    task_to_hyp_scores[task] = merged_hyp_scores_list
    task_to_scores[task]     = merged_contrib_scores_list

    logging.debug("shape of hyp_score " + str(task_to_hyp_scores['task0'][0].shape))
    logging.debug("shape of score " + str(task_to_scores['task0'][0].shape))


def create_track_set(task_to_hyp_scores, task_to_scores, onehot_data):
    from modisco import core
    contrib_scores_tracks = [core.DataTrack(name=key + "_contrib_scores",
                                            fwd_tracks=task_to_scores[key],
                                            rev_tracks=[x[::-1, ::-1] for x in
                                                        task_to_scores[key]],
                                            has_pos_axis=True) for key in task_names]
    hypothetical_contribs_tracks = [core.DataTrack(name=key + "_hypothetical_contribs",
                                                   fwd_tracks=task_to_hyp_scores[key],
                                                   rev_tracks=[x[::-1, ::-1] for x in
                                                               task_to_hyp_scores[key]],
                                                   has_pos_axis=True) for key in task_names]
    onehot_track = core.DataTrack(name="sequence", fwd_tracks=onehot_data,
                                  rev_tracks=[x[::-1, ::-1] for x in onehot_data],
                                  has_pos_axis=True)
    track_set = core.TrackSet(data_tracks=contrib_scores_tracks
                                          + hypothetical_contribs_tracks + [onehot_track])
    return track_set

import h5py
import numpy as np
import modisco

import motif_hit.incr_workflow

track_set = create_track_set(task_to_hyp_scores, task_to_scores, merged_onehot_list)

prior_results = motif_hit.incr_workflow.TfModiscoPriorResults(results_fn, track_set)

factory = modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                        trim_to_window_size=30,
                        initial_flank_to_add=10,
                        kmer_len=8, num_gaps=3,
                        num_mismatches=2,
                        final_min_cluster_size=30)

workflow = motif_hit.incr_workflow.TfModiscoIncrementalWorkflow(
                        sliding_window_size=21,
                        flank_size=10,
                        target_seqlet_fdr=0.01,
                        seqlets_to_patterns_factory=factory,
                        prior_results=prior_results)  # new argument for incremental clustering

tfmodisco_results = workflow(task_names=task_names,
                        contrib_scores=task_to_scores,
                        hypothetical_contribs=task_to_hyp_scores,
                        one_hot=merged_onehot_list)


"""
#Slight modifications from the default settings
factory = modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
              trim_to_window_size=15,
              initial_flank_to_add=5,
              kmer_len=5, num_gaps=1,
              num_mismatches=0,
              final_min_cluster_size=60)

tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                            sliding_window_size=15,
                            flank_size=5,
                            target_seqlet_fdr=0.01,
                            seqlets_to_patterns_factory=factory
                        )(
                            task_names=task_names,
                            contrib_scores        = task_to_scores,
                            hypothetical_contribs = task_to_hyp_scores,
                            one_hot=merged_onehot_list)
"""

logging.debug("**************** workflow done *********************")

# ## Save and print the results

# In[8]:

import h5py
import modisco.util
#reload(modisco.util)
os.system('rm -f results2.hdf5')
grp = h5py.File("results2.hdf5")
#tfmodisco_results.save_hdf5(grp)

logging.debug("**************** result saved *********************")



