
# coding: utf-8

# In[1]:

from __future__ import print_function, division
#get_ipython().magic('matplotlib inline')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os
os.chdir("../examples/simulated_tf_binding/")
os.system('mkdir -p figures')




try:
    reload  # Python 2.7
except NameError:
    try:
        from importlib import reload  # Python 3.4+
    except ImportError:
        from imp import reload  # Python 3.0 - 3.3


# # Preliminaries
# 
# This notebook experiments incremental clustering strategies.
# 
# ## Setup
# 
# Gather all the necessary data for running TF-MoDISco

# In[2]:

import numpy as np
import modisco
import theano
print("Theano version:",theano.__version__)
import sys
print (sys.version)


# ### Grab the input data

# In[3]:

#get_ipython().system('./grab_scores_for_modisco.sh')
import os
import logging
logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')
os.system('./grab_scores_for_modisco.sh')





# ### One-hot encode the fasta sequences

# In[4]:

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

#read in the data in the testing set
test_ids_fh = gzip.open("test.txt.gz","rb")
ids_to_load = set([x.rstrip() for x in test_ids_fh])

fasta_sequences = []
for i,a_line in enumerate(gzip.open("sequences.simdata.gz","rb")):
    if (i==0):
        next
    a_line = a_line.rstrip()
    seq_id,seq_fasta,embeddings,task1,task2,task3 = a_line.split(b"\t")
    if seq_id in ids_to_load:
        fasta_sequences.append(seq_fasta.decode("utf-8"))


# ### Load importance scores into numpy arrays
# 
# You need a numpy array of importance scores and hypothetical importance scores for every task. See `Generate Importance Scores.ipynb` for how to prepare these data tracks.

# In[5]:

import h5py
from collections import OrderedDict

task_to_scores = OrderedDict()
task_to_hyp_scores = OrderedDict()

f = h5py.File("scores.h5","r")
tasks = f["contrib_scores"].keys()
for task in tasks:
    #Note that the sequences can be of variable lengths;
    #in this example they all have the same length (200bp) but that is
    #not necessary.
    task_to_scores[task] = [np.array(x) for x in f['contrib_scores'][task][:]]
    task_to_hyp_scores[task] = [np.array(x) for x in f['hyp_contrib_scores'][task][:]]

onehot_data = [one_hot_encode_along_channel_axis(seq) for seq in fasta_sequences]


def create_track_set(score_file, sequence_file, test_file):
    import gzip
    import h5py
    from collections import OrderedDict

    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()

    # f = h5py.File("scores.h5", "r")
    f = h5py.File(score_file, "r")
    tasks = f["contrib_scores"].keys()
    for task in tasks:
        # Note that the sequences can be of variable lengths;
        # in this example they all have the same length (200bp) but that is
        # not necessary.
        task_to_scores[task] = [np.array(x) for x in f['contrib_scores'][task][:]]
        task_to_hyp_scores[task] = [np.array(x) for x in f['hyp_contrib_scores'][task][:]]

    test_ids_fh = gzip.open(test_file, "rb")
    ids_to_load = set([x.rstrip() for x in test_ids_fh])

    fasta_sequences = []
    # for i, a_line in enumerate(gzip.open("sequences.simdata.gz", "rb")):
    for i, a_line in enumerate(gzip.open(sequence_file, "rb")):
        if (i == 0):
            next
        a_line = a_line.rstrip()
        seq_id, seq_fasta, embeddings, task1, task2, task3 = a_line.split(b"\t")
        if seq_id in ids_to_load:
            fasta_sequences.append(seq_fasta.decode("utf-8"))

    onehot_data = [one_hot_encode_along_channel_axis(seq) for seq in fasta_sequences]

    task_names = tasks

    ##############################
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



import motif_hit.incr_workflow

track_set = create_track_set("scores.h5", "sequences.simdata.gz", "test.txt.gz")

prior_results = motif_hit.incr_workflow.TfModiscoPriorResults("results.hdf5", track_set)

factory = modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                        trim_to_window_size=15,
                        initial_flank_to_add=5,
                        kmer_len=5, num_gaps=1,
                        num_mismatches=0,
                        final_min_cluster_size=60)

workflow = motif_hit.incr_workflow.TfModiscoIncrementalWorkflow(
                        #Slight modifications from the default settings
                        sliding_window_size=15,
                        flank_size=5,
                        target_seqlet_fdr=0.01,
                        prior_results=prior_results,
                        seqlets_to_patterns_factory=factory)

tfmodisco_results = workflow(task_names=["task0", "task1", "task2"],
                        contrib_scores=task_to_scores,
                        hypothetical_contribs=task_to_hyp_scores,
                        one_hot=onehot_data)



print("tfmodisco_results: ", tfmodisco_results)
#
# ## Save and print the results

# In[29]:

import h5py
import modisco.util
#reload(modisco.util)
#get_ipython().system('rm results.hdf5')
os.system('rm results2.hdf5')
grp = h5py.File("results2.hdf5")
#tfmodisco_results.save_hdf5(grp)


# In[30]: visualize the results

"""
from collections import Counter
from modisco.visualization import viz_sequence
#reload(viz_sequence)
from matplotlib import pyplot as plt

import modisco.affinitymat.core
#reload(modisco.affinitymat.core)
import modisco.cluster.phenograph.core
#reload(modisco.cluster.phenograph.core)
import modisco.cluster.phenograph.cluster
#reload(modisco.cluster.phenograph.cluster)
#import modisco.cluster.core
#reload(modisco.cluster.core)
import modisco.aggregator
#reload(modisco.aggregator)

hdf5_results = h5py.File("results2.hdf5","r")

print("Metaclusters heatmap")
import seaborn as sns
activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
                    np.array(
        [x[0] for x in sorted(
                enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
               key=lambda x: x[1])])]
sns.heatmap(activity_patterns, center=0)
plt.show()

metacluster_names = [
    x.decode("utf-8") for x in 
    list(hdf5_results["metaclustering_results"]
         ["all_metacluster_names"][:])]

all_patterns = []

for metacluster_name in metacluster_names:
    print(metacluster_name)
    metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                   [metacluster_name])
    print("activity pattern:",metacluster_grp["activity_pattern"][:])
    all_pattern_names = [x.decode("utf-8") for x in 
                         list(metacluster_grp["seqlets_to_patterns_result"]
                                             ["patterns"]["all_pattern_names"][:])]
    if (len(all_pattern_names)==0):
        print("No motifs found for this activity pattern")
    for pattern_name in all_pattern_names:
        print(metacluster_name, pattern_name)
        all_patterns.append((metacluster_name, pattern_name))
        pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
        print("total seqlets:",len(pattern["seqlets_and_alnmts"]["seqlets"]))
        background = np.array([0.27, 0.23, 0.23, 0.27])
        print("Task 0 hypothetical scores:")
        viz_sequence.plot_weights(pattern["task0_hypothetical_contribs"]["fwd"])
        print("Task 0 actual importance scores:")
        viz_sequence.plot_weights(pattern["task0_contrib_scores"]["fwd"])
        print("Task 1 hypothetical scores:")
        viz_sequence.plot_weights(pattern["task1_hypothetical_contribs"]["fwd"])
        print("Task 1 actual importance scores:")
        viz_sequence.plot_weights(pattern["task1_contrib_scores"]["fwd"])
        print("Task 2 hypothetical scores:")
        viz_sequence.plot_weights(pattern["task2_hypothetical_contribs"]["fwd"])
        print("Task 2 actual importance scores:")
        viz_sequence.plot_weights(pattern["task2_contrib_scores"]["fwd"])
        print("onehot, fwd and rev:")
        viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                        background=background)) 
        viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
                                                        background=background)) 
        
hdf5_results.close()

"""
