from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import unittest
from unittest import skip
import sys
import os
import numpy as np
import gzip
import h5py
from collections import OrderedDict


def one_hot_encode_along_channel_axis(sequence):
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


class TestTfmodiscoWorkflow(unittest.TestCase):

    def setUp(self):

        try:
            reload  # Python 2.7
        except NameError:
            try:
                from importlib import reload  # Python 3.4+
            except ImportError:
                from imp import reload  # Python 3.0 - 3.3

        if (os.path.isfile("scores.h5")==False):
            os.system("curl -o scores.h5 https://raw.githubusercontent.com/AvantiShri/model_storage/23d8f3ffc89af210f6f0bf7e65585eff259ba672/modisco/scores.h5")
        if (os.path.isfile("sequences.simdata.gz")==False):
            os.system("wget https://raw.githubusercontent.com/AvantiShri/model_storage/db919b12f750e5844402153233249bb3d24e9e9a/deeplift/genomics/sequences.simdata.gz -O sequences.simdata.gz")
        if (os.path.isfile("test.txt.gz")==False):
            os.system("wget https://raw.githubusercontent.com/AvantiShri/model_storage/9aadb769735c60eb90f7d3d896632ac749a1bdd2/deeplift/genomics/test.txt.gz -O test.txt.gz")
        
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

        task_to_scores = OrderedDict()
        task_to_hyp_scores = OrderedDict()
        f = h5py.File("scores.h5","r")
        tasks = f["contrib_scores"].keys()
        n = 100 #since this is just a test run, for speed I am limiting to 100 sequences
        for task in tasks:
            #Note that the sequences can be of variable lengths;
            #in this example they all have the same length (200bp) but that is
            #not necessary.
            task_to_scores[task] = [np.array(x) for x in f['contrib_scores'][task][:n]]
            task_to_hyp_scores[task] = [np.array(x) for x in f['hyp_contrib_scores'][task][:n]]

        f.close()

        onehot_data = [one_hot_encode_along_channel_axis(seq)
                       for seq in fasta_sequences][:n]

        self.onehot_data = onehot_data
        self.task_to_scores = task_to_scores
        self.task_to_hyp_scores = task_to_hyp_scores

    #@skip
    def test_base_workflow(self): 

        onehot_data = self.onehot_data
        task_to_scores = self.task_to_scores
        task_to_hyp_scores = self.task_to_hyp_scores

        import modisco
        null_per_pos_scores = (modisco.coordproducers
                               .LaplaceNullDist(num_to_samp=5000))
        tfmodisco_results = (modisco.tfmodisco_workflow
                                    .workflow.TfModiscoWorkflow(
                #Slight modifications from the default settings
                sliding_window_size=15,
                flank_size=5,
                target_seqlet_fdr=0.15,
                seqlets_to_patterns_factory=
                 modisco.tfmodisco_workflow
                  .seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                    trim_to_window_size=15,
                    initial_flank_to_add=5,
                    kmer_len=5, num_gaps=1,
                    num_mismatches=0,
                    final_min_cluster_size=60)
            )(
             task_names=["task0", "task1", "task2"],
             contrib_scores=task_to_scores,
             hypothetical_contribs=task_to_hyp_scores,
             one_hot=onehot_data,
             null_per_pos_scores = null_per_pos_scores,
             plot_save_dir="plot_save_directory"))

    #@skip
    def test_memeinit_workflow(self): 

        onehot_data = self.onehot_data
        task_to_scores = self.task_to_scores
        task_to_hyp_scores = self.task_to_hyp_scores

        import modisco
        null_per_pos_scores = (modisco.coordproducers
                               .LaplaceNullDist(num_to_samp=5000))
        tfmodisco_results = (modisco.tfmodisco_workflow
            .workflow.TfModiscoWorkflow(
                #Slight modifications from the default settings
                sliding_window_size=15,
                flank_size=5,
                target_seqlet_fdr=0.15,
                seqlets_to_patterns_factory=
                 modisco.tfmodisco_workflow
                  .seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                    initclusterer_factory=
                      modisco.clusterinit.memeinit.MemeInitClustererFactory(
                        meme_command="meme", base_outdir="meme_out",
                        max_num_seqlets_to_use=10000, nmotifs=3,
                        n_jobs=4),
                    trim_to_window_size=15,
                    initial_flank_to_add=5,
                    kmer_len=5, num_gaps=1,
                    num_mismatches=0,
                    final_min_cluster_size=60)
            )(
             task_names=["task0", "task1", "task2"],
             contrib_scores=task_to_scores,
             hypothetical_contribs=task_to_hyp_scores,
             one_hot=onehot_data,
             null_per_pos_scores = null_per_pos_scores,
             plot_save_dir="plot_save_directory"))

    #@skip
    def test_norevcomp_memeinit_workflow(self): 

        onehot_data = self.onehot_data
        task_to_scores = self.task_to_scores
        task_to_hyp_scores = self.task_to_hyp_scores

        import modisco
        null_per_pos_scores = (modisco.coordproducers
                               .LaplaceNullDist(num_to_samp=5000))
        tfmodisco_results = (modisco.tfmodisco_workflow
            .workflow.TfModiscoWorkflow(
                #Slight modifications from the default settings
                sliding_window_size=15,
                flank_size=5,
                target_seqlet_fdr=0.15,
                seqlets_to_patterns_factory=
                 modisco.tfmodisco_workflow
                  .seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                    initclusterer_factory=
                      modisco.clusterinit.memeinit.MemeInitClustererFactory(
                        meme_command="meme", base_outdir="meme_out",
                        max_num_seqlets_to_use=10000, nmotifs=3,
                        n_jobs=4),
                    trim_to_window_size=15,
                    initial_flank_to_add=5,
                    kmer_len=5, num_gaps=1,
                    num_mismatches=0,
                    final_min_cluster_size=60)
            )(
             task_names=["task0", "task1", "task2"],
             contrib_scores=task_to_scores,
             hypothetical_contribs=task_to_hyp_scores,
             one_hot=onehot_data,
             null_per_pos_scores = null_per_pos_scores,
             plot_save_dir="plot_save_directory",
             revcomp=False))

    #@skip
    def test_varseqlen_agkm_workflow(self): 

        onehot_data = self.onehot_data
        task_to_scores = self.task_to_scores
        task_to_hyp_scores = self.task_to_hyp_scores

        import modisco
        null_per_pos_scores = (modisco.coordproducers
                               .LaplaceNullDist(num_to_samp=5000))
        tfmodisco_results = (modisco.tfmodisco_workflow
                                    .workflow.TfModiscoWorkflow(
             #Slight modifications from the default settings
             sliding_window_size=[5,9,13,17],
             flank_size=5,
             target_seqlet_fdr=0.15,
             seqlets_to_patterns_factory=
              modisco.tfmodisco_workflow
               .seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
                 trim_to_window_size=15,
                 initial_flank_to_add=5,
                 embedder_factory=(modisco.seqlet_embedding
                  .advanced_gapped_kmer.AdvancedGappedKmerEmbedderFactory()),
                 final_min_cluster_size=60)
            )(
             task_names=["task0", "task1", "task2"],
             contrib_scores=task_to_scores,
             hypothetical_contribs=task_to_hyp_scores,
             one_hot=onehot_data,
             null_per_pos_scores = null_per_pos_scores,
             plot_save_dir="plot_save_directory"))
