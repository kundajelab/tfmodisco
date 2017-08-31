from __future__ import division, print_function, absolute_import
import numpy as np
from . import affinitymat
from . import core


class AbstractSeqletsAggregator(object):

    def aggregate_seqlets(self, seqlets):
        raise NotImplementedError()


class AbstractPatternAligner(self):

    def __init__(self, track_names, normalizer):
        self.track_names = track_names
        self.normalizer = normalizer

    def __call__(self, parent_pattern, child_pattern):
        #return an alignment
        raise NotImplementedError()     


class CrossCorrelationPatternAligner(AbstractPatternAligner):

    def __init__(self, pattern_crosscorr_settings):
        self.pattern_crosscorr_settings = pattern_crosscorr_settings

    def __call__(self, parent_pattern, child_pattern):
        fwd_data_parent, rev_data_parent = core.get_2d_data_from_seqlets(
            seqlet=parent_pattern,
            track_names=self.pattern_crosscorr_settings.track_names,
            normalizer=self.pattern_crosscorr_settings.normalizer) 
        fwd_data_child, rev_data_child = core.get_2d_data_from_seqlets(
            seqlet=child_pattern,
            track_names=self.pattern_crosscorr_settings.track_names,
            normalizer=self.pattern_crosscorr_settings.normalizer) 
        #find optimal alignments of fwd_data_child and rev_data_child
        #with fwd_data_parent.
        best_crosscorr, best_crosscorr_argmax =\
            core.get_best_alignment_crosscorr(
                parent_matrix=fwd_data_parent,
                child_matrix=fwd_data_child,
                min_overlap=self.pattern_crosscorr_settings.min_overlap)  
        best_crosscorr_rev, best_crosscorr_argmax_rev =\
            core.get_best_alignment_crosscorr(
                parent_matrix=fwd_data_parent,
                child_matrix=rev_data_child,
                min_overlap=self.pattern_crosscorr_settings.min_overlap) 
        if (best_crosscorr_rev > best_crosscorr):
            return (best_crosscorr_argmax_rev, True)
        else:
            return (best_crosscorr_argmax, False)


class HierarchicalSeqletAggregator(object):

    def __init__(self, pattern_aligner, affinity_mat_from_seqlets):
        self.pattern_aligner = pattern_aligner
        self.affinity_mat_from_seqlets = affinity_mat_from_seqlets

    def aggregate_seqlets(self, seqlets):
        affinity_mat = self.affinity_mat_from_seqlets(seqlets)
        self.aggregate_seqlets_by_affinity_mat(seqlets=seqlets,
                                               affinity_mat=affinity_mat)

    def aggregate_seqlets_by_affinity_mat(self, seqlets, affinity_mat):

        aggregated_seqlets = [AggregatedSeqlet.from_seqlet(x) for x in seqlets]
        #get the affinity mat as a list of 3-tuples
        affinity_tuples = []
        for i in range(len(affinity_mat)-1):
            for j in range(i+1,len(affinity_mat)):
                affinity_tuples.append((affinity_mat[i,j],i,j))
        #sort to get closest first
        affinity_tuples = sorted(affinity_tuples, lambda x: -x[0])

        #now repeatedly merge, unless already merged.
        for (affinity,i,j) in affinity_tuples:
            aggregated_seqlet_i = aggregate_seqlets[i]
            aggregated_seqlet_j = aggregate_seqlets[j]
            #if they are not already the same aggregation object...
            if (aggregated_seqlet_i != aggregated_seqlet_j):
                if (aggregated_seqlet_i.num_seqlets <
                    aggregated_seqlet_j.num_seqlets):
                    parent_agg_seqlet = aggregated_seqlet_j 
                    child_agg_seqlet = aggregated_seqlet_i
                else:
                    parent_agg_seqlet = aggregated_seqlet_i
                    child_agg_seqlet = aggregated_seqlet_j
                parent_agg_seqlet.add_seqlet(
                    seqlet=child_pattern,
                    aligner=self.motif_aligner)
                aggregate_seqlets[i] = parent_agg_seqlet 
                aggregate_seqlets[j] = child_agg_seqlet
 
        
class SeqletAndAlignment(object):

    def __init__(self, seqlet, alnmt):
        self.seqlet = seqlet
        #alnmt is the position of the beginning of seqlet
        #in the aggregated seqlet
        self.alnmt = alnmt 

