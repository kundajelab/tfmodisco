from __future__ import division, print_function, absolute_import
import numpy as np
from . import affinitymat
from . import core


class AbstractSeqletsAggregator(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class HierarchicalSeqletAggregator(object):

    def __init__(self, pattern_aligner, affinity_mat_from_seqlets):
        self.pattern_aligner = pattern_aligner
        self.affinity_mat_from_seqlets = affinity_mat_from_seqlets

    def __call__(self, seqlets):
        affinity_mat = self.affinity_mat_from_seqlets(seqlets)
        return self.aggregate_seqlets_by_affinity_mat(seqlets=seqlets,
                                                      affinity_mat=affinity_mat)

    def aggregate_seqlets_by_affinity_mat(self, seqlets, affinity_mat):

        aggregated_seqlets = [core.AggregatedSeqlet.from_seqlet(x)
                              for x in seqlets]
        #get the affinity mat as a list of 3-tuples
        affinity_tuples = []
        for i in range(len(affinity_mat)-1):
            for j in range(i+1,len(affinity_mat)):
                affinity_tuples.append((affinity_mat[i,j],i,j))
        #sort to get closest first
        affinity_tuples = sorted(affinity_tuples, key=lambda x: -x[0])

        #now repeatedly merge, unless already merged.
        for (affinity,i,j) in affinity_tuples:
            aggregated_seqlet_i = aggregated_seqlets[i]
            aggregated_seqlet_j = aggregated_seqlets[j]
            #if they are not already the same aggregation object...
            if (aggregated_seqlet_i != aggregated_seqlet_j):
                if (aggregated_seqlet_i.num_seqlets <
                    aggregated_seqlet_j.num_seqlets):
                    parent_agg_seqlet = aggregated_seqlet_j 
                    child_agg_seqlet = aggregated_seqlet_i
                else:
                    parent_agg_seqlet = aggregated_seqlet_i
                    child_agg_seqlet = aggregated_seqlet_j
                parent_agg_seqlet.add_pattern(
                    pattern=child_agg_seqlet,
                    aligner=self.pattern_aligner)
                aggregated_seqlets[i] = parent_agg_seqlet 
                aggregated_seqlets[j] = parent_agg_seqlet

        return sorted(list(set(aggregated_seqlets)),
                      key=lambda x: -x.num_seqlets)

