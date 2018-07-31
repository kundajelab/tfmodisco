from __future__ import division, print_function, absolute_import
import sys
import itertools
from collections import defaultdict
import numpy as np
from modisco import metaclusterers

def map_seqlet_to_metacluster(seqlet_to_metacluster, attribute_vectors,
                              possible_activity_patterns = None):
    '''
        Based on metaclusterers.SignBasedPatternClustering.__call__(), TODO: could merge later.

    Args:
        seqlet_to_metacluster: instance of SignBasedPatternClustering
        attribute_vectors:     array of activity patterns, one row per seqlet
        possible_activity_patterns: list of existing activity patterns from previous clustering

    Returns:
        instance of metaclusterers.MetaclusteringResults

    '''

    if possible_activity_patterns == None:
        possible_activity_patterns = \
            list(itertools.product(*[(1, -1, 0) for x
                                     in range(attribute_vectors.shape[1])]))

    activity_pattern_to_attribute_vectors = defaultdict(list)
    for vector in attribute_vectors:
        vector_activity_pattern = seqlet_to_metacluster.vector_to_pattern(vector)
        compatible_activity_patterns = \
            seqlet_to_metacluster.get_compatible_patterns(
                vector_activity_pattern, possible_activity_patterns)
        for compatible_activity_pattern in compatible_activity_patterns:
            activity_pattern_to_attribute_vectors[
                seqlet_to_metacluster.pattern_to_str(
                    compatible_activity_pattern)].append(vector)

    surviving_activity_patterns = [
        activity_pattern for activity_pattern in
        possible_activity_patterns if
        (len(activity_pattern_to_attribute_vectors[
                 seqlet_to_metacluster.pattern_to_str(activity_pattern)])
         >= seqlet_to_metacluster.min_cluster_size)]

    orphans = []
    activity_patterns = []
    final_activity_pattern_to_vectors = defaultdict(list)
    for vector in attribute_vectors:
        vector_pattern = seqlet_to_metacluster.weak_vector_to_pattern(vector)  # be liberal
        compatible_activity_patterns = \
            seqlet_to_metacluster.get_compatible_patterns(vector_pattern,
                                         surviving_activity_patterns)
        if len(compatible_activity_patterns) != 0:
            best_pattern = seqlet_to_metacluster.pattern_to_str(
                max(compatible_activity_patterns,
                    key=lambda x: np.sum(x * np.array(vector))))
            activity_patterns.append(best_pattern)
            final_activity_pattern_to_vectors[best_pattern].append(vector)
        else: # this vector/seqlet does not match to any patterns
            orphans.append(vector)
            activity_patterns.append('-,-,-') # some invalid pattern
            # TODO: we should re-assign them to other pattern/clusters

    final_surviving_activity_patterns = set([
        seqlet_to_metacluster.pattern_to_str(activity_pattern) for activity_pattern in
        possible_activity_patterns if
        (len(final_activity_pattern_to_vectors[
                 seqlet_to_metacluster.pattern_to_str(activity_pattern)])
         >= seqlet_to_metacluster.min_cluster_size)])

    if (seqlet_to_metacluster.verbose):
        print(str(len(final_surviving_activity_patterns)) +
              " activity patterns with support >= "
              + str(seqlet_to_metacluster.min_cluster_size) + " out of "
              + str(len(possible_activity_patterns))
              + " possible patterns")
        print(str(len(orphans)) + " seqlets did not match to any metaclusters")
        print(orphans)
        # sort activity patterns by most to least support

    sorted_activity_patterns = sorted(
        final_surviving_activity_patterns,
        key=lambda x: -len(final_activity_pattern_to_vectors[x]))
    activity_pattern_to_cluster_idx = dict(
        [(x[1], x[0]) for x in enumerate(sorted_activity_patterns)])
    metacluster_idx_to_activity_pattern = dict(
        [(x[0], x[1])
         for x in enumerate(sorted_activity_patterns)])

    metacluster_indices = [
        activity_pattern_to_cluster_idx[activity_pattern]
        if activity_pattern in final_surviving_activity_patterns
        else -1 for activity_pattern in activity_patterns]
    return metaclusterers.MetaclusteringResults(
        metacluster_indices=np.array(metacluster_indices),
        attribute_vectors=attribute_vectors,
        metacluster_idx_to_activity_pattern=
            metacluster_idx_to_activity_pattern)


