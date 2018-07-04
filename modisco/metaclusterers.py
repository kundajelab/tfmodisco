from __future__ import division, print_function, absolute_import
import sys
import itertools
from collections import defaultdict
import numpy as np
from . import util


class MetaclusteringResults(object):

    def __init__(self, metacluster_indices,
                       attribute_vectors,
                       metacluster_idx_to_activity_pattern):
        self.metacluster_indices = metacluster_indices
        self.attribute_vectors = attribute_vectors
        self.metacluster_idx_to_activity_pattern =\
                metacluster_idx_to_activity_pattern

    def save_hdf5(self, grp):
        grp.create_dataset("metacluster_indices",
                           data=self.metacluster_indices)
        grp.create_dataset("attribute_vectors",
                           data=np.array(self.attribute_vectors))
        metacluster_idx_to_activity_pattern_grp =\
            grp.create_group("metacluster_idx_to_activity_pattern")
        all_metacluster_names = []
        for cluster_idx,activity_pattern in\
            self.metacluster_idx_to_activity_pattern.items():
            metacluster_name = "metacluster_"+str(cluster_idx)
            metacluster_idx_to_activity_pattern_grp.attrs[
                metacluster_name] = activity_pattern
            all_metacluster_names.append(metacluster_name)
        util.save_string_list(all_metacluster_names,
                              dset_name="all_metacluster_names",
                              grp=grp) 
         


class AbstractMetaclusterer(object):

    def __call__(self, attribute_vectors):
        raise NotImplementedError()


class SignBasedPatternClustering(AbstractMetaclusterer):
    
    def __init__(self, min_cluster_size, threshold_for_counting_sign,
                 weak_threshold_for_counting_sign, verbose=True):
        self.min_cluster_size = min_cluster_size
        self.threshold_for_counting_sign = threshold_for_counting_sign
        self.weak_threshold_for_counting_sign =\
             weak_threshold_for_counting_sign
        self.verbose = verbose
        
    def pattern_to_str(self, pattern):
        return ",".join([str(x) for x in pattern])
    
    def vector_to_pattern(self, vector):
        to_return = np.array([
                0 if np.abs(element) < self.threshold_for_counting_sign
                  else (1 if element > 0 else -1)
                  for element in vector])
        if (to_return[0]==0 and to_return[1]==0 and to_return[2]==0):
            print(vector)
            print(to_return)
            assert False
        return to_return
    
    def weak_vector_to_pattern(self, vector):
        to_return = np.array([
                0 if np.abs(element) < self.weak_threshold_for_counting_sign
                  else (1 if element > 0 else -1) for element in vector])
        if (to_return[0]==0 and to_return[1]==0 and to_return[2]==0):
            print(vector)
            print(to_return)
            assert False
        return to_return
    
    def check_pattern_compatibility(self, pattern_to_check, reference_pattern):
        return all([(pattern_elem==reference_elem or reference_elem==0)
                    for pattern_elem, reference_elem
                    in zip(pattern_to_check, reference_pattern)])
    
    def get_compatible_patterns(self, pattern_to_check, reference_patterns):
        return [reference_pattern for reference_pattern
                in reference_patterns
                if self.check_pattern_compatibility(
                        pattern_to_check=pattern_to_check,
                        reference_pattern=reference_pattern)]
    
    def __call__(self, attribute_vectors):

        all_possible_activity_patterns =\
            list(itertools.product(*[(1,-1,0) for x
                 in range(attribute_vectors.shape[1])]))

        activity_pattern_to_attribute_vectors = defaultdict(list)        
        for vector in attribute_vectors:
            vector_activity_pattern = self.vector_to_pattern(vector)
            compatible_activity_patterns =\
                self.get_compatible_patterns(
                     vector_activity_pattern, all_possible_activity_patterns)
            for compatible_activity_pattern in compatible_activity_patterns:
                activity_pattern_to_attribute_vectors[
                    self.pattern_to_str(
                         compatible_activity_pattern)].append(vector)
        
        surviving_activity_patterns = [
            activity_pattern for activity_pattern in 
            all_possible_activity_patterns if
            (len(activity_pattern_to_attribute_vectors[
                 self.pattern_to_str(activity_pattern)])
             >= self.min_cluster_size)]
        
        activity_patterns = []
        final_activity_pattern_to_vectors = defaultdict(list)
        for vector in attribute_vectors:
            vector_pattern = self.weak_vector_to_pattern(vector) #be liberal
            compatible_activity_patterns =\
                self.get_compatible_patterns(vector_pattern,
                                             surviving_activity_patterns)
            best_pattern = self.pattern_to_str(
                max(compatible_activity_patterns,
                key=lambda x: np.sum(x*np.array(vector) )))
            activity_patterns.append(best_pattern)
            final_activity_pattern_to_vectors[best_pattern].append(vector)
            
        final_surviving_activity_patterns = set([
            self.pattern_to_str(activity_pattern) for activity_pattern in 
            all_possible_activity_patterns if
            (len(final_activity_pattern_to_vectors[
                 self.pattern_to_str(activity_pattern)])
             >= self.min_cluster_size)])

        if (self.verbose):
            print(str(len(final_surviving_activity_patterns))+
                  " activity patterns with support >= "
                  +str(self.min_cluster_size)+" out of "
                  +str(len(all_possible_activity_patterns))
                  +" possible patterns") 
        
        #sort activity patterns by most to least support
        sorted_activity_patterns = sorted(
            final_surviving_activity_patterns,
            key=lambda x: -len(final_activity_pattern_to_vectors[x]))
        activity_pattern_to_cluster_idx = dict(
            [(x[1],x[0]) for x in enumerate(sorted_activity_patterns)])
        metacluster_idx_to_activity_pattern = dict(
            [(x[0],x[1])
             for x in enumerate(sorted_activity_patterns)])

        metacluster_indices = [
             activity_pattern_to_cluster_idx[activity_pattern]
             if activity_pattern in final_surviving_activity_patterns
             else -1 for activity_pattern in activity_patterns]
        return MetaclusteringResults(
                metacluster_indices=np.array(metacluster_indices),
                attribute_vectors=attribute_vectors,
                metacluster_idx_to_activity_pattern=
                    metacluster_idx_to_activity_pattern)

