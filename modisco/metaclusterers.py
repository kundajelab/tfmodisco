from __future__ import division, print_function, absolute_import
import sys
import itertools
from collections import defaultdict, OrderedDict
import numpy as np
from . import util
from . import value_provider


class MetaclusteringResults(object):

    def __init__(self, metacluster_indices,
                       metaclusterer,
                       attribute_vectors,
                       metacluster_idx_to_activity_pattern):
        self.metacluster_indices = metacluster_indices
        self.metaclusterer = metaclusterer
        self.attribute_vectors = attribute_vectors
        self.metacluster_idx_to_activity_pattern =\
                metacluster_idx_to_activity_pattern

    @classmethod
    def from_hdf5(cls, grp):
        metacluster_indices = np.array(grp["metacluster_indices"])
        metaclusterer = AbstractMetaclusterer.from_hdf5(grp) 
        attribute_vectors = np.array(grp["attribute_vectors"])
        all_metacluster_names = util.load_string_list(
            dset_name="all_metacluster_names",
            grp=grp) 
        metacluster_idx_to_activity_pattern_grp =\
            grp["metacluster_idx_to_activity_pattern"]
        metacluster_idx_to_activity_pattern = OrderedDict()
        for metacluster_name in all_metacluster_names:
            metacluster_idx = int(metacluster_name.split("_")[-1])
            activity_pattern =\
             metacluster_idx_to_activity_pattern_grp.attrs[metacluster_name] 
            metacluster_idx_to_activity_pattern[metacluster_idx] =\
                activity_pattern
        return cls(metacluster_indices=metacluster_indices,
                   metaclusterer=metaclusterer,
                   attribute_vectors=attribute_vectors,
                   metacluster_idx_to_activity_pattern=
                    metacluster_idx_to_activity_pattern)

    def save_hdf5(self, grp):
        grp.create_dataset("metacluster_indices",
                           data=self.metacluster_indices)
        self.metaclusterer.save_hdf5(grp) 
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

    def __init__(self, task_name_to_value_provider, task_names):
        self.task_names = task_names
        self.task_name_to_value_provider = task_name_to_value_provider
        self.fit_called = False

    def get_vector_from_seqlet(self, seqlet):
        vector =  np.array([
                    self.task_name_to_value_provider[task_name](seqlet)
                    for task_name in self.task_names])
        return vector 

    def transform(self, seqlets):
        assert (self.fit_called == True), "fit has not been called"
        attribute_vectors = np.array([self.get_vector_from_seqlet(seqlet) 
                                      for seqlet in seqlets])
        metacluster_indices = [
            self._transform_vector(x) for x in attribute_vectors]
        metacluster_idx_to_activity_pattern =\
            self.get_metacluster_idx_to_activity_pattern()
        return MetaclusteringResults(
                       metacluster_indices=metacluster_indices,
                       metaclusterer=self,
                       attribute_vectors=attribute_vectors,
                       metacluster_idx_to_activity_pattern=
                        metacluster_idx_to_activity_pattern)

    def get_metacluster_idx_to_activity_pattern(self):
        raise NotImplementedError()

    def _transform_vector(self, vector):
        raise NotImplementedError()

    def fit_transform(self, seqlets):
        self.fit(seqlets)
        return self.transform(seqlets)

    def fit(self, seqlets):
        attribute_vectors = (np.array([
                             self.get_vector_from_seqlet(x) 
                             for x in seqlets]))
        self._fit(attribute_vectors)
        self.fit_called = True 

    def _fit(self, attribute_vectors):
        raise NotImplementedError()

    @classmethod
    def from_hdf5(cls, grp):
        the_class = eval(grp.attrs["class"])
        return the_class.from_hdf5(grp)


class SignBasedPatternClustering(AbstractMetaclusterer):
    
    def __init__(self, task_name_to_value_provider,
                       task_names,
                       min_cluster_size,
                       threshold_for_counting_sign,
                       weak_threshold_for_counting_sign, verbose=True):
        super(SignBasedPatternClustering, self).__init__(
            task_name_to_value_provider=task_name_to_value_provider,
            task_names=task_names)
        self.min_cluster_size = min_cluster_size
        self.threshold_for_counting_sign = threshold_for_counting_sign
        self.weak_threshold_for_counting_sign =\
             weak_threshold_for_counting_sign
        self.verbose = verbose

    def get_metacluster_idx_to_activity_pattern(self):
        return self.metacluster_idx_to_activity_pattern
        
    def pattern_to_str(self, pattern):
        return ",".join([str(x) for x in pattern])
    
    def vector_to_pattern(self, vector):
        to_return = np.array([
                0 if np.abs(element) < self.threshold_for_counting_sign
                  else (1 if element > 0 else -1)
                  for element in vector])
        if all([ v == 0 for v in to_return ]) :
            print(vector)
            print(to_return)
            assert False
        if (np.sum(np.abs(to_return))==0):
            print(vector)
            print(to_return)
            assert False
        return to_return
    
    def weak_vector_to_pattern(self, vector):
        to_return = np.array([
                0 if np.abs(element) < self.weak_threshold_for_counting_sign
                  else (1 if element > 0 else -1) for element in vector])
        if all([ v == 0 for v in to_return ]) :
            print(vector)
            print(to_return)
            assert False
        if (np.sum(np.abs(to_return))==0):
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

    def map_vector_to_best_pattern(self, vector): 
        vector_pattern = self.weak_vector_to_pattern(vector) #be liberal
        compatible_activity_patterns =\
            self.get_compatible_patterns(vector_pattern,
                                         self.surviving_activity_patterns)
        if len(compatible_activity_patterns) > 0:
            best_pattern = self.pattern_to_str(
                max(compatible_activity_patterns,
                key=lambda x: np.sum(x*np.array(vector) )))
        else:
            best_pattern = None
        return best_pattern

    def _transform_vector(self, vector):
        best_pattern = self.map_vector_to_best_pattern(vector) 
        return (self.activity_pattern_to_cluster_idx[best_pattern]
             if best_pattern in self.final_surviving_activity_patterns
             else -1)

    def set_fit_values(self, activity_pattern_to_cluster_idx,
                             surviving_activity_patterns,
                             final_surviving_activity_patterns):
        self.activity_pattern_to_cluster_idx =\
            activity_pattern_to_cluster_idx
        self.metacluster_idx_to_activity_pattern = OrderedDict(
            [(val,key)
             for key,val in
             self.activity_pattern_to_cluster_idx.items()])
        self.surviving_activity_patterns = surviving_activity_patterns
        self.final_surviving_activity_patterns =\
            final_surviving_activity_patterns
        self.fit_called = True

    @classmethod
    def from_hdf5(cls, grp):
        from . import core
        task_names = util.load_string_list(dset_name="task_names",
                                           grp=grp) 
        task_name_to_value_provider = OrderedDict() 
        task_name_to_value_provider_grp = grp["task_name_to_value_provider"]
        for task_name in task_names:
            task_name_to_value_provider[task_name] =\
                value_provider.AbstractValueProvider.from_hdf5(
                    task_name_to_value_provider_grp[task_name])
        min_cluster_size = grp.attrs["min_cluster_size"]
        threshold_for_counting_sign = grp.attrs["threshold_for_counting_sign"]
        weak_threshold_for_counting_sign =\
            grp.attrs["weak_threshold_for_counting_sign"]
        verbose = grp.attrs["verbose"]

        sign_based_pattern_clustering =\
         cls(task_name_to_value_provider=task_name_to_value_provider,
             task_names=task_names,
             min_cluster_size=min_cluster_size,
             threshold_for_counting_sign=threshold_for_counting_sign,
             weak_threshold_for_counting_sign=weak_threshold_for_counting_sign,
             verbose=verbose)

        fit_called = grp.attrs["fit_called"]
        if (fit_called):
            activity_pattern_to_cluster_idx = OrderedDict()
            activity_pattern_to_cluster_idx_grp =\
                grp["activity_pattern_to_cluster_idx"]
            for activity_pattern in\
                activity_pattern_to_cluster_idx_grp.attrs.keys():
                activity_pattern_to_cluster_idx[activity_pattern] =\
                    activity_pattern_to_cluster_idx_grp.attrs[activity_pattern]   
            surviving_activity_patterns =\
               np.array(grp["surviving_activity_patterns"])
            final_surviving_activity_patterns =\
                set(util.load_string_list(
                 dset_name="final_surviving_activity_patterns", grp=grp))
            sign_based_pattern_clustering.set_fit_values(
                activity_pattern_to_cluster_idx=
                 activity_pattern_to_cluster_idx,
                surviving_activity_patterns=
                 surviving_activity_patterns,
                final_surviving_activity_patterns=
                 final_surviving_activity_patterns)

        return sign_based_pattern_clustering

    def save_hdf5(self, grp):
        grp.attrs["class"] = type(self).__name__
        util.save_string_list(self.task_names, dset_name="task_names",grp=grp) 
        task_name_to_value_provider_grp =(
            grp.create_group("task_name_to_value_provider"))
        for task_name,value_provider in\
            self.task_name_to_value_provider.items():
            value_provider.save_hdf5(
             task_name_to_value_provider_grp.create_group(task_name))
        grp.attrs["min_cluster_size"] = self.min_cluster_size
        grp.attrs["threshold_for_counting_sign"] =\
            self.threshold_for_counting_sign
        grp.attrs["weak_threshold_for_counting_sign"] =\
            self.weak_threshold_for_counting_sign
        grp.attrs["verbose"] = self.verbose
        grp.attrs["fit_called"] = self.fit_called
        if (self.fit_called):
            #save self.activity_pattern_to_cluster_idx
            activity_pattern_to_cluster_idx_grp =\
                grp.create_group("activity_pattern_to_cluster_idx")
            for activity_pattern,cluster_idx in\
                self.activity_pattern_to_cluster_idx.items():
                activity_pattern_to_cluster_idx_grp.attrs[
                    activity_pattern] = cluster_idx
            #save self.surviving_activity_patterns
            grp.create_dataset("surviving_activity_patterns",
                               data=np.array(self.surviving_activity_patterns))
            util.save_string_list(
                 list(self.final_surviving_activity_patterns),
                 dset_name="final_surviving_activity_patterns",
                 grp=grp) 
    
    def _fit(self, attribute_vectors):

        all_possible_activity_patterns =\
            list(itertools.product(*[(1,-1,0) for x
                 in range(attribute_vectors.shape[1])]))
        all_possible_activity_patterns.remove(
            tuple([0]*attribute_vectors.shape[1]))

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
        
        self.surviving_activity_patterns = [
            activity_pattern for activity_pattern in 
            all_possible_activity_patterns if
            (len(activity_pattern_to_attribute_vectors[
                 self.pattern_to_str(activity_pattern)])
             > self.min_cluster_size)]

        activity_patterns = []
        final_activity_pattern_to_vectors = defaultdict(list)
        for vector in attribute_vectors:
            best_pattern = self.map_vector_to_best_pattern(vector) 
            activity_patterns.append(best_pattern)
            if best_pattern is not None:
                final_activity_pattern_to_vectors[best_pattern].append(vector)

        self.final_surviving_activity_patterns = set([
            self.pattern_to_str(activity_pattern) for activity_pattern in 
            all_possible_activity_patterns if
            (len(final_activity_pattern_to_vectors[
                 self.pattern_to_str(activity_pattern)])
             >= self.min_cluster_size)])

        if (self.verbose):
            print(str(len(self.final_surviving_activity_patterns))+
                  " activity patterns with support >= "
                  +str(self.min_cluster_size)+" out of "
                  +str(len(all_possible_activity_patterns))
                  +" possible patterns") 
        
        #sort activity patterns by most to least support
        sorted_activity_patterns = sorted(
            self.final_surviving_activity_patterns,
            key=lambda x: -len(final_activity_pattern_to_vectors[x]))
        self.activity_pattern_to_cluster_idx = dict(
            [(x[1],x[0]) for x in enumerate(sorted_activity_patterns)])
        self.metacluster_idx_to_activity_pattern = dict(
            [(x[0],x[1])
             for x in enumerate(sorted_activity_patterns)])
