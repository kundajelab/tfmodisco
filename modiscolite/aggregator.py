import numpy as np
from . import affinitymat
from . import core
from . import util
from collections import OrderedDict, defaultdict
import itertools
import sys
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import time

class AbstractAggSeqletPostprocessor(object):

    def __call__(self, aggregated_seqlets):
        raise NotImplementedError() #return an array

    def chain(self, postprocessor):
        return AdhocAggSeqletPostprocessor(
                func=lambda x: postprocessor(self(x)))


class AdhocAggSeqletPostprocessor(AbstractAggSeqletPostprocessor):

    def __init__(self, func):
        self.func = func

    def __call__(self, aggregated_seqlets):
        return self.func(aggregated_seqlets)


class TrimToFracSupport(AbstractAggSeqletPostprocessor):

    def __init__(self, min_frac, min_num, verbose):
        self.min_frac = min_frac
        self.min_num = min_num
        self.verbose = verbose

    def __call__(self, aggregated_seqlets):
        return [x.trim_to_positions_with_min_support(
                  min_frac=self.min_frac,
                  min_num=self.min_num,
                  verbose=self.verbose) for x in aggregated_seqlets]


class AbstractTrimToBestWindow(AbstractAggSeqletPostprocessor):

    def __init__(self, window_size):
        self.window_size = window_size

    def score_positions(self, aggregated_seqlet):
        raise NotImplementError()

    def __call__(self, aggregated_seqlets):
        trimmed_agg_seqlets = []
        for aggregated_seqlet in aggregated_seqlets:
            start_idx = np.argmax(util.cpu_sliding_window_sum(
                arr=self.score_positions(aggregated_seqlet),
                window_size=self.window_size))
            end_idx = start_idx + self.window_size
            trimmed_agg_seqlets.append(
                aggregated_seqlet.trim_to_start_and_end_idx(
                    start_idx=start_idx, end_idx=end_idx)) 
        return trimmed_agg_seqlets


class TrimToBestWindowByIC(AbstractTrimToBestWindow):

    def __init__(self, window_size, onehot_track_name, bg_freq):
        super(TrimToBestWindowByIC, self).__init__(window_size=window_size)
        self.onehot_track_name = onehot_track_name
        self.bg_freq = bg_freq

    #sub up imp for each track, take l1 norm, average across seqlets
    def score_positions(self, aggregated_seqlet):
        ppm = aggregated_seqlet[self.onehot_track_name].fwd
        per_pos_ic = util.compute_per_position_ic(
            ppm=ppm, background=self.bg_freq, pseudocount=0.001)
        return per_pos_ic


def expand_seqlets_to_fill_pattern(pattern, track_set, left_flank_to_add, 
    right_flank_to_add, verbose=True):

    new_seqlets_and_alnmts = []
    skipped_seqlets = 0
    for seqlet_and_alnmt in pattern.seqlets_and_alnmts:
        seqlet = seqlet_and_alnmt.seqlet
        alnmt = seqlet_and_alnmt.alnmt
        left_expansion = alnmt+left_flank_to_add 
        right_expansion = ((len(pattern) - 
                           (alnmt+len(seqlet)))+right_flank_to_add)
        if (seqlet.coor.is_revcomp == False):
            start = seqlet.coor.start - left_expansion
            end = seqlet.coor.end + right_expansion
        else:
            start = seqlet.coor.start - right_expansion
            end = seqlet.coor.end + left_expansion
        if (start >= 0 and
            end <= track_set.get_example_idx_len(
                    seqlet.coor.example_idx)):
            seqlet = track_set.create_seqlet(
                coor=core.SeqletCoordinates(
                    example_idx=seqlet.coor.example_idx,
                    start=start, end=end,
                    is_revcomp=seqlet.coor.is_revcomp))

            new_seqlets_and_alnmts.append(
             core.SeqletAndAlignment(seqlet=seqlet, alnmt=0))
        else:
            skipped_seqlets += 1 
    if verbose and (skipped_seqlets > 0):
        print("Skipped "+str(skipped_seqlets)+" seqlets that went over the"
              +" sequence edge during flank expansion") 
        sys.stdout.flush()
    if (len(new_seqlets_and_alnmts) > 0):
        return core.AggregatedSeqlet(seqlets_and_alnmts_arr=
                                      new_seqlets_and_alnmts)
    else:
        return None


class ExpandSeqletsToFillPattern(AbstractAggSeqletPostprocessor):

    def __init__(self, track_set, flank_to_add=0, verbose=True):
        self.track_set = track_set 
        self.verbose = verbose
        self.flank_to_add = flank_to_add

    def __call__(self, aggregated_seqlets):
        new_aggregated_seqlets = []
        for aggregated_seqlet in aggregated_seqlets:
            new_agg_seqlet = expand_seqlets_to_fill_pattern(
                pattern=aggregated_seqlet, track_set=self.track_set,
                left_flank_to_add=self.flank_to_add,
                right_flank_to_add=self.flank_to_add,
                verbose=self.verbose)
            if new_agg_seqlet is not None:
                new_aggregated_seqlets.append(new_agg_seqlet)
        return new_aggregated_seqlets 


class DetectSpuriousMerging2(AbstractAggSeqletPostprocessor):

    def __init__(self, subcluster_settings, min_in_subcluster,
                       similar_patterns_collapser, verbose):
        self.subcluster_settings = subcluster_settings
        self.min_in_subcluster = min_in_subcluster
        self.similar_patterns_collapser = similar_patterns_collapser
        self.verbose = verbose

    def __call__(self, aggregated_seqlets):
        to_return = []
        for i,pattern in enumerate(aggregated_seqlets):
            if (self.verbose):
                print("Inspecting pattern",i,"for spurious merging")
            if (len(pattern.seqlets) > self.min_in_subcluster):
                pattern.compute_subclusters_and_embedding(
                    verbose=self.verbose,
                    compute_embedding=False,
                    **self.subcluster_settings)
                subpatterns = pattern.subcluster_to_subpattern.values()
                #pattern_collapser resturns both the merged patterns as well
                # as the pattern merge hierarchy; we return the merged patterns
                to_return.extend(
                    self.similar_patterns_collapser(subpatterns)[0]) 
            else:
                to_return.append(pattern)
        return to_return


def merge_in_seqlets_filledges(parent_pattern, seqlets_to_merge,
    aligner, track_set, verbose=True):

    parent_pattern = parent_pattern.copy()
    for seqlet in seqlets_to_merge:
        #get the alignment from the aligner 
        (alnmt, revcomp_match, alnmt_score) =\
            aligner(parent_pattern=parent_pattern, child_pattern=seqlet)
        
        if revcomp_match:
            seqlet = seqlet.revcomp()

        preexpansion_seqletlen = len(seqlet)
        #extend seqlet according to the alignment so that it fills the
        # whole pattern
        left_expansion = max(alnmt,0)
        right_expansion = max((len(parent_pattern) - (alnmt+len(seqlet))), 0)

        if seqlet.coor.is_revcomp == False:
            start = seqlet.coor.start - left_expansion
            end = seqlet.coor.end + right_expansion
        else:
            start = seqlet.coor.start - right_expansion
            end = seqlet.coor.end + left_expansion

        example_end = track_set.get_example_idx_len(seqlet.coor.example_idx)

        if start >= 0 and end <= example_end:
            seqlet = track_set.create_seqlet(
                coor=core.SeqletCoordinates(
                    example_idx=seqlet.coor.example_idx,
                    start=start, end=end,
                    is_revcomp=seqlet.coor.is_revcomp)) 
        else:
            continue #don't try adding this seqlet

        #also expand the pattern (if needed) so that the seqlet
        # doesn't go over the edge
        parent_left_expansion = max(0, -alnmt)
        parent_right_expansion = max(0, (alnmt+preexpansion_seqletlen)
                                         - len(parent_pattern))

        if (parent_left_expansion > 0) or (parent_right_expansion > 0):
            candidate_parent_pattern = expand_seqlets_to_fill_pattern(
                pattern=parent_pattern,
                track_set=track_set, left_flank_to_add=parent_left_expansion,
                right_flank_to_add=parent_right_expansion, verbose=verbose)
            if candidate_parent_pattern is not None:
                parent_pattern = candidate_parent_pattern
            else: #the flank expansion required to merge in this seqlet got
                # rid of all the other seqlets in the pattern, so we won't use
                # this seqlet
                continue

        #add the seqlet in at alignment 0, assuming it's not already
        # part of the pattern
        if seqlet not in parent_pattern.seqlets_and_alnmts:
            parent_pattern._add_pattern_with_valid_alnmt(
                            pattern=seqlet, alnmt=0)

    return parent_pattern


class GreedySeqletAggregator():
    def __init__(self, pattern_comparison_settings, seqlet_sort_metric, track_set,
        postprocessor=None):
        self.pattern_comparison_settings = pattern_comparison_settings
        self.seqlet_sort_metric = seqlet_sort_metric
        self.track_set = track_set
        self.postprocessor = postprocessor

    def __call__(self, seqlets):
        sorted_seqlets = sorted(seqlets, key=self.seqlet_sort_metric) 
        pattern = core.AggregatedSeqlet.from_seqlet(sorted_seqlets[0])
        if len(sorted_seqlets) > 1:
            pattern = merge_in_seqlets_filledges(
                aligner=core.CrossContinJaccardPatternAligner(self.pattern_comparison_settings),
                parent_pattern=pattern,
                seqlets_to_merge=sorted_seqlets[1:],
                track_set=self.track_set,
                verbose=True)

        to_return = [pattern]
        if (self.postprocessor is not None):
            to_return = self.postprocessor(to_return)

        #sort by number of seqlets in each...for the default postprocessor
        # there should just be one motif here though 
        return sorted(to_return, key=lambda x: -x.num_seqlets)


class PatternMergeHierarchy(object):

    def __init__(self, root_nodes):
        self.root_nodes = root_nodes

    def add_level(self, level_arr):
        self.levels.append(level_arr)

    def save_hdf5(self, grp):
        root_node_names = []
        for i in range(len(self.root_nodes)):
            node_name = "root_node"+str(i)
            root_node_names.append(node_name) 
            self.root_nodes[i].save_hdf5(grp.create_group(node_name))
        util.save_string_list(root_node_names,
                              dset_name="root_node_names",
                              grp=grp) 

    @classmethod
    def from_hdf5(cls, grp, track_set):
        root_node_names = util.load_string_list(dset_name="root_node_names",
                                                grp=grp) 
        root_nodes = []
        for root_node_name in root_node_names:
            root_node = PatternMergeHierarchyNode.from_hdf5(
                            grp=grp[root_node_name],
                            track_set=track_set)
            root_nodes.append(root_node)
        return cls(root_nodes=root_nodes) 


class PatternMergeHierarchyNode(object):

    def __init__(self, pattern, child_nodes=None, parent_node=None,
                       indices_merged=None, submat_crosscontam=None,
                       submat_alignersim=None): 
        self.pattern = pattern 
        if (child_nodes is None):
            child_nodes = []
        self.child_nodes = child_nodes
        self.parent_node = parent_node
        self.indices_merged = indices_merged
        self.submat_crosscontam = submat_crosscontam
        self.submat_alignersim = submat_alignersim

    def save_hdf5(self, grp):
        if (self.indices_merged is not None):
            grp.create_dataset("indices_merged",
                               data=np.array(self.indices_merged)) 
            grp.create_dataset("submat_crosscontam",
                               data=np.array(self.submat_crosscontam)) 
            grp.create_dataset("submat_alignersim",
                               data=np.array(self.submat_alignersim)) 
        self.pattern.save_hdf5(grp=grp.create_group("pattern"))
        if (self.child_nodes is not None):
            child_node_names = []
            for i in range(len(self.child_nodes)):
                child_node_name = "child_node"+str(i)
                child_node_names.append(child_node_name)
                self.child_nodes[i].save_hdf5(
                    grp.create_group(child_node_name))
            util.save_string_list(child_node_names,
                                  dset_name="child_node_names",
                                  grp=grp)

    @classmethod
    def from_hdf5(cls, grp, track_set):
        pattern = core.AggregatedSeqlet.from_hdf5(grp=grp["pattern"],
                                                  track_set=track_set)  
        if "indices_merged" in grp:
            indices_merged = tuple(grp["indices_merged"][:])
            submat_crosscontam = np.array(grp["submat_crosscontam"])
            submat_alignersim = np.array(grp["submat_alignersim"])
        else:
            (indices_merged, submat_crosscontam,
             submat_alignersim) = (None, None, None)
        if "child_node_names" in grp:
            child_node_names = util.load_string_list(
                                dset_name="child_node_names",
                                grp=grp)
            child_nodes = []
            for child_node_name in child_node_names:
                child_node = PatternMergeHierarchyNode.from_hdf5(
                               grp=grp[child_node_name],
                               track_set=track_set) 
                child_nodes.append(child_node)
               
        else:
            child_nodes = None
   
        to_return = cls(pattern=pattern,
                        child_nodes=child_nodes,
                        indices_merged=indices_merged,
                        submat_crosscontam=submat_crosscontam,
                        submat_alignersim=submat_alignersim) 

        if (child_nodes is not None):
            for child_node in child_nodes:
                child_node.parent_node = to_return

        return to_return


def compute_continjacc_vec_vs_arr(vec, arr):
    abs_vec = np.abs(vec)
    abs_arr = np.abs(arr)
    union = np.sum(np.maximum(abs_vec[None,:], abs_arr), axis=-1)
    intersection = np.sum((np.minimum(abs_vec[None,:], abs_arr)
                    *np.sign(vec[None,:])*np.sign(arr)), axis=-1)
    zeros_mask = (union==0)
    union = (union*(zeros_mask==False) + 1e-7*zeros_mask)
    return intersection/union


def compute_continjacc_arr1_vs_arr2(arr1, arr2, n_cores): 
    return np.array(
        Parallel(n_jobs=n_cores)(
            delayed(compute_continjacc_vec_vs_arr)(vec, arr2) for vec in arr1
        ))

class DynamicDistanceSimilarPatternsCollapser2(object):

    def __init__(self, pattern_comparison_settings,
                       track_set,
                       pattern_aligner,
                       collapse_condition, dealbreaker_condition,
                       postprocessor,
                       verbose=True,
                       max_seqlets_subsample=1000,
                       n_cores=1):
        self.pattern_comparison_settings = pattern_comparison_settings
        self.track_set = track_set
        self.pattern_aligner = pattern_aligner
        self.collapse_condition = collapse_condition
        self.dealbreaker_condition = dealbreaker_condition
        self.postprocessor = postprocessor
        self.verbose = verbose
        self.n_cores = n_cores
        self.max_seqlets_subsample = max_seqlets_subsample

    def subsample_pattern(self, pattern):
        return util.subsample_pattern(
                        pattern,
                        num_to_subsample=self.max_seqlets_subsample)

    def __call__(self, patterns):

        patterns = [x.copy() for x in patterns]

        merge_hierarchy_levels = []        
        current_level_nodes = [
            PatternMergeHierarchyNode(pattern=x) for x in patterns]
        merge_hierarchy_levels.append(current_level_nodes)

        merge_occurred_last_iteration = True
        merging_iteration = 0

        #negative numbers to indicate which
        # entries need to be filled (versus entries we can infer
        # from the previous iteration of the while loop)
        pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
        pairwise_sims = np.zeros((len(patterns), len(patterns)))

        #loop until no more patterns get merged
        while (merge_occurred_last_iteration):

            start  = time.time()
            
            merging_iteration += 1
            if (self.verbose):
                print("On merging iteration",merging_iteration) 
                sys.stdout.flush()
            merge_occurred_last_iteration = False

            if (self.verbose):
                print("Numbers for each pattern pre-subsample:",
                      str([len(x.seqlets) for x in patterns]))
            #Let's subsample 'patterns' to prevent runtime from being too
            # large in calculating pairwise sims. 
            subsample_patterns = [
                (x if x.num_seqlets <= self.max_seqlets_subsample
                 else self.subsample_pattern(x)) for x in patterns]
            if (self.verbose):
                print("Numbers after subsampling:",
                      str([len(x.seqlets) for x in subsample_patterns]))


            for i,(pattern1, subsample_pattern1) in enumerate(
                                            zip(patterns, subsample_patterns)):
                start = time.time()
                if (self.verbose):
                    print("Computing sims for pattern",i)
                #from modisco.visualization import viz_sequence
                #viz_sequence.plot_weights(pattern1["task0_contrib_scores"].fwd)
                for j,(pattern2, subsample_pattern2) in enumerate(
                                            zip(patterns, subsample_patterns)):
                    #Note: I compute both i,j AND j,i because although
                    # the result is the same for the sim, it can be different
                    # for the auroc because a different motif is getting
                    # shifted over.
                    if (j==i):
                        pairwise_aurocs[i,j] = 0.5
                        pairwise_sims[i,j] = 1.0
                        continue
                    if pairwise_aurocs[i,j] >= 0: #filled in from previous iter
                        assert pairwise_aurocs[j,i] >= 0
                        continue 
                        
                    #Compute best alignment between pattern pair
                    (alnmt, rc, aligner_sim) =\
                        self.pattern_aligner(pattern1, pattern2)
                    pairwise_sims[i,j] = aligner_sim

                    #get realigned pattern2
                    pattern2_coords = [x.coor
                        for x in subsample_pattern2.seqlets]
                    if (rc): #flip strand if needed to align
                        pattern2_coords  = [x.revcomp()
                         for x in pattern2_coords]
                    #now apply the alignment
                    pattern2_coords = [
                        x.shift((1 if x.is_revcomp else -1)*alnmt)
                        for x in pattern2_coords] 

                    pattern2_shifted_seqlets = self.track_set.create_seqlets(
                        coords=pattern2_coords,
                        track_names=
                         self.pattern_comparison_settings.track_names) 

                    pattern1_fwdseqdata, _ =\
                      core.get_2d_data_from_patterns(
                        patterns=subsample_pattern1.seqlets,
                        track_names=
                         self.pattern_comparison_settings.track_names,
                        track_transformer=
                         self.pattern_comparison_settings.track_transformer)
                    pattern2_fwdseqdata, _ =\
                      core.get_2d_data_from_patterns(
                        patterns=pattern2_shifted_seqlets,
                        track_names=
                         self.pattern_comparison_settings.track_names,
                        track_transformer=
                         self.pattern_comparison_settings.track_transformer)

                    #Flatten, compute continjacc sim at this alignment
                    flat_pattern1_fwdseqdata = pattern1_fwdseqdata.reshape(
                        (len(pattern1_fwdseqdata), -1))
                    flat_pattern2_fwdseqdata = pattern2_fwdseqdata.reshape(
                        (len(pattern2_fwdseqdata), -1))

                    #Do a check for all-zero scores, print warning
                    #do a check about the per-example sum
                    per_ex_sum_pattern1_zeromask = (np.sum(np.abs(
                        flat_pattern1_fwdseqdata),axis=-1))==0
                    per_ex_sum_pattern2_zeromask = (np.sum(np.abs(
                        flat_pattern2_fwdseqdata),axis=-1))==0
                    if (np.sum(per_ex_sum_pattern1_zeromask) > 0):
                        print("WARNING: Zeros present for pattern1 coords")
                        zero_seqlet_locs =\
                            np.nonzero(per_ex_sum_pattern1_zeromask) 
                        print("\n".join([str(s.coor) for s in
                                         subsample_pattern1.seqlets]))
                    if (np.sum(per_ex_sum_pattern2_zeromask) > 0):
                        print("WARNING: Zeros present for pattern2 coords")
                        zero_seqlet_locs =\
                            np.nonzero(per_ex_sum_pattern2_zeromask)
                        print("\n".join([str(coor) for coor in
                                         pattern2_coords]))

                    between_pattern_sims =\
                     compute_continjacc_arr1_vs_arr2(
                        arr1=flat_pattern1_fwdseqdata,
                        arr2=flat_pattern2_fwdseqdata,
                        n_cores=self.n_cores).ravel()

                    within_pattern1_sims =\
                     compute_continjacc_arr1_vs_arr2(
                        arr1=flat_pattern1_fwdseqdata,
                        arr2=flat_pattern1_fwdseqdata,
                        n_cores=self.n_cores).ravel()

                    auroc = roc_auc_score(
                        y_true=[0 for x in between_pattern_sims]
                               +[1 for x in within_pattern1_sims],
                        y_score=list(between_pattern_sims)
                                +list(within_pattern1_sims))

                    #The symmetrization over i,j and j,i is done later
                    pairwise_aurocs[i,j] = auroc
                if (self.verbose):
                    print("Computed sims for pattern",i,
                          "in",time.time()-start,"s")

            #pairwise_sims is not symmetric; differ based on which pattern is
            # padded with zeros.
            patterns_to_patterns_aligner_sim =\
                0.5*(pairwise_sims + pairwise_sims.T)
            cross_contamination = 2*(1-np.maximum(pairwise_aurocs,0.5))
            
            if (self.verbose):
                print("Cluster sizes")
                print(np.array([len(x.seqlets) for x in patterns]))
                print("Cross-contamination matrix:")
                print(np.round(cross_contamination,2))
                print("Pattern-to-pattern sim matrix:")
                print(np.round(patterns_to_patterns_aligner_sim,2))

            indices_to_merge = []
            merge_partners_so_far = dict([(i, set([i])) for i in
                                          range(len(patterns))])

            #merge patterns with highest similarity first
            sorted_pairs = sorted([(i,j,patterns_to_patterns_aligner_sim[i,j])
                            for i in range(len(patterns))
                            for j in range(len(patterns)) if (i < j)],
                            key=lambda x: -x[2])
            #iterate over pairs
            for (i,j,aligner_sim) in sorted_pairs:
                #symmetrize asymmetric crosscontam
                # take min rather than avg to avoid aggressive merging
                cross_contam = min(cross_contamination[i,j],
                                    cross_contamination[j,i])
                if (self.collapse_condition(prob=cross_contam,
                                            aligner_sim=aligner_sim)):
                    if (self.verbose):
                        print("Collapsing "+str(i)
                              +" & "+str(j)
                              +" with crosscontam "+str(cross_contam)+" and"
                              +" sim "+str(aligner_sim)) 
                        sys.stdout.flush()

                    collapse_passed = True
                    #check compatibility for all indices that are
                    #about to be merged
                    merge_under_consideration = set(
                        list(merge_partners_so_far[i])
                        +list(merge_partners_so_far[j]))
                    for m1 in merge_under_consideration:
                        for m2 in merge_under_consideration:
                            if (m1 < m2):
                                cross_contam_here =\
                                    0.5*(cross_contamination[m1, m2]+
                                         cross_contamination[m2, m1])
                                aligner_sim_here =\
                                    patterns_to_patterns_aligner_sim[
                                        m1, m2]
                                if (self.dealbreaker_condition(
                                        prob=cross_contam_here,
                                        aligner_sim=aligner_sim_here)):
                                    collapse_passed=False                     
                                    if (self.verbose):
                                        print("Aborting collapse as "
                                              +str(m1)
                                              +" & "+str(m2)
                                              +" have cross-contam "
                                              +str(cross_contam_here)
                                              +" and"
                                              +" sim "
                                              +str(aligner_sim_here)) 
                                        sys.stdout.flush()
                                    break

                    if (collapse_passed):
                        indices_to_merge.append((i,j))
                        for an_idx in merge_under_consideration:
                            merge_partners_so_far[an_idx]=\
                                merge_under_consideration 
                else:
                    if (self.verbose):
                        pass
                        #print("Not collapsed "+str(i)+" & "+str(j)
                        #      +" with cross-contam "+str(cross_contam)+" and"
                        #      +" sim "+str(aligner_sim)) 
                        #sys.stdout.flush()

            for i,j in indices_to_merge:
                pattern1 = patterns[i]
                pattern2 = patterns[j]
                if (pattern1 != pattern2): #if not the same object
                    if (pattern1.num_seqlets < pattern2.num_seqlets):
                        parent_pattern, child_pattern = pattern2, pattern1
                    else:
                        parent_pattern, child_pattern = pattern1, pattern2
                    new_pattern = merge_in_seqlets_filledges(
                        parent_pattern=parent_pattern,
                        seqlets_to_merge=child_pattern.seqlets,
                        aligner=self.pattern_aligner,
                        track_set=self.track_set,
                        verbose=self.verbose)
                    new_pattern =\
                        self.postprocessor([new_pattern])
                    assert len(new_pattern)==1
                    new_pattern = new_pattern[0]
                    for k in range(len(patterns)):
                        #Replace EVERY case where the parent or child
                        # pattern is present with the new pattern. This
                        # effectively does single-linkage.
                        if (patterns[k]==parent_pattern or
                            patterns[k]==child_pattern):
                            patterns[k]=new_pattern
            merge_occurred_last_iteration = (len(indices_to_merge) > 0)

            if (merge_occurred_last_iteration):
                #Once we are here, each element of 'patterns'
                #will have the new parent of the corresponding element
                #of 'old_patterns'
                old_to_new_pattern_mapping = patterns

                #sort by size and remove redundant patterns
                patterns = sorted(patterns, key=lambda x: -x.num_seqlets)
                patterns = list(OrderedDict([(x,1) for x in patterns]).keys())

                #let's figure out which indices don't require recomputation
                # and use it to repopulate pairwise_sims and pairwise_aurocs
                old_to_new_index_mappings = OrderedDict()
                for old_pattern_idx,(old_pattern_node, corresp_new_pattern)\
                    in enumerate(zip(current_level_nodes,
                                     old_to_new_pattern_mapping)):
                    #if the old pattern was NOT changed in this iteration
                    if (old_pattern_node.pattern == corresp_new_pattern):
                        new_idx = patterns.index(corresp_new_pattern) 
                        old_to_new_index_mappings[old_pattern_idx] = new_idx
                print("Unmerged patterns remapping:",old_to_new_index_mappings)
                new_pairwise_aurocs = -np.ones((len(patterns), len(patterns)))
                new_pairwise_sims = np.zeros((len(patterns), len(patterns)))
                for old_idx_i, new_idx_i in\
                    old_to_new_index_mappings.items():
                    for old_idx_j, new_idx_j in\
                        old_to_new_index_mappings.items():
                        new_pairwise_aurocs[new_idx_i, new_idx_j] =\
                            pairwise_aurocs[old_idx_i, old_idx_j]
                        new_pairwise_sims[new_idx_i, new_idx_j] =\
                            pairwise_sims[old_idx_i, old_idx_j]
                pairwise_aurocs = new_pairwise_aurocs 
                pairwise_sims = new_pairwise_sims
                     

                #update the hierarchy
                #the current 'top level' will consist of all the current
                # nodes that didn't get a new parent, plus any new parents
                # created                
                next_level_nodes = []
                for frontier_pattern in patterns:
                    #either this pattern is in old_pattern_nodes, in which
                    # case take the old_pattern_node entry, or it's a completely
                    # new pattern in which case make a node for it
                    old_pattern_node_found = False
                    for old_pattern_node in current_level_nodes:
                        if (old_pattern_node.pattern==frontier_pattern):
                            #sanity check..there should be only one node
                            # per pattern
                            assert old_pattern_node_found==False
                            next_level_nodes.append(old_pattern_node)
                            old_pattern_node_found = True 
                    if (old_pattern_node_found==False):
                       next_level_nodes.append(
                        PatternMergeHierarchyNode(frontier_pattern)) 

                for next_level_node in next_level_nodes:
                    #iterate over all the old patterns and their new parent
                    # in order to set up the child nodes correctly
                    for old_pattern_idx,(old_pattern_node, corresp_new_pattern)\
                        in enumerate(zip(current_level_nodes,
                                         old_to_new_pattern_mapping)):
 
                        #if the node has a new parent
                        if (old_pattern_node.pattern != corresp_new_pattern):
                            if (next_level_node.pattern==corresp_new_pattern):

                                
                                #corresp_new_pattern should be comprised of a 
                                # merging of all the old patterns at
                                # indices_merged_with
                                indices_merged = tuple(sorted(
                                    merge_partners_so_far[old_pattern_idx])) 
                                #get the relevant slice         
                                submat_crosscontam =\
                                 cross_contamination[indices_merged,:][:,
                                                     indices_merged]
                                submat_alignersim =\
                                 patterns_to_patterns_aligner_sim[
                                    indices_merged, :][:,indices_merged]

                                if (next_level_node.indices_merged is not None):
                                    assert (next_level_node.indices_merged
                                            ==indices_merged),\
                                     (next_level_node.indices_merged,
                                      indices_merged)
                                else:
                                    next_level_node.indices_merged =\
                                        indices_merged
                                    next_level_node.submat_crosscontam =\
                                        submat_crosscontam
                                    next_level_node.submat_alignersim =\
                                        submat_alignersim

                                next_level_node.child_nodes.append(
                                                old_pattern_node) 
                                assert old_pattern_node.parent_node is None
                                old_pattern_node.parent_node = next_level_node
                            

                current_level_nodes=next_level_nodes
                print("Time spent on merging iteration:", time.time()-start)

        return patterns, PatternMergeHierarchy(root_nodes=current_level_nodes)
    