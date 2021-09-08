from __future__ import division, print_function
from collections import defaultdict, OrderedDict, namedtuple
import numpy as np
import time
import itertools
from .. import core
from .. import affinitymat
from .. import util
from .. import cluster
from .. import aggregator
from .. import seqlet_embedding
from .. import affinitymat
from .. import coordproducers
from .. import tfmodisco_workflow
import sys
from joblib import Parallel, delayed
from tqdm import tqdm


class MakeHitScorer(object):

    def __init__(self, patterns, target_seqlet_size,
                       bg_freq,
                       task_names_and_signs,
                       n_cores,
                       run_trim_and_subcluster=True,
                       additional_trimandsubcluster_kwargs={},
                       additional_seqletscorer_kwargs={}):

        self.target_seqlet_size = target_seqlet_size

        onehot_track_name = "sequence" #the default

        task_names = [x[0] for x in task_names_and_signs]
        self.task_names = task_names
        self.task_names_and_signs = task_names_and_signs

        if (run_trim_and_subcluster):
            print("Getting trimmed patterns, subclustering them")
            self.trimmed_subclustered_patterns = trim_and_subcluster_patterns(
                patterns=patterns, window_size=target_seqlet_size,
                onehot_track_name=onehot_track_name,
                task_names=task_names, bg_freq=bg_freq,
                n_cores=n_cores, **additional_trimandsubcluster_kwargs)
        else:
            self.trimmed_subclustered_patterns = patterns

        print("Preparing seqlet scorer")

        self.seqlet_scorer = prepare_seqlet_scorer(
            patterns=self.trimmed_subclustered_patterns,
            onehot_track_name=onehot_track_name,
            task_names_and_signs=task_names_and_signs,
            n_cores=n_cores,
            bg_freq=bg_freq, **additional_seqletscorer_kwargs)

        self.tnt_results = None

    def get_coordproducer_score_track(self, contrib_scores):
        combined_contribs = np.zeros_like(contrib_scores[self.task_names[0]])
        for task_name,sign in self.task_names_and_signs:
            combined_contribs += contrib_scores[task_name]*sign
        return np.sum(combined_contribs, axis=-1)

    def set_coordproducer(self, contrib_scores,
                core_sliding_window_size,
                target_fdr,
                min_passing_windows_frac,
                max_passing_windows_frac,
                null_track=coordproducers.LaplaceNullDist(num_to_samp=10000),
                bp_to_suppress_around_core=None,
                sign_to_return=1, #should take values of 1, -1 or None
                **additional_coordproducer_kwargs):

        assert sign_to_return in [1, -1, None]

        assert (self.target_seqlet_size-core_sliding_window_size)%2==0,\
                ("Please provide a core_sliding_window_size that is an"
                 +" even number smaller than target_seqlet_size")

        if (bp_to_suppress_around_core is None):
            bp_to_suppress_around_core = core_sliding_window_size

        self.coordproducer = coordproducers.FixedWindowAroundChunks(
            #sliding=[core_sliding_window_size],
            sliding=core_sliding_window_size,
            flank=int((self.target_seqlet_size-core_sliding_window_size)/2.0),
            #the -0.5 is there for nitty-gritty reasons re. how the
            # suppression window is imposed
            suppress=bp_to_suppress_around_core-0.5,
            target_fdr=target_fdr,
            min_passing_windows_frac=min_passing_windows_frac,
            max_passing_windows_frac=max_passing_windows_frac,
            sign_to_return=sign_to_return,
            **additional_coordproducer_kwargs
        ) 
        self.core_sliding_window_size = core_sliding_window_size

        coordproducer_results = self.coordproducer(
            score_track=self.get_coordproducer_score_track(
                            contrib_scores=contrib_scores),
            null_track=null_track)

        self.tnt_results = coordproducer_results.tnt_results

        return self 

    def __call__(self, contrib_scores, hypothetical_contribs, one_hot,
                       hits_to_return_per_seqlet=1,
                       min_mod_precision=0,
                       revcomp=True, coordproducer_settings=None):
        if (coordproducer_settings is None):
            if (self.tnt_results is None):
                raise RuntimeError("Please set a coordproducer or provide"
                                   +" coordproducer settings")
        else:
            self.set_coordproducer(contrib_scores=contrib_scores,
                                   **coordproducer_settings)

        score_track = self.get_coordproducer_score_track(
                                                contrib_scores=contrib_scores)
        coords = self.coordproducer(
                    score_track=score_track, tnt_results=self.tnt_results,
                    null_track=None).coords
        track_set = tfmodisco_workflow.workflow.prep_track_set(
                        task_names=self.task_names,
                        contrib_scores=contrib_scores,
                        hypothetical_contribs=hypothetical_contribs,
                        one_hot=one_hot,
                        custom_perpos_contribs=None,
                        revcomp=revcomp, other_tracks=[])        
        seqlets = track_set.create_seqlets(coords)

        (all_seqlet_hits, patternidx_to_matches, exampleidx_to_matches) =\
            self.seqlet_scorer(seqlets=seqlets,
                hits_to_return_per_seqlet=hits_to_return_per_seqlet,
                min_mod_precision=min_mod_precision,
                trim_to_central=self.core_sliding_window_size) 

        exampleidx_to_matcheswithimportance = defaultdict(list)
        patternidx_to_matcheswithimportance = defaultdict(list)

        for exampleidx in exampleidx_to_matches:
            for motifmatch in sorted(exampleidx_to_matches[exampleidx],
                                     key=lambda x: x.start):
                total_importance = np.sum(score_track[exampleidx][
                              max(motifmatch.start,0):motifmatch.end])
                #tedious but i want to keep them tuples
                motifmatch_with_importance = MotifMatchWithImportance(
                    patternidx=motifmatch.patternidx,
                    patternidx_rank=motifmatch.patternidx_rank,
                    total_importance=total_importance,
                    exampleidx=motifmatch.exampleidx,
                    start=motifmatch.start, end=motifmatch.end,
                    trim_start=motifmatch.trim_start,
                    trim_end=motifmatch.trim_end,
                    seqlet_orig_start=motifmatch.seqlet_orig_start,
                    seqlet_orig_end=motifmatch.seqlet_orig_end,
                    seqlet_orig_revcomp=motifmatch.seqlet_orig_revcomp,
                    is_revcomp=motifmatch.is_revcomp,
                    aggregate_sim=motifmatch.aggregate_sim,
                    mod_delta=motifmatch.mod_delta,
                    mod_precision=motifmatch.mod_precision,
                    mod_percentile=motifmatch.mod_percentile,
                    fann_perclasssum_perc=motifmatch.fann_perclasssum_perc,
                    fann_perclassavg_perc=motifmatch.fann_perclassavg_perc)

                exampleidx_to_matcheswithimportance[exampleidx].append(
                    motifmatch_with_importance)
                patternidx_to_matcheswithimportance[
                    motifmatch.patternidx].append(motifmatch_with_importance)

        return (exampleidx_to_matcheswithimportance,
                patternidx_to_matcheswithimportance)


def trim_and_subcluster_patterns(patterns, window_size, onehot_track_name,
                                 task_names, bg_freq, n_cores,
                                 subpattern_perplexity=50, verbose=True):
    if (verbose):
        print("Trimming the patterns to the target length") 

    patterns = util.trim_patterns_by_ic(
                patterns=patterns, window_size=window_size,
                onehot_track_name=onehot_track_name,
                bg_freq=bg_freq)

    if (verbose):
        print("Apply subclustering") 
    
    track_names = ([x+"_contrib_scores" for x in task_names]
                  +[x+"_hypothetical_contribs" for x in task_names])
    util.apply_subclustering_to_patterns(
            patterns=patterns,
            track_names=track_names,
            n_jobs=n_cores, perplexity=subpattern_perplexity, verbose=True)

    return patterns


def prepare_seqlet_scorer(patterns,
                          onehot_track_name,
                          task_names_and_signs,
                          n_cores,
                          bg_freq,
                          max_seqlets_per_submotif=100,
                          min_overlap_size=10,
                          crosspattern_perplexity=10,
                          n_neighbors=500,
                          ic_trim_threshold=0.3,
                          verbose=True,
                          seqlet_batch_size=5000):

    assert len(set([len(x) for x in patterns]))==1, (
        "patterns should be of equal lengths - are: "
        +str(set([len(x) for x in patterns])))

    target_seqlet_size = len(patterns[0])
    if (verbose):
        print("Pattern length (and hence target seqlet size) is "
              +str(target_seqlet_size))

    assert min_overlap_size < target_seqlet_size, (
            "min_overlap_size must be < target_seqlet_size; are "
            +str(min_overlap_size)+" and "+str(target_seqlet_size))

    subpatterns = []
    subpattern_to_superpattern_mapping = {} 
    subpattern_count = 0
    for i,pattern in enumerate(patterns):
        for subpattern in pattern.subcluster_to_subpattern.values():
            if (len(subpattern.seqlets) > max_seqlets_per_submotif):
                if (verbose):
                    print("Subsampling subpattern "+str(subpattern_count))
                subpattern = util.subsample_pattern(
                    pattern=subpattern,
                    num_to_subsample=max_seqlets_per_submotif)
            subpatterns.append(subpattern)
            subpattern_to_superpattern_mapping[subpattern_count] = i
            subpattern_count += 1

    if (verbose):
        print("Prepare seqlet scorer") 

    track_names = ([x[0]+"_contrib_scores" for x in task_names_and_signs]
                +[x[0]+"_hypothetical_contribs" for x in task_names_and_signs])

    min_overlap=float(min_overlap_size/target_seqlet_size)
    seqlet_scorer = CoreDensityAdaptedSeqletScorer2(
        patterns=subpatterns,
        n_neighbors=n_neighbors,
        affmat_from_seqlets_with_alignments=
         affinitymat.core.AffmatFromSeqletsWithNNpairs(
           pattern_comparison_settings=
            affinitymat.core.PatternComparisonSettings( 
               track_names=track_names, 
               #L1 norm is important for contin jaccard sim
               track_transformer=affinitymat.L1Normalizer(), 
               min_overlap=float(min_overlap_size/target_seqlet_size)),
           sim_metric_on_nn_pairs=\
               affinitymat.core.ParallelCpuCrossMetricOnNNpairs(
                 n_cores=n_cores,
                 cross_metric_single_region=
                  affinitymat.core.CrossContinJaccardSingleRegionWithArgmax())),
        aff_to_dist_mat=
            affinitymat.transformers.AffToDistViaInvLogistic(),
        perplexity=crosspattern_perplexity,
        n_cores=n_cores,
        pattern_to_superpattern_mapping=subpattern_to_superpattern_mapping,
        superpatterns=patterns,
        bg_freq=bg_freq,
        ic_trim_threshold=ic_trim_threshold,
        onehot_track_name=onehot_track_name,
        verbose=verbose,
        seqlet_batch_size=seqlet_batch_size
    )

    return seqlet_scorer

 
class CoreDensityAdaptedSeqletScorer2(object):

    #patterns should already be subsampled 
    def __init__(self, patterns,
                       n_neighbors,
                       affmat_from_seqlets_with_alignments,
                       aff_to_dist_mat,
                       perplexity,
                       n_cores,
                       bg_freq,
                       #ic_trim_threshold is further trimming beyond the width
                       # of the seqlet to hone in on core motif region; done
                       # only at the end.
                       ic_trim_threshold,
                       onehot_track_name,
                       pattern_to_superpattern_mapping=None,
                       superpatterns=None,
                       leiden_numseedstotry=50,
                       verbose=True, seqlet_batch_size=5000): 
        self.patterns = patterns
        if (pattern_to_superpattern_mapping is None):
            pattern_to_superpattern_mapping = dict([
                                          (i,i) for i in range(len(patterns))])
            self.class_patterns = patterns
        else:
            self.class_patterns = superpatterns
        self.n_neighbors = n_neighbors
        self.affmat_from_seqlets_with_alignments =\
            affmat_from_seqlets_with_alignments
        self.pattern_to_superpattern_mapping = pattern_to_superpattern_mapping
        self.aff_to_dist_mat = aff_to_dist_mat
        self.perplexity = perplexity
        self.n_cores = n_cores
        self.bg_freq = bg_freq
        self.ic_trim_threshold = ic_trim_threshold
        self.onehot_track_name = onehot_track_name
        self.leiden_numseedstotry = leiden_numseedstotry
        self.verbose = verbose
        self.seqlet_batch_size = seqlet_batch_size
        self.build()

    #fine_affmat_nn and seqlet_neighbors are lists of lists, indicating which
    # seqlets were the closest ones
    def get_classwise_fine_affmat_nn_sumavg(self,
            fine_affmat_nn, seqlet_neighbors, exclude_self=False):
        num_classes = max(self.motifmemberships)+1
        #(not used in the density-adapted scoring) for each class, compute
        # the total fine-grained similarity for each class in the topk
        # nearest neighbors. Will be used to instantiate a class-wise
        # precision scorer
        fine_affmat_nn_perclassum = np.zeros(
            (len(fine_affmat_nn), num_classes))
        fine_affmat_nn_perclassavg = np.zeros(
            (len(fine_affmat_nn), num_classes))

        if (exclude_self):
            self_not_in_nn = 0 #keep a count for sanity-check purposes

        for i in range(len(fine_affmat_nn)):
            if (exclude_self): 
                #exclude_self means exclude the self-similarity
                # (which would be 1.0 assuming the alignment works out),
                # for the case where we are just sanity-checking
                # how this score
                # works on the original motif seqlets themselves.
                if (i not in seqlet_neighbors[i]):
                    self_not_in_nn += 1
            for classidx in range(num_classes):
                class_entries = [fine_affmat_nn[i][j] for
                   j in range(len(fine_affmat_nn[i]))
                   if ((self.motifmemberships[
                              seqlet_neighbors[i][j]]==classidx)
                       and (exclude_self==False
                            or seqlet_neighbors[i][j] != i) )]
                if (len(class_entries) > 0):
                    fine_affmat_nn_perclassum[i][classidx] =\
                        np.sum(class_entries)
                    fine_affmat_nn_perclassavg[i][classidx] =\
                        np.mean(class_entries)

        if (exclude_self):
            print(self_not_in_nn,"seqlets out of",len(fine_affmat_nn),
                  "did not have themselves in their nearest neighbs, likely"
                  "due to alignment issues") 

        return (fine_affmat_nn_perclassum, fine_affmat_nn_perclassavg)

    def pad_seqletdata_to_align(self, fwdseqletdata, revseqletdata,
                                      alignmentinfo):
        full_length = len(fwdseqletdata)
        _, pattern_alnmt_offset, pattern_alnmt_isfwd = alignmentinfo
        pattern_alnmt_offset = int(pattern_alnmt_offset)
        if (pattern_alnmt_isfwd):
            compareto_seqlet = np.pad(
              array=fwdseqletdata[max(pattern_alnmt_offset,0):
                               min(pattern_alnmt_offset+full_length,
                               full_length)],
              pad_width=[(max(-pattern_alnmt_offset,0),
                          max(pattern_alnmt_offset,0)),
                         (0,0)])
        else:
            compareto_seqlet = np.pad(
              array=revseqletdata[
                  max(-pattern_alnmt_offset,0):
                  min(full_length-pattern_alnmt_offset,
                      full_length)],
              pad_width=[(max(pattern_alnmt_offset,0),
                          max(-pattern_alnmt_offset,0)), 
                         (0,0)])
        return compareto_seqlet

    def get_similarities_to_classpatterns(self, seqlets, trim_to_central):
        full_length=len(self.class_patterns[0])
        assert len(seqlets[0])==full_length

        if (trim_to_central is None):
            seqlets_foralignment = seqlets
            trim_to_central = full_length
            trim_amount = 0
        else:
            trim_amount = int(0.5*(len(seqlets[0])-trim_to_central))
            assert len(set([len(x) for x in seqlets]))==1
            seqlets_foralignment = [seqlet.trim(
                                        start_idx=trim_amount,
                                        end_idx=len(seqlet)-trim_amount)
                                    for seqlet in seqlets]
        #all_pattern_alnmnts has dims of num_seqlets x num_patterns x 3
        # first entry of last index is the sim,
        # second index is the alignment, third entry is is_fwd.
        #The alignment holds the seqlet fixed and
        # slides the pattern across it. is_fwd==False means the pattern was
        # reverse-complemented. The seqlet is what is padded.
        all_pattern_alnmnts =\
            self.affmat_from_seqlets_with_alignments(
                seqlets=seqlets_foralignment,
                filter_seqlets=self.class_patterns,
                min_overlap_override=float(trim_to_central)/full_length)
        all_pattern_alnmnts[:,:,1] += trim_amount

        pattern_comparison_settings = (self.affmat_from_seqlets_with_alignments
                                           .pattern_comparison_settings)
        (all_seqlet_fwd_data, all_seqlet_rev_data) =\
            core.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=pattern_comparison_settings.track_names,
                track_transformer=
                    pattern_comparison_settings.track_transformer)

        assert all_seqlet_fwd_data.shape[1]==len(self.class_patterns[0])

        pattern_aggregate_sims = []

        for seqlet_batch_start in range(0,len(seqlets),self.seqlet_batch_size):
            this_batch_size = (min(len(seqlets),
              seqlet_batch_start+self.seqlet_batch_size)-seqlet_batch_start)
            if (self.verbose):
                print("On seqlets",seqlet_batch_start,"to",
                      seqlet_batch_start+this_batch_size,"out of",
                      len(seqlets))
                sys.stdout.flush() 
            batch_allpatterns_aggregate_sims = np.zeros((this_batch_size,
                                                     len(self.class_patterns)))
            for pattern_idx in (range(len(self.class_patterns)) if
                                self.verbose==False
                                else tqdm(range(len(self.class_patterns))) ): 
                compareto_seqlets = []
                for seqlet_idx in range(seqlet_batch_start,
                                        seqlet_batch_start+this_batch_size):
                    compareto_seqlet = self.pad_seqletdata_to_align(
                       fwdseqletdata=all_seqlet_fwd_data[seqlet_idx],
                       revseqletdata=all_seqlet_rev_data[seqlet_idx],
                       alignmentinfo=
                        all_pattern_alnmnts[seqlet_idx, pattern_idx]) 
                    compareto_seqlets.append(compareto_seqlet)

                compareto_seqlets = np.concatenate([x[None,:]
                            for x in compareto_seqlets], axis=0)
                flatten_compareto_seqlets = compareto_seqlets.reshape(
                                             (len(compareto_seqlets),-1))
                #aggregate sims
                batch_pattern_aggregate_sims =\
                 util.compute_continjacc_sims_1vmany(
                     vec1=self.classpattern_aggregatedata[pattern_idx].ravel(),
                     vecs2=flatten_compareto_seqlets,
                     vecs2_weighting=np.ones_like(flatten_compareto_seqlets))
                batch_allpatterns_aggregate_sims[:,pattern_idx] =\
                    batch_pattern_aggregate_sims
                
            pattern_aggregate_sims.extend(batch_allpatterns_aggregate_sims)

        pattern_aggregate_sims = np.array(pattern_aggregate_sims)
        all_pattern_alnmnts[:,:,0] = pattern_aggregate_sims 
        return all_pattern_alnmnts

    def get_similarities_to_motifseqlets(self, seqlets, trim_to_central):
        full_length=len(self.patterns[0])
        assert len(seqlets[0])==full_length

        if (trim_to_central is None):
            seqlets_foralignment = seqlets
            trim_to_central = full_length
            trim_amount = 0
        else:
            trim_amount = int(0.5*(len(seqlets[0])-trim_to_central))
            assert len(set([len(x) for x in seqlets]))==1
            seqlets_foralignment = [seqlet.trim(
                                        start_idx=trim_amount,
                                        end_idx=len(seqlet)-trim_amount)
                                    for seqlet in seqlets]
        #all_pattern_alnmnts has dims of num_seqlets x num_patterns x 3
        # first entry of last index is the sim,
        # second index is the alignment, third entry is is_fwd.
        #The alignment holds the seqlet fixed and
        # slides the pattern across it. is_fwd==False means the pattern was
        # reverse-complemented. The seqlet is what is padded.
        all_pattern_alnmnts =\
            self.affmat_from_seqlets_with_alignments(
                seqlets=seqlets_foralignment,
                filter_seqlets=self.patterns,
                min_overlap_override=float(trim_to_central)/full_length)
        all_pattern_alnmnts[:,:,1] += trim_amount

        pattern_comparison_settings = (self.affmat_from_seqlets_with_alignments
                                           .pattern_comparison_settings)
        (all_seqlet_fwd_data, all_seqlet_rev_data) =\
            core.get_2d_data_from_patterns(
                patterns=seqlets,
                track_names=pattern_comparison_settings.track_names,
                track_transformer=
                    pattern_comparison_settings.track_transformer)

        assert all_seqlet_fwd_data.shape[1]==len(self.patterns[0])

        num_motifseqlets = sum([len(x) for x in
                                self.pattern_innerseqletdata])

        seqlet_neighbors = []
        affmat_nn = []

        for seqlet_batch_start in range(0,len(seqlets),self.seqlet_batch_size):
            this_batch_size = (min(len(seqlets),
              seqlet_batch_start+self.seqlet_batch_size)-seqlet_batch_start)
            if (self.verbose):
                print("On seqlets",seqlet_batch_start,"to",
                      seqlet_batch_start+this_batch_size,"out of",
                      len(seqlets))
                sys.stdout.flush() 
            batch_allpatterns_pairwise_sims = np.zeros((this_batch_size,
                                               num_motifseqlets))
            pattern_innerseqlet_startidx = 0
            
            for pattern_idx in (range(len(self.patterns)) if
                                self.verbose==False
                                else tqdm(range(len(self.patterns))) ): 
                compareto_seqlets = []
                for seqlet_idx in range(seqlet_batch_start,
                                        seqlet_batch_start+this_batch_size):
                    compareto_seqlet = self.pad_seqletdata_to_align(
                       fwdseqletdata=all_seqlet_fwd_data[seqlet_idx],
                       revseqletdata=all_seqlet_rev_data[seqlet_idx],
                       alignmentinfo=
                        all_pattern_alnmnts[seqlet_idx, pattern_idx]) 
                    compareto_seqlets.append(compareto_seqlet)

                compareto_seqlets = np.concatenate([x[None,:]
                            for x in compareto_seqlets], axis=0)
                flatten_compareto_seqlets = compareto_seqlets.reshape(
                                             (len(compareto_seqlets),-1))

                #pairwise sims to the things in the pattern
                this_pattern_innerseqlet_fwd_data =\
                    self.pattern_innerseqletdata[pattern_idx] 
                batch_pattern_pairwise_sims =\
                    util.compute_pairwise_continjacc_sims(
                     vecs1=compareto_seqlets.reshape(
                             (len(compareto_seqlets),-1)),
                     vecs2=this_pattern_innerseqlet_fwd_data.reshape(
                             (len(this_pattern_innerseqlet_fwd_data),-1)),
                     n_jobs=self.n_cores,
                     vecs2_weighting=None,
                     verbose=False) 
                batch_allpatterns_pairwise_sims[:,
                  pattern_innerseqlet_startidx:
                   (pattern_innerseqlet_startidx
                     +len(this_pattern_innerseqlet_fwd_data))] =(
                  batch_pattern_pairwise_sims) 
                pattern_innerseqlet_startidx +=\
                    len(this_pattern_innerseqlet_fwd_data)
                
            batch_seqlet_neighbors = np.argsort(
              -batch_allpatterns_pairwise_sims, axis=-1)[:,:self.n_neighbors] 
            seqlet_neighbors.extend(batch_seqlet_neighbors)

            batch_affmat_nn = np.array([sims[neighbors]
               for sims,neighbors in
               zip(batch_allpatterns_pairwise_sims,
                   batch_seqlet_neighbors)])
            affmat_nn.extend(batch_affmat_nn)

        affmat_nn = np.array(affmat_nn)
        seqlet_neighbors = np.array(seqlet_neighbors)

        return affmat_nn, seqlet_neighbors
        
    def build(self):

        #do ic trimming to get the offsets for adjusting the motif hit
        # locations
        self.classpattern_trimindices = [util.get_ic_trimming_indices(
            ppm=classpattern[self.onehot_track_name].fwd,
            background=self.bg_freq,
            threshold=self.ic_trim_threshold)
         for classpattern in self.class_patterns]

        motifmemberships = np.array([
            self.pattern_to_superpattern_mapping[i]
            for i in range(len(self.patterns))
            for j in self.patterns[i].seqlets])
        self.motifmemberships = motifmemberships
        assert (max(self.motifmemberships)+1) == len(self.class_patterns),\
            (max(self.motifmemberships), len(self.class_patterns))

        if (self.verbose):
            print("Computing best alignments for all motifseqlets")

        motifseqlets = [seqlet for pattern in self.patterns
                               for seqlet in pattern.seqlets]

        #fetch the pattern inner seqlet data
        pattern_comparison_settings = (self.affmat_from_seqlets_with_alignments
                                           .pattern_comparison_settings)

        self.classpattern_aggregatedata = core.get_2d_data_from_patterns(
                patterns=self.class_patterns,
                track_names=pattern_comparison_settings.track_names,
                track_transformer=
                    pattern_comparison_settings.track_transformer)[0] 
        self.pattern_innerseqletdata = [core.get_2d_data_from_patterns(
                patterns=pattern.seqlets,
                track_names=pattern_comparison_settings.track_names,
                track_transformer=
                    pattern_comparison_settings.track_transformer)[0]
                for pattern in self.patterns]

        #Compute best alignment and similarities for each seqlet
        #logic for why we don't take the cluter membership of each motif
        # seqlet as the best alignment: we want to represent the
        # process of a new seqlet coming in as accurately as possible;
        # we won't know th best alignment for a new seqlet coming in
        fine_affmat_nn, seqlet_neighbors =\
            self.get_similarities_to_motifseqlets(
             seqlets=motifseqlets, trim_to_central=None)

        #fann = fine affmat nn. This is not used for density-adaptive
        # scoring; rather it's a way to get a sense of within-motif
        # similarity WITHOUT the density-adaptation step
        (fann_perclassum, fann_perclassavg) = (
            self.get_classwise_fine_affmat_nn_sumavg(
                fine_affmat_nn=fine_affmat_nn,
                seqlet_neighbors=seqlet_neighbors,
                exclude_self=True))
        if (self.verbose):
            print("Insantiating a precision scorer based on fann_perclasssum")
        self.fann_perclasssum_precscorer = util.ClasswisePrecisionScorer(
            true_classes=motifmemberships,
            class_membership_scores=fann_perclassum) 
        if (self.verbose):
            print("Insantiating a precision scorer based on fann_perclassavg")
        self.fann_perclassavg_precscorer = util.ClasswisePrecisionScorer(
            true_classes=motifmemberships,
            class_membership_scores=fann_perclassavg) 

        #As a baseline, compare to a scorer that uses aggregate similarity
        classpattern_simsandalnmnts = self.get_similarities_to_classpatterns(
                                seqlets=motifseqlets,
                                trim_to_central=0)
        if (self.verbose):
            print("Insantiating a precision scorer based on aggregate sim")
        self.aggsim_precscorer = util.ClasswisePrecisionScorer(
            true_classes=motifmemberships,
            class_membership_scores=classpattern_simsandalnmnts[:,:,0]) 
                
        if (self.verbose):
            print("Mapping affinity to distmat")

        #Map aff to dist
        distmat_nn = self.aff_to_dist_mat(affinity_mat=fine_affmat_nn) 

        if (self.verbose):
            print("Symmetrizing nearest neighbors")

        #Note: the fine-grained similarity metric isn't actually symmetric
        # because a different input will get padded with zeros depending
        # on which seqlets are specified as the filters and which seqlets
        # are specified as the 'thing to scan'. So explicit symmetrization
        # is worthwhile
        sym_seqlet_neighbors, sym_distmat_nn = util.symmetrize_nn_distmat(
            distmat_nn=distmat_nn, nn=seqlet_neighbors,
            average_with_transpose=True)
        del distmat_nn
        del seqlet_neighbors
        
        if (self.verbose):
            print("Computing betas for density adaptation")

        #Compute beta values for the density adaptation. *store it*
        betas_and_ps = Parallel(n_jobs=self.n_cores)(
                 delayed(util.binary_search_perplexity)(
                      self.perplexity, distances)
                 for distances in sym_distmat_nn)
        self.motifseqlet_betas = np.array([x[0] for x in betas_and_ps])
        del betas_and_ps

        if (self.verbose):
            print("Computing normalizing denominators")

        #also compute the normalization factor needed to get probs to sum to 1
        #note: sticking to lists here because different rows of
        # sym_distmat_nn may have different lengths after adding in
        # the symmetric pairs
        densadapted_affmat_nn_unnorm = [np.exp(-np.array(distmat_row)/beta)
            for distmat_row, beta in
            zip(sym_distmat_nn, self.motifseqlet_betas)]
        normfactors = np.array([max(np.sum(x),1e-8) for x in
                                densadapted_affmat_nn_unnorm])
        self.motifseqlet_normfactors = normfactors
        del normfactors

        if (self.verbose):
            print("Computing density-adapted nn affmat")

        #Do density-adaptation using average of self-Beta and other-Beta.
        sym_densadapted_affmat_nn = self.densadapt_wrt_motifseqlets(
                            new_rows_distmat_nn=sym_distmat_nn,
                            new_rows_nn=sym_seqlet_neighbors,
                            new_rows_betas=self.motifseqlet_betas,
                            new_rows_normfactors=self.motifseqlet_normfactors)

        #Make csr matrix
        csr_sym_density_adapted_affmat = util.coo_matrix_from_neighborsformat(
            entries=sym_densadapted_affmat_nn,
            neighbors=sym_seqlet_neighbors,
            ncols=len(sym_densadapted_affmat_nn)).tocsr()

        #Run Leiden to get clusters based on sym_densadapted_affmat_nn
        clusterer = cluster.core.LeidenClusterParallel(
                n_jobs=self.n_cores, 
                affmat_transformer=None,
                numseedstotry=self.leiden_numseedstotry,
                n_leiden_iterations=-1,
                refine=True,
                verbose=self.verbose)
        recluster_idxs = clusterer(
                            orig_affinity_mat=csr_sym_density_adapted_affmat,
                            initclusters=motifmemberships).cluster_indices
        if (self.verbose):
            print("Number of reclustered idxs:", len(set(recluster_idxs)))

        oldandreclust_pairs = set(zip(recluster_idxs, motifmemberships))
        #sanity check that 'recluster_idxs' are a stict subset of the original
        # motif memberships
        print(oldandreclust_pairs)
        assert len(oldandreclust_pairs)==len(set(recluster_idxs))
        reclusteridxs_to_motifidx = dict([
            (pair[0], pair[1])
            for pair in oldandreclust_pairs]) 
        assert np.max(np.abs(np.array([reclusteridxs_to_motifidx[x]
                      for x in recluster_idxs])-motifmemberships))==0

        if (self.verbose):
            print("Preparing modularity scorer")

        #Set up machinery needed to score modularity delta.
        self.modularity_scorer = util.ModularityScorer( 
            clusters=recluster_idxs, nn=sym_seqlet_neighbors,
            affmat_nn=sym_densadapted_affmat_nn,
            cluster_to_supercluster_mapping=reclusteridxs_to_motifidx
        )

    def densadapt_wrt_motifseqlets(self, new_rows_distmat_nn, new_rows_nn,
                                         new_rows_betas, new_rows_normfactors):
        new_rows_densadapted_affmat_nn = []
        for i in range(len(new_rows_distmat_nn)):
            densadapted_row = []
            for j,distance in zip(new_rows_nn[i], new_rows_distmat_nn[i]):
                densadapted_row.append(np.sqrt(
                  (np.exp(-distance/new_rows_betas[i])/new_rows_normfactors[i])
                 *(np.exp(-distance/self.motifseqlet_betas[j])/
                   self.motifseqlet_normfactors[j]))) 
            new_rows_densadapted_affmat_nn.append(densadapted_row)
        return new_rows_densadapted_affmat_nn

    def __call__(self, seqlets, hits_to_return_per_seqlet=1,
                       min_mod_precision=0, trim_to_central=None):
        
        fine_affmat_nn, seqlet_neighbors =\
            self.get_similarities_to_motifseqlets(
                   seqlets=seqlets, 
                   trim_to_central=trim_to_central)
        classpattern_simsandalnmnts = self.get_similarities_to_classpatterns(
                                seqlets=seqlets,
                                trim_to_central=trim_to_central)

        (fann_perclassum, fann_perclassavg) = (
            self.get_classwise_fine_affmat_nn_sumavg(
                fine_affmat_nn=fine_affmat_nn,
                seqlet_neighbors=seqlet_neighbors))

        #Map aff to dist
        distmat_nn = self.aff_to_dist_mat(affinity_mat=fine_affmat_nn) 

        betas_and_ps = Parallel(n_jobs=self.n_cores)(
                 delayed(util.binary_search_perplexity)(
                      self.perplexity, distances)
                 for distances in distmat_nn)
        betas = np.array([x[0] for x in betas_and_ps])
        del betas_and_ps

        #also compute the normalization factor needed to get probs to sum to 1
        #note: sticking to lists here because in the future I could
        # have an implementation where different rows of
        # distmat_nn may have different lengths (e.g. when considering
        # a set of initial cluster assigments produced by another method) 
        densadapted_affmat_nn_unnorm = [np.exp(-np.array(distmat)/beta)
                                        for distmat, beta in
                                        zip(distmat_nn, betas)]
        normfactors = np.array([max(np.sum(x),1e-8)
                                for x in densadapted_affmat_nn_unnorm])

        new_rows_densadapted_affmat_nn = self.densadapt_wrt_motifseqlets(
                new_rows_distmat_nn=distmat_nn,
                new_rows_nn=seqlet_neighbors,
                new_rows_betas=betas,
                new_rows_normfactors=normfactors)

        argmax_classes, mod_percentiles, mod_precisions, mod_deltas =\
            self.modularity_scorer(
                new_rows_affmat_nn=new_rows_densadapted_affmat_nn,
                new_rows_nn=seqlet_neighbors,
                hits_to_return_per_input=hits_to_return_per_seqlet) 

        fann_perclasssum_perc = (self.fann_perclasssum_precscorer.
            score_percentile(score=
                fann_perclassum[np.arange(len(argmax_classes))[:,None],
                                argmax_classes].ravel(),
                top_class=argmax_classes.ravel()).reshape(
                    argmax_classes.shape))
        
        fann_perclassavg_perc = (self.fann_perclassavg_precscorer.
            score_percentile(score=
                fann_perclassavg[np.arange(len(argmax_classes))[:,None],
                                argmax_classes].ravel(),
                top_class=argmax_classes.ravel()).reshape(
                    argmax_classes.shape))

        all_seqlet_hits = []
        for i in range(len(argmax_classes)):
            this_seqlet_hits = []
            for class_rank,class_idx in enumerate(argmax_classes[i]):
                if (mod_precisions[i][class_rank] > min_mod_precision):
                    seqlet = seqlets[i]
                    mappedtomotif = self.class_patterns[class_idx]
                    (sim, alignment, isfwd) =\
                        classpattern_simsandalnmnts[i][class_idx]
                    rc = (isfwd==False)
                    alignment = int(alignment)
                    trim_indices = self.classpattern_trimindices[class_idx]
                    #give the start and end in terms of the forward strand
                    fullpattern_start = (seqlet.coor.start+alignment
                           if seqlet.coor.is_revcomp==False
                           else (seqlet.coor.end-alignment)-len(mappedtomotif))
                    fullpattern_end = fullpattern_start+len(mappedtomotif)
                    if ((not rc) and seqlet.coor.is_revcomp==False
                        or (rc and seqlet.coor.is_revcomp)):
                        #if the fwd strand of the seqlet would have aligned
                        # to the forward motif...
                        trim_start = fullpattern_start + trim_indices[0]
                    else:
                        trim_start = fullpattern_start + (
                            len(mappedtomotif)-trim_indices[1])
                    motif_hit = MotifMatch(
                     patternidx=class_idx,
                     patternidx_rank=class_rank,
                     exampleidx=seqlet.coor.example_idx,
                     start=fullpattern_start,
                     end=fullpattern_end,
                     trim_start=trim_start,
                     trim_end=trim_start+(trim_indices[1]-trim_indices[0]),
                     is_revcomp=seqlet.coor.is_revcomp if (not rc)
                                else (seqlet.coor.is_revcomp==False),
                     seqlet_orig_start=seqlet.coor.start,
                     seqlet_orig_end=seqlet.coor.end,
                     seqlet_orig_revcomp=seqlet.coor.is_revcomp,
                     aggregate_sim=sim,
                     mod_delta=mod_deltas[i][class_rank],
                     mod_precision=mod_precisions[i][class_rank],
                     mod_percentile=mod_percentiles[i][class_rank],
                     fann_perclasssum_perc=fann_perclasssum_perc[i][class_rank],
                     fann_perclassavg_perc=fann_perclassavg_perc[i][class_rank])
                    this_seqlet_hits.append(motif_hit) 
            all_seqlet_hits.append(this_seqlet_hits)

        #organize by example/patternidx
        #Remove duplicate motif matches that can occur due to overlapping seqlets
        unique_motifmatches = dict()
        duplicates_found = 0
        for seqlet_hits in all_seqlet_hits:
            for motifmatch in seqlet_hits:
                match_identifier = (motifmatch.patternidx, motifmatch.exampleidx,
                                    motifmatch.start, motifmatch.end,
                                    motifmatch.is_revcomp)
                if match_identifier not in unique_motifmatches:
                    unique_motifmatches[match_identifier] = motifmatch
                else:
                    if (motifmatch.mod_percentile >
                        unique_motifmatches[match_identifier].mod_percentile):
                        unique_motifmatches[match_identifier] = motifmatch 
                    duplicates_found += 1
        print("Removed",duplicates_found,"duplicates")

        patternidx_to_matches = defaultdict(list)
        exampleidx_to_matches = defaultdict(list)
        for motifmatch in unique_motifmatches.values():
            patternidx_to_matches[motifmatch.patternidx].append(motifmatch)
            exampleidx_to_matches[motifmatch.exampleidx].append(motifmatch)

        return (all_seqlet_hits, patternidx_to_matches, exampleidx_to_matches)

    
MotifMatch = namedtuple("MotifMatch", 
    ["patternidx", "patternidx_rank", "exampleidx",
     "start", "end", "trim_start", "trim_end", "is_revcomp",
     "seqlet_orig_start", "seqlet_orig_end", "seqlet_orig_revcomp",
     "aggregate_sim",
     "mod_delta", "mod_precision", "mod_percentile",
     "fann_perclasssum_perc", "fann_perclassavg_perc"])

MotifMatchWithImportance = namedtuple("MotifMatchWithImportance", 
    ["patternidx", "patternidx_rank", "total_importance", "exampleidx",
     "start", "end", "trim_start", "trim_end", "is_revcomp",
     "seqlet_orig_start", "seqlet_orig_end", "seqlet_orig_revcomp",
     "aggregate_sim",
     "mod_delta", "mod_precision", "mod_percentile",
     "fann_perclasssum_perc", "fann_perclassavg_perc"])


