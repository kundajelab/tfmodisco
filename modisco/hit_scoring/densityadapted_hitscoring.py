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
from joblib import Parallel, delayed


class MakeHitScorer(object):

    def __init__(self, patterns, target_seqlet_size,
                       bg_freq,
                       task_names_and_signs,
                       n_cores,
                       additional_trimandsubcluster_kwargs={},
                       additional_seqletscorer_kwargs={}):

        self.target_seqlet_size = target_seqlet_size

        onehot_track_name = "sequence" #the default

        task_names = [x[0] for x in task_names_and_signs]
        self.task_names = task_names
        self.task_names_and_signs = task_names_and_signs

        print("Getting trimmed patterns, subclustering them")

        self.trimmed_subclustered_patterns = trim_and_subcluster_patterns(
            patterns=patterns, window_size=target_seqlet_size,
            onehot_track_name=onehot_track_name,
            task_names=task_names, bg_freq=bg_freq,
            n_cores=n_cores, **additional_trimandsubcluster_kwargs)

        print("Preparing seqlet scorer")

        self.seqlet_scorer = prepare_seqlet_scorer(
            patterns=self.trimmed_subclustered_patterns,
            onehot_track_name=onehot_track_name,
            task_names_and_signs=task_names_and_signs,
            n_cores=n_cores, **additional_seqletscorer_kwargs)

        self.tnt_results = None

    def get_coordproducer_score_track(self, contrib_scores):
        combined_contribs = np.zeros_like(contrib_scores[self.task_names[0]])
        for task_name,sign in self.task_names_and_signs:
            combined_contribs += contrib_scores[task_name]*sign
        return np.sum(combined_contribs, axis=-1)

    def set_coordproducer(self, contrib_scores,
                core_sliding_window_size, target_fdr,
                min_passing_windows_frac,
                max_passing_windows_frac,
                null_track=coordproducers.LaplaceNullDist(num_to_samp=10000),
                **additional_coordproducer_kwargs):

        assert (self.target_seqlet_size-core_sliding_window_size)%2==0,\
                ("Please provide a core_sliding_window_size that is an"
                 +" even number smaller than target_seqlet_size")

        self.coordproducer = coordproducers.VariableWindowAroundChunks(
            sliding=[core_sliding_window_size],
            flank=int((self.target_seqlet_size-core_sliding_window_size)/2.0),
            suppress=core_sliding_window_size,
            target_fdr=target_fdr,
            min_passing_windows_frac=min_passing_windows_frac,
            max_passing_windows_frac=max_passing_windows_frac,
            **additional_coordproducer_kwargs
        ) 

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
                min_mod_precision=min_mod_precision) 

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
                          max_seqlets_per_submotif=100,
                          min_overlap_size=10,
                          crosspattern_perplexity=10,
                          coarsegrained_topn=500,
                          verbose=True):

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

    seqlet_scorer = CoreDensityAdaptedSeqletScorer(
        patterns=subpatterns,
        coarsegrained_seqlet_embedder=(
            seqlet_embedding.advanced_gapped_kmer
               .AdvancedGappedKmerEmbedderFactory()(
                   onehot_track_name=onehot_track_name,
                   toscore_track_names_and_signs=[
                        (x[0]+"_hypothetical_contribs", x[1])
                        for x in task_names_and_signs],
                   n_jobs=n_cores)),
        coarsegrained_topn=coarsegrained_topn, #set to 500 for real seqs
        affmat_from_seqlets_with_nn_pairs=
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
                   affinitymat.core.CrossContinJaccardSingleRegion())),
        aff_to_dist_mat=
            affinitymat.transformers.AffToDistViaInvLogistic(),
        perplexity=crosspattern_perplexity,
        n_cores=n_cores,
        pattern_aligner=core.CrossContinJaccardPatternAligner(              
            pattern_comparison_settings=
                affinitymat.core.PatternComparisonSettings(
                   track_names=track_names, 
                   track_transformer=affinitymat.L1Normalizer(),
                   min_overlap=float(min_overlap_size)/target_seqlet_size)),
        pattern_to_superpattern_mapping=subpattern_to_superpattern_mapping,
        superpatterns=patterns,
        verbose=verbose
    )

    return seqlet_scorer

    
class CoreDensityAdaptedSeqletScorer(object):

    #patterns should already be subsampled 
    def __init__(self, patterns,
                       coarsegrained_seqlet_embedder,
                       coarsegrained_topn,
                       affmat_from_seqlets_with_nn_pairs,
                       aff_to_dist_mat,
                       perplexity,
                       n_cores,
                       pattern_aligner,
                       pattern_to_superpattern_mapping=None,
                       superpatterns=None,
                       leiden_numseedstotry=50,
                       verbose=True): 
        self.patterns = patterns
        if (pattern_to_superpattern_mapping is None):
            pattern_to_superpattern_mapping = dict([
                                          (i,i) for i in range(len(patterns))])
            self.class_patterns = patterns
        else:
            self.class_patterns = superpatterns
        self.pattern_to_superpattern_mapping = pattern_to_superpattern_mapping
        self.coarsegrained_seqlet_embedder = coarsegrained_seqlet_embedder
        self.coarsegrained_topn = coarsegrained_topn
        self.affmat_from_seqlets_with_nn_pairs =\
            affmat_from_seqlets_with_nn_pairs
        self.aff_to_dist_mat = aff_to_dist_mat
        self.perplexity = perplexity
        self.n_cores = n_cores
        self.pattern_aligner = pattern_aligner
        self.leiden_numseedstotry = leiden_numseedstotry
        self.verbose = verbose
        self.build()

    def get_classwise_fine_affmat_nn_sumavg(self,
            fine_affmat_nn, seqlet_neighbors):
        num_classes = max(self.motifmemberships)+1
        #(not used in the density-adapted scoring) for each class, compute
        # the total fine-grained similarity for each class in the topk
        # nearest neighbors. Will be used to instantiate a class-wise
        # precision scorer
        fine_affmat_nn_perclassum = np.zeros(
            (len(fine_affmat_nn), num_classes))
        fine_affmat_nn_perclassavg = np.zeros(
            (len(fine_affmat_nn), num_classes))
        for i in range(len(fine_affmat_nn)):
            for classidx in range(num_classes):
                class_entries = [fine_affmat_nn[i][j] for
                   j in range(len(fine_affmat_nn[i]))
                   if self.motifmemberships[seqlet_neighbors[i][j]]==classidx]
                if (len(class_entries) > 0):
                    fine_affmat_nn_perclassum[i][classidx] =\
                        np.sum(class_entries)
                    fine_affmat_nn_perclassavg[i][classidx] =\
                        np.mean(class_entries)
        return (fine_affmat_nn_perclassum, fine_affmat_nn_perclassavg)

    def build(self):
        #for all the seqlets in the patterns, use the       
        # coarsegrained_seqlet_embedder to compute the embeddings for fwd and
        # rev sequences. *store it*

        motifseqlets = [seqlet for pattern in self.patterns
                               for seqlet in pattern.seqlets]
        self.motifseqlets = motifseqlets

        motifmemberships = np.array([
            self.pattern_to_superpattern_mapping[i]
            for i in range(len(self.patterns))
            for j in self.patterns[i].seqlets])
        self.motifmemberships = motifmemberships
        assert (max(self.motifmemberships)+1) == len(self.class_patterns),\
            (max(self.motifmemberships), len(self.class_patterns))

        if (self.verbose):
            print("Computing coarse-grained embeddings")

        orig_embedding_fwd, orig_embedding_rev =\
            self.coarsegrained_seqlet_embedder(seqlets=motifseqlets)
        self.orig_embedding_fwd = orig_embedding_fwd
        self.orig_embedding_rev = orig_embedding_rev

        if (self.verbose):
            print("Computing coarse top k nn via cosine sim")

        #then find the topk nearest neighbors by cosine sim,
        coarse_affmat_nn, seqlet_neighbors  =\
            affinitymat.core.SparseNumpyCosineSimFromFwdAndRevOneDVecs(
                n_neighbors=self.coarsegrained_topn, verbose=self.verbose)(
                    fwd_vecs=self.orig_embedding_fwd,
                    rev_vecs=self.orig_embedding_rev,
                    initclusters=None)

        if (self.verbose):
            print("Computing fine-grained sim for top k")

        #and finegrained sims for the topk nearest neighbors.
        fine_affmat_nn = self.affmat_from_seqlets_with_nn_pairs(
                            seqlets=motifseqlets,
                            filter_seqlets=None,
                            seqlet_neighbors=seqlet_neighbors,
                            return_sparse=True)

        #fann = fine affmat nn. This is not used for density-adaptive
        # scoring; rather it's a way to get a sense of within-motif
        # similarity WITHOUT the density-adaptation step
        (fann_perclassum, fann_perclassavg) = (
            self.get_classwise_fine_affmat_nn_sumavg(
                fine_affmat_nn=fine_affmat_nn,
                seqlet_neighbors=seqlet_neighbors))
        self.fann_perclasssum_precscorer = util.ClasswisePrecisionScorer(
            true_classes=motifmemberships,
            class_membership_scores=fann_perclassum) 
        self.fann_perclassavg_precscorer = util.ClasswisePrecisionScorer(
            true_classes=motifmemberships,
            class_membership_scores=fann_perclassavg) 
                
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
                                min_mod_precision=0):
        
        embedding_fwd, _ =\
            self.coarsegrained_seqlet_embedder(seqlets=seqlets,
                                               only_compute_fwd=True)

        #then find the topk nearest neighbors by cosine sim
        coarse_affmat_nn, seqlet_neighbors  =\
            affinitymat.core.SparseNumpyCosineSimFromFwdAndRevOneDVecs(
                n_neighbors=self.coarsegrained_topn, verbose=self.verbose)(
                    fwd_vecs=self.orig_embedding_fwd,
                    rev_vecs=self.orig_embedding_rev,
                    initclusters=None,
                    fwd_vecs2=embedding_fwd)

        #and finegrained sims for the topk nearest neighbors.
        fine_affmat_nn = self.affmat_from_seqlets_with_nn_pairs(
                            seqlets=seqlets,
                            filter_seqlets=self.motifseqlets,
                            seqlet_neighbors=seqlet_neighbors,
                            return_sparse=True)

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
                    (alignment, rc, sim) = self.pattern_aligner(
                        parent_pattern=seqlet, child_pattern=mappedtomotif)
                    motif_hit = MotifMatch(
                     patternidx=class_idx,
                     patternidx_rank=class_rank,
                     exampleidx=seqlet.coor.example_idx,
                     start=seqlet.coor.start+alignment
                           if seqlet.coor.is_revcomp==False
                           else (seqlet.coor.end-alignment)-len(mappedtomotif),
                     end=seqlet.coor.start+alignment+len(mappedtomotif)
                         if seqlet.coor.is_revcomp==False
                         else (seqlet.coor.end-alignment),
                     is_revcomp=seqlet.coor.is_revcomp if rc==False
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
     "start", "end", "is_revcomp",
     "seqlet_orig_start", "seqlet_orig_end", "seqlet_orig_revcomp",
     "aggregate_sim",
     "mod_delta", "mod_precision", "mod_percentile",
     "fann_perclasssum_perc", "fann_perclassavg_perc"])

MotifMatchWithImportance = namedtuple("MotifMatchWithImportance", 
    ["patternidx", "patternidx_rank", "total_importance", "exampleidx",
     "start", "end", "is_revcomp",
     "seqlet_orig_start", "seqlet_orig_end", "seqlet_orig_revcomp",
     "aggregate_sim",
     "mod_delta", "mod_precision", "mod_percentile",
     "fann_perclasssum_perc", "fann_perclassavg_perc"])


