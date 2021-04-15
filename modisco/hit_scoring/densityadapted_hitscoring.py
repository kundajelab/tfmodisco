from __future__ import division, print_function
from collections import defaultdict, OrderedDict, namedtuple
import numpy as np
import time
from .. import affinitymat
from .. import util
from .. import cluster
from .. import aggregator
from .. import seqlet_embedding
from .. import affinitymat
from joblib import Parallel, delayed


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
        pattern_to_superpattern_mapping=subpattern_to_superpattern_mapping,
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
                       pattern_to_superpattern_mapping=None,
                       leiden_numseedstotry=50,
                       verbose=True): 
        self.patterns = patterns
        if (pattern_to_superpattern_mapping is None):
            pattern_to_superpattern_mapping = dict([
                                          (i,i) for i in range(len(patterns))])
        self.pattern_to_superpattern_mapping = pattern_to_superpattern_mapping
        self.coarsegrained_seqlet_embedder = coarsegrained_seqlet_embedder
        self.coarsegrained_topn = coarsegrained_topn
        self.affmat_from_seqlets_with_nn_pairs =\
            affmat_from_seqlets_with_nn_pairs
        self.aff_to_dist_mat = aff_to_dist_mat
        self.perplexity = perplexity
        self.n_cores = n_cores
        self.leiden_numseedstotry = leiden_numseedstotry
        self.verbose = verbose
        self.build()

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
            clusters=recluster_idxs,
            nn=sym_seqlet_neighbors,
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

    def __call__(self, seqlets):
        
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

        argmax_classes, precisions, percentiles, modularity_deltas =\
            self.modularity_scorer(
                new_rows_affmat_nn=new_rows_densadapted_affmat_nn,
                new_rows_nn=seqlet_neighbors) 

        return (argmax_classes, precisions, percentiles, modularity_deltas)


