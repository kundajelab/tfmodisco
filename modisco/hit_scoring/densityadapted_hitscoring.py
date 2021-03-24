from __future__ import division, print_function
from collections import defaultdict, OrderedDict, namedtuple
import numpy as np
import time
from .. import affinitymat
from .. import util
from joblib import Parallel, delayed


class CoreDensityAdaptedSeqletScorer(object):

    #patterns should already be subsampled 
    def __init__(self, patterns,
                       coarsegrained_seqlet_embedder,
                       coarsegrained_topn,
                       affmat_from_seqlets_with_nn_pairs,
                       aff_to_dist_mat,
                       perplexity,
                       n_cores,
                       verbose): 
        self.patterns = patterns
        self.coarsegrained_seqlet_embedder = coarsegrained_seqlet_embedder
        self.coarsegrained_topn = coarsegrained_topn
        self.affmat_from_seqlets_with_nn_pairs =\
            affmat_from_seqlets_with_nn_pairs
        self.aff_to_dist_mat = aff_to_dist_mat
        self.perplexity = perplexity
        self.n_cores = n_cores
        self.verbose = verbose
        self.build()

    def build(self):
        #for all the seqlets in the patterns, use the       
        # coarsegrained_seqlet_embedder to compute the embeddings for fwd and
        # rev sequences. *store it*

        motifseqlets = [seqlet for pattern in self.patterns
                               for seqlet in pattern.seqlets]
        self.motifseqlets = motifseqlets

        motifmemberships = np.array([i for i in range(len(self.patterns))
                                     for j in pattern.seqlets])
        orig_embedding_fwd, orig_embedding_rev =\
            self.coarsegrained_seqlet_embedder(seqlets=motifseqlets)
        self.orig_embedding_fwd = orig_embedding_fwd
        self.orig_embedding_rev = orig_embedding_rev

        #then find the topk nearest neighbors by cosine sim,
        coarse_affmat_nn, seqlet_neighbors  =\
            affinitymat.core.SparseNumpyCosineSimFromFwdAndRevOneDVecs(
                n_neighbors=self.coarsegrained_topn, verbose=self.verbose)(
                    fwd_vecs=self.orig_embedding_fwd,
                    rev_vecs=self.orig_embedding_rev,
                    initclusters=None)

        #and finegrained sims for the topk nearest neighbors.
        fine_affmat_nn = self.affmat_from_seqlets_with_nn_pairs(
                            seqlets=motifseqlets,
                            filter_seqlets=None,
                            seqlet_neighbors=seqlet_neighbors,
                            return_sparse=True)

        #Map aff to dist
        distmat_nn = self.aff_to_dist_mat(affinity_mat=fine_affmat_nn) 

        sym_seqlet_neighbors, sym_distmat_nn = util.symmetrize_nn_distmat(
            distmat_nn=distmat_nn, nn=seqlet_neighbors)
        del distmat_nn
        del seqlet_neighbors
        
        #Compute beta values for the density adaptation. *store it*
        betas = Parallel(n_jobs=self.n_cores)(
                 delayed(util.binary_search_perplexity)(
                      self.perplexity, distances)
                 for distances in sym_distmat_nn)
        self.motifseqlet_betas = betas
        del betas

        #also compute the normalization factor needed to get probs to sum to 1
        #note: sticking to lists here because different rows of
        # sym_distmat_nn may have different lengths after adding in
        # the symmetric pairs
        densadapted_affmat_nn_unnorm = [np.exp(-np.array(distmat)/beta) in
                                   zip(sym_distmat_nn, self.motifseqlet_betas)]
        normfactors = [np.sum(x) for x in densadapted_affmat_nn_unnorm]
        self.motifseqlet_normfactors = normfactors
        del normfactors
        
        #Do density-adaptation using average of self-Beta and other-Beta.
        sym_densadapted_affmat_nn = self.densadapt_wrt_motifseqlets(
                            new_rows_distmat_nn=sym_distmat_nn,
                            new_rows_nn=sym_seqlet_neighbors,
                            new_rows_betas=self.motifseqlet_betas,
                            new_rows_normfactors=self.motifseqlet_normfactors)

        #Set up machinery needed to score modularity delta.
        self.modularity_scorer = util.ModularityScorer( 
            clusters=motifmemberships,
            nn=sym_seqlet_neighbors,
            affmat_nn=sym_densadapted_affmat_nn)

    def densadapt_wrt_motifseqlets(self, new_rows_distmat_nn, new_rows_nn,
                                         new_rows_betas, new_rows_normfactors):
        new_rows_densadapted_affmat_nn = []
        for i in range(len(new_rows_distmat_nn)):
            densadapted_row = []
            for j,distance in zip(new_rows_nn[i], new_rows_distmat_nn[i]):
                densadapted_row.append(0.5*(
                  (np.exp(-distance/new_rows_betas[i])/new_rows_normfactors[i])
                + (np.exp(-distance/self.motifseqlet_betas[j])/
                   self.motifseqlet_norm_factors[j]))) 
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

        betas = Parallel(n_jobs=self.n_cores)(
                 delayed(util.binary_search_perplexity)(
                      self.perplexity, distances)
                 for distances in distmat_nn)

        #also compute the normalization factor needed to get probs to sum to 1
        #note: sticking to lists here because in the future I could
        # have an implementation where different rows of
        # distmat_nn may have different lengths (e.g. when considering
        # a set of initial cluster assigments produced by another method) 
        densadapted_affmat_nn_unnorm = [np.exp(-np.array(distmat)/beta) in
                                          zip(distmat_nn, betas)]
        normfactors = [np.sum(x) for x in densadapted_affmat_nn_unnorm]

        new_rows_densadapted_affmat_nn = self.densadapt_wrt_motifseqlets(
                new_rows_distmat_nn=distmat_nn,
                new_rows_nn=seqlet_neighbors,
                new_rows_betas=betas,
                new_rows_normfactors=normfactors)

        argmax_classes, precisions, modularity_deltas = self.modularity_scorer(
            new_rows_affmat_nn=new_rows_densadapted_affmat_nn,
            new_rows_nn=seqlet_neighbors) 

        return (argmax_classes, precisions, modularity_deltas)


