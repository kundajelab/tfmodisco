from __future__ import division, print_function, absolute_import
from . import affinitymat as affmat
from . import cluster
from . import aggregator
from . import core
from . import util
from collections import defaultdict, OrderedDict
from sklearn.neighbors import NearestNeighbors
import numpy as np
import itertools
import time
import sys


class SeqletsToPatternsResults(object):

    def __init__(self, patterns, **kwargs):
        self.patterns = patterns
        self.__dict__.update(**kwargs)


class AbstractSeqletsToPatterns(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class SeqletsToPatterns(AbstractSeqletsToPatterns):

    def __init__(self, track_set,
                       onehot_track_name,
                       contrib_scores_track_names,
                       hypothetical_contribs_track_names,
                       track_signs,

                       n_cores=20,
                       min_overlap_while_sliding=0.7,

                       #gapped kmer embedding arguments
                       alphabet_size=4,
                       kmer_len=8, num_gaps=3, num_mismatches=2,
                       embedding_track_names_and_signs=\
                        embedding_track_names_and_signs,
                       onehot_track_name="sequence",

                       nn_n_jobs=4,
                       nearest_neighbors_multiplier=3,

                       affmat_correlation_threshold=0.15,

                       tsne_perplexities = [10],
                       louvain_min_cluster_size=10,
                       louvain_level_to_return=1,

                       frac_support_to_trim_to=0.2,
                       trim_to_window_size=30,
                       initial_flank_to_add=10,

                       neg_kldiv_merging_threshold=-0.25,
                       crosscorr_similarity_merging_threshold=None,

                       per_track_min_similarity_for_seqlet_assignment=0.1,
                       final_min_cluster_size=10,

                       final_flank_to_add=10,
                       verbose=True,
                       batch_size=50):

        #mandatory arguments
        self.track_set = track_set
        self.onehot_track_name = onehot_track_name
        self.contrib_scores_track_names = contrib_scores_track_names
        self.hypothetical_contribs_track_names =\
              hypothetical_contribs_track_names
        self.track_signs = track_signs

        assert len(track_signs)==len(hypothetical_contribs_track_names)
        assert len(track_signs)==len(contrib_scores_track_names)

        #affinity_mat calculation
        self.n_cores = n_cores
        self.min_overlap_while_sliding = min_overlap_while_sliding

        #gapped kmer embedding arguments
        self.alphabet_size = alphabet_size
        self.kmer_len = kmer_len
        self.num_gaps = num_gaps
        self.num_mismatches = num_mismatches

        self.nn_n_jobs = nn_n_jobs
        self.nearest_neighbors_multiplier = nearest_neighbors_multiplier

        self.affmat_correlation_threshold = affmat_correlation_threshold

        #affinity mat to tsne dist mat setting
        self.tsne_perplexity = tsne_perplexity

        #clustering settings
        self.louvain_min_cluster_size = louvain_min_cluster_size
        self.louvain_level_to_return = louvain_level_to_return

        #postprocessor1 settings
        self.frac_support_to_trim_to = frac_support_to_trim_to
        self.trim_to_window_size = trim_to_window_size
        self.initial_flank_to_add = initial_flank_to_add

        #reassignment settings
        self.per_track_min_similarity_for_seqlet_assignment =\
            per_track_min_similarity_for_seqlet_assignment
        self.final_min_cluster_size = final_min_cluster_size

        #merging settings
        self.neg_kldiv_merging_threshold = neg_kldiv_merging_threshold
        if (crosscorr_similarity_merging_threshold is None):
            crosscorr_similarity_merging_threshold = 0.85*len(track_names)
        self.crosscorr_similarity_merging_threshold =\
                crosscorr_similarity_merging_threshold

        #final postprocessor settings
        self.final_flank_to_add=final_flank_to_add

        #other settings
        self.verbose = verbose
        self.batch_size = batch_size

        self.build() 

    def build(self):

        self.pattern_comparison_settings =\
            affmat.core.PatternComparisonSettings(
                track_names=self.track_names,                                     
                track_transformer=affmat.L1Normalizer(),   
                min_overlap=self.min_overlap_while_sliding)

        #gapped kmer embedder
        self.gkmer_embedder = affmat.core.GappedKmerEmbedder(
            alphabet_size=self.alphabet_size,
            kmer_len=self.kmer_len,
            num_gaps=self.num_gaps,
            num_mismatches=self.num_mismatches,
            num_filters_to_retain=None,
            onehot_track_name=self.onehot_track_name,
            toscore_track_names_and_signs=list(
                zip(self.hypothetical_contribs_track_names,
                    [np.sign(x) for x in self.track_signs])),
            normalizer=affmat.core.MeanNormalizer())

        #affinity matrix from embeddings
        self.affinity_mat_from_seqlet_embeddings =\
            affmat.core.AffmatFromSeqletEmbeddings(
                seqlets_to_1d_embedder=self.gkmer_embedder,
                affinity_mat_from_1d=\
                    affmat.core.NumpyCosineSimilarity(verbose=self.verbose),
                verbose=self.verbose)

        self.nearest_neighbors_object = NearestNeighbors( 
            algorithm="brute", metric="precomputed",
            n_jobs=self.nn_n_jobs)

        self.affmat_from_seqlets_with_nn_pairs =\
            affmat.core.AffmatFromSeqletsWithNNpairs(
                pattern_comparison_settings=self.pattern_comparison_settings,
                sim_metric_on_nn_pairs=\
                    affmat.core.ParallelCpuCrossMetricOnNNpairs(
                        n_cores=self.n_cores,
                        cross_metric_single_region=
                            affmat.core.CrossContinJaccardSingleRegion()))

        self.affinity_mat_from_seqlets =\
            affmat.core.MaxCrossMetricAffinityMatrixFromSeqlets(                       
                pattern_comparison_settings=self.pattern_comparison_settings,
                cross_metric=affmat.core.CrossContinJaccardMultiCoreCPU(
                             verbose=True, n_cores=self.n_cores))

        self.filter_mask_from_correlation =\
            affmat.core.FilterMaskFromCorrelation(
                correlation_threshold=self.affmat_correlation_threshold,
                verbose=self.verbose)

        self.aff_to_dist_mat = affmat.transformers.AffToDistViaInvLogistic() 

        self.tsne_conditional_probs_transformers =\
            [affmat.transformers.TsneConditionalProbs(
                    perplexity=perplexity, aff_to_dist_mat=aff_to_dist_mat)
             for perplexity in self.tsne_perplexities]

        self.clusterer = cluster.core.LouvainCluster(
            level_to_return=-1,
            affmat_transformer=None,
            min_cluster_size=self.louvain_min_cluster_size,
            verbose=self.verbose)

        self.expand_trim_expand1 =\
            aggregator.ExpandSeqletsToFillPattern(
                track_set=self.track_set,
                flank_to_add=self.initial_flank_to_add).chain(
            aggregator.TrimToBestWindow(
                window_size=self.trim_to_window_size,
                track_names=self.track_names)).chain(
            aggregator.ExpandSeqletsToFillPattern(
                track_set=self.track_set,
                flank_to_add=self.initial_flank_to_add))

        self.postprocessor1 =\
            aggregator.TrimToFracSupport(
                        frac=self.frac_support_to_trim_to).chain(
            self.expand_trim_expand1)

        self.seqlet_aggregator = aggregator.GreedySeqletAggregator(
            pattern_aligner=core.CrossContinJaccardPatternAligner(
            pattern_comparison_settings=self.pattern_comparison_settings),
            seqlet_sort_metric=
                lambda x: -sum([np.sum(np.abs(x[track_name].fwd)) for
                           track_name in self.track_names]),
            postprocessor=self.postprocessor1)

        self.patterns_to_pattern_sim_computer =\
            affmat.core.AffmatFromSeqletsWithNNpairs(
                pattern_comparison_settings=self.pattern_comparison_settings,
                sim_metric_on_nn_pairs=\
                    affmat.core.ParallelCpuCrossMetricOnNNpairs(
                        n_cores=self.n_cores,
                        cross_metric_single_region=\
                            affmat.core.CrossContinJaccardSingleRegion(),
                        verbose=self.verbose))

        self.min_similarity_for_seqlet_assignment =\
            (len(self.track_names)*
             self.per_track_min_similarity_for_seqlet_assignment)

        self.seqlet_reassigner =\
           aggregator.ReassignSeqletsFromSmallClusters(
            seqlet_assigner=aggregator.AssignSeqletsByBestMetric(
                pattern_comparison_settings=self.pattern_comparison_settings,
                individual_aligner_metric=
                    core.get_best_alignment_crosscontinjaccard,
                matrix_affinity_metric=
                    affmat.core.CrossContinJaccardMultiCoreCPU(
                        verbose=self.verbose, n_cores=self.n_cores)),
            min_cluster_size=self.final_min_cluster_size,
            postprocessor=self.expand_trim_expand1,
            verbose=self.verbose) 

        self.final_postprocessor = aggregator.ExpandSeqletsToFillPattern(
                                        track_set=self.track_set,
                                        flank_to_add=self.final_flank_to_add) 

    def __call__(self, seqlets):

        start = time.time()

        if (self.verbose):
            print("Computing affinity matrix using seqlet embeddings")
            sys.stdout.flush()

        affinity_mat_from_seqlet_embeddings =\
            self.affinity_mat_from_seqlet_embeddings(seqlets)

        nn_start = time.time() 
        if (self.verbose):
            print("Compute nearest neighbors using affmat embeddings")
            sys.stdout.flush()

        seqlet_neighbors =\
            self.nearest_neighbors_object.fit(
                -affinity_mat_from_seqlet_embeddings).kneighbors(
                X=-affinity_mat_from_seqlet_embeddings,
                n_neighbors=
                  min(self.nearest_neighbors_multiplier
                      *3*self.tsne_perplexity+1,
                      len(seqlets)),
                return_distance=False)

        if (self.verbose):
            print("Computed nearest neighbors in",
                  round(time.time()-nn_start,2),"s")
            sys.stdout.flush()

        nn_affmat_start = time.time() 
        if (self.verbose):
            print("Computing affinity matrix on nearest neighbors")
            sys.stdout.flush()
        nn_affmat = self.affmat_from_seqlets_with_nn_pairs(
                                    seqlet_neighbors=seqlet_neighbors,
                                    seqlets=seqlets) 
        if (self.verbose):
            print("Computed affinity matrix on nearest neighbors in",
                  round(time.time()-nn_affmat_start,2),"s")

        #filter by correlation
        filtered_rows_mask = self.filter_mask_from_correlation(
                            main_affmat=nn_affmat,
                            other_affmat=affinity_mat_from_seqlet_embeddings) 

        filtered_seqlets = [x[0] for x in
                            zip(seqlets, filtered_rows_mask) if (x[1])]
        filtered_final_affmat =\
            nn_affmat[filtered_rows_mask][:,filtered_rows_mask]

        if (self.verbose):
            print("Computing tsne conditional probs")
            sys.stdout.flush() 

        multiscale_tsne_conditional_probs =\
            np.mean([conditional_prob_transformer(filtered_final_affmat)
                     for tsne_conditional_prob_transformer in
                         self.tsne_conditional_probs_transformers], axis=0)

        if (self.verbose):
            print("Computing clustering")
            sys.stdout.flush()
        cluster_results = self.clusterer(multiscale_tsne_conditional_probs)

        num_clusters = max(cluster_results.cluster_indices+1)
        if (self.verbose):
            print("Got "+str(num_clusters)+" clusters initially")
            sys.stdout.flush()

        if (self.verbose):
            print("Aggregating seqlets in each cluster")
            sys.stdout.flush()

        cluster_to_eliminated_motif = OrderedDict()
        cluster_to_motif = OrderedDict()
        for i in range(num_clusters):
            if (self.verbose):
                print("Aggregating for cluster "+str(i))
                sys.stdout.flush()
            motifs = self.seqlet_aggregator(cluster_to_seqlets[i])
            motif = motifs[0]
            motif_track_signs = [
                np.sign(np.sum(motif[contrib_scores_track_name].fwd)) for
                contrib_scores_track_name in self.contrib_scores_track_names] 
            if (all([(x==y) for x,y in
                    zip(motif_track_signs, self.track_signs)])):
                cluster_to_motif[i] = motifs[0]
            else:
                if (self.verbose):
                    print("Dropping cluster "+str(i)+
                          "with "+str(len(motifs[0].seqlets_and_alnmts))
                          +" due to sign disagreement")
                cluster_to_eliminated_motif[i] = motifs[0]
         

        
        #if (self.verbose):
        #    print("Performing seqlet reassignment for small clusters")
        #    sys.stdout.flush()
        #patterns = self.seqlet_reassigner(patterns)
        #patterns = self.final_postprocessor(patterns)

        #if (self.verbose):
        #    print("Got "+str(len(patterns))+" patterns after reassignment")
        #    sys.stdout.flush()

        if (self.verbose):
            print("Total time taken is "
                  +str(round(time.time()-start,2))+"s")
            sys.stdout.flush()

        results = SeqletsToPatternsResults(
            patterns=patterns,
            affinity_mat_from_seqlet_embeddings=\
                affinity_mat_from_seqlet_embeddings,
            seqlet_neighbors=seqlet_neighbors,
            nn_affmat=nn_affmat,   
            filtered_rows_mask=filtered_rows_mask,
            filtered_seqlets=filtered_seqlets,
            final_affmat=final_affmat,
            cluster_results=cluster_results,
            cluster_to_seqlets=cluster_to_seqlets,
            cluster_to_aggregated_seqlets=cluster_to_aggregated_seqlets)

        return results 

