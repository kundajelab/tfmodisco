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

                       nn_n_jobs=4,
                       nearest_neighbors_to_compute=500,

                       affmat_correlation_threshold=0.15,

                       tsne_perplexities = [10],
                       louvain_min_cluster_size=10,

                       frac_support_to_trim_to=0.2,
                       trim_to_window_size=30,
                       initial_flank_to_add=10,

                       prob_and_pertrack_sim_merge_thresholds=[
                        (0.0001,0.84), (0.00001, 0.87), (0.000001, 0.9)],

                       prob_and_pertrack_sim_dealbreaker_thresholds=[
                        (0.1,0.75), (0.01, 0.8), (0.001, 0.83),
                        (0.0000001,0.9)],

                       min_similarity_for_seqlet_assignment=0.2,
                       final_min_cluster_size=50,

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
        self.nearest_neighbors_to_compute = nearest_neighbors_to_compute

        self.affmat_correlation_threshold = affmat_correlation_threshold

        #affinity mat to tsne dist mat setting
        self.tsne_perplexities = tsne_perplexities

        #clustering settings
        self.louvain_min_cluster_size = louvain_min_cluster_size

        #postprocessor1 settings
        self.frac_support_to_trim_to = frac_support_to_trim_to
        self.trim_to_window_size = trim_to_window_size
        self.initial_flank_to_add = initial_flank_to_add 

        #similarity settings for merging
        self.prob_and_sim_merge_thresholds =\
            [(x[0], x[1]*2*len(contrib_scores_track_names))
             for x in prob_and_pertrack_sim_merge_thresholds]
        self.prob_and_sim_dealbreaker_thresholds =\
            [(x[0], x[1]*2*len(contrib_scores_track_names))
             for x in prob_and_pertrack_sim_dealbreaker_thresholds]

        #reassignment settings
        self.min_similarity_for_seqlet_assignment =\
            min_similarity_for_seqlet_assignment
        self.final_min_cluster_size = final_min_cluster_size

        #final postprocessor settings
        self.final_flank_to_add=final_flank_to_add

        #other settings
        self.verbose = verbose
        self.batch_size = batch_size

        self.build() 

    def build(self):

        self.pattern_comparison_settings =\
            affmat.core.PatternComparisonSettings(
                track_names=self.hypothetical_contribs_track_names
                            +self.contrib_scores_track_names,                                     
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
                    perplexity=perplexity,
                    aff_to_dist_mat=self.aff_to_dist_mat)
             for perplexity in self.tsne_perplexities]

        #self.clusterer = cluster.core.LouvainCluster(
        #    level_to_return=-1,
        #    affmat_transformer=affmat.transformers.SymmetrizeByAddition(
        #                            probability_normalize=True),
        #    min_cluster_size=self.louvain_min_cluster_size,
        #    verbose=self.verbose)


        self.clusterer = cluster.core.CollectComponents(
            dealbreaker_threshold=0.5,
            join_threshold=0.75,
            transformer=affmat.transformers.SymmetrizeByAddition(
                                    probability_normalize=True).chain(
                                affmat.transformers.LouvainMembershipAverage(
                                    max_runs=100)),
            min_cluster_size=self.louvain_min_cluster_size)

        self.expand_trim_expand1 =\
            aggregator.ExpandSeqletsToFillPattern(
                track_set=self.track_set,
                flank_to_add=self.initial_flank_to_add).chain(
            aggregator.TrimToBestWindow(
                window_size=self.trim_to_window_size,
                track_names=self.contrib_scores_track_names)).chain(
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
                               track_name in self.contrib_scores_track_names]),
            postprocessor=self.postprocessor1)

        self.pattern_to_pattern_sim_computer =\
            affmat.core.AffmatFromSeqletsWithNNpairs(
                pattern_comparison_settings=self.pattern_comparison_settings,
                sim_metric_on_nn_pairs=\
                    affmat.core.ParallelCpuCrossMetricOnNNpairs(
                        n_cores=self.n_cores,
                        cross_metric_single_region=\
                            affmat.core.CrossContinJaccardSingleRegion(),
                        verbose=False))
        self.dynamic_distance_similar_patterns_collapser =\
            aggregator.DynamicDistanceSimilarPatternsCollapser(
                pattern_to_pattern_sim_computer=\
                    self.pattern_to_pattern_sim_computer,
                aff_to_dist_mat=self.aff_to_dist_mat,
                pattern_aligner=core.CrossCorrelationPatternAligner(
                    pattern_comparison_settings=
                        affmat.core.PatternComparisonSettings(
                            track_names=(
                                self.hypothetical_contribs_track_names+
                                self.contrib_scores_track_names), 
                            track_transformer=affmat.MeanNormalizer().chain(
                                              affmat.MagnitudeNormalizer()), 
                            min_overlap=self.min_overlap_while_sliding)),
                collapse_condition=(lambda dist_prob, aligner_sim:
                    any([(dist_prob > x[0] and aligner_sim > x[1])
                         for x in self.prob_and_sim_merge_thresholds])),
                dealbreaker_condition=(lambda dist_prob, aligner_sim:
                    any([(dist_prob < x[0] and aligner_sim < x[1])              
                         for x in self.prob_and_sim_dealbreaker_thresholds])),
                postprocessor=self.postprocessor1,
                verbose=self.verbose) 

        self.seqlet_reassigner =\
           aggregator.ReassignSeqletsFromSmallClusters(
            seqlet_assigner=aggregator.AssignSeqletsByBestMetric(
                pattern_comparison_settings=self.pattern_comparison_settings,
                individual_aligner_metric=
                    core.get_best_alignment_crosscontinjaccard,
                matrix_affinity_metric=
                    affmat.core.CrossContinJaccardMultiCoreCPU(
                        verbose=self.verbose, n_cores=self.n_cores),
                min_similarity=self.min_similarity_for_seqlet_assignment),
            min_cluster_size=self.final_min_cluster_size,
            postprocessor=self.expand_trim_expand1,
            verbose=self.verbose) 

        self.final_postprocessor = aggregator.ExpandSeqletsToFillPattern(
                                        track_set=self.track_set,
                                        flank_to_add=self.final_flank_to_add) 

    def __call__(self, seqlets):

        start = time.time()

        if (self.verbose):
            print("(Round 1) Computing affinity matrix from seqlet embeddings")
            sys.stdout.flush()

        affinity_mat_from_seqlet_embeddings1 =\
            self.affinity_mat_from_seqlet_embeddings(seqlets)

        nn_start1 = time.time() 
        if (self.verbose):
            print("(Round 1) Compute nearest neighbors from affmat embeddings")
            sys.stdout.flush()

        seqlet_neighbors1 =\
            self.nearest_neighbors_object.fit(
                -affinity_mat_from_seqlet_embeddings1).kneighbors(
                X=-affinity_mat_from_seqlet_embeddings1,
                n_neighbors=min(self.nearest_neighbors_to_compute+1,
                                len(seqlets)),
                return_distance=False)

        if (self.verbose):
            print("Computed nearest neighbors in",
                  round(time.time()-nn_start1,2),"s")
            sys.stdout.flush()

        nn_affmat_start1 = time.time() 
        if (self.verbose):
            print("(Round 1) Computing affinity matrix on nearest neighbors")
            sys.stdout.flush()
        nn_affmat1 = self.affmat_from_seqlets_with_nn_pairs(
                                    seqlet_neighbors=seqlet_neighbors1,
                                    seqlets=seqlets) 
        if (self.verbose):
            print("(Round 1) Computed affinity matrix on nearest neighbors in",
                  round(time.time()-nn_affmat_start1,2),"s")
            sys.stdout.flush()

        #filter by correlation
        filtered_rows_mask1 = self.filter_mask_from_correlation(
                            main_affmat=nn_affmat1,
                            other_affmat=affinity_mat_from_seqlet_embeddings1) 

        filtered_seqlets1 = [x[0] for x in
                             zip(seqlets, filtered_rows_mask1) if (x[1])]
        filtered_affmat1 =\
            nn_affmat1[filtered_rows_mask1][:,filtered_rows_mask1]

        if (self.verbose):
            print("(Round 1) Computing tsne conditional probs")
            sys.stdout.flush() 

        multiscale_tsne_conditional_probs1 =\
            np.mean([tsne_conditional_prob_transformer(filtered_affmat1)
                     for tsne_conditional_prob_transformer in
                         self.tsne_conditional_probs_transformers], axis=0)

        if (self.verbose):
            print("(Round 1) Computing clustering")
            sys.stdout.flush()
        cluster_results1 = self.clusterer(multiscale_tsne_conditional_probs1)

        num_clusters1 = max(cluster_results1.cluster_indices+1)
        if (self.verbose):
            print("Got "+str(num_clusters1)+" clusters after round 1")
            sys.stdout.flush()

        if (self.verbose):
            print("(Round 1) Aggregating seqlets in each cluster")
            sys.stdout.flush()

        cluster_to_seqlets1 = defaultdict(list) 
        assert len(filtered_seqlets1)==len(cluster_results1.cluster_indices)
        for seqlet,idx in zip(filtered_seqlets1,
                              cluster_results1.cluster_indices):
            cluster_to_seqlets1[idx].append(seqlet)

        cluster_to_eliminated_motif1 = OrderedDict()
        cluster_to_motif1 = OrderedDict()
        for i in range(num_clusters1):
            if (self.verbose):
                print("Aggregating for cluster "+str(i)+" with "
                      +str(len(cluster_to_seqlets1[i]))+" seqlets")
                sys.stdout.flush()
            motifs = self.seqlet_aggregator(cluster_to_seqlets1[i])
            motif = motifs[0]
            motif_track_signs = [
                np.sign(np.sum(motif[contrib_scores_track_name].fwd)) for
                contrib_scores_track_name in self.contrib_scores_track_names] 
            if (all([(x==y) for x,y in
                    zip(motif_track_signs, self.track_signs)])):
                cluster_to_motif1[i] = motif
            else:
                if (self.verbose):
                    print("Dropping cluster "+str(i)+
                          " with "+str(motif.num_seqlets)
                          +" seqlets due to sign disagreement")
                cluster_to_eliminated_motif1[i] = motif

        #Do another round of affinity matrix calculation and clustering on the
        #seqlets from the newly recentered motifs
        if (self.verbose):
            print("Beginning round 2")
            sys.stdout.flush()
        
        seqlets2 = list(itertools.chain(*[[x.seqlet for x in                 
                    motif.seqlets_and_alnmts] for motif in
                    cluster_to_motif1.values()]))

        if (self.verbose):
            print("(Round 2) Computing affinity matrix from seqlet embeddings")
            sys.stdout.flush()

        affinity_mat_from_seqlet_embeddings2 =\
            self.affinity_mat_from_seqlet_embeddings(seqlets2)

        if (self.verbose):
            print("(Round 2) Compute nearest neighbors from affmat embeddings")
            sys.stdout.flush()

        seqlet_neighbors2 =\
            self.nearest_neighbors_object.fit(
                -affinity_mat_from_seqlet_embeddings2).kneighbors(
                X=-affinity_mat_from_seqlet_embeddings2,
                n_neighbors=min(self.nearest_neighbors_to_compute+1,
                                len(seqlets2)),
                return_distance=False)

        nn_affmat_start2 = time.time() 
        if (self.verbose):
            print("(Round 2) Computing affinity matrix on nearest neighbors")
            sys.stdout.flush()
        nn_affmat2 = self.affmat_from_seqlets_with_nn_pairs(
                                    seqlet_neighbors=seqlet_neighbors2,
                                    seqlets=seqlets2) 
        if (self.verbose):
            print("(Round 2) Computed affinity matrix on nearest neighbors in",
                  round(time.time()-nn_affmat_start2,2),"s")
            sys.stdout.flush()

        if (self.verbose):
            print("(Round 2) Computing tsne conditional probs")
            sys.stdout.flush() 

        multiscale_tsne_conditional_probs2 =\
            np.mean([tsne_conditional_prob_transformer(nn_affmat2)
                     for tsne_conditional_prob_transformer in
                         self.tsne_conditional_probs_transformers], axis=0)

        if (self.verbose):
            print("(Round 2) Computing clustering")
            sys.stdout.flush()
        cluster_results2 = self.clusterer(multiscale_tsne_conditional_probs2)

        num_clusters2 = max(cluster_results2.cluster_indices+1)
        if (self.verbose):
            print("Got "+str(num_clusters2)+" clusters after round 2")
            sys.stdout.flush()

        if (self.verbose):
            print("(Round 2) Aggregating seqlets in each cluster")
            sys.stdout.flush()

        cluster_to_seqlets2 = defaultdict(list) 
        assert len(seqlets2)==len(cluster_results2.cluster_indices)
        for seqlet,idx in zip(seqlets2,
                              cluster_results2.cluster_indices):
            cluster_to_seqlets2[idx].append(seqlet)

        cluster_to_eliminated_motif2 = OrderedDict()
        cluster_to_motif2 = OrderedDict()
        for i in range(num_clusters2):
            if (self.verbose):
                print("Aggregating for cluster "+str(i)+" with "
                      +str(len(cluster_to_seqlets2[i]))+" seqlets")
                sys.stdout.flush()
            motifs = self.seqlet_aggregator(cluster_to_seqlets2[i])
            motif = motifs[0]
            motif_track_signs = [
                np.sign(np.sum(motif[contrib_scores_track_name].fwd)) for
                contrib_scores_track_name in self.contrib_scores_track_names] 
            if (all([(x==y) for x,y in
                    zip(motif_track_signs, self.track_signs)])):
                cluster_to_motif2[i] = motif
            else:
                if (self.verbose):
                    print("Dropping cluster "+str(i)+
                          " with "+str(motif.num_seqlets)
                          +" seqlets due to sign disagreement")
                cluster_to_eliminated_motif2[i] = motif

        #Now start merging patterns 
        if (self.verbose):
            print("Merging clusters")
            sys.stdout.flush()
        merged_patterns = self.dynamic_distance_similar_patterns_collapser( 
            patterns=cluster_to_motif2.values(),
            seqlets=seqlets2) 
        merged_patterns = sorted(merged_patterns, key=lambda x: -x.num_seqlets)
        if (self.verbose):
            print("Got "+str(len(merged_patterns))+" patterns after merging")
            sys.stdout.flush()
        
        if (self.verbose):
            print("Performing seqlet reassignment for small clusters")
            sys.stdout.flush()
        too_small_patterns = [x for x in merged_patterns if
                              x.num_seqlets < self.final_min_cluster_size]
        final_patterns = self.seqlet_reassigner(merged_patterns)
        final_patterns = self.final_postprocessor(final_patterns)
        if (self.verbose):
            print("Got "+str(len(final_patterns))
                  +" patterns after reassignment")
            sys.stdout.flush()

        if (self.verbose):
            print("Total time taken is "
                  +str(round(time.time()-start,2))+"s")
            sys.stdout.flush()

        results = SeqletsToPatternsResults(
            patterns=final_patterns,

            affinity_mat_from_seqlet_embeddings1=\
                affinity_mat_from_seqlet_embeddings1,
            seqlet_neighbors1=seqlet_neighbors1,
            nn_affmat1=nn_affmat1,   
            filtered_mask1=filtered_rows_mask1,
            filtered_seqlets1=filtered_seqlets1,
            filtered_affmat1=filtered_affmat1,
            cluster_results1=cluster_results1,
            cluster_to_motif1=cluster_to_motif1,

            seqlets2=seqlets2,
            affinity_mat_from_seqlet_embeddings2=\
                affinity_mat_from_seqlet_embeddings2,
            seqlet_neighbors2=seqlet_neighbors2,
            nn_affmat2=nn_affmat2,   
            cluster_results2=cluster_results2,
            cluster_to_motif2=cluster_to_motif2,
            cluster_to_eliminated_motif2=cluster_to_eliminated_motif2,

            merged_patterns=merged_patterns,
            too_small_patterns=too_small_patterns,
            final_patterns=final_patterns)

        return results 

