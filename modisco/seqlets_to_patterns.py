from __future__ import division, print_function, absolute_import
from . import affinitymat as affmat
from . import cluster
from . import aggregator
from . import core
from collections import defaultdict, OrderedDict
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

    def __init__(self, track_names,
                       track_set,
                       n_cores=20,
                       min_overlap_while_sliding=0.7,
                       tsne_perplexity=50,
                       min_edges_per_row=None, 
                       louvain_min_cluster_size=10,
                       frac_support_to_trim_to=0.2,
                       trim_to_window_size=30,
                       initial_flank_to_add=10,
                       per_track_min_similarity_for_seqlet_assignment=0,
                       final_min_cluster_size=30,
                       similarity_splitting_threshold=0.9,
                       similarity_merging_threshold=None,
                       final_flank_to_add=10,
                       verbose=True,
                       batch_size=50):

        #mandatory arguments
        self.track_names = track_names
        self.track_set = track_set

        #affinity_mat calculation
        self.n_cores = n_cores
        self.min_overlap_while_sliding = min_overlap_while_sliding

        #affinity mat to tsne dist mat setting
        self.tsne_perplexity = tsne_perplexity

        #seqlet filtering based on affinity_mat
        if (min_edges_per_row is None):
            min_edges_per_row = tsne_perplexity/5.0
        self.min_edges_per_row = min_edges_per_row

        #clustering settings
        self.louvain_min_cluster_size = louvain_min_cluster_size

        #postprocessor1 settings
        self.frac_support_to_trim_to = frac_support_to_trim_to
        self.trim_to_window_size = trim_to_window_size
        self.initial_flank_to_add = initial_flank_to_add

        #reassignment settings
        self.per_track_min_similarity_for_seqlet_assignment =\
            per_track_min_similarity_for_seqlet_assignment
        self.final_min_cluster_size = final_min_cluster_size

        #split detection settings
        self.similarity_splitting_threshold = similarity_splitting_threshold

        #merging settings
        if (similarity_merging_threshold is None):
            similarity_merging_threshold = 0.85*len(track_names)
        self.similarity_merging_threshold = similarity_merging_threshold

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

        self.affinity_mat_from_seqlets =\
            affmat.core.MaxCrossMetricAffinityMatrixFromSeqlets(                       
                pattern_comparison_settings=self.pattern_comparison_settings,
                cross_metric=affmat.core.CrossContinJaccardMultiCoreCPU(
                             verbose=True, n_cores=self.n_cores))

        self.tsne_affinitymat_transformer =\
            affmat.transformers.TsneJointProbs(perplexity=self.tsne_perplexity)

        self.filtered_rows_mask_producer =\
           affmat.core.FilterSparseRows(
            affmat_transformer=\
                affmat.transformers.PerNodeThresholdBinarizer(
                 thresholder=affmat.transformers.NonzeroMeanThreshold(
                                expected_nonzeros=self.tsne_perplexity*3))
                 .chain(affmat.transformers.SymmetrizeByMultiplying()),
            min_rows_before_applying_filtering=0,
            min_edges_per_row=self.min_edges_per_row,
            verbose=self.verbose)

        self.clusterer = cluster.core.LouvainCluster(
            affmat_transformer=self.tsne_affinitymat_transformer,
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

        self.seqlet_aggregator = aggregator.HierarchicalSeqletAggregator(
            pattern_aligner=core.CrossContinJaccardPatternAligner(
            pattern_comparison_settings=self.pattern_comparison_settings),
            affinity_mat_from_seqlets=self.affinity_mat_from_seqlets,
            postprocessor=self.postprocessor1) 

        self.split_detector = aggregator.DetectSpuriousMerging(
                track_names=self.track_names,
                track_transformer=affmat.L1Normalizer(),
                subclusters_detector=aggregator.RecursiveKmeans(
                    threshold=self.similarity_splitting_threshold,
                    minimum_size_for_splitting=self.final_min_cluster_size,
                    verbose=self.verbose))

        self.similar_patterns_collapser =\
            aggregator.SimilarPatternsCollapser(
                pattern_aligner=core.CrossCorrelationPatternAligner(
                    pattern_comparison_settings=
                        affmat.core.PatternComparisonSettings(
                            track_names=self.track_names,
                            track_transformer=affmat.MeanNormalizer().chain(
                                              affmat.MagnitudeNormalizer()),
                            min_overlap=self.min_overlap_while_sliding)),
                merging_threshold=self.similarity_merging_threshold,
                postprocessor=self.expand_trim_expand1,
                verbose=self.verbose) 

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
            print("Computing affinity matrix")
            sys.stdout.flush()
        affmat_start = time.time()
        affinity_mat = self.affinity_mat_from_seqlets(seqlets)
        if (self.verbose):
            print("Affinity mat computed in "
                  +str(round(time.time()-affmat_start,2))+"s")
            sys.stdout.flush()

        tsne_mat = self.tsne_affinitymat_transformer(affinity_mat)

        if (self.verbose):
            print("Applying filtering")
        filtering_start_time = time.time()
        filtered_rows_mask = self.filtered_rows_mask_producer(tsne_mat)
        if (self.verbose):
            print("Rows filtering took "+
                  str(round(time.time()-filtering_start_time))+"s")

        seqlets = [x[0] for x in zip(seqlets, filtered_rows_mask) if (x[1])]

        if (self.verbose):
            print("Computing clustering")
            sys.stdout.flush()
        cluster_results = self.clusterer(
            affinity_mat[filtered_rows_mask][:,filtered_rows_mask])
        num_clusters = max(cluster_results.cluster_indices+1)
        if (self.verbose):
            print("Got "+str(num_clusters)+" clusters initially")
            sys.stdout.flush()

        if (self.verbose):
            print("Aggregating seqlets in each cluster")
            sys.stdout.flush()

        cluster_to_seqlets = defaultdict(list)
        for cluster_val, seqlet in zip(cluster_results.cluster_indices,
                                       seqlets):
            cluster_to_seqlets[cluster_val].append(seqlet)

        cluster_to_aggregated_seqlets = OrderedDict()
        for i in range(num_clusters):
            if (self.verbose):
                print("Aggregating for cluster "+str(i))
                sys.stdout.flush()
            cluster_to_aggregated_seqlets[i] =\
                self.seqlet_aggregator(cluster_to_seqlets[i])
        patterns = list(itertools.chain(
                        *cluster_to_aggregated_seqlets.values()))

        if (self.verbose):
            print("Detecting splits")
            sys.stdout.flush()
        patterns = self.split_detector(patterns)
        if (self.verbose):
            print("Got "+str(len(patterns))+" after split detection")
            sys.stdout.flush()

        if (self.verbose):
            print("Collapsing similar patterns")
            sys.stdout.flush()
        patterns = self.similar_patterns_collapser(patterns)
        if (self.verbose):
            print("Got "+str(len(patterns))+" after collapsing")
            sys.stdout.flush()

        if (self.verbose):
            print("Performing seqlet reassignment")
            sys.stdout.flush()
        patterns = self.seqlet_reassigner(patterns)

        patterns = self.final_postprocessor(patterns)

        if (self.verbose):
            print("Got "+str(len(patterns))+" patterns")
            sys.stdout.flush()

        if (self.verbose):
            print("Total time taken is "
                  +str(round(time.time()-start,2))+"s")
            sys.stdout.flush()

        results = SeqletsToPatternsResults(
            patterns=patterns,
            affinity_mat=affinity_mat,
            cluster_results=cluster_results,
            filtered_seqlets=filtered_seqlets)

        return patterns 

