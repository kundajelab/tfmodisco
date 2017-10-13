from __future__ import division, print_function, absolute_import
from . import affinitymat as affmat
from . import cluster
from . import aggregator
from . import core
from collections import defaultdict, OrderedDict
import itertools


class AbstractSeqletsToPatterns(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class SeqletsToPatterns1(AbstractSeqletsToPatterns):

    def __init__(self, crosscorr_track_names,
                       track_set,
                       crosscorr_min_overlap=0.5,
                       tsne_perplexity=50,
                       louvain_min_cluster_size=10,
                       frac_support_to_trim_to=0.2,
                       trim_to_window_size=30,
                       initial_flank_to_add=10,
                       similarity_splitting_threshold=0.85,
                       minimum_size_for_splitting=50,
                       per_track_similarity_merging_threshold=0.85,
                       per_track_min_similarity_for_seqlet_assignment=0.5,
                       final_min_cluster_size=20,
                       final_flank_to_add=20,
                       verbose=True,
                       batch_size=50,
                       progress_update=None):
        self.crosscorr_track_names = crosscorr_track_names
        self.track_set = track_set
        self.crosscorr_min_overlap = crosscorr_min_overlap
        self.tsne_perplexity = tsne_perplexity
        self.louvain_min_cluster_size = louvain_min_cluster_size
        self.frac_support_to_trim_to = frac_support_to_trim_to
        self.trim_to_window_size = trim_to_window_size
        self.initial_flank_to_add = initial_flank_to_add
        self.similarity_splitting_threshold = similarity_splitting_threshold
        self.minimum_size_for_splitting = minimum_size_for_splitting
        self.per_track_similarity_merging_threshold =\
            per_track_similarity_merging_threshold
        self.per_track_min_similarity_for_seqlet_assignment =\
            per_track_min_similarity_for_seqlet_assignment
        self.final_min_cluster_size = final_min_cluster_size
        self.final_flank_to_add=final_flank_to_add
        self.verbose = verbose
        self.batch_size = batch_size
        self.progress_update = progress_update
        self.build() 

    def build(self):

        self.pattern_crosscorr_settings = affmat.core.PatternCrossCorrSettings(
            track_names=self.crosscorr_track_names,                                     
            track_transformer=affmat.MeanNormalizer().chain(
                              affmat.MagnitudeNormalizer()),   
            min_overlap=self.crosscorr_min_overlap)

        self.affinity_mat_from_seqlets =\
            affmat.MaxCrossCorrAffinityMatrixFromSeqlets(
                pattern_crosscorr_settings=self.pattern_crosscorr_settings,
                progress_update=self.progress_update)

        self.clusterer = cluster.core.LouvainCluster(
            affmat_transformer=\
                affmat.transformers.TsneJointProbs(
                    perplexity=self.tsne_perplexity),
            min_cluster_size=self.louvain_min_cluster_size,
            verbose=self.verbose)

        self.expand_trim_expand1 =\
            aggregator.ExpandSeqletsToFillPattern(
                track_set=self.track_set,
                flank_to_add=self.initial_flank_to_add).chain(
            aggregator.TrimToBestWindow(
                window_size=self.trim_to_window_size,
                track_names=self.crosscorr_track_names)).chain(
            aggregator.ExpandSeqletsToFillPattern(
                track_set=self.track_set,
                flank_to_add=self.initial_flank_to_add))

        self.postprocessor1 =\
            aggregator.TrimToFracSupport(
                        frac=self.frac_support_to_trim_to).chain(
            self.expand_trim_expand1).chain(
            aggregator.DetectSpuriousMerging(
                track_names=self.crosscorr_track_names,
                track_transformer=affmat.MeanNormalizer().chain(
                                  affmat.MagnitudeNormalizer()),
                subclusters_detector=aggregator.RecursiveKmeans(
                    threshold=self.similarity_splitting_threshold,
                    minimum_size_for_splitting=self.minimum_size_for_splitting,
                    verbose=self.verbose)))

        self.pattern_aligner = core.CrossCorrelationPatternAligner(
            pattern_crosscorr_settings=self.pattern_crosscorr_settings)
        
        self.seqlet_aggregator = aggregator.HierarchicalSeqletAggregator(
            pattern_aligner=self.pattern_aligner,
            affinity_mat_from_seqlets=self.affinity_mat_from_seqlets,
            postprocessor=self.postprocessor1) 

        self.similarity_merging_threshold =\
            (len(self.crosscorr_track_names)*
             self.per_track_similarity_merging_threshold)

        self.similar_patterns_collapser =\
            aggregator.SimilarPatternsCollapser(
                pattern_aligner=self.pattern_aligner,
                merging_threshold=self.similarity_merging_threshold,
                postprocessor=self.expand_trim_expand1,
                verbose=self.verbose) 

        self.min_similarity_for_seqlet_assignment =\
            (len(self.crosscorr_track_names)*
             self.per_track_min_similarity_for_seqlet_assignment)

        self.expand_trim_expand2 =\
            aggregator.ExpandSeqletsToFillPattern(
                track_set=self.track_set,
                flank_to_add=self.initial_flank_to_add).chain(
            aggregator.TrimToBestWindow(
                window_size=self.trim_to_window_size,
                track_names=self.crosscorr_track_names)).chain(
            aggregator.ExpandSeqletsToFillPattern(
                track_set=self.track_set,
                flank_to_add=self.final_flank_to_add))

        self.small_clusters_eliminator =\
           aggregator.ReassignSeqletsToLargerClusters(
            seqlet_assigner=aggregator.AssignSeqletsByBestCrossCorr(
                pattern_crosscorr_settings=self.pattern_crosscorr_settings,
                min_similarity=self.min_similarity_for_seqlet_assignment,
                batch_size=self.batch_size,
                progress_update=self.progress_update),
            min_cluster_size=self.final_min_cluster_size,
            postprocessor=self.expand_trim_expand2) 
        

    def __call__(self, seqlets):

        if (self.verbose):
            print("Computing affinity matrix")
        affinity_mat = self.affinity_mat_from_seqlets(seqlets)
        if (self.verbose):
            print("Computing clustering")
        cluster_results = self.clusterer(affinity_mat)
        num_clusters = max(cluster_results.cluster_indices+1)
        if (self.verbose):
            print("Got "+str(num_clusters)+" clusters initially")

        cluster_to_seqlets = defaultdict(list)
        for cluster_val, seqlet in zip(cluster_results.cluster_indices,
                                       seqlets):
            cluster_to_seqlets[cluster_val].append(seqlet)

        cluster_to_aggregated_seqlets = OrderedDict()
        for i in range(num_clusters):
            cluster_to_aggregated_seqlets[i] =\
                self.seqlet_aggregator(cluster_to_seqlets[i])

        if (self.verbose):
            print("Collapsing similar patterns")

        merged_patterns = self.similar_patterns_collapser(
            list(itertools.chain(*cluster_to_aggregated_seqlets.values())))

        if (self.verbose):
            print("Eliminating clusters smaller than "
                  +(str(self.final_min_cluster_size)))

        merged_patterns = self.small_clusters_eliminator(merged_patterns)

        if (self.verbose):
            print("Got "+str(len(merged_patterns))+" patterns")

        return merged_patterns 

