from __future__ import division, print_function, absolute_import
from . import affinitymat as affmat
from . import cluster
from . import aggregator
from collections import defaultdict, OrderedDict


class AbstractSeqletsToPatterns(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class DefaultSeqletsToPatterns(object):

    def __init__(self, crosscorr_track_names,
                       crosscorr_min_overlap=0.5,
                       tsne_perplexity=50,
                       frac_support_to_trim_to=0.2,
                       initial_trim_to_window_size=30,
                       initial_flank_to_add=10,
                       per_track_similarity_splitting_threshold=0.85,
                       minimum_size_for_splitting=50,
                       verbose=True):
        self.crosscorr_track_names = crosscorr_track_names
        self.crosscorr_min_overlap = crosscorr_min_overlap
        self.tsne_perplexity = tsne_perplexity
        self.frac_support_to_trim_to = frac_support_to_trim_to
        self.initial_trim_to_window_size = initial_trim_to_window_size
        self.initial_flank_to_add = initial_flank_to_add
        self.per_track_similarity_splitting_threshold =\
            per_track_similarity_splitting_threshold
        self.minimum_size_for_splitting = minimum_size_for_splitting
        self.verbose = verbose
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
                verbose=self.verbose)

        self.clusterer = cluster.core.LouvainCluster(
            affmat_transformer=\
                affmat.transformers.TsneJointProbs(perplexity=self.perplexity),
            verbose=self.verbose)

        self.similarity_splitting_threshold =\
            (len(self.crosscorr_track_names)
             *self.per_track_similarity_splitting_threshold)

        self.expand_trim_expand1 =\
            aggregator.ExpandSeqletsToFillPattern(
                track_set=track_set,
                flank_to_add=self.initial_flank_to_add)).chain(
            aggregator.TrimToBestWindow(
                window_size=self.initial_trim_to_window_size,
                track_names=self.crosscorr_track_names)).chain(
            aggregator.ExpandSeqletsToFillPattern(
                track_set=track_set,
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
                    minimum_size_for_splitting=minimum_size_for_splitting)))

        self.pattern_aligner = core.CrossCorrelationPatternAligner(
            pattern_crosscorr_settings=self.pattern_crosscorr_settings)
        
        self.seqlet_aggregator = aggregator.HierarchicalSeqletAggregator(
            pattern_aligner=self.pattern_aligner,
            affinity_mat_from_seqlets=self.affinity_mat_from_seqlets,
            postprocessor = postprocessor1) 
        

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




        
         
         
