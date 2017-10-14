from __future__ import division, print_function, absolute_import
from . import affinitymat as affmat
from . import cluster
from . import aggregator
from . import core
from collections import defaultdict, OrderedDict
import itertools
import time
import sys


class AbstractSeqletsToPatterns(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class SeqletsToPatterns1(AbstractSeqletsToPatterns):

    def __init__(self, crosscorr_track_names,
                       track_set,
                       crosscorr_min_overlap=0.5,
                       affmat_progress_update=5000,
                       min_rows_before_applying_filtering=1000,
                       bins_for_curvature_threshold=15,
                       min_jaccard_sim=0.2,
                       min_edges_per_row=5, 
                       tsne_perplexity=50,
                       louvain_min_cluster_size=10,
                       frac_support_to_trim_to=0.2,
                       trim_to_window_size=30,
                       initial_flank_to_add=10,
                       similarity_splitting_threshold=0.85,
                       per_track_similarity_merging_threshold=0.85,
                       per_track_min_similarity_for_seqlet_assignment=0,
                       final_min_cluster_size=20,
                       final_flank_to_add=10,
                       percent_change_in_assignments_tolerance=0.1,
                       max_reassignment_rounds=3, 
                       verbose=True,
                       batch_size=50):

        #mandatory arguments
        self.crosscorr_track_names = crosscorr_track_names
        self.track_set = track_set

        #affinity_mat calculation
        self.crosscorr_min_overlap = crosscorr_min_overlap
        self.affmat_progress_update = affmat_progress_update

        #seqlet filtering based on affinity_mat
        self.min_rows_before_applying_filtering =\
            min_rows_before_applying_filtering
        self.bins_for_curvature_threshold = bins_for_curvature_threshold
        self.min_jaccard_sim = min_jaccard_sim
        self.min_edges_per_row = min_edges_per_row

        #clustering settings
        self.tsne_perplexity = tsne_perplexity
        self.louvain_min_cluster_size = louvain_min_cluster_size

        #postprocessor1 settings
        self.frac_support_to_trim_to = frac_support_to_trim_to
        self.trim_to_window_size = trim_to_window_size
        self.initial_flank_to_add = initial_flank_to_add

        #split detection settings
        self.similarity_splitting_threshold = similarity_splitting_threshold

        #cluster merging settings
        self.per_track_similarity_merging_threshold =\
            per_track_similarity_merging_threshold
        self.per_track_min_similarity_for_seqlet_assignment =\
            per_track_min_similarity_for_seqlet_assignment

        #reassignment settings
        self.percent_change_in_assignments_tolerance =\
            percent_change_in_assignments_tolerance
        self.max_reassignment_rounds = max_reassignment_rounds
        self.final_min_cluster_size = final_min_cluster_size

        #final postprocessor settings
        self.final_flank_to_add=final_flank_to_add

        #other settings
        self.verbose = verbose
        self.batch_size = batch_size

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
                batch_size=self.batch_size,
                progress_update=self.affmat_progress_update)

        self.filtered_rows_mask_producer =\
           affmat.core.FilterSparseRows(
            affmat_transformer=\
                affmat.transformers.PerNodeThresholdDistanceBinarizer(
                    thresholder=affmat.transformers.CurvatureBasedThreshold(
                                    bins=self.bins_for_curvature_threshold))
                .chain(affmat.transformers.SymmetrizeByMultiplying())
                .chain(affmat.transformers.JaccardSimCPU())
                .chain(affmat.transformers.MinVal(self.min_jaccard_sim)),
            min_rows_before_applying_filtering=\
                self.min_rows_before_applying_filtering,
            min_edges_per_row=self.min_edges_per_row,
            verbose=self.verbose)  

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
                    minimum_size_for_splitting=self.final_min_cluster_size,
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

        self.final_postprocessor = aggregator.ExpandSeqletsToFillPattern(
                                        track_set=self.track_set,
                                        flank_to_add=self.final_flank_to_add)

        self.seqlet_reassigner =\
           aggregator.ReassignSeqletsTillConvergence(
            seqlet_assigner=aggregator.AssignSeqletsByBestCrossCorr(
                pattern_crosscorr_settings=self.pattern_crosscorr_settings,
                min_similarity=self.min_similarity_for_seqlet_assignment,
                batch_size=self.batch_size,
                progress_update=None),
            percent_change_tolerance=
                self.percent_change_in_assignments_tolerance,
            max_rounds=self.max_reassignment_rounds,
            min_cluster_size=self.final_min_cluster_size,
            postprocessor=self.expand_trim_expand1,
            verbose=self.verbose) 
        

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
        #silence it for later steps
        self.affinity_mat_from_seqlets.progress_update = None

        if (self.verbose):
            print("Applying filtering")
        filtering_start_time = time.time()
        filtered_rows_mask = self.filtered_rows_mask_producer(affinity_mat)
        if (self.verbose):
            print("Rows filtering took "+
                  str(round(time.time()-filtering_start_time))+"s")

        affinity_mat = (affinity_mat[filtered_rows_mask])[:,filtered_rows_mask]
        seqlets = [x[0] for x in zip(seqlets, filtered_rows_mask) if (x[1])]

        if (self.verbose):
            print("Computing clustering")
            sys.stdout.flush()
        cluster_results = self.clusterer(affinity_mat)
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

        if (self.verbose):
            print("Collapsing similar patterns")
            sys.stdout.flush()

        merged_patterns = self.similar_patterns_collapser(
            list(itertools.chain(*cluster_to_aggregated_seqlets.values())))

        if (self.verbose):
            print("Performing seqlet reassignment")
            sys.stdout.flush()

        merged_patterns = self.seqlet_reassigner(merged_patterns)
        merged_patterns = self.final_postprocessor(merged_patterns)

        if (self.verbose):
            print("Got "+str(len(merged_patterns))+" patterns")
            sys.stdout.flush()

        if (self.verbose):
            print("Total time taken is "
                  +str(round(time.time()-start,2))+"s")
            sys.stdout.flush()

        return merged_patterns 

