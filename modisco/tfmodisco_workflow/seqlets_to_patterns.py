from __future__ import division, absolute_import, print_function
from .. import affinitymat
from .. import nearest_neighbors
from .. import cluster
from .. import aggregator
from .. import core
from .. import util
from .. import seqlet_embedding
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import time
import sys
import gc


def print_memory_use():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("MEMORY",process.memory_info().rss/1000000000)


def get_seqlet_neighbors_with_initcluster(
        nearest_neighbors_to_compute,
        coarse_affmat, initclusters): 
    if (initclusters is not None):
        assert len(initclusters)==len(coarse_affmat)
    #get the argsort for coarse_affmat
    coarse_affmat_argsort = np.argsort(-coarse_affmat, axis=-1)
    nearest_neighbors = []
    for row_idx,argsort_row in enumerate(coarse_affmat_argsort):
        combined_neighbor_row = []
        neighbor_row_topnn = argsort_row[:nearest_neighbors_to_compute+1]
        neighbor_set_topnn = set(neighbor_row_topnn)
        combined_neighbor_row.extend(neighbor_row_topnn)
        if (initclusters is not None):
            combined_neighbor_row.extend([
                y for y in ([x for x in argsort_row
                             if initclusters[x]==initclusters[row_idx]][
                             :nearest_neighbors_to_compute+1])
                if y not in neighbor_set_topnn])
        nearest_neighbors.append(combined_neighbor_row) 
    return nearest_neighbors


def fish_out_kwargs(orig_kwargs, to_fish_out):
    fished_out = {}
    for kwarg_name in to_fish_out:
        if kwarg_name in orig_kwargs:
            fished_out[kwarg_name] = orig_kwargs[kwarg_name]
            del orig_kwargs[kwarg_name] 
    return fished_out


#adds backwargs compatibility re gapped kmer arguments
def legacy_tfmodiscoseqletstopatternsfactory(current_constructor):

    def new_constructor(*args, **kwargs): 
        gapped_kmer_kwargs = fish_out_kwargs(
            orig_kwargs=kwargs,
            to_fish_out=['kmer_len', 'num_gaps',
                         'num_mismatches', 'gpu_batch_size'])
        if (len(gapped_kmer_kwargs) > 0):
            assert 'embedder_factory' not in kwargs,\
                ("Cannot both specify embedder_factory and "
                 +str(gapped_kmer_kwargs))
            kwargs['embedder_factory'] = (
                  seqlet_embedding.gapped_kmer
                  .GappedKmerEmbedderFactory(**gapped_kmer_kwargs))
        return current_constructor(*args, **kwargs) 
    return new_constructor 


##legacy
#alphabet_size=None,
#kmer_len=None, num_gaps=3, num_mismatches=2,
#gpu_batch_size=20,
class TfModiscoSeqletsToPatternsFactory(object):

    @legacy_tfmodiscoseqletstopatternsfactory
    def __init__(self, n_cores=4,
                       min_overlap_while_sliding=0.7,

                       #init clusterer factory
                       initclusterer_factory=None,                       

                       embedder_factory=(
                        seqlet_embedding.gapped_kmer
                                        .GappedKmerEmbedderFactory()),

                       nn_n_jobs=4,
                       nearest_neighbors_to_compute=500,

                       affmat_correlation_threshold=0.15,
                       filter_beyond_first_round=False,
                       skip_fine_grained=False,

                       tsne_perplexity = 10,
                       use_louvain=False,
                       louvain_initclusters_weight=1.0,
                       n_leiden_iterations_r1=-1,
                       n_leiden_iterations_r2=-1,
                       louvain_num_runs_and_levels_r1=[(200,-1)],
                       louvain_num_runs_and_levels_r2=[(200,-1)], 
                       contin_runs_r1 = 50,
                       contin_runs_r2 = 50,
                       final_louvain_level_to_return=1,

                       frac_support_to_trim_to=0.2,
                       min_num_to_trim_to=30,
                       trim_to_window_size=30,
                       initial_flank_to_add=10,

                       prob_and_pertrack_sim_merge_thresholds=[
                        (0.0001,0.84), (0.00001, 0.87), (0.000001, 0.9)],

                       prob_and_pertrack_sim_dealbreaker_thresholds=[
                        (0.1,0.75), (0.01, 0.8), (0.001, 0.83),
                        (0.0000001,0.9)],

                       threshold_for_spurious_merge_detection=0.8,

                       min_similarity_for_seqlet_assignment=0.2,
                       final_min_cluster_size=30,

                       final_flank_to_add=10,
                       verbose=True, seed=1234):

        self.initclusterer_factory = initclusterer_factory
        if (use_louvain==True):
            assert self.initclusterer_factory==None,\
                    ("Louvain doesn't support cluster initialization;"
                     +" set use_louvain to False")

        #affinity_mat calculation
        self.n_cores = n_cores
        self.min_overlap_while_sliding = min_overlap_while_sliding

        self.embedder_factory = embedder_factory

        self.nn_n_jobs = nn_n_jobs
        self.nearest_neighbors_to_compute = nearest_neighbors_to_compute

        self.affmat_correlation_threshold = affmat_correlation_threshold
        self.filter_beyond_first_round = filter_beyond_first_round
        self.skip_fine_grained = skip_fine_grained

        #affinity mat to tsne dist mat setting
        self.tsne_perplexity = tsne_perplexity

        #clustering settings
        self.use_louvain = use_louvain
        self.louvain_initclusters_weight = louvain_initclusters_weight
        self.n_leiden_iterations_r1 = n_leiden_iterations_r1
        self.n_leiden_iterations_r2 = n_leiden_iterations_r2
        self.contin_runs_r1 = contin_runs_r1
        self.contin_runs_r2 = contin_runs_r2
        self.louvain_num_runs_and_levels_r1 = louvain_num_runs_and_levels_r1
        self.louvain_num_runs_and_levels_r2 = louvain_num_runs_and_levels_r2
        self.final_louvain_level_to_return = final_louvain_level_to_return

        #postprocessor1 settings
        self.frac_support_to_trim_to = frac_support_to_trim_to
        self.min_num_to_trim_to = min_num_to_trim_to
        self.trim_to_window_size = trim_to_window_size
        self.initial_flank_to_add = initial_flank_to_add 

        #merging similar patterns
        self.prob_and_pertrack_sim_merge_thresholds =\
            prob_and_pertrack_sim_merge_thresholds
        self.prob_and_pertrack_sim_dealbreaker_thresholds =\
            prob_and_pertrack_sim_dealbreaker_thresholds

        self.threshold_for_spurious_merge_detection =\
            threshold_for_spurious_merge_detection

        #reassignment settings
        self.min_similarity_for_seqlet_assignment =\
            min_similarity_for_seqlet_assignment
        self.final_min_cluster_size = final_min_cluster_size

        #final postprocessor settings
        self.final_flank_to_add=final_flank_to_add

        #other settings
        self.verbose = verbose
        self.seed = seed

    def get_jsonable_config(self):
        to_return =  OrderedDict([
                ('class_name', type(self).__name__),
                ('n_cores', self.n_cores),
                ('initclusterer_factory',
                 self.initclusterer_factory.get_jsonable_config())
                ('min_overlap_while_sliding', self.min_overlap_while_sliding),
                ('embedder_factory',
                 self.embedder_factory.get_jsonable_config()),
                ('num_mismatches', self.num_mismatches),
                ('nn_n_jobs', self.nn_n_jobs),
                ('nearest_neighbors_to_compute',
                 self.nearest_neighbors_to_compute),
                ('affmat_correlation_threshold',
                 self.affmat_correlation_threshold),
                ('filter_beyond_first_round', filter_beyond_first_round),
                ('tsne_perplexity', self.tsne_perplexity),
                ('use_louvain', self.use_louvain),
                ('louvain_num_runs_and_levels_r1',
                 self.louvain_num_runs_and_levels_r1),
                ('louvain_num_runs_and_levels_r2',
                 self.louvain_num_runs_and_levels_r2),
                ('final_louvain_level_to_return',
                 self.final_louvain_level_to_return),
                ('contin_runs_r1',
                 self.contin_runs_r1),
                ('contin_runs_r2',
                 self.contin_runs_r2),
                ('frac_support_to_trim_to', self.frac_support_to_trim_to),
                ('min_num_to_trim_to', self.min_num_to_trim_to),
                ('trim_to_window_size', self.trim_to_window_size),
                ('initial_flank_to_add', self.initial_flank_to_add),
                ('prob_and_pertrack_sim_merge_thresholds',
                 self.prob_and_pertrack_sim_merge_thresholds),
                ('prob_and_pertrack_sim_dealbreaker_thresholds',
                 self.prob_and_pertrack_sim_dealbreaker_thresholds),
                ('threshold_for_spurious_merge_detection',
                 self.threshold_for_spurious_merge_detection),
                ('min_similarity_for_seqlet_assignment',
                 self.min_similarity_for_seqlet_assignment),
                ('final_min_cluster_size', self.final_min_cluster_size),
                ('final_flank_to_add', self.final_flank_to_add)]) 
        return to_return

    def __call__(self, track_set, onehot_track_name,
                       contrib_scores_track_names,
                       hypothetical_contribs_track_names,
                       track_signs,
                       other_comparison_track_names=[]):

        assert len(track_signs)==len(hypothetical_contribs_track_names)
        assert len(track_signs)==len(contrib_scores_track_names)

        seqlets_sorter = (lambda arr:
                          sorted(arr,
                                 key=lambda x:
                                  -np.sum([np.sum(np.abs(x[track_name].fwd))
                                     for track_name
                                     in contrib_scores_track_names])))

        if (self.initclusterer_factory is not None):
            self.initclusterer_factory.set_onehot_track_name(onehot_track_name)
        initclusterer_factory = self.initclusterer_factory

        pattern_comparison_settings =\
            affinitymat.core.PatternComparisonSettings(
                track_names=hypothetical_contribs_track_names
                            +contrib_scores_track_names
                            +other_comparison_track_names, 
                track_transformer=affinitymat.L1Normalizer(), 
                min_overlap=self.min_overlap_while_sliding)

        #coarse_grained 1d embedder
        seqlets_to_1d_embedder = self.embedder_factory(
                onehot_track_name=onehot_track_name,
                toscore_track_names_and_signs=list(
                zip(hypothetical_contribs_track_names,
                    [np.sign(x) for x in track_signs])))

        #affinity matrix from embeddings
        coarse_affmat_computer =\
            affinitymat.core.AffmatFromSeqletEmbeddings(
                seqlets_to_1d_embedder=seqlets_to_1d_embedder,
                affinity_mat_from_1d=\
                    affinitymat.core.NumpyCosineSimilarity(
                        verbose=self.verbose),
                verbose=self.verbose)

        affmat_from_seqlets_with_nn_pairs =\
            affinitymat.core.AffmatFromSeqletsWithNNpairs(
                pattern_comparison_settings=pattern_comparison_settings,
                sim_metric_on_nn_pairs=\
                    affinitymat.core.ParallelCpuCrossMetricOnNNpairs(
                        n_cores=self.n_cores,
                        cross_metric_single_region=
                            affinitymat.core.CrossContinJaccardSingleRegion()))

        filter_mask_from_correlation =\
            affinitymat.core.FilterMaskFromCorrelation(
                correlation_threshold=self.affmat_correlation_threshold,
                verbose=self.verbose)

        aff_to_dist_mat = affinitymat.transformers.AffToDistViaInvLogistic() 
        density_adapted_affmat_transformer =\
            affinitymat.transformers.TsneConditionalProbs(
                perplexity=self.tsne_perplexity,
                aff_to_dist_mat=aff_to_dist_mat)

        #prepare the clusterers for the different rounds
        affmat_transformer_r1 = affinitymat.transformers.SymmetrizeByAddition(
                                probability_normalize=True)
        print("TfModiscoSeqletsToPatternsFactory: seed=%d" % self.seed)
        if (self.use_louvain):
            for n_runs, level_to_return in self.louvain_num_runs_and_levels_r1:
                affmat_transformer_r1 = affmat_transformer_r1.chain(
                    affinitymat.transformers.LouvainMembershipAverage(
                        n_runs=n_runs,
                        level_to_return=level_to_return,
                        parallel_threads=self.n_cores, seed=self.seed))
            clusterer_r1 = cluster.core.LouvainCluster(
                level_to_return=self.final_louvain_level_to_return,
                affmat_transformer=affmat_transformer_r1,
                contin_runs=self.contin_runs_r1,
                verbose=self.verbose, seed=self.seed)
        else:
            clusterer_r1 = cluster.core.LeidenCluster(
                affmat_transformer=affmat_transformer_r1,
                contin_runs=self.contin_runs_r1,
                n_leiden_iterations=self.n_leiden_iterations_r1,
                verbose=self.verbose)

        affmat_transformer_r2 = affinitymat.transformers.SymmetrizeByAddition(
                                probability_normalize=True)
        if (self.use_louvain):
            for n_runs, level_to_return in self.louvain_num_runs_and_levels_r2:
                affmat_transformer_r2 = affmat_transformer_r2.chain(
                    affinitymat.transformers.LouvainMembershipAverage(
                        n_runs=n_runs,
                        level_to_return=level_to_return,
                        parallel_threads=self.n_cores, seed=self.seed))
            clusterer_r2 = cluster.core.LouvainCluster(
                level_to_return=self.final_louvain_level_to_return,
                affmat_transformer=affmat_transformer_r2,
                contin_runs=self.contin_runs_r2,
                verbose=self.verbose, seed=self.seed,
                initclusters_weight=self.louvain_initclusters_weight)
        else:
            clusterer_r2 = cluster.core.LeidenCluster(
                affmat_transformer=affmat_transformer_r2,
                contin_runs=self.contin_runs_r2,
                n_leiden_iterations=self.n_leiden_iterations_r2,
                verbose=self.verbose)
        
        clusterer_per_round = [clusterer_r1, clusterer_r2]

        #prepare the seqlet aggregator
        expand_trim_expand1 =\
            aggregator.ExpandSeqletsToFillPattern(
                track_set=track_set,
                flank_to_add=self.initial_flank_to_add).chain(
            aggregator.TrimToBestWindow(
                window_size=self.trim_to_window_size,
                track_names=contrib_scores_track_names)).chain(
            aggregator.ExpandSeqletsToFillPattern(
                track_set=track_set,
                flank_to_add=self.initial_flank_to_add))
        postprocessor1 =\
            aggregator.TrimToFracSupport(
                        min_frac=self.frac_support_to_trim_to,
                        min_num=self.min_num_to_trim_to,
                        verbose=self.verbose)\
                      .chain(expand_trim_expand1)
        seqlet_aggregator = aggregator.GreedySeqletAggregator(
            pattern_aligner=core.CrossContinJaccardPatternAligner(
                pattern_comparison_settings=pattern_comparison_settings),
                seqlet_sort_metric=
                    lambda x: -sum([np.sum(np.abs(x[track_name].fwd)) for
                               track_name in contrib_scores_track_names]),
            postprocessor=postprocessor1)

        def sign_consistency_func(motif):
            motif_track_signs = [
                np.sign(np.sum(motif[contrib_scores_track_name].fwd)) for
                contrib_scores_track_name in contrib_scores_track_names]
            return all([(x==y) for x,y in zip(motif_track_signs, track_signs)])

        #prepare the similar patterns collapser
        pattern_to_seqlet_sim_computer =\
            affinitymat.core.AffmatFromSeqletsWithNNpairs(
                pattern_comparison_settings=pattern_comparison_settings,
                sim_metric_on_nn_pairs=\
                    affinitymat.core.ParallelCpuCrossMetricOnNNpairs(
                        n_cores=self.n_cores,
                        cross_metric_single_region=\
                            affinitymat.core.CrossContinJaccardSingleRegion(),
                        verbose=False))

        #similarity settings for merging
        prob_and_sim_merge_thresholds =\
            [(x[0], x[1]*(len(contrib_scores_track_names)
                          +len(other_comparison_track_names)))
             for x in self.prob_and_pertrack_sim_merge_thresholds]
        prob_and_sim_dealbreaker_thresholds =\
            [(x[0], x[1]*(len(contrib_scores_track_names)
                          +len(other_comparison_track_names)))
             for x in self.prob_and_pertrack_sim_dealbreaker_thresholds]

        spurious_merge_detector = aggregator.DetectSpuriousMerging(
            track_names=contrib_scores_track_names,
            track_transformer=affinitymat.core.L1Normalizer(),
            affmat_from_1d=affinitymat.core.ContinJaccardSimilarity(
                            make_positive=True, verbose=False),
            diclusterer=cluster.core.LouvainCluster(
                            level_to_return=1,
                            max_clusters=2, contin_runs=20,
                            verbose=False, seed=self.seed),
            is_dissimilar_func=aggregator.PearsonCorrIsDissimilarFunc(
                        threshold=self.threshold_for_spurious_merge_detection,
                        verbose=self.verbose),
            min_in_subcluster=self.final_min_cluster_size)

        #similar_patterns_collapser =\
        #    aggregator.DynamicThresholdSimilarPatternsCollapser(
        #        pattern_to_seqlet_sim_computer=
        #            pattern_to_seqlet_sim_computer,
        #        pattern_aligner=core.CrossCorrelationPatternAligner(
        #            pattern_comparison_settings=
        #                affinitymat.core.PatternComparisonSettings(
        #                    track_names=(
        #                        contrib_scores_track_names+
        #                        other_comparison_track_names), 
        #                    track_transformer=
        #                        affinitymat.MeanNormalizer().chain(
        #                        affinitymat.MagnitudeNormalizer()), 
        #                    min_overlap=self.min_overlap_while_sliding)),
        #        collapse_condition=(lambda dist_prob, aligner_sim:
        #            any([(dist_prob > x[0] and aligner_sim > x[1])
        #                 for x in prob_and_sim_merge_thresholds])),
        #        dealbreaker_condition=(lambda dist_prob, aligner_sim:
        #            any([(dist_prob < x[0] and aligner_sim < x[1])              
        #                 for x in prob_and_sim_dealbreaker_thresholds])),
        #        postprocessor=postprocessor1,
        #        verbose=self.verbose) 
        similar_patterns_collapser =\
            aggregator.DynamicDistanceSimilarPatternsCollapser(
                pattern_to_pattern_sim_computer=
                    pattern_to_seqlet_sim_computer,
                aff_to_dist_mat=aff_to_dist_mat,
                pattern_aligner=core.CrossCorrelationPatternAligner(
                    pattern_comparison_settings=
                        affinitymat.core.PatternComparisonSettings(
                            track_names=(
                                contrib_scores_track_names+
                                other_comparison_track_names), 
                            track_transformer=
                                affinitymat.MeanNormalizer().chain(
                                affinitymat.MagnitudeNormalizer()), 
                            min_overlap=self.min_overlap_while_sliding)),
                collapse_condition=(lambda dist_prob, aligner_sim:
                    any([(dist_prob > x[0] and aligner_sim > x[1])
                         for x in prob_and_sim_merge_thresholds])),
                dealbreaker_condition=(lambda dist_prob, aligner_sim:
                    any([(dist_prob < x[0] and aligner_sim < x[1])              
                         for x in prob_and_sim_dealbreaker_thresholds])),
                postprocessor=postprocessor1,
                verbose=self.verbose)

        seqlet_reassigner =\
           aggregator.ReassignSeqletsFromSmallClusters(
            seqlet_assigner=aggregator.AssignSeqletsByBestMetric(
                pattern_comparison_settings=pattern_comparison_settings,
                individual_aligner_metric=
                    core.get_best_alignment_crosscontinjaccard,
                matrix_affinity_metric=
                    affinitymat.core.CrossContinJaccardMultiCoreCPU(
                        verbose=self.verbose, n_cores=self.n_cores),
                min_similarity=self.min_similarity_for_seqlet_assignment),
            min_cluster_size=self.final_min_cluster_size,
            postprocessor=expand_trim_expand1,
            verbose=self.verbose) 

        final_postprocessor = aggregator.ExpandSeqletsToFillPattern(
                                        track_set=track_set,
                                        flank_to_add=self.final_flank_to_add) 

        return TfModiscoSeqletsToPatterns(
                seqlets_sorter=seqlets_sorter,
                initclusterer_factory=initclusterer_factory,
                coarse_affmat_computer=coarse_affmat_computer,
                nearest_neighbors_to_compute=self.nearest_neighbors_to_compute,
                affmat_from_seqlets_with_nn_pairs=
                    affmat_from_seqlets_with_nn_pairs, 
                filter_mask_from_correlation=filter_mask_from_correlation,
                filter_beyond_first_round=self.filter_beyond_first_round,
                skip_fine_grained=self.skip_fine_grained,
                density_adapted_affmat_transformer=
                    density_adapted_affmat_transformer,
                clusterer_per_round=clusterer_per_round,
                seqlet_aggregator=seqlet_aggregator,
                sign_consistency_func=sign_consistency_func,
                spurious_merge_detector=spurious_merge_detector,
                similar_patterns_collapser=similar_patterns_collapser,
                seqlet_reassigner=seqlet_reassigner,
                final_postprocessor=final_postprocessor,
                verbose=self.verbose)

    def save_hdf5(self, grp):
        grp.attrs['jsonable_config'] =\
            json.dumps(self.jsonable_config, indent=4, separators=(',', ': ')) 


class SeqletsToPatternsResults(object):

    def __init__(self,
                 each_round_initcluster_motifs, 
                 patterns, cluster_results,
                 total_time_taken, success=True, **kwargs):
        self.each_round_initcluster_motifs = each_round_initcluster_motifs
        self.success = success
        self.patterns = patterns
        self.cluster_results = cluster_results
        self.total_time_taken = total_time_taken
        self.__dict__.update(**kwargs)

    def save_each_round_initcluster_motifs(self, grp):
        all_round_names = []
        for (round_idx,initcluster_motifs)\
            in enumerate(self.each_round_initcluster_motifs):
            round_name = "round_"+str(round_idx)
            util.save_patterns(patterns=initcluster_motifs,
                               grp=grp.create_group(round_name)) 
        util.save_string_list(
            string_list=all_round_names,
            dset_name="all_round_names",          
            grp=grp)

    @classmethod
    def load_each_round_initcluster_motifs(cls, grp, track_set):
        all_round_names = util.load_string_list(dset_name="all_round_names",
                                                grp=grp) 
        each_round_initcluster_motifs = [] 
        for round_name in all_round_names:
            round_grp = grp[round_name]
            initcluster_motifs = load_patterns(grp=round_grp,
                                               track_set=track_set)
            each_round_initcluster_motifs.append(initcluster_motifs)
        return each_round_initcluster_motifs
             
    @classmethod
    def from_hdf5(cls, grp, track_set):
        success = grp.attrs.get("success", False)
        if (success):
            if ("each_round_initcluster_motifs" not in grp):
                each_round_initcluster_motifs = None 
            else:
                each_round_initcluster_motifs =\
                    cls.load_each_round_initcluster_motifs(
                        grp=grp["each_round_initcluster_motifs"],
                        track_set=track_set)
            patterns = util.load_patterns(grp=grp["patterns"],
                                          track_set=track_set) 
            cluster_results = None
            total_time_taken = None
            return cls(
                each_round_initcluster_motifs=each_round_initcluster_motifs,
                patterns=patterns,
                cluster_results=cluster_results,
                total_time_taken=total_time_taken)
        else:
            return cls(success=False, patterns=None, cluster_results=None,
                       total_time_taken=None)

    def save_hdf5(self, grp):
        grp.attrs["success"] = self.success
        if (self.success):
            if (self.each_round_initcluster_motifs is not None):
                self.save_each_round_initcluster_motifs(
                    grp=grp.create_group("each_round_initcluster_motifs"))
            util.save_patterns(self.patterns,
                               grp.create_group("patterns"))
            self.cluster_results.save_hdf5(grp.create_group("cluster_results"))   
            grp.attrs['total_time_taken'] = self.total_time_taken


class AbstractSeqletsToPatterns(object):

    def __call__(self, seqlets):
        raise NotImplementedError()


class TfModiscoSeqletsToPatterns(AbstractSeqletsToPatterns):

    def __init__(self, seqlets_sorter, 
                       initclusterer_factory,
                       coarse_affmat_computer,
                       nearest_neighbors_to_compute,
                       affmat_from_seqlets_with_nn_pairs, 
                       filter_mask_from_correlation,
                       filter_beyond_first_round,
                       skip_fine_grained,
                       density_adapted_affmat_transformer,
                       clusterer_per_round,
                       seqlet_aggregator,
                       sign_consistency_func,
                       spurious_merge_detector,
                       similar_patterns_collapser,
                       seqlet_reassigner,
                       final_postprocessor,
                       verbose=True):

        self.seqlets_sorter = seqlets_sorter
        self.initclusterer_factory = initclusterer_factory
        self.coarse_affmat_computer = coarse_affmat_computer
        self.nearest_neighbors_to_compute = nearest_neighbors_to_compute
        self.affmat_from_seqlets_with_nn_pairs =\
            affmat_from_seqlets_with_nn_pairs
        self.filter_mask_from_correlation = filter_mask_from_correlation
        self.filter_beyond_first_round = filter_beyond_first_round
        self.skip_fine_grained = skip_fine_grained
        self.density_adapted_affmat_transformer =\
            density_adapted_affmat_transformer
        self.clusterer_per_round = clusterer_per_round 
        self.seqlet_aggregator = seqlet_aggregator
        self.sign_consistency_func = sign_consistency_func
        
        self.spurious_merge_detector = spurious_merge_detector
        self.similar_patterns_collapser = similar_patterns_collapser
        self.seqlet_reassigner = seqlet_reassigner
        self.final_postprocessor = final_postprocessor

        self.verbose = verbose


    def get_cluster_to_aggregate_motif(self, seqlets, cluster_indices,
                                       sign_consistency_check,
                                       min_seqlets_in_motif):
        num_clusters = max(cluster_indices+1)
        cluster_to_seqlets = defaultdict(list) 
        assert len(seqlets)==len(cluster_indices)
        for seqlet,idx in zip(seqlets, cluster_indices):
            cluster_to_seqlets[idx].append(seqlet)

        cluster_to_motif = OrderedDict()
        cluster_to_eliminated_motif = OrderedDict()
        for i in range(num_clusters):
            if (len(cluster_to_seqlets[i]) >= min_seqlets_in_motif):
                if (self.verbose):
                    print("Aggregating for cluster "+str(i)+" with "
                          +str(len(cluster_to_seqlets[i]))+" seqlets")
                    print_memory_use()
                    sys.stdout.flush()
                motifs = self.seqlet_aggregator(cluster_to_seqlets[i])
                assert len(motifs)<=1
                if (len(motifs) > 0):
                    motif = motifs[0]
                    if (sign_consistency_check==False or
                        self.sign_consistency_func(motif)):
                        cluster_to_motif[i] = motif
                    else:
                        if (self.verbose):
                            print("Dropping cluster "+str(i)+
                                  " with "+str(motif.num_seqlets)
                                  +" seqlets due to sign disagreement")
                        cluster_to_eliminated_motif[i] = motif
                cluster_to_motif[i] = motif
        return cluster_to_motif, cluster_to_eliminated_motif


    def __call__(self, seqlets):

        seqlets = self.seqlets_sorter(seqlets)
        if (self.initclusterer_factory is not None):
            initclusterer = self.initclusterer_factory(seqlets=seqlets) 
        else:
            initclusterer = None

        start = time.time()

        #seqlets_sets = []
        #coarse_affmats = []
        #nn_affmats = []
        #filtered_seqlets_sets = []
        #filtered_affmats = []
        #density_adapted_affmats = []
        #cluster_results_sets = []
        #cluster_to_motif_sets = []
        #cluster_to_eliminated_motif_sets = []

        if (initclusterer is not None):
            each_round_initcluster_motifs = []
        else:
            each_round_initcluster_motifs = None

        for round_idx, clusterer in enumerate(self.clusterer_per_round):
            import gc
            gc.collect()

            round_num = round_idx+1

            if (initclusterer is not None): 
                initclusters = initclusterer(seqlets=seqlets)
                initcluster_motifs =\
                    list(self.get_cluster_to_aggregate_motif(
                            seqlets=seqlets,
                            cluster_indices=initclusters,
                            sign_consistency_check=False,
                            min_seqlets_in_motif=2)[0].values())
                each_round_initcluster_motifs.append(initcluster_motifs)
            else:
                initclusters = None
                initcluster_motifs = None
            
            if (len(seqlets)==0):
                if (self.verbose):
                    print("len(seqlets) is 0 - bailing!")
                return SeqletsToPatternsResults(
                        patterns=None,
                        seqlets=None,
                        affmat=None,
                        cluster_results=None, 
                        total_time_taken=None,
                        success=False)

            if (self.verbose):
                print("(Round "+str(round_num)+
                      ") num seqlets: "+str(len(seqlets)))
                print("(Round "+str(round_num)+") Computing coarse affmat")
                print_memory_use()
                sys.stdout.flush()
            coarse_affmat = self.coarse_affmat_computer(seqlets)
            #coarse_affmats.append(coarse_affmat)

            if (self.skip_fine_grained==False):
                nn_start = time.time() 
                if (self.verbose):
                    print("(Round "+str(round_num)+") Compute nearest neighbors"
                          +" from coarse affmat")
                    print_memory_use()
                    sys.stdout.flush()

                seqlet_neighbors = get_seqlet_neighbors_with_initcluster(
                    nearest_neighbors_to_compute=
                     self.nearest_neighbors_to_compute,
                    coarse_affmat=coarse_affmat,
                    initclusters=initclusters)

                if (self.verbose):
                    print("Computed nearest neighbors in",
                          round(time.time()-nn_start,2),"s")
                    print_memory_use()
                    sys.stdout.flush()

                nn_affmat_start = time.time() 
                if (self.verbose):
                    print("(Round "+str(round_num)+") Computing affinity matrix"
                          +" on nearest neighbors")
                    print_memory_use()
                    sys.stdout.flush()
                nn_affmat = self.affmat_from_seqlets_with_nn_pairs(
                                    seqlet_neighbors=seqlet_neighbors,
                                    seqlets=seqlets) 
                #nn_affmats.append(nn_affmat)
                
                if (self.verbose):
                    print("(Round "+str(round_num)+") Computed affinity matrix"
                          +" on nearest neighbors in",
                          round(time.time()-nn_affmat_start,2),"s")
                    print_memory_use()
                    sys.stdout.flush()

                #filter by correlation
                if (round_idx == 0 or self.filter_beyond_first_round==True):
                    #the filter_mask_from_correlation function only operates
                    # on columns in which np.abs(main_affmat) > 0
                    filtered_rows_mask = self.filter_mask_from_correlation(
                                            main_affmat=nn_affmat,
                                            other_affmat=coarse_affmat) 
                    if (self.verbose):
                        print("(Round "+str(round_num)+") Retained "
                              +str(np.sum(filtered_rows_mask))
                              +" rows out of "+str(len(filtered_rows_mask))
                              +" after filtering")
                        print_memory_use()
                        sys.stdout.flush()
                else:
                    filtered_rows_mask = np.array([True for x in seqlets])
                    if (self.verbose):
                        print("Not applying filtering for "
                              +"rounds above first round")
                        print_memory_use()
                        sys.stdout.flush()

                filtered_seqlets = [x[0] for x in
                    zip(seqlets, filtered_rows_mask) if (x[1])]
                if (initclusters is not None):
                    filtered_initclusters = initclusters[filtered_rows_mask] 
                else:
                    filtered_initclusters = None
                #filtered_seqlets_sets.append(filtered_seqlets)

                filtered_affmat =\
                    nn_affmat[filtered_rows_mask][:,filtered_rows_mask]
                del coarse_affmat
                del nn_affmat
            else:
                filtered_affmat = coarse_affmat
                filtered_seqlets = seqlets
                if (initclusters is not None):
                    filtered_initclusters = initclusters
                else:
                    filtered_initclusters = None

            if (self.verbose):
                print("(Round "+str(round_num)+") Computing density "
                      +"adapted affmat")
                print_memory_use()
                sys.stdout.flush() 

            density_adapted_affmat =\
                self.density_adapted_affmat_transformer(filtered_affmat)
            del filtered_affmat
            #density_adapted_affmats.append(density_adapted_affmat)

            if (self.verbose):
                print("(Round "+str(round_num)+") Computing clustering")
                print_memory_use()
                sys.stdout.flush() 

            cluster_results = clusterer(density_adapted_affmat,
                                        initclusters=filtered_initclusters)
            del density_adapted_affmat
            #cluster_results_sets.append(cluster_results)
            num_clusters = max(cluster_results.cluster_indices+1)
            cluster_idx_counts = Counter(cluster_results.cluster_indices)
            if (self.verbose):
                print("Got "+str(num_clusters)
                      +" clusters after round "+str(round_num))
                print("Counts:")
                print(dict([x for x in cluster_idx_counts.items()]))
                print_memory_use()
                sys.stdout.flush()

            if (self.verbose):
                print("(Round "+str(round_num)+") Aggregating seqlets"
                      +" in each cluster")
                print_memory_use()
                sys.stdout.flush()

            cluster_to_motif, cluster_to_eliminated_motif =\
                self.get_cluster_to_aggregate_motif(
                    seqlets=filtered_seqlets,
                    cluster_indices=cluster_results.cluster_indices,
                    sign_consistency_check=True,
                    min_seqlets_in_motif=0)

            #obtain unique seqlets from adjusted motifs
            seqlets = list(dict([(y.exidx_start_end_string, y)
                             for x in cluster_to_motif.values()
                             for y in x.seqlets]).values())

        if (self.verbose):
            print("Got "+str(len(cluster_to_motif.values()))+" clusters")
            print("Splitting into subclusters...")
            print_memory_use()
            sys.stdout.flush()

        split_patterns = self.spurious_merge_detector(
                            cluster_to_motif.values())

        if (len(split_patterns)==0):
            if (self.verbose):
                print("No more surviving patterns - bailing!")
            return SeqletsToPatternsResults(
                    patterns=None,
                    seqlets=None,
                    affmat=None,
                    cluster_results=None, 
                    total_time_taken=None,
                    success=False)

        #Now start merging patterns 
        if (self.verbose):
            print("Merging on "+str(len(split_patterns))+" clusters")
            print_memory_use()
            sys.stdout.flush()
        merged_patterns, pattern_merge_hierarchy =\
            self.similar_patterns_collapser( 
                patterns=split_patterns, seqlets=seqlets) 
        merged_patterns = sorted(merged_patterns, key=lambda x: -x.num_seqlets)
        if (self.verbose):
            print("Got "+str(len(merged_patterns))+" patterns after merging")
            print_memory_use()
            sys.stdout.flush()

        if (self.verbose):
            print("Performing seqlet reassignment")
            print_memory_use()
            sys.stdout.flush()
        reassigned_patterns = self.seqlet_reassigner(merged_patterns)
        final_patterns = self.final_postprocessor(reassigned_patterns)
        if (self.verbose):
            print("Got "+str(len(final_patterns))
                  +" patterns after reassignment")
            print_memory_use()
            sys.stdout.flush()

        total_time_taken = round(time.time()-start,2)
        if (self.verbose):
            print("Total time taken is "
                  +str(total_time_taken)+"s")
            print_memory_use()
            sys.stdout.flush()

        results = SeqletsToPatternsResults(
            each_round_initcluster_motifs=each_round_initcluster_motifs,             
            patterns=final_patterns,
            seqlets=filtered_seqlets, #last stage of filtered seqlets
            #affmat=filtered_affmat,
            cluster_results=cluster_results, 
            total_time_taken=total_time_taken,
           
            #seqlets_sets=seqlets_sets,
            #coarse_affmats=coarse_affmats,
            #nn_affmats=nn_affmats,
            #filtered_seqlets_sets=filtered_seqlets_sets,
            #filtered_affmats=filtered_affmats,
            #density_adapted_affmats=density_adapted_affmats,
            #cluster_results_sets=cluster_results_sets,
            #cluster_to_motif_sets=cluster_to_motif_sets,
            #cluster_to_eliminated_motif_sets=cluster_to_eliminated_motif_sets,

            merged_patterns=merged_patterns,
            pattern_merge_hierarchy=pattern_merge_hierarchy,
            reassigned_patterns=reassigned_patterns)

        return results
