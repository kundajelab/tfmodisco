from __future__ import division, absolute_import, print_function
from modisco import affinitymat
from modisco import nearest_neighbors
from modisco import cluster
from modisco import aggregator
from modisco import core
from modisco import util
from collections import defaultdict, OrderedDict, Counter
import numpy as np
import time
import sys
import gc
from modisco.tfmodisco_workflow import seqlets_to_patterns as seq_to_pat

import logging

def calc_jaccard_seqlets_and_patterns(seqlets_to_patterns, seqlets, patterns):
    '''
    calculate jaccard similarity between seqlets and patterns
    Args:
        seqlets_to_patterns: TfModiscoSeqletsToPatterns created for a given metacluster
        seqlets:  seqlets to be classified
        patterns: patterns in given metacluster to be mapped to
    Returns:
        jaccard similarity matrix
    '''

    nn_affmat = seqlets_to_patterns.affmat_from_seqlets_with_nn_pairs(
        seqlets=seqlets,
        filter_seqlets=patterns,
        seqlet_neighbors=None)
    return nn_affmat

def prepare_ref_affmat_list(seqlets_to_patterns, seqlets_per_pattern, patterns):
    '''
    Prepare the reference affmat for the calculation of normalized jaccard similarities
    Args:
        seqlets_to_patterns: TfModiscoSeqletsToPatterns created for a given metacluster
        seqlets_per_pattern: list of lists, each list contains the 'old' seqlets of a pattern
        patterns: patterns in given metacluster
    Returns:
        list of sorted affmat, an 1-dim array
    '''
    ref_affmat_list = []
    # for each pattern
    for c, pattern_seqlet, prior_seqlets in zip(range(len(patterns)),
                                                patterns,
                                                seqlets_per_pattern):
        # set up sorted Jaccard 'reference' values for this pattern in order
        # to calculate percentile of new seqlet's Jaccard distance later on
        reference_affmat = calc_jaccard_seqlets_and_patterns(seqlets_to_patterns, prior_seqlets,
                                                             [pattern_seqlet])
        ref_affmat_sorted = np.sort(reference_affmat[:,0]) # make a one dim array
        ref_affmat_list.append(ref_affmat_sorted)

    return ref_affmat_list

def calc_norm_jaccard_seqlets_and_patterns(seqlets_to_patterns, seqlets, patterns, ref_affmat_list):
    '''
    Calculate normalized jaccard similarity between seqlets and patterns
    Args:
        seqlet_to_pattern: TfModiscoSeqletsToPatterns created for a given metacluster
        seqlets: seqlets to be classified
        patterns: patterns in given metacluster to be mapped to
        ref_affmat_list: list of 1-dim arrays containing Jaccard similarities between each pattern and the seqlets supporting that pattern. The 1-dim arraies may have different lengths.
    Returns:
        normalized Jaccard similarities
    '''

    # nn_affmat: (# of seqlets) x (# of patterns)
    nn_affmat = calc_jaccard_seqlets_and_patterns(seqlets_to_patterns, seqlets, patterns)
    norm_nn_affmat = np.empty(nn_affmat.shape)

    for c, pattern_seqlet, prior_affmat_sorted in \
            zip(range(len(patterns)), patterns, ref_affmat_list):
        for r in range(len(seqlets)):
            norm_nn_affmat[r][c] = np.searchsorted(prior_affmat_sorted, nn_affmat[r][c]) / len(prior_affmat_sorted)
    return norm_nn_affmat

def map_seqlets_to_patterns(seqlets_to_patterns, seqlets, prior_results, metacluster_idx):
    '''
    Based on tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatterns.__call__().
    TODO: could merge later
    This function is called for each metacluster from incr_workflow.TfModiscoIncrementalWorkflow.__call(),
    to map seqlets to existing patterns from previous Modisco results.
    Args:
        seqlets_to_patterns: instance of TfModiscoSeqletsToPatterns
        seqlets:             list of new seqlets to be clustered
        prior_results:       previous clustering results
        metacluster_idx:     idx of the meterculster
    Returns:
    '''
    seqlets = seqlets_to_patterns.seqlets_sorter(seqlets) # TODO: why do we need to sort?

    start = time.time()

    ###################################
    metacluster_name = prior_results.metacluster_names[metacluster_idx]
    metacluster = prior_results.metaclusters[metacluster_name]
    patterns = metacluster.pattern_seqlets

    logging.debug("map_seqlets_to_patterns(): metacluster " + metacluster_name)

    # calculate jaccard similarity between seqlets and patterns
    nn_affmat_start = time.time()

    ref_affmat_list = prepare_ref_affmat_list(seqlets_to_patterns, metacluster.seqlets_per_pattern, patterns)

    norm_nn_affmat = calc_norm_jaccard_seqlets_and_patterns(seqlets_to_patterns, seqlets, patterns, ref_affmat_list)
    # for every pattern, get prior_affmat & sort

    max_aff = np.argmax(norm_nn_affmat, axis=1)

    #print("max_aff=")
    #print(max_aff)
    #sys.stdout.flush()

    ###################################
    # check the old seqlets

    all_prior_seqlets = []
    for prior_seqlets in metacluster.seqlets_per_pattern:
        all_prior_seqlets += prior_seqlets

    prior_norm_nn_affmat = calc_norm_jaccard_seqlets_and_patterns(seqlets_to_patterns, all_prior_seqlets, patterns, ref_affmat_list)
    prior_max_aff = np.argmax(prior_norm_nn_affmat, axis=1)

    # evaluate accuracy of motif hits algorithm
    all_exp_max = []
    for idx, prior_seqlets in enumerate(metacluster.seqlets_per_pattern):
        exp_max = [idx]*(len(prior_seqlets))
        all_exp_max += exp_max

    cmp = [x==y for (x,y) in zip (list(prior_max_aff), all_exp_max)]

    logging.debug("Compare with prior clustering: %d correct out of %d (%f)" % (sum(cmp), len(cmp), sum(cmp)/len(cmp)))
    return None # will return proper results later.

    ###################################
    """
    seqlets_sets = []
    coarse_affmats = []
    nn_affmats = []
    filtered_seqlets_sets = []
    filtered_affmats = []
    density_adapted_affmats = []
    cluster_results_sets = []
    cluster_to_motif_sets = []
    cluster_to_eliminated_motif_sets = []


    # following are from TfModiscoSeqletsToPatterns.__call__(). 
    # They are kept here since they are useful for subsequent experiments.
    for round_idx, clusterer in enumerate(seqlet_to_pattern.clusterer_per_round):
        for i in range(3): gc.collect()

        round_num = round_idx + 1

        seqlets_sets.append(seqlets)

        if (len(seqlets) == 0):
            if (seqlet_to_pattern.verbose):
                print("len(seqlets) is 0 - bailing!")
            return SeqletsToPatternsResults(
                patterns=None,
                seqlets=None,
                affmat=None,
                cluster_results=None,
                total_time_taken=None,
                success=False)
        '''
        if (seqlet_to_pattern.verbose):
            print("(Round " + str(round_num) +
                  ") num seqlets: " + str(len(seqlets)))
            print("(Round " + str(round_num) + ") Computing coarse affmat")
            sys.stdout.flush()

        coarse_affmat = seqlet_to_pattern.coarse_affmat_computer(seqlets)
        coarse_affmats.append(coarse_affmat)

        nn_start = time.time()
        if (seqlet_to_pattern.verbose):
            print("(Round " + str(round_num) + ") Compute nearest neighbors"
                  + " from coarse affmat")
            sys.stdout.flush()

        seqlet_neighbors = seqlet_to_pattern.nearest_neighbors_computer(coarse_affmat)

        if (seqlet_to_pattern.verbose):
            print("Computed nearest neighbors in",
                  round(time.time() - nn_start, 2), "s")
            sys.stdout.flush()
        '''

        # calculate jaccard similarity between seqlets and patterns, given the seqlets_neighbors_new array
        nn_affmat_start = time.time()
        if (seqlet_to_pattern.verbose):
            print("(Round " + str(round_num) + ") Computing affinity matrix"
                  + " on nearest neighbors")
            sys.stdout.flush()
        nn_affmat = seqlet_to_pattern.affmat_from_seqlets_with_nn_pairs(
            seqlet_neighbors=seqlet_neighbors_new,
            seqlets=seqlets)
        nn_affmats.append(nn_affmat)
        if (seqlet_to_pattern.verbose):
            print("(Round " + str(round_num) + ") Computed affinity matrix"
                  + " on nearest neighbors in",
                  round(time.time() - nn_affmat_start, 2), "s")
            sys.stdout.flush()

        # seqlet_neighbors_new.shape == (len(seqlets_ext), num_patterns)
        norm_nn_affmat = np.zeros(seqlet_neighbors_new.shape)
        for r in range(len(seqlets_ext)):
            for c in range(num_patterns):
                norm_nn_affmat = np.searchsorted(nn_affmat_sorted, nn_affmat[r][c])

        '''
        # filter by correlation, but not useful for incremental run
        if (round_idx == 0 or seqlet_to_pattern.filter_beyond_first_round == True):
            filtered_rows_mask = seqlet_to_pattern.filter_mask_from_correlation(
                main_affmat=nn_affmat,
                other_affmat=coarse_affmat)
            if (seqlet_to_pattern.verbose):
                print("(Round " + str(round_num) + ") Retained "
                      + str(np.sum(filtered_rows_mask))
                      + " rows out of " + str(len(filtered_rows_mask))
                      + " after filtering")
                sys.stdout.flush()
        else:
            filtered_rows_mask = np.array([True for x in seqlets])
            if (seqlet_to_pattern.verbose):
                print("Not applying filtering for "
                      + "rounds above first round")
                sys.stdout.flush()

        filtered_seqlets = [x[0] for x in
                            zip(seqlets, filtered_rows_mask) if (x[1])]

        filtered_affmat = \
            nn_affmat[filtered_rows_mask][:, filtered_rows_mask]
        '''
        filtered_affmat  = nn_affmat
        filtered_seqlets = seqlets
        filtered_seqlets_sets.append(filtered_seqlets)
        filtered_affmats.append(filtered_affmat)

        if (seqlet_to_pattern.verbose):
            print("(Round " + str(round_num) + ") Computing density "
                  + "adapted affmat")
            sys.stdout.flush()

        density_adapted_affmat = \
            seqlet_to_pattern.density_adapted_affmat_transformer(filtered_affmat)
        density_adapted_affmats.append(density_adapted_affmat)

        if (seqlet_to_pattern.verbose):
            print("(Round " + str(round_num) + ") Computing clustering")
            sys.stdout.flush()

        cluster_results = clusterer(density_adapted_affmat)
        cluster_results_sets.append(cluster_results)
        num_clusters = max(cluster_results.cluster_indices + 1)
        cluster_idx_counts = Counter(cluster_results.cluster_indices)
        if (seqlet_to_pattern.verbose):
            print("Got " + str(num_clusters)
                  + " clusters after round " + str(round_num))
            print("Counts:")
            print(dict([x for x in cluster_idx_counts.items()]))
            sys.stdout.flush()

        if (seqlet_to_pattern.verbose):
            print("(Round " + str(round_num) + ") Aggregating seqlets"
                  + " in each cluster")
            sys.stdout.flush()

        cluster_to_seqlets = defaultdict(list)
        assert len(filtered_seqlets) == len(cluster_results.cluster_indices)
        for seqlet, idx in zip(filtered_seqlets,
                               cluster_results.cluster_indices):
            cluster_to_seqlets[idx].append(seqlet)

        cluster_to_eliminated_motif = OrderedDict()
        cluster_to_motif = OrderedDict()
        cluster_to_motif_sets.append(cluster_to_motif)
        cluster_to_eliminated_motif_sets.append(
            cluster_to_eliminated_motif)
        for i in range(num_clusters):
            if (seqlet_to_pattern.verbose):
                print("Aggregating for cluster " + str(i) + " with "
                      + str(len(cluster_to_seqlets[i])) + " seqlets")
                sys.stdout.flush()
            motifs = seqlet_to_pattern.seqlet_aggregator(cluster_to_seqlets[i])
            assert len(motifs) == 1
            motif = motifs[0]
            if (seqlet_to_pattern.sign_consistency_func(motif)):
                cluster_to_motif[i] = motif
            else:
                if (seqlet_to_pattern.verbose):
                    print("Dropping cluster " + str(i) +
                          " with " + str(motif.num_seqlets)
                          + " seqlets due to sign disagreement")
                cluster_to_eliminated_motif[i] = motif

        # obtain unique seqlets from adjusted motifs
        seqlets = dict([(y.exidx_start_end_string, y)
                        for x in cluster_to_motif.values()
                        for y in x.seqlets]).values()
    if (seqlet_to_pattern.verbose):
        print("Got " + str(len(cluster_to_motif.values())) + " clusters")
        print("Splitting into subclusters...")
        sys.stdout.flush()

    split_patterns = seqlet_to_pattern.spurious_merge_detector(
        cluster_to_motif.values())

    # Now start merging patterns
    if (seqlet_to_pattern.verbose):
        print("Merging on " + str(len(split_patterns)) + " clusters")
        sys.stdout.flush()
    merged_patterns, pattern_merge_hierarchy = \
        seqlet_to_pattern.similar_patterns_collapser(
            patterns=split_patterns, seqlets=seqlets)
    merged_patterns = sorted(merged_patterns, key=lambda x: -x.num_seqlets)
    if (seqlet_to_pattern.verbose):
        print("Got " + str(len(merged_patterns)) + " patterns after merging")
        sys.stdout.flush()

    if (seqlet_to_pattern.verbose):
        print("Performing seqlet reassignment")
        sys.stdout.flush()
    reassigned_patterns = seqlet_to_pattern.seqlet_reassigner(merged_patterns)
    final_patterns = seqlet_to_pattern.final_postprocessor(reassigned_patterns)
    if (seqlet_to_pattern.verbose):
        print("Got " + str(len(final_patterns))
              + " patterns after reassignment")
        sys.stdout.flush()

    total_time_taken = round(time.time() - start, 2)
    if (seqlets_to_patterns.verbose):
        print("Total time taken is "
              + str(total_time_taken) + "s")
        sys.stdout.flush()

    results = seq_to_pat.SeqletsToPatternsResults(
        patterns=final_patterns,
        seqlets=filtered_seqlets,  # last stage of filtered seqlets
        affmat=filtered_affmat,
        cluster_results=cluster_results,
        total_time_taken=total_time_taken,

        seqlets_sets=seqlets_sets,
        coarse_affmats=coarse_affmats,
        nn_affmats=nn_affmats,
        filtered_seqlets_sets=filtered_seqlets_sets,
        filtered_affmats=filtered_affmats,
        density_adapted_affmats=density_adapted_affmats,
        cluster_results_sets=cluster_results_sets,
        cluster_to_motif_sets=cluster_to_motif_sets,
        cluster_to_eliminated_motif_sets=cluster_to_eliminated_motif_sets,

        merged_patterns=merged_patterns,
        pattern_merge_hierarchy=pattern_merge_hierarchy,
        reassigned_patterns=reassigned_patterns)

    return results
    """

