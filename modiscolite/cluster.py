import os
import re
import uuid
import leidenalg
import numpy as np
import igraph as ig

from joblib import Parallel, delayed

def LeidenCluster(orig_affinity_mat, initclusters, n_jobs,
    numseedstotry=10, n_leiden_iterations=-1,
    partitiontype=leidenalg.ModularityVertexPartition,
    affmat_transformer=None, refine=False, verbose=True):

    if affmat_transformer is not None:
        affinity_mat = affmat_transformer(orig_affinity_mat)
    else:
        affinity_mat = orig_affinity_mat

    best_clustering = None
    best_quality = None

    toiterover = range(numseedstotry)

    #if an initclustering is specified, we would want to try the Leiden
    # both with and without that initialization and take the one that
    # gets the best modularity. EXCEPT when refinement is also specified.
    if refine:
        initclusters_to_try_list = [True]
    else:
        initclusters_to_try_list = [False]
        if (initclusters is not None):
            initclusters_to_try_list.append(True)

    #write out the contents of affinity_mat and initclusters if applicable
    uid = uuid.uuid1().hex
    
    sources, targets = affinity_mat.nonzero()
    weights = affinity_mat[sources, targets]

    np.save(uid+"_sources.npy", sources)
    np.save(uid+"_targets.npy", targets)
    np.save(uid+"_weights.npy", weights.A1) #A1 is the same as ravel()

    del sources, targets, weights

    if (initclusters is not None):
        np.save(uid+"_initclusters.npy", initclusters)

    for use_initclusters in initclusters_to_try_list:
        parallel_leiden_results = (
            Parallel(n_jobs=1, verbose=verbose)(delayed(run_leiden)(
                uid, use_initclusters, affinity_mat.shape[0], partitiontype,
                n_leiden_iterations, seed*100, refine) for seed in [1, 2])) 

        for quality, membership in parallel_leiden_results:
            if (best_quality is None) or (quality > best_quality):
                best_quality = quality
                best_clustering = membership
                if verbose:
                    print("Quality:",best_quality)

    # clean up
    for f in os.listdir(os.getcwd()):
        if re.search(uid, f):
            os.remove(f)

    return {
        'cluster_indices': best_clustering,
        'quality': best_quality
    }

     
#based on https://github.com/theislab/scanpy/blob/8131b05b7a8729eae3d3a5e146292f377dd736f7/scanpy/_utils.py#L159
def get_igraph(sources_idxs_file, targets_idxs_file, weights_file, n_vertices):
    sources = np.load(sources_idxs_file) 
    targets = np.load(targets_idxs_file) 
    weights = np.load(weights_file)
    
    g = ig.Graph(directed=None) 
    g.add_vertices(n_vertices)
    g.add_edges(zip(sources, targets)) 
    g.es['weight'] = weights
    return g 

def run_leiden(fileprefix, use_initclusters, n_vertices, partition_type, n_iterations, seed, refine):
    sources_idxs_file = fileprefix + "_sources.npy"
    targets_idxs_file = fileprefix + "_targets.npy"
    weights_file = fileprefix + "_weights.npy"

    partition_type = partition_type.__name__

    if use_initclusters:
        initial_membership_file = fileprefix + "_initclusters.npy"
    else:
        initial_membership_file = None


    the_graph = get_igraph(
        sources_idxs_file=sources_idxs_file,
        targets_idxs_file=targets_idxs_file,
        weights_file=weights_file,
        n_vertices=n_vertices)

    partition_type = eval("leidenalg." + partition_type) 

    initial_membership = (None if initial_membership_file is None
                          else np.load(initial_membership_file).tolist())

    #weights = np.array(the_graph.es['weight']).astype(np.float64).tolist()
    weights = (np.array(the_graph.es['weight']).astype(np.float64)).tolist()

    if refine == False:
        partition = leidenalg.find_partition(
            graph=the_graph,
            partition_type=partition_type,
            weights=weights, 
            n_iterations=n_iterations,
            initial_membership=initial_membership,    
            seed=seed) 
    else:
        #Refine the partition suggested by initial_membership
        #code here is based on a combination of find_partition code
        # at https://github.com/vtraag/leidenalg/blob/9ffd92ada566d7cce094afd4ec5c70209609af26/src/functions.py#L26
        # and the discussion of refining communities in https://leidenalg.readthedocs.io/en/stable/advanced.html#optimiser
        constraining_partition = partition_type(
                        graph=the_graph, initial_membership=initial_membership,
                        weights=weights) 
        #if initial_membership is not specified, each node is placed in its
        # own cluster, as per https://leidenalg.readthedocs.io/en/stable/reference.html#mutablevertexpartition
        refined_partition_movenodes =\
            partition_type(graph=the_graph, weights=weights) 
        refined_partition_mergenodes =\
            partition_type(graph=the_graph, weights=weights) 

        #Github issue discussing things is here: https://github.com/vtraag/leidenalg/issues/61
        #move_nodes is what is used in the original louvain
        # (as per https://leidenalg.readthedocs.io/en/stable/advanced.html#optimiser)
        #merge nodes is used for the refinement step of Leiden

        #With Leiden 0.8.4, due to segfault bug, can't do move nodes:
        # https://github.com/vtraag/leidenalg/issues/68

        #with move nodes
        optimiser = leidenalg.Optimiser() 
        optimiser.set_rng_seed(seed)
        optimiser.move_nodes_constrained(refined_partition_movenodes,
                                         constraining_partition)

        #with merge nodes
        optimiser = leidenalg.Optimiser() 
        optimiser.set_rng_seed(seed)
        optimiser.merge_nodes_constrained(refined_partition_mergenodes,
                                          constraining_partition)
        #partition = refined_partition_mergenodes

        #take the partition with the best quality
        partition = (refined_partition_movenodes if (
                          refined_partition_movenodes.quality() >
                          refined_partition_mergenodes.quality()) 
                          else refined_partition_mergenodes)

    return np.array(partition.quality()), np.array(partition.membership)
