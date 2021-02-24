from __future__ import division, print_function
import os
import signal
import subprocess
import numpy as np
import h5py
import traceback
import scipy.sparse


def print_memory_use():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print("MEMORY",process.memory_info().rss/1000000000)


def load_patterns(grp, track_set):
    from modisco.core import AggregatedSeqlet
    all_pattern_names = load_string_list(dset_name="all_pattern_names",
                                         grp=grp)
    patterns = []
    for pattern_name in all_pattern_names:
        pattern_grp = grp[pattern_name] 
        patterns.append(AggregatedSeqlet.from_hdf5(grp=pattern_grp,
                                                   track_set=track_set))
    return patterns


def save_patterns(patterns, grp):
    all_pattern_names = []
    for idx, pattern in enumerate(patterns):
        pattern_name = "pattern_"+str(idx)
        all_pattern_names.append(pattern_name)
        pattern_grp = grp.create_group(pattern_name) 
        pattern.save_hdf5(pattern_grp)
    save_string_list(all_pattern_names, dset_name="all_pattern_names",
                     grp=grp)


def flatten_seqlet_impscore_features(seqlet_impscores):
    return np.reshape(seqlet_impscores, (len(seqlet_impscores), -1))


def coo_matrix_from_neighborsformat(entries, neighbors, ncols):
    coo_mat = scipy.sparse.coo_matrix(
            (np.concatenate(entries, axis=0),
             (np.array([i for i in range(len(neighbors))
                           for j in neighbors[i]]).astype("int"),
              np.concatenate(neighbors, axis=0)) ),
            shape=(len(entries), ncols)) 
    return coo_mat


def load_string_list(dset_name, grp):
    return [x.decode("utf-8") for x in grp[dset_name][:]]


def save_string_list(string_list, dset_name, grp):
    dset = grp.create_dataset(dset_name, (len(string_list),),
                              dtype=h5py.special_dtype(vlen=bytes))
    dset[:] = string_list


def load_seqlet_coords(dset_name, grp):
    from modisco.core import SeqletCoordinates
    coords_strings = load_string_list(dset_name=dset_name, grp=grp)
    return [SeqletCoordinates.from_string(x) for x in coords_strings] 


def save_seqlet_coords(seqlets, dset_name, grp):
    coords_strings = [str(x.coor) for x in seqlets] 
    save_string_list(string_list=coords_strings,
                     dset_name=dset_name, grp=grp)


def save_list_of_objects(grp, list_of_objects):
    grp.attrs["num_objects"] = len(list_of_objects) 
    for idx,obj in enumerate(list_of_objects):
        obj.save_hdf5(grp=grp.create_group("obj"+str(idx)))


def load_list_of_objects(grp, obj_class):
    num_objects = grp.attrs["num_objects"]
    list_of_objects = []
    for idx in range(num_objects):
        list_of_objects.append(obj_class.from_hdf5(grp=grp["obj"+str(idx)]))
    return list_of_objects


def factorial(val):
    to_return = 1
    for i in range(1,val+1):
        to_return *= i
    return to_return


def first_curvature_max(values, bins, bandwidth):
    from sklearn.neighbors.kde import KernelDensity
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
                [[x,0] for x in values])
    midpoints = np.min(values)+((np.arange(bins)+0.5)
                                *(np.max(values)-np.min(values))/bins)
    densities = np.exp(kde.score_samples([[x,0] for x in midpoints]))

    global_max_x = max(zip(densities,midpoints), key=lambda x: x[0])[1]
    firstd_x, firstd_y = angle_firstd(x_values=midpoints, y_values=densities) 
    secondd_x, secondd_y = firstd(x_values=firstd_x, y_values=firstd_y)
    thirdd_x, thirdd_y = firstd(x_values=secondd_x, y_values=secondd_y)
    #find curvature maxima i.e. points where thirdd crosses 0
    maxima_x = [0.5*(prev_x+after_x) for (prev_x, after_x),(prev_y,after_y)
                  in zip(zip(thirdd_x[0:-1], thirdd_x[1:]),
                         zip(thirdd_y[0:-1], thirdd_y[1:]))
                  if (prev_y > 0 and after_y < 0)
                  and 0.5*(prev_x+after_x)]
    maxima_x_after_global_max = [x for x in maxima_x if x > global_max_x]
    maxima_x_before_global_max = [x for x in maxima_x if x < global_max_x]
    threshold_before = maxima_x_before_global_max[-1] if\
                        len(maxima_x_before_global_max) > 0 else global_max_x
    threshold_after = maxima_x_after_global_max[0] if\
                        len(maxima_x_after_global_max) > 0 else global_max_x

    from matplotlib import pyplot as plt
    hist_y, _, _ = plt.hist(values, bins=100)
    max_y = np.max(hist_y)
    plt.plot(midpoints, densities*(max_y/np.max(densities)))
    plt.plot([threshold_before, threshold_before], [0, max_y])
    plt.plot([threshold_after, threshold_after], [0, max_y])
    plt.show()

    return threshold_before, threshold_after


def cosine_firstd(x_values, y_values):
    x_differences = x_values[1:] - x_values[:-1]
    x_midpoints = 0.5*(x_values[1:] + x_values[:-1])
    y_differences = y_values[1:] - y_values[:-1]
    hypotenueses = np.sqrt(np.square(y_differences) + np.square(x_differences))
    cosine_first_d = x_differences/hypotenueses 
    return x_midpoints, cosine_first_d


def angle_firstd(x_values, y_values):
    x_differences = x_values[1:] - x_values[:-1]
    x_midpoints = 0.5*(x_values[1:] + x_values[:-1])
    y_differences = y_values[1:] - y_values[:-1]
    return x_midpoints, np.arctan2(y_differences, x_differences)


def angle_curvature(x_values, y_values):
    x_midpoints, y_angles = angle_firstd(x_values, y_values)
    y_midpoints = 0.5*(y_values[1:] + y_values[:-1])
    x_midmidpoints, y_anglechange = firstd(x_midpoints, y_angles)
    x_differences = x_midpoints[1:] - x_midpoints[:-1] 
    y_differences = y_midpoints[1:] - y_midpoints[:-1]
    distance_travelled = np.sqrt(np.square(x_differences)+
                                 np.square(y_differences))
    angle_change_w_dist = y_anglechange/distance_travelled
    return x_midmidpoints, angle_change_w_dist


def firstd(x_values, y_values):
    x_differences = x_values[1:] - x_values[:-1]
    x_midpoints = 0.5*(x_values[1:] + x_values[:-1])
    y_differences = y_values[1:] - y_values[:-1]
    rise_over_run = y_differences/x_differences
    return x_midpoints, rise_over_run


def cpu_sliding_window_sum(arr, window_size):
    assert len(arr) >= window_size, str(len(arr))+" "+str(window_size)
    to_return = np.zeros(len(arr)-window_size+1)
    current_sum = np.sum(arr[0:window_size])
    to_return[0] = current_sum
    idx_to_include = window_size
    idx_to_exclude = 0
    while idx_to_include < len(arr):
        current_sum += (arr[idx_to_include] - arr[idx_to_exclude]) 
        to_return[idx_to_exclude+1] = current_sum
        idx_to_include += 1
        idx_to_exclude += 1
    return to_return


def identify_peaks(arr):
    #use a state machine to identify peaks
    #"peaks" as defined by larger than neighbours
    #for tied region, take the middle of the tie.
    #return tuples of idx + peak val
    previous_val = None
    potential_peak_start_idx = None
    found_peaks = []
    for idx, val in enumerate(arr):
        if (previous_val is not None):
            if (val > previous_val):
                potential_peak_start_idx = idx
            elif (val < previous_val):
                if (potential_peak_start_idx is not None):
                    #peak found!
                    found_peaks.append(
        (int(0.5*(potential_peak_start_idx+(idx-1))), previous_val))
                potential_peak_start_idx = None
                potential_peak_start_val = None
            else:
                #tie...don't change anything.
                pass
        previous_val = val
    return found_peaks


def get_top_N_scores_per_region(scores, N, exclude_hits_within_window):
    scores = scores.copy()
    assert len(scores.shape)==2, scores.shape
    if (N==1):
        return np.max(scores, axis=1)[:,None]
    else:
        top_n_scores = []
        for i in range(scores.shape[0]):
            top_n_scores_for_region=[]
            for n in range(N):
                max_idx = np.argmax(scores[i]) 
                top_n_scores_for_region.append(scores[i][max_idx])
                scores[i][max_idx-exclude_hits_within_window:
                          max_idx+exclude_hits_within_window-1] = -np.inf
            top_n_scores.append(top_n_scores_for_region) 
        return np.array(top_n_scores)


def phenojaccard_sim_mat(sim_mat, k):
    from collections import defaultdict
    node_to_nearest = defaultdict(set)
    for node,neighbours_affs in enumerate(sim_mat):
        sorted_neighbours_affs = sorted(enumerate(neighbours_affs), key=lambda x: -x[1])
        node_to_nearest[node].update([x[0] for x in sorted_neighbours_affs[:k]])
    new_sim_mat = np.zeros_like(sim_mat)
    for node1 in node_to_nearest:
        for node2 in node_to_nearest:
            intersection = set(node_to_nearest[node1])
            intersection.intersection_update(node_to_nearest[node2])
            union = set(node_to_nearest[node1])
            union.update(node_to_nearest[node2])
            jaccard = float(len(intersection))/float(len(union))
            new_sim_mat[node1,node2] = jaccard
    return new_sim_mat
 

def jaccardify_sim_mat(sim_mat, verbose=True, power=1):
    print("Seriously consider using phenojaccard")
    if (verbose):
        print("calling jaccardify")
    sim_mat = np.power(sim_mat, power)
    import time
    t1 = time.time()
    minimum_sum = np.sum(np.minimum(sim_mat[:,None,:],
                         sim_mat[None,:,:]), axis=-1)
    maximum_sum = np.sum(np.maximum(sim_mat[:,None,:],
                         sim_mat[None,:,:]), axis=-1)
    ratio = minimum_sum/maximum_sum
    t2 = time.time()
    if (verbose):
        print("time taken in jaccardify",t2-t1)
    return ratio 


def compute_jaccardify(sim_mat, start_job, end_job):
    num_nodes = sim_mat.shape[0]
    distances = []
    for job_num in xrange(start_job, end_job):
        row_idx = int(job_num/num_nodes)
        col_idx = job_num%num_nodes
        minimum_sum = np.sum(np.minimum(sim_mat[row_idx,:],
                                        sim_mat[col_idx,:]))
        maximum_sum = np.sum(np.maximum(sim_mat[row_idx,:],
                                        sim_mat[col_idx,:]))
        ratio = minimum_sum/maximum_sum
        distances.append(ratio)
    return distances


#should be speed-upable further by recognizing that the distance is symmetric
def parallel_jaccardify(sim_mat, num_processes=4,
                        verbose=True, power=1,
                        temp_file_dir="tmp",
                        temp_file_prefix="jaccardify_h5"):

    if (os.path.isdir(temp_file_dir)==False):
        os.system("mkdir "+temp_file_dir)
    sim_mat = np.power(sim_mat, power)

    num_nodes = sim_mat.shape[0]
    total_tasks = num_nodes**2
    tasks_per_job = int(np.ceil(total_tasks/num_processes))

    launched_pids = []
    print(num_processes)
    for i in xrange(num_processes):
        pid = os.fork() 
        print(pid)
        if pid==0:
            try:
                #set a signal handler for interrupt signals
                signal.signal(signal.SIGINT,
                              (lambda signum, frame: os._exit(os.EX_TEMPFAIL)))
                start_job = tasks_per_job*i
                end_job = min(total_tasks, tasks_per_job*(i+1))
                distances = compute_jaccardify(sim_mat, start_job, end_job) 
                #write the distances to an h5 file
                h5_file_name = temp_file_dir+"/"\
                               +temp_file_prefix+"_"+str(i)+".h5"
                f = h5py.File(h5_file_name, "w")
                dset = f.create_dataset("/distances", data=distances)
                f.close()
                print("Exit!")
                os._exit(os.EX_OK) #exit the child
            except (Exception, _):
                raise RuntimeError("Exception in job "+str(i)+\
                                   "\n"+traceback.format_exc()) 
                os._exit(os.EX_SOFTWARE)
        else:
            launched_pids.append(pid)

    try:
        while len(launched_pids) > 0:
            pid, return_code = os.wait()
            if return_code != os.EX_OK:  
                raise RuntimeError(
                "pid "+str(pid)+" gave error code "+str(return_code))
            if pid in launched_pids:
                launched_pids.remove(pid)

        #child processes would have all exited
        collated_distances = []
        #now collate all the stuff written to the various h5 files
        for i in xrange(num_processes):
            h5_file_name = temp_file_dir+"/"\
                           +temp_file_prefix+"_"+str(i)+".h5"
            f = h5py.File(h5_file_name)
            collated_distances.extend(f['/distances'])
            f.close()
            os.system("rm "+h5_file_name)
        assert len(collated_distances) == total_tasks 
        to_return = np.zeros((num_nodes, num_nodes))
        #now reshape the collated distances into a numpy array
        for i in xrange(len(collated_distances)):
            row_idx = int(i/num_nodes)
            col_idx = i%num_nodes
            to_return[row_idx, col_idx] = collated_distances[i]
        return to_return
    except (KeyboardInterrupt, OSError):
        for pid in launched_pids:
            try:
                os.kill(pid, signal.SIGHUP)
            except:
                pass
        raise


def make_graph_from_sim_mat(sim_mat):
    import networkx as nx
    G = nx.Graph()
    print("Adding nodes")
    for i in range(len(sim_mat)):
        G.add_node(i)
    print("nodes added")
    edges_to_add = []
    print("Preparing edges")
    for i in range(len(sim_mat)):
        for j in range(i+1,len(sim_mat)):
            edges_to_add.append((i,j,{'weight':sim_mat[i,j]})) 
    print("Done preparing edges")
    G.add_edges_from(edges_to_add)
    print("Done adding edges")
    return G


def cluster_louvain(sim_mat):
    import community
    graph = make_graph_from_sim_mat(sim_mat)
    print("making partition")
    partition = community.best_partition(graph)
    print("done making partition")
    louvain_labels = [partition[i] for i in range(len(partition.keys()))]
    return louvain_labels


def get_betas_from_tsne_conditional_probs(conditional_probs,
                                          original_affmat, aff_to_dist_mat):
    dist_mat = aff_to_dist_mat(original_affmat)
    betas = []
    for i,(prob_row, distances, affinities) in\
        enumerate(zip(conditional_probs,
                      dist_mat, original_affmat)):
        nonzero_probs = prob_row[prob_row > 0.0]
        nonzero_distances = distances[prob_row > 0.0]
        prob1, dist1 = max(zip(nonzero_probs, nonzero_distances),
                           key=lambda x: x[1])
        prob2, dist2 = min(zip(nonzero_probs, nonzero_distances),
                           key=lambda x: x[1])
        beta = np.log(prob2/prob1)/(dist1-dist2)
        betas.append(beta)
        #sanity check
        recomputed_probs = np.exp(-beta*(distances))*(affinities > 0.0)
        recomputed_probs[i] = 0
        recomputed_probs = recomputed_probs/np.sum(recomputed_probs)
        test_recomputed_probs = recomputed_probs[prob_row > 0.0]/\
                                 np.sum(recomputed_probs[prob_row > 0.0])
        maxdiff = np.max(np.abs(prob_row[prob_row > 0.0]
                                - test_recomputed_probs))
        assert maxdiff < 10**-5,\
               (np.sum(prob_row), maxdiff, test_recomputed_probs)
    return np.array(betas)


def convert_to_percentiles(vals):
    to_return = np.zeros(len(vals))
    argsort = np.argsort(vals)
    to_return[argsort] = np.arange(len(vals))/float(len(vals))
    #sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
    #for sort_idx,(orig_idx,val) in enumerate(sorted_vals):
    #    to_return[orig_idx] = sort_idx/float(len(vals))
    return to_return


def binary_search_perplexity(desired_perplexity, distances):
    
    EPSILON_DBL = 1e-8
    PERPLEXITY_TOLERANCE = 1e-5
    n_steps = 100
    
    desired_entropy = np.log(desired_perplexity)
    
    beta_min = -np.inf
    beta_max = np.inf
    beta = 1.0
    
    for l in range(n_steps):
        ps = np.exp(-distances * beta)
        sum_ps = np.sum(ps)
        ps = ps/(max(sum_ps,EPSILON_DBL))
        sum_disti_Pi = np.sum(distances*ps)
        entropy = np.log(sum_ps) + beta * sum_disti_Pi
        
        entropy_diff = entropy - desired_entropy
        #print(beta, np.exp(entropy), entropy_diff)
        if np.abs(entropy_diff) <= PERPLEXITY_TOLERANCE:
            break
        
        if entropy_diff > 0.0:
            beta_min = beta
            if beta_max == np.inf:
                beta *= 2.0
            else:
                beta = (beta + beta_max) / 2.0
        else:
            beta_max = beta
            if beta_min == -np.inf:
                beta /= 2.0
            else:
                beta = (beta + beta_min) / 2.0
    return beta, ps


def get_ic_trimming_indices(ppm, background, threshold, pseudocount=0.001):
    """Return tuple of indices to trim to if ppm is trimmed by info content.

    The ppm will be trimmed from the left and from the right until a position
     that meets the information content specified by threshold is found. A
     base of 2 is used for the infromation content.

    Arguments:
        threshold: the minimum information content.
        remaining arguments same as for compute_per_position_ic

    Returns:
        (start_idx, end_idx). start_idx is inclusive, end_idx is exclusive.
    """
    per_position_ic = compute_per_position_ic(
                       ppm=ppm, background=background, pseudocount=pseudocount)
    passing_positions = np.where(per_position_ic >= threshold)
    return (passing_positions[0][0], passing_positions[0][-1]+1)


def compute_per_position_ic(ppm, background, pseudocount):
    """Compute information content at each position of ppm.

    Arguments:
        ppm: should have dimensions of length x alphabet. Entries along the
            alphabet axis should sum to 1.
        background: the background base frequencies
        pseudocount: pseudocount to be added to the probabilities of the ppm
            to prevent overflow/underflow.

    Returns:
        total information content at each positon of the ppm.
    """
    assert len(ppm.shape)==2
    assert ppm.shape[1]==len(background),\
            "Make sure the letter axis is the second axis"
    if (not np.allclose(np.sum(ppm, axis=1), 1.0, atol=1.0e-5)):
        print("WARNING: Probabilities don't sum to 1 in all the rows; this can"
              +" be caused by zero-padding. Will renormalize. PPM:\n"
              +str(ppm)
              +"\nProbability sums:\n"
              +str(np.sum(ppm, axis=1)))
        ppm = ppm/np.sum(ppm, axis=1)[:,None]

    alphabet_len = len(background)
    ic = ((np.log((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/np.log(2))
          *ppm - (np.log(background)*background/np.log(2))[None,:])
    return np.sum(ic,axis=1)


#rolling_window is from this blog post by Erik Rigtorp:
# https://rigtorp.se/2011/01/01/rolling-statistics-numpy.html
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def sliding_window_max(a, window):
    rolling_windows_a = rolling_window(a, window)
    return np.max(rolling_windows_a, axis=-1) 


def compute_masked_cosine_sim(imp_scores, onehot_seq, weightmat): 
    strided_impscores = rolling_window(
        imp_scores.transpose((0,2,1)),
        window=len(weightmat)).transpose((0,2,3,1))
    strided_onehotseq = rolling_window(
        onehot_seq.transpose((0,2,1)),
        window=len(weightmat)).transpose((0,2,3,1))

    #this finds the cosine similarity with a masked version of the weightmat
    # where only the positions that are nonzero in the deeplift scores are
    # considered
    dot_product_imp_weightmat = np.sum(
        strided_impscores*weightmat[None,None,:,:], axis=(2,3))
    norm_deeplift_scores = np.sqrt(np.sum(np.square(strided_impscores),
                                   axis=(2,3)))
    norm_masked_weightmat = np.sqrt(np.sum(np.square(
                                strided_onehotseq*weightmat[None,None,:,:]),
                                axis=(2,3)))
    cosine_sim = dot_product_imp_weightmat/(
                  norm_deeplift_scores*norm_masked_weightmat)
    return cosine_sim


def get_logodds_pwm(ppm, background, pseudocount):
    assert len(ppm.shape)==2
    assert ppm.shape[1]==len(background),\
            "Make sure the letter axis is the second axis"
    assert (np.max(np.abs(np.sum(ppm, axis=1)-1.0)) < 1e-7),(
             "Probabilities don't sum to 1 along axis 1 in "
             +str(ppm)+"\n"+str(np.sum(ppm, axis=1)))
    alphabet_len = len(background)
    odds_ratio = ((ppm+pseudocount)/(1 + pseudocount*alphabet_len))/(
                  background[None,:])
    return np.log(odds_ratio)


def compute_pwm_scan(onehot_seq, weightmat):
    strided_onehotseq = rolling_window(
        onehot_seq.transpose((0,2,1)),
        window=len(weightmat)).transpose((0,2,3,1))
    pwm_scan = np.sum(
        strided_onehotseq*weightmat[None,None,:,:], axis=(2,3)) 
    return pwm_scan


def compute_sum_scores(imp_scores, window_size):
    strided_impscores = rolling_window(
        imp_scores.transpose((0,2,1)),
        window=window_size).transpose((0,2,3,1))
    sum_scores = np.sum(strided_impscores, axis=(2,3))
    return sum_scores


def trim_ppm(ppm, t=0.45):
    maxes = np.max(ppm,-1)
    maxes = np.where(maxes>=t)
    return ppm[maxes[0][0]:maxes[0][-1]+1] 
        

def write_meme_file(ppm, bg, fname):
    f = open(fname, 'w')
    f.write('MEME version 4\n\n')
    f.write('ALPHABET= ACGT\n\n')
    f.write('strands: + -\n\n')
    f.write('Background letter frequencies (from unknown source):\n')
    f.write('A %.3f C %.3f G %.3f T %.3f\n\n' % tuple(list(bg)))
    f.write('MOTIF 1 TEMP\n\n')
    f.write('letter-probability matrix: alength= 4 w= %d nsites= 1 E= 0e+0\n' % ppm.shape[0])
    for s in ppm:
        f.write('%.5f %.5f %.5f %.5f\n' % tuple(s))
    f.close()


def fetch_tomtom_matches(ppm, background=[0.25, 0.25, 0.25, 0.25], tomtom_exec_path='tomtom', motifs_db='HOCOMOCOv11_core_HUMAN_mono_meme_format.meme' , n=5, temp_dir='./', trim_threshold=0.45):
    """Fetches top matches from a motifs database using TomTom.
    
    Args:
        ppm: position probability matrix- numpy matrix of dimension (N,4)
        background: list with ACGT background probabilities
        tomtom_exec_path: path to TomTom executable
        motifs_db: path to motifs database in meme format
        n: number of top matches to return, ordered by p-value
        temp_dir: directory for storing temp files
        trim_threshold: the ppm is trimmed from left till first position for which
            probability for any base pair >= trim_threshold. Similarly from right.
    
    Returns:
        list: a list of up to n results returned by tomtom, each entry is a
            dictionary with keys 'Target ID', 'p-value', 'E-value', 'q-value'  
    """
    
    fname = os.path.join(temp_dir, 'query_file')
    
    # trim and prepare meme file
    write_meme_file(trim_ppm(ppm, t=trim_threshold), background, fname)
    
    # run tomtom
    cmd = '%s -no-ssc -oc . -verbosity 1 -text -min-overlap 5 -mi 1 -dist pearson -evalue -thresh 10.0 %s %s' % (tomtom_exec_path, fname, motifs_db)
    #print(cmd)
    out = subprocess.check_output(cmd, shell=True)
    
    # prepare output
    dat = [x.split('\\t') for x in str(out).split('\\n')]
    schema = dat[0]
    tget_idx, pval_idx, eval_idx, qval_idx = schema.index('Target ID'), schema.index('p-value'), schema.index('E-value'), schema.index('q-value')
    
    r = []
    for t in dat[1:1+n]:
        mtf = {}
        mtf['Target ID'] = t[tget_idx]
        mtf['p-value'] = float(t[pval_idx])
        mtf['E-value'] = float(t[eval_idx])
        mtf['q-value'] = float(t[qval_idx])
        r.append(mtf)
    
    os.system('rm ' + fname)
    return r


def show_or_savefig(plot_save_dir, filename):
    from matplotlib import pyplot as plt
    if plt.isinteractive():
        plt.show()
    else:
        import os, errno
        try:
            os.makedirs(plot_save_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        fname = (plot_save_dir+"/"+filename)
        plt.savefig(fname)
        print("saving plot to " + fname)
