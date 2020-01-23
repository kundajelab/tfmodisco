from __future__ import division, print_function
import os
import signal
import subprocess
import numpy as np
import h5py
import traceback
from sklearn.neighbors.kde import KernelDensity
from datetime import datetime


def print_memory_use():
    import os
    import psutil
    process = psutil.Process(os.getpid())
    now = datetime.now()
    print("MEMORY",process.memory_info().rss/(2**30),now)


def percentile_transform(vals):
    ordering = np.argsort(vals)
    percentiles = np.arange(len(vals))/len(vals)
    to_return = np.zeros(len(vals))
    to_return[ordering] = percentiles
    return to_return


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


def factorial(val):
    to_return = 1
    for i in range(1,val+1):
        to_return *= i
    return to_return


def first_curvature_max(values, bins, bandwidth):
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


def convert_to_percentiles(vals):
    to_return = np.zeros(len(vals))
    sorted_vals = sorted(enumerate(vals), key=lambda x: x[1])
    for sort_idx,(orig_idx,val) in enumerate(sorted_vals):
        to_return[orig_idx] = sort_idx/float(len(vals))
    return to_return


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
    assert (np.max(np.abs(np.sum(ppm, axis=1)-1.0)) <= 2e-3),(
             "Probabilities don't sum to 1 along axis 1 in "
             +str(ppm)+"\n"+str(np.sum(ppm, axis=1)))
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
    assert (np.max(np.abs(np.sum(ppm, axis=1)-1.0)) <= 2e-3),(
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


def l1norm_contin_jaccard_sim(arr1, arr2):
    assert len(arr1.shape)==3, arr1.shape
    assert arr1.shape[1:]==arr2.shape[1:], (arr1.shape, arr2.shape)
    absarr1 = np.abs(arr1) 
    absarr1 = absarr1/np.sum(absarr1, axis=(1,2))[:,None,None] #l1 norm
    absarr2 = np.abs(arr2)
    absarr2 = absarr2/np.sum(absarr2, axis=(1,2))[:,None,None] #l1 norm
    signarr1 = np.sign(arr1)
    signarr2 = np.sign(arr2)
    return (np.sum(np.minimum(absarr1,absarr2)*signarr1*signarr2, axis=(1,2))/
            np.sum(np.maximum(absarr1, absarr2), axis=(1,2)))


def facility_locator(distmat, num_exemplars):
    exemplars = [] 
    current_bestrep = np.inf*np.ones(distmat.shape[0])
    for i in range(min(num_exemplars, len(distmat))):
        candidate_newbestrep = np.minimum(distmat, current_bestrep[None,:])  
        candidate_objective = np.sum(candidate_newbestrep, axis=-1) 
        next_best_exemplar = np.argmin(candidate_objective) 
        exemplars.append(next_best_exemplar)
        current_bestrep = candidate_newbestrep[next_best_exemplar]
    return exemplars


#From: https://github.com/theislab/scanpy/blob/8131b05b7a8729eae3d3a5e146292f377dd736f7/scanpy/_utils.py#L159
def get_igraph_from_adjacency(adjacency, directed=None):
    """Get igraph graph from adjacency matrix."""
    import igraph as ig
    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    if isinstance(weights, np.matrix):
        weights = weights.A1
    g = ig.Graph(directed=directed)
    g.add_vertices(adjacency.shape[0])  # this adds adjacency.shap[0] vertices
    g.add_edges(list(zip(sources, targets)))
    try:
        g.es['weight'] = weights
    except:
        pass
    if g.vcount() != adjacency.shape[0]:
        logg.warning(
            f'The constructed graph has only {g.vcount()} nodes. '
            'Your adjacency matrix contained redundant nodes.'
        )
    return g
