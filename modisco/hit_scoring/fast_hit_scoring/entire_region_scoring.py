from __future__ import division, print_function, absolute_import
from joblib import Parallel, delayed


def score_trackset_with_pattern(track_set, pattern,
                                track_names, track_transformer, n_cores):
    


def parallelcpu_score_full_regionarrs_with_perpos_continjaccard(
    regionarrs_to_scan, arr_to_scan_with, n_cores):

    assert len(arr_to_scan_with.shape)==2
    for regionarr in regionarrs_to_scan:
        assert len(regionarr.shape)==2
        assert arr_to_scan_with.shape[-1] == regionarr.shape[-1]

    
    start = time.time()
    #parallelize by input
    job_arguments = []
    for regionarr_to_scan in regionarrs_to_scan:
        job_arguments.append((regionarr_to_scan, arr_to_scan_with))
    
    full_crosscontinjaccards =\
        list(
         Parallel(n_jobs=n_cores)(
            delayed(score_full_regionarr_with_perpos_continjaccard)(*jobargs)
            for jobargs in job_arguments))
    return full_crosscontinjaccards


def score_full_regionarr_with_perpos_continjaccard(regionarr_to_scan,
                                                   arr_to_scan_with):
    arr_to_scan_with_norm = np.sum(np.abs(arr_to_scan_with))

    window_len = len(arr_to_scan_with)

    #np.cumsum will give the cumulative sum at the *end* of every bin
    per_pos_cum_sum_abs = np.cumsum(np.sum(np.abs(region_to_scan),axis=1))
    #by subtracting per_pos_cumsum_abs[0], we can get the cumulative
    # sum at the *start* of every bin. Then we can take the difference of the
    # two at an offset of (window_len-1) to get the sums in windows of
    # length window_len
    per_pos_window_sum = (per_pos_cum_sum_abs[window_len-1:]
                          - (per_pos_cum_sum_abs[:len(per_pos_cum_sum_abs)-
                                                  (window_len-1)]-
                             per_pos_cum_sum_abs[0]))
    #per_pos_scale_factor is how much to scale the window sum by to equal
    # the norm of arr_to_scan_with
    per_pos_scale_factor = arr_to_scan_with_norm/(
                            per_pos_window_sum+(0.0000001*
                                                (per_pos_window_sum==0)))

    full_crossmetric = np.zeros(len(regionarr_to_scan)+1-window_len)

    absolute_arr_to_scan_with = np.abs(absolute_arr_to_scan_with)
    signed_arr_to_scan_with = np.sign(absolute_arr_to_scan_with)

    for idx in range(len(full_crossmetric)):
        region_snapshot = (region_to_scan[idx:(idx+window_len)]
                           *per_pos_scale_factor[idx]) 
        abs_region_snapshot = np.abs(region_snapshot)
        union = np.sum(np.maximum(abs_region_snapshot,
                                  absolute_arr_to_scan_with))
        intersection = np.sum(np.minimum(abs_region_snapshot,
                                         absolute_arr_to_scan_with)*
                              np.sign(region_snapshot)*
                              signed_arr_to_scan_with)
        full_crossmetric[:,idx] = intersection/union 
    return full_crossmetric
